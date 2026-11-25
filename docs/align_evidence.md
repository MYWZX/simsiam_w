# NSS峰度与图文一致性证据：独立实验版本

本协议不修改 current_tuning_reference/current_baseline，不复用历史输出目录，不把旧成绩标成新模式结果。云端只执行CPU测试和配置预览，真实训练必须在服务器CUDA环境运行。

## NSS第六维

新增 descriptor=`legacy_kurt6_v1`，签名 `[5,6]`。继续使用原7×7均值滤波、零填充和原MSCN-like计算，前五维及6→256→256注入MLP、262→256→1门控保持原样，仅把重复的第六维替换为区域系数的超额峰度：

`k_i = mean[(z-mean(z))^4] / mean[(z-mean(z))^2]^2 - 3`。

中心矩采用FP32，同一自适应池化区域；方差≤1e-12返回0，其余使用方差下界保护分母。不任意裁剪峰度。224→14的整除网格使用unfold并行计算；非整除尺寸使用精确自适应池化区间。k没有额外标准化或GGD拟合。z是归一化亮度系数，不是融合输出。

旧六维和新六维虽然形状相同，语义不同：禁止检查点混载，不做“等价迁移”。新K组从原始CLIP/ResNet初始化重新预训练，保留官方train/val，按val_component_anchor_SROCC选模，不resume项目旧实验。下游严格使用本组已完成、SHA256核验通过的best_component_anchor_pretrain.pth。

## 一致性顺序对照

两条NSS arm：legacy使用固定历史来源fa702194…；kurt6使用上面新来源。每条arm在3K/2023各有五组，组间独立训练而非接着上一组训练：

| 组 | 相对前组的唯一设计变化 |
|---|---|
| diag | 原局部路径、固定温度1，仅增加诊断；不增加参数，不改变评分 |
| temperature | 原局部路径，增加可训练的有界匹配温度 |
| tokenmean | 换成CLIP区域/BPE词元路径，逐词元检索后普通均值汇总；独立新投影明确随机初始化 |
| coverage | 同样的区域/词元路径，改用低匹配内容获得较高汇总权重 |
| reference | 再加入当前CLIP原生空间相似度的可训练校准与分数混合 |

legacy/diag对kurt6/diag检验NSS替换。temperature→tokenmean用于识别换局部路径的影响；必须用tokenmean→coverage检验覆盖度，不能把新路径的影响算成覆盖度收益。累加组只能解释该上下文下的增量，不等同于完整析因实验。

温度：`tau=.03+1.97*sigmoid(logit)`，初始化1，归入align组，LR1e-4。旧align_local_temperature仍保持1，禁止重复启用；本协议也不与CLIP后的global_adapter叠加，以免把adapter输出误称为原生CLIP空间。不是直接照抄CLIP的logit_scale。

词元：排除SOT/上下文/EOT/pad，额外过滤仅含标点的BPE片段。保留词法子词，不宣称完成分词、词性筛选或实体解析；全标点Prompt回退原内容mask并报告fallback。两组tokenmean/coverage采用同一mask。覆盖度先计算每个词元检索到的视觉证据与词元的cos，再用`softmax(-match/0.1)`对有效词元加权，同时汇总视觉和文本特征。它是软的低匹配加权，不是严格逻辑AND，不用伪造的缺失标签。

reference明确不是独立冻结的原始CLIP教师：使用当前同一配套CLIP图像/文本编码器的原生输出，在进入256维投影前做cos，文本仍复用同样5个上下文；图像编码器保持基线可训练设置。`s_ref=sigmoid(a*cos+b)`，a=1,b=0；最终`(1-mix)*s_align_base+mix*s_ref`，mix初始化0.05且可训练。四个新增标量（含temperature）明确记录初值SHA和align组归属。没有另加权重来源、下载或第二个编码器，也不以任意两个256维投影冒充CLIP空间。

保持projection_only_v1，无跨注意力和融合z。各组保留质量/NSS路径、原损失权重及LR策略、PI/PT LR1e-4和global_projection LR1e-5；seed/split42；下游60轮、不合并train+val。对同一60轮轨迹分别按验证Quality SROCC、Quality/Align SROCC算术均值选模；测试仅训练结束后对所选模型评估，不据测试选模型。1K没有一致性标签，只有K来源的原结构60轮质量对照（subset1078/split3407）。

## 诊断与验证

验证/所选测试模型记录：空间归一化熵、区域最大权重、匹配温度、Prompt实际截断比例、三个一致性分数的MAE；词元组再记录平均/最低词元匹配、覆盖度最大权重和词法mask回退率；reference记录原生cos、校准分数、混合比例及原生cos与人工标签的相关性。各均值只统计有效一致性标签样本。

旧配置无新模块、参数或随机数消耗；diag测试检查输出逐元素一致。温度、上下文及新投影的梯度、NSS中心矩、常量区域、AMP有限性、跨版本载入拒绝、LR归属均有CPU合成测试。训练器与来源检查沿用已验证协议；新增参数初值记录到run_metadata和CSV。真实数据表现、峰度动态范围和多seed稳定性必须由服务器实验判断。

## 服务器运行

Python默认 `/home/sll/anaconda3/envs/WZX-AI/bin/python`，可用ABLATION_PYTHON覆盖。先 `git pull --ff-only origin work`。

先只预训练K来源：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_nss_kurtosis.sh pretrain
```

两条完整队列可以独立放在两张卡。kurt6队列会先核验/跳过已完成预训练，再完成三个K下游对照，之后按diag→temperature→tokenmean→coverage→reference运行3K/2023。legacy队列直接使用历史来源。两队列无相互等待：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_evidence_chain.sh legacy
CUDA_VISIBLE_DEVICES=1 bash scripts/run_evidence_chain.sh kurt6
```

如需防VS Code断连，可用后台启动器同时启动上述两队列（仅服务器双GPU可用时允许）：

```bash
bash scripts/launch_evidence_background.sh
```

启动器输出独立日志目录和PID，不结束或迁移其他进程。使用完成核查器，全部完成且通过核验时才返回成功：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/check_align_evidence.py legacy
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/check_align_evidence.py kurt6
```

完整矩阵为1次新预训练＋23次60轮下游（legacy10＋kurt6的3个来源对照和10个证据实验）。不需要一次全部运行；单组可执行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_evidence.sh legacy temperature --dataset agiqa3k --continue
CUDA_VISIBLE_DEVICES=1 bash scripts/run_align_evidence.sh kurt6 coverage --dataset agiqa3k --continue
```

`--continue`只核验并跳过完整结果，不恢复或覆盖半成品。部分输出存在时明确停止，保留目录；不删除、不补造完成记录。队列采用flock防止同一arm重复启动。独立新目录和RNG行为变化意味着不承诺训练逐位复现。

配置预览，不要求模型权重和CUDA：

```bash
bash scripts/run_align_evidence.sh legacy all --dry-run
bash scripts/run_align_evidence.sh kurt6 all --dry-run
bash scripts/run_nss_kurtosis.sh all --dry-run
```

完成后分别打包；包含CSV、日志、配置、元数据、划分、完成记录和来源SHA，不包含大权重，已有同名包时拒绝覆盖：

```bash
bash scripts/run_nss_kurtosis.sh pack-downstream
bash scripts/run_align_evidence.sh legacy pack
bash scripts/run_align_evidence.sh kurt6 pack
```

生成六面板验证曲线图（PNG/PDF），输出目录必须是新目录；只读取已记录指标，不重新推理、不重新选模：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/plot_align_evidence.py legacy --output run_logs/evidence_legacy_figures_v1
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/plot_align_evidence.py kurt6 --output run_logs/evidence_kurt6_figures_v1
```

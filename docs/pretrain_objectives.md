# 原六维基线：五项预训练单因素调参

当前历史最好基线保留。本次实现5个因素×2个候选=10次独立新预训练；每组可随后运行三个下游，共30次下游。不是组合十个参数，也不从上一组权重续训。

| 因素 | 配置位置（train下） | 原值 | low队列（较低值） | high队列（较高值） |
|---|---|---:|---|---|
| 图像—Answer关系权重 | lambda_img_answer | 0.20 | answer010 = 0.10 | answer030 = 0.30 |
| Prompt—Answer关系权重 | lambda_prompt_answer | 0.05 | pa025 = 0.025 | pa100 = 0.10 |
| 局部图像—Prompt关系权重 | lambda_img_prompt_local | 0.05 | local025 = 0.025 | local100 = 0.10 |
| 词级Prompt—Answer关系权重 | lambda_prompt_answer_word | 0.05 | word025 = 0.025 | word100 = 0.10 |
| Component Anchor温度 | quality_component_anchor_ordinal.anchor_temperature | 0.20 | anchortau010 = 0.10 | anchortau030 = 0.30 |

每组仅改表中一个叶子参数，另改独立run_id/输出路径。I-I权重0.15、I-P权重0.25、Artifact/Aesthetics Anchor各0.01、Ordinal各0.015保持，不叠加已经测试的候选。

I-A、P-A仍分别使用既有Answer权重和Prompt/Answer关系权重，样本关系权重生成方式不变。local/word两项仍是现有局部与词级关系监督，负关系保持关闭；提高系数不会打开push、负样本或质量文本CE。

Anchor温度共同作用于Artifact/Aesthetics各自的五级概率、Anchor CE及Ordinal BCE，并影响用于预训练选模的anchor概率评分。不是只调CE强度，也不是改变两个loss的系数。比较预训练val_component_anchor_SROCC时需留意评分映射随温度而变，不能直接将其提高当成表示质量提升；最终还需看统一下游评估。

## 固定协议与来源

- projection_only_v1，原六维NSS-like/both路径，CLIP图像编码器可训练，5个可训练上下文词元，预训练I-I和图文使用g→PI，P0绕过PI；保留双质量残差、NSS、Component Anchors/Ordinal等原有目标；质量文本五档CE和负关系仍关闭。
- 每组从原始CLIP/ResNet预训练初始化开始。不加载本项目旧预训练/下游检查点，不resume其他实验。train_seed42、预训练split_seed42，保留RichHF官方train/val，不合并。
- 预训练按最大val_component_anchor_SROCC选模，原50轮上限、patience15及原学习率/优化器/调度保持；不改成最低loss选模。
- 每组的三个下游只加载本组新选出的best_component_anchor_pretrain.pth。保持当前v2参考的60轮、最大val_quality_SROCC选模、train/val/test分离；train_seed42，1K subset1078 split3407，3K/2023 split42。
- 下游恢复原共享Soft-SROCC温度0.02、Quality Soft-SROCC系数0.03/0.02/0.04，其余参数原样。PI/PT初始LR1e-4、shared_projection_mult1.0，全局投影仍属fusion组、初始LR1e-5。没有采用五维NSS、标准MSCN、GGD、冻结CLIP或global Adapter候选。
- 启动/完成/配置/权重来源SHA核验复用严格运行器。十组输出、源指纹和配置目录各自独立，拒绝跨组、旧模式或其他来源的错误加载。下游在源预训练未完成时拒绝启动。3K没有完整控制成员表时不伪造划分证明，按现有策略报告种子/可用证据。

## 两张卡执行

在服务器仓库干净时同步work。默认动作只有pretrain；以下命令显式要求每组预训练完成后自动顺序跑其三个下游，再进入下一组。两条队列可并行，各自最多一项训练任务。

```bash
# 终端一：五个较低候选，共5次预训练、15个下游
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_objectives.sh low all

# 终端二：五个较高候选，共5次预训练、15个下游
CUDA_VISIBLE_DEVICES=1 bash scripts/run_pretrain_objectives.sh high all
```

low顺序：answer010 → pa025 → local025 → word025 → anchortau010。
high顺序：answer030 → pa100 → local100 → word100 → anchortau030。

每个预训练都重新恢复原始模型初始化；队列前一组结果不作为后一组初始权重。每项结束并验证成功后才开始下一项，遇到失败立即停止。

只排队跑预训练，不自动下游：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_objectives.sh low pretrain
CUDA_VISIBLE_DEVICES=1 bash scripts/run_pretrain_objectives.sh high pretrain
```

也支持单独一组或某个数据集，例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_objectives.sh answer010 all
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_objectives.sh answer010 agiqa1k
```

上面是不同用法示例，不要重复启动同一组。默认Python沿用/home/sll/anaconda3/envs/WZX-AI/bin/python；ABLATION_PYTHON可覆盖；HF_HUB_OFFLINE默认1，需要原始CLIP/ResNet缓存已可用。

## 预检、完成复用与失败处理

```bash
bash scripts/run_pretrain_objectives.sh check-configs
bash scripts/run_pretrain_objectives.sh low all --dry-run
bash scripts/run_pretrain_objectives.sh high all --dry-run
```

check-configs验证全部40个静态配置；dry-run显示计划，检查配置及已存在阶段证据，不启动训练、不创建实验输出。不存在的未来预训练权重会在实际进入下游前核验。

队列开始前会预检队列内五组，发现已有未完成输出或配置不符时不会先启动前面的昂贵训练。all/continue只复用已完成且验证通过的阶段，拒绝覆盖或自动resume未完成阶段。若服务器重启或失败，先保留目录/CSV/日志发回诊断，不删除或伪造completion_manifest。此前两组NSS专用recover-all入口不适用于本次新组。

## 打包

所有任务完成后分别打包两队列，每条命令生成五个独立结果包，不启动训练：

```bash
bash scripts/run_pretrain_objectives.sh low pack-downstream
bash scripts/run_pretrain_objectives.sh high pack-downstream
```

单组也可用 `answer010 pack-downstream`；仅预训练结果用 `answer010 pack`。包名统一为 `projection_only_v1_ctx5_obj_<variant>_downstream_results.tar.gz`，含该组预训练及三个下游的CSV、配置、日志、元数据和来源指纹，不含权重。已有同名包拒绝覆盖；如果已单组打包，不要重复用队列命令覆盖。

如需一次上传全部十组，完成上面两次打包后可将十个小结果包再合并（仅匹配结果包，不匹配权重）：

```bash
test ! -e projection_only_v1_ctx5_objectives_all_results.tar.gz && \
tar -czf projection_only_v1_ctx5_objectives_all_results.tar.gz \
  projection_only_v1_ctx5_obj_*_downstream_results.tar.gz
```

合并文件已存在时换新文件名保留旧包。此合并包为嵌套的十个结果包，上传后按组比较全部Quality/Alignment SROCC与PLCC。新结果独立报告，不自动改动当前最好基线；这是固定且已查看划分上的探索性调参，不能视为独立确认。

## 本次验证

10项CPU测试通过：40个配置严格单因素、独立路径、无旧项目权重/resume；10个假预训练及30个假下游的来源/完成记录/选模规则/60轮/划分证据核查与无权重打包；CPU拒绝真实训练；改配置/改权重指纹/已有输出拒绝；两队列全量预检、dry-run不训练、失败停止。

真实训练器的evaluate损失汇总在假骨干上验证I-A/P-A系数仅改变相应贡献；local/word汇总使用隔离的固定原始loss，另由实际map/word估计器测试确认四个正系数均保持positive-only、无push、raw loss不被系数重复缩放且梯度有限。真实Component Anchor函数验证两个分量都收到0.1/0.3温度、概率评分与CE改变，CE+Ordinal的梯度有限。

两队列all --dry-run、40配置check-configs、shell语法与git diff --check通过。只新增运行器、静态配置、文档和测试，没有修改模型、训练器或损失实现，也未在云端运行真实训练或加载服务器大权重。

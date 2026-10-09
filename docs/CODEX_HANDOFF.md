# AIGC-IQA 项目跨账号交接（更新至 2026-10-01 00:49 CST）

> 2026-10-01 后续更新：已实现 `tools/diagnose_prompt_sensitivity.py`，
> 用已有 3K/2023 检查点在验证集比较原 Prompt 与主体/动作替换 Prompt 的 Align 分数；候选不是人工真值。
> 新增 `--isolate-text-context-rng` 到预训练和下游，用于 seed123 固定17轮前缀配对复核；默认关闭。
> 完整命令和边界见 `docs/text_context_next_experiments.md`；66 项单元测试通过，
> 两个 3K 检查点的 CPU 小批次成对推理通过；完整诊断和新预训练尚未启动。
> 本段优先于下文旧的“正在运行/尚无输出”时间点描述；使用前重新核查进程和 CSV。


这是一份**会话摘要，不是逐字聊天导出**。新账号不会自动继承旧账号的聊天记录；将本文件的绝对路径发给新 Codex，并要求先读本文及文末证据文件。项目目录：`/home1/wzx/aigc_naturalness_code/simsiam`。本文只记录核查过的代码、输出和用户明确表达的研究选择；运行中的结果须重新检查。

## 给新 Codex 的第一条消息（可直接复制）

> 请先完整阅读 `/home1/wzx/aigc_naturalness_code/simsiam/docs/CODEX_HANDOFF.md`，再核对其中列出的现有代码、配置、CSV、正在运行的进程及最新检查点。不要假设新账号继承了旧聊天。先报告你核查到的当前状态和下一步，不要自动重启/覆盖实验，也不要把进行中的实验写成最终结论。

## 研究目标和用户偏好

- 目标是 AIGC 图片质量评价，独立给出 **Quality**（图像视觉质量）和 **Align**（图片与生成 Prompt 的一致性），重点改善 AGIQA-1K 的 Quality、AGIQA-3K 与 AIGCIQA2023 的 Align。1K 没有可用于评测的人工 Align 标签。
- 用户坚持保留 SimSiam 作为核心创新，但允许**诊断对照**暂时把常规 SimSiam 损失关掉，以寻找有效条件。Prompt「猫在吃饭」对应「狗在吃饭」或「猫在拉屎」都应判低一致性；同时希望模型理解主体和动作的局部关系。
- 最终论文方法希望在三个下游数据集使用同一方案，不主张为每个数据集临时拼不同结构。探索阶段某一个 seed 有提升可以保留为候选，但必须记录所有结果；正式泛化/领先结论须在固定协议和公平划分上验证。
- 历史合成 Final MOS 是 Quality/Align 的人为加权值，**没有独立人工 Final MOS**。当前正式评价应分别报告 Quality/Align PLCC、SROCC，不把内部仍计算的 `s_final` 当作人工标注分数。
- 用户喜欢可执行指令；给预训练指令时紧接着给下游指令；**不要指定 GPU 编号**。未经请求不要自动发起长时间训练或覆盖已有输出。
- CVPR 2026 DPGF-Net 分数只是希望超越的目标，不要求照搬结构；目前没有核验其与本项目标签、划分、指标完全一致，不可宣称公平 SOTA 比较。

## 当前模型和迁移路径

- 主要代码：`models/two_stage_models.py`、`models/backbones.py`、`models/fusion.py`、`train_cm_pretrain.py`、`train_cm_iqa.py`、`datasets/mixed_pretrain_dataset.py`、`datasets/multi_iqa_dataset.py`。
- 当前主线预训练配置：`configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml`。RichHF-18K 提供 Prompt、Answer、人工 Quality/Align、Artifact/Aesthetics、区域 misalignment map 与词级反馈。严格三视图：SafeAug1/2 用于 Image–Image SimSiam；Clean View 用于跨模态与人工标签/局部/Component Anchor 监督。
- 图像侧 OpenCLIP ViT-B/16 + ResNet50 + SemanticStructureFusion；文本侧 OpenCLIP ViT-B/16 文本 Transformer 原参数冻结。普通四项：Image–Image、Image–Prompt、Image–Answer、Prompt–Answer SimSiam。还有 RichHF 人工 Quality/Align、局部 pull–push、Artifact/Aesthetics Quality Auxiliary 与 Component Anchor/Ordinal、预训练 PCGrad。各项是否有独立下游增益尚未全部证明。
- 下游主线配置：`configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml`、`configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml`、`configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml`；`pretrained_load_scope: representation`。加载视觉编码器/Fusion、`text_to_fused`、Align adapters，以及质量 projector → Quality Adapter；SimSiam predictor 和大部分预训练评分头不迁移。Quality 分支视觉主导，Align 用 Prompt 与全局/局部图像特征，Align gate 按样本自适应；质量内部 `quality_mix` 是可学习全局标量。
- 1K 的 `align_mask=0`、Align 损失权重为 0；3K 和 2023 有独立 Align 标签。两个输出协议启用且 Final 损失为 0。旧接口仍可能返回 `s_final`，不可误解为有效人工标签。

## 已做实验与应避免的误读

- 历史主线依次探索：四路 SimSiam、联合 IQA 预训练、分层学习率/局部反馈/相对排序、质量有序关系与三元组、同源 CLIP 视觉编码器、五级 Quality Anchor+Ordinal、SafeAug/MultiCrop、Artifact/Aesthetics 分项、严格三视图 Clean 监督、质量采样/损失重加权/镜像、去冗余、两输出解耦、Quality/Align PCGrad、Adapter 迁移与学习率消融。完整历史表在 `outputs/experiment_summary_after_quality_anchor.html`，但它**已过时**，不能作为当前最佳成绩的唯一依据。
- 更可信的当前审计：`outputs/project_experiment_audit_20260929.md` 和同名 CSV/HTML。它区分实际 seed、split、检查点和 `best_test`，并指出过时 HTML 中的错误。历史跨配置最高分多为探索单次结果，不能与当前两输出实验或 SOTA 直接横比。
- 当前研究诊断设计在 `docs/rq_diagnostics.md`。RQ1 比较四个常规 SimSiam 损失全开 vs 全 0（仍保留其他 Siamese 风格/人工监督）；两组均 seed42、固定17轮、相同 scheduler 和下游协议。已有 `best_test`：1K Quality SROCC **0.8797 vs 0.8697**；3K Align SROCC **0.8203 vs 0.8136**；2023 Align SROCC **0.7560 vs 0.7518**（前者全开）。这支持“四项合起来在这次受控运行中有用”，**不证明每一分支独立有效或多 seed 稳定**。3K 关掉后 Quality PLCC 反而更高，须报告权衡。结果 CSV：`outputs/rq_diagnostics_v1/all_results_rq1_{agiqa1k,agiqa3k,aigciqa2023}.csv`。
- RQ3 比较同一预训练检查点下的 Quality Adapter `replace/parallel_score × pretrained/random`。截至 2026-10-01 00:49 CST，1K/3K/2023 各四组的 CSV 均有 `best_test` 和 `final`。按 replace-pretrained、replace-random、parallel-pretrained、parallel-random 的顺序，测试 SROCC（Quality / Align）为：1K `0.8797 / 不适用`、`0.8795 / 不适用`、`0.8838 / 不适用`、`0.8755 / 不适用`；3K `0.8762 / 0.8203`、`0.8744 / 0.8207`、`0.8591 / 0.7804`、`0.8594 / 0.7843`；2023 `0.8567 / 0.7560`、`0.8512 / 0.7616`、`0.8427 / 0.7630`、`0.8428 / 0.7815`。2023 的 parallel-random 虽有最高 Align，却降低 Quality；不能只挑一个指标宣称整体改进。结果文件：`outputs/rq_diagnostics_v1/all_results_rq3_{agiqa1k,agiqa3k,aigciqa2023}.csv`。**同模式** pretrained/random 才较适合检验 Adapter 预训练权重贡献；不同模式改变结构且消耗随机数的方式不同。以上仍是单 seed、各组按验证集选择 epoch 的探索结果。
- 新增 RQ1 的 **seed123 成对复核**：两组均用 `configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml`、训练 seed123、原配置的 split seed42，固定17轮；`full` 保持四项常规 SimSiam，`no_regular_simsiam` 仅用 `--disable-regular-simsiam` 将 Image–Image、Image–Prompt、Image–Answer、Prompt–Answer 四项的损失权重清零。后者**不是完全无 SimSiam 架构/监督**：编码器、人工 Quality/Align、局部反馈、Component Anchor 等仍保持相同。它与文本前缀实验无关，两组都没有 `--text-context-tokens 4`。截至上述时间，两组预训练进程都在运行；各自结果 CSV 只有 `run_start`，**尚无 epoch 或 final 行，更无下游结论**。结果文件：`outputs/rq_diagnostics_v1/all_results_rq1_pretrain_{full,no_regular_simsiam}_seed123.csv`；检查点目录：`outputs/rq_diagnostics_v1/rq1_pretrain_{full,no_regular_simsiam}_seed123/`。不要覆盖或重复启动。
- 2026-09-29 模块审计指出当前预训练的 Anchor train/val 差距大、局部质量和 Prompt–Answer 等模块缺独立有效性证据；预训练 PCGrad 是逐参数版本，下游 `global_shared` 是另一个实现。详见审计文档，勿凭启用/非零 loss 直接宣称有效。

## 最新实现：CLIP 可学习文本前缀（正在测）

- 2026-09-30 新增 `--text-context-tokens N`（建议 N=4）到预训练和下游。冻结 CLIP 原文本参数，在 `[SOT, C1..C4, 原 Prompt, EOT]` 输入中只训练 4×512 个共享上下文向量；不是 image-conditioned token，也不是论文完整 triple-prompt。文本经冻结 Transformer 后继续走原 `text_to_fused` / Align 路径；所有原 SimSiam、损失和图像分支保持不变。Image–Image 不直接更新文本词元，其他涉及文本的损失可更新。
- `encode_cached` 在该实验开启时绕开 detached 缓存，避免断梯度；文本词元进入精简 `best_component_anchor_pretrain.pth`，下游 representation 加载；启用时若检查点缺词元或形状不符则报错。下游词元独立 LR 组默认为 1e-5，预训练用原 `other` 组 2e-5。默认 N=0 时历史路径不变。
- 新增 `tests/test_learnable_text_context.py`；真实 OpenCLIP ViT-B/16 的 CPU 前向/反向通过，原 CLIP 冻结，词元有梯度；本交接时全套单元测试 **62 项通过**。**未因此证明下游指标提升**。
- 截至 2026-10-01 00:49 CST，预训练进程仍在运行：`outputs/richhf18k_cm_pretrain_decoupled_text_context4/`；CSV `outputs/all_results_richhf_text_context4.csv` 已记录至 epoch 21，**没有 `final` 行**。目录中有临时最佳检查点，但不能据此认定训练完成。新会话须重新检查进程、CSV 与检查点时间戳。先前记录的 RQ3 下游进程已结束，三个数据集的四组对照均已完成。
- 尚未看到 `outputs/agiqa1k_decoupled_text_context4_seed42`、`outputs/agiqa3k_decoupled_text_context4_seed42`、`outputs/aigciqa2023_decoupled_text_context4_seed42`；下游必须等预训练完成并确认所选检查点。优先检验 3K/2023 Align SROCC/PLCC 与质量非劣；1K 只看 Quality。困难负例（主体/动作替换）仍需要单独验证。

## 最新实验的复现命令（仅核对状态后运行，不重复覆盖）

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
/home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_pretrain.py --config configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml --seed 42 --text-context-tokens 4 --output-dir outputs/richhf18k_cm_pretrain_decoupled_text_context4 --results-csv outputs/all_results_richhf_text_context4.csv
```

预训练完成后，三个下游都带 `--text-context-tokens 4`，并显式传同一个新检查点：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml --seed 42 --text-context-tokens 4 --pretrained-checkpoint outputs/richhf18k_cm_pretrain_decoupled_text_context4/best_component_anchor_pretrain.pth --output-dir outputs/agiqa1k_decoupled_text_context4_seed42
/home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml --seed 42 --text-context-tokens 4 --pretrained-checkpoint outputs/richhf18k_cm_pretrain_decoupled_text_context4/best_component_anchor_pretrain.pth --output-dir outputs/agiqa3k_decoupled_text_context4_seed42
/home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml --seed 42 --text-context-tokens 4 --pretrained-checkpoint outputs/richhf18k_cm_pretrain_decoupled_text_context4/best_component_anchor_pretrain.pth --output-dir outputs/aigciqa2023_decoupled_text_context4_seed42
```

下游加载日志应显示 `text_context=1`。已有新实验运行时**不要执行上面的预训练命令**；检查状态用 `ps -eo pid,etime,args | rg 'train_cm_pretrain.py|train_cm_iqa.py'`，并查 CSV 的 `final`/`best_test` 记录。

## 相关论文与证据边界

- 本地 PDF：`Image_Text_Guided_CLIP_Based_Framework_forNo_Reference_Stereoscopic_Image_Quality_Assessment(1).pdf`。曾分析其冻结 CLIP、可学习上下文、图像条件化词元和质量提示词消融；只借鉴**文本端少量参数可训练**的思路。它处理立体图像质量，并未验证 AIGC 图文一致性。PDF 含 `[Author Names]`、图表占位符，未核实正式发表或数据；不能把其中 SRCC 数值当作可信 SOTA 依据。
- 已核实的 prompt-learning 原始来源之一是 CVPR 2022 CoCoOp：https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.html 。当前实现只是共享 learnable prefix，不是 CoCoOp 的每图条件化词元。

## 新会话优先顺序

1. 读本文件、`outputs/project_experiment_audit_20260929.md`、`docs/rq_diagnostics.md`；重新核查所有“正在运行”状态。
2. 对文本前缀预训练和 RQ1 seed123 两组预训练分别确认 `final`、选择指标与检查点有效性；不以中途 epoch 或文件存在代替完成。三者均不可重复启动/覆盖。RQ3 已完成，不必重跑。
3. 文本前缀完成后，在同 seed/split、同原配置下只比较文本词元开关；RQ1 seed123 两组完成后再按下文成对运行下游。分别报告 1K Quality、3K/2023 Quality 与 Align 的 PLCC/SROCC，保留失败结果；勿混用文本前缀与 RQ1 检查点。
4. 对主体/动作困难负例做额外诊断，防止只提升总体相关却仍不懂 Prompt。若文本实验有效，再考虑按图像变化的条件词元或分句/实体动作关系；若无效，先查 Prompt 截断、文本敏感性及该前缀是否只学到质量捷径，不直接叠加更多模块。

## 新增 RQ1 seed123 成对复核：运行与后续下游

两组预训练已启动；以下命令仅供核对，**不要再次执行或共用已有 output-dir**。两组结果 CSV 分开写，避免同时运行时相互追加：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_pretrain.py --config configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml --seed 123 --fixed-epochs 17 --output-dir outputs/rq_diagnostics_v1/rq1_pretrain_full_seed123 --results-csv outputs/rq_diagnostics_v1/all_results_rq1_pretrain_full_seed123.csv
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_pretrain.py --config configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml --seed 123 --fixed-epochs 17 --disable-regular-simsiam --output-dir outputs/rq_diagnostics_v1/rq1_pretrain_no_regular_simsiam_seed123 --results-csv outputs/rq_diagnostics_v1/all_results_rq1_pretrain_no_regular_simsiam_seed123.csv
```

待**两组均有 `final` 行和 `best_quality_pretrain.pth`** 后，以下各条指令分别在一个数据集上依次运行 full / no_regular_simsiam 两个下游；不指定 GPU，输出目录由脚本防覆盖：

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/run_rq_diagnostics.py downstream --dataset 1k --seed 123 --pretrain-seed 123 --output-root outputs/rq_diagnostics_v1 --execute
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/run_rq_diagnostics.py downstream --dataset 3k --seed 123 --pretrain-seed 123 --output-root outputs/rq_diagnostics_v1 --execute
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/run_rq_diagnostics.py downstream --dataset 2023 --seed 123 --pretrain-seed 123 --output-root outputs/rq_diagnostics_v1 --execute
```

注意：`--seed 123` 控制训练随机性，源 YAML 的 `split_seed: 42` 仍保持固定；`--fixed-epochs 17` 关闭早停但不把原 50 轮 cosine 调度周期改为 17。脚本统一使用验证质量选择的 `best_quality_pretrain.pth`，不可从下游测试结果反选预训练检查点。

## 2026-10-01：五级质量词 Clean View 实验（已实现，未运行）

- 只借鉴本地 PDF 的五个固定质量词 `terrible / poor / fair / good / excellent`，不是复现其三重 Prompt 或立体图像结构。新增 `QualityTextPrototypeBank`：冻结的原始 CLIP 文本编码产生五个短句原型；仅训练图像质量 SimSiam 256维投影到 CLIP 文本空间的映射。**不使用真实生成 Prompt/Answer，也不使用四个可学习文本前缀**；这四个词元继续只服务原 Prompt/Answer 路径。
- `train_cm_pretrain.py --quality-text-levels` 启用 Clean View 的五级软标签交叉熵（权重 0.005），SafeAug 两视图和原四项 SimSiam 不变；现有 Artifact/Aesthetics 数值锚点保留，总体数值锚点仍关闭。五级中心是 RichHF 官方 train CSV 中 answer_status=ok 的总体质量分数 5/25/50/75/95 分位数；写入有效配置与检查点，不使用验证/下游标签设定中心。跨预训练数据集复用前必须重估中心。
- 精简的 `best_component_anchor_pretrain.pth` 已包含质量文本映射、固定短句向量和中心。下游 `train_cm_iqa.py --quality-text-blend 0` 是“仅预训练质量表征”消融；`--quality-text-blend 0.10` 将五级词推断的质量分数以 10% 融入质量输出。Align 与两输出定义不变；0.10 是预设实验值，不是已证明最优值。下游还输出文本单独分数和视觉分数的验证指标。
- 当前没有启动这项预训练或下游，也没有结果。新增 `tests/test_quality_text_bank.py`，全套 71 项单元测试通过；此结果只证明接口与数学实现，不证明 IQA 指标改善。当前已有其他下游任务在运行，不得覆盖其目录。


## 2026-10-01：五级图文一致性词实验（已实现，未启动）

- 与五级质量词预训练**分开运行**，不覆盖其输出。CLI：预训练 `--align-text-levels`；下游 `--align-text-blend 0`（仅迁移表征）或 `0.10`（将文本五档分数以预设 10% 混入 Align 输出）。不启用开关时原实验默认路径不变；SimSiam/质量损失和双输出协议保留。
- 固定五句 `the match between the image and its prompt is terrible/poor/fair/good/excellent` 用冻结基础 CLIP 编码；真实生成 Prompt 同图像全局/局部证据先形成配对特征，再映射到五句空间。因此预测条件同时含图像和真实 Prompt；五句本身并不逐样本拼接 Prompt。原四个可学习前缀词元仍用于真实 Prompt 的编码路径。
- Clean View 上用人工 `align_score` 对五档做相邻软标签 CE（预设权重 0.005）；低一致性样本与其他样本等权。五个中心固定 `[0,0.25,0.5,0.75,1]`，覆盖训练人工标签完整范围；RichHF 官方训练集 `answer_status=ok` 的 15810 行中有 568 行低于 1/3，不能采用最低中心 1/3 的设计。SafeAug1/2 仍只服务常规 Image–Image SimSiam。
- 新分支权重和固定文本向量保存在精简检查点；新实验另按 Clean View 的验证集 `align_text_SROCC` 预先选出 `best_align_text_pretrain.pth`，不会改变质量实验的原选点规则。下游 representation scope 严格核验中心与必需键。新增验证指标 `align_text_PLCC/SROCC`、下游 `align_base` 和 `align_text` 指标；尚无训练结果，不能宣称提升。新增 `tests/test_align_text_bank.py`；当前回归测试 75 项通过。若并行运行，给一致性实验单独的 `output-dir` 和 `results-csv`，勿覆盖质量词实验。

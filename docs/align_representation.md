# 一致性表示与验证选模对照（独立实验，未更换基线）

本轮不重新预训练、不恢复负关系监督，也不修改旧结果。仅 AGIQA-3K、AIGCIQA2023（1K 无一致性标签，不重跑）。训练 seed=42、split_seed=42、60 轮、保留 train/val/test。全部从当前 v2 原基线预训练权重初始化，SHA256 固定为 `fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686`。沿用原学习率、损失、6维 NSS、5个上下文词元及下游评分头重置策略。各次实验输出目录包含独立的 aprobe/aregionword 标识；新旧成绩分别报告。

## 两种模型，每种两个数据集：共四次下游训练

- **aprobe**：保持原模型，加入只观察的梯度记录和同轨迹验证选模比较。
- **aregionword / clip_region_word_v1**：仅将局部一致性证据换成 CLIP 区域—Prompt BPE token 匹配。质量分支计算不改；共享 CLIP 仍被两个任务更新，因此不保证训练后的质量成绩不变。

每次训练都在同一条 60 轮轨迹上分别保留最大验证 Quality SROCC、最大 `(val Quality SROCC + val Align SROCC)/2` 对应的模型内存快照。严格要求验证集有一致性标签；并列时保留最早轮次。两个选模模型只在训练结束后各评估一次测试集（相同轮次只评估一次）；不根据测试成绩选轮次。不合并标签、不产生新的 Final MOS。旧的 hmean 比较模式仍保留默认行为；本实验显式 `quality_mean_v1`。

## 梯度观察

第1、10、30、60轮各前两个训练批次，使用同一个前向结果的加权 Quality task、Alignment task 目标，分别记录 CLIP 图像编码器、ResNet 局部编码器、生成 g 的全局投影的梯度范数、Alignment/Quality 范数比、余弦、单侧/共同/未连接参数张量数量。记录 `task_gradient_probe.jsonl` 并纳入打包。

`autograd.grad` 不写参数 `.grad`，不修改优化器，不启用 PCGrad，不增加前向、不另采样、不改变模型模式。AMP时使用当前GradScaler缩放计算梯度，再还原范数，降低未缩放半精度梯度下溢的影响；记录缩放值。图和正常反向传播保留。它增加少量批次的反向开销与峰值内存；不承诺 GPU 训练逐位复现。余弦为负仅说明该批次的梯度方向冲突，不能单凭这一点认定泛化下降的原因；零范数时余弦记 null。不能把所有 Fusion 名下参数混成一个结论，此处仅记录 semantic_proj。

不在随机初始化的下游头上作“已训练模型冲突”判断。默认不再单独跑短程参数对照，诊断随四次正式服务器下游训练记录。

## 新局部一致性路径

```text
同一次 CLIP ViT-L/14 前向
  ├─ 原 pooled 全局特征768 → 原全局投影256 → 原 PI/AG → 原全局一致性头
  └─ 256个区域token ×1024 → 新Linear(1024,256) → L2归一化
Prompt（SOT后5个可训练上下文）
  ├─ 原 pooled 表示 → 原 PT/AT → 原全局一致性头
  └─ 冻结CLIP文本Transformer的LN后token ×768 → 新Linear(768,256) → L2归一化
区域与每个有效文本token逐一cos → 沿256个区域softmax → 每个文本token的区域证据
有效token平均：区域证据e、文本表示t
concat[e,t,|e−t|] → 原局部一致性评分头 → 局部分数
原样本级门控 → 全局/局部分数加权 → Alignment分数
```

文本有效位置排除 SOT、5个上下文位置、EOT和padding；内容粒度是 **BPE子词**，不是人工标注的词。上下文位置不直接参与匹配，但通过冻结文本Transformer继续获得梯度。空Prompt报错。空间softmax温度固定1，未新增温度搜索。CLIP区域来自 ViT，不是 ResNet 的196个局部token。无跨注意力模块、无融合z、无Transformer解码器。

质量路径仍是 CLIP全局投影g、ResNet局部P₀、原QA/QL/NSS/评分和分数混合。局部一致性不再执行 AP、WI、WT；这些旧模块仍完整接收旧表示权重，但在新实验中冻结并排除优化器，不再形成局部一致性的ResNet梯度。全局一致性依旧使用PI、AG、PT、AT。

原检查点的全部规定表示参数仍由原严格加载器核验并迁移，不用其他项目检查点。新增的两个投影**明确随机初始化**，不能声称已预训练；记录参数名、形状和初始张量SHA256。旧评分头仍按原下游协议重置。新增参数沿用 align 组初始LR=1e-4；全局投影1e-5，PI/PT 1e-4、上下文1e-5保持原样。初始化隔离随机数，使原有模块及评分头初始化不因增加投影而变化。新模式只支持标准CLS池化OpenCLIP ViT、下游双任务projection_only_v1；不允许叠加历史跨注意力、V-Mamba、局部温度消融和门控消融。

## 服务器命令

先同步（工作区有改动时先保留）：

```bash
git status --short
git pull --ff-only origin work
```

只检查配置，不读模型权重、不训练：

```bash
bash scripts/run_align_representation.sh all --dry-run
```

建议先基线诊断/选模对照，再看新区域匹配结果。两卡可并行，各自两个数据集依次执行：

```bash
# 终端一：原模型+梯度诊断+双验证选模（3K、2023，共2次）
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_representation.sh probe
# 终端二：CLIP区域/BPE匹配+梯度诊断+双验证选模（3K、2023，共2次）
CUDA_VISIBLE_DEVICES=1 bash scripts/run_align_representation.sh region
```

单卡顺序执行全部四次：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_representation.sh all
```

只运行某一个实验/数据集：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_representation.sh aprobe agiqa3k
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_representation.sh aregionword aigciqa2023
```

只有已完整通过核验的实验可以通过 `--continue` 跳过；不恢复、不覆盖半成品。脚本训练前核对原权重SHA/完成状态和历史对照，CPU拒绝真实训练。AGIQA3K旧对照未记录成员清单时，仍只能确认其配置/运行seed；本次记录实际划分，不伪造旧清单。

全部结束后打包（不包含大模型权重）：

```bash
bash scripts/run_align_representation.sh pack
```

输出 `projection_only_v1_v2_align_representation_results.tar.gz`，包含四次CSV、配置、元数据、训练日志、划分、来源指纹、梯度JSONL和旧对照资料。核验两种策略是否选对轮次、测试结果是否与最终事件一致；不混用旧Quality选模结果和新joint选模结果。仍不保存下游大模型权重，不支持训练中断续跑。本轮成功与否须由实际服务器结果判断，尚未验证性能提升。

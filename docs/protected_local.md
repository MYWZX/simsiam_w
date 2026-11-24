# 成熟全局路径保护与局部预训练对照

版本：`projection_only_v1_v2_protected_local_v1`。这是独立新实验，不替换当前最好基线，也不把旧结果标为本版本结果。仅对有一致性标签的 AGIQA-3K 和 AIGCIQA2023 运行；1K不重跑。云端CPU仅做合成测试，脚本拒绝CPU真实训练。

## 三个阶段及预算

1. **Foundation：两个数据集各60轮。** 沿用当前v2最好基线配置，seed=42、split_seed=42、train/val/test不合并，最大验证 Quality SROCC 选模。唯一新增行为是结束后保存选中模型的**完整参数和缓冲区**到 `best_selected_downstream.pth`。CLIP文本权重、5个上下文、PI/PT、QA/QL、NSS、全部评分头和混合参数都包含在内，固定归一化等非持久注册缓冲区也显式保存和恢复。旧结果包只有CSV，没有这些下游权重，不能从旧CSV直接恢复成熟评分头，因此本阶段需要重新运行。它是新的同批对照，成绩不保证与历史基线逐位一致。
2. **可选局部路径预训练：RichHF原始官方train/val各自独立，固定20轮。** 从原最好基线的项目预训练权重初始化，冻结所有继承模块，仅训练两个新投影和局部一致性评分头；只使用clean view及人工图文一致性评分。最大验证**局部** Alignment SROCC 选模，不评估RichHF测试分数。它是“新局部路径的预训练热身”，不是从原始CLIP重新做完整四关系SimSiam联合预训练，不修改既有质量/OAS预训练。保存 `best_local.pth`，其中只有局部模块；下游**仅迁移两个新投影**，RichHF评分头明确丢弃，使用成熟下游自己的局部评分头。
3. **受保护的下游局部训练：每组每数据集20轮。** 从同一个Foundation选中检查点完整加载，比较下面三组；都最大验证 Alignment SROCC 选模，训练结束后才评估测试一次。质量损失及扰动损失系数置零，只优化原Alignment MSE/rank/Soft-SROCC（原权重、tau、margin不额外搜索）。全局分数和质量分数继续计算、报告，但所有相关参数及运行统计冻结。

| 组 | 可训练模块 | 新投影来源 |
|---|---|---|
| `old`：预算对照 | 原AP、WI、WT、局部评分头、样本级一致性门控 | 无新增投影 |
| `scratch`：直接新路径 | 新区域/文本投影、局部评分头、一致性门控 | 显式随机初始化 |
| `warm`：局部预训练后迁移 | 与scratch完全相同 | RichHF选中检查点的两个投影 |

三组都继承Foundation的成熟局部评分头和门控，然后允许它们更新；仅warm使用RichHF投影，不能混入RichHF头或其他全局权重。预训练只针对局部评分，门控和全局评分头不参与更新。旧局部模块在新区域路径中不执行，保留原权重并冻结。

共 **2次Foundation + 1次局部预训练 + 6次下游局部训练 = 9次运行**。每个下游方案付出60轮Foundation + 20轮局部训练；warm另有独立的RichHF 20轮成本。原60轮历史结果不构成匹配预算对照，应首先与同批Foundation及old组比较。会记录实际批次、成功更新和AMP跳过次数，不能只按名义轮次宣称预算完全相同。

## 冻结与兼容性

冻结：CLIP图像、完整ResNet、全局投影g、独立局部P₀投影、PI/PT、文本编码器、5个上下文、AG/AT、全局一致性头、全部质量QA/QL/NSS/评分头及其混合参数。冻结同时控制 `requires_grad`、BatchNorm运行统计和Dropout模式；仅表中局部模块切回train模式。参数及缓冲区在选模恢复前后的SHA256摘要必须相同。验证集质量及全局一致性指标必须保持不变（绝对容差1e-6），并与Foundation选中记录对照。最终测试的质量及全局一致性指标也会核验。冻结后算子可能选择不同实现，不承诺GPU逐位相同。

完整Foundation加载不走旧的“仅表示迁移、重置评分头”逻辑。所有原参数必须完整、形状匹配；新模式只允许两个新投影缺失，并明确记录初始化。局部预训练迁移对两个投影严格检查键和形状，明确报告未迁移的RichHF评分头。来源模式、run_id、训练完成状态、配置、选模轮次、数据集、seed/split及权重SHA256不符时拒绝运行；不resume旧优化器。

Foundation的原参数分组与学习率不改：全局投影1e-5、PI/PT1e-4等沿用基线。后续局部阶段仅有align组可训练，初始LR=1e-4、weight decay=.002、cosine min LR=1e-6，固定20轮。冻结模块不会误入其他组。

整个方案仍是projection_only_v1，没有跨注意力融合z。新增区域路径沿用上一轮CLIP patch/BPE-token的cosine匹配与空间softmax，质量路径保持原6维NSS。局部热身不使用Answer置信度、负关系、质量文本CE，也不把局部标注误称为SimSiam负样本。

## 执行

工作区有源码改动时先保留；从仓库根目录同步：

```bash
git status --short
git pull --ff-only origin work
```

只看全部9份配置，不加载权重、不训练：

```bash
bash scripts/run_protected_local.sh all --dry-run
```

**单卡顺序完成全部流程（包含局部预训练）：**

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_protected_local.sh all --continue
```

**两卡分工：**

```bash
# 终端一／GPU0：Foundation + old/scratch，两数据集依次跑，共6次运行
CUDA_VISIBLE_DEVICES=0 bash scripts/run_protected_local.sh direct --continue
# 终端二／GPU1：仅RichHF局部预训练，共1次运行
CUDA_VISIBLE_DEVICES=1 bash scripts/run_protected_local.sh warmup --continue
```

以上两条都结束后，再运行warm组两个下游。不要在Foundation尚未完成时启动warm，也不要重复启动Foundation：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_protected_local.sh warm --continue
```

可单独执行 `foundation`、`old`、`scratch`、`warm`，用 `--dataset agiqa3k` 或 `--dataset aigciqa2023` 限定数据集。`warmup`固定RichHF、不需要dataset参数。`--continue`仅核验并跳过完整完成的运行；半成品仍拒绝，不删除、不覆盖、不自动补写完成清单。训练失败时保留日志及所有文件，不能通过删除目录掩盖失败。源检查点改变时拒绝继续。

全部9次完成后打包：

```bash
bash scripts/run_protected_local.sh pack
```

输出 `projection_only_v1_v2_protected_local_v1_results.tar.gz`。包含9份CSV、配置快照、元数据、实际划分、训练日志、来源指纹、检查点SHA256清单和冻结证明，**不包含pth权重**。大权重留在服务器新输出目录：两个完整Foundation以及各组的小型局部检查点。局部检查点不能脱离其指定Foundation独立使用。

当前代码实现与测试不代表性能已提升；需要服务器实际结果，按同批Foundation/old/scratch/warm分别报告Quality和Alignment的SROCC/PLCC。

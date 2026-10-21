# 当前代码审查与框架核对

审查基线：`eecf914ad00391a0d2ec5c867d6ded9d633a5d77`（work）。
范围：当前 Large / residual_feature / class-context Quality bank 主线，两个训练入口、损失、加载器、数据输入、优化器、选模及运行溯源。
本次仅添加审查文档和框架图，不修改模型、损失、依赖或已有实验结果。严重度指潜在影响，不代表已证明导致性能下降。

## 1. 优先处理：OpenAI 权重与激活函数配置不一致

位置：[视觉构造器](../../models/backbones.py#L137)、[文本构造器](../../models/two_stage_models.py#L69)。两处都没有设置 `force_quick_gelu`。

在 OpenCLIP 3.3.0 上核对结果：

- `get_model_config('ViT-L-14')` 的 quick_gelu 默认 False；Base 同样如此。
- `get_pretrained_cfg('ViT-L-14', 'openai')` 的 quick_gelu 为 True。
- meta-device 构造原生 ViT-L-14 后，实际有 36 个 GELU、0 个 QuickGELU，无需下载权重。
- OpenCLIP factory 对这种组合发出 mismatch warning，但不会自动设置正确激活函数。

**结论：当前构造方式与权重标签声明不一致，属于明确的初始化兼容性风险。** 上传运行记录同为 OpenCLIP 3.3.0，但服务器 dirty 工作区的具体改动未归档，仍不能把本地观察直接当成服务器实际模块取证。该差异不是此次跨运行差异的已证实原因。

建议：在未来新预训练前打印视觉和文本实际激活类型，并显式记录模型变体。若采用 QuickGELU，两个阶段和两个塔保持一致。不得直接给已经训练好的 checkpoint 切换激活函数并把新结果当作原方法续跑；参数 shape 匹配不代表函数一致。本次保留现有行为。

## 2. 优先处理：加载检查不完整，strict 参数也不严格

位置：[load_pretrained_backbone](../../models/two_stage_models.py#L1668)。

当前视觉参数仅在“同名键存在且 shape 不符”时报错；缺失的视觉键不会被拒绝。Adapter 只检查是否加载了任意一个 `quality_adapter.*` 键，不检查该模块完整性。`strict=True` 在末尾仍只是打印 missing/unexpected，没有抛错。

已用小模型与模拟 torch.load 复现：checkpoint 仅含一个 `backbone.global_encoder.weight`，representation 加载仍成功，local encoder 完全保持初始化。主程序在本地 checkpoint 存在时还会关闭视觉权重的在线预训练初始化，因此不完整加载可能留下随机视觉参数。

**结论：这是可复现的加载完整性漏洞；不表示当前 Large 文件已损坏或漏载。** 两组现有 SHA256 相同只证明输入相同，不能证明所有预期键被加载。

建议：按 scope 定义应完整恢复的模块，输出每模块 missing / mismatch / loaded 参数数量；关键视觉模块和启用的 Adapter 拒绝部分恢复。representation 故意丢弃评分头，因此不能简单对整个模型使用 strict=True，需按模块检查。

## 3. 中等：检查点直接映射到 GPU，增加加载峰值显存

位置：[torch.load](../../models/two_stage_models.py#L1717)。模型已 `.to(device)` 后，checkpoint 又按 model_device 加载。同一时刻存在目标模型及来源 checkpoint，完整预训练 checkpoint 还可能带 optimizer 状态。

这是代码可见的显存峰值风险，本机 CPU 环境未测 GPU 峰值。建议来源 checkpoint 在 CPU 读取，再将选中的 tensor copy 到目标参数；结合第 2 项改动单独验证。普通模式的最佳模型仍 `copy.deepcopy(model.state_dict())`（[训练循环](../../train_cm_iqa.py#L2094)），GPU 模型会额外保留一套 GPU 权重；双选模模式已使用 CPU 独立快照，不受此问题影响。

## 4. 中等设计问题：主线不是完全相同的 Quality 损失系数

三个下游结构统一，但 YAML 中 `lambda_srocc_quality`：1K 为 0.03，3K/2023 为 0.02。1K 无 Align 标签而关闭 Align 是合理的标签条件处理；Quality soft-SROCC 系数不同则是另一个独立超参数差异。

位置：[1K](../../configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml#L117)、[3K](../../configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml#L124)、[2023](../../configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml#L129)。

如果论文宣称“三个数据集完全同一训练配方”，应披露该差异或事先制定统一值。结构统一的说法仍成立。不要为了修辞立即改动并要求重跑已有基线。

## 5. 中等设计问题：Prompt 重复投影、Answer 路径不对称

位置：[三视图](../../models/two_stage_models.py#L1488)、[重复 Prompt 投影](../../models/two_stage_models.py#L1507)，双视图也有相同调用模式。

相同的 `align_text_feature` 连续调用两次 `_project_text`，分别用于全局和区域关系。该 projector / predictor 包含 BN，projector 还有 dropout。小模型 hook 复现：一次三视图 forward，text_projector 调用 3 次（Prompt 两次、Answer 一次），前两次输入完全相同，BN counter 增加 3。

这两份 Prompt 结果并非死代码：区域 pull–push 会消费 `align_prompt_*`。因此不能直接称第二次前向无用；但它确实让同一 Prompt 以不同 dropout 实现投影，且按 Prompt:Answer=2:1 更新 BN 统计。Answer 只走 text_to_fused，Prompt 已走 align_text_adapter；两个语义路径也不完全对称。

这是需说明意图的训练设计，不是已证实错误。若目标是同一 Prompt 表征用于全局/局部关系，可在未来受控改动中复用投影；这会改变 dropout RNG 与 BN 更新，不能称为“完全不影响轨迹的优化”。不建议为了清理代码直接改现有预训练。

## 6. 较低：NSS 六维中两个通道完全重复

位置：[NSS 构造](../../models/two_stage_models.py#L932)。`sigma_proxy = pooled_var`，拼接时既包含 pooled_var 又包含 sigma_proxy。小模型验证两个输出通道逐元素相等。

当前实际至多五种不同的统计量，不能表述为六种独立 NSS 描述符。它不是精确 GGD/NSS 拟合，代码已有 proxy 说明。重复通道不意味着模型一定出错，也不是独立创新。可修正文档；若替换为新的统计量，会改变 checkpoint 输入语义，应另做实验。

## 7. 设计限制：五级 Quality 监督与各数据集 MOS 分布

Quality bank 的固定中心来自 RichHF，范围约 [0.375, 0.9583]。下游 `_compute_text_bank_aux_loss` 会把端点以外标签饱和到首/末类别，类内的更细粒度次序由主评分头的 MSE / ranking 承担。

blend=0 时这一范围不限制最终 Quality 输出；仍存在 CE 梯度。不能把五级辅助分数范围与最终 Quality 范围混淆。此前训练集分位数重校准未显示统一收益，不建议重复要求该实验，也不能单凭 loss 数值大就删除五级分支。下一步诊断应关注共享参数上目标梯度的方向与范数，而非堆叠新结构。

## 8. 溯源限制：dirty=true 不知道具体改了什么

当前 run_metadata 记录 commit 和 dirty 布尔值，但没有记录变更路径或代码差异摘要；仅双选模入口写这些信息。已上传两组运行都 dirty=true。不能据此判断是模型源码被修改，还是仅有无关的未跟踪文件。

现有证据足以支持同一 run 内两种选模策略的比较，不足以完整重建服务器 dirty 工作区。未来可以附加非敏感源码路径/摘要、模型激活配置和数据 split 指纹；不应记录密钥或整个进程环境。当前两次同轨迹比较不需重跑。

## 当前检查未发现的明显错误

- ImageNet 输入在 CLIP 视觉支路内转换到 CLIP normalization；ResNet 沿用 ImageNet normalization。
- 真实 Prompt 参与 Align，不进入当前 quality_branch_use_text_gate=false 的 Quality 主评分。
- frozen CLIP 文本 Transformer 的权重冻结，但可学习共享 contexts、class contexts 仍能通过它回传梯度。
- residual_feature 保留 z 旁路，alpha 由 sigmoid logit 得到，尺度 norm 被 detach；质量文本辅助仍使用 Adapter 输出，而非残差 r。
- beta 与 alpha 是不同参数；beta 融合 patch/global 质量分数。
- representation scope 有意重置任务评分头，保留视觉、Fusion、文本 contexts/projection、Align adapters、Quality bank，并将 quality projector 迁移至 Adapter。
- 1K 只计 Quality 标签；3K/2023 各自报告 Quality / Align，不用合成 Final MOS 作主监督。
- 双选模每轮共用一次验证、训练结束才测试，不用测试指标选点，CPU 快照包括独立 tensor copy 和 state_dict metadata。
- 旧错误负样本过滤、整图安全输入、3K crop 接通、验证损失统计的先前修复仍在。当前正式运行关闭 legacy negatives / perturbation，未启用可选 attention / V-Mamba / high-res crop / Align text bank。

## 验证与限制

- 104 项现有测试通过（CPU）。
- 小模型复现 partial-load 接受、NSS 重复通道、Prompt 重复投影 / BN 更新。
- 本地 OpenCLIP 3.3.0 配置与 meta-device 原生 Large 模块核查了 GELU mismatch；未下载新权重、未改激活函数。
- 没有在此云环境运行真实数据/GPU 训练，不声明上述潜在问题已导致特定分数差距。
- 框架图：[当前主线](../figures/aigc_iqa_current_framework.png)，细节依据：[图示说明](../figures/aigc_iqa_current_framework.md)。

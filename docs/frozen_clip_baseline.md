# 冻结 CLIP 图像编码器的独立基线对照

仅在预训练与下游同时将 `model.global_encoder.freeze` 从 false 改为 true。
预训练复制 projection_only_v1_baseline；三个下游复制 current_tuning_reference v2。除冻结开关、独立运行路径及本组检查点来源外，完整配置逐项相同。
这不是 Prompt 权重0.20/0.30实验的延伸：I-I=0.15、图像—Prompt=0.25、图像—Answer=0.20、Prompt—Answer=0.05，Anchor各0.01、Ordinal各0.015。

## 冻结范围

- 冻结原始 OpenAI CLIP ViT-L/14 图像编码器：`requires_grad=False`，不进入优化器，不更新权重。
- 复用源码现有 `ImageTextBackbone.train()` 规则：父模型 `train()` 后视觉塔仍保持 `eval()`，不更新其 BN 统计或启用其 Dropout。不是仅把LR设为0。
- CLIP ViT-L/14 视觉输出为768D，与包装器output_dim相同；包装器没有新增需要单独训练的随机输出投影。
- `backbone.fusion.semantic_proj` 是独立的全局Linear投影，**仍可训练**；没有冻结整个g路径。P₀局部投影、ResNet、PI/PT、QA/QL、NSS、AG/AP/AT及所有当前评分/预测模块仍按基线训练。
- 原CLIP文本编码器权重保持冻结，5个上下文词元仍可训练。
- 保留现有所有层级LR及weight decay配置。被冻结的global_encoder组自然从有效优化器组中消失；其余组的参数归属、LR和weight decay不变。全局semantic_proj初始LR仍1e-5，PI/PT仍1e-4。
- 不修改模型核心源码或历史配置；使用已有冻结机制。

## 训练与来源协议

本组从原始CLIP/ResNet预训练初始化重新预训练，不加载本项目旧预训练/下游权重、不resume。
预训练保持官方train/val，seed=42、split_seed=42，50轮预算、patience=15，按最大val_component_anchor_SROCC选模。
随后三个下游统一加载本组 `best_component_anchor_pretrain.pth`，保留train/val/test、60轮、按最大val_quality_SROCC选模。
训练seed=42，1K subset1078/split3407；3K和2023 split42；Quality SoftSROCC系数0.03/0.02/0.04。quality text-bank CE与negative仍关闭。

来源验证沿用现有pipeline：配置、模式、独立run_id、完成记录、权重SHA256、加载后来源记录均校验。旧的可训练视觉塔基线不能混入本组pipeline。
已有目录或CSV拒绝覆盖；失败停止后续阶段并保留产物；`continue`只跳过通过验证的完整阶段，不自动恢复不完整训练。
1K/2023严格要求划分清单；3K若原配置未记录实际成员，只核对配置与runtime seed并明确报告，不伪造清单。
HF_HUB_OFFLINE默认1，使用已有原始模型缓存；云端CPU拒绝真实启动。

## 服务器命令

先同步：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && git pull --ff-only origin work
```

在空闲GPU执行（此处以GPU0为例）：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=0 bash scripts/run_frozen_clip.sh all
```

`all` = 本组重新预训练 → 1K → 3K → 2023；共1次预训练、3次下游。
入口默认仅预训练；可将 `all` 换为 `pretrain`。还支持单数据集、`downstream`、`continue`、`--dry-run`。

完成后打包：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && bash scripts/run_frozen_clip.sh pack-downstream
```

生成 `projection_only_v1_ctx5_frozen_clip_downstream_results.tar.gz`。
包含CSV、配置、元数据、日志、完成记录与来源指纹，不含大模型权重；已有包拒绝覆盖。
沿用现有打包白名单，不含独立data_split_manifest.json；1K/2023的成员指纹摘要保存在run_metadata.json。

结果独立报告，当前最好基线不变。冻结改变有效优化参数与train/eval行为，不承诺训练逐位复现或必然提高成绩。

## 验证

完整回归：`PYTHONPATH=.:tests /workspace/.venvs/simsiam/bin/python -m unittest discover -s tests`，341项通过。
新增6项覆盖：冻结参数/BN状态不变与父train模式保持、非冻结分支和上下文梯度/实际优化步骤、非冻结优化器组完整一致、权重迁移与同权重eval输出一致、四份配置仅冻结开关差异、CPU拒绝/失败停止/不覆盖/本组三个下游来源与打包。
小模型使用伪编码器，pipeline使用伪产物；未下载或训练真实CLIP/ResNet。测试隔离并恢复torch随机状态，不影响其他用例。`all --dry-run`及shell语法检查通过。

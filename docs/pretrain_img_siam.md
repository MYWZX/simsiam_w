# projection_only_v1 I-I SimSiam 预训练权重比较

当前最好基线保持不变；上一轮 Anchor/Ordinal 实验没有整体提升，新的两组回到原始 Anchor=0.01、Ordinal=0.015（artifact/aesthetics 各自）。
本轮只比较预训练 lambda_img_siam=0.10（siam010）、0.20（siam020），原基线为0.15。
保留 I-I 目标，不改变 g→PI、图像—文本关系、P₀独立投影、5个上下文词元、QA残差、NSS、Component Anchors/Ordinal、PCGrad及所有其他目标。
新实验从原始 CLIP/ResNet 初始化，不载入旧项目预训练、不 resume；保留官方 train/val、seed=42、split_seed=42、50轮预算、patience=15、val_component_anchor_SROCC最大选模。
默认入口仍只预训练；显式 all 才自动执行本组预训练及三个下游。

下游严格复制 current_tuning_reference v2，仅变更来源检查点和运行路径：训练seed=42；1K subset1078 split3407；3K/2023 split42；2023质量SoftSROCC=0.04。
60轮、train/val/test独立、按val_quality_SROCC最大选模。PI/PT初始LR=1e-4、全局投影=1e-5，优化器分组不变。
每组权重来自本组 best_component_anchor_pretrain.pth；核验配置、完成记录和来源SHA256，独立输出，既有结果拒绝覆盖。
复用修复后的划分核验：1K/2023要求清单；3K未启用成员清单记录，核验配置和运行元数据seed并明确报告限制，不伪造清单。

先同步一次：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && git pull --ff-only origin work
```

终端一，GPU0：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_img_siam.sh siam010 all
```

终端二，GPU1：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=1 bash scripts/run_pretrain_img_siam.sh siam020 all
```

共两次预训练、六次下游。GPU编号可调整；任一步失败停止，保留结果。云端CPU拒绝真实启动。
HF_HUB_OFFLINE默认1，要求本地缓存原始模型权重，不自动替换预训练来源。
支持 `pretrain`、`downstream`、单个数据集、`continue`、`--dry-run`。
`continue`仅跳过通过配置、哈希及完成状态检查的阶段，启动尚无输出的阶段；不自动恢复不完整训练。

全部结束后打包：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && bash scripts/run_pretrain_img_siam.sh siam010 pack-downstream && bash scripts/run_pretrain_img_siam.sh siam020 pack-downstream
```

生成 projection_only_v1_ctx5_img_siam010_downstream_results.tar.gz 和 projection_only_v1_ctx5_img_siam020_downstream_results.tar.gz。
包含预训练与三个下游的CSV、配置、日志、运行元数据和来源指纹，不包含模型权重。
原打包白名单不含独立data_split_manifest.json，1K/2023的成员指纹摘要在run_metadata.json中。

结果作为独立实验报告；原最好基线及其成绩不变。单seed比较不代表多seed稳定提升；不保证训练逐位复现。

本轮结果已核查，0.10/0.20均未超过当前基线，继续保留I-I=0.15。详见 [结果记录](pretrain_img_siam_results.md)。下一轮单独比较 [图像—Prompt权重](pretrain_img_prompt.md)。

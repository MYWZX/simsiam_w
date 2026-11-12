# CLIP视觉塔：预训练/下游分阶段冻结对照

只改变CLIP图像编码器在不同阶段是否可训练。文本塔原权重始终冻结、5个上下文词元始终可训练；ResNet、全局semantic_proj、PI/PT、QA/QL等仍按原基线训练。

| 条件 | 预训练CLIP图像塔 | 下游CLIP图像塔 | 预训练来源 |
|---|---|---|---|
| 当前基线（已有） | 可训练 | 可训练 | 当前最好基线的固定预训练 |
| pretrain_only | 冻结 | 可训练 | 与both共享冻结预训练 |
| downstream_only | 可训练 | 冻结 | 当前最好基线的固定预训练 |
| both | 冻结 | 冻结 | 原run_frozen_clip实验，原样复用 |

“全部冻结”仅指这两个阶段的CLIP图像编码器，不是冻结整个模型。
完整配置与原基线逐项比较：除global_encoder.freeze、独立输出标识与对应来源路径外不变。I-I=0.15、图像—Prompt=0.25；没有使用Prompt0.20/0.30或I-I0.10/0.20的权重。
冻结为requires_grad=False且父模型train()后维持视觉塔eval()；解冻阶段从已载入权重正常训练。CLIP图像输出包装器没有额外随机投影；semantic_proj保持可训练。除去被冻结的视觉组，其余LR/参数分组不变。

## 为什么复用两个预训练来源

不重新训练三份预训练。使用同一预训练权重比较下游冻结与否，避免源权重差异干扰结论：

- 可训练预训练：锁定docs/current_tuning_reference.json记载的原最好基线来源。校验版本、配置SHA、权重SHA、选中epoch和completion SHA，**不允许静默换一份重新训练的来源**。缺失则报错，不自动重新预训练。
- 冻结预训练：沿用configs/pretrain_frozen_clip/frozen/pretrain.yaml及旧输出目录。已有完整记录/权重则校验并复用；完全没有输出时才从原始CLIP/ResNet初始化训练一次；已有不完整产物则停止保留。
- pretrain_only与both的预训练SHA必须相同；both与之前run_frozen_clip.sh的输出、配置和版本完全一致，不创建新目录冒充旧实验，也不覆盖旧数据。

共3组×3数据集=9次下游；已有both完整结果可直接复用。冻结预训练最多新增1次（若已有则0次），仅下游冻结组不新增预训练。仅下游冻结锁定的是已经经过原预训练更新的CLIP权重，不是换回原始CLIP。仅预训练冻结组在下游会恢复这些参数的可训练状态。
使用旧冻结权重前仍校验其projection_only_v1模式、5ctx、完整配置、选模身份、来源及完成状态。所有下游加载后来源SHA须与对应预训练记录匹配。

## 固定训练协议

预训练：seed=42、split_seed=42，官方train/val分开，50轮预算、patience=15，最大val_component_anchor_SROCC选模。
下游：train_seed=42；1K subset1078/split3407，3K/2023 split42；train/val/test分开，60轮，最大val_quality_SROCC选模；Quality SoftSROCC系数0.03/0.02/0.04。全局投影LR1e-5、PI/PT LR1e-4及其他组保持当前基线。
质量文本五档CE、negative仍关闭，保留双质量残差、NSS、Component Anchors/Ordinal、现有监督。
1K/2023严格校验已有划分清单；3K原配置未启用完整成员记录，只能核对配置/runtime seed，不伪造清单。

## 服务器命令：两张卡完成三组

先同步一次：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && git pull --ff-only origin work
```

终端一，GPU0：先完成/复用both，再运行仅预训练冻结组。

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_freeze_stages.sh both all && CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_freeze_stages.sh pretrain_only all
```

终端二，GPU1：仅下游冻结，使用原最好基线的预训练。

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=1 bash scripts/run_clip_freeze_stages.sh downstream_only all
```

`all`在这个新入口中会**校验并跳过完整阶段**，不会覆盖或重跑已完成阶段；与旧入口严格拒绝已有结果的all语义不同。`continue`等效可用。验证失败或训练失败则停止。
不要同时启动both和pretrain_only去创建同一个冻结预训练目录。推荐以上顺序；若旧冻结任务还在运行，先让它完成。已有部分产物绝不当成完整结果跳过，也不自动恢复其训练。
`pretrain`对pretrain_only/both只运行旧冻结预训练入口；对downstream_only只验证固定基线，不训练。
`--dry-run`只打印计划/配置，**不证明来源文件已完成或SHA有效**；真实运行和打包才执行全部检查。云端CPU拒绝真实启动。

三个结果包：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && bash scripts/run_clip_freeze_stages.sh pretrain_only pack-downstream && bash scripts/run_clip_freeze_stages.sh downstream_only pack-downstream && bash scripts/run_clip_freeze_stages.sh both pack-downstream
```

- projection_only_v1_ctx5_clip_pretrain_only_downstream_results.tar.gz
- projection_only_v1_ctx5_clip_downstream_only_downstream_results.tar.gz
- projection_only_v1_ctx5_frozen_clip_downstream_results.tar.gz（与之前both包相同）

已有同名包拒绝覆盖。已打包的both可保留，确认对应同一完整实验后直接上传原包。
每包包含对应预训练证据及3个下游CSV、配置、日志、元数据、完成记录、来源指纹；不含模型权重。打包白名单沿用旧行为，不含独立data_split_manifest.json；1K/2023摘要在run_metadata.json。
三组独立报告，与已有可训练/可训练基线形成2×2对照。当前最好基线不被替换；单seed结果不代表多seed稳定提升。

## 验证

完整CPU回归347项通过；新增6项覆盖三组单因素配置、共享预训练只运行一次、固定基线来源拒绝替换、三组九次模拟下游与打包、完成/部分产物保护、三种迁移方向的目标freeze状态与梯度。测试隔离并恢复随机状态。三组all --dry-run和shell语法检查通过，未在云端启动真实训练。

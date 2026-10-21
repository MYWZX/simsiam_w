# 预训练 PCGrad 关闭消融

唯一训练处理：`--disable-pretrain-pcgrad` 将 `train.pcgrad.enabled` 改为 false；训练循环从 `_pcgrad_backward` 切换为总损失普通反传。旧基线 YAML 与已训练权重不修改。关闭的是预训练的逐参数 PCGrad，不是下游 global_shared PCGrad。

保留 Large、seed42 / split42、shared context4、Quality class contexts / CE、全部四路 SimSiam、Clean View、人工 / 区域 / 词级监督、Component Anchor、Quality auxiliary、原学习率和选点规则。不要同时修改 QuickGELU、增强、五档中心或其他模块。

## 1. 同步并开始新的预训练

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git fetch origin
git checkout work
git pull --ff-only origin work
```

以下输出路径必须是新路径；不要重用已有基线目录。模型仍由原 OpenAI Large 初始权重开始，不带 `--resume`：

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_pretrain.py \
  --config configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 --quality-text-levels \
  --disable-pretrain-pcgrad \
  --output-dir outputs/richhf18k_quality_classprompt_cliplarge_nopcgrad_seed42 \
  --results-csv outputs/all_results_richhf18k_quality_classprompt_cliplarge_nopcgrad_seed42.csv
```

启动日志应有 `Pretraining PCGrad enabled=False`；有效 config_snapshot 和 CSV run_start 的 train.pcgrad.enabled 应为 false。原规则为最多50轮、cosine周期50、patience15；此命令保留原规则，不额外关闭早停。因此并不保证实际运行满50轮。预训练按验证 `val_component_anchor_SROCC` 选点，不能根据下游测试分数挑预训练 epoch。

需要软时限时，可在同一命令末尾增加 `--max-runtime-hours 5`；这会按 epoch 间软时限停止，不保证精确截止。只有这次 no-PCGrad 自己的完整 checkpoint 才可用于恢复：保持以上全部参数，再加 `--resume outputs/richhf18k_quality_classprompt_cliplarge_nopcgrad_seed42/last_pretrain.pth`。预训练完整恢复权重仍照常原子保存；不要把旧 PCGrad 基线文件拿来恢复本消融。

## 2. 完成后运行三个下游

先确认预训练 CSV 有 final、相应 checkpoint 有效。所有下游统一 residual_feature、初始alpha0.1、正常BN、Quality CE0.005、blend0、整图安全输入、关闭perturbation和旧negative；仅换预训练来源。无须重新训练已有 PCGrad 基线。

### AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_nopcgrad_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_nopcgrad_safeinput_noneg_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_nopcgrad_safeinput_noneg_seed42.csv
```

### AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_nopcgrad_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_nopcgrad_safeinput_noneg_dualselect_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_nopcgrad_safeinput_noneg_dualselect_seed42.csv
```

### AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_nopcgrad_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_nopcgrad_safeinput_noneg_dualselect_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_nopcgrad_safeinput_noneg_dualselect_seed42.csv
```

3K/2023 保留最近已完成对照的60轮同轨迹双选模开关，从中按预先确定的 **joint_selected** 报告主要结果，quality_selected 仅作同轨迹附加记录，不按测试成绩另选策略。1K没有Align标签，按原Quality选模，不使用双选模。

## 3. 比较与证据边界

比较已有 PCGrad 预训练来源 vs 新 no-PCGrad 来源的下游测试 Quality/Align PLCC、SROCC。还应核对有效配置、预训练完成状态和checkpoint选择指标，不仅看目录名。1K 的参照需使用相同 safe-input、无negative配置，不能拿legacy增强或global-only结果混比。

原 Large 基线曾中断并从旧完整checkpoint恢复，缺少完整RNG状态；本次新训练不是对旧中断轨迹的位级复刻。保留这个历史限制，不把单seed差异宣称为统计显著收益，也不要求重跑已完成基线或多seed。

当前云端仅验证CLI和关闭开关，不具备真实数据/GPU；未启动或完成这次预训练。

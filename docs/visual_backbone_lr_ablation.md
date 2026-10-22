# 仅降低视觉骨干学习率的下游消融

新增 `--visual-backbone-lr-mult 0.01`，同时将 global_encoder_mult、local_encoder_mult 从 0.02 改为 0.01。基础 LR=1e-4 时，CLIP 视觉与 ResNet 初始 LR 均为 1e-6，Fusion 仍为 1e-5，Adapter/Quality/Align 仍为 1e-4。文本、权重衰减、调度器保持原配置；未传参数时默认不变。注意余弦调度的 eta_min=1e-6，因此本轮视觉骨干 LR 保持 1e-6，其他组继续原余弦调度；不是整条 LR 曲线都变成原来的一半。

只做这一个消融：原 PCGrad 开启预训练来源、Large、layer3、CE=0.005、Adapter 原学习率、seed/split_seed=42；关闭逐图门控（不传 --quality-sample-gate），不传 --quality-adapter-lr-mult。对照已经完成的原 layer3 基线，不重跑基线。1K 保留质量选模和早停，3K/2023 保留 60 轮双选模、主要报告 joint_selected。保留现有质量/Align分支诊断，不增加损失或保存下游 pth。

配置快照应只新增/覆盖两个视觉编码器倍率（除路径标识），`quality_sample_gate` 不应启用；运行日志应显示 global_encoder=1.00e-06、local_encoder=1.00e-06。预训练来源 SHA256 应为 9a140f598c8bd09812f92e8355dce7148d4b649c89646094823f39f8bcc3bf34。

## 同步代码

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git fetch origin
git checkout work
git pull --ff-only origin work
```

输出目录与CSV带bblr001，保持独立；若路径已存在并含结果，使用新的实验名，不覆盖。以下每条命令在项目根目录执行，不指定GPU。

## AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --visual-backbone-lr-mult 0.01 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_bblr001_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_bblr001_seed42.csv
```

## AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --visual-backbone-lr-mult 0.01 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_bblr001_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_bblr001_seed42.csv
```

## AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --visual-backbone-lr-mult 0.01 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_bblr001_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_bblr001_seed42.csv
```


## 打包结果

```bash
tar -czf cliplarge_layer3_bblr001_results.tar.gz \
  outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_bblr001_seed42 \
  outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_bblr001_seed42 \
  outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_bblr001_seed42 \
  outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_bblr001_seed42.csv \
  outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_bblr001_seed42.csv \
  outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_bblr001_seed42.csv
```

CPU 测试仅验证配置与优化器分组，不代表真实数据性能提升。

# 逐图 Quality 融合消融与分支诊断

默认继续使用标量 beta；`--quality-sample-gate` 开启 beta(x)=sigmoid(beta_logit+MLP([z, sum(weights*patch_tokens)]))。MLP 为 2D→32→1，GELU，无 Dropout。最后一层权重/偏置为零，初始化时精确复现当前标量融合；保留可学习标量。新增模块初始化隔离 RNG，现有头权重保持一致。门控不接收 MOS、文本或测试指标，以质量组学习率训练；不能与 global-only 同时启用。它可能通过共享骨干间接影响 Align。

评估自动增加 quality_global / quality_patch 的 PLCC、SROCC、RMSE 等指标，quality_mix_mean/std/min/max，以及 quality_branch_error_PLCC（两分支相对 MOS 的残差 Pearson；方差为零时省略）。这些只用于诊断，不改变选模。alpha 仍由 quality_adapter_gate 记录；Align 全局/局部指标原已存在。

诊断不增加训练损失或额外模型前向。旧日志缺少实际全局和 patch 质量预测，不能回溯补算；下游不保存权重，所以不能承诺仅靠旧包完成该诊断。本实现不导出逐样本散点数据。

本轮为可选消融，不宣称门控已经有收益。恢复统一 layer3 基线：CE=0.005、Adapter LR=1e-4、PCGrad 开启的预训练权重；不使用 qalr01 降学习率参数。1K 仍质量选模，3K/2023 仍联合选模。对照已有 layer3 基线，不重复训练 baseline，不按测试结果逐数据集挑配置。

## 同步代码

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git fetch origin
git checkout work
git pull --ff-only origin work
```

输出目录与CSV带qgate，保持独立；若路径已存在并含结果，使用新的实验名，不覆盖。以下每条命令在项目根目录执行，不指定GPU。

## AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-sample-gate \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_qgate_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_qgate_seed42.csv
```

## AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-sample-gate \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_qgate_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_qgate_seed42.csv
```

## AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-sample-gate \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_qgate_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_qgate_seed42.csv
```


云端仅做 CPU 功能/回归验证，真实收益需服务器实验；下游继续不保存 pth。

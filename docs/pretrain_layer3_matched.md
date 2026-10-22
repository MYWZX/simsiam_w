# 预训练与下游统一使用 ResNet layer3

本轮只改变预训练局部特征阶段，和已有下游 layer3 匹配。训练代码已支持 `--local-feature-stage layer3`；新增独立 YAML，旧 layer4 配置及权重继续保留。新增启动日志明确显示有效预训练阶段。

- layer3原始1024×14×14 → ResNet.proj 1024→256 → 256×14×14（224输入时196个token）。Fusion、Quality、Align及区域/词级关系均使用该局部来源；不只是Quality换层。
- 新预训练从原OpenAI Large / ImageNet ResNet初始化开始；不是对旧layer4 checkpoint改名字或恢复旧训练。
- 保留PCGrad开启、Large、seed/split42、共享context4、Quality独立class contexts4、质量五档CE0.005、blend0、四路SimSiam、Clean监督、Component Anchor/Ordinal与辅助回归。不启用Align五档CE，不加逐图质量门控，不降低骨干或Adapter LR。
- 预训练仍最多50轮、patience15、原cosine与选模规则；不保证跑满50轮。下游主要使用 best_component_anchor_pretrain.pth。
- 新下游与新预训练都取layer3，局部投影应完整加载。**不要传 --reinit-local-projection**。该参数仅用于旧layer4→layer3的不匹配处理。
- 同时改变预训练输出阶段、token数和局部投影形状/初始化，不能把结果解释为纯token数量收益。无需重跑已经完成的旧来源下游。

## 同步

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git fetch origin
git checkout work
git pull --ff-only origin work
```

## 新预训练

使用新目录；若已存在这次结果，不重复启动同名新训练。

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_pretrain.py \
  --config configs/richhf18k_cm_pretrain_decoupled_quality_align_layer3.yaml \
  --clip-large --seed 42 --text-context-tokens 4 --quality-text-levels \
  --local-feature-stage layer3 \
  --output-dir outputs/richhf18k_quality_classprompt_cliplarge_layer3_seed42 \
  --results-csv outputs/all_results_richhf18k_quality_classprompt_cliplarge_layer3_seed42.csv
```

启动时确认 `Pretraining ResNet local feature stage=layer3`；实际config_snapshot与CSV run_start中local_encoder.feature_stage=layer3、train.pcgrad.enabled=true、lambda_quality_text_ce=.005。

完整可恢复检查点 `last_pretrain.pth` 和 `best_pretrain.pth` 按原逻辑保存优化器/调度器等状态；Component等辅助最佳文件是representation转移权重，不是完整恢复文件。只允许在相同配置/参数下恢复本轮自己的last文件。恢复：完整复用上面的命令，再加：

```bash
--resume outputs/richhf18k_quality_classprompt_cliplarge_layer3_seed42/last_pretrain.pth
```

不能把旧layer4预训练文件传给新实验的resume。已有恢复逻辑检查model/dataset/train等配置并strict加载模型。

## 下游

等待这次预训练完成后再执行以下命令。使用新的component checkpoint；三个数据集维持原统一基线、原骨干倍率0.02、Adapter原LR；1K仍按质量选模，3K/2023主要报告joint_selected。下游不保存pth，按原规则将选中权重保留在内存。

## AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_layer3_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_pretrainlayer3_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_pretrainlayer3_seed42.csv
```

## AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_layer3_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_pretrainlayer3_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_pretrainlayer3_seed42.csv
```

## AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_layer3_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_pretrainlayer3_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_pretrainlayer3_seed42.csv
```


## 验证范围

CPU使用真实ResNet layer3与真实Fusion/三视图预训练流程，CLIP与文本用小型离线替身：核对三个视图均14×14、区域map适配196个token、梯度到局部主干及PQ、实际representation提取以及匹配下游完整恢复局部投影和PQ→QA。不代表Large/GPU训练已完成或有性能提升。

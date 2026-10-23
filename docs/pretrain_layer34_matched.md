# layer3＋layer4 多尺度预训练与下游

新增可选 `--local-feature-stage layer3_4`。原layer3、layer4默认结构与state_dict命名保留，不修改已有基线。

## 实际结构

同一套ResNet-50在一次前向中返回两层（不是两套ResNet）：

```text
F3: 1024×14×14 → proj3: 1×1 Conv → 256×14×14 ──────────┐
                                                       ├→ 加权融合 → 256×14×14
F4: 2048×7×7 → proj4: 1×1 Conv → 256×7×7 → bilinear ↑ ─┘
```

F = w3·proj3(F3) + (1−w3)·upsample(proj4(F4))，w3=sigmoid(stage_mix_logit)，新预训练初始化logit0、两层各0.5；下游继承预训练学到的融合标量并继续微调。两套卷积权重独立；上采样按实际F3尺寸计算，align_corners=false；不加额外LN、注意力或融合卷积。上采样不会恢复丢失的细节。

- 输出仍是196个局部token（224输入时），不是245或392个token。Fusion、Quality、Align都消费同一个融合局部图；NSS和人类区域map按该14×14网格对齐。
- w3在所有样本和三个任务中共享，是全局可学习标量。它与Quality分数融合beta、Align逐样本gamma不同。
- 预训练train/val及下游val/test指标记录local_layer3_weight/local_layer4_weight；下游有val_或test_前缀。其梯度跟随原local_encoder优化器分组；权重衰减及学习率策略不另改。
- 两阶段都必须启用layer3_4。新representation权重必须包含两套投影、融合标量和完整请求的局部主干；尺寸不匹配、漏载或单/双层架构混用会报错。**不要传--reinit-local-projection**。
- 需要重新进行这次RichHF预训练，从原CLIP/OpenAI和ImageNet ResNet初始化；不加载旧单层任务检查点，不使用旧检查点resume。完整last/best恢复、原子保存逻辑保留。
- 保持PCGrad开启、Large、seed/split42、contexts4、质量CE0.005/blend0、Component/Ordinal/原损失、增强、学习率、早停、选模规则；不启用逐图Quality gate或Align五档CE。
- 新增融合参数/投影和更深stage会改变初始化与训练轨迹，固定seed不等于单层基线的逐参数相同初始化。不声称已获得性能提升。

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
  --config configs/richhf18k_cm_pretrain_decoupled_quality_align_layer34.yaml \
  --clip-large --seed 42 --text-context-tokens 4 --quality-text-levels \
  --local-feature-stage layer3_4 \
  --output-dir outputs/richhf18k_quality_classprompt_cliplarge_layer34_seed42 \
  --results-csv outputs/all_results_richhf18k_quality_classprompt_cliplarge_layer34_seed42.csv
```

启动时确认 `Pretraining ResNet local feature stage=layer3_4`；实际config_snapshot与CSV run_start中local_encoder.feature_stage=layer3_4、train.pcgrad.enabled=true、lambda_quality_text_ce=.005。

完整可恢复检查点 `last_pretrain.pth` 和 `best_pretrain.pth` 按原逻辑保存优化器/调度器等状态；Component等辅助最佳文件是representation转移权重，不是完整恢复文件。只允许在相同配置/参数下恢复本轮自己的last文件。恢复：完整复用上面的命令，再加：

```bash
--resume outputs/richhf18k_quality_classprompt_cliplarge_layer34_seed42/last_pretrain.pth
```

不能把旧layer4预训练文件传给新实验的resume。已有恢复逻辑检查model/dataset/train等配置并strict加载模型。

## 下游

等待这次预训练完成后再执行以下命令。使用新的component checkpoint；三个数据集维持原统一基线、原骨干倍率0.02、Adapter原LR；1K仍按质量选模，3K/2023主要报告joint_selected。下游不保存pth，按原规则将选中权重保留在内存。

## AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3_4 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_layer34_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_layer34_safeinput_noneg_pretrainlayer34_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_layer34_safeinput_noneg_pretrainlayer34_seed42.csv
```

## AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3_4 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_layer34_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_layer34_safeinput_noneg_dualselect_pretrainlayer34_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_layer34_safeinput_noneg_dualselect_pretrainlayer34_seed42.csv
```

## AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3_4 \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_layer34_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_layer34_safeinput_noneg_dualselect_pretrainlayer34_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_layer34_safeinput_noneg_dualselect_pretrainlayer34_seed42.csv
```


## 打包日志与结果（不包含大权重）

```bash
tar -czf cliplarge_pretrainlayer34_results.tar.gz \
  --exclude='*.pth' \
  outputs/richhf18k_quality_classprompt_cliplarge_layer34_seed42 \
  outputs/all_results_richhf18k_quality_classprompt_cliplarge_layer34_seed42.csv \
  outputs/agiqa1k_quality_cliplarge_layer34_safeinput_noneg_pretrainlayer34_seed42 \
  outputs/agiqa3k_quality_cliplarge_layer34_safeinput_noneg_dualselect_pretrainlayer34_seed42 \
  outputs/aigciqa2023_quality_cliplarge_layer34_safeinput_noneg_dualselect_pretrainlayer34_seed42 \
  outputs/all_results_agiqa1k_quality_cliplarge_layer34_safeinput_noneg_pretrainlayer34_seed42.csv \
  outputs/all_results_agiqa3k_quality_cliplarge_layer34_safeinput_noneg_dualselect_pretrainlayer34_seed42.csv \
  outputs/all_results_aigciqa2023_quality_cliplarge_layer34_safeinput_noneg_dualselect_pretrainlayer34_seed42.csv
```

## 验证

实际ResNet双层输出尺寸、数值融合公式、两套投影与主干/标量梯度、256输入动态尺寸、实际三视图联合模型、区域map196对齐、下游双头与优化器分组及诊断、representation提取/完整匹配恢复、单/双层双向拒载和缺失/错形张量拒载。CLIP/文本用离线小替身，不代表真实Large数据/GPU实验完成。

完整回归：124项测试通过；尚未在服务器真实数据/GPU运行本消融。

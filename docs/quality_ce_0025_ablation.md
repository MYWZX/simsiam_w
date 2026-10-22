# layer3 下游五档 Quality CE 减半消融

这次仅将下游 `--quality-text-lambda` 从 **0.005 改为 0.0025**。现有CLI已经支持该值，无需更改损失实现、旧YAML或默认行为。

## 固定条件

- 使用已有 **预训练PCGrad开启** 的 Large Component Anchor checkpoint，不用no-PCGrad来源；不重新预训练。
- 下游ResNet layer3，显式重置旧layer4局部投影，保留与已完成layer3基线相同的加载策略。
- Large、seed42 / split42、shared contexts4、residual_feature、alpha初始0.1、正常BN、blend0、原五档中心。
- 整图安全输入，关闭perturbation和旧negative；学习率、ranking/soft-SROCC系数均不改。
- 1K保持Quality选模/原早停规则；3K/2023保持60轮双选模，主要结果固定取joint_selected。
- 各数据集使用同一个0.0025权重，不按测试成绩分别挑系数。本轮不同时测试0.001或降低ranking权重。

降低CE系数不等于关闭五档分支，也不等于修改blend、预训练损失、模型参数或归一化。固定同一模型/输入时，原始CE和预测不变，CE对总损失及其梯度的加权贡献减半；训练后的参数轨迹和最终指标可能变化，不保证收益。

## 同步代码

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git fetch origin
git checkout work
git pull --ff-only origin work
```

输出目录与CSV带qce0025，保持独立；若路径已存在并含结果，使用新的实验名，不覆盖。以下每条命令在项目根目录执行，不指定GPU。

## AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-text-blend 0 --quality-text-lambda 0.0025 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_qce0025_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_qce0025_seed42.csv
```

## AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-text-blend 0 --quality-text-lambda 0.0025 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_qce0025_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_qce0025_seed42.csv
```

## AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-text-blend 0 --quality-text-lambda 0.0025 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_qce0025_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_qce0025_seed42.csv
```

## 结果核对

- config_snapshot.yaml和run_start.config_json中 `train.lambda_quality_text_ce` 应为0.0025；`model.quality_text_bank.score_blend` 应为0；local_encoder.feature_stage应为layer3。
- 预训练来源仍为原PCGrad开启的文件。已有基线SHA256为 `9a140f598c8bd09812f92e8355dce7148d4b649c89646094823f39f8bcc3bf34`；3K/2023会在run_metadata中记录实际加载文件哈希。
- 对比已完成ON-PCGrad/layer3/CE0.005实验，不重跑基线。分别报告Quality/Align测试PLCC、SROCC，保留所有下降结果。
- 仍不保存下游pth；中断会失去内存最佳权重。当前云端无数据/GPU，本次只核对参数传递、实际配置快照与小模型损失/梯度，不宣称训练完成。

## 本次验证记录

- 使用实际main配置流程，在模型构建前捕获并真实写出/读回六份临时配置快照（3个数据集×基线/新系数）；去除输出路径标识后，唯一差异为train.lambda_quality_text_ce。
- 使用包含残差Adapter、Quality bank和双输出的CPU小模型，实际调用compute_losses与autograd.grad：原始CE和Quality/Align预测一致，加权CE与image_to_text参数上的辅助梯度均减半，总损失差等于减少的CE项。
- 未改动训练源文件、模型结构、默认系数或旧实验配置；验证不等于真实数据指标改善。

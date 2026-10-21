# ResNet layer3 局部特征消融

新增 `--local-feature-stage layer3`；不传参数时仍为 layer4，已有 YAML 与默认模型不改变。

224输入时：

- layer4：2048×7×7 → Conv1×1(2048→256) → 256×7×7，49 tokens。
- layer3：1024×14×14 → Conv1×1(1024→256) → 256×14×14，196 tokens。

layer3模式的timm features_only网络不执行/保留layer4。该局部特征同时进入Fusion、Quality和Align，不只改Quality，也不是高分辨率crop或多层融合。

## 检查点策略

已有Large预训练是layer4。首次下游layer3实验同时传：

```text
--local-feature-stage layer3 --reinit-local-projection
```

兼容的ResNet stem/layer1–3权重照常恢复，旧layer4张量不属于目标网络，明确不使用。新局部投影weight和bias整体保留seed初始化；其他模块继续按原representation协议加载。日志明确打印该操作；有效配置记录feature_stage与reinit_local_projection。

未传reset开关遇到投影shape不同，直接报错。局部投影缺键报错；显式跨阶段迁移时还核查目标局部主干所有tensor都已恢复。reset开关仅支持已知ResNet50的2048→1024输入通道变更，不允许对同阶段checkpoint意外重置。

如果未来重新预训练layer3：预训练和下游都传 `--local-feature-stage layer3`，下游使用匹配的layer3 checkpoint，**不传** `--reinit-local-projection`。不能用旧layer4完整checkpoint恢复layer3预训练。

这是阶段＋投影重新初始化的结构消融，不是纯粹的token数量变化。新投影shape还可能改变后续初始化的RNG消耗；相同seed不保证所有评分头初始权重一致。当前实现不宣称初始化完全配对，也没有重跑基线或保证指标改善。

## 从已有基线做下游对照

先同步work：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git fetch origin
git checkout work
git pull --ff-only origin work
```

下列命令使用已有 **PCGrad开启的layer4 Large预训练**，与目前已完成layer4下游基线比较。不要把正在进行的no-PCGrad预训练结果同时换入这项实验；若以后用no-PCGrad来源，两种层级都应使用相同来源比较。无需重启或修改正在运行的PCGrad消融。

### AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_seed42.csv
```

### AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_seed42.csv
```

### AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_seed42.csv
```

3K/2023主要对比joint_selected；1K仅Quality。继续不保存下游pth。使用新输出路径，不覆盖已有结果。

## 验证

7项新增测试覆盖真实ResNet阶段尺寸、默认state_dict键兼容、显式加载/拒绝缺失或错配参数，以及真实layer3局部分支配合小型全局/文本替身时的Fusion、196个Quality/Align权重和反向梯度。全套111项CPU测试通过。命令解析已核对；未执行真实数据/GPU训练。

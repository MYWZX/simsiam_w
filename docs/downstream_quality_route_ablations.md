# 下游 Quality 路径两项独立消融

## 固定对照与输入定义

以下6条命令/启动器使用当前保留的统一基线：原PCGrad开启的layer4来源checkpoint→layer3下游，显式重置1024→256局部投影；Large、seed/split42、context4、residual_feature/alpha0.1、Quality CE0.005/blend0、安全输入、无perturbation/旧negative、原骨干倍率0.02、Adapter原LR、无逐图Quality门控。A同时改变Quality和Align全局输入，B只改变Quality路径，不重新预训练。

基线来源：outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth。不要把本默认启动器与刚完成的layer3来源或尚在准备的layer3_4来源混着比较。原始保留基线和这两组应使用同一来源；新来源需要独立输出目录及相应匹配阶段。

`g`是现有Fusion.semantic_proj(CLIP全局特征)，Large为768→256，位于MHA、LayerNorm、FFN之前。它复用已有线性权重，不新增投影、不增加归一化。`z`是融合局部信息并经过残差/LN/FFN后的256维表示。

## A：g替代Quality和Align全局路径的z，保留局部质量评分

开关：`--quality-global-feature g --align-global-feature g`。

```text
CLIP → 现有semantic_proj → g → QA → a
                            g + alpha*sg(||g||)*normalize(a) → global_quality_head → Qg
局部特征 → NSS / patch scoring → Ql
Q=(1-beta)Qg+beta Ql
```

替换覆盖QA输入、残差旁路、范数尺度和全局诊断；没有通过残差偷偷保留z。五档CE仍消费QA输出a。Quality总损失仍通过保留的局部分支到达ResNet，但不通过Fusion.attn/local_proj/FFN/norms；semantic_proj仍接收Quality梯度。

**Align全局路径同样改为g→A_G→v**，全局pair head及逐图门控使用这个v；局部图文匹配、文本分支和Quality局部分支保留。共享编码器仍会计算z供诊断及兼容接口使用，但本组Quality/Align评分不依赖z，两个任务的损失均不回传Fusion跨注意力、局部投影、LN或FFN。Quality全局路径同时绕过MHA及其后LN/FFN，所以此实验回答“g替换z”的效果，不是单独删除MHA而保留其后层的纯注意力消融。图中g通常有不同于z的尺度分布，本轮按用户要求直接使用g，不额外插入LN。

## B：保留z，严格移除Quality局部分支

开关：`--quality-global-feature z --quality-no-local-branch`。

```text
CLIP + ResNet → Fusion → z → QA → a
                           z + alpha*sg(||z||)*normalize(a) → global_quality_head → Q
```

Quality不执行NSS、NSS投影/门控、patch质量头、局部加权和与beta融合；相关私有参数冻结，但保留模块定义/state_dict以保持其他模块初始化和检查点兼容。局部分数返回None，验证/测试不记录伪造的patch成绩或误差相关性。

L/local_patch_proj仍为Align使用，不能冻结它；ResNet与Fusion仍由z路径接收Quality梯度。Align全局/局部路径均保留。禁止与Quality局部attention、局部辅助头或逐图Quality门控并用。

旧`--quality-global-only`仅绕过局部分数融合但仍计算局部分支，本轮使用新严格开关；旧实验行为不修改。严格跳过局部Dropout会改变训练时随机数消费顺序，不声称与旧global-only训练轨迹逐位相同。

## 关于“直接预测”与不变的Adapter

两组均保留当前`residual_feature`，不是同时切换成adapter-only/replace。第二组是“直接用全局路径预测质量、不与Ql混合”，其全局头仍接收残差r。Quality五档CE保持开启、blend0；不添加或移除其他辅助目标。两个开关可组合形成第三个控制，但本次命令不组合这两项。

A的Align全局分数及门控会改变；相同权重和eval模式下局部图文分数保持一致。B的Align路径不变。微调后共享参数变化仍会影响Align。未开启实验时旧主路径不变。

## 同步与六条可执行命令

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
```

启动器内包含全部Python参数，不指定GPU。若同名输出已包含结果，请先保留旧结果并使用新实验名称，避免CSV追加和元数据覆盖。

```bash
# A：g + 局部质量评分
bash scripts/run_quality_route_ablation.sh g_local agiqa1k
bash scripts/run_quality_route_ablation.sh g_local agiqa3k
bash scripts/run_quality_route_ablation.sh g_local aigciqa2023

# B：z + 严格关闭局部质量评分
bash scripts/run_quality_route_ablation.sh z_global agiqa1k
bash scripts/run_quality_route_ablation.sh z_global agiqa3k
bash scripts/run_quality_route_ablation.sh z_global aigciqa2023
```

加`--dry-run`可以只打印完整命令，不启动训练。默认Python为服务器WZX-AI环境，可用ABLATION_PYTHON显式覆盖。1K质量选模，3K/2023仍跑60轮比较策略、主要取joint_selected；下游不保存pth。

## A组完整原生命令

## AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-global-feature g --align-global-feature g \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_qualityaligng_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_qualityaligng_seed42.csv
```

## AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-global-feature g --align-global-feature g \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_qualityaligng_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_qualityaligng_seed42.csv
```

## AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-global-feature g --align-global-feature g \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_qualityaligng_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_qualityaligng_seed42.csv
```


## B组完整原生命令

## AGIQA-1K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa1k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-global-feature z --quality-no-local-branch \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_zglobalonly_seed42 \
  --results-csv outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_zglobalonly_seed42.csv
```

## AGIQA-3K

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/agiqa3k_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-global-feature z --quality-no-local-branch \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_zglobalonly_seed42 \
  --results-csv outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_zglobalonly_seed42.csv
```

## AIGCIQA2023

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
  --config configs/aigciqa2023_cm_iqa_decoupled_quality_align.yaml \
  --clip-large --seed 42 --text-context-tokens 4 \
  --local-feature-stage layer3 --reinit-local-projection \
  --quality-global-feature z --quality-no-local-branch \
  --quality-text-blend 0 --quality-text-lambda 0.005 \
  --quality-adapter-mode residual_feature --quality-adapter-residual-init 0.1 \
  --quality-safe-input --disable-perturbation --disable-legacy-negative --audit-loss-gradients \
  --compare-selection-policies \
  --pretrained-checkpoint outputs/richhf18k_quality_classprompt_cliplarge_seed42/best_component_anchor_pretrain.pth \
  --output-dir outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_zglobalonly_seed42 \
  --results-csv outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_zglobalonly_seed42.csv
```


## 检查项

- 配置快照：A为model.quality_global_feature=g、model.align_global_feature=g，Quality局部仍开启；B为quality_global_feature=z、quality_no_local_branch=true、quality_global_only=true。
- 验证指标：quality_global_uses_g和align_global_uses_g均为A=1/B=0；quality_local_enabled为A=1/B=0；B没有quality_patch_*和quality_branch_error_PLCC，quality_mix为0。
- 来源checkpoint/数据划分/学习率一致，保留所有下降项；不按测试成绩在每数据集切换方法。
- CPU小模型使用真实Fusion及Quality/Align头，检查g的QA/残差输入、梯度依赖、局部模块调用次数、Align eval一致性、严格分支冻结、CE梯度及诊断；不宣称真实数据或GPU训练完成。

完整回归129项测试通过；额外检查了双g模式的Align局部分数保持一致、Quality/Align均不向Fusion跨注意力回传梯度。

A的新结果名称含qualityaligng，与旧的仅Quality替换实验qualityg分开，避免混用结果。CPU验证不代表真实数据或GPU训练完成。

## 完成后打包两组结果

```bash
tar -czf cliplarge_quality_route_ablations_results.tar.gz \
  outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_qualityaligng_seed42 \
  outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_qualityaligng_seed42.csv \
  outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_qualityaligng_seed42 \
  outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_qualityaligng_seed42.csv \
  outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_qualityaligng_seed42 \
  outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_qualityaligng_seed42.csv \
  outputs/agiqa1k_quality_cliplarge_layer3_safeinput_noneg_zglobalonly_seed42 \
  outputs/all_results_agiqa1k_quality_cliplarge_layer3_safeinput_noneg_zglobalonly_seed42.csv \
  outputs/agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_zglobalonly_seed42 \
  outputs/all_results_agiqa3k_quality_cliplarge_layer3_safeinput_noneg_dualselect_zglobalonly_seed42.csv \
  outputs/aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_zglobalonly_seed42 \
  outputs/all_results_aigciqa2023_quality_cliplarge_layer3_safeinput_noneg_dualselect_zglobalonly_seed42.csv
```

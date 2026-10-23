# 以双 g 下游实验①为基线的三个独立消融

三个实验相互独立，不逐项叠加。每组都必须重新预训练，再分别微调三个数据集；不能把旧权重配一个新下游开关当作完整实验。

共同设置沿用已完成的 qualityaligng 实验：Large、seed/split42、context4；预训练保留原 layer4、PCGrad 开启、三视图、50轮上限/15轮早停与 Component Anchor 验证选模；下游 layer3、显式重置局部投影，Quality/Align 全局输入均为 g，保留两项任务的局部分支。下游 Quality Adapter 为 residual_feature，alpha 初始0.1。学习率、安全输入、损失权重、选模策略均沿用原实验；没有开启逐图 Quality 门控。

这里继续使用 layer4→layer3，是为了与已完成实验①固定同一个对照。不要同时换成新 layer3 或 layer3+4 来源，否则会混入额外变量。新预训练从基础视觉权重开始，不续训原 RichHF 检查点。

| 组 | 新预训练变化 | 下游变化 |
|---|---|---|
| noce | 关闭质量文本五档 bank 和 CE | 同样关闭 bank 与 CE |
| mos_pq | Clean 全局质量特征先经过独立 PQ，再进入质量及 Artifact/Aesthetics 评分路径 | 结构仍为原实验①，QA继承新预训练PQ |
| shared_pi_pt | I-I改用共享PI；图文图像关系也使用同一个PI；PI/PT保存为可迁移表示；独立PQ保留 | Align的g与P0共用PI，再进各自Adapter；文本U后接PT再进AT；QA仍只继承PQ |

## 1. noce：移除预训练与下游质量文本五档 CE

关闭的是 terrible/poor/fair/good/excellent 的 CLIP 质量文本辅助监督：model.quality_text_bank.enabled=false，train.lambda_quality_text_ce=0。启动器不传会重新开启bank的 --quality-text-levels 或 --quality-text-blend。

Component Anchors、Ordinal、MOS监督全部保留。虽然 Component Anchors 也有五档锚点，它们并非本次移除的文本CE。Align原本没有五档文本CE，本组也不新增。

## 2. mos_pq：质量 MOS/辅助评分路径经过 PQ

```text
Clean → 双视觉编码器/Fusion → z_c → PQ → normalize → q_c
                                                    ├→ 全局质量头 → Qg
                                                    ├→ Component Anchors / Ordinal / 质量文本CE
                                                    └→ 与局部证据拼接 → Artifact/Aesthetics 辅助头
局部视觉特征 → 原NSS/patch评分 → Ql
Q = (1-beta)Qg + beta Ql → 质量MOS监督
```

所有路径共用同一次Clean PQ输出，避免额外的PQ/BatchNorm/Dropout调用。局部Ql路径、Quality全局/局部分数融合仍保留，质量MOS损失因此能直接更新PQ。Artifact/Aesthetics头仍保留原来的全局/局部/差值拼接形式，只把其中全局特征改为q_c。

按用户确认，Align MOS不经过PQ，保留原图文评分路径。SafeAug I-I仍用独立PQ，与原预训练相同。下游不额外添加“再经过一次PQ”：新PQ直接初始化原有Quality Adapter，其余结构不变。

## 3. shared_pi_pt：移动 I-I，并迁移 PI/PT

**独立PQ没有删除，也没有合并为PI。** 只把I-I的投影路径从PQ改到PI。质量Component Anchors/Ordinal/文本CE仍使用Clean z_c→PQ，下游QA也仍继承PQ，不继承PI。

预训练主路径：

```text
SafeAug z1,z2 → 共享PI → normalize → DI → I-I SimSiam
图像 z       → 同一个PI → normalize → AG → normalize → DI → 图像侧关系
Prompt       → 冻结TextEncoder → U → PT → normalize → AT → normalize → DT
Answer       → 冻结TextEncoder → U → 同一个PT → normalize → DT
Clean z_c    → 独立PQ → normalize → Component Anchors / Ordinal / 质量文本CE
```

四项常规SimSiam为 I-I、Image-Prompt、Image-Answer、Prompt-Answer。前三项的图像侧共享PI；第四项只有文本，不会把文本错误送入PI。图像预测器DI复用原图文image_predictor；独立PQ及其原预测器定义保留，保持质量模块和检查点布局。预测器都不迁移。

Prompt与Answer保留原先是否经过AT的区别。本组同时将图文PI/PT放到Adapter之前，使预训练与下游的投影/Adapter顺序一致；它是用户指定的组合架构消融，不能将最终变化单独归因于某一个共享操作。

局部反馈路径使用已经经过PI/AP的patch表示，区域池化后不重复执行PI。

下游：

```text
CLIP全局特征 → 原semantic_proj → g  ──→ PI ──→ AG → v
                                      同一套PI参数
ResNet局部特征 → 原L投影 → P0     ──→ PI ──→ AP → p
Prompt+contexts → 冻结TextEncoder → U(768→256) → PT → AT → u

v,u → 原全局pair head → Ag
p,u → 原局部图文匹配及pair head → Al
原逐图门控 → A = gamma Ag + (1-gamma)Al

Quality：g → QA(由独立PQ初始化) → 原残差/全局质量头
         P0 → 原NSS局部质量分支
```

PI和PT均为256→512→512→256的原MLP投影结构（沿用MLP内部BN/ReLU/Dropout），输出L2归一化。由于冻结TextEncoder输出768维，保留U先映射为256维，再接PT，不能把256维PT直接接在768维原始输出上。PI对局部token逐个应用；实现中将B×N×256展平为(BN)×256，避免BN把token数量当作通道。全局和局部PI共享全部参数及BN统计，不是复制两个独立PI。

权重只注册在iqa.shared_image_projector / iqa.shared_text_projector一个位置。迁移检查包含全部参数和BN缓冲：缺失、形状错误、旧架构误配都会报错，不允许静默留下随机PI/PT。

## 执行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

# 每组先完成自己的预训练
bash scripts/run_projection_ablation.sh noce pretrain
bash scripts/run_projection_ablation.sh mos_pq pretrain
bash scripts/run_projection_ablation.sh shared_pi_pt pretrain

# 再使用每组自己的 best_component_anchor_pretrain.pth 微调
for experiment in noce mos_pq shared_pi_pt; do
  for dataset in agiqa1k agiqa3k aigciqa2023; do
    bash scripts/run_projection_ablation.sh "$experiment" "$dataset" || break 2
  done
done
```

也可逐组预训练后立即跑该组的三个数据集。命令末尾加 --dry-run 可查看全部实际参数，不加载数据或启动训练。默认Python沿用WZX-AI，可通过ABLATION_PYTHON覆盖。没有指定GPU编号。输出已存在时启动器拒绝覆盖/追加；预训练如需恢复，请使用train_cm_pretrain.py现有--resume参数及该组同一配置。

输出：预训练为outputs/richhf18k_cliplarge_<组名>_seed42；下游为outputs/<数据集>_quality_cliplarge_layer3_safeinput_noneg[_dualselect]_qualityaligng_<组名>_seed42，各自另有all_results_*.csv。下游不保存pth；预训练保留完整恢复检查点及表示迁移检查点。1K按质量验证选模，3K/2023记录双策略、主比较仍取joint_selected。

结果打包（排除大型权重）：

```bash
files=()
for experiment in noce mos_pq shared_pi_pt; do
  pretrain_name="richhf18k_cliplarge_${experiment}_seed42"
  files+=("outputs/$pretrain_name" "outputs/all_results_${pretrain_name}.csv")
  for dataset in agiqa1k agiqa3k aigciqa2023; do
    selection=""
    [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
    name="${dataset}_quality_cliplarge_layer3_safeinput_noneg${selection}_qualityaligng_${experiment}_seed42"
    files+=("outputs/$name" "outputs/all_results_${name}.csv")
  done
done
tar --exclude='*.pth' --exclude='*.pt' -czf cliplarge_projection_ablations_results.tar.gz "${files[@]}"
```

新实验需要真实数据/GPU运行才能得出性能结论，代码测试不代表训练已经完成。

## 验证范围

136项CPU回归测试通过。新增检查覆盖MOS到PQ的梯度、I-I到PI而非PQ的梯度、独立PQ保留、PI/PT完整迁移及缺失报错、下游Adapter输入顺序和共享参数、无CE实际损失、12条启动命令的有效main配置，以及真实三视图训练循环的一批次PCGrad更新和验证。以上训练循环使用小型替代视觉/文本编码器与真实Fusion/评分头，不代表RichHF或三个下游数据集的GPU实验完成。

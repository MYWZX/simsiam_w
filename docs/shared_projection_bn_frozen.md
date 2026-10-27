# 共享PI/PT：下游冻结BN统计量的对照

对照为已完成的shared_pi_pt双质量残差/layer3实验。本组仅在下游设置model.shared_projection_freeze_bn_stats=true，不重新预训练，不更换来源，不叠加noce、Joint来源或layer3+4。

固定来源：outputs/richhf18k_cliplarge_dualquality_residual_layer3_shared_pi_pt_seed42/best_component_anchor_pretrain.pth。

## 实际冻结内容

只对shared_image_projector（PI）和shared_text_projector（PT）内部的BatchNorm调用eval：

- running_mean、running_var和num_batches_tracked保持从预训练加载的值，训练前向和验证前向都使用这些保存的统计。
- BN的可学习gamma/beta仍可求梯度、仍在原优化器中更新；最后一个affine=false的BN本来就没有这两个参数。
- PI/PT的Linear及其余参数仍训练，Dropout仍在训练阶段工作；没有对整个PI/PT调用eval或requires_grad_(False)。
- ResNet、CLIP、QA、局部质量编码器、NSS汇总、alpha/eta/分数beta、Align Adapters及评分头沿用原策略，不冻结它们的BN或参数。

实现会在模型构建时及每次model.train()后重新应用BN策略，不会被训练循环或eval→train切换取消。它是运行策略，不添加参数或改变state_dict结构，所以可以完整继承原shared预训练权重，包括PI/PT的BN统计和alpha。

默认开关为false，原shared及其他实验不改变；如果未开启shared_simsiam_projection却设置此开关，会报错。

## 对照目的与限制

先前3K共享结构的训练Align持续上升、验证Align下降，值得检查共享PI同时处理g和P0时的BN统计变化。冻结统计可以减少这一项训练/验证差异，但Dropout仍有训练/验证差异，不能称为两种模式完全相同。

本组不能直接证明原退化一定由BN导致，也不能保证提升；若结果改善，仍需结合分支曲线解释。只冻结统计，不修改BN结构或替换成LayerNorm，也不降低PI/PT学习率。

保持Large、seed/split42、contexts4、匹配layer3、独立PQ/QA质量分支、残差评分、CE0.005/blend0、原学习率、安全输入和损失设置。1K按质量验证选模；3K/2023按原双策略完成60轮，主结果仍取joint_selected。三个数据集统一应用同一策略，不按测试集切换模型。

## 执行：无需重新预训练

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

bash scripts/run_shared_projection_bn_frozen.sh agiqa1k
bash scripts/run_shared_projection_bn_frozen.sh agiqa3k
bash scripts/run_shared_projection_bn_frozen.sh aigciqa2023
```

每条命令末尾加--dry-run可查看完整参数。启动器没有pretrain入口；来源文件不存在会报错，不回退到其他模型。默认Python为/home/sll/anaconda3/envs/WZX-AI/bin/python，可通过ABLATION_PYTHON覆盖，不指定GPU编号。下游不保存pth。

输出增加_shared_pi_pt_bnfrozen_seed42，与原shared_pi_pt结果分开；同名目录/CSV存在时拒绝覆盖或追加。

验证/测试CSV增加val_shared_projection_bn_stats_frozen/test_shared_projection_bn_stats_frozen=1。该字段表示配置的持久化冻结策略，不是因评估阶段所有BN临时处于eval就将原基线标记为冻结。

## 打包

```bash
files=()
for dataset in agiqa1k agiqa3k aigciqa2023; do
  selection=""
  [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
  name="${dataset}_quality_cliplarge_layer3_dualquality_residual${selection}_shared_pi_pt_bnfrozen_seed42"
  files+=("outputs/$name" "outputs/all_results_${name}.csv")
done
tar --exclude='*.pth' --exclude='*.pt' -czf cliplarge_shared_pi_pt_bnfrozen_results.tar.gz "${files[@]}"
```

代码检查包括反复train/eval切换、仅PI/PT的BN为eval而Dropout正常训练、运行统计不变但BN仿射和Linear参数更新、QA BN正常更新、原基线统计仍更新、相同权重在初始eval前向完全一致、原checkpoint完整迁移、三条命令的有效main配置及禁止pretrain阶段。训练循环使用小型替代视觉/文本编码器和实际模型头，不代表真实数据/GPU训练。

完整CPU回归158项测试通过；预训练非默认BN统计完整迁移检查通过。尚未运行真实数据/GPU实验。

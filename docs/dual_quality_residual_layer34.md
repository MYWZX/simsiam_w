# 双质量残差：匹配layer3+4对照

对照为dual_quality_residual的layer3→layer3实验。本组仅将预训练、下游的local_encoder.feature_stage同时改为layer3_4，另设实验名称、输出目录和来源checkpoint。不是此前普通质量结构的layer3+4实验，也不是无全局残差的dual_quality/replace实验。

沿用已有多层局部编码：同一个ResNet-50输出layer3（1024通道）与layer4（2048通道），各自投影到256通道；layer4投影上采样到layer3网格后，使用可学习标量sigmoid权重融合。224输入仍得到256×14×14、196个局部token。初始两层权重各0.5；不是两个ResNet，也不是拼接成512维或245个token。

所有原来消费局部特征的路径（质量局部编码/NSS汇总、Align局部匹配、预训练Fusion）都使用该多层融合图；所以这是整套局部表示的换层对照，不是只改变某一个Quality模块。

保持不变：

- Large、seed/split42、contexts4、原学习率和损失系数。
- g→PQ/QA→qg，r=g+alpha*stopgrad(norm(g))*normalize(qg)，全局头读取r。
- qg与局部ql的特征融合q接受Component Anchors/Ordinal/质量五档CE，CE权重0.005，blend0；下游仍仅用原有五档CE辅助目标。
- 局部头读取ql；Qg/Ql以原beta融合预测质量；Align下游仍使用g与局部匹配。
- alpha预训练初始化0.1，下游继承学到的alpha；eta、局部质量编码/NSS汇总也完整迁移。
- 预训练50轮上限/15轮早停、PCGrad开启、val_component_anchor_SROCC选择主checkpoint。1K按质量验证选模；3K/2023跑60轮比较验证策略，主结果取joint_selected。

## 权重迁移与诊断

两阶段都必须使用layer3_4及本组的新checkpoint，不传--reinit-local-projection。现有严格检查会要求两层ResNet主干、proj3/proj4、stage_mix_logit和全部双质量表示参数完整恢复；缺失或与单层检查点混用会报错。不得加载此前普通layer3+4或layer3双质量残差检查点来代替这次完整实验。

CSV同时记录四类不同权重：

| 名称 | 含义 |
|---|---|
| local_layer3_weight / local_layer4_weight | ResNet两层特征融合 |
| quality_feature_local_weight / global_weight | qg/ql的辅助质量表示融合eta |
| quality_residual_alpha | g的质量残差强度alpha |
| quality_mix_* | Qg/Ql分数融合beta |

预训练记录train_/val_前缀；下游为val_/test_前缀。它们不是同一个权重，也不直接代表某模块的性能贡献比例。

## 执行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

# 先完成本组新预训练
bash scripts/run_dual_quality_residual_layer34.sh pretrain

# 再微调三个数据集
bash scripts/run_dual_quality_residual_layer34.sh agiqa1k
bash scripts/run_dual_quality_residual_layer34.sh agiqa3k
bash scripts/run_dual_quality_residual_layer34.sh aigciqa2023
```

每条命令末尾加--dry-run可以查看完整参数。默认Python为/home/sll/anaconda3/envs/WZX-AI/bin/python，可由ABLATION_PYTHON覆盖，不指定GPU编号。输出已存在时拒绝覆盖/追加；恢复预训练请使用现有train_cm_pretrain.py --resume与本组同一配置。

新来源：outputs/richhf18k_cliplarge_dualquality_residual_layer34_seed42/best_component_anchor_pretrain.pth。预训练保存完整恢复及表示checkpoint；下游不保存pth。

## 打包

```bash
files=(
  outputs/richhf18k_cliplarge_dualquality_residual_layer34_seed42
  outputs/all_results_richhf18k_cliplarge_dualquality_residual_layer34_seed42.csv
)
for dataset in agiqa1k agiqa3k aigciqa2023; do
  selection=""
  [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
  name="${dataset}_quality_cliplarge_layer34_dualquality_residual${selection}_seed42"
  files+=("outputs/$name" "outputs/all_results_${name}.csv")
done
tar --exclude='*.pth' --exclude='*.pt' -czf cliplarge_dualquality_residual_layer34_results.tar.gz "${files[@]}"
```

验证覆盖配置差异隔离、四条启动命令的有效main配置、实际ResNet-50三视图前向、质量CE到两层投影/层融合权重的梯度、完整局部参数及alpha迁移、缺失和单层误配拒绝。CLIP/文本使用小型替代编码器，不代表真实数据/GPU训练完成。

完整CPU回归149项测试通过；尚未运行真实数据/GPU实验。

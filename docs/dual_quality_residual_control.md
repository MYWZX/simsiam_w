# 双质量表示的全局残差评分对照

对照为已完成的dual_quality/replace实验（不是原layer4→layer3基线，也不是layer3+4）。新方案只改变全局质量头的输入：

```text
旧：g → PQ/QA → normalize → qg → 全局质量头 → Qg
新：g → PQ/QA → normalize → qg
    r = g + alpha * stopgrad(norm(g)) * normalize(qg)
    r → 原全局质量头 → Qg
```

alpha=sigmoid(logit)，预训练初始化0.1、可学习。g是现有semantic_proj输出的CLIP全局特征，不是融合z。只对范数估计停止梯度；g旁路、PQ/QA与alpha都可以由质量MOS损失训练。

不改变的路径：

```text
qg 与局部ql → 原eta特征融合 → q → Component Anchors / Ordinal / 质量五档CE
局部特征 → 原局部编码/NSS汇总 → ql → 原局部质量头 → Ql
Q = (1-beta)Qg + beta Ql
```

辅助监督使用qg与ql的融合，不使用r；alpha不进入这些辅助损失。SafeAug I-I仍使用全局g→PQ；Align、质量五档CE0.005/blend0、两阶段layer3、学习率、PCGrad、划分和验证选模策略等全部沿用上一组。预训练必须重跑，不能只给旧replace检查点加一个下游开关。

## alpha保存与迁移

alpha的logit存为iqa.quality_adapter_gate_logit（预训练），下游为quality_adapter_gate_logit。完整恢复checkpoint与表示checkpoint都保存它；下游完整迁移该参数，不重置成0.1。

启动参数里的--quality-adapter-residual-init 0.1只用于构建模型，随后严格加载新预训练学到的alpha。缺失该参数会报错。dual_quality_signature=1标记旧replace，2标记本残差方案，禁止两种评分方式误配检查点。局部投影/质量编码/NSS汇总/eta仍完整继承；数据集评分头和beta仍按原策略重新学习。

原replace启动器及配置保持不变；新四份配置与原配置仅有全局残差模式、alpha初始化，以及实验名/输出路径/来源checkpoint的区别。新输出名称含dualquality_residual，不覆盖旧结果。

## 运行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

# 先完成新的预训练
bash scripts/run_dual_quality_residual.sh pretrain

# 然后运行三个下游
bash scripts/run_dual_quality_residual.sh agiqa1k
bash scripts/run_dual_quality_residual.sh agiqa3k
bash scripts/run_dual_quality_residual.sh aigciqa2023
```

每条命令末尾加--dry-run可打印完整参数。默认Python为/home/sll/anaconda3/envs/WZX-AI/bin/python，不指定GPU编号。预训练保持50轮上限/15轮早停、val_component_anchor_SROCC选主checkpoint；下游1K按质量验证选模，3K/2023跑60轮比较策略、主要取joint_selected。预训练保存恢复和迁移checkpoint，下游不保存pth。输出已存在时拒绝覆盖/追加；恢复训练应使用原train_cm_pretrain.py --resume与同一新配置。

新预训练目录：outputs/richhf18k_cliplarge_dualquality_residual_layer3_seed42。

CSV新增quality_residual_alpha（下游带val_/test_前缀）；quality_feature_local_weight继续表示eta，quality_mix_*继续表示beta，三者不混用。

## 打包

```bash
files=(
  outputs/richhf18k_cliplarge_dualquality_residual_layer3_seed42
  outputs/all_results_richhf18k_cliplarge_dualquality_residual_layer3_seed42.csv
)
for dataset in agiqa1k agiqa3k aigciqa2023; do
  selection=""
  [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
  name="${dataset}_quality_cliplarge_layer3_dualquality_residual${selection}_seed42"
  files+=("outputs/$name" "outputs/all_results_${name}.csv")
done
tar --exclude='*.pth' --exclude='*.pt' -czf cliplarge_dualquality_residual_layer3_results.tar.gz "${files[@]}"
```

验证范围：检查残差公式/范数停止梯度、全局头输入、其他表示及Align前向不变、辅助损失不训练alpha但仍更新两个视觉编码、alpha完整保存/迁移及缺失/模式误配报错、训练循环和四条启动命令的有效配置。训练循环使用小型替代视觉/文本编码器，不代表真实数据/GPU训练完成。

完整CPU回归：146项测试通过；尚未运行真实数据/GPU实验。

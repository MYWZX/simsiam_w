# 双质量残差layer3：预训练联合选模来源对照

“残差/layer3”指预训练和下游的ResNet均取layer3（224输入为14×14网格、196个局部token），不是仅下游layer3。

本组固定刚完成的双质量残差/layer3结构，仅将三个下游的统一来源从

outputs/richhf18k_cliplarge_dualquality_residual_layer3_seed42/best_component_anchor_pretrain.pth

改为同一次预训练已保存的

outputs/richhf18k_cliplarge_dualquality_residual_layer3_seed42/best_joint_pretrain.pth

不重新预训练、不换layer3+4、不改变模型或损失。上一轮日志中Component Anchor来源为第8轮，Joint来源为第23轮。第23轮预训练验证Quality/Align SROCC为0.764765/0.571971；第8轮为0.761273/0.528276。这说明该控制有验证集依据，不保证下游改善。

预训练val_joint_SROCC在当前两输出配置下是Quality与Align SROCC的算术均值；不要与下游joint_selected的调和平均指标混淆。0.668368的Joint指标与0.713700的Component Anchor指标定义不同，不能直接比较数值大小。

## 保持不变

模型仍为g→PQ/QA→qg，r=g+alpha*stopgrad(norm(g))*normalize(qg)，全局头读取r。辅助监督继续使用qg/ql融合质量表示；局部编码/NSS汇总及质量分数融合不变。两阶段layer3匹配，局部投影不重置。

Large、seed/split42、contexts4、CE0.005/blend0、安全输入、原学习率、Align路径及下游训练/选模策略都保持不变。alpha、eta、PQ/QA、ResNet/L/局部质量编码/NSS汇总等表示参数均从Joint checkpoint完整继承；不是把旧Component Anchor模型中的某些权重与Joint拼接。评分头和beta按原策略重新学习。

命令中的residual-init0.1是构建默认值，加载后由Joint checkpoint的alpha覆盖。旧的双质量模式/完整迁移检查继续生效。不能为了更高测试成绩逐数据集选择Component或Joint来源，三个数据集统一使用Joint。

## 执行：只需三次微调

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

bash scripts/run_dual_quality_residual_joint.sh agiqa1k
bash scripts/run_dual_quality_residual_joint.sh agiqa3k
bash scripts/run_dual_quality_residual_joint.sh aigciqa2023
```

末尾加--dry-run可查看完整命令。默认Python为/home/sll/anaconda3/envs/WZX-AI/bin/python，可通过ABLATION_PYTHON覆盖，不指定GPU编号。启动器没有pretrain阶段，会拒绝未知阶段；来源文件不存在时报错，不回退到Component Anchor检查点。同名输出存在时拒绝覆盖或追加。

输出名称增加_pretrainjoint_seed42，与上一轮结果隔离。1K按质量验证选模，3K/2023仍完成60轮、记录双验证策略，主要报告joint_selected；不保存下游pth。

## 打包

```bash
files=()
for dataset in agiqa1k agiqa3k aigciqa2023; do
  selection=""
  [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
  name="${dataset}_quality_cliplarge_layer3_dualquality_residual${selection}_pretrainjoint_seed42"
  files+=("outputs/$name" "outputs/all_results_${name}.csv")
done
tar --exclude='*.pth' --exclude='*.pt' -czf cliplarge_dualquality_residual_layer3_pretrainjoint_results.tar.gz "${files[@]}"
```

配置检查：三份原始YAML及三条命令经过实际main的有效配置快照流程，与原来源命令比较，除输入checkpoint与实验输出标识外没有其他差异；脚本语法和拒绝pretrain阶段检查通过。未构建数据集或进行真实数据/GPU训练。本次只新增配置、启动器和文档，不修改模型与训练代码。

# 下游仅全局g经过PI，P0绕过PI

这是在shared_projection_bn_frozen对照基础上的独立下游路由实验。保持PI/PT的BN统计冻结，仅设置model.shared_projection_project_patches=false（默认true，原实验行为不变）。

```text
原：g  → PI → normalize → AG
    P0 → 同一个PI → normalize → AP
    文本 → 冻结TextEncoder → U → PT → normalize → AT

新：g  → PI → normalize → AG
    P0 ───────────────────→ AP
    文本 → 冻结TextEncoder → U → PT → normalize → AT
```

P0本来就不经过PT；本次移除它在图像侧经过PI的步骤。文本PT保留。局部路径不额外插入新的投影或归一化：AP实际接收原L投影产生的P0。因而本对照同时绕过局部PI内的MLP/BN/Dropout及其后L2归一化，不是单独删掉某一个线性层。

Quality的g→QA残差评分、局部质量编码/NSS汇总、融合辅助监督保持原样。全局/局部pair head与最终Align门控仍按原结构工作。局部评分不直接依赖PI参数，但仍依赖PT的文本表示；全局评分与门控仍可训练PI。两个任务依然共享视觉编码器及L，不能保证微调后全局分数或Quality与原实验不变。

## 来源与权重

无需重新预训练。统一使用已有共享PI/PT来源：

outputs/richhf18k_cliplarge_dualquality_residual_layer3_shared_pi_pt_seed42/best_component_anchor_pretrain.pth

预训练配置保持P0经过PI；本次只改变下游。所有原有表示权重照常完整加载，包括PI、PT、AP、独立PQ/QA、局部质量编码/NSS汇总、alpha/eta。没有重新初始化AP；它现在接收未经PI处理的P0，输入分布因此改变，这是本实验要检验的变化，不保证会改善。

保持匹配layer3、Large、seed/split42、contexts4、质量CE0.005/blend0、残差评分、原学习率、安全输入及验证选模策略。PI/PT的BN统计仍固定为预训练值，BN仿射参数及Linear仍训练；QA、ResNet等其他模块不新增冻结。1K质量验证选模；3K/2023完成60轮、主结果取joint_selected。

移除局部PI也移除了对应训练Dropout调用，可能改变后续随机数消费；同seed不代表两条训练轨迹逐位可比。同权重eval前向检查可确认Quality与全局Align未被直接改动。

## 执行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

bash scripts/run_shared_projection_global_only.sh agiqa1k
bash scripts/run_shared_projection_global_only.sh agiqa3k
bash scripts/run_shared_projection_global_only.sh aigciqa2023
```

没有pretrain入口。不指定GPU编号，Python默认/home/sll/anaconda3/envs/WZX-AI/bin/python，可通过ABLATION_PYTHON覆盖。命令末尾加--dry-run打印完整参数；缺少来源或输出目录/CSV已存在时拒绝启动。下游不保存pth。

结果目录增加_shared_pi_pt_bnfrozen_piglobal_seed42，原shared和BN冻结实验不被覆盖。验证/测试CSV记录align_patch_uses_shared_pi=0（带val_/test_前缀），并保留shared_projection_bn_stats_frozen=1。

## 打包

```bash
files=()
for dataset in agiqa1k agiqa3k aigciqa2023; do
  selection=""
  [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
  name="${dataset}_quality_cliplarge_layer3_dualquality_residual${selection}_shared_pi_pt_bnfrozen_piglobal_seed42"
  files+=("outputs/$name" "outputs/all_results_${name}.csv")
done
tar --exclude='*.pth' --exclude='*.pt' -czf cliplarge_shared_pi_pt_bnfrozen_piglobal_results.tar.gz "${files[@]}"
```

检查覆盖PI仅调用一次且接收g、AP接收未经PI归一化的P0、PT路径保留、局部评分到PI梯度断开而到PT/ResNet仍连通、相同权重eval下Quality/全局Align一致、完整原权重迁移、BN冻结保留与三条启动命令的有效配置。小模型检查不代表真实数据/GPU训练。

完整CPU回归161项测试通过，尚未运行真实数据/GPU实验。

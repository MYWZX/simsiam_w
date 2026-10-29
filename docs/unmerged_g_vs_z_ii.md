# 两阶段去CE/negative，下游不合并：g-I-I对照与z-I-I新实验

用户要求：在此前预训练也关闭CE/negative的设置上，下游不合并；另外只把预训练I-I从g输入PI改为融合z输入PI。

两套都使用RichHF原划分预训练，不合并RichHF；下游也保留训练/验证/测试集合。CE和负例push关闭，Soft-SROCC原系数保留，其余模型/损失保持。

| 项目 | g对照 | z新实验 |
|---|---|---|
| I-I输入 | g→PI→DI | z→PI→DI |
| 图文图像输入 | z→PI→AG→DI | 相同 |
| Clean质量路径 | g→PQ/QA，全局残差＋局部质量 | 相同 |
| 预训练来源 | 已完成的richhf18k_protocolv2_both_noce_noneg_seed42/best_component_anchor_pretrain.pth | 新richhf18k_zii_noce_noneg_pretrain_dev_seed42/best_component_anchor_pretrain.pth |
| 下游合并 | 不合并 | 不合并 |
| 下游选模 | val_quality_SROCC最大 | 相同 |

下游划分：1K独立subset1078保持755/108/215；3K为2087/298/597；2023为1680/240/480。没有refit阶段，测试仅在验证选模完成后对所选模型执行，测试不参与选轮。

## 一键运行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_unmerged_ii_experiments.sh
```

默认all流程：校验/准备subset1078 → 核验并复用三份已完成g开发结果（不存在时才训练）→ 新z预训练 → 三个z下游。

**g组不合并的结果已经在上一包中完成**，不是要重跑它的refit80：

| 数据集 | 选中轮 | Q SR | Q PL | A SR | A PL |
|---|---:|---:|---:|---:|---:|
| 1K subset1078 | 56 | .881930 | .886969 | — | — |
| 3K | 59 | .895417 | .930940 | .819549 | .890800 |
| 2023 | 40 | .867057 | .880193 | .805659 | .807477 |

入口对g组复用时检查完整单次CSV、连续60轮、科学配置一致、环境记录中来源已加载校验，以及当前pth指纹与旧来源一致。只忽略输出路径和加载器记录的派生统计。不能把refit80合并快照冒充不合并结果。不完整或不一致时停止，不覆盖或删除旧目录。

Z预训练如果已有匹配配置的完整日志和来源checkpoint，也会复用；否则新训练，保存独立zii来源。它仍是50轮上限、15轮patience、component-anchor验证选预训练权重。额外两个记录开关（batch预算/划分清单）用于可追溯，不改变loss或模型。CUDA随机性仍可能影响结果。

## 只运行需要的部分

```bash
# 只查看/复用已完成g对照，不重跑预训练
bash scripts/run_unmerged_ii_experiments.sh g
# 只跑z新实验：先新预训练，再三个不合并下游
bash scripts/run_unmerged_ii_experiments.sh z
# z预训练已经完成后，只跑某个下游
bash scripts/run_unmerged_ii_experiments.sh z-downstream agiqa3k
# 只跑z预训练
bash scripts/run_unmerged_ii_experiments.sh pretrain-z
# 预览命令，不训练
bash scripts/run_unmerged_ii_experiments.sh all --dry-run
```

默认Python为`/home/sll/anaconda3/envs/WZX-AI/bin/python`，用ABLATION_PYTHON覆盖。不开GPU编号，子进程使用同一Python环境、HF_HUB_OFFLINE=1。中断后重新调用会复用已经完整完成的阶段；尚未完成的下游输出不支持自动恢复（未保存下游pth），会停止并要求保留/调查。

## 输出

G组复用：

```text
outputs/agiqa1k_subset1078_protocolv2_both_noce_noneg_dev_seed42.csv
outputs/agiqa3k_protocolv2_both_noce_noneg_dev_seed42.csv
outputs/aigciqa2023_protocolv2_both_noce_noneg_dev_seed42.csv
```

Z组：

```text
outputs/richhf18k_zii_noce_noneg_pretrain_dev_seed42.csv
outputs/agiqa1k_subset1078_zii_noce_noneg_unmerged_seed42.csv
outputs/agiqa3k_zii_noce_noneg_unmerged_seed42.csv
outputs/aigciqa2023_zii_noce_noneg_unmerged_seed42.csv
```

配置在`configs/zii_noce_noneg/`。下游新配置相对对应g组只改预训练来源和输出标识，不更改分支、损失、学习率、split、SafeAug/整图输入、文本上下文或BN策略。两套不能按测试分别挑数据集或Quality/Align epoch后拼接。

## 检查范围

配置单因素检查、流程无合并命令检查、完整结果复用不启动训练、来源或超参数变化拒绝、refit快照拒绝、不完整结果拒绝，以及此前I-I-z对Fusion/局部分支有梯度、PQ独立的模型测试。本地未运行真实数据/GPU训练。

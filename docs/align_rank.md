# 一致性成对排序权重单因素对照

本轮只有两档`lambda_rank_align`：原0.03 → **arank050=0.015 / arank200=0.06**。每档只跑AGIQA-3K、AIGCIQA2023，共4次下游；不跑1K、不重新预训练，不自动接其他候选实验。

全部恢复原v2基线：一致性LR=1e-4、Soft-SROCC权重=0.04、共享温度=0.02、PI/PT初始LR=1e-4、上下文/全局投影各1e-5；Quality设置及其他参数不变（3K Quality Soft-SROCC=0.02，2023=0.04）。原六维legacy NSS、projection_only_v1、5上下文词元，negative和质量文本CE关闭。`lambda_rank_align`是当前启用的成对排序项，与关闭的Prompt负样本排序`lambda_align_rank`不是同一参数；不改变共享rank_margin=0.02或hard_rank_ratio=0.35。

固定原基线预训练SHA256：`fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686`。训练seed42、split_seed42、train/val/test分开、60轮，按max val_quality_SROCC选择同一轮，统一报告Quality/Alignment SROCC/PLCC。不因关注一致性而改变选模标准，不按测试集选择轮次。

运行器使用独立命名空间复用上一轮严格核查流程，不修改上一轮变体或输出。配置在`launch_config.yaml`生成并完整保存；增加实际划分成员和优化器初始化记录。运行前核验原预训练来源、完成状态、SHA及基线CSV；运行后核验60轮记录、最大验证Quality选择、初始参数组/LR、实际划分成员、有效总损失及来源加载记录。CPU拒绝真实训练；所有旧结果保持。

## 服务器执行

```bash
pwd
git status --short
git pull --ff-only origin work
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/run_align_rank.py check-configs
```

终端一：0.015，依次跑3K、2023。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_rank.sh low
```

终端二：0.06，依次跑3K、2023。

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_align_rank.sh high
```

单卡一次运行四次：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_rank.sh all
```

GPU编号按可用情况调整。已有输出默认拒绝覆盖。可显式加`--continue`，只跳过核验通过的已完成任务；未完成目录保留并报错，不自动resume。

只打印计划，不读取大模型或启动训练：

```bash
bash scripts/run_align_rank.sh all --dry-run
```

四次全部完成后打包：

```bash
bash scripts/run_align_rank.sh pack
```

输出`projection_only_v1_v2_align_rank_results.tar.gz`，含候选及原基线CSV、配置、日志、运行/来源/完成指纹和成员清单，不含大模型权重。同名已有包拒绝覆盖。

## 只读分支诊断

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/audit_align_branches.py
```

默认检查两份固定基线CSV并核验其SHA；可用`--csv outputs/某次运行.csv`指定完整CSV，或`--output 新文件.json`保存（不覆盖）。始终读取原Quality选中轮的best_val/best_test，不选择其他轮或改用测试最优分支。

根据已上传的基线控制CSV，当前选中轮的测试诊断：

| 数据集 | 轮次 | 全局SROCC | 局部SROCC | 混合SROCC | 全局PLCC | 局部PLCC | 混合PLCC | 平均全局门控γ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3K | 58 | 0.83064 | 0.75883 | 0.83079 | 0.89407 | 0.83390 | 0.89508 | 0.93490 |
| 2023 | 52 | 0.81006 | 0.66306 | 0.80703 | 0.80959 | 0.51064 | 0.80650 | 0.92713 |

对应验证集：3K全局/局部/混合SROCC=0.77066/0.71803/0.77457；2023=0.80692/0.67835/0.80655。局部分支较弱，3K混合略有收益，2023在这两个已记录集合上略低于全局分支。门控γ是`A=γ*A_global+(1-γ)*A_local`中的样本平均系数，不代表每张图固定使用同一权重，也不是重要性或因果归因。

这提示后续可针对局部匹配和门控做独立对照，但本批只改排序权重；不能据这些测试结果直接删除局部分支并把它当成未经过选择的新结果。

## 验证记录

云端CPU：6项专项测试、20项原有相关回归测试通过，4份配置及shell dry-run通过。专项覆盖单因素配置、协议命名空间隔离、真实成对排序损失/梯度、CPU启动拒绝、四次模拟下游/打包、历史结果保护和分支同轮诊断。实际模型与训练器源码未改动，未启动真实训练。

另只读分析了上一轮结果包中的10份完整CSV（2份基线、8份候选）：在各自Quality选中轮，10份的局部测试SROCC均低于全局分支。该观察用于提出后续局部匹配/门控消融方向，不直接替换当前模型。

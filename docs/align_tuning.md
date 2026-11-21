# 原基线下游一致性四组单因素调参

只用已接受v2原基线预训练来源（best_component_anchor_pretrain.pth，SHA256 `fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686`）；不重新预训练，不使用quality_first/residual_oas或其他实验的权重，不覆盖任何旧结果。

固定原六维legacy NSS、projection_only_v1、5上下文词元、完整QA/QL及全局/局部路径，negative和质量文本CE关闭。只运行3K、2023，训练seed42、split_seed42，60轮、train/val/test分开，仍按max val_quality_SROCC选模。两个任务全部测试SROCC/PLCC均报告同一选中轮，不按测试集挑选轮次。1K没有一致性评价，本批不运行1K。

| 变体 | 唯一变化 | 实际初始值 |
|---|---|---|
| alignlr050 | layerwise_lr.align_mult=0.5 | align组LR=5e-5 |
| alignlr200 | layerwise_lr.align_mult=2.0 | align组LR=2e-4 |
| asrocc050 | lambda_srocc_align=0.02 | 对照原0.04 |
| asrocc200 | lambda_srocc_align=0.08 | 对照原0.04 |

align组包括文本维度映射T、AG/AP/AT、局部匹配投影、一致性评分头及gate。PI/PT属于shared_projection组，不随align_mult改变，仍为1e-4；上下文1e-5、全局投影1e-5、CLIP/ResNet各2e-6，QA及局部质量组按基线。Quality Soft-SROCC在3K为0.02、2023为0.04；共享Soft-SROCC温度0.02；其他损失权重及优化器/调度策略原样。修改一致性目标可能通过共享视觉特征影响质量，不能承诺质量成绩不变。

新增配置只增加split成员表及优化器初始化记录，便于核验。执行前验证原预训练完成记录、指纹及基线控制CSV，3K/2023控制必须匹配current_tuning_reference.json的CSV SHA。执行后核验实际成员、全部LR组、来源加载记录、完整60轮及验证选模，重构日志中的有效总目标以检查质量/一致性权重。打包含候选和基线控制证据，不含模型权重。

## 执行

```bash
pwd
git status --short
git pull --ff-only origin work
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/run_align_tuning.py check-configs
```

终端一，两个LR候选，各跑两个数据集，4次下游：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_tuning.sh lr
```

终端二，两个Soft-SROCC权重候选，各跑两个数据集，4次下游：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_align_tuning.sh srocc
```

GPU编号按可用情况选择。单卡顺序跑8次：`CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_tuning.sh all`。单次例子：`bash scripts/run_align_tuning.sh alignlr050 agiqa3k`。队列可以添加数据集名仅跑该数据集。

`--dry-run`只打印配置计划，不读取大权重，不启动训练或建立输出。实际运行仍会核验来源。

```bash
bash scripts/run_align_tuning.sh all --dry-run
```

已有同名输出默认拒绝覆盖。需要跳过已完成任务时显式加`--continue`，仅在完成证据核验通过后跳过；部分输出仍拒绝，不自动resume，不改名或删除历史结果。

## 打包

八次下游全部完成后：

```bash
bash scripts/run_align_tuning.sh pack
```

输出`projection_only_v1_v2_align_tuning_results.tar.gz`。含CSV、配置、日志、运行元数据、划分成员清单、来源/完成指纹和原基线控制CSV，排除模型权重；已有同名包拒绝覆盖。

## 只读选模诊断

服务器核查固定基线CSV：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/audit_align_selection.py
```

也可`--csv outputs/某次运行.csv`，或`--output 新文件.json`保存报告（不覆盖）。报告只用验证集，不构造未保存轮次的测试指标，不改选模协议。

基于此前已上传、SHA与固定参考一致的控制CSV，诊断如下：

| 数据集 | Quality选中轮 | Alignment验证最佳轮 | Quality轮的Val Align SROCC | 最佳Val Align SROCC | 差距 |
|---|---:|---:|---:|---:|---:|
| 3K | 58 | 51 | 0.774572 | 0.778310 | 0.003738 |
| 2023 | 52 | 36 | 0.806550 | 0.814372 | 0.007822 |

两者存在选模差距，但本轮仍保留原规则。以上是验证指标差异，不能据此推断另一个轮次的测试成绩；后续如改为双任务选模，应独立定义和报告。

## 验证

云端CPU：10项专项测试、10项既有Soft-SROCC运行器回归测试通过；8份配置检查与shell dry-run通过。专项测试使用真实优化器验证参数成员完全不变、仅align组LR变化，PI/PT和上下文/全局投影LR不变；使用真实损失验证仅一致性Soft-SROCC加权项变化，并核验来源/预算/配置篡改拒绝、实际成员表、选模一致性、失败停止、部分目录保护及打包排除权重。另以已审计基线CSV的120轮真实日志验证总损失重构，最大绝对误差分别约1.06e-8、2.24e-8。未启动真实训练或改动模型与训练器源码。

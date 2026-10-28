# 两阶段去质量五档 CE / negative，再做合并训练

这是新实验 `both_noce_noneg`，与之前只在下游关闭的 `noce_noneg` 分开保存。

## 具体修改

预训练：

- 质量五档文本分支关闭，lambda_quality_text_ce=0，lambda_quality_text_anchor=0。
- 新开关 `train.disable_negative_relations=true`：RichHF 局部区域/短语监督只保留正例 pull，不执行负例区域/短语投影，也不执行 push/负例相对排序。保留原正例的 0.5 系数，不因去掉负例而把正例权重翻倍。
- Prompt–Answer gap 推远项权重保持 0。
- 预训练没有下游旧错配 Prompt 的 lambda_neg 机制，不能通过给预训练随便加 lambda_neg=0 来声称负例已经去除。
- 四路常规 SimSiam、MOS 监督、Artifact/Aesthetics Component Anchors/Ordinal、监督样本排序、PCGrad、QuickGELU、layer3 等沿用原设置。Soft-SROCC 按各项原配置保留，不额外开关。

下游开发与合并重训：

- 五档质量分支关闭，CE/anchor 权重为 0；旧错配 Prompt 的 lambda_neg=0。
- QA、NSS 局部质量、PI/PT、真实 Prompt 文本路径、Align 全局/局部分支、MSE/排序/Soft-SROCC 保留。
- 三个数据集统一使用这次新预训练的 `best_component_anchor_pretrain.pth`，不再使用带质量 CE 预训练的旧权重。

## 一条命令执行完整流程

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_two_stage_noce_noneg.sh
```

入口包含 `set -euo pipefail`，任一步报错立即停止。若新预训练已完成而下游尚未开始，使用 `bash scripts/run_two_stage_noce_noneg.sh downstream`。若下游已有部分输出，按下面分阶段命令从未完成阶段继续，不覆盖既有输出。

## 分阶段执行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

# 1. 两阶段一致的新预训练来源，只需一组
bash scripts/run_protocol_v2.sh pretrain both_noce_noneg

# 2. 用新来源估计各数据集停止预算，再合并 train+val 重新训练
for ds in agiqa1k agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh dev both_noce_noneg "$ds"
  bash scripts/run_protocol_v2.sh plan both_noce_noneg "$ds"
  bash scripts/run_protocol_v2.sh refit both_noce_noneg "$ds"
done
```

每条命令成功后再运行下一条；循环内任一步失败应停止，检查错误后从对应阶段继续。若希望整段自动遇错停止，可在脚本文件开头使用 `set -euo pipefail`。已有输出不会被覆盖，重新启动不需要再重复完成的预训练。

新预训练来源：

```text
outputs/richhf18k_protocolv2_both_noce_noneg_seed42/best_component_anchor_pretrain.pth
```

开发结果：

```text
outputs/<dataset>_protocolv2_both_noce_noneg_dev_seed42.csv
```

合并重训结果：

```text
outputs/<dataset>_protocolv2_both_noce_noneg_refit80_seed42.csv
```

预算计划：`outputs/protocolv2_plans/<dataset>_both_noce_noneg.{json,yaml}`。其中记录完整原 split、来源 SHA256、开发 CSV SHA256、配置指纹以及验证指标选出的固定处理 batch 数。

## 合并训练的协议

开发阶段仍为 70/10/20，按验证质量 SROCC 选停止预算。因为预训练来源及其学习到的特征改变，旧来源下的 3008/9620/10710 batch 预算不能直接当作新来源已验证的最优预算，因此必须重新运行这一步。

合并阶段从新预训练来源重新初始化，使用原 train+val 样本作为 80% 训练集，保留原 20% 测试索引；无验证选模。保持批大小并使用确定的处理 batch 预算，最后一轮可截断。LR 按原开发预算轨迹换算，成功更新/AMP 跳过另记。结束后只评估一次测试，不按测试挑权重。原开发模型不会被继续训练，也不会用它的参数代替来源权重。

预训练本身仍保留原 RichHF 官方划分和早停/选模规则，本次合并的是三个下游数据集的 train+val。新模型结束时不保证是所有中间模型中最好的，这是固定预算协议的限制。80/20 与开发阶段 70/10/20 结果分别报告。

## 本地检查

负例关闭开关默认 false，因此旧配置行为保持。新增测试检查：负例投影/push/rank 不运行、正例梯度保留、正例系数不翻倍、旧默认仍执行原负例、新预训练配置与来源路由正确。原固定预算测试覆盖 split 不重叠、测试集身份不变、精确步数截止及最终单次测试。云端无真实数据/GPU，未代跑训练。

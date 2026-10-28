# 下一组下游实验：关闭质量五档 CE 和旧错配 Prompt 损失

按用户最新要求，新增 `noce_noneg`，从 protocol v2 baseline 独立初始化：

- `model.quality_text_bank.enabled=false`：关闭五档质量文本辅助分支，不只是将最终分数 blend 置零。
- `lambda_quality_text_ce=0`，`lambda_quality_text_anchor=0`。
- 按此前要求，`lambda_neg=0`，不再运行旧错配 Prompt 负样本前向。
- Quality/Align 的 MSE、样本排序、Soft-SROCC 等其余配置保持 baseline 设置。
- 保留 QA、NSS 局部质量分支、PI/PT、Align adapters 和真实 Prompt 文本路径。关闭质量等级文本分支不等于关闭 Prompt 文本编码器。

本组是下一套候选训练设置，同时改变 CE 与旧 negative，不能用于单独判断 CE 的贡献。旧实验配置和结果保持不变。

这里关闭的是接下来的**下游**五档 CE；预训练保持现有 protocol v2 权重，它在预训练时使用过五档 CE。因此不能将本组描述为“两阶段都无 CE”。不需要重新预训练。

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
for ds in agiqa1k agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh dev noce_noneg "$ds"
done
```

结果文件：`outputs/<dataset>_protocolv2_noce_noneg_dev_seed42.csv`，配置/环境记录在同名目录。不覆盖旧 baseline。仍使用验证质量 SROCC 选模，不按 loss 最小或测试成绩选模。

先完成开发实验并审查验证结果，再决定是否合并 train+val。若采用本设置，必须从新 CSV 生成预算，不能复用旧 baseline 预算：

```bash
# 审查开发结果后执行，不与上一段连续盲跑。
for ds in agiqa1k agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh plan noce_noneg "$ds"
  bash scripts/run_protocol_v2.sh refit noce_noneg "$ds"
done
```

固定预算重训不能保证结束模型是所有中间轮次里最好的；它执行提前确定的停止规则，避免以测试集代替验证集。

# 用户确认的独立 AGIQA-1K 1078 图子集协议

用户确认：先用剔除冲突图片的1078图独立子集继续，并单独报告。本实现不再等待原始勘误，也不声称恢复了完整1080图的正确标签。

## 数据处理和划分

- 原 `AIGC-1K_answer.csv`、Excel、图像和历史结果全部保留。
- 只剔除 `dreamStudio_8_9_1_1.jpg` 的两条冲突记录（原0-based行1078、1079），不任选一个 MOS、不平均，不把任何 MOS/Prompt/Answer 分配给未引用图片 `dreamStudio_8_5_0_0.jpg`。
- 1078条保留记录的所有字段与原CSV完全相同，按原行序存入 `data/AGIQA-1K/revisions/subset1078_v1/AIGC-1K_answer_unambiguous1078.csv`。
- **不对剩余样本重新洗牌**：先重建原1080行 seed42 划分，再从各split中移除冲突记录。保留其他样本原来的训练/验证/测试归属。

| 阶段 | train | val | test |
|---|---:|---:|---:|
| 原1080行（存在冲突） | 756 | 108 | 216 |
| 新1078图开发实验 | 755 | 108 | 215 |
| 新1078图合并重训 | 863 | 0 | 215 |

因此最终训练比例约80%，测试集合相较旧版仅移除了那张冲突图片，其余215张原测试图片仍在测试集合。原1K开发模型曾接触测试图片，不继续用它微调；各新实验从自己原来的 RichHF 预训练来源重新初始化。

使用显式 split_manifest.json 锁定集合、原行映射和来源摘要；加载时核对清单和CSV内容。缺失清单、修改划分/标签或更换来源表都会报错，不能静默退回随机划分。训练 min/max 只拟合755张训练样本；合并时仅重新拟合863张train+val。

## 一键运行两套1K新实验，不重跑预训练

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_agiqa1k_subset1078.sh
```

自动准备/校验新数据版本，接着顺序运行：

1. `both`：使用此前预训练不合并的 `richhf18k_protocolv2_both_noce_noneg_seed42/best_component_anchor_pretrain.pth`，重新做1078图开发选预算，再合并863张训练、测试215张。
2. `allmerged`：使用此前已合并预训练的 `richhf18k_allmerged_noce_noneg_refit_seed42/final_merged_pretrain.pth`，执行同样的1078图开发和合并过程。

两阶段 CE/负例推远关闭、Soft-SROCC 等损失按对应旧配置保留。这个脚本只运行1K，不影响3K/2023。两套都共用同一1078图数据与split；旧数据/模型结果不覆盖。

数据准备可重复执行：来源指纹匹配且现有版本完全一致时只校验，不覆盖。来源与用户审查的CSV/Excel不同会停止，不自动处理其他标注版本。训练输出拒绝覆盖已有结果；任一步失败立即停止。

## 单套或分阶段执行

```bash
# 只准备和校验子集，不训练
bash scripts/run_agiqa1k_subset1078.sh prepare

# 只运行其中一套的完整开发→计划→合并训练
bash scripts/run_agiqa1k_subset1078.sh both
bash scripts/run_agiqa1k_subset1078.sh allmerged

# 某套开发已经完成时，从尚未完成的阶段继续
bash scripts/run_agiqa1k_subset1078.sh both plan
bash scripts/run_agiqa1k_subset1078.sh both refit
# allmerged 同理；不要重跑已有的dev阶段覆盖结果
```

新CSV摘要：`60e7d83a36bf5daa545d05ef979618d72821fcb3bd9e5e87874fe2563b22fde3`。同目录 `annotation_audit.json` 记录原表指纹和剔除策略，`split_manifest.json` 记录新/旧索引对应。

## 输出与报告

```text
outputs/agiqa1k_subset1078_protocolv2_both_noce_noneg_dev_seed42.csv
outputs/agiqa1k_subset1078_protocolv2_both_noce_noneg_refit80_seed42.csv
outputs/agiqa1k_subset1078_allmerged_noce_noneg_dev_seed42.csv
outputs/agiqa1k_subset1078_allmerged_noce_noneg_refit80_seed42.csv
outputs/subset1078_plans/both.{json,yaml}
outputs/subset1078_plans/allmerged.{json,yaml}
```

新开发仍按验证质量SROCC确定预算，不沿用旧1080行最优轮/步数。合并阶段固定预算结束后只测试一次，无验证选模，不用测试选终点。最终看 refit80 CSV 的 final.metrics_json.test。

必须标记为“AGIQA-1K subset1078，剔除冲突图片后的独立协议，test N=215”。不要和旧1080行/test216或论文标准协议混为同一基准，也不能用这份子集成绩直接宣称标准AGIQA-1K SOTA。可在自己的两套新方法间做同子集对照。

## 验证范围

已对用户上传的真实标注生成并核对候选、来源与保留字段，验证准备脚本幂等。自动测试覆盖真实数据加载器的755/108/215划分、未变动成员归属、跨split无图片重叠、缺失或改动清单/CSV拒绝、863图合并与215图测试独立、新预算和两套脚本路由。不包含真实图像/GPU训练。

# AGIQA-1K 提供的 Excel 与 Answer CSV 全表核查

用户上传 `agiqa1k_annotations.tar.gz` 内的两份标注已逐行核对。原文件未覆盖，原训练/测试结果未更改。

| 核查项 | 结果 |
|---|---:|
| CSV 行数 / 唯一图片名 | 1080 / 1079 |
| Excel 行数 / 唯一图片名 | 1080 / 1079 |
| 按原行序图片名不一致 | 0 |
| Prompt 去首尾空白后不一致 | 0 |
| MOS 最大绝对差 | 0.00004998126432997907 |

MOS 差异与 CSV 保留四位小数一致。问题不是 CSV 相对这份 Excel 的 name/MOS/Prompt 发生行序错位：**两份表自身都存在相同的冲突**。

| 原表0-based行索引 | 两表图片名 | CSV MOS | Excel MOS（约） |
|---:|---|---:|---:|
| 1078 | dreamStudio_8_9_1_1.jpg | .4414 | .441416 |
| 1079 | dreamStudio_8_9_1_1.jpg | 2.0841 | 2.084116 |

用户服务器目录核查另确认 `dreamStudio_8_5_0_0.jpg` 存在，却没有被两表引用。不能仅凭唯一未引用图片，就认定最后一行 MOS/Answer/Prompt 属于它。所提供 Excel 不能成为独立的消歧依据，也不能在未核实来源的情况下宣称上游官方数据一定有同样错误。

来源指纹：

- Answer CSV：`40c19836a31a3d07dd3ad7f61824962fc259c23fd3ccb813eb2c0aa99fba8bbd`
- Excel：`f958adab2712d6ea6c2b0777f1ca892f3d35d3956a893e78cf6090309e6a8ea3`

## 可供选择的后续方案

1. 暂停1K，获取可靠的原始 name–MOS/Prompt 标注或勘误；确认后另存完整修正版。3K/2023 独立继续，无需重跑预训练。
2. 新建明确标记为 **1078图独立子集** 的实验：排除冲突图片的全部两条记录，不任选其中一个MOS、不平均、不重新分配漏标图片的标签。未标注的另一张图继续不使用。重新开发选预算、合并训练；该结果不是标准1080图协议，不能直接用于宣称与论文同协议的 SOTA 对比。现有测试图片重叠导致已完成1K开发结果不能继续作为无泄漏结果使用。

已在云端准备候选1078行CSV供审查，但未接入训练配置；等待用户选择，不自动改变科研协议。候选只保留原记录字段，并不声称逐张检查了生成 Answer 与图像是否相符。

## 可重复的只读核查/候选生成

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/prepare_agiqa1k_annotation_audit.py \
  --csv data/AGIQA-1K/AIGC-1K_answer.csv \
  --xlsx data/AGIQA-1K/AIGC_MOS_Zscore.xlsx \
  --output-dir outputs/agiqa1k_annotation_audit_v1
```

只生成报告；若明确要候选子集，在命令末尾加 `--write-subset`，它会写到新目录，拒绝覆盖已有目录，且不修改源表或既有划分。新表需要独立划分与新实验标记，不能直接替换旧文件接着跑旧预算。

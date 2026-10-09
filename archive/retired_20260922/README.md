# 已退出当前实验路线的文件归档（2026-09-22）

本目录保留历史文件原内容，未删除实验 CSV、数据集、日志或检查点。
归档依据是对话中以共享投影的 Quality Anchor + Ordinal 替代 embedding triplet 的路线决定，以及当前运行配置的引用核查；不以一次低分自动认定方案无效。

## 归档清单

归档路径相对本目录；原路径相对项目根目录。configs 子目录保留原目录结构。

| 原路径 | 归档路径 | 原因 |
| --- | --- | --- |
| configs/richhf18k_cm_pretrain_local_feedback_dual_projection_quality_triplet_balanced_lr.yaml | configs/richhf18k_cm_pretrain_local_feedback_dual_projection_quality_triplet_balanced_lr.yaml | 已被共享投影 Anchor/Ordinal 路线替代的双投影三元组预训练配置 |
| configs/agiqa1k_cm_iqa_dual_projection_quality_adapter_srocc.yaml | configs/agiqa1k_cm_iqa_dual_projection_quality_adapter_srocc.yaml | 上述历史预训练的 adapter 下游配置 |
| configs/agiqa1k_cm_iqa_dual_projection_quality_adapter_raw_srocc.yaml | configs/agiqa1k_cm_iqa_dual_projection_quality_adapter_raw_srocc.yaml | 上述历史预训练的 raw adapter 下游配置 |
| configs/agiqa1k_cm_iqa_dual_projection_quality_adapter_parallel_score_srocc.yaml | configs/agiqa1k_cm_iqa_dual_projection_quality_adapter_parallel_score_srocc.yaml | 上述历史预训练的 parallel-score adapter 下游配置 |
| train_cm_iqa.py.orig | ../../old/train_cm_iqa_legacy.py | 旧代码备份已集中到 `old/`；当前入口为 train_cm_iqa.py |
| train_cm_pretrain.py.bak_before_clean_anchor | ../../old/train_cm_pretrain_before_clean_anchor.py | Clean Anchor 修改前备份已集中到 `old/`；当前入口为 train_cm_pretrain.py |

## 保留范围

- 当前 Clean View、独立/合并损失加权和采样对照仍在共用配置中，通过参数启用；不归档共用配置。
- 保留当前三个下游配置、IDE 中的 mean_residual 配置，以及有效基线和未明确舍弃的历史配置。
- models、datasets、utils、训练入口仍被当前实验使用。旧方案在这些共享模块中的可选分支不作拆分或删除。
- 数据处理工具和诊断工具保留；不因近期未运行就视为废弃。
- 核查时三个 best_component 下游正在运行；其配置不在归档清单内。

## 恢复与复现

将清单中的归档文件移动回对应原路径即可恢复；备份源码现位于 `old/`，详见 `old/README.md`。若原路径已有文件，先比较内容，不覆盖。
旧日志中的配置路径保留历史原值，可通过上表查找归档位置。
历史 YAML 内容未修改，其中数据、输出和检查点路径仍相对项目根目录；从项目根目录运行时，可将 --config 改为本目录下的 YAML 路径。


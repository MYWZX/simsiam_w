# 当前下游调参对照 v2（2026-11-06 用户指定）

1K和3K保留此前参数；2023使用Quality Soft-SROCC权重0.04。以后新调参应明确从 configs/current_tuning_reference 复制，并以本页对应运行作为对照。当前未启动新实验。

| 数据集 | 训练seed | 划分seed | Soft-SROCC权重 | Q-SROCC | Q-PLCC | A-SROCC | A-PLCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| agiqa1k | 42 | 3407 | 0.03 | 0.885922 | 0.909385 | 无标签 | 无标签 |
| agiqa3k | 42 | 42 | 0.02 | 0.901767 | 0.936847 | 0.830785 | 0.895084 |
| aigciqa2023 | 42 | 42 | 0.04 | 0.880600 | 0.892680 | 0.807026 | 0.806497 |

三数据集平均Quality SROCC：0.889429；PLCC：0.912971。

其他参数恢复原对照：CLIP/ResNet初始LR2e-6，QA/PI/PT/quality初始LR1e-4，semantic_proj初始LR1e-5，alpha继承，beta起点0.50，weight_decay0.002，评分头dropout0.20，质量pairwise排序权重0.03，Soft-SROCC温度0.02，60轮、验证Quality SROCC最大选模，不合并train+val。

预训练来源保持原seed42、第4轮Component Anchor选点；SHA256：`fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686`。

本次为用户指定的数据集分别设定训练超参数的探索对照，不是统一Soft-SROCC权重，也不代表全新训练或独立确认。原始seed42基线、跨seed/跨划分及失败调参记录保留。

原 docs/current_baseline.json 与 configs/projection_only_v1_baseline 保持历史原样，供既有来源指纹检查使用；旧运行脚本仍使用各自原对照，不会自动采用v2。未来新运行器必须显式接入v2，不能把旧对照比较结果标成v2收益。

本目录配置用于记录所选已完成运行，不要直接对现有output_dir执行训练；新实验必须生成独立run_id/output_dir/CSV。完整精度指标与CSV哈希见 current_tuning_reference.json。

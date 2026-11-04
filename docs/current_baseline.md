# 当前统一最好基线（用户指定，2026-11-04）

后续实验统一与本页的完整 projection_only_v1 基线比较。旧 attention 模式、旧1080图及 train+val 合并实验保留为历史记录，不再作为默认比较基准。不得拼接不同配置的单数据集最优值。

“最好基线”表示用户选定的当前工作基准，不表示其每项指标优于全部历史结果，也不构成多 seed 显著性结论。

| 数据集 | 选中epoch | Q-SROCC | Q-PLCC | A-SROCC | A-PLCC |
|---|---:|---:|---:|---:|---:|
| agiqa1k_subset1078 | 60 | 0.890823 | 0.892999 | 无标签 | 无标签 |
| agiqa3k | 58 | 0.901767 | 0.936847 | 0.830785 | 0.895084 |
| aigciqa2023 | 57 | 0.874495 | 0.888681 | 0.807236 | 0.806281 |

平均 Quality SROCC：0.889028；平均 Quality PLCC：0.906176。

协议：5上下文词元；g-only PI/PT；P0绕过PI；双质量残差、NSS、Component Anchor及Ordinal保留；质量文本CE和negative关闭。seed=42、split_seed=42。下游PI/PT初始LR=1e-4，全局投影LR=1e-5。1K为subset1078（train755/val108/test215）；不合并train+val；三个下游均60轮预算，按验证Quality SROCC最大选模。

预训练第19轮早停，按验证Component Anchor SROCC选中第4轮（0.7163169500600057）。三个下游使用同一新来源：

```text
/home1/wzx/aigc_naturalness_code/simsiam/outputs/richhf18k_projection_only_v1_baseline_ctx5_dev_seed42/best_component_anchor_pretrain.pth
SHA256: fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686
```

代码提交：`b52d1fd0bd6074bd37e1bd3876ddd9518a0cf1b0`。配置：`configs/projection_only_v1_baseline`。

原始结果包：`projection_only_v1_baseline_ctx5_downstream_results.tar.gz`；SHA256：`49e5699798628798f93fc4f0f36ef0301cbae6377ae5272b80d8cb29d7a517e7`。

包内完成记录、配置及来源指纹已核查。包不含权重，云端未独立重算实际权重SHA256；指纹来自服务器校验记录。workspace_dirty不能单独证明源码变化。

完整精度指标、各CSV哈希与来源指纹见 [current_baseline.json](current_baseline.json)，表格见 [current_baseline.csv](current_baseline.csv)。历史 current_baseline_merged_results.* 不再是当前默认基线。

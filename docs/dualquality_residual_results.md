# 双质量残差layer3 / layer3+4实验结果

来源：cliplarge_dualquality_residual_layer3_results.tar.gz及cliplarge_dualquality_residual_layer34_results.tar.gz。对照为此前双质量replace/layer3，不与普通layer3+4结构混为同一方法。

固定选模：1K取质量验证选模；3K/2023取joint_selected，同一epoch同时报告Quality和Align。不依据测试结果改选quality_selected。

| 数据集 | 方案 | 选中轮次 | Q PLCC | Q SROCC | A PLCC | A SROCC |
|---|---|---:|---:|---:|---:|---:|
| agiqa1k | replace/layer3 | 49 | 0.875121 | 0.881721 | — | — |
| agiqa1k | residual/layer3 | 10 | 0.880435 | 0.888737 | — | — |
| agiqa1k | residual/layer3+4 | 11 | 0.877692 | 0.882941 | — | — |
| agiqa3k | replace/layer3 | 31 | 0.931198 | 0.893735 | 0.891616 | 0.814252 |
| agiqa3k | residual/layer3 | 31 | 0.933937 | 0.898755 | 0.890514 | 0.811653 |
| agiqa3k | residual/layer3+4 | 40 | 0.932284 | 0.894309 | 0.891668 | 0.814126 |
| aigciqa2023 | replace/layer3 | 59 | 0.885208 | 0.869393 | 0.788612 | 0.789584 |
| aigciqa2023 | residual/layer3 | 57 | 0.888763 | 0.874559 | 0.788686 | 0.789401 |
| aigciqa2023 | residual/layer3+4 | 45 | 0.888513 | 0.873777 | 0.788825 | 0.786597 |

## 配对差值

| 数据集 | 残差layer3−replace ΔQ PLCC | ΔQ SROCC | ΔA PLCC | ΔA SROCC |
|---|---:|---:|---:|---:|
| agiqa1k | +0.005314 | +0.007015 | — | — |
| agiqa3k | +0.002739 | +0.005020 | -0.001102 | -0.002599 |
| aigciqa2023 | +0.003555 | +0.005166 | +0.000074 | -0.000183 |

| 数据集 | 残差layer3+4−残差layer3 ΔQ PLCC | ΔQ SROCC | ΔA PLCC | ΔA SROCC |
|---|---:|---:|---:|---:|
| agiqa1k | -0.002743 | -0.005796 | — | — |
| agiqa3k | -0.001652 | -0.004446 | +0.001153 | +0.002473 |
| aigciqa2023 | -0.000250 | -0.000781 | +0.000139 | -0.002804 |

## 完成及来源核对

八个CSV各有一次run_start和一次final，未见重复追加启动。预训练layer3选8轮、23轮正常早停；layer3+4选3轮、18轮正常早停；均为原50轮上限/15轮patience。1K分别选10/11轮，在30/31轮正常早停。四次3K/2023均完成60轮。

配置均为seed/split42、Large、质量CE0.005、双质量残差模式。下游未重置匹配局部投影，质量与Align全局输入均为g。3K/2023四份元数据均为af2730b18f0d9258a6aeff66d5922ba3336583cd、workspace_dirty=false、checkpoint_verified_after_load=true，同一方案两个数据集的来源哈希一致。

layer3来源SHA256：ffc52bc71ec6fcdea7cc4a1fce345aea8912f0de8657f1bc85f327de3a2246cd；layer3+4来源：95a1d0faf3116ee21ac502465ffcc00e366c0fbbab31bcc7ef43d7be2575427f。未上传pth，未独立复算文件哈希；1K只有配置路径，没有上述同等级运行元数据。

## 权重及分支诊断

layer3主来源alpha=0.122259，三个下游选中点约0.1282–0.1318；eta约0.4091–0.4103，beta约0.4306–0.4866。layer3+4来源alpha=0.108290，下游约0.1151–0.1184；eta约0.4548–0.4582。参数没有停在初始alpha=0.1，也没有出现eta趋近0关闭局部辅助表示。

layer3+4的layer3权重约0.4993–0.4995，仍接近等权；新增layer4没有给当前双质量残差方案带来质量提升。不能把这个结果推广为所有模型中多层一定无用。

layer3在1K的Qg PLCC由replace的0.844584提高到0.865354，最终Q由0.875121提高到0.880435；支持保留原始g旁路对当前结构有帮助。2023的Qg=0.888478/0.874116，最终Q=0.888763/0.874559，局部融合补充较小，不能把beta约0.48解释为局部贡献了48%的预测能力。

## 历史最好与限制

2023 Quality SROCC=0.874558700，为目前已上传SafeInput系列的新最好值；比此前逐图质量门控的0.874286130高0.000272570。其PLCC=0.888763368仍略低于此前0.888853770（差-0.000090402）。两者不是同一模型的双指标纪录，不能拼接。

3K质量历史最好仍是普通layer3+4：0.935095599/0.902454561。1K PLCC/SROCC历史最好仍为原统一基线0.888547943及CE0.0025实验0.889329674（不同模型）。当前残差layer3尚未解决1K PLCC和Align差距。单seed的小差异不等于统计显著或SOTA。

## 更值得验证的下一项：预训练选模

当前主来源按Component Anchor选择，选中较早，并不直接优化图文一致性。现有同一轨迹已保存best_joint_pretrain.pth，无需重新预训练即可做独立来源控制。

| 局部阶段 | 来源 | 轮次 | RichHF val Quality SROCC | val Align SROCC | val Component Anchor SROCC |
|---|---|---:|---:|---:|---:|
| layer3 | component_anchor | 8 | 0.761273 | 0.528276 | 0.713700 |
| layer3 | joint | 23 | 0.764765 | 0.571971 | 0.712218 |
| layer34 | component_anchor | 3 | 0.749832 | 0.464462 | 0.715805 |
| layer34 | joint | 16 | 0.756082 | 0.558612 | 0.700877 |

预训练val_joint_SROCC在本两输出配置下是Quality与Align SROCC的算术均值，与下游joint_selected使用的调和平均指标不同。

建议先固定残差layer3，只将下游三个数据集的统一来源由best_component_anchor_pretrain.pth换为best_joint_pretrain.pth，另设目录。该来源由预训练验证集确定，不按下游测试成绩挑epoch；不能事后按数据集选择两种来源。现有日志只说明这个控制值得做，不保证下游Align一定改善。暂不继续叠加多层或同时调整损失权重。

# 双质量残差layer3：Joint预训练来源结果

来源：用户上传cliplarge_dualquality_residual_layer3_pretrainjoint_results.tar.gz。对照为同次layer3预训练的Component Anchor来源，不是改变下游验证选模策略。

预训练和下游都使用layer3。比较仍固定1K质量验证选模、3K/2023 joint_selected；Quality与Align来自同一个下游选中epoch。

| 数据集 | 预训练来源 | 下游选中轮次 | Q PLCC | Q SROCC | A PLCC | A SROCC |
|---|---|---:|---:|---:|---:|---:|
| agiqa1k | Component Anchor（预训练8轮） | 10 | 0.880435 | 0.888737 | — | — |
| agiqa1k | Joint（预训练23轮） | 28 | 0.880836 | 0.879686 | — | — |
| agiqa3k | Component Anchor（预训练8轮） | 31 | 0.933937 | 0.898755 | 0.890514 | 0.811653 |
| agiqa3k | Joint（预训练23轮） | 29 | 0.932404 | 0.898612 | 0.892954 | 0.818054 |
| aigciqa2023 | Component Anchor（预训练8轮） | 57 | 0.888763 | 0.874559 | 0.788686 | 0.789401 |
| aigciqa2023 | Joint（预训练23轮） | 52 | 0.881094 | 0.862647 | 0.790975 | 0.790638 |

| 数据集 | Joint−Component ΔQ PLCC | ΔQ SROCC | ΔA PLCC | ΔA SROCC |
|---|---:|---:|---:|---:|
| agiqa1k | +0.000400 | -0.009051 | — | — |
| agiqa3k | -0.001533 | -0.000143 | +0.002440 | +0.006401 |
| aigciqa2023 | -0.007669 | -0.011912 | +0.002288 | +0.001237 |

## 完成及来源检查

三个CSV各含一次run_start和一次final，未见重复启动。1K选28轮、48轮正常早停；3K与2023均完成60轮，joint_selected分别选29/52轮。

与上一轮CSV逐字段比较，三组配置均只改变train.pretrained_checkpoint与run_id/output_dir/results_csv；Large、seed/split42、layer3、残差评分、CE0.005、学习率、安全输入、下游选模策略均保持一致。

3K/2023元数据均记录cab5f6c25a1446d1fc110d59a9475f1d9ed1cf8a、checkpoint_verified_after_load=true，并记录相同Joint来源SHA256：a8327029f420c0f80c8c13d7988b149d02a82e0c1a8e9b7ddd5bec3c8fb5c45b。与上一轮Component来源ffc52bc71ec6fcdea7cc4a1fce345aea8912f0de8657f1bc85f327de3a2246cd不同，支持实际切换了来源。未上传pth，未独立复算文件哈希。1K只有配置路径，没有同等级哈希元数据。

本轮3K/2023记录workspace_dirty=true；包内未附未提交补丁，无法确认其内容。上一轮Component来源记录workspace_dirty=false。有效配置是一项来源对照，但不能据此断言运行时的未提交代码完全相同。

## 权重与解释

Joint来源第23轮预训练alpha=0.142837、eta=0.372170；下游alpha约0.1492–0.1534，eta约0.3723–0.3740。没有回到初始alpha=0.1或eta=0.5的迹象，与完整迁移预期一致。beta约0.4324–0.4825。

3K Align PLCC/SROCC提高0.002440/0.006401，2023提高0.002288/0.001237，支持预训练来源选择会影响Align迁移。但Joint来源同时包含更晚的共享骨干、PQ/QA、局部编码与文本/Align表示，不能把变化单独归因于某一个Adapter。

2023质量PLCC/SROCC下降0.007669/0.011912；1K SROCC下降0.009051。更高的预训练Quality/Align验证指标没有保证更好的下游质量泛化。当前不将Joint来源作为统一替代。

quality_selected的补充记录：3K第54轮Q=0.932755/0.899161，A=0.892660/0.819094；2023第5轮Q=0.880023/0.865468，A=0.781084/0.782116。主结果不因测试成绩改用这些记录。

## 结论

保留Component Anchor来源作为当前双质量残差layer3的主对照；Joint来源保留为选模消融。不能分别取Component的质量指标和Joint的一致性指标，拼成同一个模型结果，也不按数据集切换来源来报告同一套方法。

本轮没有刷新已上传SafeInput系列的单项历史最好：2023 Q SROCC仍为Component来源0.874558700；3K Q仍为普通layer3+4的0.935095599/0.902454561。Joint来源虽然改善了当前结构的Align，仍低于原统一基线3K A=0.893545928/0.824962269、2023 A=0.794016896/0.796561074。单seed结果不构成统计显著证据。

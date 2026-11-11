# I-I 预训练损失权重实验结果

对照 current_tuning_reference v2（I-I=0.15），siam010=0.10、siam020=0.20 在十项下游测试指标上均未超过基线；保持0.15及当前最佳预训练来源不变。完整指标见 pretrain_img_siam_results.csv。

配置与快照均符合单因素方案，完成记录内的CSV/配置/日志/元数据SHA256已核对。两次预训练分别在23/21轮正常早停，按val_component_anchor_SROCC选中第8/6轮，验证SROCC为0.7120728367132345 / 0.7117291849831568（基线0.7163169500600057）。六次下游均完成60轮，选中轮确为val_quality_SROCC最大轮，测试来自该轮而非按test选模。

来源SHA256：
- siam010：a1e8f4a56802afc8f29d44812b6c2ad4ce2afd97ea91331787eaf10bd37c58bb
- siam020：d9df42d27ffcee2923a3cf610d9f0c2475c63fff7a780e6b200f91a27dcdb076

每组3个下游均与本组完成记录、source_fingerprint以及checkpoint_verified_after_load记录吻合。结果包不含权重，未独立重新计算权重文件SHA256。包内未提供独立成员清单，运行seed/配置符合 train42、1K split3407 subset1078、其余split42；不声称重新核对了包外实际成员。workspace_dirty=true本身不证明源码修改。运行记录commit为50463fe8dd081684d9dd81fad7537ea9b47a2f9a。

Quality均值：基线SROCC/PLCC=0.8894294907295097/0.9129708659787698；0.10=0.8813641205950257/0.9063274029062187；0.20=0.8790866032821171/0.9039927017032588。Alignment均值仅计算3K/2023；1K缺失标签不记为0。单seed结果不证明多seed稳定差异。

下一轮改为图像—Prompt权重0.20/0.30，I-I恢复/保持0.15，详情见pretrain_img_prompt.md。新实验不替换基线，未在云端启动真实训练。

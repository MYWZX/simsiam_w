# 当前有效代码审查、过拟合与train+val重训协议

审查对象：当前双质量残差/layer3/shared PI/PT/BN统计冻结路径及P0绕过PI对照，包含数据加载、归一化、预训练投影/损失、下游评分/损失、优化器、BN模式、表示迁移、验证选模与日志。检查对应源码与已有CSV；没有原服务器数据、GPU或未提交补丁，不能证明每张图像/标签/Prompt对应正确，也不声称形式化验证了所有历史可选分支。

## 已修复的代码问题

| 问题 | 证据与影响 | 修复 |
|---|---|---|
| 2023图像/MOS数量不一致时静默截断 | datasets/multi_iqa_dataset.py原使用min(images,q,a)，可能把文件丢失转成错误的位置映射 | 数量必须完全一致且图像非空，否则报错；合法输入顺序不变 |
| MAT字段名错误时回退到第一个数组 | 配置错字可能变成读取错误标签而非报错 | 显式key不存在立即报错；自动识别要求唯一数值向量；拒绝空、矩阵、非有限标签 |
| 配置的Prompt文件不存在时返回空列表 | 整个一致性分支可能默默收到空Prompt | 配置路径不存在立即报错；未配置Prompt的旧默认分支没有自动改协议 |
| FP16零向量归一化产生NaN | 本地直接复现F.normalize(zeros(float16))非有限，默认eps=1e-12在FP16下不可表示 | 投影、Align及SimSiam相关归一化使用eps=1e-6；零输出与普通FP16向量测试通过 |
| 显式冻结视觉编码器只关参数梯度 | 旧maybe_freeze只设requires_grad=false，父model.train会重新开启BN/Dropout | freeze=true时维持encoder eval；默认未冻结配置不变，Fusion仍可训练 |
| 双质量迁移可能部分漏载Align Adapter/U | 原严格检查覆盖QA/局部/PI/PT，却没要求AG/AP/AT/U全部兼容 | 当前双质量模式中追加完整性检查，缺失或形状不兼容立即报错；数据集评分头仍有意重置 |
| 无效指标被当作0相关或0误差 | 非有限值/长度错误可能被safe包装吞掉；N=1误差原本返回0 | 显式拒绝非有限值和长度不符；N=1按实际误差计算，空Align仍允许 |
| 空验证集可能静默选第一轮 | 普通模式的空指标为0，双策略会报样本不足，均不是合法train/test协议 | 训练入口拒绝空/不足2样本的验证集，提示需要独立固定预算refit模式 |

没有更换激活函数、标签量表、数据划分、模型结构或训练预算；没有重算历史成绩。归一化稳定性修复会改变零/极小向量情况的数值行为，未来实验应记录新commit，不与旧二进制运行视作逐位相同。

## 确认存在、但需单独版本处理的问题

**1K归一化使用了完整CSV的MOS极值。** AGIQA1KIQADataset在划分前从整表计算min_mos/max_mos，含测试标签信息；即使传mos_max，min仍来自整表。应使用明确的官方量表上下限，或只在训练划分拟合变换并应用到留出集。当前没有原MOS数据/量表证据，未猜测范围或悄悄更改旧标签；相关修正应与旧基线分开报告。这不证明它导致当前PLCC下降，反而可能使协议偏乐观；PLCC/SROCC对共同线性量表变换本身不敏感，训练目标和固定margin/五档监督仍会受量表影响。

**OpenCLIP激活兼容性。** 本地open-clip-torch=3.3.0，get_model_config('ViT-L-14').quick_gelu=false，而OpenAI tag quick_gelu=true，factory会提示不匹配；项目视觉/文本构造没有显式force_quick_gelu。服务器日志包版本同为3.3.0。需核对实际服务器模块，若修正，应在新版本中统一视觉/文本的预训练与下游，重新预训练；不能只把已有checkpoint的下游切成QuickGELU。现有同一系列内部沿用同种构造，不能把所有近期跌分都归因于此。

**2023仍依赖排序和位置映射。** MAT没有在本接口内按唯一图像ID关联，Prompt采用轮次/每Prompt图像数映射。新增数量检查不能证明排序与发布文件一致，必须在原数据上核对文件编号、MAT顺序和Prompt来源。

**当前划分是按图像randperm，不是Prompt分组。** 同Prompt可能跨训练/验证/测试。对同分布图像评估不自动等于泄漏；若要声称未见Prompt泛化，应增加单独的Prompt分组协议，不能与当前划分混报。

**NSS有冗余通道。** sigma_proxy=pooled_var，两路完全相同，六个通道实际上至少有一项重复；不是维度报错，但不能宣称六个独立统计量。改变描述子要同时考虑预训练NSS投影的语义兼容，不直接替换旧权重输入。

## 与过拟合相关的实现细节

- PI/PT未被单列优化器组，而是进入other，当前LR=1e-4，是视觉编码器2e-6的50倍。冻结BN统计不等于冻结投影权重，仍有约百万级投影参数可调整。以后若想只降低PI/PT LR，应单独建组，不能把other_mult变化误称为只影响PI/PT。
- batch=8时，ResNet和QA的BN仍会更新；目前只冻结PI/PT统计。梯度小不意味着运行统计变化小，视觉LR很低也不能限制BN漂移。
- 当前Align辅助项为lambda_align_aux*(global_MSE+local_MSE)，lambda=0.3。3K所选第54轮局部辅助加权值0.00133772，主Align MSE0.00029951（约4.5倍）；2023第57轮分别0.00050333/0.00007049（约7.1倍）。Soft-SROCC项还会叠加。损失值比例不等于梯度比例，但值得检验辅助目标是否过强。
- 1K quality_only在完整forward_single之后才覆盖Final，并没有跳过Align前向。Align损失关闭也会执行其Dropout/BN，属于无效计算与随机状态副作用；局部PI旁路会改变随机数消费。应将quality-only快速前向作为独立复现修正，而不是把1K质量变化直接解释为Align模块贡献。
- 当前双g下游仍计算z，但Fusion除semantic_proj外的注意力/FFN不参与实际质量与一致性损失。Fusion梯度审计为0是预期，不证明ResNet/L等真正共享模块没有任务冲突。跳过无用Fusion可节省计算，但会改变Dropout随机消费，应记录版本，不能删掉预训练仍需要的z路径。
- shuffle训练DataLoader与验证DataLoader没有各自显式generator，初始化/迭代可能消费全局CPU RNG；set_seed不等于完全隔离模型初始化、加载器顺序及GPU非确定性。
- 训练相关系数是当轮更新过程的在线统计，不能当作同一个最终eval模型的训练集分数。正式过拟合诊断宜使用选中模型+训练索引+eval变换，额外计算train-eval，与val/test同条件比较。
- 单策略最佳state仍deepcopy到原设备，Large模型可额外占用显存；双策略已经采用不可变CPU快照。这不是已观察到的分数错误，但可进一步优化内存。

## 已核对的正确行为

当前主路径中g只来自CLIP全局投影；局部质量来自ResNet/L/NSS/局部编码；qg/ql融合后供辅助质量监督；alpha只影响全局残差评分，eta和beta分别为特征/分数融合。PI/PT位于Align分支，独立PQ/QA没有被意外替换为PI。PI/PT BN冻结通过train覆盖持续生效，仿射参数仍训练。

当前双质量/多层/共享投影迁移有架构与完整性检查，predictor及数据集评分头不迁移是既定设计。选模代码只读取验证指标，测试在训练结束后评估被选中模型；未发现自动根据测试集更新best的路径。但开发中已反复查看同一测试集，所以不能将该测试集描述为从未参与开发判断的盲测集。

## 降低过拟合：建议的实验顺序

在保留现有70/10/20划分和同一个预训练来源的前提下，一次只改一项：

1. **先降低Align辅助评分权重0.3→0.1。** 目的不是追求更低训练损失，而是检验减少局部辅助回归压力是否改善留出泛化；同时记录主/辅损失与真正共享参数梯度。
2. **单独降低PI/PT LR 1e-4→1e-5。** 保留BN冻结，Linear/仿射仍训练。先单列参数组再实验，其余模块LR不动。不要重复此前已做过的全视觉LR降低实验来冒充此控制。
3. **再单独测试QA或ResNet的BN统计冻结。** QA已有freeze_bn_stats控制；ResNet统计冻结需要独立设置，不能等同于冻结全部ResNet参数。本次修复的是显式encoder.freeze=true时完整冻结的语义，不自动开启新实验。
4. 再考虑减少可训练参数（先冻结预训练投影、只训Adapter/头的短阶段，再选择性解冻）或增加评分头正则。不要同时增加dropout、weight decay、换来源、换选模和合并验证集，否则无法定位原因。

IQA增强必须保持标签含义；不建议直接用强颜色抖动、模糊或大裁剪“防过拟合”。换选模指标只改变选择偏好，不等于从训练层面解决过拟合。

以上是建议，不是已完成的新GPU实验，未改变现有启动器参数。

## 可以只用训练集和测试集吗？

可以，但推荐“开发阶段有验证，最终train+val refit无验证”的两阶段协议：

1. 先用原训练/验证集确定模型、损失、学习率与停止预算，禁止用测试成绩确定这些规则。质量选模还是联合选模也提前固定。
2. 固定原测试索引，将原训练和验证索引合并，重新用训练dataset构建Subset（不要简单Concat训练/验证dataset而保留验证变换）。从同一预训练权重重新初始化训练，不是从已选下游模型继续追加训练。
3. 不再每轮选best/早停，只按预定更新步数或轮数训练，结束后一次评估原测试集。输出应标记fixed_budget_refit/final_epoch，不叫best_test模型。
4. 合并后同epoch的更新次数增加，若要保持旧预算，使用预先规定的optimizer steps，或依据新旧len(train_loader)换算。不要直接照搬旧epoch再额外训练约14.3%的步数并称为只增加数据。
5. 保留原测试索引/图像ID及哈希。不能只重算train_ratio=0.8、val_ratio=0就默认测试集合完全相同；整数取整或数据顺序变化可能改变边界。

训练比例70%→80%，训练图片增加约14.3%，不是增加一倍，也不保证解决过拟合。1K模型容量相对数据仍大。新的80/20结果必须与旧70/10/20分开标注；和论文比较仍需核对划分、种子汇总及PLCC计算协议。

如果从始至终都不想设验证集，则训练轮数、超参数和结构必须事先固定（例如采用外部既定训练方案），不能每轮看测试指标再选最好epoch。当前入口是验证选模入口，不支持直接val_ratio=0；本次增加显式拒绝，防止空验证指标导致选第一轮。尚未实现固定预算80/20模式，现有划分未改变。

## 原服务器只读数据审计

新增scripts/audit_downstream_data.py：不加载模型、不训练，只根据实际config_snapshot核对索引和路径重复、文件存在、原始标签范围与归一化参数、Prompt重合及样例映射，保存划分哈希。Prompt重合是诊断，不自动当成泄漏；不同文件内容相同不在路径审计能力内。

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
for dataset in agiqa1k agiqa3k aigciqa2023; do
  selection=""
  [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
  name="${dataset}_quality_cliplarge_layer3_dualquality_residual${selection}_shared_pi_pt_bnfrozen_seed42"
  /home/sll/anaconda3/envs/WZX-AI/bin/python scripts/audit_downstream_data.py \
    --config "outputs/$name/config_snapshot.yaml" \
    --output-json "outputs/$name/data_audit.json" || break
done
```

仅产生审计JSON，不修改图片、标签或划分。若路径/数量/字段错误，会报错而不生成“通过”的结论。当前云环境没有原数据，因此此步骤需要在原服务器运行。

## 验证与交付范围

170项CPU回归测试通过。新增覆盖数据字段/数量/有限性、指标无效输入及单样本误差、空验证拒绝、FP16零投影/SimSiam有限性与普通向量一致性、显式冻结视觉编码器的BN/Dropout模式、Align Adapter/U不兼容迁移拒绝，以及只读数据审计划分哈希复现/重复图像引用检测。旧测试中的无验证占位loader已更新为非空占位，不改变其权重初始化检查目的。

尚未运行真实数据/GPU实验；没有开启任何新的正则化、激活切换或80/20训练。相关改动必须在后续独立版本/实验中预先固定并记录。


后续实现：修正协议及固定预算重训已在 [protocol_v2_and_fixed_budget_refit.md](protocol_v2_and_fixed_budget_refit.md) 提供独立配置、执行脚本和验证范围。旧实验保持原协议。

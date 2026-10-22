# 可编辑框架图：逐模块说明与逐线核对

审查对象：`paper_framework_editable.pptx` 第1页，修正后为84条逻辑连线。一条折线的多个物理线段按 E###.1、E###.2 命名；PowerPoint“开始→选择→选择窗格”可查编号。第2页是带已知错误的原PNG参考，不作为正确架构图。

## 核对结论

旧PPT并非全部正确：局部质量存在重复连线，部分依赖漏画；Fusion独立local_proj未明确；门控v/u条件输入缺失；Quality Adapter三层FC误标成三个MLP。新版已修正。主图属于模块级概览：包含显式命名端口和框内复合计算，不等于每个PyTorch算子都画成独立节点。未把一个模型前向都计算到的所有诊断路径画出来。

- E033仅为阶段分隔线，不是迁移或梯度。
- E065紫色虚线是辅助损失前向；不是跨阶段迁移。
- PQ→QA迁移写在说明框中，不是一条从预训练实时流到下游的前向边。
- 共享V/R/F是多次调用同一组参数，不是把三个图像相加后编码。
- 相同z/P0/u/v符号是命名复用端口；全局z已经融合了局部信息，不能解释为纯CLIP向量。
- AP值向量进入pooling、u进入local pair、sg目标、softmax、sigmoid、BN/激活、Fusion残差/LN/FFN等部分操作采用框内公式/名称概括。完整计算见下文，不以缺少独立外部箭头推断不存在。
- 不声称视觉热图是实际模型注意力：图中格子为示意，无真实样本分析含义。

## 每个模块与实际计算

共享视觉路径：CLIP ViT-L/14全局输出768维，经Fusion.semantic_proj变为256维；ResNet layer3原始1024×14×14，经ResNet.proj变256×14×14。Fusion有另一套独立的256→256局部1×1投影，展平为196个token，再由全局query做MHA，依次残差+LayerNorm、FFN、残差+LayerNorm，输出256维z。MHA仍有内部Q/K/V权重。预训练源checkpoint取layer4，原始2048×7×7；下游取layer3且重置其1024→256投影。

第三套1×1投影L为local_patch_proj，直接消费ResNet.proj输出，而不消费Fusion.local_proj输出。L的权重供Q与Align共用，输出P0∈R^(196×256)。196是特征图位置数，不是196张裁剪原图，也不是196个CLIP全局向量。

质量全局：QA从预训练PQ复制初始化，单个MLP为256→512→512→256，含BN/ReLU/Dropout；输出归一化为a。r=z+α sg(||z||) normalize(a)，α可学习初值0.1；只停止norm的梯度，不停止z或a整条路径。Global head为256→256→128→1，最后sigmoid得到Qg。

质量局部：图像产生每位置6维NSS-like描述n（MSCN均值、二阶矩、绝对值均值、梯度、alpha代理、sigma代理）；不是完整GGD拟合，二阶矩与sigma代理当前重复。N为6→256→256 MLP，Pq=P0+N(n)。Patch head为256→256→1及sigmoid，生成s_i。Gate为262→256→1，对[Pq_i,n_i]输出logit，经patch维softmax得到w_i。Ql=sum_i w_i*s_i。Q=(1−β)Qg+βQl，β=sigmoid(一个标量)，不是逐图门控。文本门控、额外attention、distribution分支均未纳入当前主图。

五档CE：a经独立image_to_text映射256→768，和5个归一化文本参考算相似度/温度，软标签CE权重0.005。五档词terrible/poor/fair/good/excellent；用5组独立class contexts，不与真实Prompt的共享contexts混为一谈。冻结的是文本骨干，context仍可学习。blend=0表示不将五档期望分数直接混入Q，不表示CE关闭。

对齐：Prompt+共享context经冻结文本骨干产生768维特征，U线性映射到256维，再经AT得到u。AG(z)=v、AP(P0)=p_i；AG/AT/AP是三个独立ResidualAdapter（LayerNorm→Linear→GELU→Dropout→Linear，加0.1倍残差），不是同一个Adapter。

全局Pair head吃[v,u,abs(v−u)]（768维），MLP768→256→128→1+sigmoid得到Ag。局部匹配以normalize(XI(p_i))与normalize(XT(u))点积，patch维softmax生成t_i；e=sum_i t_i*p_i，值向量是AP(P0)，不是XI的归一化结果。局部Pair head吃[e,u,abs(e−u)]得到Al。全局/局部Pair head结构相同，权重独立。Gate吃[v,u,Ag,Al]共514维，经MLP514→256→1+sigmoid得到逐样本γ；A=γAg+(1−γ)Al。

预训练：SafeAug1/2与Clean独立前向通过同一组视觉权重。q1/q2/qc=normalize(PQ(z1/z2/zc))；p1/p2=normalize(DQ(q1/q2))。I-I对称损失比较p1与sg(q2)、p2与sg(q1)。Anchor/Ordinal和五档CE使用qc，不使用DQ输出。质量、Artifact/Aesthetics回归头不直接吃PQ输出。图文关系另用PI/PT及DI/DT，PQ/PI/PT相互独立。DI、DT、DQ为256→128→256；PQ/PI/PT为256→512→512→256。

真实文本路由并不完全相同：主Prompt先T+C→U→AT→XT→normalize→PT；Answer为T+C→U→PT；辅助短语为T+C→U→AT→PT。Clean全局图像为AG(zc)→PI，区域关系先L/AP/XI归一化及区域pool，再AG/PI。图中E026/E029和关系目标框是这几类路径的概括，不是省掉这些层的实现。

迁移：PQ→QA，骨干兼容权重、上下文、U、AG/AT/AP、Quality bank复制初始化。PI/PT与各predictor不迁移；下游评分头、L/N/NSSgate、XI/XT、Align gate重新初始化，局部layer3投影也重新初始化。冻结基础文本骨干从对应CLIP来源初始化，不等于把预训练checkpoint的全部文本骨干权重恢复过来。迁移是训练阶段间的初始化，不是实时计算线。

代码仍计算legacy s_final，但当前two-output-only不监督它；图只保留Q与A，不代表新增Final MOS或Authenticity头。AGIQA-1K没有Align标签，图是统一模型结构，不能据此推断该数据集监督了所有分支。

## 逐线清单

| 编号 | 含义与审查结论 |
|---|---|
| E001 | SafeAug1 → 共享视觉编码器（单独前向，不与另两张图混合）。已按代码核对。 |
| E002 | SafeAug2 → 同一个视觉编码器（第二次独立前向）。已按代码核对。 |
| E003 | Clean → 同一个视觉编码器（用于有标签监督）。已按代码核对。 |
| E004 | 预训练 CLIP → 全局特征 g；768维。已按代码核对。 |
| E005 | 全局特征 → Fusion，包含其 semantic_proj。已按代码核对。 |
| E006 | 预训练 ResNet layer4 → 原始2048通道局部图。已按代码核对。 |
| E007 | 局部图 → Fusion；该概括路径包含 ResNet.proj 2048→256，与 Fusion.local_proj 独立。已按代码核对。 |
| E008 | Fusion → 各视图对应的 z；不是三张图共用一个数值张量。已按代码核对。 |
| E009 | z1 → PQ（同一个质量投影器）。已按代码核对。 |
| E010 | PQ → q1（含L2归一化）。已按代码核对。 |
| E011 | q1 → DQ。已按代码核对。 |
| E012 | DQ → p1（含L2归一化）。已按代码核对。 |
| E013 | z2 → 同一个PQ。已按代码核对。 |
| E014 | PQ → q2（含L2归一化）。已按代码核对。 |
| E015 | q2 → 同一个DQ。已按代码核对。 |
| E016 | DQ → p2（含L2归一化）。已按代码核对。 |
| E017 | zc → 同一个PQ。已按代码核对。 |
| E018 | PQ → qc（含L2归一化）。已按代码核对。 |
| E019 | qc → Component Anchor/Ordinal 与五档CE；不经过DQ。已按代码核对。 |
| E020 | 第一视图 z1 的命名路由；不是同一个z复制给全部视图。已按代码核对。 |
| E021 | 第二视图 z2 的命名路由。已按代码核对。 |
| E022 | Clean视图 zc 的命名路由。已按代码核对。 |
| E023 | p1 → SimSiam损失；目标为sg(q2)，目标接入在损失框公式中表示。已按代码核对。 |
| E024 | p2 → SimSiam损失；目标为sg(q1)，目标接入在损失框公式中表示。已按代码核对。 |
| E025 | Clean共享特征 → 独立评分路径；质量/一致性及Artifact/Aesthetics监督，不是PQ输出。已按代码核对。 |
| E026 | Clean全局/区域对齐特征 → PI；前置AG或区域池化按不同关系走相应路径。已按代码核对。 |
| E027 | PI → DI；含投影输出归一化。已按代码核对。 |
| E028 | 带可学习context的文本tokens → 冻结文本编码器。已按代码核对。 |
| E029 | 文本编码器 → U及关系特定适配路径 → PT；不是绕过U直接进PT。已按代码核对。 |
| E030 | PT → DT；含投影输出归一化。已按代码核对。 |
| E031 | 图像预测/目标表示 → 图文关系目标（图中合并表示）。已按代码核对。 |
| E032 | 文本预测/目标表示 → 图文关系目标（图中合并表示）。已按代码核对。 |
| E033 | 阶段分隔线，无数据流或梯度含义。已按代码核对。 |
| E034 | 下游原图 → CLIP视觉编码器。已按代码核对。 |
| E035 | 同一原图 → ResNet-50 layer3。已按代码核对。 |
| E036 | CLIP全局向量768维 → Fusion.semantic_proj（768→256）。已按代码核对。 |
| E037 | ResNet layer3原始1024通道图 → ResNet.proj（1024→256）。已按代码核对。 |
| E038 | Fusion的global query → 注意力。已按代码核对。 |
| E039 | K → 注意力；K来自Fusion.local_proj输出，MHA内部仍有K线性变换。已按代码核对。 |
| E040 | V → 注意力；V来自同一局部特征，MHA内部仍有V线性变换。已按代码核对。 |
| E041 | semantic_proj后的256维全局特征 → Fusion query。已按代码核对。 |
| E042 | ResNet.proj后的局部图 → Fusion独立local_proj（256→256）。已按代码核对。 |
| E043 | Fusion.local_proj输出 → 局部K表示。已按代码核对。 |
| E044 | 同一Fusion.local_proj输出 → 局部V表示。已按代码核对。 |
| E045 | ResNet.proj输出 → 独立且Q/Align共享的local_patch_proj L；不是Fusion.local_proj输出。已按代码核对。 |
| E046 | z命名端口 → Quality Adapter。已按代码核对。 |
| E047 | Quality Adapter第一FC → 第二FC；BN/ReLU/Dropout在块内概括。已按代码核对。 |
| E048 | Quality Adapter第二FC → 第三FC；整个Adapter是一个MLP。已按代码核对。 |
| E049 | Quality Adapter输出 → a；按当前配置L2归一化。已按代码核对。 |
| E050 | a → 残差加法；缩放alpha和stopgrad(norm(z))由公式表示。已按代码核对。 |
| E051 | z旁路 → 同一残差加法。已按代码核对。 |
| E052 | r → 全局质量头；不是a单独作为主评分输入。已按代码核对。 |
| E053 | 全局质量头 → sigmoid后的Qg。已按代码核对。 |
| E054 | n=NSS(image) → N(n)并加到P0；N在公式块内表示。已按代码核对。 |
| E055 | P0命名端口 → Pq=P0+N(n)。已按代码核对。 |
| E056 | Pq → patch质量头，sigmoid输出各patch分数。已按代码核对。 |
| E057 | Pq → NSS gate。已按代码核对。 |
| E058 | 原始n → 同一个NSS gate，与Pq拼接；不经过N(n)后替代原始n。已按代码核对。 |
| E059 | 各patch分数 → 加权和Σ（与权重相乘再求和，不是普通相加）。已按代码核对。 |
| E060 | NSS gate经softmax的权重 → 同一个加权和Σ。已按代码核对。 |
| E061 | 加权和 → Ql。已按代码核对。 |
| E062 | Qg → 质量标量融合。已按代码核对。 |
| E063 | Ql → 同一个质量标量融合。已按代码核对。 |
| E064 | β融合 → 最终质量Q；β对所有样本共享。已按代码核对。 |
| E065 | a → 五档文本CE辅助分支；实际含image_to_text及独立文本参考，此虚线不是参数迁移。已按代码核对。 |
| E066 | z命名端口 → 图像对齐Adapter AG。已按代码核对。 |
| E067 | AG → 全局对齐图像特征v。已按代码核对。 |
| E068 | v → 全局Pair Head。已按代码核对。 |
| E069 | Prompt与可学习context → token表示（示意token化/embedding）。已按代码核对。 |
| E070 | 这些tokens → 冻结CLIP文本编码器。已按代码核对。 |
| E071 | 文本编码器768维输出 → U（768→256）。已按代码核对。 |
| E072 | U输出 → 独立文本对齐Adapter AT。已按代码核对。 |
| E073 | AT → 文本对齐特征u。已按代码核对。 |
| E074 | u命名端口 → 全局Pair Head，与v及绝对差拼接。已按代码核对。 |
| E075 | AP(P0) → XI并归一化后参与相似度计算；XI在分组块中表示。已按代码核对。 |
| E076 | u → XT并归一化后参与相似度计算；不是把文本变成图像patch。已按代码核对。 |
| E077 | 相似度经patch维softmax → Attention pooling权重。已按代码核对。 |
| E078 | Attention pooling结果e → Local Pair Head，另接u并拼绝对差（在头注释内表示）。已按代码核对。 |
| E079 | P0命名端口 → 局部匹配分组内部AP。已按代码核对。 |
| E080 | 同一个u → 局部匹配文本输入端口。已按代码核对。 |
| E081 | v、u命名端口 → γ门控网络；与Ag、Al共同作为输入。已按代码核对。 |
| E082 | 全局一致性Ag → 样本门控及融合。已按代码核对。 |
| E083 | 局部一致性Al → 同一个样本门控及融合。已按代码核对。 |
| E084 | γ融合 → 最终一致性A。已按代码核对。 |

## 验证范围与依据

`framework_route_checks.json`记录17项下游/实际Fusion CPU张量一致性检查及预训练共享投影检查。检查真实model heads实际输入/输出；编码器用小模型，Fusion单独使用真实实现。此验证不是训练实验，不证明模块有性能贡献，也不声称检查了PowerPoint客户端的每种渲染行为。

- `models/two_stage_models.py`: `_quality_from_patches`、`_quality_residual_feature`、`_align_from_global_and_patch`、`_forward_three_views`、`load_pretrained_backbone`。
- `models/fusion.py`: `SemanticStructureFusion.forward`。
- `models/backbones.py`: `ResNet50LocalEncoder`。
- 当前实验CLI覆盖YAML默认值，说明依据为Large/layer3/residual_feature/CE0.005/Quality标量融合，不按YAML未覆盖的Base默认值解释。

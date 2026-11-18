# Component Anchors + Ordinal：无示例最终图

当前最终图：Component_Anchors_Ordinal_Symbolic.png（1774×887）。仅交付渲染PNG，不制作PPT。此图是栅格图，不声称矢量或可编辑对象。

按用户修订，图中不含示例概率柱、特定软目标模式或样本分数，统一以A_c、p_c、t_c、r_c、u_c表示锚点及分布。K=5和256-D均为实际结构参数，不是举例。此前带示意分布的Component_Anchors_Ordinal_Final.png保留为历史版本，不再作为当前交付。

同一256维质量特征q复用到Artifact、Aesthetics两路。两路独立固定五级锚点，经余弦相似度与Softmax产生各自的五档概率p_c；同一人工分量分数插值得到软目标t_c。Anchor CE比较p_c与t_c。尾部累加分别产生r_c和u_c，两者共同进入Ordinal BCE。两种监督并行，不增加预测网络。

实线为前向计算，灰色点线为标签监督；固定锚点为非可训练常量，不是文本嵌入。温度和损失权重为简洁而省略，不代表未启用。图经实际渲染与目视核查，未更改源码或运行训练。

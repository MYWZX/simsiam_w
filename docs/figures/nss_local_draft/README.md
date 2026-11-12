# NSS局部质量分支草图

借鉴用户上传论文 Xia et al., Visual-Language Multi-Task Blind Image Quality Assessment With Local Quality Weighting (TMM2025), 图2“统计支路与深度特征支路分开”的布局。没有复制原图图像或照搬其GGD/分数乘权重方法，计算遵循本项目当前源码。

上半部分：图像还原像素、转灰度、MSCN-like局部归一化得到z；另外从灰度算梯度幅值。对z、z²、|z|及梯度幅值分别自适应平均池化到14×14。输入224×224时，每格平均16×16=256个数。例如先逐像素平方得到z²，再平均，不是把平均值平方。四个均值加比值mabs/sqrt(m2)与重复m2，拼成14×14×6，展平196×6。实际代码含epsilon保护。第6项确实重复第2项，草图明确注明，不把它画成独立统计或GGD拟合。

空间网格与ResNet token对齐，但不意味着ResNet单个token的感受野仅16×16像素。

下半部分：n通过6→256→256 MLP嵌入，与P0逐元素相加；QL残差Adapter、L2后得到r196×256。原始n与r拼接262维，经gate与跨196位置softmax得到w，再池化r、归一化得到qL256维，最后经局部质量头输出sL。sL还需与全局质量分数混合，不是最终整个模型输出。

n在两面板重复绘制，明确标记reuse，表示同一张量；不是重复计算的不同网络。原始统计和可学习MLP、加法与加权池化已区分。没有质量文本gate、跨注意力或融合z。

PPT原生可编辑、PDF矢量、PNG6000×3810，均从同一PPT导出；已实际查看渲染，无页外文字、替换字符或PDF栅格对象。可编辑公式是文本分组而非Office公式对象。

复现：安装requirements.txt、LibreOffice，在本目录运行 `python generate_draft.py`；依赖同目录generate_srm.py、drawing_primitives.py和fonts/。未改训练代码或实验结果。

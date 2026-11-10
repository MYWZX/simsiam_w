# SRM 直观流程版

将模块内部结构改为三个明确步骤：
1. 用 u 经过 Predictor 和 L2 归一化预测 v，与 sg(v) 计算负余弦距离。
2. 交换角色，用 v 预测 u，与 sg(u) 计算负余弦距离。
3. 对两个距离取等权平均，得到关系损失。

重复的 u、v 是同一对输入的复用，不是额外的编码过程。暖橙为停止梯度目标，青绿为预测器，淡紫为匹配损失。停止梯度只作用于目标路径，不等于冻结整个编码器。

四种调用的输入和 Predictor 配对独立列出。全项目仅复用一套图像 Predictor h_I、一套文本 Predictor h_T；不同模态不共用 Predictor 参数。h_m/h_n 按调用选择。两个 Predictor 结构均为 Linear256→128、BN、ReLU、Linear128→256，后接 L2 归一化。PI/PT 属于外部表征模块。

逐样本权重为 I-I:1、I-Prompt:a、I-Answer:c、Prompt-Answer:a*c，图中为易读省略。a 是人工一致性分数，c 是 CLIP 派生的权重而非正确率概率。实际加权批次归约及各关系外部损失系数仍在 SRM 外，不因画法变化而改变。当前关闭负关系，无拉远；局部区域及词级额外监督不属于这四个主调用。

符号：v1/v2 为双增强视图经 PI 后的特征；a^c 为 Clean 图像一致性表示；t_P 为 Prompt 经 PT、文本 adapter 和匹配投影后的表示；t_A 为 Answer 经 PT 后的表示。所有输入 256 维。未引入跨注意力或融合 z。

交付：原生可编辑 PPTX、由该 PPT 导出的矢量 PDF，以及由 PDF 渲染的 6000×3525 PNG。已实际查看最终渲染，未发现裁切和主要连线遮挡。所有图形、公式文本及连接线均可编辑，公式为分组文本而非 Office 公式对象；不存在栅格化框架对象。

复现：安装 requirements.txt 与 LibreOffice，运行 `python generate_srm.py --output-dir ./rendered`。随包提供字体、许可证及渲染审计记录。

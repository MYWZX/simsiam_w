# SRM 论文模块图：新版配色

文件：SRM_publication.pptx（原生可编辑），同源 PDF（矢量）与 PNG（6000×3750）。

青绿：可训练 Predictor 路径；暖橙：stop-gradient 目标路径；淡紫：负余弦距离与对称平均。sg 仅截断该目标路径的梯度，不代表编码器永久冻结。两个输入各自分叉为预测与目标；两个方向的距离等权平均。

四种主调用复用一套图像 Predictor h_I 和一套文本 Predictor h_T，均为 256→Linear128→BN→ReLU→Linear256，输出再 L2 归一化。PI/PT 位于 SRM 外。输入均为已投影的 256 维表示。

下方卡片的 w 是逐样本关系权重，不是外部损失系数：I-I 为 1；I-Prompt 为人工一致性 a；I-Answer 为 CLIP 派生的 Answer 权重 c；Prompt-Answer 为 a*c。c 不是正确率概率。加权批次归约在模块外执行。当前关闭负关系，仅正向匹配；区域及词级额外监督不包含在这四个调用中。

图中 v1/v2 是双增强视图经 PI 后的特征；a^c 为 Clean 图像一致性表示；t_P 为 Prompt 经 PT、文本 adapter 与匹配投影后的表示；t_A 为 Answer 经 PT 后的表示。没有跨注意力或融合 z。

生成：安装 requirements.txt 和 LibreOffice，运行 `python generate_srm.py --output-dir ./rendered`。字体随包附带。所有公式为分组原生文本对象，非 Office 公式编辑器对象；图中没有栅格化模块。PNG 由同一 PPT 导出的 PDF 渲染。render_audit.json 记录尺寸、原生对象数量、字体和 SHA256。已实际查看完整渲染，未发现截断、越界或主要连线遮挡。

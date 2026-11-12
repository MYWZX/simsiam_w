# SRM：图像预测 Prompt 的单向完整路径

只展示 I–P 对称损失中的第一个方向，不代表全部 SRM。基于 projection_only_v1 基线，CLIP 图像编码器可训练，CLIP 文本权重冻结。冻结图像编码器的对照实验中，前向路径一致，但不会更新其权重。

上排：Clean Image → CLIP 图像编码器 → L2 → 全局 Linear → PI + L2 → AG + L2 → 图像 Predictor + L2 → pu。
下排：Prompt 与共享的五个上下文向量 → 冻结 CLIP 文本编码器 + L2 → T → PT + L2 → AT → WT + L2 → v → sg(v)。
损失为 −cos(pu, sg(v))。只通过预测侧回传，目标侧的上下文、T、PT、AT、WT 不接收此项梯度；它们可以从反方向或其他目标获得梯度。五个上下文参数并非冻结，且由 Prompt、Answer 共享。

图中红色虚线概括梯度方向，并非跳过中间模块的直接连接。PI/PT 模块内部为 Linear/BN/ReLU 等组成的投影 MLP，图内维度为摘要。此图为逐样本单向项，完整训练另外包含反方向平均、样本对齐权重与关系损失系数。未画局部 P0、QA 和最终评分头，因为它们不在这一项的有效路径上。

PPT 中模块、文字和连接线均为原生可编辑对象，数学式为普通可编辑文字。PDF、6000×3000 PNG 来自同一 PPT，已实际渲染并目视检查。运行 python generate_path.py 可复现；需要 requirements.txt 所列库及 LibreOffice。字体与许可证位于 fonts/。

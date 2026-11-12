# 有序分量监督（Ordered Component Supervision）

本图对应 projection_only_v1 预训练的 Component Anchors + Ordinal 辅助监督，不改变模型或启动训练。

## 阅读方式

- 第一页为可放入 QAM 的模块详图：质量特征 q 分别与 Artifact、Aesthetics 各自的五个固定锚点比较，生成每个分量独立的五档概率。不是十分类。
- Anchor CE：匹配由人工分数插值得到的相邻档位软标签。
- Ordinal BCE：复用该五档概率，经固定累计变换得到四个 P(level > k)，与同一人工标签导出的四个阈值目标比较。不是新增 MLP 或独立分类头。
- 第二页演示 y=0.70 如何产生软标签与有序阈值目标。数字按源码公式计算后四舍五入，仅为示例，不是实验结果。
- 实线为预测计算，灰色点线为标签监督；青色与橙色区分两类监督，紫色表示累计概率变换。
- 锚点圆弧为 256 维固定向量的二维示意；两组锚点位于不同平面。它们是常量，不是文本嵌入，也不是可训练参数。
- 两项损失都反传至 q 及其上游全局/局部质量表示；仅辅助监督用于预训练，不代表上游冻结。关闭的质量文本五档 CE 与此处的 Component Anchor CE 不同。

源码依据：train_cm_pretrain.py 的 _compute_quality_component_anchor_ordinal_losses 及固定锚点、软标签构造；质量特征 q 的生成由现有双质量分支完成。本图不含跨注意力或 z。

## 文件与编辑

Component_Anchor_Ordinal.pptx 为两页原生可编辑主文件。模块、文字、概率柱、点、线和数学标注均为原生对象；数学标注使用可编辑字符组合，非公式编辑器对象。主要对象按功能分组并命名。
PDF 和两个 6000×3000 PNG 均由该 PPT 统一导出。第一页为模块图，第二页为解释页，后者建议用于方法说明或补充材料。文字较多的说明可在论文排版时保留于图注。

## 复现

安装 requirements.txt 中的 Python 库及 LibreOffice（soffice），在本目录运行：

    python generate_module.py

字体和许可证在 fonts/。生成器与 drawing_primitives.py、generate_srm.py 放在同一目录。render_audit.json 记录页面尺寸、矢量检查与出界检查。已实际渲染并目视检查两页。

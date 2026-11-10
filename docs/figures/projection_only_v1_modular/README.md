# projection_only_v1 模块化论文框架图

本版按用户反馈改为 **一张总图＋两张独立模块展开图**，不再把所有底层细节塞入总图。三个图均有独立原生可编辑 PPTX、同源高清 PNG 和矢量 PDF；另提供三页合并 PDF。

## 1. 模块命名

**DAC — Dual-Branch Assessment Core（双分支评估核心）**

将两个阶段重复出现的视觉/文本编码、全局/局部投影、Quality 与 Alignment 两条分支封装为统一的结构模块。这是图稿层面的封装命名，不是添加新网络或修改 forward。

- 总图中的两个 DAC 形状、尺寸和内部图标一致。
- 预训练的 View1 / View2 / Clean 调用同一阶段的共享表示参数。只有 Clean 执行完整评分分支并承接人工标签监督；增强视图的关系损失使用图像特征路径。
- 下游使用相同结构，但表示参数通过初始化迁移，随后独立微调；不是跨阶段持续绑定共享权重。
- PI/PT 是 DAC 内持久化的特征投影器，预训练与下游均会使用它们，不画成评分头，也不在 SRM 内重复创建投影器。
- DAC 详细展开图保留真实 Quality / Alignment 计算，无跨注意力、无融合 z。Quality 全局 g→QA，I-I 与图像语义关系 g→PI，P₀ 绕过 PI。

**SRM — SimSiam Relation Module（SimSiam关系模块）**

将 Predictor、stop-gradient、双向负余弦损失及平均封装为训练关系模块。对已投影的特征 u/v，内部执行：

`L = 1/2 [D(h_m(u), sg(v)) + D(h_n(v), sg(u))]`

每个 D 使用归一化余弦相似度。按调用处的置信度/有效样本掩码加权，再进行批内归约；主图省略这些数值权重，以突出拓扑。

## 2. 四个主要 SRM 的输入

| 主图调用 | 输入 u | 输入 v | 两侧 Predictor |
|---|---|---|---|
| SRM 1：I–I | v₁ | v₂ | h_I / h_I |
| SRM 2：I–Prompt | aᶜ | tP | h_I / h_T |
| SRM 3：I–Answer | aᶜ | tA | h_I / h_T |
| SRM 4：Prompt–Answer | tP | tA | h_T / h_T |

**这是同一次预训练中的四项主要关系损失，不是四个独立模型，也不是四次预训练实验。**

所有图像侧调用复用 h_I，所有文本侧调用复用 h_T；h_I 与 h_T 彼此不是同一套权重。每个 Predictor 为 256→128→256，只用于预训练，不迁移为下游评分头。

实际特征定义：

- `vᵢ = norm(PI(gᵢ))`：增强视图特征，不经过 AG。
- `aᶜ = norm(AG(norm(PI(g_clean))))`：Clean 图像语义关系特征。
- `tP = norm(W_T(AT(norm(PT(T(E_T(P)))))))`：Prompt 关系特征，复用 Clean Align 文本特征。
- `tA = norm(PT(T(E_T(A))))`：Answer 关系特征，不额外经过 AT/W_T。
- `q = norm[(1−ρ)qA + ρqL]`：分量锚点/Ordinal监督的质量表示。

总图从同一个 aᶜ、tP、tA 特征节点实线分叉，表达张量复用；交叉处有留白、没有分叉点的线不连接。

## 3. 没有被四个 SRM 替代的目标

Clean View 的 Quality / Alignment 回归与排序、Artifact/Aesthetics 辅助监督、Component Anchors/Ordinal 均保留。正向局部/词级关系也保留，在总图监督区简记；这些局部调用仍使用相应共享 Predictor 家族。

预训练的质量任务/一致性任务继续使用 PCGrad。没有打开 negative 推离或质量文本五档 CE。不能把“4 个主要关系”理解为只有 4 项总损失。

## 4. 迁移关系

紫色虚线只表示表示参数初始化。迁移包括视觉编码器、全局和局部投影、5个上下文词元、T、PI/PT、QA、局部质量残差编码、NSS投影/门控、Align三个残差Adapter、α、ρ。

下游重置数据集评分头、Align匹配 Linear W_I/W_T、逐样本 Align 分数门控，以及 Quality 分数混合 β。固定 CLIP 文本塔由同一原始来源载入，其原始权重冻结；上下文词元可训练。不要把两个阶段的完整 DAC 全部参数理解为无差别迁移。

## 5. 文件与编辑

- `01_modular_framework.pptx/png/pdf`：完整模块化总图。
- `02_DAC_module.pptx/png/pdf`：DAC 内部结构与特征端口定义。
- `03_SRM_module.pptx/png/pdf`：SRM 内部结构与四类调用表。
- `all_modular_figures.pdf`：以上三图的三页矢量 PDF。
- `generate_modular.py`、`drawing_primitives.py`：可复现脚本。
- `assets/source_DAC_native.pptx`：DAC 原生形状底稿；生成器复制可编辑对象，不栅格化。
- `fonts/`：DejaVu Sans 字体及许可。编辑前安装可避免字体替换。
- `render_audit.json`：尺寸、原生对象计数与文件指纹。

文字、模块、箭头、token、锁与输入示意均为原生可编辑对象。公式是可编辑文字，不是图片。每页有命名分组，主要模块/连线可单独选择和调整。

实线表示前向数据流；灰色点线表示训练监督；紫色虚线表示参数初始化。SRM 的重复名称/编号表示模块调用，Predictor 共享由 h_I/h_T 标识；DAC 架构复用和跨阶段迁移是不同关系。

## 6. 复现

安装 python-pptx、PyMuPDF、Pillow 及 LibreOffice，并在机器上安装附带字体：

```bash
python -m pip install -r requirements.txt
python generate_modular.py --output-dir ./rendered
```

每份 PNG 均从对应 PPTX 经 LibreOffice PDF 导出后渲染，不独立重画。代码核对依据：MYWZX/simsiam_w 的 projection_only_v1 forward 与当前配置。未修改训练代码、未启动训练、未覆盖旧版图稿或实验结果。

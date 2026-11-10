# projection_only_v1 论文框架图：说明与复现

## 文件

- `projection_only_v1_framework.pptx`：一页、26 × 14.65 英寸横向自定义画布。所有视觉元素均为 PowerPoint 原生可编辑对象，没有以整张图片充当框架。主要分组为 `Panel_A_Pretraining`、`Panel_B_Downstream_Assessment`、`Legend_and_Transfer`，内部包含命名的 Module、Feature、Operator、Connector 子分组。
- `projection_only_v1_framework.png`：5600 × 3156 像素，完整主图。
- `projection_only_v1_framework.pdf`：与 PPT 同源的一页 PDF，字体和线条保留为矢量。
- `generate_figure.py`：完整、自包含的生成脚本。
- `render_audit.json`：尺寸、原生对象计数、字体、来源哈希及导出检查。
- `source_verification.json`：源码与配置核对记录。
- `fonts/`：DejaVu Sans 字体与许可证。若编辑电脑未安装该字体，请先安装，避免 PowerPoint 自动替换字体。

PNG 的导出链为 **PPTX → LibreOffice PDF → PyMuPDF PNG**，不是另外绘制的独立 PNG。PNG/PDF/PPT 采用相同布局。无需打开网络服务。

## 内容依据

以本次对话中用户认可的最新框架说明为事实依据，核对 `MYWZX/simsiam_w` 的实际代码与配置。参考上传的 DPGF-Net 论文及截图的浅色分区、视觉层级与模块化表达；未移植 TCPGA、FIM 或其余论文特有结构。

主图对应当前 `projection_only_v1`。基线与新开的 I-I 权重实验结构相同；本图不把尚未获得结果的新实验标记成最好基线，也没有加入旧跨注意力或融合输出 z。

## 读图方式

- **实线箭头**：特征、输入或分数的前向数据流。
- **灰色点线箭头**：训练阶段监督或损失输入。
- **紫色虚线箭头**：跨阶段表示参数初始化，既不是同一张量的复用，也不是跨阶段绑定共享参数。
- **相同模块 ID**：同一阶段重复调用相同参数。例如预训练多个 V/G/PI/Pred I 具有 shared weights。两个面板之间只通过紫色迁移标记关联，不表示之后训练仍绑定权重。
- **实线分叉点**：复用同一个特征。交叉处留白且没有连接点表示不连接。
- **圆圈 `+`**：残差或逐元素相加；图中从未将 g 与 P₀ 相加。
- **圆圈 `C`**：配对拼接 `[u,t,|u−t|]`。
- **`P` / Text pool**：局部 token 加权池化。Quality 由 NSS 门控产生位置权重；Alignment 使用文本引导位置相似度产生权重。
- **圆圈 `M` / Gate γ 中的 M**：标量分数加权混合。
- **`sg`**：stop-gradient。I-I 为对称跨视图 stop-gradient；Quality 全局残差只对 g 的范数估计 stop-gradient，不截断整个 g 或 QA 分支。
- **锁**：CLIP 文本塔原始权重冻结。5 个上下文词元单独绘制、可训练，梯度仍可通过冻结的文本塔传播到这些词元。

## 模块与实际路径

- `V`：CLIP ViT-L/14 全局图像编码器，输出 768 维（已含其 CLIP 输出投影），QuickGELU。
- `R`：ResNet-50 layer3 局部图像编码。
- `G`：`backbone.fusion.semantic_proj`，Linear 768→256，生成 g。虽沿用 fusion 命名及优化器归属，但不是跨注意力。
- `L`：局部编码器内部 1024→256 的 1×1 卷积，以及独立 `local_patch_proj` 的 256→256 1×1 卷积。224×224 输入下得到 196×256 的 P₀。为避免在主图堆叠两个维度转换框，以 L 分组表示两个连续局部投影，并在图中注明 two 1×1 projections。
- `Eₜ`：冻结的 CLIP 文本塔，另有 5 个可学习上下文词元。
- `T`：`text_to_fused`，Linear 768→256。
- `PI` / `PT`：独立图像/文本特征投影器，非评分头；MLP 为 256→512→512→256，含 BN/ReLU/Dropout 与末端非仿射 BN；图中 norm 为输出 L2 归一化。
- `QA`：质量 MLP 256→512→512→256，输出归一化为 qA。全局质量评分特征为 `g + α·sg(‖g‖)·qA`，不经过 PI，相加后不再整体归一化。
- `QL`：局部质量残差 Adapter，`x + 0.1·MLP(LN(x))`，之后 L2 归一化。输入为 P₀ 加 NSS 的 MLP 投影。
- NSS：每位置 6 维轻量 MSCN/灰度梯度描述，不是完整 GGD 拟合。描述分为两路，一路 MLP 后加入 P₀，另一路直接进入局部质量池化门控。
- `AG/AP/AT`：全局图像、局部图像、文本各自的残差 Adapter。三者不共享彼此权重。
- `H_QG/H_QL`：全局/局部 Quality 评分头，均含 sigmoid，输出标量。Quality 的 β 是可学习的全局标量。
- `H_AG/H_AP`：全局/局部 Alignment 评分头，均含 sigmoid。Alignment 的 γ 为以全局图像特征、文本特征及两个分数为输入的逐样本门控。
- Alignment 的 Text pool 包含独立的匹配 Linear W_I/W_T、L2 归一化、点积 softmax 和加权池化；这里没有多头跨注意力。

## 预训练面板的展开规则

Clean View 中的 Quality/Alignment 模块在面板 (b) 展开；两种阶段采用同样的计算结构，但后续参数通过迁移初始化，不跨阶段绑定。因而不会把面板 (a) 的整个分支直接无条件导入下游。

- 两个 SafeAug 图像视图只为 I-I 自监督提供视图不变性。实际路径 g→PI→Pred I；PI 的投影输出为 stop-gradient 目标。
- 人工 MOS/Align、Artifact/Aesthetics 与 Component Anchors/Ordinal 监督 Clean View。
- Clean 的语义图像关系路径为 g→PI→AG→norm，之后使用 Pred I。
- Prompt 关系复用 Align 文本特征，实际为 `PT→norm→AT→W_T→norm`。
- Answer 关系为 `PT→norm`，然后与 Prompt 共用文本 Predictor；图中 † 专门说明两条路径的区别。
- 图中 I–Prompt / I–Answer / Prompt–Answer 关系区包含对应的预训练 Predictors；为避免主图拥挤，没有将每一次语义关系调用重复展开。共享与 stop-gradient 规则由该区和 I-I 示意共同说明。
- RichHF 正向局部/词级关系使用注释和相应特征进行附加训练，集中在图下注明，未展开为密集旁路；negative 推离关闭。
- `q=norm[(1−ρ)qA+ρqL]` 是 Component Anchors/Ordinal 约束的表示。固定分量锚点在不同正交平面中构造五个等级，Anchor 软标签 CE 与累积 Ordinal BCE 仍开启，和关闭的“质量文本五档 CE”不是一项损失。
- `H_art/H_aes` 是预训练分量辅助回归头，输入 `[qA,qL,|qA−qL|]`，不是把最终 Quality 分数改成分量锚点评分。

## 参数迁移的精确范围

紫色 `θᵣ` 包括：V/R、G/L、5 个上下文词元、T、PI/PT、QA、QL、NSS 投影/门控、AG/AP/AT、α 与质量表示混合参数 ρ。

下游重新初始化：四个数据集评分头 H_QG/H_QL/H_AG/H_AP、Alignment 的匹配 Linear W_I/W_T、逐样本分数门控 γ，以及 Quality 分数混合参数 β（从0.5开始）。预训练辅助分量评分头和 SimSiam Predictors 不作为下游评分模块迁移。冻结文本塔的原始权重从相同 CLIP 来源载入，不是通过预训练学到再迁移的参数。

图中表示迁移并不表示所有迁移参数在每个下游都有非零梯度。例如下游未启用分量锚点监督时，ρ 不直接参与 Quality 主评分。

主图只报告独立 Quality / Alignment 输出；没有第三个有效 Final 任务。1K 下游只有 Quality 标签，3K/2023 使用两个任务。

## 可编辑性与复现

所有模块、文字、线条、token、锁、公式和输入示意均可编辑；没有位图对象。公式使用普通可编辑文本及原生上下标属性，不是专用 Office 公式对象。折线由成组的原生连接器线段组成；能匹配模块端口的位置已设置原生连接锚点，其他拐点可单独调整。

使用 DejaVu Sans 以覆盖希腊字母和上下标。PDF 已嵌入字体；PPT 在未安装该字体的机器上仍可能发生替换，因此附字体与许可。脚本依赖 python-pptx、PyMuPDF、Pillow，以及系统 LibreOffice：

```bash
python -m pip install python-pptx PyMuPDF Pillow
python generate_figure.py --output-dir ./rendered
```

建议论文排版使用同源 PDF；PNG 为完整高清预览。已实际检查完整渲染及关键区域，校验无漏模块、无非预期文本换行、输出未裁切、字体来自 DejaVu 家族。由于没有执行真实模型训练，本交付只改变图稿，不改变训练代码或结果。

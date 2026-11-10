# SRM 独立论文成图

本版为可直接排入论文的 SRM 独立精修图。不是布局建议或草图；已从原生可编辑 PPTX 实际导出 PDF 和高清 PNG。

## 文件

- `SRM_publication.png`：长边6000像素的完整高清渲染图。
- `SRM_publication.pdf`：同源单页矢量PDF，建议论文优先使用。
- `SRM_publication.pptx`：一页原生可编辑PPT，16 × 9.35英寸横向画布。
- `generate_srm.py`、`drawing_primitives.py`：完整生成脚本。
- `fonts/`：使用的DejaVu Sans/Serif字体及许可。编辑前安装可避免字体替换。
- `render_audit.json`：原生对象统计、像素尺寸、字体与文件哈希。

## 内容

SRM接收已投影、已归一化的256维特征u/v。每个输入分出可训练预测路径与stop-gradient目标路径：

`p_u = norm(h_m(u))`，`p_v = norm(h_n(v))`

`L_SRM = 1/2 [D(p_u,sg(v)) + D(p_v,sg(u))]`

D为负余弦。图中显示每对样本的对称损失；实际代码按调用处的置信度/有效样本权重做批内加权归约。sg不是冻结整个编码器，故不使用锁图标。

四个主要调用共享两套模态Predictor：

| 关系 | 输入(u,v) | Predictor(h_m,h_n) |
|---|---|---|
| I-I | v₁,v₂ | h_I,h_I |
| I-Prompt | aᶜ,t_P | h_I,h_T |
| I-Answer | aᶜ,t_A | h_I,h_T |
| Prompt-Answer | t_P,t_A | h_T,h_T |

这里t_P/t_A是对上一版tP/tA的规范下标排版，特征定义未变：vᵢ=norm(PI(gᵢ))；aᶜ=norm(AG(norm(PI(g_clean))))；t_P=norm(W_T(AT(norm(PT(T(E_T(P)))))))；t_A=norm(PT(T(E_T(A))))。

PI/PT属于可迁移的共享表示部分，不在SRM里重新创建。h_I/h_T为预训练Predictor（256→128→256），不作为下游评分头迁移。两侧h_m/h_n由输入模态选择，不能理解成四类关系各自有独立参数。

这次只精修图稿，不修改训练代码、损失、权重或基线结果。

## 排版与可编辑性

全部框、线、归一化节点和文字都采用PowerPoint原生对象，没有位图。公式使用分组的可编辑文字精确定位上下标，避免PowerPoint/LibreOffice对自动上下标缩放的差异；它们不是Office公式编辑器对象。主要公式和模块均有可识别的分组名称。

导出链：PPTX→LibreOffice PDF→PyMuPDF PNG。PDF文字和线条保留为矢量。图中各个关系共享同一符号定义，未新增negative关系、质量文本CE或其他历史模块。

建议论文导入矢量PDF；需要插入位图时使用完整6000像素PNG。无需裁去说明块才能使用，本图的公式和四类调用说明已统一排版。

## 复现

安装LibreOffice及随附字体，运行：

```bash
python -m pip install -r requirements.txt
python generate_srm.py --output-dir ./rendered
```

复现脚本不依赖仓库或其他旧版PPT素材。已检查完整实际渲染、无页面外文字和替换字符、无PDF位图对象。

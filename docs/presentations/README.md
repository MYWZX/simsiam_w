# 可编辑模型框架 PPT

- [下载 PPTX](https://raw.githubusercontent.com/MYWZX/simsiam_w/work/docs/presentations/paper_framework_editable.pptx)
- [PDF 预览](paper_framework_editable.pdf)
- [第 1 页图片预览](paper_framework_editable_preview.png)

共三页：

1. **原生可编辑重绘版**：文字、公式、模块、线条、特征张量和热图均由 PowerPoint 对象组成。不是一张 PNG 覆盖整个幻灯片。保留原图双面板、配色和张量/注意力图的视觉语言，同时重建关键连接。
2. **原始设计图参考**：整张 PNG，仅用于视觉比较；不具备图层可编辑性。原生成图的连接遗漏不作为实现依据。
3. **中文编辑说明**。

编辑方法：点击文字即可修改；选中箭头拖动端点；右键组合对象 → 取消组合，可以修改张量/热图中每个小格。图内 z / P₀ / u 为命名复用端口；保持同名特征对应关系。局部匹配是示意分组，具体 XI/XT 与 AP 的计算以技术文档为准。

重绘版并非原始 PNG 的逐像素分层，也不声称图像能无损转换为原生 PPT；它是独立的可编辑重建。公式以可编辑 Unicode 文本保存，非 Office 公式对象。连线为原生可编辑连接线，但不保证移动模块时所有端点自动吸附，请移动后检查。图片参考页可删除。建议在 PowerPoint 中导出 PDF 用于论文排版。

已检查：PPTX 压缩包完整，python-pptx 可重新读取，第一页面无整页图片，LibreOffice 能打开并导出三页 PDF，并检查第 1 页渲染结果。未在微软 PowerPoint 客户端测试，字体在不同设备上可能替换。

再生成：安装 `python-pptx` 与 `Pillow` 后，在仓库根目录运行 `python docs/presentations/build_editable_framework.py`。模型依据及迁移细节见 [技术说明](../figures/model_forward_and_sharing.md)。

## 逐模块与逐线核对

[完整讲解与84条逻辑连线清单](framework_line_audit.md)。连接线在选择窗格以E001.1、E001.2等命名；同编号不同后缀为同一折线的物理线段。E033仅为阶段分隔线。

新版修正了局部质量重复边/遗漏输入、Fusion独立局部投影、γ的v/u条件输入和Adapter的FC标注。主图仍是模块级概览，投影归一化、局部匹配值向量、目标端stop-gradient等由公式和配套说明展开。原PNG参考页保留已知问题，已加醒目标注。

实际CPU路径核对见 [结果](framework_route_checks.json)：17项下游/Fusion与7项预训练结构/张量检查。编码器用小模型，不是数据集/GPU训练验证。运行仓库根目录下的 `python docs/presentations/check_framework_routes.py`、`python docs/presentations/check_pretrain_routes.py` 可复核（使用项目依赖，两个脚本按顺序运行）。

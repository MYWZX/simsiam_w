# projection_only_v1：整套可编辑论文框架图

## 文件与页序

`ProjectionOnly_Modular_Framework.pptx`：一份 6 页原生可编辑 PPT，自定义 20×13.5 英寸横向画布。
`ProjectionOnly_Modular_Framework.pdf`：由这份 PPT 直接导出的同源矢量 PDF。
六张 PNG 均由上述 PDF 对应页渲染，均为 **6000×4050**：

1. `01_Framework.png`：最终总框架，预训练与下游上下分区。
2. `02_SRE.png`：共享表征编码模块。
3. `03_QAM.png`：质量模块。
4. `04_AAM.png`：一致性模块。
5. `05_TPM.png`：AAM 内文本引导局部匹配的独立展开图。
6. `06_SRM.png`：两方向匹配后取平均的 SimSiam 关系模块。

`Modules_overview.png` 仅为五张模块图的联系表预览；正式单图使用上列 6000px PNG。

## 事实依据与图形封装

SRE、QAM、AAM、TPM、SRM 是对现有计算的图形封装名称，不代表新增网络或新实验版本。依据当前 `work` 分支源码以及 projection_only_v1 基线配置核对，源码版本/哈希见 `source_audit.json`。上传参考图用于配色与层级参考；图中并未继承参考图的旧版本结构。

- **SRE**：全局 CLIP ViT-L/14 输出先归一化，再经原 `backbone.fusion.semantic_proj` Linear768→256 得 g。g 直接供 Quality，另经 PI+L2 得 v，供一致性/I-I。ResNet-50 layer3 后保留 `local_encoder.proj` 与 `local_patch_proj` 两个独立 1×1 投影，得 P₀（196×256）；P₀ 绕过 PI。文本为冻结 CLIP 文本编码器 + 5 个可训练上下文词元，文本输出先归一化，再经 T 和 PT。NSS n 直接来自图像的每 patch 六维轻量统计。
- **QAM**：qA=norm(QA(g))；全局评分输入为 g+α·sg(‖g‖)·qA。局部 P₀ 加 NSS 嵌入，经 QL 残差 adapter 与归一化；raw NSS 与局部 token 共同生成 softmax 权重，池化且归一化后得到 qL，再预测局部分数。Quality=(1−β)sG+βsL。训练 inset 明确复用上方 qA/qL；ρ 表征混合用于 Component Anchors/Ordinal，而非替代评分路径；auxiliary heads 用 [qA,qL,|qA−qL|]。
- **AAM**：AG、AP、AT 是三套独立残差 adapter。全局 [aG,aT,|aG−aT|] 与局部 [e,aT,|e−aT|] 各自经评分头得到标量。Gate 输入为 [aG,aT,sG,sP]，输出每图 γ；Alignment=γsG+(1−γ)sP。为了避免四路长线交叉，Gate 框内直接列明复用输入，旁注 reuse；这不表示额外编码。
- **TPM**：norm(W_I(U)) 与 norm(W_T(aT)) 点积，跨 patch softmax；权重池化原始适配 token U=AP(P₀)，而不是 W_I(U)。它是文本引导点积池化，不是已删除的跨注意力融合；没有融合 z。
- **SRM**：hI/hT 是不同网络，按模态在各调用之间共享；不是四套独立 Predictor，也不是图像和文本共用一套 Predictor。结构为 Linear256→128、BN、ReLU、Linear128→256，后接 L2。目标分支 sg，两个负余弦项等权平均。图中重复的输入是相同特征的角色交换，不是重新编码。

## 四个主关系与额外监督

I-I 使用 v1/v2（双增强视图，PI 后），hI/hI；I-Prompt 使用 aᶜ/tP，hI/hT；I-Answer 使用 aᶜ/tA，hI/hT；Prompt-Answer 使用 tP/tA，hT/hT。

aᶜ=norm(AG(norm(PI(gclean))))；Prompt tP=norm(W_T(AT(norm(PT(T(Etext(P)))))))；Answer tA=norm(PT(T(Etext(A))))，绕过 AAM。三者不是同一条投影路径。

图中 w=1、a、c、a·c 为逐样本权重，外部各关系损失系数及加权批次归约省略。a 是人工一致性分数；c 是 CLIP 分数按训练集分位端点归一化得到的伪监督权重，不是答案正确率概率。I-I 计入 Quality 任务，三个图文/文本关系计入 Alignment 任务，再用 PCGrad 处理任务梯度。

Clean-view 的人工质量/一致性、artifact/aesthetics、Component Anchors/Ordinal 仍启用。局部 misalignment_map 与词级标签的额外关系损失保留，但不伪装成四个全局 SRM 调用的共同标签。当前 disable_negative_relations=true，仅拉近；质量文本五档 CE 关闭，与 Component Anchor 监督不是一回事。

## 共享、复用与迁移的区别

- 实线箭头：数据前向；圆点：同一特征分叉。
- 灰色点线：训练监督关系，不表示参数共享。
- 紫色虚线：跨阶段参数初始化，不是数据流，也不是两阶段参数永久绑定。
- `shared θ` / shared predictors：同一阶段多次调用同一组参数。
- `reuse`：已有特征/分数再次使用，不新建网络。QAM 训练 inset 复用 qA/qL；AAM Gate 复用框内列出的四个输入。
- `+`：特征残差相加。`C`：拼接（SRE 中拼接 token；AAM 中拼接 x、t、绝对差）。Weighted pool：跨 token 加权求和。Score mix：标量分数混合。这些操作不等价。
- 锁仅在 CLIP 文本编码器处出现；5 个上下文词元可训练。sg 只截断目标路径梯度，不等于冻结整个编码器。

预训练到下游完整继承所需表示：视觉与局部投影、全局 semantic_proj、上下文/T、PI/PT、QA/QL、NSS MLP/gate、AG/AP/AT、α/ρ 等表示参数。冻结文本塔仍使用同一原始 CLIP 来源，未被训练成另一组文本权重。下游重置四个评分头、W_I/W_T、Alignment gate 与 Quality 分数混合 β；预训练辅助 component heads / SRM Predictors 不作为下游评分模块。具体键级迁移规则仍以源码加载器为准，图不替代加载校验。

## 编辑与复现

所有模块、文字、token、张量示意、图例、公式标签和连线为 PowerPoint 原生对象，合理分组并命名；不含整页 PNG 或栅格化模块。公式为原生分组文本，不是 Office 公式编辑器对象。输入图标与权重小热图均为可编辑的示意，不是实际实验样本或测量结果。

安装 `requirements.txt`，系统安装 LibreOffice；运行：

```bash
python generate_all.py --output-dir ./rendered
```

生成脚本依赖同目录 `generate_srm.py`、`drawing_primitives.py` 与 `fonts/`。字体及许可证随包。PNG 必须通过最终 PPT→PDF→PNG 链路统一导出。

## 实际核查

六页已实际渲染并逐页查看，修正了文字重叠和评分输出路径。`render_audit.json` 记录各页尺寸和原生对象数；PDF 无栅格图、无替换字符、无出界文字；PPT 各页对象 ID 唯一，连接器引用有效；输出 PNG 与同源 PDF 的逐像素渲染相同。未修改训练代码、未启动训练、未覆盖旧结果或旧图。

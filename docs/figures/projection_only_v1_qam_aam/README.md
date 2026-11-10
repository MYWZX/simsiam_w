# QAM / AAM 分离的 projection_only_v1 论文图

按最新反馈，已撤销总图中的 DAC 大模块，拆成两个独立任务模块，并各自提供独立展开图。原模块化版本保留在原目录，本版不覆盖它。

## 模块划分

- **QAM — Quality Assessment Module（质量评估模块）**：输入原始全局投影 g、独立局部 token P₀ 和每位置 NSS 描述 n；输出 Quality 分数。预训练时另导出质量表示和分量辅助预测，用于分量锚点/Ordinal及辅助回归/排序。
- **AAM — Alignment Assessment Module（一致性评估模块）**：输入已投影图像向量 v、P₀ 与已投影文本向量 t；输出 Alignment 分数，并提供语义关系的训练特征端口。
- **共享编码／投影部分**单独放在两模块之外，包含 V/R、G/L、冻结文本塔 Eₜ 与5个可训练上下文词元、T、PI/PT及原始NSS描述提取。图像编码器可训练；原始文本塔权重冻结；上下文词元和文本维度转换可训练。
- **SRM — SimSiam Relation Module** 保留四类主要调用，Predictor仍按图像/文本模态共享，不复制四套独立网络。

这只是图形封装边界调整，没有改变源码、参数、优化器或训练协议。

## 边界必须这样理解

1. `g = G(V(I))`，不是 PI(g)。QAM 接收 g，因此 Quality 主评分不经过 PI。
2. `P₀ = L(R(I))`；L 分组包括局部编码器内部 1×1 投影与独立 local_patch_proj。P₀在两个任务中复用，绕过PI。
3. `n` 是同位置 NSS 描述（N×6），不是NSS MLP输出。NSS MLP和NSS门控属于QAM。
4. `v = norm(PI(g))`。I-I用增强视图的v₁/v₂，直接送SRM，不经过QAM/AAM。
5. `t = norm(PT(T(Eₜ(P))))` 是AAM输入；总图中的`tP⁰`表示这个尚未经过AT/W_T的Prompt基础特征。
6. AAM的图像关系出口`aᶜ = norm(AG(vᶜ))`；Prompt出口`tP = norm(W_T(AT(tP⁰)))`。
7. Answer出口`tA = norm(PT(T(Eₜ(A))))`直接来自共享投影部分，绕过AAM。不能给Answer补画AT/W_T。
8. I-I被归入预训练质量任务的优化目标，不代表它在网络前向上经过QAM。多个目标仍会通过共享表示参数产生耦合梯度。

## QAM展开图

- 全局：`qA = norm(QA(g))`；`hG = g + α·sg(‖g‖)·qA`；`sG = sigmoid(H_QG(hG))`。
- 局部：`uᵢ = P₀ᵢ + MLP_NSS(nᵢ)`；`rᵢ = norm(QL(uᵢ))`，其中QL是固定残差尺度0.1的可训练残差Adapter。
- NSS门控按`[rᵢ,nᵢ]`产生softmax位置权重；池化并归一化得到qL，再通过H_QL得到局部标量sL。
- `S_Q = (1−β)sG + βsL`是分数混合，不是跨注意力。
- 下方Pretraining taps区域为训练监督出口：`q = norm[(1−ρ)qA+ρqL]`约束分量锚点/Ordinal；`C(qA,qL)=[qA,qL,|qA−qL|]`送Artifact/Aesthetics辅助头。
- 辅助分量头在当前下游配置不启用；总图的下游QAM只报告S_Q，不把辅助头画成下游有效预测。

## AAM展开图

- 三个独立残差Adapter：AG用于全局图像向量v，AP用于P₀，AT用于文本向量t。
- 全局配对：`C(aG,aT)=[aG,aT,|aG−aT|]`送H_AG。
- 局部：W_I/W_T用于归一化匹配与softmax位置加权，池化AP(P₀)得到局部证据e；`C(e,aT)`送H_AP。
- `γ=Gate(aG,aT,sG,sP)`是逐样本门控；`S_A=γsG+(1−γ)sP`。
- 有文本引导池化，没有多头跨注意力，也没有全局/局部融合输出z。

## 四类SRM不变

| 调用 | 输入 | Predictor复用 |
|---|---|---|
| I-I | v₁, v₂ | h_I / h_I |
| I-Prompt | aᶜ, tP | h_I / h_T |
| I-Answer | aᶜ, tA | h_I / h_T |
| Prompt-Answer | tP, tA | h_T / h_T |

四类是同次预训练的四项主要关系损失，不是四次实验。Clean View人工评分、分量锚点/Ordinal、辅助头和正向局部/词级目标仍保留，PCGrad不变。关闭的质量文本五档CE与启用的分量Anchor CE仍严格区分。

## 共享、复用与迁移

- 同一阶段中的多个视图共享编码和投影参数。QAM与AAM共享的是外部编码/基础特征，不是各自所有任务参数。
- 实线分叉点复用同一张量；跨线留白、没有分叉点的位置不连接。
- 紫色虚线仅表示跨阶段表示初始化，不是跨阶段绑定权重。
- 迁移包括共享表示、上下文词元、QA/QL、NSS MLP/门控、AG/AP/AT、α、ρ等保留表示参数。
- 数据集评分头、匹配Linear W_I/W_T、逐样本Alignment评分门控和Quality分数混合β重置。完整QAM/AAM并不是全部参数无差别继承。
- h_I/h_T为预训练Predictor，不作为下游评分头迁移。

## 文件与复现

- `01_split_framework.*`：更新后的完整总图。
- `02_QAM_module.*`：质量模块独立展开。
- `03_AAM_module.*`：一致性模块独立展开。
- `04_SRM_module.*`：SimSiam关系模块。
- 每张图均有PPTX、高清PNG、矢量PDF；另有四页合并PDF `all_split_figures.pdf`。
- 所有图形、文字、公式标注、token及连线均为PowerPoint原生对象，没有位图冒充可编辑模块。
- `generate_split.py`、`drawing_primitives.py`为完整生成代码，不依赖旧版PPT素材。
- 附DejaVu Sans字体和许可证；编辑前安装字体可减少替换。

```bash
python -m pip install -r requirements.txt
python generate_split.py --output-dir ./rendered
```

导出链始终为PPTX→LibreOffice PDF→PNG。四张图长边均至少5600像素。字体、页面边界、原生对象和实际渲染均检查；未启动训练，未修改历史结果。

# 质量优先预训练与残差后 OAS 两组独立对照

原基线及已有实验目录不变。两组均保留 projection_only_v1、5个上下文词元、原六维 legacy NSS（第六维仍重复二阶矩，不是讨论中的峰度替换版）、P0 绕过 PI、原始 RichHF train/val/test 划分、seed42；质量文本五档CE和negative关闭。所有新运行有独立 run_id。云端只运行合成测试，不进行真实训练。

## 第一组：quality_first

1. **quality_stage**：从原始 CLIP/ResNet 初始化，只训练质量目标，含 I-I 的 g→PI 路径、质量回归/排序、组件辅助目标、Component Anchors/Ordinal。图文/词级/局部一致性目标系数为零，文本上下文、PT和一致性模块冻结并 eval。PI、图像 Predictor 因 I-I 可训练。
2. 按验证集 `val_component_anchor_SROCC` 最大选择质量阶段的完整 `best_pretrain.pth`。完整模型（包括两个编码器、投影、全部评分头、Predictor、BN缓冲和未训练的一致性侧初始参数）strict=True迁移到下一阶段，不重新随机初始化部分模块，不加载旧项目检查点。
3. **alignment阶段**：新优化器/调度器、新seed42起点，非resume。冻结 CLIP、ResNet、全局Linear、独立局部投影、QA/QL、NSS、质量头及两种质量混合参数；冻结模块保持eval，BN/Dropout不更新。仅 PI/PT、共享文本维度映射T、5上下文词元、AG/AP/AT、局部匹配投影、一致性评分头/gate、图文Predictor可训练。质量损失和I-I损失系数为零，一致性目标沿用基线权重。共享文本Transformer本体始终冻结，梯度可以经过它更新上下文。
4. 按 `val_align_proxy_SROCC` 最大选 `best_align_pretrain.pth`，三个下游均使用该文件。不能继续用被冻结的Component Anchor指标选择第二阶段，否则无法选择一致性学习结果。

两个阶段**各50轮上限、15轮patience**，保留基线学习率倍数与cosine策略；每阶段只有一种任务，不启用跨任务PCGrad。第一组增加预训练预算，因此是训练协议对照，不能宣称为相同总预算的单因素实验。所有分支仍沿用现有forward用于诊断，零系数表示目标关闭，不意味着所有中间计算被删除。

冻结审计对所有冻结参数和冻结模块缓冲区计算训练前/后SHA256，末尾不一致则拒绝完成；来源完整checkpoint在加载前/后核验SHA256。阶段记录为`phase_manifest.json`，质量来源为`quality_stage_source.json`，两者纳入完成清单和打包。

下游恢复原v2基线的训练设置（CLIP/ResNet可训练），只继承原规则允许的表示参数；全局/局部/一致性数据集评分头按原规则重置，阶段冻结策略不带入下游。

## 第二组：residual_oas

保持联合预训练和所有基线目标/权重/训练预算，仅为Component Anchors/Ordinal增加显式配置：

```yaml
model:
  dual_quality_representation:
    anchor_global_source: residual  # 默认adapter，历史行为不变
```

原OAS：`q = normalize((1-rho)*qA + rho*qL)`。

新OAS：`q_oas = normalize((1-rho)*hG + rho*qL)`，其中`hG = g + alpha*stopgrad(norm(g))*normalize(qA)`，复用实际送入Global Head的残差相加结果。不额外单独归一化hG，不改变混合权重初始化，不增加QA/BN/Dropout调用。

只有组件锚点CE/Ordinal使用新`component_anchor_clean_z`。MOS评分、辅助组件回归、I-I、图文关系、原`quality_clean_z`以及其他质量监督均保留原路径。模型不新增权重，默认配置数值行为不变。新OAS梯度可以经过残差直接进入g，并作用于alpha。

按 `val_component_anchor_SROCC` 最大选 `best_component_anchor_pretrain.pth`，供三个下游使用。下游恢复原配置；OAS是预训练监督，推理评分结构不增加模块。

## 服务器命令

先同步并检查环境，避免在同一目录重复启动同一组。`all`明确授权预训练后自动接三个下游；默认不带action只进行预训练（第一组包含两个预训练阶段）。所有存在的阶段必须先通过完成核验才跳过；未完成目录拒绝覆盖，不自动resume。

```bash
pwd
git status --short
git branch --show-current
git pull --ff-only origin work
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/run_quality_oas_controls.py check-configs
```

终端一：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_quality_oas_controls.sh quality_first all
```

终端二（GPU1确有空闲且可用时）：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_quality_oas_controls.sh residual_oas all
```

只预训练：将`all`换成`pretrain`。只运行第一组质量阶段：`quality_first quality`。后续独立下游：将action换成`agiqa1k`、`agiqa3k`、`aigciqa2023`或`downstream`。下游保持train/val/test、60轮、max val_quality_SROCC，训练seed42；1K subset1078 split3407，另两集split42；PI/PT初始LR1e-4、global projection 1e-5，其他参数不额外调节。

只读计划：

```bash
bash scripts/run_quality_oas_controls.sh quality_first all --dry-run
bash scripts/run_quality_oas_controls.sh residual_oas all --dry-run
```

## 打包

两组分别打包，包含CSV、配置、日志、来源/阶段/完成记录及存在的划分清单，不含`.pth`等模型权重。第一组包同时包含两个预训练阶段证据。同名已有包拒绝覆盖。

```bash
bash scripts/run_quality_oas_controls.sh quality_first pack-downstream
bash scripts/run_quality_oas_controls.sh residual_oas pack-downstream
```

生成`projection_only_v1_ctx5_quality_first_downstream_results.tar.gz`和`projection_only_v1_ctx5_residual_oas_downstream_results.tar.gz`。预训练阶段完成但下游未完成时，使用`pack`仅打包预训练。

结果属于独立代码版本与训练协议；不承诺逐位复现，不把旧成绩标为新结果，不替换已接受基线。第二组未改统计描述内容，第一组也未改NSS；本批不能回答峰度替换是否有效。

## 验证记录

云端CPU合成验证：11项专项测试通过；最终全套432项unittest通过。包括实际训练入口的微型两阶段保存/加载/选模链路、完整张量迁移、冻结参数及BN缓冲区不变、可训练PI的无affine BN继续更新、OAS残差来源及alpha梯度、相同权重下原评分输出不变、配置/来源篡改拒绝、阶段顺序、CPU真实启动拒绝、打包排除权重。9份配置检查及两组shell dry-run通过。未启动真实预训练或下游，未加载服务器的大模型检查点。

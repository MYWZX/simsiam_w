# CLIP后共享残差Adapter：冻结预训练，冻结/解冻下游对照

## 结构版本与位置

独立结构版本 `clip_adapter_v1`，仍使用projection_only_v1，不恢复跨注意力或融合z。原最好基线与已有冻结实验不变。

```text
CLIP图像编码器 → 原有L2输出 x（768维）
                     ↓
  新共享Adapter：x + 0.1·Linear256→768(GELU(Linear768→256(LayerNorm(x))))
                     ↓ 768维，无额外末端归一化
  原semantic_proj：Linear768→256 → g
                     ├─ Quality路径
                     └─ PI → Alignment / I-I关系
```

Adapter名为 `backbone.global_adapter`，独立于 `backbone.global_encoder`，因此冻结CLIP不会冻结它。预训练与下游都启用且训练Adapter，按现有权重迁移完整继承。
不同视图/任务共享同一组Adapter参数；不是每个任务/视图新建一个Adapter。文本塔、文本T/PT、QA/QL和ResNet局部路径保持原样。P₀仍独立投影、绕过PI。

瓶颈256、固定残差系数0.1，不增加Dropout。最终上投影weight/bias零初始化，所以初始是直通x；第一步学习上投影，后续下投影也开始获得梯度。创建新模块时隔离CPU初始化RNG，以保留既有模块的初始化；不承诺训练逐位复现。
Adapter共395,776个可训练参数。新增独立 `global_adapter` 优化器组，两阶段初始LR均1e-4（预训练mult=5、下游mult=1）；weight decay沿用各阶段基线。全局semantic_proj仍1e-5、PI/PT仍1e-4。梯度审计在启用时单独报告global_adapter的质量/一致性梯度，不与全局投影或跨注意力混计。旧有参数组及其归属/LR不变。

## 两组与固定协议

| 条件 | 预训练CLIP图像塔 | 下游CLIP图像塔 | Adapter |
|---|---|---|---|
| frozen | 冻结 | 冻结 | 两阶段始终训练 |
| unfrozen | 冻结 | 可训练 | 两阶段始终训练 |

两组共享**本次新增Adapter后重新预训练**的同一份权重，不使用旧无Adapter的冻结权重，更不使用Prompt/I-I权重变体来源。
预训练从原始CLIP/ResNet初始化，官方train/val分开，seed/split_seed42，50轮预算、patience15，最大val_component_anchor_SROCC选模。
下游当前基线60轮预算，train/val/test分开，最大val_quality_SROCC选模；train_seed42，1K subset1078/split3407，3K/2023 split42；Quality SoftSROCC0.03/0.02/0.04。
I-I=0.15、图像—Prompt=0.25、图像—Answer=0.20、Prompt—Answer=0.05；保留双质量残差、NSS、Component Anchors/Ordinal和现有目标；质量文本五档CE与negative仍关闭。
除此新增结构/参数组与冻结边界外，完整配置逐项复制当前基线。

## 严格兼容性

- 未启用Adapter的旧配置默认保持原计算与参数键，不新增随机层或缓冲区。
- Adapter只允许projection_only_v1，错误组合明确报错。
- 模块的dim/hidden/scale签名与全部参数都保存入representation checkpoint。
- 下游严格要求完整且匹配的Adapter；新/旧架构互载、缺键、多余键、尺寸或残差比例不匹配报错，不允许新模块静默随机初始化。
- 原有QA、PI/PT、局部投影、NSS、上下文和相关表示的完整迁移校验保留。全局Adapter参数在both任务间共享，现有PCGrad按实际梯度覆盖它。
- 来源config、版本、checkpoint_scope、选中epoch、完成记录、权重SHA及加载后来源继续检查。
- 未完成输出拒绝覆盖/自动resume；已完成源可校验复用。新all等同continue，会核验并跳过完整阶段。

## 服务器执行

先同步：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && git pull --ff-only origin work
```

第一步：只运行一次共享预训练（GPU0示例）。

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_adapter.sh frozen pretrain
```

**共享预训练完成后**，两个终端可并行运行下游：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=0 bash scripts/run_clip_adapter.sh frozen all
```

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=1 bash scripts/run_clip_adapter.sh unfrozen all
```

all会校验并跳过已完成的共享预训练，分别跑三个数据集，共1次预训练、6次下游。
不要从空目录同时启动两组all创建同一个预训练目录。
若希望先预训练后自动接下游，可直接先执行frozen all；只有确认其共享预训练已经完成后，才能在另一张卡启动unfrozen all。
若共享预训练已有完整结果，使用all复用，不再执行会拒绝已有输出的pretrain命令。
支持单数据集、downstream、continue、--dry-run；默认只预训练。HF_HUB_OFFLINE默认1，不自动换初始化来源。云端CPU拒绝真实启动。

全部完成后：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && bash scripts/run_clip_adapter.sh frozen pack-downstream && bash scripts/run_clip_adapter.sh unfrozen pack-downstream
```

生成：
- projection_only_v1_ctx5_clip_adapter_v1_frozen_downstream_results.tar.gz
- projection_only_v1_ctx5_clip_adapter_v1_unfrozen_downstream_results.tar.gz

含CSV、配置快照、元数据、日志、完成记录和来源指纹，不含大模型权重；同名包拒绝覆盖。
沿用划分检查/打包规则：1K/2023要求运行清单；3K未启用成员记录时明确报告仅seed核查。独立data_split_manifest.json不在原打包白名单，1K/2023成员摘要在run_metadata.json。
该新结构独立报告，旧基线成绩不算作新结构成绩，冻结CLIP和新增Adapter均不保证改善。

## 本次验证

完整CPU回归355项通过。新增8项覆盖：旧配置无状态键/RNG/输出变化、零初始化eval一致、两步Adapter更新及冻结边界、三视图与两个任务梯度、独立梯度审计与参数分组/学习率、完整迁移/旧新结构拒绝/缺键/维度/比例/CLIP缺键拒绝、两组共享来源/六次模拟下游/打包/拒绝覆盖。新用例隔离并恢复随机状态。两组all --dry-run及shell语法检查通过，未启动真实训练。

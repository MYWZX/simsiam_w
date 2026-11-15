# 五维NSS清理与受控消融

当前历史最好基线及所有旧配置不变。新实验版本为 `projection_only_v1_ctx5_nss5_<variant>`；不把旧结果改名为新结果。

## 四个独立组

| variant | 归一化 | 注入NSS到P0 | gate直接读取n | gate输入维度 |
|---|---|---|---|---:|
| legacy_both | 原MSCN-like | 是 | 是 | 261 |
| mscn_both | 标准MSCN | 是 | 是 | 261 |
| legacy_inject | 原MSCN-like | 是 | 否，仅读取局部特征r | 256 |
| legacy_gate | 原MSCN-like | 否 | 是 | 261 |

归一化对照：legacy_both vs mscn_both，只有归一化算子不同。路径消融固定原归一化：legacy_both / legacy_inject / legacy_gate。复用共同的both组，总计4次独立预训练、12个下游，不重复训练同一个控制组。本次不是2×3完整交叉设计；不从这四组推断标准MSCN下哪条路径更好。

每个区域的五维向量顺序固定：

    n5 = [mean(z), mean(z²), mean(|z|), mean(|gradient(Y)|), R]
    R = mean(|z|) / (sqrt(mean(z²) + 1e-6) + 1e-6)

删除原第六维 mean(z²) 的重复项，不增加新统计量。第二维是原始二阶矩，不称中心方差；R是统计比值，不称GGD形状alpha。当前局部网格通常14×14，所以n为B×196×5。

旧归一化保持方框窗口、零填充、残差能量分母和原稳定常数；标准MSCN使用7×7、sigma7/6的归一化高斯核、replicate边界、真实局部二阶矩方差和C=1/255。两版五维统计、梯度算子、ratio常数、池化、后续网络完全相同。“归一化算子”这一因素含窗口、方差定义、边界与C，不能再将差异归为其中单一因素。

三条路径的确切含义：

- both：n→MLP→与P0相加→QL/L2→r；[r,n]→gate→token Softmax→加权汇聚r。
- inject_only：仍执行n注入，gate只读取r。不直接拼接n，但r间接含有统计信息，不能称gate完全不受NSS影响。不是均匀池化。
- gate_only：删除NSS投影器的可训练参数且不执行注入，r由P0经QL产生；[r,n]输入gate。仍使用学习式加权特征池化，不改成加权标量分数。

## 匹配初始化与训练策略

五维模式在初始化时先按历史6维结构生成共同模板，再合并重复列；额外构造所需参数时隔离RNG。因此同seed下legacy_both与折叠后的旧6维初始化一致，四组公共参数初始化相同，不因删输入/模块改变其他任务头的初始化。gate_only不保留无效投影参数，inject_only保留canonical gate的r输入列和偏置。标准归一化组使用完全相同的权重初值。

所有组预训练与下游均启用同版描述/路径，projection_only_v1、5上下文、g→PI、P0绕过PI、双质量残差等目标保持。每组都从原始CLIP/ResNet初始化重新预训练，不加载本项目旧预训练/迁移权重。预训练官方train/val保留，最大val_component_anchor_SROCC选模。三个下游使用本组best_component_anchor_pretrain.pth，60轮、最大val_quality_SROCC选模，train_seed42；1K subset1078 split3407，3K/2023 split42。其他参数取现有固定基线。

全局投影与其他公共参数保留原优化器分组、学习率与权重衰减。仅被移除的NSS参数不再进入优化器。下游全局投影初始LR仍1e-5，PI/PT仍1e-4。

NSS描述和路径分别有持久签名，迁移时均须一致。运行器检查配置/完成状态/SHA/源权重形状，拒绝四组之间互用、旧6维/前次Stats6/GGD混入、漏载或重复初始化局部参数。历史模式缺省仍旧6维/both，数值计算与初始化行为保留。

## 旧检查点等价迁移验证

第2与第6通道相等，所以NSS MLP第一层将两列相加再删第6列；gate第一层对拼接后对应的两列作同样操作。其余权重保持原样，两条MLP的偏置也不变。这是函数重参数化，不是新训练，也不是宣称权重衰减和优化轨迹等价。

在服务器仓库执行（仅CPU验证和生成独立验证文件；不训练、不下载或混入其他预训练来源）：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/migrate_nss5_checkpoint.py \
  outputs/richhf18k_projection_only_v1_baseline_ctx5_dev_seed42/best_component_anchor_pretrain.pth \
  --output outputs/nss5_equivalence_only/legacy_best_folded5.pth
```

原文件不改写；目标存在时拒绝覆盖。报告 `legacy_best_folded5.pth.migration.json` 包含父SHA、新SHA、未改动张量数和实际源权重两条MLP的eval数值误差。验证输入使用由固定seed生成的图像计算真实五维描述，并使用实际旧权重；不实例化CLIP/ResNet，因此这个服务器工具不宣称已跑完整编码器输出等价测试。

新文件明确标为 `checkpoint_scope=migration_verification_only`，不保存optimizer/scheduler等续训状态。四组实验不使用它，正常选模来源核验会拒绝把它当成新预训练结果。完整模型质量/一致性eval输出等价性由无外部权重的假骨干单元测试覆盖；浮点合并改变运算次序，只要求误差容限内一致，不承诺逐位相同。云端没有真实大权重，实际旧基线数值误差须运行上述命令后读取报告。

## 服务器运行

默认动作只预训练；`all`明确表示完成本组预训练后依次运行三个下游。只验证已完成阶段后复用，不覆盖或resume未完成输出。

先在仓库干净时同步work，然后两张卡先跑归一化对照：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_nss5_controls.sh legacy_both all
CUDA_VISIBLE_DEVICES=1 bash scripts/run_nss5_controls.sh mscn_both all
```

上述任务完成并释放GPU后，再分别跑路径消融：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_nss5_controls.sh legacy_inject all
CUDA_VISIBLE_DEVICES=1 bash scripts/run_nss5_controls.sh legacy_gate all
```

不想自动跑下游时，把all改为pretrain。已完成预训练后可用downstream或指定agiqa1k / agiqa3k / aigciqa2023。配置/计划核查：

```bash
bash scripts/run_nss5_controls.sh legacy_both all --dry-run
bash scripts/run_nss5_controls.sh mscn_both all --dry-run
bash scripts/run_nss5_controls.sh legacy_inject all --dry-run
bash scripts/run_nss5_controls.sh legacy_gate all --dry-run
```

打包命令：

```bash
bash scripts/run_nss5_controls.sh legacy_both pack-downstream
bash scripts/run_nss5_controls.sh mscn_both pack-downstream
bash scripts/run_nss5_controls.sh legacy_inject pack-downstream
bash scripts/run_nss5_controls.sh legacy_gate pack-downstream
```

输出为 `projection_only_v1_ctx5_nss5_<variant>_downstream_results.tar.gz`，包含预训练和三个下游的CSV、配置、日志、元数据、来源指纹，不含权重。只有完成且核验通过才允许打包。3K沿用原未记录完整成员表的配置，不能据种子核查宣称实际划分成员已独立验证。

这些新组独立报告，不自动替换当前最好基线。CUDA随机性、改变参数化后的优化和权重衰减差异均存在；即使初始化/eval等价，也不保证完整训练逐位复现。

## 本次实现验证

- 全量unittest：395项通过；五维数值与协议定向：15项通过。
- 测试覆盖旧6维前五通道逐元素相等、重复通道确实相等、原/标准五维统计定义一致、四组公共初始化及旧6维折叠初始化一致。
- 合并权重后完整假骨干模型eval质量、一致性及所有相关浮点输出在atol/rtol=3e-6内一致，实测误差记录于nss5_validation.json；不冒称实际大模型检查点已在云端核验。
- 被关闭的注入不执行且无可训练冗余投影参数、gate输入维度正确、各启用路径有质量梯度、多裁剪token顺序与AMP/平坦输入有限。
- 同版表示权重完整迁移、归一化或路径错配/缺签名/缺参数拒绝；参数分组与全局投影LR保持。
- 四个新预训练和十二个下游使用假产物验证来源/完成记录/选模/预算/划分证据及不含权重的打包流程；CUDA缺失拒绝真实训练。
- 四组all --dry-run、shell语法与git diff --check通过。没有运行真实训练，没有覆盖旧实验或修改当前最好基线。

# 标准 MSCN 两版独立实验

版本：`mscn_stats6_v1`、`mscn_ggd_v1`。旧配置不指定 `model.nss_text_gate.descriptor` 时仍为 `legacy`，原计算、参数名和检查点行为保持。新模式要求 projection_only_v1 和双质量表示。当前实现不启动训练，不改历史结果。

## 共同的标准 MSCN

先将 ImageNet 标准化的输入恢复为 RGB [0,1]，取亮度 Y=0.299R+0.587G+0.114B。

- 7×7、sigma=7/6 的归一化高斯核，边界采用 replicate。
- mu=G*Y；sigma=sqrt(max(G*(Y²)−mu²,0))。
- Z=(Y−mu)/(sigma+1/255)。这是在 [0,1] 尺度上的 C=1（8-bit）MSCN。
- 采用 float32 计算并关闭该区域的 AMP；方差下限仅作数值保护，不估计 GGD 的位置参数。
- 按局部特征图的实际网格 adaptive-average-pool。224×224 对应当前 14×14 网格时每格 16×16 个系数；这是统计区域，不代表 ResNet 感受野大小。

`models/nss.py` 包含两种无可训练参数的描述器。高斯核及 GGD 查表数据是确定性的非持久 buffer，随模型迁移设备。它们无需外部 scipy 或逐块 CPU 优化。

## A：MSCN + Stats6

每个区域输出：

| 通道 | 含义 |
|---|---|
| 1 | MSCN 系数均值 |
| 2 | MSCN 系数标准差（中心二阶矩的平方根） |
| 3 | MSCN 系数绝对值均值 |
| 4 | MSCN 系数偏度（中心三阶矩 / 标准差³） |
| 5 | MSCN 系数超额峰度（中心四阶矩 / 标准差⁴ − 3） |
| 6 | 原亮度梯度幅值均值（水平/垂直前向差分） |

这六项有统计相关性，但没有原版“第二维和第六维完全相同”的重复项。极低方差区域的标准差、偏度、峰度置零，避免不可辨识高阶标准化矩。高阶矩可能受离群值影响，这是否改善质量评价需要实验验证。

尺寸仍为 B×196×6；NSS MLP 输入6维，gate输入256+6=262维。原 P0、QL、特征池化及评分方式不变。

## B：MSCN + GGD

对每个区域的 Z，拟合零均值对称广义高斯分布：

    p(z)=alpha/[2 beta Gamma(1/alpha)] * exp(−(|z|/beta)^alpha)
    variance=beta² Gamma(3/alpha)/Gamma(1/alpha)

用矩匹配估计形状 alpha：

    R=(mean |Z|)² / mean(Z²)
    R=Gamma(2/alpha)² / [Gamma(1/alpha) Gamma(3/alpha)]

在 alpha∈[0.2,10]、步长0.001的理论比值表中查找并线性插值；超出可拟合范围时截断。查找索引离散，区间内插值可导。第二个输出为 mean(Z²)，即零均值假设下估计方差，而非 MSCN 系数的有限样本中心方差。没有用“代理比值”冒充 alpha。

每格输出 `[alpha, variance]`，B×196×2。NSS MLP输入改为2，gate输入为258。beta可由以上公式推导，因此不重复输入；不额外拼接全图 GGD，不拟合邻域乘积的 AGGD。这是局部单尺度零均值 GGD 版本，并非某篇论文整套 NSS 特征的复刻。

当二阶矩≤1e-10时，形状不可辨识，明确返回alpha=2，保留实际二阶矩；这不是宣称平坦区域真的服从高斯分布。每区域通常只有256个样本，GGD估计存在采样误差，alpha范围也有限。

两版均维持 `n → NSS MLP → 与P0相加 → QL → r`，以及 `[r,n] → gate → token Softmax → 特征加权池化`。不改成分数加权。

## 来源与训练协议

两版的预训练和三个下游均使用同版 NSS。重新从原始 CLIP/ResNet 初始化预训练，不加载旧项目权重。预训练保留官方 train/val，最大 val_component_anchor_SROCC 选模。下游只加载对应新预训练 best_component_anchor_pretrain.pth，保持60轮、最大 val_quality_SROCC、训练seed42；1K subset1078 split3407，3K/2023 split42。所有其他参数由当前基线生成，保留五个上下文词元、PI/PT学习率、全局投影分组及现有损失。

检查点带持久的 `nss_descriptor_signature`。迁移时拒绝旧版/Stats6/GGD混用，并核查所有NSS MLP/gate参数完整和形状匹配。运行器另核对配置、完成状态、来源SHA256、签名和NSS权重形状。不能通过旧6维权重形状恰好相同来混入Stats6。

新统计会改变前向表示；GGD维度还改变了部分参数初始化和参数量。结果必须按独立版本报告，不承诺逐位训练复现，也不能把差异全部归因于高斯窗口这一项。两版同时改变归一化和描述方式；若要隔离各个因素，需要另加对照。

## 服务器命令

在服务器仓库目录先同步干净的 work 工作区；有修改先保留，不强制重置。默认动作仅预训练。下列两个独立输出可分别放在两张卡：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_nss_ablation.sh stats6 pretrain
CUDA_VISIBLE_DEVICES=1 bash scripts/run_nss_ablation.sh ggd pretrain
```

如果希望各自预训练结束后自动依次运行三个下游，分别改用：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_nss_ablation.sh stats6 all
CUDA_VISIBLE_DEVICES=1 bash scripts/run_nss_ablation.sh ggd all
```

`all`采用验证完成后跳过的continue流程；不会resume未完成训练。已完成阶段核对通过才复用，失败目录拒绝覆盖。不要同时启动同一个版本的pretrain和all。

只检查配置和阶段计划（不训练）：

```bash
bash scripts/run_nss_ablation.sh stats6 all --dry-run
bash scripts/run_nss_ablation.sh ggd all --dry-run
```

预训练已完成后，可用 `stats6 downstream` / `ggd downstream` 跑三个下游，或用具体 `agiqa1k` / `agiqa3k` / `aigciqa2023`。下游来源未完成或不匹配时会拒绝启动。3K沿用原record_split_manifest=false：报告种子核查，不伪造样本成员证据。

打包（配置、CSV、元数据、日志和来源指纹，不含模型权重）：

```bash
bash scripts/run_nss_ablation.sh stats6 pack
bash scripts/run_nss_ablation.sh ggd pack
bash scripts/run_nss_ablation.sh stats6 pack-downstream
bash scripts/run_nss_ablation.sh ggd pack-downstream
```

文件名前缀分别为 `projection_only_v1_ctx5_mscn_stats6_v1` 和 `projection_only_v1_ctx5_mscn_ggd_v1`。CPU云端只运行数值、假骨干、假产物协议测试，不运行真实训练。脚本默认Python路径沿用服务器WZX-AI环境，可用ABLATION_PYTHON覆盖。

## 本次验证记录

- `OMP_NUM_THREADS=1 python -m unittest discover -s tests`：369项通过。
- 在上述检查后补充了源检查点签名/缺参拒绝测试，`python -m unittest discover -s tests -p 'test_nss_*.py'`：15项通过。
- Gaussian MSCN与独立float64公式比对；六维描述与逐区域中心矩比对；GGD理论矩恢复已知形状与方差；平坦/低对比/随机输入、CPU AMP及输入梯度有限。
- 两种模式假骨干前向/反向、多裁剪token顺序、旧版状态与输出保持、预训练和下游参数组归属一致。
- 同版表示权重逐张量迁移一致；旧版/两新版互相混载、签名错误、NSS参数缺失均拒绝。
- 假训练产物检验两次预训练及六个下游的完成记录、来源SHA、60轮预算、选模规则、划分证据和无权重打包；不代表真实GPU训练已验证。
- 两种 `all --dry-run`、shell语法及 `git diff --check` 通过。云端没有访问训练数据或创建真实实验结果。

## 预训练未完成时的诊断包

`pack-downstream`要求来源预训练有完整且可验证的完成记录。缺少`completion_manifest.json`时，不应伪造该文件、直接宣称成功或先删除日志；它只能说明完成验证不可用，不能据此判断最初失败原因。

在服务器仓库运行以下独立脚本（默认同时收集两版）：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/pack_nss_diagnostics.py
```

生成`projection_only_v1_nss_diagnostics_<时间戳>.tar.gz`。无需CUDA、无需torch、不启动训练，也不更改结果目录。包含当前及最近三个同名.failed_/.incomplete_目录的日志/元数据、当前与归档CSV、相关配置/源码和git状态。单文件超过2MiB仅收集末尾2MiB并在报告中标明。模型权重只记录文件名与大小，不读取其内容，不打入包内。已删除的日志无法恢复。这个诊断包不属于已完成实验结果包。

## 服务器重启后的第16轮断点恢复

仅这两组NSS运行器增加显式 `recover-pretrain` / `recover-all` 动作；普通 `all` 仍不会自行恢复未完成训练。两份服务器断点确认保存epoch16、字段齐全、签名[1,6]/[2,2]、最佳epoch5、no_improve11，因此应从epoch17继续。第17轮未保存的部分重算，早停耐心不清零；若始终无提升且patience15不变，会在epoch20达到早停，而非重新额外训练15轮。

同步干净的work后，分别在两个终端运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_nss_ablation.sh stats6 recover-all
CUDA_VISIBLE_DEVICES=1 bash scripts/run_nss_ablation.sh ggd recover-all
```

各自恢复预训练，正常结束且核验通过后自动依次启动其三个下游。只恢复预训练则改用 `recover-pretrain`。只核查断点和计划、不修改文件也不训练：

```bash
bash scripts/run_nss_ablation.sh stats6 recover-pretrain --dry-run
bash scripts/run_nss_ablation.sh ggd recover-pretrain --dry-run
```

恢复前严格核对原启动身份、配置SHA、实际配置快照、完整断点模式及签名、CSV与断点轮次、历史最佳和早停计数，以及已有最佳Component检查点。比较原启动提交与当前的训练、模型、数据、损失、utils源码；这些计算代码发生变化时拒绝恢复，允许本次运行器/核验逻辑的改动。脚本会拒绝正在运行的同配置训练，采用每组独立的进程锁，并拒绝覆盖完成记录或已有恢复尝试。

开始训练前将原CSV、日志、配置、元数据保存在 `recovery_inputs/`。原`.pth`也会保留：优先建立硬链接，文件系统不支持时复制；训练器通过原子替换保存新权重，所以旧链接保留原始断点。不要手动原地修改备份文件。随后调用现有训练器的`--resume`恢复模型、optimizer、scheduler、AMP scaler、Python/NumPy/Torch/CUDA随机数状态。

新增 `recovery_manifest.json` 记录输入断点SHA、原选模状态、恢复命令、代码兼容核查和新运行环境；成功后新增 `recovery_complete.json`。完成核验检查原CSV字节前缀未被改写、恰好一次resume、连续完整轮次、恢复输入指纹、全历史最大val_component_anchor_SROCC选模，以及配置预算/早停规则。只通过上述核验后才发布正常completion_manifest并允许下游加载。

来源指纹增加 `training_continuity=interruption_recovery_v1` 与恢复清单SHA；这些结果应报告为中断恢复实验，不冒充未中断训练，也不保证数据加载器工作进程或CUDA运算逐位复现。没有改变两个NSS描述器、目标或训练超参数，没有重置第5轮最佳状态。

原有 `pack` / `pack-downstream` 命令保持使用，新增恢复清单和原始日志/CSV等证据会一起打包，仍不包含原始或恢复后的权重。其他实验的默认完成核验继续拒绝resume记录，不宽泛放开历史实验来源。

当前入口支持一次显式恢复。若恢复本身再次异常，会保留原始和新增证据并拒绝再次覆盖；需要单独诊断，不能删除恢复记录后强行继续。已完成且验证通过的恢复运行，再次执行 `recover-all` 会复用完成状态，不重复预训练。若下游已有未完成输出，仍按原规则拒绝覆盖。

恢复入口验证：全量unittest 378项通过；补充CLI顺序/失败中止与训练代码兼容检查后，NSS定向25项通过。包括CPU与dry-run不训练不创建备份、活动进程拒绝、模式/轮次/配置/早停状态错误拒绝、两版从16到17的模拟恢复、原证据与权重保留、来源标记及下游接受、打包排除权重、失败不发布完成记录和禁止覆盖恢复尝试。云端未运行真实训练；实际GPU恢复效果仍需服务器执行验证。

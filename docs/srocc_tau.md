# 原六维基线：Soft-SROCC温度微调

复用当前历史最好基线及选定v2配置，不采用五维、标准MSCN、GGD、冻结CLIP或Adapter候选。只做下游，不重新预训练、不resume。

| variant | train.srocc_tau |
|---|---:|
| 原基线（复用旧结果，不重跑） | 0.02 |
| tau010 | 0.01 |
| tau040 | 0.04 |

总计两个参数值×三个数据集=6个下游。温度控制mini-batch软排序的平滑度：sigmoid((xj−xi)/tau)，同时用于预测值与标签的soft rank。更小通常更接近硬排序，也更容易饱和，梯度行为需实测；不保证更小就更好。评价集的真实SROCC/PLCC算法不改。

这是现有的共享温度：1K只有Quality Soft-SROCC有效，3K/2023同时影响Quality和Alignment的已启用Soft-SROCC。不是只修改Quality，也不是修改损失系数。保留Quality Soft-SROCC权重1K/3K/2023=0.03/0.02/0.04、Alignment权重、Quality pairwise rank0.03及其margin/hard ratio、MSE和其余目标。

完全保留六维NSS-like、5上下文、全局投影、独立P0及全部基线结构。CLIP/ResNet LR2e-6，全局投影fusion组1e-5，QA/PI/PT1e-4，shared_projection_mult1.0、weight_decay0.002及原scheduler。没有叠加以前的局部LR/dropout/alpha/beta等候选。

固定原预训练best_component_anchor_pretrain.pth，SHA256：

    fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686

训练seed42，1K subset1078 split3407，3K/2023 split42；保留train/val/test，60轮，按最大val_quality_SROCC选模，不拼接train+val。验证源的模式/配置/完成记录/SHA及原同划分控制CSV指纹；2023明确复用已接受的Quality Soft-SROCC0.04控制。运行时仅额外开启划分和优化器初始化记录，用于核验，不改变训练目标。

开始前检查每个输出是否已存在，拒绝覆盖和无声恢复。完成后核查60轮、实际划分、优化器组/学习率、损失系数贡献与来源SHA。打包使用元数据/CSV/日志/配置白名单，不包含大权重。

## 服务器执行

在仓库同步干净work后，默认Python沿用WZX-AI环境；ABLATION_PYTHON可覆盖，默认HF_HUB_OFFLINE=1。云端CPU会拒绝真实训练。

单卡顺序跑完六个下游：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_srocc_tau.sh all
```

也可分两张卡，各跑一个温度的三个数据集，两个方案任选其一，不要和all同时运行：

```bash
# 终端一
CUDA_VISIBLE_DEVICES=0 bash scripts/run_srocc_tau.sh tau010 agiqa1k && \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_srocc_tau.sh tau010 agiqa3k && \
CUDA_VISIBLE_DEVICES=0 bash scripts/run_srocc_tau.sh tau010 aigciqa2023

# 终端二
CUDA_VISIBLE_DEVICES=1 bash scripts/run_srocc_tau.sh tau040 agiqa1k && \
CUDA_VISIBLE_DEVICES=1 bash scripts/run_srocc_tau.sh tau040 agiqa3k && \
CUDA_VISIBLE_DEVICES=1 bash scripts/run_srocc_tau.sh tau040 aigciqa2023
```

如果部分完成或失败，保留其输出，不重跑all；只执行未启动的具体variant/dataset。此入口不支持中断续训，避免将未核查的恢复结果混入本次单因素对照。

仅核对配置：

```bash
bash scripts/run_srocc_tau.sh check-configs
```

全链路只读预检（需要服务器已有源和控制结果；不训练、不创建输出）：

```bash
bash scripts/run_srocc_tau.sh all --dry-run
```

六个下游完成后打包：

```bash
bash scripts/run_srocc_tau.sh pack
```

上传 `projection_only_v1_v2_srocc_tau_1ksplit3407_results.tar.gz`，里面同时含原同划分控制与六组候选，以及当前参考配置和指纹。不会自动选择新基线。需比较所有Quality/Alignment SROCC/PLCC，不按测试集另挑轮次；这是已查看划分上的探索性调参，不视为独立确认。

## 本次验证

10项CPU测试通过：六个配置仅温度/运行标识/证据记录标志改变；原六维模型与其他损失、优化配置相同；错误源SHA/五维替换/旧2023控制/缺控制/改动产物/错误学习率与损失贡献拒绝；CUDA缺失不训练；六次调度与失败中止；白名单打包无权重。真实compute_losses测试确认三个soft-rank调用共享tau，固定权重预测及MSE/成对排序项不变，总损失变化正好来自Quality与Alignment Soft-SROCC项，梯度有限。

check-configs、shell语法、git diff --check通过。只添加新运行器、文档和测试，没有修改模型、训练器或损失实现。云端没有启动真实训练，也未读取服务器大检查点；实际六个下游的结果仍待服务器运行。

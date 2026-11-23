# 局部一致性监督、匹配温度和文本上下文LR：六组独立单因素

固定原v2基线预训练`best_component_anchor_pretrain.pth`，SHA256 `fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686`。不重新预训练，不加载quality_first/residual_oas等其他来源，不继承前几轮候选。只运行3K/2023，训练seed42、划分seed42、60轮、train/val/test分开，仍按max val_quality_SROCC选择同一轮，统一报告两个任务SROCC/PLCC。

| 变体 | 唯一有效变化 | 原基线 |
|---|---|---|
| alocalaux050 | 局部一致性辅助MSE权重0.15 | 0.30 |
| alocalaux200 | 局部一致性辅助MSE权重0.60 | 0.30 |
| amatchtau050 | 局部空间匹配温度0.5 | 1.0 |
| amatchtau200 | 局部空间匹配温度2.0 | 1.0 |
| actxlr050 | 5个可训练上下文词元LR=5e-6 | 1e-5 |
| actxlr200 | 5个可训练上下文词元LR=2e-5 | 1e-5 |

每组两个数据集，共12次下游。单因素分别比较，不把局部权重、温度、上下文LR组合起来。

## 计算与兼容

`lambda_align_global_aux`与`lambda_align_local_aux`分别控制全局/局部辅助MSE；未指定时各自回退到旧`lambda_align_aux`，默认0.3。权重必须有限且非负。两个权重相等时保留旧的先相加再乘系数的计算顺序；所有旧配置行为保持。局部监督候选显式固定全局辅助权重0.3，只改变局部权重。日志`weighted_align_patch`使用实际局部系数，核查器按两个独立权重重构总损失。

`model.align_local_temperature`作用于`softmax(cos(WI(AP(P0)), WT(AT(PT(T(text)))))/tau)`的局部位置轴，原实现tau=1。0.5使空间分布相对集中，2.0使其相对平滑。它不是Soft-SROCC的软排序温度；后者仍为0.02。没有跨注意力，也不改变P0绕过PI的路径。温度1跳过除法，严格保留默认算术；参数有限且>0。非默认温度与分支消融或联合预训练组合时明确报错。实际温度写入验证/测试运行诊断。

上下文候选只改`layerwise_lr.text_context_mult`为0.05/0.20；基础LR=1e-4。CLIP文本Transformer权重仍冻结，5个上下文向量继续训练。PI/PT、T、AG/AP/AT和其他参数分组均不因该项改名或重新分组。

所有候选保留原学习门控和文本引导局部池化；原六维NSS、projection_only_v1、5上下文、Quality设置、align LR1e-4、rank_align0.03、srocc_align0.04、全局投影1e-5及其他参数不变。Quality Soft-SROCC在3K为0.02，2023为0.04。不会承诺训练后质量分数不变，因为一致性梯度可更新共享视觉编码器。

模型没有新增权重、缓冲区或随机初始化模块；基线表示参数按原规则完整加载，评分头按原规则初始化。新结果独立报告，不替换原基线，不承诺训练逐位复现。

## 服务器命令

```bash
pwd
git status --short
git pull --ff-only origin work
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/run_align_local_tuning.py check-configs
```

终端一：三个较小参数值候选，每个两个数据集，共6次：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_local_tuning.sh low
```

终端二：三个较大参数值候选，每个两个数据集，共6次：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_align_local_tuning.sh high
```

GPU按可用情况选择。单卡12次：`CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_local_tuning.sh all`。分任务队列为`aux`、`match`、`context`，各4次。单次可用`bash scripts/run_align_local_tuning.sh alocalaux050 agiqa3k`。

只读计划：`bash scripts/run_align_local_tuning.sh all --dry-run`。不会读取大模型或创建结果目录。已有输出默认拒绝覆盖；显式`--continue`仅跳过核验通过的已完成运行，未完成目录保留并报错，不自动resume。

## 打包

12次全部完成后：

```bash
bash scripts/run_align_local_tuning.sh pack
```

输出`projection_only_v1_v2_align_local_tuning_results.tar.gz`，含候选/原基线CSV、完整配置、日志、元数据、实际划分和来源/完成指纹，不含模型权重，同名包拒绝覆盖。打包前核验实际初始化LR、有效损失权重、实际温度、来源加载记录、60轮及验证选模；不能跳过缺失证据。

## 验证记录

云端CPU合成验证：6项专项测试通过，最终全套461项unittest通过；12份配置及shell dry-run通过。验证旧辅助权重回退/等权算术不变、仅局部MSE系数变化、质量目标不变、实际Softmax匹配温度公式、温度1及模型初始化RNG/state_dict兼容、上下文优化器组成员及独立LR、12次模拟运行/打包、错误温度或有效局部监督记录拒绝、CPU真实启动拒绝。更新后的核查器还验证了此前18次真实已完成运行的原始结果包证据，旧结果保持兼容。未启动真实预训练/下游，不改旧基线或旧结果。

# 区域—词元匹配固定温度：1与0.1

独立版本 `projection_only_v1_v2_align_fixed_temperature_v1`，共4个新下游任务。云端仅CPU测试，不运行真实训练。不修改当前最好基线，不覆盖旧输出，不重新预训练。

## 唯一变量

- `tau100`：固定匹配温度1。
- `tau010`：固定匹配温度0.1。

均从上一轮legacy/tokenmean配置复制，保留CLIP区域/BPE词元路径、同样的词法子词mask（过滤纯标点；全标点Prompt回退并报告）、逐词元检索后普通均值汇总。温度在模型中是固定标量，不创建temperature_logit或任何温度优化参数。两组除温度和输出标识外，配置完全一致。

只比较上述两组。不能复用上一轮的“从1开始学习温度”结果冒充固定温度1对照；也不能用原ResNet局部路径的diag与固定0.1的CLIP区域路径比较来声称单独温度收益。

旧六维NSS保留原样，不使用峰度、GGD、五维或标准MSCN。pretraining来源固定为原最好基线SHA256 `fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686`，加载前后核验，完整representation迁移。两个区域/词元投影沿用此前明确新增随机初始化的方式；不混用其他预训练源。

projection_only_v1、5上下文、全局g→PI、P0绕过PI、质量路径及损失权重不变。不叠加softmin覆盖度、CLIP相似度分数参照、跨注意力、冻结或40+20阶段。新增标量温度无可训练参数；global_projection初始LR仍1e-5，PI/PT和一致性组仍1e-4，其余组不变。

3K和2023均seed42/split42，各跑60轮，保留train/val/test。2023质量Soft-SROCC权重0.04，3K0.02，其他参数取当前v2基线。1K无一致性标签，本轮不跑1K。对同一60轮轨迹分别按验证Quality SROCC和Quality/Align SROCC算术均值选模，不提前停止，不根据test选轮次。

## 诊断与核验

逐轮保存空间归一化熵、最大区域权重、实际固定温度、Prompt截断率、词元匹配均值/最小值、各一致性分支MAE、质量MSE和全部Q/A相关系数。上述指标用于判断匹配是否集中以及质量是否受损，不作为事后更换选模规则的依据。

完成核验包括：60轮和双选模记录；实际处理批次、更新/AMP跳过计数；连续60轮余弦LR；源SHA及配置快照；四个所选模型的温度记录；无新增温度/校准参数；两组区域/词元投影的初始SHA、参数组/参数量/LR、实际划分成员匹配。不承诺CUDA训练逐位复现，也不预设0.1一定更好。源权重之外的两套新投影在不同温度组应具有相同初始参数。

CPU验证包含解析softmax公式、实际区域—词元前向路径和梯度、固定标量无梯度、同权重质量和全局一致性输出相等、初始张量/RNG和参数组一致、有效/无效产物核验、无权重打包及图像渲染。

## 服务器命令

默认Python为`/home/sll/anaconda3/envs/WZX-AI/bin/python`，可用ABLATION_PYTHON覆盖。

两卡后台完成全部4次下游：GPU0按顺序跑tau=1的3K/2023，GPU1按顺序跑tau=0.1的3K/2023。启动前只读核验来源、历史对照、已有任务状态，失败不启动任何队列。正常VS Code断连后继续；服务器重启仍会终止。

```bash
git pull --ff-only origin work && bash scripts/launch_align_fixed_temperature_background.sh
```

启动器会打印独立日志目录、PID及tail命令。不操作其他人的任务；不要重复启动。已完整完成且核验通过的任务跳过；不完整任务明确停止保留，不覆盖、不自动从头重跑，不把中断说成可恢复。

单卡顺序跑全部，或只跑一个温度/数据集：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_fixed_temperature.sh all --continue
CUDA_VISIBLE_DEVICES=1 bash scripts/run_align_fixed_temperature.sh tau010 agiqa3k --continue
```

只预览四组配置，不要求权重或CUDA：

```bash
bash scripts/run_align_fixed_temperature.sh all --dry-run
```

只读完成检查（全部完成才退出码0；活动任务会报告PID）：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/preflight_align_fixed_temperature.py --require-complete
```

完成后打包：

```bash
bash scripts/run_align_fixed_temperature.sh pack
```

生成`projection_only_v1_v2_align_fixed_temperature_v1_results.tar.gz`，包含四个新下游和两个历史对照的CSV、配置、日志、元数据、完成记录、来源和实际划分；不含大权重。已有同名包时拒绝覆盖。新温度不是模型参数，不存在“温度权重漏载”。

生成验证曲线PNG/PDF（新输出目录，不重新推理或选模）：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/plot_align_fixed_temperature.py --output run_logs/fixed_temperature_figures_v1
```

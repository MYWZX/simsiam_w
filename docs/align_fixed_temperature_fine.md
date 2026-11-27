# 固定温度小范围对照：0.05 / 0.1 / 0.2

独立版本`projection_only_v1_v2_align_fixed_temperature_fine_v1`。新增`tau005=0.05`、`tau020=0.2`，各跑3K/2023，共4次新下游。已有0.1作为必需对照，核验并复用，不重训。没有重新预训练，不修改已接受的最好基线，不覆盖上一轮结果。

## 固定协议

与上一轮固定0.1配置相比，只改变温度和运行标识/输出路径。区域/BPE词元路径、词法子词mask、普通均值汇总不变。使用旧六维NSS和原预训练来源`fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686`，不引入峰度、覆盖度、CLIP分数参照或跨注意力。

温度始终是固定标量，没有温度参数/梯度。模型代码、损失和训练器不变；两套新投影保持此前明确的新初始化方式。global_projection初始LR1e-5，PI/PT及一致性组1e-4，其他分组/参数量不变。两数据集均seed42/split42、60轮、不合并train+val，分别按验证Quality SROCC与Quality/Align SROCC算术均值选模；test不参与选模。2023质量Soft-SROCC系数0.04，3K0.02。1K没有一致性标签，本轮不训练1K。

验证器要求：固定温度全程正确，60轮连续LR和实际处理/更新/AMP跳过记录完整；新增两组与彼此及旧0.1对照的投影初值SHA、优化器分组/参数量/LR、实际划分成员一致。0.1对照缺失、不完整或核验失败时不启动新实验，也不自动重训或补造完成标记。

同一seed的CUDA实验不承诺逐位复现；本次只是围绕0.1做相邻温度探索，不预设更低温度一定更好。最终比较质量/一致性全部SROCC、PLCC及空间熵、区域最大权重、质量损失，而不是只挑一个test指标。

## 服务器运行

默认Python`/home/sll/anaconda3/envs/WZX-AI/bin/python`，可通过ABLATION_PYTHON覆盖。

双卡后台：GPU0跑0.05的3K→2023，GPU1跑0.2的3K→2023。先只读预检两集旧0.1对照、源SHA和已有新目录；失败不启动队列。已有完整任务核验后跳过，半成品保留并停止。不自动删除/重跑中断目录，不操作其他训练进程。

```bash
git pull --ff-only origin work && bash scripts/launch_align_fixed_temperature_fine_background.sh
```

正常VS Code断连后仍继续，服务器重启仍会终止。启动器会打印独立PID、日志路径和tail命令，不要重复启动同一队列。

单卡或单任务运行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_fixed_temperature_fine.sh all --continue
CUDA_VISIBLE_DEVICES=1 bash scripts/run_align_fixed_temperature_fine.sh tau020 aigciqa2023 --continue
```

只预览配置、不读取权重、不启动训练：

```bash
bash scripts/run_align_fixed_temperature_fine.sh all --dry-run
```

只读检查完成状态，全部四次新任务完成并通过核验时退出码0：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/preflight_align_fixed_temperature_fine.py --require-complete
```

## 打包与画图

完成后打包：

```bash
bash scripts/run_align_fixed_temperature_fine.sh pack
```

生成`projection_only_v1_v2_align_fixed_temperature_fine_v1_results.tar.gz`。包含四次新运行和两次旧0.1对照（合计6个CSV）的配置、日志、元数据、完成记录、源指纹与实际划分，以及单独的fine协议文件。只核验/读取旧对照；不启动或改写旧任务。包内不含大模型权重；同名包存在则拒绝覆盖。历史整体基线指标另外保留在current_tuning_reference记录中。

生成0.05/0.1/0.2三条曲线的PNG/PDF：

```bash
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/plot_align_fixed_temperature_fine.py --output run_logs/fixed_temperature_fine_figures_v1
```

图包含验证Q/A SROCC、质量MSE、空间熵、区域最大权重、固定温度。仅读取日志，旧0.1曲线从原目录读取，不重新推理或选模；输出目录必须是新目录。

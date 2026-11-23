# 一致性分支三组独立消融

固定已接受原基线预训练SHA `fa702194827ea2023375cef8dfec6014e9dd91fd80207a00ecebbb7ae6e34686`，仅下游3K、2023，每组各2次，共6次。原NSS六维、projection_only_v1、5上下文、训练/划分seed42、60轮、按max val_quality_SROCC选模，Quality和所有任务损失系数及学习率恢复原基线（align LR1e-4，rank_align=0.03，srocc_align=0.04，PI/PT1e-4，全局投影1e-5），不继承前批调参候选，不改变质量路径，不重新预训练。

| 变体 | 唯一因素 | 实际路径 |
|---|---|---|
| aglobal | 最终只输出全局一致性分数 | A=A_global，γ=1；跳过并冻结align_gate |
| ahalf | 固定全局/局部分数各50% | A=0.5*(A_global+A_local)，γ=0.5；跳过并冻结align_gate |
| auniform | 去掉文本引导的空间加权 | 每个局部位置权重1/N，直接均值池化AP输出；最终仍用原学习门控 |

三组都保持`lambda_align_aux=0.3`，全局/局部辅助MSE都训练。**aglobal是输出路径消融，不是整个局部分支训练消融**：局部特征、AP、局部评分头仍执行、仍从辅助MSE获得梯度。不会把该结果解释为完全移除局部路径。

uniform仅移除局部匹配投影、点积和空间softmax；Prompt仍进入AT以及局部评分头的图文拼接，所以不是去掉所有文本条件。未执行的两个匹配Linear保留原结构和初始化以兼容检查点，明确冻结并从优化器排除。门控固定组同理。QA/QL、NSS、质量分数混合及局部质量池化均不变。

新配置`model.align_branch_control`显式记录`version=alignment_branch_v1`、`output=global/half/learned`、`local_pool=text/uniform`；缺省仍是原学习门控+文本加权。每个实验只改一个因素，不支持同时改变两条路径；额外门控监督、文本评分bank、跨注意力、V-Mamba或预训练模式与这些控制组合时明确报错。

模型不增加权重或缓冲区；表示加载沿用完整旧基线来源，评分头仍按原规则初始化。保留相同构造顺序，不创建随机替代网络。运行器通过快照、真实运行诊断、常数门控记录、初始化LR组、实际成员表、有效目标、来源指纹、60轮及同轮验证选模核验。固定门控/均匀池化会移除部分计算，可能改变训练随机数消耗，属于独立代码版本，不承诺训练逐位相同，不能把旧成绩当新模式结果。

## 服务器

```bash
pwd
git status --short
git pull --ff-only origin work
/home/sll/anaconda3/envs/WZX-AI/bin/python scripts/run_align_branch_controls.py check-configs
```

终端一：全局输出和固定半混合，两组各跑3K/2023，共4次：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_branch_controls.sh gate
```

终端二：均匀池化，依次跑3K/2023，共2次：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_align_branch_controls.sh pool
```

GPU编号按可用情况调整。单卡六次：`CUDA_VISIBLE_DEVICES=0 bash scripts/run_align_branch_controls.sh all`。

单次：`bash scripts/run_align_branch_controls.sh aglobal agiqa3k`；另外两组为`ahalf`、`auniform`。只读计划：`bash scripts/run_align_branch_controls.sh all --dry-run`。默认拒绝已有输出；显式加`--continue`仅在完成核验通过后跳过，未完成目录仍保留并报错，不自动resume。

六次全部完成后：

```bash
bash scripts/run_align_branch_controls.sh pack
```

输出`projection_only_v1_v2_align_branch_results.tar.gz`。含候选及原基线CSV、配置、运行/来源/完成证据、日志和成员清单；不含模型权重，不覆盖旧包。缺任一阶段证据则拒绝打包，不重建完成记录。

## 验证

云端CPU合成验证：7项专项测试通过，最终全套455项unittest通过。验证同权重/同输入下质量评分与全局一致性分支数值不变、构造RNG与state_dict结构相同、门控/匹配投影确实未执行、局部辅助损失梯度保留、固定混合与均匀池化公式正确、基线表示完整加载、真实evaluate输出控制诊断、错误组合拒绝、六次模拟完成与打包核查。六份配置及shell dry-run通过。未启动真实预训练/下游，未改动原基线或已有结果。

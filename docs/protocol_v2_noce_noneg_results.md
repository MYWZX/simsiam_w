# Protocol v2 下游 noce_noneg 结果

来源：用户上传 `cliplarge_protocolv2_noce_noneg_results.tar.gz`，对照上一包 protocol v2 baseline。SR=SROCC，PL=PLCC；测试结果均来自验证质量 SR 选中模型，不按测试另挑 epoch。

## 审计

- 三组均有一次 run_start/final，epoch 连续 1–60。final 的 best_val 与选中轮日志一致。
- 每轮质量五档 CE、旧错配 Prompt 损失均为 0。Quality text bank 关闭，lambda_quality_text_ce=0，lambda_quality_text_anchor=0，lambda_neg=0。其余科学配置与 baseline 相同；1K 原本 lambda_neg=0，本次该数据集主要改变的是关闭质量文本辅助分支。
- 三组来源 SHA256 均为 `9655e75a1fa05a8a33500cddeb3e8dd9530d70bc243209ef704b0f0a5f92e5bc`，checkpoint_verified_after_load=true，提交为 `33ea18cdd7003a74c7d552c36a549844f3c32496`。
- **三组 workspace_dirty=true**，而上一轮 baseline 为 false。包中没有运行时 git diff/status 文件，无法确认额外改动范围；不应把它们描述成已证实的完全干净代码对照。配置快照的变化符合预期，实际代码是否另有修改尚未独立核验。
- 所有 JSON 数值未发现 NaN/Inf。累计 AMP 跳步为 1K 4、3K 8、2023 7，分布与 baseline 一致。
- 预训练没有重跑，仍使用带质量五档 CE 训练过的来源。本组是下游关闭 CE，不是两阶段均无 CE。

## 结果

| 数据集 | 设置 | 轮次 | 验证Q SR | 测试Q SR | 测试Q PL | 测试A SR | 测试A PL |
|---|---|---:|---:|---:|---:|---:|---:|
| agiqa1k | baseline | 23 | 0.893586 | 0.877712 | 0.879750 | — | — |
| agiqa1k | noce_noneg | 32 | 0.896386 | 0.880167 | 0.878391 | — | — |
| agiqa3k | baseline | 60 | 0.897960 | 0.897005 | 0.932850 | 0.793943 | 0.879062 |
| agiqa3k | noce_noneg | 37 | 0.900799 | 0.895559 | 0.932731 | 0.807961 | 0.885542 |
| aigciqa2023 | baseline | 53 | 0.837244 | 0.870584 | 0.885429 | 0.762946 | 0.754777 |
| aigciqa2023 | noce_noneg | 51 | 0.824739 | 0.866585 | 0.882475 | 0.792772 | 0.792322 |

## 相对 baseline 的变化

| 数据集 | 验证Q SR变化 | 测试Q SR变化 | 测试Q PL变化 | 验证A SR变化 | 测试A SR变化 |
|---|---:|---:|---:|---:|---:|
| agiqa1k | +0.002801 | +0.002455 | -0.001358 | +0.000000 | +0.000000 |
| agiqa3k | +0.002838 | -0.001446 | -0.000120 | +0.029913 | +0.014018 |
| aigciqa2023 | -0.012505 | -0.004000 | -0.002954 | +0.021345 | +0.029826 |

三数据集验证质量 SR 等权平均：

- baseline: 0.876263192
- noce_noneg: 0.873974776

不能把结果概括为全面提升。1K/3K 验证质量排序略升，2023 验证质量 SR 和 PL 都下降；3K/2023 验证和测试 Align SR 均上升，呈现任务取舍。单种子小幅变化尚不证明稳定收益。3K/2023 同时关闭 CE 和旧 negative，不能区分二者贡献；关闭整个 bank 也改变了模块初始化 RNG 消耗，不能把差异完全归因于某一个梯度项。

泛化差距依然存在：noce_noneg 所选轮在线训练Q PL 分别 .987423、.995231、.995813，验证Q PL 分别 .899491、.928928、.868485。训练指标来自训练过程中累计，并非最终模型的全训练集 eval。尚无证据表明已解决过拟合。

## 下一步

按用户已确定的后续约束，保持下游 CE=0、旧 negative=0、Soft-SROCC 不变；不因看到测试结果重新恢复 CE。该组仍需先核对 dirty 工作区的实际代码改动，再作为固定预算重训来源。服务器可只读检查：

```bash
git status --short
git diff --stat
git diff --cached --stat
```

不要为了清理状态而直接丢弃文件。当前检查只能展示当前工作区，不能重建训练当时已被撤销的修改；若确认只是无关文件，可记录原因。若是训练/模型/数据代码修改，应先归档并核实真实配置和代码。

若代码差异核实无关，按新开发结果生成预算，不使用旧 baseline 预算：

| 数据集 | 验证选中轮 | 处理batch预算 |
|---|---:|---:|
| agiqa1k | 32 | 3008 |
| agiqa3k | 37 | 9620 |
| aigciqa2023 | 51 | 10710 |

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
for ds in agiqa1k agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh plan noce_noneg "$ds"
  bash scripts/run_protocol_v2.sh refit noce_noneg "$ds"
done
```

这是预先规划的增加训练样本实验，不表示 noce_noneg 已证明优于 baseline，也不保证最后模型是所有中间模型中最优。原测试索引保持不变，训练结束后只测试一次，80/20 协议结果独立报告。

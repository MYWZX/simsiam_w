# 全局/局部双质量表示：联合辅助监督与完整迁移

这是独立的新架构，不覆盖双g基线或noce/mos_pq/shared_pi_pt三个独立消融。配置开关为model.dual_quality_representation.enabled，默认关闭。

## 对应模型

```text
全局：CLIP → semantic_proj → g → PQ/QA → normalize → qg → 全局质量头 → Qg
                                                                          \
                                                                           → Q → 整图质量MOS
                                                                          /
局部：ResNet layer3 → L → P0 → + NSS投影 → 局部质量编码器 → normalize → 局部tokens
                                                            ↓
                                                NSS gate → softmax权重
                                                            ↓
                                            加权汇总 + normalize → ql → 局部质量头 → Ql

qg ──→ (1-eta) ──┐
                 ├→ 相加 → normalize → q → Component Anchors / Ordinal / 质量五档CE
ql ──→ eta ──────┘

最终：Q = (1-beta) Qg + beta Ql
```

eta和beta是不同参数：eta融合质量特征，beta融合质量分数。两者均由sigmoid参数化、初始0.5。eta从预训练迁移到下游；beta与数据集评分头一起重新学习。

全局qg只来自CLIP的g，不使用融合z；局部ql只来自ResNet和图像NSS，不接收g或文本。局部质量编码器采用小型ResidualAdapter：x+0.1*MLP(LayerNorm(x))，默认256→256→256、GELU、dropout0.1，再归一化。它处理每个局部token；NSS gate产生softmax权重，汇总为整图局部质量表示ql。标签监督的是整图汇总结果，不把整图MOS分配给每个patch。

辅助监督的q=normalize((1-eta)qg+eta*ql)。因此Component Anchors、Ordinal和五档CE都能经qg训练CLIP/PQ，经ql训练ResNet、L、NSS投影、局部编码器、NSS gate，同时学习eta。下游五档CE也使用相同的q；Anchors/Ordinal仍只在预训练使用，与已有下游任务定义相同。

注意本方案的局部评分为head(ql)，不是旧方案中逐patch预测分数再加权。CSV为兼容现有分析继续使用quality_patch_*记录这个局部整图评分，不能把它当作已监督的逐patch质量图。NSS权重仍可单独分析。

## 预训练/下游对应

- 全局PQ与下游QA是同一结构：256→512→512→256的MLP，含BN/ReLU/Dropout，输出L2归一化。
- 为使预训练/下游图像质量路径一致，PQ在预训练中直接注册为iqa.quality_adapter，保存到表示checkpoint；下游加载quality_adapter。不再额外新建第二个相同PQ，也不复制注册同一个模块。
- 全局质量头直接消费qg，使用replace模式，**没有旧residual_feature中g绕过QA的旁路**。这对应用户批准的“qg→全局质量头”方案，是独立架构变化。
- SafeAug I-I仍保留：两张SafeAug的g经过同一个PQ/QA及质量predictor；融合后的Clean q用于Anchors/Ordinal/CE。Clean只调用一次PQ和bank，避免重复BN/Dropout前向。
- 原图文SimSiam、文本路径及预训练Align路径保留，不叠加共享PI/PT实验。下游Align继续使用实验①的g与原局部图文匹配。Quality局部编码器不改写Align的P0。
- 原Artifact/Aesthetics MOS辅助头继续通过qg、ql及二者差值拼接学习；最终质量MOS监督Q，两项质量评分头、局部NSS路径和PQ一起训练。

## 为什么两阶段统一layer3

本次必须迁移L和局部质量编码/汇总，故预训练、下游都使用ResNet layer3，224输入对应14×14局部特征、196个token。禁止--reinit-local-projection，禁止跳过QA或只加载backbone。原layer4→layer3实验会重置局部投影，不能作为本方案的完整迁移。

模型检查点带dual_quality_signature。加载时要求完整恢复PQ/QA（含BN缓冲）、ResNet局部编码器、L、NSS投影、局部质量编码器、NSS gate及eta，缺失/不兼容会报错。旧架构与新架构不能互换检查点。

表示checkpoint还保留原有CLIP/Fusion/文本contexts/U/Align Adapters/质量文本bank等模块；数据集相关全局和局部质量评分头、Artifact/Aesthetics头及最终分数融合beta不迁移。原实验的迁移白名单保持原样，仅本模式新增局部表示参数。

本次有多个结构变化，又从layer4→layer3切到匹配layer3；不能把与旧实验①的分差单独归因于某一个模块。已有layer3→layer3实验可作为背景对照，但与其相比全局输入和质量结构也有变化。

## 固定设置

Large、seed/split42、4个可学习contexts；预训练50轮上限/15轮早停、原PCGrad开启、官方RichHF划分，仍以val_component_anchor_SROCC选择主迁移checkpoint。质量五档CE权重0.005、blend0，原Component Anchors/Ordinal/MOS目标保留。除显式质量结构、匹配局部层及对应权重迁移外，学习率/损失系数沿用现有配置。下游Quality-safe输入，无perturbation/旧negative；不启用逐图Quality门控。1K按质量验证选模；3K/2023比较两种验证策略，主结果仍取joint_selected。

不改变为关闭PCGrad、降低骨干LR或降低CE权重的其他实验设置。

## 运行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work

# 必须先完成新预训练，不能使用旧checkpoint
bash scripts/run_dual_quality.sh pretrain

# 然后分别微调
bash scripts/run_dual_quality.sh agiqa1k
bash scripts/run_dual_quality.sh agiqa3k
bash scripts/run_dual_quality.sh aigciqa2023
```

可在任意命令末尾加--dry-run查看完整参数。Python默认/home/sll/anaconda3/envs/WZX-AI/bin/python，可通过ABLATION_PYTHON覆盖。不指定GPU编号。若同名输出已存在，启动器拒绝覆盖或追加；如要恢复预训练，请使用现有train_cm_pretrain.py --resume参数与同一配置。

新预训练目录：outputs/richhf18k_cliplarge_dualquality_layer3_seed42。预训练保留完整恢复检查点与best_component_anchor_pretrain.pth表示权重；下游不保存pth。

## 诊断

CSV新增quality_feature_local_weight（eta）及quality_feature_global_weight（1-eta），下游带val_/test_前缀。quality_mix_*仍表示beta。eta下降说明融合质量辅助表示更依赖全局，不能把它解读为最终质量分数中局部贡献的比例；同样beta不是预训练知识贡献百分比。梯度路径连通不保证每阶段都能得到相等的学习量，应结合eta和分支指标判断是否出现偏向。

## 打包结果

完成四次实验后，打包配置、日志和CSV，排除大型权重：

```bash
files=(
  outputs/richhf18k_cliplarge_dualquality_layer3_seed42
  outputs/all_results_richhf18k_cliplarge_dualquality_layer3_seed42.csv
)
for dataset in agiqa1k agiqa3k aigciqa2023; do
  selection=""
  [[ "$dataset" == agiqa1k ]] || selection="_dualselect"
  name="${dataset}_quality_cliplarge_layer3_dualquality${selection}_seed42"
  files+=("outputs/$name" "outputs/all_results_${name}.csv")
done
tar --exclude='*.pth' --exclude='*.pt' -czf cliplarge_dualquality_layer3_results.tar.gz "${files[@]}"
```

代码验证不等于真实数据/GPU训练完成，性能需要运行上述实验后评估。

## 已完成的代码验证

142项CPU回归测试通过。新测试逐项验证Artifact/Aesthetics Anchor、Ordinal及质量CE都能更新CLIP、ResNet、全局PQ、局部编码器、NSS gate与eta；全局/局部评分的输入依赖相互分离；Align前向保持原路径；局部参数及BN缓冲完整迁移、损坏检查点报错；还覆盖三视图真实训练循环的一批次PCGrad更新、下游CE梯度、完整state_dict恢复和四条启动命令的有效配置。

上述训练循环使用小型替代视觉/文本编码器及真实Fusion/质量/Align模块，不是RichHF或下游数据集的GPU训练结果。

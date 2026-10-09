# 下一步：先补诊断对照，不改变正式方法

依据：outputs/project_experiment_audit_20260929.md 的已有 CSV 描述性结果，以及科研问题澄清。
不是新的训练结果，不宣称统计显著性或超过 DPGF-Net。

## 本轮范围

- RQ1：完整四路普通 SimSiam vs 四路普通损失权重全0。
- RQ3：同一预训练 checkpoint，replace / parallel_score × pretrained / random。
- RQ2 的细粒度关系重构本轮不实现；当前已经有区域/词级 pull–push，先避免重复叠加监督。
- 所有原 YAML 保留。正式方案保留 SimSiam，关闭项只作诊断对照。

## 控制与边界

四路权重全0只覆盖 lambda_img_siam、lambda_img_answer、lambda_img_prompt、
lambda_prompt_answer；若旧配置有图像–图像系数别名 lambda_quality_view_siam，也一并清零。
Quality/Align 评分、辅助头、Component Anchor/Ordinal、Map/词级
损失与 PCGrad 保持原配置；模型和 forward 也保持，因此这是“常规四路损失增量”对照，
不是完全移除一切 Siamese 风格监督。原始 loss 日志仍可非零，但这些四项对总梯度贡献为0。

两组从头以同一 seed 训练，均固定17轮，关闭早停但保持原50轮 cosine 学习率轨迹。
17轮是控制计算预算的探索设置，不是最优轮数结论。可统一改 --fixed-epochs，不应只改一组。
不直接拿旧 early-stop checkpoint 与新固定预算对照作单因素结论。
两组统一使用 best_quality_pretrain.pth（验证质量分数选择），不根据下游测试挑 checkpoint。
下游 split_seed 沿用源配置42；CLI seed只改变训练随机性，预训练seed由 --pretrain-seed 独立指定。

random 保留相同 Adapter、可训练参数及其随机初始化 BN 状态，只排除该模块的checkpoint加载。
其余编码器、Fusion、文本投影和 Align Adapter 仍照常迁移，预测头仍按 representation 协议重置。
parallel_score 是两个质量分支的分数融合，不是 Quality + Align 的合成 Final MOS；
不同模式参数量不同，所以同模式 pretrained/random 对照才用于区分权重知识与容量。
不同模式仍会改变结构初始化的随机数消耗；不能把不同模式的差异直接全归因于质量知识。

不指定GPU、不另存完整/瘦身权重副本、不覆盖现有输出。新结果单独写在 --output-root 下，
训练脚本自动保存有效 config_snapshot.yaml 和 CSV config_json。
预训练补记 final 完成事件；失败停止，不自动重试，重跑须选新 output-root。
不加 --execute 只打印命令，不写文件、不训练。

## 运行指令

项目目录：/home1/wzx/aigc_naturalness_code/simsiam。
以下均用 /home/sll/anaconda3/envs/WZX-AI/bin/python。

1. RQ1 两组预训练（依次运行）：

    HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/run_rq_diagnostics.py pretrain --fixed-epochs 17 --execute

2. 预训练完成后，各数据集两检查点的下游：

    HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/run_rq_diagnostics.py downstream --dataset 1k --execute

    HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/run_rq_diagnostics.py downstream --dataset 3k --execute

    HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/run_rq_diagnostics.py downstream --dataset 2023 --execute

3. RQ3 先用已有 best_quality checkpoint 测1K，依次执行四种条件，无需再预训练：

    HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/run_rq_diagnostics.py adapter --dataset 1k --execute

## 如何解释

首轮只用一个训练seed探索；保留有效单seed候选，同时保存失败条件，不声称多seed稳定性。
分别报告 Quality/Align 的 PLCC、SROCC，不用合成 Final 指标替代它们。
质量“不能明显退步”的 PLCC/SROCC 非劣界值尚未确定，必须在看新测试结果前确定，
目前不根据结果倒推阈值。不加任何“只有提升的seed才写入”的过滤。
RQ1若完整方案明确劣于关闭对照，只能否定当前常规四路的收益假设，不要求删除正式方案的SimSiam。
RQ3若随机Adapter与预训练Adapter相当，不能把并行结构的收益全部解释为预训练质量知识。
DPGF-Net 的划分、标签、聚合方式仍需核验，本轮不作公平 SOTA 排名声明。

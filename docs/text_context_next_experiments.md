# 文本前缀下一步诊断（2026-10-01）

本文件是实验协议，不是结果。依据为本项目现有代码、配置快照与 seed42 下游 CSV；
不引入新人工标注或外部数据。保留四路 SimSiam 为正式方法；以下只测试文本前缀
和 Align 对主体/动作错误的敏感性。正式指标仍分别为 Quality、Align 的 PLCC/SROCC。

## A. 现有检查点的 Prompt 错配探针（无需训练）

`tools/diagnose_prompt_sensitivity.py` 在同一张图上比较原 Prompt 与只替换一个
主体词或动作词的 Prompt。输出 `gap = Align(original) - Align(changed)`；
`positive_gap_rate` 是 gap>0 的比例。仅选人工 Align>=0.4 的样本，候选覆盖率另报。
该阈值只是预先固定的探针范围，不是正确/错误界限。替换文本没有人工复核，
可能仍描述图片或语法不自然，因此 **gap 不是准确率，不可代替人工细粒度标签**。
先查看 `*.pairs.csv` 的正负文本，剔除无效候选后才解释结论；
验证集可用于探索，测试集不可反复挑模型。

在项目根目录执行（不指定 GPU；脚本自动选设备）：

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/diagnose_prompt_sensitivity.py \
  --checkpoint baseline=outputs/agiqa3k_cm_iqa_decoupled_quality_align_seed42/best.pth \
  --checkpoint context4=outputs/agiqa3k_decoupled_text_context4_seed42/best.pth \
  --split val --output-prefix outputs/prompt_sensitivity_3k_context4_val

HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python tools/diagnose_prompt_sensitivity.py \
  --checkpoint baseline=outputs/aigciqa2023_cm_iqa_decoupled_quality_align_seed42/best.pth \
  --checkpoint context4=outputs/aigciqa2023_decoupled_text_context4_seed42/best.pth \
  --split val --output-prefix outputs/prompt_sensitivity_2023_context4_val
```

脚本只做推理，读取检查点同目录的 `config_snapshot.yaml`，核对数据配置与划分；
拒绝覆盖已有 CSV。对 3K/2023，初步词表覆盖有限，尤其动作词；不能只看均值。

## B. 隔离初始化随机数的配对复核（需要训练）

旧 N=4 前缀初始化会额外消耗随机数，改变后续模块初值。新增
`--isolate-text-context-rng` 只隔离这一步 CPU RNG 消耗；默认关闭，
所以旧实验路径不变。seed123 的 N=0 RQ1 full 已有固定17轮基线，
使用相同配置、相同 split seed42 与 `best_quality_pretrain.pth` 选点。
N=4 实验与其比较；不要与早停的 seed42 前缀实验混作单变量对照。

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_pretrain.py \
  --config configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml \
  --seed 123 --fixed-epochs 17 --text-context-tokens 4 --isolate-text-context-rng \
  --output-dir outputs/rq_diagnostics_v1/rq_text_context4_isolated_pretrain_seed123 \
  --results-csv outputs/rq_diagnostics_v1/all_results_rq_text_context4_isolated_pretrain_seed123.csv
```

完成并确认检查点存在后，三个下游使用同一个新预训练检查点；各自沿用源 YAML 的
`split_seed: 42`。正在运行的 RQ1 seed123 下游不要重启、覆盖或与本组共用输出目录。

```bash
for spec in 'agiqa1k agiqa1k' 'agiqa3k agiqa3k' 'aigciqa2023 aigciqa2023'; do
  set -- $spec
  HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python train_cm_iqa.py \
    --config "configs/${1}_cm_iqa_decoupled_quality_align.yaml" \
    --seed 123 --text-context-tokens 4 --isolate-text-context-rng \
    --pretrained-checkpoint outputs/rq_diagnostics_v1/rq_text_context4_isolated_pretrain_seed123/best_quality_pretrain.pth \
    --output-dir "outputs/rq_diagnostics_v1/${2}_text_context4_isolated_seed123" \
    --results-csv "outputs/rq_diagnostics_v1/all_results_${2}_text_context4_isolated_seed123.csv" || break

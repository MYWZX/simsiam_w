# 独立新实验：预训练和下游均合并 train+val

入口：`scripts/run_allmerged_noce_noneg.sh`。保留此前 `run_two_stage_noce_noneg.sh` 的行为及输出，本实验使用 `allmerged_noce_noneg` 名称和新目录。

## 一键完整执行

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_allmerged_noce_noneg.sh
```

默认使用 `/home/sll/anaconda3/envs/WZX-AI/bin/python`，可通过 `ABLATION_PYTHON` 覆盖；不指定 GPU 编号，设置 HF_HUB_OFFLINE=1。所有阶段自动串行，任一步失败就停止，拒绝覆盖已有输出。

## 执行顺序

1. **RichHF 原划分开发预训练**：关闭质量五档 CE 和负样本推远项，按验证 component-anchor SROCC 选停止点。记录每轮处理 batch/成功更新/AMP 跳过，以及原三份划分的图像身份、CSV SHA256。仍保留 MOS、Component Anchors/Ordinal、四路 SimSiam、正例局部/短语对齐；其余参数与先前两阶段 noce_noneg 相同。
2. **RichHF 合并预训练**：核对原划分/标签来源没有变化、三份划分无图像交集；将原 train 与原 val 合并，原 val 样本也使用训练变换。保留原 uniform replacement 采样方式，扩展采样总体，不引入原 test。重新从初始 CLIP/ResNet 编码器开始，不加载开发训练好的模型。使用选中轮累计的原训练 batch 数作预算，按原余弦轨迹换算 LR，最后一轮可以截断。此阶段没有验证选模，也不评估 RichHF 测试集。
3. **保存固定结束点预训练表示**：`final_merged_pretrain.pth`，不是根据合并数据再挑选的 best。
4. **下游开发**：三个数据集分别从同一份 `final_merged_pretrain.pth` 初始化，使用原 70/10/20 划分、验证质量 SROCC 确定各自预算。
5. **下游合并重训和测试**：各自合并 train+val，约 80% 训练，保留原 20% 测试索引。从合并预训练来源重新初始化，不继续训练下游开发模型。按固定预算结束，只做一次最终测试，不用测试早停或挑权重。

预训练的合并比例由 RichHF 官方实际 train/val/test 样本数决定，不称为固定 80/20；下游三个数据集保持分别训练，不互相拼接。

新负例开关、CE 关闭与之前 `both_noce_noneg` 相同。Soft-SROCC 保留原系数（预训练中本来为 0 的项仍为 0）；预训练 PCGrad 保留。下游旧 negative=0；共享 PI/PT、QA、NSS 局部质量、Align 分支等保留。

## 文件位置

```text
# 原划分开发预训练
outputs/richhf18k_allmerged_noce_noneg_pretrain_dev_seed42.csv
outputs/richhf18k_allmerged_noce_noneg_pretrain_dev_seed42/config_snapshot.yaml

# 真正合并后的预训练来源；下游统一加载这里
outputs/richhf18k_allmerged_noce_noneg_refit_seed42/final_merged_pretrain.pth
outputs/richhf18k_allmerged_noce_noneg_refit_seed42/refit_plan.json
outputs/richhf18k_allmerged_noce_noneg_refit_seed42.csv

# 下游开发 / 最终合并结果，dataset 为 agiqa1k/agiqa3k/aigciqa2023
outputs/<dataset>_allmerged_noce_noneg_dev_seed42.csv
outputs/<dataset>_allmerged_noce_noneg_refit80_seed42.csv
outputs/allmerged_noce_noneg_plans/<dataset>.json
```

看最终成绩时取下游 `refit80` CSV 的最后一条 `final`，其 `metrics_json.test` 为测试指标。下游依旧不保存 pth；预训练保存最终迁移表示。未修改此前实验的权重、结果或停止预算。

## 分阶段与中断

```bash
# 只运行原划分预训练，完成后停止
bash scripts/run_allmerged_noce_noneg.sh pretrain-dev
# 已有本新实验的开发预训练，再运行合并预训练
bash scripts/run_allmerged_noce_noneg.sh pretrain-refit
# 新合并预训练已经完成、下游尚未开始
bash scripts/run_allmerged_noce_noneg.sh downstream
```

若下游已有部分阶段完成，应从对应未完成阶段执行，不要重跑整个 downstream 覆盖结果。例如 3K 开发尚未运行：

```bash
python_bin=/home/sll/anaconda3/envs/WZX-AI/bin/python
HF_HUB_OFFLINE=1 "$python_bin" train_cm_iqa.py --config configs/allmerged_noce_noneg/agiqa3k.yaml
"$python_bin" scripts/prepare_protocol_refit.py \
  --snapshot outputs/agiqa3k_allmerged_noce_noneg_dev_seed42/config_snapshot.yaml \
  --csv outputs/agiqa3k_allmerged_noce_noneg_dev_seed42.csv \
  --prefix outputs/allmerged_noce_noneg_plans/agiqa3k
HF_HUB_OFFLINE=1 "$python_bin" train_cm_iqa.py \
  --config outputs/allmerged_noce_noneg_plans/agiqa3k.yaml \
  --refit-plan outputs/allmerged_noce_noneg_plans/agiqa3k.json
```

合并预训练入口仅发布结束点表示，不支持中途恢复 optimizer/RNG；中断时保留已有日志，不要当作已完成结果。脚本不会自动删除或覆盖这些目录。

## 限制与检查

- 固定预算不是“保证结束模型全程最好”；它是先用独立验证确定训练长度，再增加训练数据的协议。
- 预算按处理 batch/更新尝试记录，成功更新和 AMP 跳过另记；合并数据后采样/epoch 边界改变，不能保证轨迹相同。
- 当前合并预训练明确支持单一官方划分 RichHF、uniform replacement sampling；对混合数据源、关系/平衡/重加权采样拒绝运行，避免无声改变协议。
- 只核查图像路径身份和 CSV 内容指纹，不对全部图像字节做哈希，也不宣称排除不同文件名的近重复图像。
- 本地测试覆盖原测试独立、集合交集拒绝、val 训练变换、标签来源变化、验证选预算、真实训练 callback、精确部分 epoch 停止、固定结束点 checkpoint 发布，以及原下游单次测试协议。无真实数据/GPU，未代跑训练。

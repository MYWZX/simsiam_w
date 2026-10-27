# 修正协议、独立防过拟合对照与 80/20 固定预算重训

本版本从双质量残差 layer3、共享 PI/PT、PI/PT BN 统计冻结基线出发。预训练与下游都使用 layer3；Quality/Align 全局路径都使用 g，P0 保留经过共享 PI 的路径。旧配置、旧权重、旧结果不覆盖。这里没有宣称新的实验成绩，真实训练需要服务器数据和 GPU。

## 1. 先建立修正后的共同基线

- 强制视觉、文本 CLIP 使用 QuickGELU，与 OpenAI 权重匹配。必须重新预训练一次，再让所有下游共享这个新来源；旧 GELU 权重不混用。表示检查点新增激活签名，跨协议迁移会报错。
- AGIQA-1K 的 min/max 仅由训练索引计算；验证、测试使用同一组训练统计，超出范围仍按现有评分接口裁剪至 [0,1]。快照记录拟合范围及样本数。合并重训时重新拟合于 train+val，仍不读取 test 的极值。固定官方量表若以后确认，可作为另一个明确标记的协议。
- 三数据集统一 `val_quality_SROCC` 选模，60 epoch 上限；保持 split_seed=42、train seed=42 和原始 70/10/20 索引。1K 无 Align 标签。
- DataLoader 使用独立 generator，worker 的 Python/NumPy RNG 显式播种，验证迭代不会消耗训练 loader 的随机序列。仍不承诺 CUDA 位级一致。
- 每轮记录实际处理的训练 batch 数、成功 optimizer 更新数和 AMP 跳过数，记录来源权重 SHA256 与运行环境。

服务器执行：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
bash scripts/run_protocol_v2.sh pretrain
for ds in agiqa1k agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh dev baseline "$ds"
done
```

默认 Python 是 `/home/sll/anaconda3/envs/WZX-AI/bin/python`，可用 `ABLATION_PYTHON` 覆盖。脚本不指定 GPU 编号。新来源是 `outputs/richhf18k_protocolv2_shared_seed42/best_component_anchor_pretrain.pth`。脚本拒绝覆盖已有输出。

## 2. 再做独立防过拟合对照

所有对照共用上一步预训练权重，各自从该权重重新微调，不从 baseline 下游权重继续训练。每个配置只改变一个因素：

| mode | 相对新 baseline 的唯一变化 | 建议顺序 |
|---|---|---|
| align_aux | 3K/2023 的 lambda_align_aux 0.3 → 0.1 | 第一轮 |
| proj_lr | PI/PT 独立学习率组，mult 1 → 0.1，即初始 1e-4 → 1e-5 | 第一轮 |
| qa_bn | 只冻结 QA 的 BN 运行统计，affine 权重仍训练 | 第一轮无明显收益时 |
| resnet_bn | 只冻结 ResNet 分支 BN 运行统计，affine 权重仍训练 | 独立于 qa_bn |

```bash
for ds in agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh dev align_aux "$ds"
done
for ds in agiqa1k agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh dev proj_lr "$ds"
done
# 必要时再分别运行：
# bash scripts/run_protocol_v2.sh dev qa_bn agiqa1k
# bash scripts/run_protocol_v2.sh dev resnet_bn agiqa1k
```

1K 的 Align 权重本来是 0；`align_aux agiqa1k` 会明确拒绝重复训练，请复用 baseline。按验证质量 SROCC 比较、同时观察验证 PLCC 与 Align 退化，不能依据测试结果挑 mode。若需一个跨数据集共同方法，提前声明共同规则（例如三数据集验证质量 SROCC 的等权平均），不要逐个测试指标拼接最优值。不同因素暂不叠加。开发阶段的既有脚本会在选模完成后报告测试值；制定下一步规则时必须排除这些值。

## 3. 最后才合并 train+val

先确定一个 mode，再为各数据集生成计划。示例用 baseline；如验证支持 proj_lr，将下面两处 baseline 一起替换为 proj_lr。

```bash
for ds in agiqa1k agiqa3k aigciqa2023; do
  bash scripts/run_protocol_v2.sh plan baseline "$ds"
  bash scripts/run_protocol_v2.sh refit baseline "$ds"
done
```

计划在 `outputs/protocolv2_plans/`，包括 JSON 及可查看的 YAML。预算生成器只读取开发 CSV 的 epoch 验证指标与训练 batch 数，忽略 final、best_test、测试指标；必须是完整单次开发运行。使用严格改善规则选择验证最优 epoch，预算是从第 1 轮累计到该轮的处理 batch 数。

重训模式会：

1. 验证计划自校验 SHA256、配置、原 CSV、来源检查点、数据顺序及三份 split 索引；发现不一致即停止。
2. 保留原测试索引及图像身份哈希，合并原 train+val，并全部使用 train dataset 的训练变换。有效比例为 80/0/20；快照另存实际样本数，原比例参数用于重建原 split。
3. 从同一预训练来源重新初始化下游，不加载已选中的开发模型。
4. 按固定训练 batch 预算运行，最后一轮可截断；不跑验证、不选 best、不用测试早停。批大小保持不变，避免 80% 数据配相同 epoch 导致更新预算增加。
5. LR 沿原开发阶段的余弦轨迹变化，按“原训练集每 epoch 的 batch 数”换算进度，保留原 60-epoch T_max；不在提前停止位置强制衰减到 min_lr。
6. 全部训练结束后只评估一次测试，CSV 使用独立 `train_val_fixed_budget` 协议，最终指标在 `test` 字段；不保存下游检查点。

预算单位是已处理 batch / optimizer 更新尝试，和开发阶段训练暴露预算对应；AMP 溢出时成功更新数可能少于预算，CSV 会另记成功次数与跳过次数。合并后 shuffle/epoch 边界改变，轨迹不会等同于开发训练。

80% 相比 70% 多约 14.3% 样本，但不保证成绩改善；这种结果应独立报告为固定预算 80/20 重训，不与原 70/10/20 混为同一协议，也不能把测试集当作替代验证集。已有测试集曾多次被观察，因此该流程只能改善后续纪律，不能消除历史上的人工选择影响。

## 本地验证范围

CPU 单元检查覆盖：训练统计不读取 held-out 极值、split/图像身份核对、计划与来源完整性、只按验证选预算、精确截断 batch、只做一次最终测试、独立 loader RNG、QuickGELU 双塔传参和跨协议加载拒绝、PI/PT 独立 LR 分组、ResNet BN 统计冻结而 affine 保持可训练。实际图像-MOS-Prompt 对应关系仍需在服务器用现有 `scripts/audit_downstream_data.py` 审计；云端没有原始数据。

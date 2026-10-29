# 1K 重复图片标签冲突与两套实验续跑

用户上传的 overlap 报告已经确认，下列三组开发实验均使用同一份存在冲突的 1K CSV：`allmerged_noce_noneg`、`protocolv2_both_noce_noneg`、`protocolv2_noce_noneg`。

| 过滤后 row_index（从0开始） | name | MOS | seed42 所在 split |
|---:|---|---:|---|
| 1078 | dreamStudio_8_9_1_1.jpg | 0.4414 | test |
| 1079 | dreamStudio_8_9_1_1.jpg | 2.0841 | train |

Prompt 相同，实际解析路径也相同。train_val、val_test 无重叠；唯一 train_test 重叠是上图。不能通过换一个 seed、跳过检查、平均标签或随便删除一行来掩盖。可能是文件名录入错误或标注记录冲突，但原始正确对应关系尚未确认。

目前两套新实验：1K dev 已完成60轮，1K plan 被拦截；1K refit/3K/2023 均未开始。预训练来源不受该下游 CSV 冲突影响，不需要为此重跑。

## 先让未受影响的数据集继续

两个入口新增 `downstream <dataset>`，只运行指定数据集的 dev→plan→refit，不重跑预训练、不触碰1K。输出已存在仍拒绝覆盖；不要同时在同一 GPU 上启动全部命令，下面按顺序运行。

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git pull --ff-only origin work

# 预训练不合并、下游合并的实验
bash scripts/run_two_stage_noce_noneg.sh downstream agiqa3k
bash scripts/run_two_stage_noce_noneg.sh downstream aigciqa2023

# 预训练和下游均合并的实验
bash scripts/run_allmerged_noce_noneg.sh downstream agiqa3k
bash scripts/run_allmerged_noce_noneg.sh downstream aigciqa2023
```

用户报告中已有的旧 noce_noneg 3K/2023 划分没有图片交集；新运行的 plan 阶段仍会逐一核查。不要因旧报告无重叠而绕过新来源下的检查。

## 1K 先核对原始数据，不自动修 MOS

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python \
  scripts/inspect_agiqa1k_csv_conflict.py \
  --output outputs/agiqa1k_csv_conflict.json
```

上传该 JSON。它只读原 CSV，列出重复记录全部字段、相邻文件名及对应 CSV 记录、原 CSV SHA256；不训练、不改变 CSV 或划分。邻近文件名仅用于定位，不能据此猜测 2.0841 应对应哪张图。仍可能需要原始官方 name–MOS 表确认。

1K 冲突解决后要用新版本数据重新跑受影响的下游开发/预算与合并训练，原开发模型已经接触到测试图片，不能作为无泄漏结果继续沿用。修正版本应单独命名并保留旧 CSV/结果，明确说明测试成员或标签是否变化。同 CSV、同划分的历史 1K 结果也应复核，不把现有成绩当作已证明无泄漏的正式结果。

本次不擅自改变科学划分协议、CSV、标签或历史测试索引。已经完成的 1K 结果继续保留用于诊断。

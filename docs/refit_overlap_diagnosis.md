# 合并计划报 Repeated image IDs across splits

两个实验都使用 `utils.refit_protocol.split_manifest`，且沿用相同下游 row-wise split，所以同样的数据重复问题会在两组同时触发。这个异常出现在合并计划生成阶段，尚未开始合并训练；不要因此从头重跑预训练。

旧检查读取 CSV 的完整 name（不是仅截取 basename）或图像相对路径，发现相同标识跨 train/val/test 即停止。仅凭 traceback 还不能知道哪个数据集、哪几行、是否标签冲突。现在检查使用数据加载器真实路径解析器，再规范化路径；同时列出跨集合例子、行索引、MOS 和 Prompt。不同目录里同名文件不会因为 basename 一样而被误报，`sub/../image` 等同文件路径别名也能检出。不是图像内容/近重复哈希审计。

## 服务器一次只读诊断

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam
git checkout work
git pull --ff-only origin work
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python \
  scripts/audit_refit_overlaps.py --output outputs/refit_overlap_audit.json
```

上传 `outputs/refit_overlap_audit.json`。自动查找两阶段 noce_noneg、allmerged_noce_noneg 及此前下游 noce_noneg 已完成的开发快照；也可指定：

```bash
HF_HUB_OFFLINE=1 /home/sll/anaconda3/envs/WZX-AI/bin/python \
  scripts/audit_refit_overlaps.py \
  --snapshot outputs/agiqa1k_protocolv2_both_noce_noneg_dev_seed42/config_snapshot.yaml \
  --output outputs/refit_overlap_audit.json
```

不构建模型、不训练、不改 CSV、划分或已有结果。诊断中的行索引是过滤后 DataFrame 的 0-based 索引，不一定等于原 CSV 物理行号。JSON 含本地图片路径及少量标签/Prompt，用于用户审查，不要盲目公开原数据。

重新运行原 `plan` 时，如果仍发现重叠，现在也会自动写 `<prefix>.overlap.json` 并报出各 split 对的数量，不会生成可绕过检查的计划。无需反复运行 GPU 训练来得到诊断。

## 后续如何处理

- 若是图片路径身份误判，仅修身份判定，才有可能保留现有开发结果和预算。
- 若同一图片真实出现在 train 与 test，不能删掉这条检查继续测试；需要明确处理重复记录及划分，再重新跑受影响下游开发/预算。预训练不受这个下游 CSV 问题影响。
- 若只在 train 与 val 重复，合并后虽都成为训练样本，原验证选预算已存在泄漏，不能把这个预算当作干净验证预算。应先确认是否多 Prompt/多标签记录，然后按图片分组建立独立验证划分。
- 同图片若标签相同或不同，处理也不同；在没有重复记录报告前不能自行 deduplicate 或覆盖原测试索引。

新版本不改变安全历史计划的 JSON schema/原身份摘要，避免仅因增加诊断使已有无重叠计划失效。额外发现实际路径别名重叠时仍会停止，以保护测试独立性。

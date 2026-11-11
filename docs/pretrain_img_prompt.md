# projection_only_v1 图像—Prompt 关系预训练权重比较

本轮仅改变预训练 `lambda_img_prompt`：当前基线 0.25，对照 prompt020=0.20、prompt030=0.30。
上一轮 I-I 权重 0.10/0.20 的两组在全部十项下游测试指标上均低于基线；因此 I-I 保持 0.15，原最好基线及其权重不变。
本轮不改变全局图像—Prompt 关系的样本权重来源（人工一致性 a）；也不改变局部区域关系和词级关系的损失系数。

固定 projection_only_v1、5 个可训练上下文词元、g→PI、P₀独立局部投影并绕过 PI、QA/QL 双质量残差、NSS、Component Anchors（各0.01）/Ordinal（各0.015）、PCGrad。
质量文本五档 CE、negative 仍关闭；其余参数完整复制原最好基线。
两组都从原始 CLIP/ResNet 初始化，**不加载旧项目预训练、不 resume**；预训练保留官方 train/val，seed=42、split_seed=42，50轮预算、patience=15，按最大 val_component_anchor_SROCC 选模。

三个下游分别严格复制 current_tuning_reference v2，只替换本组来源与独立运行路径。
train_seed=42；1K subset1078、split_seed=3407；3K/2023 split_seed=42；Quality SoftSROCC 依次0.03/0.02/0.04。
60轮、保留 train/val/test，按 val_quality_SROCC 最大选模。PI/PT 初始 LR=1e-4、全局 semantic_proj=1e-5，其他分组不变。

入口默认仅预训练；显式 `all` 才自动按顺序执行预训练→1K→3K→2023。
每个下游只允许加载本组完成并校验的 best_component_anchor_pretrain.pth，核对配置、运行标识、模式、完成状态与SHA256；加载后再次核对来源。
两组动态 runner 隔离，输出目录不共享。已有目录或CSV拒绝覆盖；失败保留输出并停止后续阶段。
`continue` 仅跳过通过全部检查的完成阶段，启动尚无输出的阶段；不自动恢复不完整训练。
1K/2023执行已有划分清单核查；3K若原配置未启用成员清单，明确报告仅核对配置/runtime seed，不伪造成员记录。
本地模型缓存由 HF_HUB_OFFLINE=1 默认约束，不自动换预训练来源。云端 CPU 拒绝真实训练。

服务器先同步一次：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && git pull --ff-only origin work
```

终端一 / GPU0：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_img_prompt.sh prompt020 all
```

终端二 / GPU1：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=1 bash scripts/run_pretrain_img_prompt.sh prompt030 all
```

共两组重新预训练与六次下游，GPU编号可自行更改。
若只需预训练，将 `all` 改成 `pretrain`；单独下游支持 `agiqa1k`、`agiqa3k`、`aigciqa2023`，也支持 `downstream`。
无需真实启动的预览：`ABLATION_PYTHON=python bash scripts/run_pretrain_img_prompt.sh prompt020 all --dry-run`。

全部完成后打包（两组分别核验完整状态后才生成文件）：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && bash scripts/run_pretrain_img_prompt.sh prompt020 pack-downstream && bash scripts/run_pretrain_img_prompt.sh prompt030 pack-downstream
```

生成：

- projection_only_v1_ctx5_img_prompt020_downstream_results.tar.gz
- projection_only_v1_ctx5_img_prompt030_downstream_results.tar.gz

包含预训练及三个下游的 CSV、配置、元数据、日志、完成记录和来源指纹；不含大模型权重。打包文件已存在同样拒绝覆盖。
沿用原打包白名单，不包含独立 data_split_manifest.json；1K/2023 成员指纹摘要保存在 run_metadata.json。
实验独立报告，单 seed 不视作稳定性证明，不保证训练逐位复现。CPU 测试使用伪产物，不产生实际训练成绩。

本次验证：`PYTHONPATH=.:tests /workspace/.venvs/simsiam/bin/python -m unittest discover -s tests`，335项通过；新增6项包含两组全部六个下游的伪产物来源/划分/60轮/打包校验。两组 `all --dry-run` 和 shell语法检查通过；未启动真实训练。

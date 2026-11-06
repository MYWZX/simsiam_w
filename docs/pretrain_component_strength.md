# projection_only_v1：Component Anchor / Ordinal 单因素预训练

四组独立新预训练；每组只改变一类分量损失系数，artifact / aesthetics 同步缩放。

| 变体 | 每个分量 Anchor | 每个分量 Ordinal |
|---|---:|---:|
| 原最好基线（已有结果，不重跑） | 0.01 | 0.015 |
| anchor050 | 0.005 | 0.015 |
| anchor200 | 0.02 | 0.015 |
| ordinal050 | 0.01 | 0.0075 |
| ordinal200 | 0.01 | 0.03 |

配置位于 `configs/pretrain_component_strength/<variant>/pretrain.yaml`。
运行器严格对比原 projection_only_v1 预训练配置：除上述系数、run_id、输出路径外，任何变化均拒绝。

保留 projection_only_v1、5 个上下文词元、g→PI、P₀ 独立局部路径、QA 双质量残差、NSS、PCGrad、现有辅助目标；质量文本 CE 和 negative 关闭。
从原始 CLIP / ResNet 预训练初始化，不加载本项目检查点、不 resume。seed=42、split_seed=42，使用 RichHF 官方 train/val，不合并。
预算 50 轮、early_stop_patience=15，按 val_component_anchor_SROCC 最大选模，保存 best_component_anchor_pretrain.pth。
学习率、优化器分组和其他目标权重均保持原预训练策略。

## 两张卡执行

先同步代码（只执行一次，完成后再启动两个终端）：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && git pull --ff-only origin work
```

终端一，GPU 0 顺序运行 Anchor 两组：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_component_strength.sh anchor050 && CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_component_strength.sh anchor200
```

终端二，GPU 1 顺序运行 Ordinal 两组：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=1 bash scripts/run_pretrain_component_strength.sh ordinal050 && CUDA_VISIBLE_DEVICES=1 bash scripts/run_pretrain_component_strength.sh ordinal200
```

GPU 编号按服务器空闲卡调整。每个终端中前一组失败即停止，保留现场；不要重复启动同一变体。
所有输出与 CSV 独立，遇到已有目录或 CSV 立即拒绝，不覆盖旧基线或其他组。CPU 环境拒绝真实启动。
默认仅使用本地 Hugging Face 缓存（HF_HUB_OFFLINE=1），需已缓存原始模型权重；不自动更换预训练来源。

本地无训练检查：

```bash
ABLATION_PYTHON=python bash scripts/run_pretrain_component_strength.sh anchor050 --dry-run
```

## 完成后打包

四组全部完成后：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && for variant in anchor050 anchor200 ordinal050 ordinal200; do bash scripts/run_pretrain_component_strength.sh "$variant" pack || break; done
```

生成四个 `projection_only_v1_ctx5_component_<variant>_pretrain_results.tar.gz`。
每个包包含 CSV、配置、日志、运行元数据、启动与完成记录、选中检查点 SHA256 / 选中轮次的来源指纹，不包含模型权重。
打包会再次核验完成状态、配置、检查点模式与来源；未完成、被改写或错误来源的实验拒绝打包，已有结果包也不覆盖。

默认仍只预训练。显式传入 `all` 时，顺序运行预训练、1K、3K、2023；传入 `downstream` 时，核验已经完成的预训练后仅运行三个下游。每组下游只使用本组 best_component_anchor_pretrain.pth，核验模式、配置、完成状态和 SHA256。
旧下游入口固定了旧来源，不能直接拿来加载这四组权重。预训练分数不能单独作为替换最好下游基线的依据。
这四组是独立实验，不承诺与历史实验训练逐位复现，也不沿用历史分数作为新结果。


## 显式自动接下游

先同步代码，再在两个终端分别执行：

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_component_strength.sh anchor050 all && CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_component_strength.sh anchor200 all
```

```bash
cd /home1/wzx/aigc_naturalness_code/simsiam && CUDA_VISIBLE_DEVICES=1 bash scripts/run_pretrain_component_strength.sh ordinal050 all && CUDA_VISIBLE_DEVICES=1 bash scripts/run_pretrain_component_strength.sh ordinal200 all
```

每组预训练完成后，依次运行三个下游；每个下游 60 轮、按 val_quality_SROCC 最大选模，保留 train/val/test。
训练 seed=42；1K 为 subset1078、split=3407，3K/2023 split=42；2023 质量 SoftSROCC=0.04，其余下游参数按 current_tuning_reference v2 保留。
每组使用独立来源与输出，任一步失败即停止。`all` 会预先检查该组四个阶段都不存在旧输出，已有预训练输出时不会跳过或重跑。

如果预训练已经完成，不再使用 `all`，改用例如：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_pretrain_component_strength.sh anchor050 downstream
```

若某个下游也已完成或失败留有输出，`downstream` 会拒绝覆盖；可使用明确的数据集 action（agiqa1k、agiqa3k、aigciqa2023）运行尚未启动的阶段。
运行中的进程不会因 git pull 自动接下游，须等待其退出，再使用 downstream。

四组全部预训练和下游完成后打包（每组一个包，含预训练及三个下游的轻量结果和来源指纹，不含模型权重）：

```bash
for variant in anchor050 anchor200 ordinal050 ordinal200; do bash scripts/run_pretrain_component_strength.sh "$variant" pack-downstream || break; done
```

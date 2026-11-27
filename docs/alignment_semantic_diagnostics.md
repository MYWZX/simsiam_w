# 冻结模型的验证集语义诊断

本工具不训练、不建立优化器、不修改实验目录，不从预训练权重或日志拼造下游模型。比较原最好基线与固定温度0.1的区域—词元分支，均采用既定验证Quality SROCC选出的完整60轮模型。输出CSV、人工复核状态、分支排序统计、门控变化、匹配热图及HTML报告。

## 必须先确认权重

历史普通下游和双选模运行默认只把所选权重留在内存，CSV不是模型检查点。默认只检查各自精确run_id目录中的：

- `semantic_quality.pth`＋`semantic_quality.json`：本次新增的完整推理快照；或
- `best_selected_downstream.pth`＋`selected_checkpoint_manifest.json`：此前显式保存的完整下游格式。

不会跨目录找别的实验，拒绝last_pretrain、representation-only、评分头缺失、错误轮次/模式/来源或未完成模型。清单核验文件SHA；实际推理还会核对payload、原CSV选模、全部参数及非持久缓冲区的键/形状/dtype，并严格载入。构造网络时关闭所有模型权重下载，冻结整个模型、设为eval、只做inference_mode前向。

```bash
mkdir -p run_logs
bash scripts/run_alignment_semantic_diagnostics.sh audit --output "run_logs/semantic_inventory_$(date -u +%Y%m%dT%H%M%SZ).json"
```

退出码2及`blocked`表示缺少前提，不是训练启动失败。清单的`snapshot_manifest_verified`仅表示文件清单、SHA及run证据通过；全量张量和数据成员在推理时继续检查。若历史权重已随进程退出而丢失，就无法直接诊断那次模型，也无法从CSV恢复。

## 准备验证集候选与人工复核

准备过程不要求已有模型权重，只使用已完成0.1运行的配置/划分证据及本地数据。默认取实际验证划分原随机顺序中的前24张，不读取test做诊断。复核图像、原描述和候选描述：

```bash
bash scripts/run_alignment_semantic_diagnostics.sh prepare --dataset agiqa3k --limit 24 --output run_logs/semantic_pairs_3k_v1
bash scripts/run_alignment_semantic_diagnostics.sh prepare --dataset aigciqa2023 --limit 24 --output run_logs/semantic_pairs_2023_v1
```

每个目录包含`review.html`、图像缩略图、`validation_samples.json`、`pairs.json`。候选生成只是有限的词法替换（实体/颜色/数量/方位），不是标注器、完整分词器或可靠负样本生成器。不把Answer当作真值或置信度来源。

在`pairs.json`中核对图像后编辑：

- `candidate`：不确定，默认状态；不计入正确率。
- `confirmed_original_better`：人工确认原描述比改动描述更符合这张图，才作为有序配对；原Prompt本身可能不完全正确，不能仅因改动词语就确认。
- `confirmed_equivalent`：人工确认两种描述为有效等价表达，单独报告分数不变性，不作为负例。
- `rejected`：排除。

确认状态必须填写`reviewer`和基于图像的`review_note`。不能改动行号、原Prompt、图像SHA及所属验证划分来迎合模型结果；需要添加人工案例时从validation_samples中选图，并填写相同身份字段。自动替换可能仍符合图像，含糊案例保留candidate/rejected。

验证划分必须有实际成员记录，或能与已记录的成员摘要匹配。只知道seed=42不够：若旧3K基线没有保存实际划分证据，比较会明确停止，不暗中重建并冒充已核验的历史划分。Prompt在CLIP中会被截断时也拒绝推理，避免把截断当语义错误。

## 冻结前向与图像

前提均满足且人工复核完成后：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_alignment_semantic_diagnostics.sh run --cases run_logs/semantic_pairs_3k_v1/pairs.json --output run_logs/semantic_diagnostics_3k_v1
CUDA_VISIBLE_DEVICES=1 bash scripts/run_alignment_semantic_diagnostics.sh run --cases run_logs/semantic_pairs_2023_v1/pairs.json --output run_logs/semantic_diagnostics_2023_v1
```

可用`--device cpu`做小规模推理；不会因此训练。`--include-candidates`允许未复核案例只参与可视化，仍从所有排序正确率中排除；默认只跑确认案例。`--figures 12`限制图像案例数量，CSV始终包含全部参与案例。

评分表包含原/改描述的全局、局部、最终一致性，分数差、全局门控权重及质量输出。质量应与文本无关，出现超出容差的变化会停止。统计按类别和模型报告N、严格排序正确率、平局比例、平均分数下降；等价描述单独报告绝对变化。没有确认配对时正确率为null，不伪造为0，也不以候选替换的模型反应推断标签可信度。

热图语义严格区分：原基线只具有“整句Prompt条件下的区域权重”，不能画成词级定位；区域—词元分支从真实前向钩子取得BPE片段对应的权重，对改动词的匹配BPE取均值；找不到该片段时明确标为整句均值回退。BPE片段不等同于人工语义词标注。原/改描述共享实际空间概率色标，不对各图分别做min-max拉伸制造虚假的显著区域。不同模型网格不同，颜色亮度不能跨模型直接当置信度比较；注意力也不是正确定位证明。

代表图按已确认配对中分数变化的两端交替选择，覆盖成功/失败；剩余候选单独标注，完整CSV不删行。NPZ保存的是派生注意力图，不是模型参数。所有输入图像、病例和检查点保留指纹，变化时停止。无人工复核或缺少权重时，不宣称已经得出真实语义诊断结论。

打包仅包含派生CSV/JSON/HTML/PNG/PDF/NPZ，不包含pth/pt/bin或数据集原始权重：

```bash
bash scripts/run_alignment_semantic_diagnostics.sh pack --input run_logs/semantic_diagnostics_3k_v1 --output semantic_diagnostics_3k_v1.tar.gz
bash scripts/run_alignment_semantic_diagnostics.sh pack --input run_logs/semantic_diagnostics_2023_v1 --output semantic_diagnostics_2023_v1.tar.gz
```

## 为以后明确需要的新运行导出快照

新增训练配置开关`train.export_semantic_snapshots: semantic_selected_v1`，默认关闭。只有完整60轮完成后才导出：普通Quality选模保存一份，Quality/mean-joint双选模保存两份。完整保留冻结CLIP文本编码器、评分头、共享/独立投影、所有持久及选中轮的非持久缓冲区；保存源SHA、配置摘要、真实轮次、验证/测试记录和独立SHA清单。它是推理快照，没有优化器，不能作为训练resume断点。

默认旧训练的RNG、模型选择和文件行为保持；默认不增加磁盘权重。导出不改变优化器/训练目标；选中轮缓冲区也被保存，避免错误混用末轮缓冲区。**新功能不能追溯性生成已经丢失的历史模型。**

如果确实需要以后新跑诊断专用模型，可先生成可审阅计划，不执行任何训练：

```bash
bash scripts/run_alignment_semantic_diagnostics.sh capture-plan --output run_logs/semantic_capture_plan_v1
```

生成四个独立新run_id的配置、`models.json`与`plan.json`中的未执行命令。它们是新运行，不能标成原最好基线或旧0.1成绩；本次不自动启动这些命令。若之后另行决定运行并完成，再通过`--models run_logs/semantic_capture_plan_v1/models.json`使用新模型做audit/prepare/run，标签也明确显示new_capture。新计划同时记录实际划分，避免旧记录缺失。

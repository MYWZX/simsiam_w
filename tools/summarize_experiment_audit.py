"""Read existing experiment logs and generate reports; never load/train models."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import statistics
from collections import Counter
from datetime import date
from pathlib import Path


DATASETS = ("agiqa1k", "agiqa3k", "aigciqa2023")
LABELS = {"agiqa1k": "1K", "agiqa3k": "3K", "aigciqa2023": "2023"}
METRICS = ("test_quality_PLCC", "test_quality_SROCC", "test_align_PLCC", "test_align_SROCC")
MODULES = [
    ("Matched CLIP visual / frozen text", "启用；视觉可微调，文本冻结", "共享图像编码器与文本编码参与预测；文本冻结不是失效。尚无当前方案单独移除 CLIP 的严格消融。", "保留；不要把同源初始化收益等同于所有跨模态损失都有效", "models/two_stage_models.py:54,109"),
    ("ResNet50 local encoder", "启用；可微调", "同时进入 Fusion、局部质量和局部一致性；缺少当前方案去局部分支消融。", "保留，先记录局部分支分数", "models/two_stage_models.py:159,598"),
    ("SemanticStructureFusion", "启用；共享", "全局 query 与局部 tokens 注意力融合；两个任务都可更新它。分开 projector 不等于编码器完全解耦。", "保留；下一步补分模块梯度诊断", "models/fusion.py:9; models/two_stage_models.py:192"),
    ("SafeAug1 / SafeAug2 / Clean View", "启用；严格三视图", "增强图仅承担 Image–Image 的有效训练目标；干净图承担跨模态、人类评分、局部标签和 Component Anchor。SafeAug 为整图 resize + 随机翻转，不含裁剪/颜色扰动。", "保留监督路由；不能仅凭 loss 下降断言增强提高泛化", "datasets/mixed_pretrain_dataset.py:23,1367; train_cm_pretrain.py:1774"),
    ("Quality Image–Image SimSiam", "启用；λ=0.15", "质量私有 projector/predictor 比较两个 SafeAug；停止梯度实现存在。最后 train_img≈−0.954 说明正对拟合，不证明质量知识或排除坍塌。", "保留；补特征方差/有效秩及关闭该项的消融", "train_cm_pretrain.py:566,1853; models/two_stage_models.py:1110"),
    ("Clean Image–Prompt / Image–Answer SimSiam", "启用；λ=0.25 / 0.20", "进入一致性私有 adapter 与语义 projector；当前末轮对应 loss 非零，但缺独立增益消融。", "分别做消融，不能整体宣称都有收益", "train_cm_pretrain.py:1781,1875"),
    ("Prompt–Answer SimSiam", "启用；λ=0.05", "当前没有关闭；末轮 loss≈−0.168，监督由 align×answer confidence 加权。", "优先候选消融，不直接删除", "configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml; train_cm_pretrain.py:1820"),
    ("Map 区域反馈 / Prompt 词级反馈", "启用；λ均为0.05", "当前采用 pull_push（不是历史 relative_rank）；Clean Map 监督局部关系。末轮相应 loss 非零；没有独立有效样本计数。", "先补有效比例，再分别关闭；没有这些标签时可配置禁用", "train_cm_pretrain.py:449,528"),
    ("Component Anchor + Ordinal", "启用；Artifact/Aesthetics各0.01+0.015", "直接监督 quality_clean_z；训练 SROCC 上升但验证早停滞。历史1K的 Component Anchor 结果0.9018是联合变更/检查点选择证据，非当前单因素消融。", "有质量结构信号，但泛化有瓶颈；需移除 Anchor、移除 Ordinal 的单项消融", "train_cm_pretrain.py:590; outputs/all_results_richhf_decoupled_quality_align.csv"),
    ("Quality score regression + rank", "启用；λ=0.20+0.10", "监督 Fusion + NSS局部/全局质量头；预训练 quality_adapter.disabled，因此这一路不经过私有 quality_image_projector。", "保留；注意评分质量空间与锚点投影空间不是完全同一路", "models/two_stage_models.py:635,680; train_cm_pretrain.py:1855"),
    ("Artifact/Aesthetics auxiliary heads", "启用；各reg0.05+rank0.025", "输入Fusion/global与局部evidence，而非私有SimSiam质量投影。历史 no_quality_aux 的2023质量SROCC下降0.0392，不能直接视为冗余。当前缺独立消融。", "先保留，后做当前方案的关闭对照", "models/two_stage_models.py:680; train_cm_pretrain.py:1808"),
    ("Quality projector → downstream Adapter", "预训练/下游均生效；表示权重迁移", "仅 projector 迁移，predictor丢弃；同种子消融仅改变enabled。3K一致性有较明确收益，质量PLCC却下降；1K/2023质量有种子相关收益。", "保留为候选，但不能宣称所有指标均受益", "models/two_stage_models.py:1134,1188"),
    ("NSS gate + Patch Quality + quality_mix", "启用；质量输出内部融合", "s_quality=α·patch+(1−α)·global，α是可学习全局标量，不是样本自适应；Align也读取含NSS的patch_tokens。未记录patch质量成绩/α。", "最高优先级诊断；再做验证集校准", "models/two_stage_models.py:617,665; train_cm_iqa.py:1395"),
    ("Align adapters + global/patch heads + gate", "3K/2023启用；1K无Align标签", "Align gate为样本自适应；全局/局部评分已有日志，融合有小幅互补。私有Align adapters尚无单独关闭消融。", "保留；局部分支相关低不等于无用", "models/two_stage_models.py:697"),
    ("Downstream MSE / local rank / Soft-SROCC", "Quality/Align按各自人工标签启用", "回归约束绝对分数，rank/SROCC约束排序；无当前逐项移除实验，无法判断各项边际收益。", "按单因素消融评估，暂不同时删除", "train_cm_iqa.py:1126"),
    ("Quality perturbation rank / mismatched prompt", "下游启用；λ=0.02 / 0.05", "质量退化排序与错误Prompt监督分别属于两个任务；尚无当前损失独立消融。", "补退化成功比例/负样本有效性再判断", "train_cm_iqa.py:1126,1137"),
    ("Pretraining PCGrad", "启用；逐参数版本", "train_cm_pretrain._pcgrad_backward按参数张量判断冲突；没有当前预训练启用/禁用严格对照，不能用下游global_shared收益证明它有效。", "保留现有对照；不要混称为global_shared", "train_cm_pretrain.py:1667"),
    ("Downstream PCGrad", "CLI选择；1K自动回退单任务", "global_shared η=1对3K有有效种子；2023质量多种子均下降；η=0.5/低Adapter LR未统一改善。", "保留有效候选，不设为所有数据集必选项", "utils/pcgrad.py:19; train_cm_iqa.py:1226"),
    ("Final MOS / final fusion", "当前监督和指标已关闭；兼容计算仍在", "模型仍计算s_final，predict_score仍返回它；two_output_only仅在训练/评估协议层关闭监督/统计。主CSV有效指标是Quality/Align。", "不要恢复合成MOS；后续单独清理旧推理接口", "models/two_stage_models.py:793,824; train_cm_iqa.py:1412"),
    ("Triplet / MeanResidual / single Quality Anchor / VMamba / MultiCrop", "当前主方案未启用", "代码存在不等于参与当前训练。近远质量关系、单Quality Anchor和Residual损失权重为零；当前配置无原分辨率多crop。", "保留兼容代码，简化文档，不作为当前模块计数", "configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml"),
    ("Mirror / balanced sampler / loss reweighting", "当前主方案关闭", "历史收益随任务变化；镜像在1K/3K质量可提升，却损害3K/2023一致性。", "不设为统一默认，保留历史实验", "configs/richhf18k_cm_pretrain_decoupled_quality_align.yaml"),
]


def load_sessions(root):
    sessions, errors = [], []
    files = sorted((root / "outputs").glob("all_results*.csv"))
    for path in files:
        active, attempts = {}, Counter()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for line, row in enumerate(csv.DictReader(stream), 2):
                rid = row.get("run_id", "")
                if row.get("event") == "run_start" or rid not in active:
                    attempts[rid] += 1
                    session = {"run_id": rid, "attempt": attempts[rid], "source_csv": str(path.relative_to(root)), "source_line": line, "stage": row.get("stage", ""), "start_time": row.get("time", ""), "config_path": row.get("config_path", ""), "output_dir": row.get("output_dir", ""), "cfg": {}, "events": {}, "epochs": []}
                    sessions.append(session)
                    active[rid] = session
                s = active[rid]
                try:
                    cfg = json.loads(row.get("config_json") or row.get("config") or "{}")
                    metrics = json.loads(row.get("metrics_json") or row.get("metrics") or "{}")
                except (ValueError, TypeError) as exc:
                    errors.append((str(path), line, str(exc)))
                    continue
                if cfg:
                    s["cfg"] = cfg
                s["last_time"] = row.get("time", "")
                s["events"][row.get("event")] = {"metrics": metrics, "epoch": row.get("epoch"), "best_epoch": row.get("best_epoch"), "line": line}
                if row.get("event") == "epoch":
                    s["epochs"].append(s["events"]["epoch"])
    return files, sessions, errors


def dataset(s):
    dtype = str(s["cfg"].get("dataset", {}).get("type", "")).lower()
    for key in DATASETS:
        if key in dtype:
            return key
    for key in DATASETS:
        if key in s["run_id"].lower() or key in s["config_path"].lower():
            return key
    return dtype or "未知"


def get_metrics(s, event="best_test"):
    return s["events"].get(event, {}).get("metrics", {})


def value(s, key):
    if "align" in key and dataset(s) == "agiqa1k":
        return None
    m = get_metrics(s)
    if m.get(key.removesuffix("PLCC").removesuffix("SROCC") + "N", 2) < 2:
        return None
    v = m.get(key)
    return v if isinstance(v, (float, int)) and math.isfinite(v) else None


def protocol(s):
    t = s["cfg"].get("train", {})
    if t.get("quality_only"):
        return "Quality-only"
    if t.get("two_output_only"):
        return "Quality+Align；无Final"
    return "历史含Final/其他；不与双输出直接混算"


def compact_method(s):
    rid = s["run_id"]
    for key in DATASETS:
        if rid.startswith(key + "_"):
            return rid[len(key) + 1:]
    return Path(s["config_path"]).stem or rid


def flatten(v, prefix=""):
    if not isinstance(v, dict):
        return {prefix: v}
    out = {}
    for k, item in v.items():
        out.update(flatten(item, f"{prefix}.{k}" if prefix else k))
    return out


def differences(a, b):
    aa, bb = flatten(a["cfg"]), flatten(b["cfg"])
    ignore = {"run_id", "output_dir", "results_csv", "source_config"}
    return [k for k in sorted(aa.keys() | bb.keys()) if k not in ignore and aa.get(k) != bb.get(k)]


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def delta(v, base, color=False):
    if v is None or base is None:
        return "—"
    d = v - base
    text = f"{d:+.4f}"
    if not color:
        return text
    cls = "up" if d > 5e-5 else "down" if d < -5e-5 else "flat"
    return f'<span class="{cls}">{text}</span>'


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows) + "</tbody></table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prefix", default="project_experiment_audit_20260929")
    args = parser.parse_args()
    root = args.root.resolve()
    if Path(args.prefix).name != args.prefix:
        parser.error("prefix must be a filename stem")
    files, sessions, errors = load_sessions(root)
    tests = [s for s in sessions if "best_test" in s["events"]]
    pretrains = [s for s in sessions if s["stage"] == "pretrain"]
    latest = {s["run_id"]: s for s in sorted(tests, key=lambda s: s["last_time"])}
    out = root / "outputs"
    csv_path, html_path, md_path = [out / (args.prefix + ext) for ext in (".csv", ".html", ".md")]
    fields = ["dataset", "method", "run_id", "attempt", "seed", "split_seed", "protocol", "best_epoch", "quality_PLCC", "quality_SROCC", "align_PLCC", "align_SROCC", "quality_RMSE", "align_RMSE", "quality_global_adapter_SROCC", "align_global_SROCC", "align_patch_SROCC", "legacy_final_PLCC", "legacy_final_SROCC", "selection_metric", "val_selection_value", "pretrained_checkpoint", "load_scope", "pcgrad_mode", "pcgrad_strength", "quality_adapter_lr_mult", "start_time", "last_time", "output_dir", "source_csv", "source_line", "config_path", "config_fingerprint", "has_final_log"]
    flat_rows = []
    for s in sorted(tests, key=lambda s: (dataset(s), s["start_time"], s["source_csv"], s["attempt"])):
        c, m, f = s["cfg"], get_metrics(s), get_metrics(s, "final")
        t = c.get("train", {})
        canonical = copy.deepcopy(c)
        for k in ("run_id", "output_dir", "results_csv", "source_config"):
            canonical.pop(k, None)
        pc = t.get("pcgrad", {})
        entry = {"dataset": dataset(s), "method": compact_method(s), "run_id": s["run_id"], "attempt": s["attempt"], "seed": c.get("seed", ""), "split_seed": c.get("split_seed", c.get("seed", "")), "protocol": protocol(s), "best_epoch": f.get("best_epoch", s["events"]["best_test"].get("best_epoch")), "selection_metric": f.get("selection_metric", t.get("selection_metric", "")), "val_selection_value": f.get("best_selection_value", ""), "pretrained_checkpoint": t.get("pretrained_checkpoint", ""), "load_scope": t.get("pretrained_load_scope", "auto"), "pcgrad_mode": pc.get("mode", "per_parameter") if pc.get("enabled") else "off", "pcgrad_strength": pc.get("projection_strength", 1.0) if pc.get("enabled") and pc.get("mode") == "global_shared" else "", "quality_adapter_lr_mult": t.get("layerwise_lr", {}).get("quality_adapter_mult", t.get("layerwise_lr", {}).get("quality_mult", 1.0)), "source_line": s["events"]["best_test"]["line"], "config_fingerprint": hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16], "has_final_log": "final" in s["events"]}
        for k in ("start_time", "last_time", "output_dir", "source_csv", "config_path"):
            entry[k] = s.get(k, "")
        for key in fields[8:17]:
            entry[key] = value(s, "test_" + key)
        if not t.get("two_output_only") and not t.get("quality_only"):
            entry["legacy_final_PLCC"] = m.get("PLCC", m.get("test_PLCC"))
            entry["legacy_final_SROCC"] = m.get("SROCC", m.get("test_SROCC"))
        flat_rows.append(entry)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)

    current_rows, current_md = [], []
    for ds in DATASETS:
        for stem in ("decoupled_best_quality", "decoupled_no_adapter", "decoupled_best_quality_pcgrad", "decoupled_best_quality_pcgrad_global", "decoupled_best_quality_pcgrad_global_eta05", "decoupled_best_quality_pcgrad_global_eta05_qa_lr01"):
            for seed in (42, 123, 3407):
                s = latest.get(f"{ds}_{stem}_seed{seed}")
                if not s:
                    continue
                control_stem = "decoupled_best_quality_pcgrad_global_eta05" if "qa_lr01" in stem and ds != "agiqa1k" else "decoupled_best_quality"
                control = latest.get(f"{ds}_{control_stem}_seed{seed}")
                values = [value(s, k) for k in METRICS]
                dsrow = [html.escape(LABELS[ds]), html.escape(stem), str(seed)] + [fmt(v) for v in values]
                dsrow += [delta(v, value(control, k), True) if control and control is not s else "—" for v, k in zip(values, METRICS)]
                dsrow += [html.escape(control_stem if control and control is not s else "基线"), html.escape(", ".join(differences(control, s))) if control and control is not s else "—"]
                current_rows.append(dsrow)
                current_md.append([LABELS[ds], stem, str(seed)] + [fmt(v) for v in values])

    mean_rows = []
    for ds in DATASETS:
        for stem in ("decoupled_best_quality", "decoupled_no_adapter", "decoupled_best_quality_pcgrad", "decoupled_best_quality_pcgrad_global"):
            group = [latest[f"{ds}_{stem}_seed{seed}"] for seed in (42,123,3407) if f"{ds}_{stem}_seed{seed}" in latest]
            if not group:
                continue
            cells = [LABELS[ds], stem, str(len(group))]
            for k in METRICS:
                vs = [value(s, k) for s in group if value(s,k) is not None]
                cells.append(f"{statistics.mean(vs):.4f} ± {statistics.stdev(vs):.4f}" if len(vs)>1 else fmt(vs[0]) if vs else "—")
            mean_rows.append(cells)

    historical_rows = []
    def historical_result(rid, seed):
        matches = [s for s in tests if s["run_id"] == rid and s["cfg"].get("seed") == seed]
        return max(matches, key=lambda s:s["last_time"]) if matches else None

    for ds in DATASETS:
        pairs = [
            ("镜像预训练", "mirror_pretrain_seed123", "nomirror_best_component_seed123", 123),
            ("移除Quality Auxiliary", "no_quality_aux_seed123", "component_loss_reweighting_best_component_seed123", 123),
            ("移除预训练Final Loss", "no_final_loss_seed123", "component_loss_reweighting_best_component_seed123", 123),
        ]
        if ds == "agiqa1k":
            pairs += [
                ("Component Anchor历史方案", "quality_components_bestartifact_repr_seed42", "quality_anchor_ordinal_matched_clip_repr_seed42", 42),
                ("SafeAug历史方案", "quality_anchor_ordinal_matched_clip_repr_safeaug_seed42", "quality_anchor_ordinal_matched_clip_repr_seed42", 42),
                ("SafeAug+MultiCrop历史方案", "quality_anchor_ordinal_matched_clip_repr_multicrop_safeaug_seed42", "quality_anchor_ordinal_matched_clip_repr_seed42", 42),
            ]
        for label, candidate_name, control_name, seed in pairs:
            s = historical_result(f"{ds}_{candidate_name}", seed)
            control = historical_result(f"{ds}_{control_name}", seed)
            if not s or not control:
                continue
            cs, ct = s["cfg"], control["cfg"]
            if cs.get("split_seed",cs.get("seed")) != ct.get("split_seed",ct.get("seed")):
                continue
            cells = [LABELS[ds], html.escape(label), str(seed)]
            cells += [fmt(value(s,k)) for k in METRICS]
            cells += [delta(value(s,k),value(control,k),True) for k in METRICS]
            cells += [html.escape(control["run_id"]), html.escape(s["run_id"])]
            historical_rows.append(cells)

    best_rows = []
    for ds in DATASETS:
        group = [s for s in tests if dataset(s) == ds]
        for k in METRICS:
            valid = [s for s in group if value(s,k) is not None]
            if not valid:
                continue
            best = max(valid, key=lambda s: value(s,k))
            best_rows.append([LABELS[ds], k.removeprefix("test_"), fmt(value(best,k)), html.escape(best["run_id"]), str(best["cfg"].get("seed", "未知")), html.escape(protocol(best))])

    pre_rows = []
    for s in pretrains:
        last = s["epochs"][-1] if s["epochs"] else {}
        lm = last.get("metrics", {})
        status = "有final日志" if "final" in s["events"] else "已early_stop；无final日志" if "early_stop" in s["events"] else "无完成证据；不猜测原因"
        row = [html.escape(s["run_id"]), str(s["attempt"]), status, str(last.get("epoch", "—"))]
        for metric in ("val_quality_proxy_SROCC", "val_align_proxy_SROCC", "val_artifact_anchor_SROCC", "val_aesthetics_anchor_SROCC"):
            candidates = [e for e in s["epochs"] if isinstance(e["metrics"].get(metric), (int,float)) and math.isfinite(e["metrics"][metric])]
            if candidates:
                e = max(candidates, key=lambda e:e["metrics"][metric])
                row.append(f"{e['metrics'][metric]:.4f} @ep{e['epoch']}；last={lm.get(metric, 0):.4f}")
            else:
                row.append("—")
        row += [html.escape(s["source_csv"])]
        pre_rows.append(row)

    main_notes = [
        "当前不是每个模块都有独立有效性证据：启用/非零loss只说明参与优化，不等于带来下游增益。",
        "当前迁移路径：CLIP visual + ResNet + Fusion + text_to_fused + Align adapters，以及 quality_image_projector → Quality Adapter；IQA预测头在representation加载时重新初始化。",
        "质量评分损失作用于Fusion/global-local质量头，Component Anchor与Image–Image作用于私有质量projector；共享编码器相连，但并非完全相同的监督路径。",
        "保持两输出协议，不把历史合成Final MOS当成当前人工质量/一致性成绩，也不因旧Final指标下降就恢复不具备独立人工标签的Final监督。",
        "最高优先级：已有checkpoint导出s_quality_global、s_quality_patch、quality_mix和人工MOS，在验证集比较全局、局部、原融合；再仅用验证集校准α，锁定后评估测试集。此分析尚未执行模型推理。",
        "第二优先级：补Fusion/CLIP/ResNet/共享patch投影的分模块梯度方向与大小，区分任务冲突和单任务泛化问题。不要把1K质量下降归因于不存在的Align训练。",
        "第三优先级：固定同一预训练checkpoint、下游协议和split，分别消融Prompt–Answer、词级反馈、区域反馈、quality auxiliary、Anchor或Ordinal、Image–Image；每次只改一项。先用一个固定种子探索，保留有效单次候选，但完整记录失败对照。",
        "低Adapter LR不设默认；尚未证实单纯遗忘假设。预训练不必立即重跑。1K没有Align标签，不能测试Quality/Align PCGrad收益。",
    ]
    findings = [
        "Anchor过拟合：当前预训练末轮Artifact train/val SROCC=0.9152/0.7023，Aesthetics=0.9101/0.6355；best Component在ep2，best Quality在ep15，best Align在ep10。不可将best Component视为通用最佳。",
        "当前预训练PCGrad是逐参数版本；下游global_shared是整共享向量版本。两者不是同一个实现，不能把下游收益直接归因于预训练PCGrad。",
        "_forward_three_views中Prompt和AlignPrompt对同一输入重复调用_project_text，且语义img1/img2投影仍计算，训练端augmented image-text loss随后被clean loss替代。存在计算冗余；Dropout/BatchNorm使直接复用不一定数值等价，应单独优化验证。",
        "train_cm_pretrain.py尾部构造final_metrics但没有logger.log(final)，解释了近期预训练CSV缺final行；early_stop和epoch记录仍能证明训练终止。未修改此代码。",
        "predict_score仍返回s_final，尽管当前训练/评估关闭Final；旧API/文档残留应另行清理，不影响本次按Quality/Align解析的成绩。",
        "quality_adapter_mode=replace时quality_global_shared与quality_global_adapter是同一分数，quality_adapter_gate固定0，不代表Adapter没起作用，也不是两条独立分支。",
        "原HTML把No Quality Auxiliary 1K的PLCC下降写成上升；还以历史Final成绩宣称当前最强，现已过时。原文件保留不覆盖。",
        "发现run_id名为seed42、实际config_json.seed=3407的历史记录；按run_start拆分尝试、读取真实配置，不从名字猜seed。不完整/重复运行保留，不能当独立统计重复。",
        "三个decoupled下游YAML默认仍指向best_component_anchor_pretrain.pth，而近期对照命令覆写为best_quality_pretrain.pth。后续若漏掉checkpoint参数，会切回另一预训练epoch；1K源YAML的seed=123与默认输出目录seed42也不一致，应以日志实际配置为准。",
    ]
    summary = f"扫描{len(files)}个CSV，{len(sessions)}次run_start分段运行；{len(tests)}条best_test对应运行，{len(pretrains)}次预训练运行；解析异常{len(errors)}。"
    md = ["## Material Passport", "", f"Origin Skill: experiment-agent；Origin Mode: validate；Origin Date: {date.today().isoformat()}；Verification Status: ANALYZED；Version Label: project_audit_v1。", "", "# 项目模块与实验审计", "", summary, "", "仅生成报告和汇总脚本；未训练、未加载checkpoint、未修改模型/训练配置。未重新计算图像级相关性，数值来自已有CSV。", "", "## 核心结论与下一步", ""]
    md += [f"{i}. {text}" for i,text in enumerate(main_notes,1)]
    md += ["", "## 模块审计", "", "| 模块 | 当前状态 | 证据与限制 | 建议 | 代码证据 |", "|---|---|---|---|---|"]
    md += ["| " + " | ".join(row) + " |" for row in MODULES]
    md += ["", "## 当前解耦方案全部对照", "", "| 数据集 | 方法 | seed | Q PLCC | Q SROCC | A PLCC | A SROCC |", "|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(row) + " |" for row in current_md]
    md += ["", "## 代码与记录问题", ""] + [f"- {text}" for text in findings]
    md += ["", "## 统计检查", "", "11/11 checked。CAUTION：按数据集/任务/真实seed与split分别比较，防止聚合反转；不从运行均值推断单张图效果；筛选/碰撞偏差缺原始图像证据；基率不适用；保留未完成尝试与重复记录；最佳单种子和反复调参具有极值、均值回归及探索路径偏差；梯度/损失相关不等于泛化因果证据；反向因果未见适用证据。未计算p值/置信区间，未声称显著性，未重跑验证。", "", "历史单次最高分属于探索上限，不是同一统一方法，也不是公平SOTA比较。接受一个种子上升作为探索候选，不必等所有种子上升；正式结论必须披露选择方式。", "", f"完整成绩：{csv_path.name}；彩色表：{html_path.name}。"]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    content = [f"<h1>项目模块与全部实验审计</h1><p>{html.escape(summary)}</p>", "<p>ANALYZED：来自已有CSV；没有重新训练/推理。蓝色↑提升、红色↓下降，差值对照明确；不是显著性检验。</p>", "<h2>下一步</h2><ol>" + "".join(f"<li>{html.escape(t)}</li>" for t in main_notes) + "</ol>", "<h2>模块是否起作用</h2>", table(["模块", "当前状态", "证据与限制", "建议", "代码证据"], [[html.escape(x) for x in row] for row in MODULES]), "<h2>当前解耦方案：同种子对照</h2>", table(["数据集","方法","seed","Q PLCC","Q SROCC","A PLCC","A SROCC","ΔQ PLCC","ΔQ SROCC","ΔA PLCC","ΔA SROCC","对照","实际配置差异"], current_rows), "<h2>当前已完成三种子统计</h2><p>均值±样本标准差；训练种子变化不等于多个独立数据划分。保留最佳单次候选，均值不作为必须全部上涨的门槛。</p>", table(["数据集","方法","n","Q PLCC","Q SROCC","A PLCC","A SROCC"],mean_rows), "<h2>历史单指标最高分（探索上限，不是统一方案）</h2>", table(["数据集","指标","值","运行","真实seed","协议"], best_rows), "<h2>代码/日志问题</h2><ul>" + "".join(f"<li>{html.escape(t)}</li>" for t in findings) + "</ul>", "<h2>全部下游成绩</h2><p>每条best_test对应一次运行尝试；1K Align为空，不把0当有效相关性。历史Final只在CSV独立列保存。保留重复运行，不作为独立重复计算。</p><input id='search' placeholder='筛选方法、数据集、seed、检查点…' style='width:70%;padding:10px'>"]
    full = []
    for r in flat_rows:
        full.append([html.escape(str(r[k])) if r[k] is not None else "—" for k in ("dataset","method","seed","split_seed","attempt","protocol","best_epoch")] + [fmt(r[k]) for k in ("quality_PLCC","quality_SROCC","align_PLCC","align_SROCC")] + [html.escape(str(r[k])) for k in ("pretrained_checkpoint","selection_metric","source_csv","source_line")])
    content += ["<div id='allruns'>" + table(["数据集","方法","seed","split","尝试","协议","best epoch","Q PLCC","Q SROCC","A PLCC","A SROCC","预训练checkpoint","验证选择指标","来源CSV","记录行"],full) + "</div>", "<h2>历史模块候选实验：分别看Quality和Align</h2><p>真实seed、split一致才显示差值；这些是历史配置/预训练检查点对照，不自动视为当前框架的严格单项消融。Δ不能跨任务抵消，也不使用合成Final成绩决定是否有效。</p>", table(["数据集","改变","seed","Q PLCC","Q SROCC","A PLCC","A SROCC","ΔQ PLCC","ΔQ SROCC","ΔA PLCC","ΔA SROCC","对照运行","候选运行"],historical_rows), "<h2>全部预训练日志</h2><p>各列为该验证指标的独立最大值及所在epoch，后面是最后一轮值；不同列不一定来自同一checkpoint。缺final日志按early_stop判断停止，不把它伪装成测试结果。</p>", table(["运行","尝试","停止证据","last epoch","Quality最大/末轮","Align最大/末轮","Artifact Anchor最大/末轮","Aesthetics Anchor最大/末轮","来源CSV"],pre_rows)]
    incomplete = [s for s in sessions if "best_test" not in s["events"] and s["stage"] != "pretrain"]
    content += ["<details><summary>无best_test的非预训练运行（不猜测失败原因）</summary>" + table(["运行","尝试","来源","已有事件"],[[html.escape(s["run_id"]),str(s["attempt"]),html.escape(s["source_csv"]),html.escape(",".join(s["events"]))] for s in incomplete]) + "</details>", "<p>统计风险11/11 checked；未执行重跑，未计算p值/置信区间，收益为描述性分析；反复测试选择最佳种子不能作为无偏泛化证明。</p>"]
    css = "body{font:14px Arial,'Microsoft YaHei',sans-serif;margin:24px;color:#243247}h1{font-size:24px}h2{margin-top:32px}table{border-collapse:collapse;width:100%;font-size:12px;margin:14px 0;display:block;overflow-x:auto}th,td{border:1px solid #cbd5e1;padding:7px;vertical-align:top;max-width:400px;overflow-wrap:anywhere}th{background:#eef3f8;position:sticky;top:0}tr:nth-child(even){background:#fafafa}.up{color:#1565c0;font-weight:bold}.down{color:#c62828;font-weight:bold}.flat{color:#64748b}li{margin-bottom:9px}"
    script = "document.getElementById('search').addEventListener('input',e=>{let q=e.target.value.toLowerCase();document.querySelectorAll('#allruns tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))});"
    html_path.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>项目模块与实验审计</title><style>'+css+'</style></head><body>' + "\n".join(content) + '<script>'+script+'</script></body></html>',encoding="utf-8")
    print(json.dumps({"csv_files":len(files),"sessions":len(sessions),"best_test_runs":len(tests),"pretrain_runs":len(pretrains),"parse_errors":errors,"reports":[str(p) for p in (html_path,csv_path,md_path)]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

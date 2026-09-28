#!/usr/bin/env python3
"""Inventory saved SurfEmb evaluations without running inference or training."""

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "submodules/RoboPEPP/logs"
STATS = ("mean", "median", "rmse", "p90", "p95", "max")


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


def csv_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def dump_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def fmt(value):
    value = number(value)
    return "--" if value is None else f"{value:.3f}"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows),
        "",
    ]


def checkpoint_metadata(path, cache):
    if not path:
        return {}
    if path not in cache:
        p = Path(path)
        if not p.is_file():
            cache[path] = {"exists": False}
        else:
            ckpt = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
            args = ckpt.get("args", {})
            args = vars(args) if hasattr(args, "__dict__") else args
            cache[path] = {
                "exists": True,
                "iter_now": ckpt.get("iter"),
                "epoch_now": ckpt.get("epoch"),
                "args": args,
            }
            del ckpt
    return cache[path]


def family(args):
    name = args.get("name", "")
    if "augfix" in name:
        return "Wrist / LND pre-refine / Augfix / " + args.get("surfemb_similarity", "unknown")
    if "fullsurfneg" in name:
        return "Wrist / LND pre-refine / full-surface negatives"
    if "prerefine" in name:
        return "Wrist / LND pre-refine / visible negatives"
    if "wristonly_lnd" in name:
        return "Wrist / LND refined / visible negatives"
    if "wristonly_visible" in name:
        return "Wrist / RARP+LND refined / visible negatives"
    if "full_corrected" in name:
        return "Full ResNet / triangle positives"
    if "keypoint_crop_part206020" in name:
        return "Full DINO / point positives"
    if "resnet_crop224" in name:
        return "Full ResNet / point positives"
    return name or "unknown"


def training_fields(meta):
    args = meta.get("args", {})
    memory = args.get("lnd_refine_memory", "")
    negatives = args.get("wrist_negative_source")
    neg_evidence = "checkpoint args"
    if negatives is None:
        # Historical wrist runners predate the negative-source argument.
        name = args.get("name", "")
        if "wristonly_visible" in name or "wristonly_lnd_visible" in name or "wristonly_lnd_prerefine_p" in name:
            negatives, neg_evidence = "visible", "historical runner/log and run family"
        else:
            negatives, neg_evidence = "not recorded", "not recorded in checkpoint"
    return {
        "train_family": family(args),
        "training_run": args.get("name", "unknown"),
        "training_data": "RARP+LND" if args.get("train_dataset_names") else ("LND-only" if args else "unknown"),
        "train_pose": "pre-refine" if "action_gt_full" in memory else ("refined" if "refine_memory" in memory else "unknown"),
        "train_n_pos": args.get("surfemb_n_pos"),
        "train_n_neg": args.get("surfemb_n_neg"),
        "train_negative_source": negatives,
        "negative_source_evidence": neg_evidence,
        "train_similarity": args.get("surfemb_similarity", "legacy raw_dot"),
        "train_temperature": args.get("surfemb_temperature"),
        "batch_per_gpu": args.get("batch_size"),
        "workers_per_rank": args.get("num_workers"),
    }


def check_lnd_csv(path, model, result):
    per_frame = path.parent / "surfemb_per_frame.csv"
    if not per_frame.is_file():
        return {"csv_check": "missing"}
    rows = [row for row in csv_rows(per_frame) if row.get("model") == model]
    valid = [r for r in rows if r.get("status") == "ok"]
    metrics = result.get("metrics", {})
    errors = []
    for col in ("canonical_trans_err_mm", "canonical_rot_err_deg"):
        if col not in metrics:
            continue
        values = [number(r.get(col)) for r in valid]
        values = np.array([v for v in values if v is not None], dtype=float)
        saved = metrics[col]
        if len(values) != saved["count"]:
            errors.append(f"{col}: CSV count {len(values)} != summary {saved['count']}")
        if len(values):
            computed = {"mean": values.mean(), "median": np.median(values), "rmse": np.sqrt(np.mean(values ** 2))}
            for stat, value in computed.items():
                if stat in saved and not np.isclose(value, saved[stat], rtol=1e-7, atol=1e-6):
                    errors.append(f"{col}/{stat}: CSV {value} != summary {saved[stat]}")
    result = {"csv_check": "ok" if not errors else "; ".join(errors), "csv_valid_rows": len(valid)}
    if valid:
        for key in ("crop_mask_source", "wrist_roi_source", "predicted_wrist_mask_mode", "pose_estimator"):
            result["csv_" + key] = ",".join(sorted({r[key] for r in valid if r.get(key)}))
    return result


def collect_lnd(paths, cache):
    rows = []
    for path, data in paths:
        if "surfemb" not in data:
            continue
        protocol = data["protocol"]
        for model, result in data["surfemb"].items():
            metrics = result["metrics"]
            meta = checkpoint_metadata(result.get("checkpoint"), cache)
            check = check_lnd_csv(path, model, result)
            row = {
                "source": str(path), "model": model,
                "dataset": data["dataset"], "eval_iteration": result.get("checkpoint_iter"),
                "valid": result.get("success"), "total": result.get("total"),
                "excluded_frame_ids": json.dumps(protocol.get("excluded_frame_ids")),
                "crop": check.get("csv_crop_mask_source") or protocol["crop"],
                "matching_roi": check.get("csv_wrist_roi_source") or protocol.get("wrist_roi_source", "not recorded"),
                "estimator": check.get("csv_pose_estimator") or protocol.get("pose_estimator", "legacy probability"),
                "pose_protocol": protocol.get("pose"),
                "similarity": protocol.get("correspondence_similarity", "not recorded"),
                "eval_temperature": protocol.get("correspondence_temperature"),
                "rotation_ensemble": protocol.get("rotation_ensemble"),
                "checkpoint": result.get("checkpoint"),
                "checkpoint_iter_now": meta.get("iter_now"),
                "checkpoint_iter_matches_eval": meta.get("iter_now") == result.get("checkpoint_iter"),
                **training_fields(meta), **check,
            }
            for prefix, key in (("t", "canonical_trans_err_mm"), ("r", "canonical_rot_err_deg")):
                for stat in STATS:
                    row[f"{prefix}_{stat}"] = metrics.get(key, {}).get(stat)
            rows.append(row)
    return rows


def collect_rarp(paths, cache):
    rows = []
    for path, data in paths:
        if "evaluation_config" not in data and "models" not in data:
            continue
        config = data.get("evaluation_config", data.get("config", {}))
        for model, result in data.get("models", data).items():
            if not isinstance(result, dict) or "checkpoint" not in result:
                continue
            meta = checkpoint_metadata(result["checkpoint"], cache)
            groups = {"articulated": result["metrics"]} if "metrics" in result else {
                key: value for key, value in result.items()
                if isinstance(value, dict) and any(isinstance(v, dict) and "mean" in v for v in value.values())
            }
            for group, metrics in groups.items():
                row = {
                    "source": str(path), "model": model, "group": group,
                    "dataset": config.get("dataset"),
                    "scope": "smoke" if "smoke" in str(path) else "full evaluation",
                    "total": result.get("total"),
                    "eval_iteration": result.get("checkpoint_iter"),
                    "checkpoint": result["checkpoint"],
                    "protocol": json.dumps(config, sort_keys=True),
                    **training_fields(meta),
                }
                for metric, stats in metrics.items():
                    if isinstance(stats, dict) and "mean" in stats:
                        for stat, value in stats.items():
                            row[f"{metric}/{stat}"] = value
                rows.append(row)
    return rows


def collect_training(cache):
    rows = []
    run_names = sorted({meta.get("args", {}).get("name") for meta in cache.values() if meta.get("args", {}).get("name")})
    for name in run_names:
        path = LOGS / f"{name}_train.log"
        if not path.is_file():
            continue
        iteration = None
        train = None
        val = {}
        for line in path.read_text(errors="replace").splitlines():
            if line.startswith("iter "):
                match = re.match(r"iter (\d+)", line)
                if match:
                    iteration = int(match[1])
                    train = dict(re.findall(r"\|\s*(loss|nce|mask)\s+([\d.eE+-]+)", line))
            if line.startswith(("VAL_LND ", "VAL_RARP ")):
                dataset, _ = line.split(" ", 1)
                values = dict(re.findall(r"(val_\w+)\s+([\d.eE+-]+)", line))
                val[dataset] = (iteration, values)
        if train:
            for dataset, (val_iter, values) in val.items():
                rows.append({
                    "run": name, "last_logged_train_iteration": iteration,
                    "last_train_nce": train.get("nce"), "last_train_bce": train.get("mask"),
                    "validation_stream": dataset, "last_val_iteration": val_iter,
                    "last_val_nce": values.get("val_nce"), "last_val_bce": values.get("val_mask_bce"),
                    "last_val_total": values.get("val_total"), "source": str(path),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "logs/surfemb_experiment_summary_20260908")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = subprocess.check_output(
        ["rg", "--files", "-g", "*summary*.json", str(LOGS), str(ROOT / "logs"), str(ROOT / "eval_outputs")],
        text=True,
    ).splitlines()
    sources = []
    for name in sorted(files):
        path = Path(name)
        if args.output in path.parents or not any(token in name.lower() for token in ("surfemb", "augfix")):
            continue
        data = read_json(path)
        if isinstance(data, dict):
            sources.append((path, data))
    cache = {}
    lnd = collect_lnd(sources, cache)
    rarp = collect_rarp(sources, cache)
    training = collect_training(cache)
    for index, row in enumerate(lnd, 1):
        row["id"] = f"L{index:03d}"
    dump_csv(args.output / "lnd_all_evaluations.csv", lnd)
    dump_csv(args.output / "rarp_all_evaluations.csv", rarp)
    dump_csv(args.output / "training_loss_endpoints.csv", training)
    (args.output / "checkpoint_metadata.json").write_text(json.dumps(cache, indent=2, default=str))
    (args.output / "source_summary_archive.json").write_text(json.dumps({str(p): d for p, d in sources}, indent=2))

    # Link every displayed number to the immutable evaluation record, not last.pt.
    lines = [
        "# SurfEmb full-patch and wrist zoom-in experiment inventory", "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} from saved local evaluations; no inference was run.", "",
        "Full patch means the entire instrument instance is cropped to 224x224. Wrist zoom-in means the wrist ROI is cropped to 224x224.", "",
        "LND tables report wrist translation in mm and rotation in degrees. T/R are per-frame error means, not RMSE. Median, RMSE and tails are in the CSV. Rotation is the saved symmetry-canonicalized geodesic error.", "",
        "Valid/total denotes accepted pose estimates, not an accuracy-threshold success rate. Means exclude failed poses. Results with different valid subsets are not directly comparable.", "",
        "Most LND records use TEST excluding frame 210 (372 frames). Original detector-based evaluations use 373. Crop source and correspondence matching ROI are separate controls.", "",
        "The original full/wrist comparison uses the same 4 GPUs and batch 56 per GPU. Full has 1024 positives/negatives with 614 wrist samples and other-part negatives; wrist-only has 614/614 and visible-wrist negatives. Crop is not the only changed variable.", "",
        "Historical full and wrist runs use RARP+LND and refined LND TRAIN poses. Later LND-only pre-refine, full-surface-negative, Augfix and cosine runs are separate experiments.", "",
        "The full baseline reached last 60000 and the mixed-data wrist-only baseline saved last 50000. Their highest saved LND pose evaluations found here are 43000 and 46000 respectively. Do not relabel historical scores with the current last.pt iteration.", "",
        "The old full model uses point-projected positive labels; the corrected full model uses triangle-rasterized per-pixel labels. Code changes after a checkpoint do not change how it was trained.", "",
        "## LND results by training family", "",
    ]
    grouped = defaultdict(list)
    for row in lnd:
        grouped[row["train_family"]].append(row)
    for group, rows in sorted(grouped.items()):
        lines.extend([f"### {group}", ""])
        lines.extend(table(
            ["Record", "Model", "Iter", "Crop", "Matching ROI", "Estimator", "Valid/total", "T mean mm", "R mean deg"],
            [[f"[{r['id']}]({r['source']})", r["model"], r["eval_iteration"], r["crop"], r["matching_roi"], r["estimator"], f"{r['valid']}/{r['total']}", fmt(r["t_mean"]), fmt(r["r_mean"])] for r in sorted(rows, key=lambda x: (x["eval_iteration"] or 0, x["source"]))],
        ))

    lines.extend(["## Matched full/wrist checkpoint comparisons", "",
        "Both models in each pair use RARP+LND refined training, triangle positives and Top-K RANSAC/LM. Matching ROI is fixed within the pair; training crop and negative distribution differ. Checkpoint iterations are close but not identical.", ""])
    paired = [
        ("18k/19k, predicted ROI", "surfemb_current_last_lnd_eval_20260804/full_instance_crop/summary.json", "surfemb_current_last_lnd_eval_20260804/wrist_crop/summary.json"),
        ("21k/22k, predicted ROI", "surfemb_current_last_lnd_eval_20260804_iter21000_22000/full_instance_crop/summary.json", "surfemb_current_last_lnd_eval_20260804_iter21000_22000/wrist_crop/summary.json"),
        ("21k/22k, GT ROI", "surfemb_current_last_lnd_eval_20260804_iter21000_22000/full_instance_crop_gt_wrist_roi/summary.json", "surfemb_current_last_lnd_eval_20260804_iter21000_22000/wrist_crop_gt_wrist_roi/summary.json"),
        ("36k/38k, GT ROI", "surfemb_training_progress_lnd_20260805/full_last36k_gtroi/summary.json", "surfemb_training_progress_lnd_20260805/wrist_last38k_gtroi/summary.json"),
        ("43k/43k, GT ROI, different accepted subsets", "surfemb_crop_checkpoint_ablation_20260805/full43k_gtroi/summary.json", "surfemb_crop_checkpoint_ablation_20260805/wrist43k_gtroi/summary.json"),
    ]
    compare_rows = []
    for label, left, right in paired:
        a = next(r for r in lnd if r["source"] == str(LOGS / left))
        b = next(r for r in lnd if r["source"] == str(LOGS / right))
        compare_rows.append([label, f"{fmt(a['t_mean'])} / {fmt(a['r_mean'])}", f"{fmt(b['t_mean'])} / {fmt(b['r_mean'])}", f"{a['valid']} / {b['valid']}"])
    lines.extend(table(["Pair", "Full T / R", "Wrist T / R", "Full / wrist valid"], compare_rows))
    lines.extend(["## RARP evaluations", "",
        "These are full-instrument training/crop models. A wrist-only fitting ablation here does not mean a wrist zoom-in model was trained. Smoke subsets are indexed separately in rarp_all_evaluations.csv.", ""])
    rarp_display = []
    for r in rarp:
        if r["scope"] == "smoke":
            continue
        tkey = next((k for k in r if "trans_err_mm/mean" in k and "wrist" in k), None)
        rkey = next((k for k in r if "rot_err_deg/mean" in k and "wrist" in k), None)
        if not tkey or not rkey:
            continue
        nkey = tkey.replace("/mean", "/count")
        rarp_display.append([f"[{Path(r['source']).parent.name}]({r['source']})", r["model"], r["group"], f"{r.get(nkey)}/{r['total']}", fmt(r[tkey]), fmt(r[rkey]), fmt(r.get("joint_mae_deg/mean"))])
    lines.extend(table(["Source", "Model", "Fitting", "N / manifest", "T mean mm", "R mean deg", "Joint MAE deg"], rarp_display))
    reference = LOGS / "rarp_pose_comparison_surfemb_topk_vs_hcce_robopepp_lastpt/comparison.md"
    if reference.exists():
        lines.extend(["### Same 2301-sample historical RARP manifest", "", f"Source: [comparison]({reference})", "", reference.read_text(), ""])

    lines.extend(["## Training and validation loss endpoints", "",
        "Training values are logged running averages and validation values are stream aggregates. Full-instrument LND validation uses missing-action placeholders and is diagnostic. Old runs use different label/sampling/validation rules; do not compare all NCE numbers as one metric. Raw-dot and cosine/temperature losses are different objectives.", ""])
    lines.extend(table(["Run", "Train iter", "Train NCE", "Val stream", "Val iter", "Val NCE"], [[f"[{r['run']}]({r['source']})", r["last_logged_train_iteration"], r["last_train_nce"], r["validation_stream"], r["last_val_iteration"], r["last_val_nce"]] for r in training]))
    lines.extend(["## Checkpoint provenance and verification", "",
        f"Indexed {len(lnd)} LND model/evaluation rows, {len(rarp)} RARP model/fitting rows (including smoke), and {len(sources)} summary files. Every LND mean/median/RMSE is checked against its per-frame CSV.", "",
        "Some best.pt/last.pt paths were subsequently overwritten. A mismatch below does not invalidate the saved CSV; reproducing that historical evaluation requires the corresponding iter checkpoint/snapshot.", ""])
    mismatches = [r for r in lnd if not r["checkpoint_iter_matches_eval"]]
    lines.extend(table(["Record", "Evaluated iter", "Current file iter", "Checkpoint"], [[r["id"], r["eval_iteration"], r["checkpoint_iter_now"], r["checkpoint"]] for r in mismatches]))
    checks = [r for r in lnd if r["csv_check"] != "ok"]
    lines.extend(["CSV consistency exceptions: " + str(len(checks)), ""])
    for r in checks:
        lines.extend([f"- {r['id']}: {r['csv_check']}"])
    lines.extend(["", "## Complete sources", ""])
    for p, _ in sources:
        lines.append(f"- [{p.relative_to(ROOT)}]({p})")
    lines.extend(["", "Unreported settings remain 'not recorded'. Current script defaults are not evidence of historical launch arguments.", ""])
    (args.output / "REPORT.md").write_text("\n".join(lines))
    audit = {
        "lnd_rows": len(lnd), "rarp_rows": len(rarp), "source_summaries": len(sources),
        "csv_exceptions": [{"id": r["id"], "source": r["source"], "detail": r["csv_check"]} for r in checks],
        "mutable_checkpoint_references": len(mismatches),
        "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in sources},
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({k: v for k, v in audit.items() if k != "sources_sha256"}, indent=2), flush=True)
    print(args.output / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()

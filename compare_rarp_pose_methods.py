import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


HISTORICAL_METHODS = {
    "robopepp_direct": "robopepp_direct",
    "robopepp_keypoint_pnp": "robopepp_pnp",
    "hcce_direct": "hcce_direct",
    "hcce_keypoint_pnp": "hcce_kp_pnp",
    "hcce_fit": "hcce_fit",
}
METRICS = ("wrist_trans_err_mm", "wrist_rot_err_deg", "joint_mae_deg")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def stats(values):
    values = np.asarray([float(value) for value in values if finite(value)], dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "rmse": float("nan")}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


def normalize_surfemb(row):
    return {
        "status": row.get("status", "missing"),
        "wrist_trans_err_mm": row.get("wrist_trans_err_mm"),
        "wrist_rot_err_deg": row.get("wrist_rot_err_deg"),
        "joint_mae_deg": row.get("joint_mae_deg"),
    }


def normalize_historical(row, prefix):
    trans_m = row.get(f"{prefix}_trans_err_m")
    return {
        "status": row.get(f"{prefix}_status", "missing"),
        "wrist_trans_err_mm": float(trans_m) * 1000.0 if finite(trans_m) else float("nan"),
        "wrist_rot_err_deg": row.get(f"{prefix}_rot_err_deg"),
        "joint_mae_deg": row.get(f"{prefix}_joint_mae_deg"),
    }


def method_summary(values, selected_indices):
    selected = [values[index] for index in selected_indices]
    valid = [item for item in selected if item["status"] == "ok"]
    return {
        "total": len(selected),
        "success": len(valid),
        "success_rate": float(len(valid) / max(1, len(selected))),
        "metrics": {metric: stats([item[metric] for item in valid]) for metric in METRICS},
    }


def write_markdown(path, summary):
    lines = [
        "# RARP needleGrasping pose comparison",
        "",
        f"- historical manifest samples: {summary['num_manifest_samples']}",
        f"- all-method common successes: {summary['num_strict_common_successes']}",
        "- translation: mm; rotation and joint MAE: degree",
        "",
        "## Same manifest, per-method valid samples",
        "",
        "| method | valid | trans mean / median | rot mean / median | joint mean / median |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, item in summary["manifest_comparison"].items():
        trans = item["metrics"]["wrist_trans_err_mm"]
        rot = item["metrics"]["wrist_rot_err_deg"]
        joint = item["metrics"]["joint_mae_deg"]
        lines.append(
            f"| {method} | {item['success']}/{item['total']} | "
            f"{trans['mean']:.3f} / {trans['median']:.3f} | "
            f"{rot['mean']:.3f} / {rot['median']:.3f} | "
            f"{joint['mean']:.3f} / {joint['median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict all-method common successes",
            "",
            "| method | N | trans mean / median | rot mean / median | joint mean / median |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, item in summary["strict_common_comparison"].items():
        trans = item["metrics"]["wrist_trans_err_mm"]
        rot = item["metrics"]["wrist_rot_err_deg"]
        joint = item["metrics"]["joint_mae_deg"]
        lines.append(
            f"| {method} | {item['success']} | "
            f"{trans['mean']:.3f} / {trans['median']:.3f} | "
            f"{rot['mean']:.3f} / {rot['median']:.3f} | "
            f"{joint['mean']:.3f} / {joint['median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "HCCE/RoboPEPP keypoint PnP changes wrist SE(3) only; its joint angles come from the same model's direct action head.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main(args):
    surfemb_rows = read_csv(args.surfemb_csv)
    historical_rows = read_csv(args.historical_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    historical = {int(row["dataset_idx"]): row for row in historical_rows}
    manifest_indices = sorted(historical)
    methods = {}
    for model_name in sorted({row["model"] for row in surfemb_rows}):
        model_rows = {
            int(row["dataset_idx"]): normalize_surfemb(row)
            for row in surfemb_rows
            if row["model"] == model_name
        }
        missing = set(manifest_indices) - set(model_rows)
        if missing:
            raise RuntimeError(f"SurfEmb model {model_name} is missing {len(missing)} manifest samples")
        methods[f"surfemb_{model_name}_topk_ransac"] = model_rows

    for method_name, prefix in HISTORICAL_METHODS.items():
        methods[method_name] = {
            index: normalize_historical(row, prefix) for index, row in historical.items()
        }

    strict_indices = [
        index
        for index in manifest_indices
        if all(method[index]["status"] == "ok" for method in methods.values())
    ]
    summary = {
        "surfemb_csv": str(Path(args.surfemb_csv).resolve()),
        "historical_csv": str(Path(args.historical_csv).resolve()),
        "num_manifest_samples": len(manifest_indices),
        "num_strict_common_successes": len(strict_indices),
        "manifest_comparison": {
            name: method_summary(values, manifest_indices) for name, values in methods.items()
        },
        "strict_common_comparison": {
            name: method_summary(values, strict_indices) for name, values in methods.items()
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    write_markdown(output_dir / "comparison.md", summary)
    print((output_dir / "comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--surfemb_csv", required=True)
    parser.add_argument("--historical_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    main(parser.parse_args())

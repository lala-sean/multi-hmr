#!/usr/bin/env python3
"""Summarize RARP pose comparison CSVs into one compact markdown table."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


CURRENT_METHODS = {
    "robopepp_pnp": ("robopepp_pnp", "RoboPEPP keypoint PnP (SE3 only; joints from direct)"),
    "robopepp_direct": ("robopepp_direct", "RoboPEPP direct"),
    "hcce_direct": ("hcce_direct", "Crop-HCCE direct"),
    "hcce_fit": ("hcce_fit", "Crop-HCCE fit"),
    "hcce_kp_pnp": ("hcce_kp_pnp", "Crop-HCCE keypoint PnP (SE3 only; joints from direct)"),
}

HIGHRES_METHODS = {
    "hcce": ("hcce", "High-res densepart HCCE fit"),
    "posehead": ("posehead", "High-res densepart pose head"),
}


def finite_array(values):
    out = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append(val)
    return np.asarray(out, dtype=np.float64)


def stats(arr):
    if arr.size == 0:
        return {"n": 0, "mean": math.nan, "median": math.nan, "rmse": math.nan}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
    }


def read_rows(path):
    if not path or not Path(path).is_file():
        return []
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def has_pose(row):
    value = str(row.get("has_pose_gt", "1")).strip().lower()
    return value not in {"", "0", "false", "nan", "none"}


def status_ok(row, prefix):
    status = str(row.get(f"{prefix}_status", "ok")).strip().lower()
    return status in {"ok", "success", "1", "true", ""}


def joint_mae_deg_from_row(row, prefix):
    direct = row.get(f"{prefix}_joint_mae_deg")
    try:
        val = float(direct)
        if math.isfinite(val):
            return val
    except (TypeError, ValueError):
        pass
    errs = []
    for name in ("alpha", "theta_l", "theta_r"):
        raw = row.get(f"{prefix}_{name}_err_rad")
        try:
            val = abs(float(raw)) * 180.0 / math.pi
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            errs.append(val)
    if not errs:
        return math.nan
    return float(np.mean(errs))


def summarize_method(rows, prefix, display_name, dataset_name=None):
    pose_rows = [row for row in rows if has_pose(row)]
    ok_rows = [row for row in pose_rows if status_ok(row, prefix)]
    trans = finite_array(row.get(f"{prefix}_trans_err_m") for row in ok_rows)
    rot = finite_array(row.get(f"{prefix}_rot_err_deg") for row in ok_rows)
    joint = finite_array(joint_mae_deg_from_row(row, prefix) for row in ok_rows)
    status_rows = [row for row in rows if status_ok(row, prefix)]
    return {
        "dataset": dataset_name or rows[0].get("dataset", "") if rows else (dataset_name or ""),
        "method": display_name,
        "prefix": prefix,
        "rows": len(rows),
        "pose_rows": len(pose_rows),
        "ok_rows": len(ok_rows),
        "ok_all_rows": len(status_rows),
        "trans": stats(trans),
        "rot": stats(rot),
        "joint": stats(joint),
    }


def group_by_dataset(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row.get("dataset", ""), []).append(row)
    return groups


def fmt(value, digits=4):
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def table_for_summaries(title, summaries):
    lines = [f"## {title}", ""]
    lines.append(
        "| dataset | method | pose n | ok n | trans mean/median/rmse m | rot mean/median/rmse deg | joint MAE mean/median/rmse deg |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        trans = s["trans"]
        rot = s["rot"]
        joint = s["joint"]
        lines.append(
            "| {dataset} | {method} | {pose_rows} | {ok_rows} | {tm}/{td}/{tr} | {rm}/{rd}/{rr} | {jm}/{jd}/{jr} |".format(
                dataset=s["dataset"],
                method=s["method"],
                pose_rows=s["pose_rows"],
                ok_rows=s["ok_rows"],
                tm=fmt(trans["mean"]),
                td=fmt(trans["median"]),
                tr=fmt(trans["rmse"]),
                rm=fmt(rot["mean"], 2),
                rd=fmt(rot["median"], 2),
                rr=fmt(rot["rmse"], 2),
                jm=fmt(joint["mean"], 2),
                jd=fmt(joint["median"], 2),
                jr=fmt(joint["rmse"], 2),
            )
        )
    lines.append("")
    return lines


def status_table(title, rows, methods):
    lines = [f"## {title}", ""]
    lines.append("| dataset | method | rows | ok rows | pose gt rows |")
    lines.append("|---|---:|---:|---:|---:|")
    for dataset, dataset_rows in group_by_dataset(rows).items():
        pose_rows = sum(has_pose(row) for row in dataset_rows)
        for prefix, name in methods.values():
            ok_count = sum(status_ok(row, prefix) for row in dataset_rows)
            lines.append(f"| {dataset} | {name} | {len(dataset_rows)} | {ok_count} | {pose_rows} |")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_csv", required=True)
    parser.add_argument("--previous_crop_csv", default="")
    parser.add_argument("--highres_csv", default="")
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    current = read_rows(args.current_csv)
    previous = read_rows(args.previous_crop_csv)
    highres = read_rows(args.highres_csv)

    lines = [
        "# RARP Method Comparison",
        "",
        f"- current_csv: `{args.current_csv}`",
        f"- previous_crop_csv: `{args.previous_crop_csv}`",
        f"- highres_csv: `{args.highres_csv}`",
        "",
        "Metric note: keypoint PnP methods solve only wrist SE3 from heatmap keypoints. Their `alpha/theta_l/theta_r` values are copied from the same model's direct/action head, so their joint MAE is not an independent keypoint-PnP joint estimate.",
        "",
    ]

    current_summaries = []
    for dataset, rows in group_by_dataset(current).items():
        for prefix, name in CURRENT_METHODS.values():
            current_summaries.append(summarize_method(rows, prefix, name, dataset))
    lines.extend(table_for_summaries("Current Crop Run", current_summaries))
    lines.extend(status_table("Current Status Counts", current, CURRENT_METHODS))

    if previous:
        prev_summaries = []
        for dataset, rows in group_by_dataset(previous).items():
            for prefix, name in CURRENT_METHODS.values():
                prev_summaries.append(summarize_method(rows, prefix, name, dataset))
        lines.extend(table_for_summaries("Previous Crop Run", prev_summaries))

    if highres:
        high_summaries = []
        for dataset, rows in group_by_dataset(highres).items():
            for prefix, name in HIGHRES_METHODS.values():
                high_summaries.append(summarize_method(rows, prefix, name, dataset))
        lines.extend(table_for_summaries("Old High-Res Densepart Run", high_summaries))

    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()

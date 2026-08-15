import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "logs/crop_fullcompare_surface32_random_lastpt_needle_test/per_instance.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "logs/hcce_fit_worstcase_analysis_lastpt_needle"


def finite_series(df, col):
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def stat(values):
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def add_gaps(df):
    out = df.copy()
    for metric in ("trans_err_m", "rot_err_deg", "joint_mae_deg", "kp_reproj_rmse_px"):
        fit = f"hcce_fit_{metric}"
        pnp = f"hcce_kp_pnp_{metric}"
        if fit in out.columns and pnp in out.columns:
            out[fit] = finite_series(out, fit)
            out[pnp] = finite_series(out, pnp)
            out[f"gap_{metric}"] = out[fit] - out[pnp]
    out["fit_bad_score"] = (
        finite_series(out, "hcce_fit_trans_err_m") / 0.01
        + finite_series(out, "hcce_fit_rot_err_deg") / 20.0
        + finite_series(out, "hcce_fit_joint_mae_deg") / 15.0
    )
    out["fit_vs_pnp_gap_score"] = (
        finite_series(out, "gap_trans_err_m") / 0.005
        + finite_series(out, "gap_rot_err_deg") / 10.0
        + finite_series(out, "gap_joint_mae_deg") / 10.0
    )
    return out


def annotate_case(row):
    tags = []
    status = str(row.get("hcce_fit_status", ""))
    if status != "ok":
        if "Not enough shaft" in status:
            tags.append("shaft_missing_after_x_filter")
        if "maximum number" in status:
            tags.append("optim_not_converged")
        return tags or ["fit_failed"]

    fit_rot = float(row.get("hcce_fit_rot_err_deg", np.nan))
    pnp_rot = float(row.get("hcce_kp_pnp_rot_err_deg", np.nan))
    gap_rot = float(row.get("gap_rot_err_deg", np.nan))
    fit_trans = float(row.get("hcce_fit_trans_err_m", np.nan))
    gap_trans = float(row.get("gap_trans_err_m", np.nan))
    gap_joint = float(row.get("gap_joint_mae_deg", np.nan))
    reproj_all = float(row.get("hcce_fit_reproj_rmse_all_px", np.nan))
    reproj_wg = float(row.get("hcce_fit_reproj_rmse_wrist_gripper_px", np.nan))
    reproj_shaft = float(row.get("hcce_fit_reproj_rmse_shaft_px", np.nan))
    wrist_iou = float(row.get("hcce_model_crop_seg_part_iou_wrist", np.nan))
    gripper_iou = float(row.get("hcce_model_crop_seg_part_iou_gripper", np.nan))
    shaft_iou = float(row.get("hcce_model_crop_seg_part_iou_shaft", np.nan))
    wrist_pts = float(row.get("hcce_fit_wrist_candidate_points", np.nan))
    grip_pts = float(row.get("hcce_fit_gripper_candidate_points", np.nan))
    shaft_pts = float(row.get("hcce_fit_shaft_candidate_points", np.nan))
    kp_score = float(row.get("hcce_kp_hm_score_min", np.nan))
    kp_rmse = float(row.get("hcce_kp_pnp_hm_crop_rmse_px", np.nan))

    if math.isfinite(gap_rot) and gap_rot > 45:
        tags.append("fit_rotation_much_worse")
    if math.isfinite(gap_rot) and gap_rot < -45:
        tags.append("pnp_rotation_much_worse")
    if math.isfinite(fit_rot) and fit_rot > 120:
        tags.append("fit_180deg_flip")
    if math.isfinite(pnp_rot) and pnp_rot > 120:
        tags.append("pnp_180deg_flip")
    if math.isfinite(gap_trans) and gap_trans > 0.02:
        tags.append("fit_translation_much_worse")
    if math.isfinite(gap_joint) and gap_joint > 20:
        tags.append("fit_joint_much_worse")
    if math.isfinite(reproj_all) and reproj_all > 12:
        tags.append("dense_fit_high_reprojection")
    if math.isfinite(reproj_shaft) and math.isfinite(reproj_wg) and reproj_shaft > reproj_wg + 6:
        tags.append("shaft_residual_dominates")
    if math.isfinite(wrist_iou) and wrist_iou < 0.75:
        tags.append("wrist_seg_low")
    if math.isfinite(gripper_iou) and gripper_iou < 0.70:
        tags.append("gripper_seg_low_or_occluded")
    if math.isfinite(shaft_iou) and shaft_iou < 0.85:
        tags.append("shaft_seg_low")
    if math.isfinite(wrist_pts) and wrist_pts < 500:
        tags.append("few_wrist_dense_points")
    if math.isfinite(grip_pts) and grip_pts < 300:
        tags.append("few_gripper_dense_points")
    if math.isfinite(shaft_pts) and shaft_pts < 800:
        tags.append("few_shaft_front_points")
    if math.isfinite(kp_score) and kp_score < 0.02:
        tags.append("low_keypoint_heatmap_score")
    if math.isfinite(kp_rmse) and kp_rmse > 6:
        tags.append("keypoint_heatmap_bad")
    if not tags:
        tags.append("ordinary")
    return tags


def write_table(df, path, n=50):
    cols = [
        "ordinal",
        "video",
        "frame_id",
        "instance_id",
        "case_tags",
        "hcce_fit_status",
        "fit_vs_pnp_gap_score",
        "fit_bad_score",
        "hcce_fit_trans_err_m",
        "hcce_kp_pnp_trans_err_m",
        "gap_trans_err_m",
        "hcce_fit_rot_err_deg",
        "hcce_kp_pnp_rot_err_deg",
        "gap_rot_err_deg",
        "hcce_fit_joint_mae_deg",
        "hcce_kp_pnp_joint_mae_deg",
        "gap_joint_mae_deg",
        "hcce_fit_reproj_rmse_all_px",
        "hcce_fit_reproj_rmse_wrist_gripper_px",
        "hcce_fit_reproj_rmse_shaft_px",
        "hcce_fit_shaft_candidate_points",
        "hcce_fit_wrist_candidate_points",
        "hcce_fit_gripper_candidate_points",
        "hcce_model_crop_seg_part_iou_mean",
        "hcce_model_crop_seg_part_iou_shaft",
        "hcce_model_crop_seg_part_iou_wrist",
        "hcce_model_crop_seg_part_iou_gripper",
        "hcce_kp_hm_score_min",
        "hcce_kp_pnp_hm_crop_rmse_px",
    ]
    keep = [c for c in cols if c in df.columns]
    df[keep].head(n).to_csv(path, index=False)


def maybe_make_plots(ok, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plot_error": repr(exc)}

    plots = {}
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=160)
    x = ok["hcce_kp_pnp_rot_err_deg"]
    y = ok["hcce_fit_rot_err_deg"]
    colors = np.where(ok["gap_rot_err_deg"] > 45, "tab:red", np.where(ok["gap_rot_err_deg"] < -45, "tab:blue", "0.65"))
    ax.scatter(x, y, s=8, c=colors, alpha=0.65, linewidths=0)
    ax.plot([0, 180], [0, 180], "k--", lw=1)
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 180)
    ax.set_xlabel("HCCE keypoint PnP rotation error (deg)")
    ax.set_ylabel("HCCE dense fit rotation error (deg)")
    ax.set_title("Rotation Error: Dense HCCE Fit vs Keypoint PnP")
    fig.tight_layout()
    path = out_dir / "rot_fit_vs_keypoint_pnp_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    plots["rot_scatter"] = str(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=160)
    ax.hist(ok["gap_rot_err_deg"].dropna(), bins=80, color="0.35")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(45, color="tab:red", lw=1)
    ax.axvline(-45, color="tab:blue", lw=1)
    ax.set_xlabel("rotation gap: HCCE fit - keypoint PnP (deg)")
    ax.set_ylabel("count")
    ax.set_title("Rotation Gap Distribution")
    fig.tight_layout()
    path = out_dir / "rot_gap_hist.png"
    fig.savefig(path)
    plt.close(fig)
    plots["rot_gap_hist"] = str(path)

    return plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top_n", type=int, default=60)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = add_gaps(df)
    df["case_tags"] = [";".join(annotate_case(row)) for _, row in df.iterrows()]

    ok = df[(df["hcce_fit_status"] == "ok") & df["hcce_fit_rot_err_deg"].notna() & df["hcce_kp_pnp_rot_err_deg"].notna()].copy()
    failures = df[df["hcce_fit_status"] != "ok"].copy()

    fit_worse = ok.sort_values("fit_vs_pnp_gap_score", ascending=False)
    fit_bad = ok.sort_values("fit_bad_score", ascending=False)
    rot_worse = ok.sort_values("gap_rot_err_deg", ascending=False)
    pnp_rot_worse = ok.sort_values("gap_rot_err_deg", ascending=True)

    write_table(fit_worse, out_dir / "fit_worse_than_keypoint_pnp_top.csv", args.top_n)
    write_table(fit_bad, out_dir / "hcce_fit_absolute_worst_top.csv", args.top_n)
    write_table(rot_worse, out_dir / "fit_rotation_worse_than_keypoint_pnp_top.csv", args.top_n)
    write_table(pnp_rot_worse, out_dir / "keypoint_pnp_rotation_worse_than_fit_top.csv", args.top_n)
    write_table(failures, out_dir / "hcce_fit_failures.csv", args.top_n)

    selected = pd.concat([
        fit_worse.head(20),
        fit_bad.head(20),
        rot_worse.head(20),
        pnp_rot_worse.head(20),
        failures.head(20),
    ], ignore_index=True)
    selected = selected.drop_duplicates(subset=["video", "frame_id", "instance_id"]).copy()
    selected[["video", "frame_id", "instance_id"]].to_csv(out_dir / "selected_worstcase_manifest.csv", index=False)
    write_table(selected, out_dir / "selected_worstcase_rows.csv", len(selected))

    tag_counts = {}
    for tags in df["case_tags"]:
        for tag in str(tags).split(";"):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    bad = ok[(ok["gap_rot_err_deg"] > 45) | (ok["gap_trans_err_m"] > 0.02) | (ok["gap_joint_mae_deg"] > 20)]
    bad_tag_counts = {}
    for tags in bad["case_tags"]:
        for tag in str(tags).split(";"):
            bad_tag_counts[tag] = bad_tag_counts.get(tag, 0) + 1

    summary = {
        "input_csv": str(input_csv),
        "rows": int(len(df)),
        "fit_ok_rows": int(len(ok)),
        "fit_failure_rows": int(len(failures)),
        "failure_status_counts": failures["hcce_fit_status"].value_counts(dropna=False).to_dict(),
        "metrics": {
            "fit_trans_err_m": stat(ok["hcce_fit_trans_err_m"]),
            "keypoint_pnp_trans_err_m": stat(ok["hcce_kp_pnp_trans_err_m"]),
            "gap_trans_err_m": stat(ok["gap_trans_err_m"]),
            "fit_rot_err_deg": stat(ok["hcce_fit_rot_err_deg"]),
            "keypoint_pnp_rot_err_deg": stat(ok["hcce_kp_pnp_rot_err_deg"]),
            "gap_rot_err_deg": stat(ok["gap_rot_err_deg"]),
            "fit_joint_mae_deg": stat(ok["hcce_fit_joint_mae_deg"]),
            "keypoint_pnp_joint_mae_deg": stat(ok["hcce_kp_pnp_joint_mae_deg"]),
            "gap_joint_mae_deg": stat(ok["gap_joint_mae_deg"]),
        },
        "fractions": {
            "fit_trans_worse": float((ok["gap_trans_err_m"] > 0).mean()),
            "fit_trans_worse_by_5mm": float((ok["gap_trans_err_m"] > 0.005).mean()),
            "fit_rot_worse": float((ok["gap_rot_err_deg"] > 0).mean()),
            "fit_rot_worse_by_5deg": float((ok["gap_rot_err_deg"] > 5).mean()),
            "fit_rot_worse_by_45deg": float((ok["gap_rot_err_deg"] > 45).mean()),
            "keypoint_pnp_rot_worse_by_45deg": float((ok["gap_rot_err_deg"] < -45).mean()),
            "fit_joint_worse": float((ok["gap_joint_mae_deg"] > 0).mean()),
            "fit_joint_worse_by_5deg": float((ok["gap_joint_mae_deg"] > 5).mean()),
        },
        "rot_error_threshold_counts": {
            str(t): {
                "fit_count": int((ok["hcce_fit_rot_err_deg"] > t).sum()),
                "keypoint_pnp_count": int((ok["hcce_kp_pnp_rot_err_deg"] > t).sum()),
                "fit_gap_worse_count": int((ok["gap_rot_err_deg"] > t).sum()),
                "pnp_gap_worse_count": int((ok["gap_rot_err_deg"] < -t).sum()),
            }
            for t in [5, 10, 20, 45, 90, 120, 150]
        },
        "tag_counts_all_rows": tag_counts,
        "tag_counts_bad_gap_rows": bad_tag_counts,
        "video_gap_summary": ok.groupby("video").agg(
            n=("video", "size"),
            fit_rot_mean=("hcce_fit_rot_err_deg", "mean"),
            pnp_rot_mean=("hcce_kp_pnp_rot_err_deg", "mean"),
            gap_rot_mean=("gap_rot_err_deg", "mean"),
            fit_trans_mean=("hcce_fit_trans_err_m", "mean"),
            pnp_trans_mean=("hcce_kp_pnp_trans_err_m", "mean"),
            fit_reproj_mean=("hcce_fit_reproj_rmse_all_px", "mean"),
            shaft_iou_mean=("hcce_model_crop_seg_part_iou_shaft", "mean"),
            wrist_iou_mean=("hcce_model_crop_seg_part_iou_wrist", "mean"),
            gripper_iou_mean=("hcce_model_crop_seg_part_iou_gripper", "mean"),
        ).sort_values("gap_rot_mean", ascending=False).head(20).reset_index().to_dict(orient="records"),
    }
    summary["plots"] = maybe_make_plots(ok, out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    lines = [
        "# HCCE Fit Worst-Case Analysis",
        "",
        f"- input: `{input_csv}`",
        f"- rows: {len(df)}",
        f"- fit ok: {len(ok)}",
        f"- fit failures: {len(failures)}",
        "",
        "## Main Metrics",
    ]
    for name, values in summary["metrics"].items():
        lines.append(
            f"- {name}: mean={values.get('mean', float('nan')):.6g}, "
            f"median={values.get('median', float('nan')):.6g}, p90={values.get('p90', float('nan')):.6g}, n={values.get('n', 0)}"
        )
    lines.extend([
        "",
        "## Fractions",
    ])
    for key, value in summary["fractions"].items():
        lines.append(f"- {key}: {value:.4f}")
    lines.extend([
        "",
        "## Failure Status Counts",
        "```json",
        json.dumps(summary["failure_status_counts"], indent=2, allow_nan=True),
        "```",
        "",
        "## Bad Gap Tag Counts",
        "```json",
        json.dumps(summary["tag_counts_bad_gap_rows"], indent=2, allow_nan=True),
        "```",
        "",
        "## Files",
        "- `fit_worse_than_keypoint_pnp_top.csv`",
        "- `hcce_fit_absolute_worst_top.csv`",
        "- `fit_rotation_worse_than_keypoint_pnp_top.csv`",
        "- `keypoint_pnp_rotation_worse_than_fit_top.csv`",
        "- `hcce_fit_failures.csv`",
        "- `selected_worstcase_manifest.csv`",
        "- `rot_fit_vs_keypoint_pnp_scatter.png`",
        "- `rot_gap_hist.png`",
        "",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot HCCE wrist-pose fitting error distributions from per-instance metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = {
    "HCCE fit": ("hcce_fit_trans_err_m", "hcce_fit_rot_err_deg", "#d1495b"),
    "HCCE direct": ("hcce_direct_trans_err_m", "hcce_direct_rot_err_deg", "#397367"),
    "HCCE keypoint PnP": ("hcce_kp_pnp_trans_err_m", "hcce_kp_pnp_rot_err_deg", "#0077b6"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-title", default="NeedleGrasping")
    parser.add_argument("--fit-label", default="HCCE fit")
    parser.add_argument("--secondary-input", type=Path)
    parser.add_argument("--secondary-fit-label", default="HCCE wrist-only fit")
    return parser.parse_args()


def finite(values: pd.Series, scale: float = 1.0) -> np.ndarray:
    array = values.to_numpy(dtype=np.float64) * scale
    return array[np.isfinite(array)]


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def summarize(values: np.ndarray) -> dict[str, float | int]:
    quantiles = np.quantile(values, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    ordered = np.sort(values)[::-1]
    total = float(ordered.sum())

    def tail_share(fraction: float) -> float:
        count = max(1, int(np.ceil(len(ordered) * fraction)))
        return float(ordered[:count].sum() / total) if total > 0 else 0.0

    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q75": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "max": float(np.max(values)),
        "top_1pct_error_sum_share": tail_share(0.01),
        "top_5pct_error_sum_share": tail_share(0.05),
        "trimmed_mean_drop_top_1pct": float(np.mean(np.sort(values)[: int(np.floor(len(values) * 0.99))])),
        "trimmed_mean_drop_top_5pct": float(np.mean(np.sort(values)[: int(np.floor(len(values) * 0.95))])),
    }


def add_reference_lines(ax: plt.Axes, median: float, p90: float, threshold: float) -> None:
    ax.axvline(median, color="#222222", linestyle="--", linewidth=1.4, label=f"median {median:.2f}")
    ax.axvline(p90, color="#e09f3e", linestyle="--", linewidth=1.4, label=f"p90 {p90:.2f}")
    ax.axvline(threshold, color="#7b2cbf", linestyle=":", linewidth=1.8, label=f"tail threshold {threshold:g}")


def plot_fit_distribution(
    trans_mm: np.ndarray,
    rot_deg: np.ndarray,
    output_dir: Path,
    dataset_title: str,
    fit_label: str,
) -> None:
    trans_stats = summarize(trans_mm)
    rot_stats = summarize(rot_deg)
    paired_n = min(len(trans_mm), len(rot_deg))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"{dataset_title}: {fit_label} wrist-pose error distribution (valid fits: {paired_n})", fontsize=17)

    ax = axes[0, 0]
    ax.hist(trans_mm, bins=60, color="#d1495b", alpha=0.88, edgecolor="white", linewidth=0.35)
    add_reference_lines(ax, trans_stats["median"], trans_stats["p90"], 30.0)
    ax.set(title="Translation: full range", xlabel="L2 translation error (mm)", ylabel="Count")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    trans_zoom_max = float(trans_stats["p95"])
    ax.hist(trans_mm[trans_mm <= trans_zoom_max], bins=45, color="#d1495b", alpha=0.88, edgecolor="white", linewidth=0.35)
    add_reference_lines(ax, trans_stats["median"], trans_stats["p90"], 30.0)
    ax.set(title="Translation: central 95%", xlabel="L2 translation error (mm)", ylabel="Count", xlim=(0, trans_zoom_max))
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    x, y = ecdf(trans_mm)
    ax.plot(x, y, color="#d1495b", linewidth=2.2)
    ax.axvline(30.0, color="#7b2cbf", linestyle=":", linewidth=1.8)
    ax.axhline(0.9, color="#777777", linestyle="--", linewidth=1.0)
    ax.set(title="Translation ECDF", xlabel="L2 translation error (mm)", ylabel="Fraction <= error", ylim=(0, 1.01))
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.hist(rot_deg, bins=np.arange(0, 185, 5), color="#d1495b", alpha=0.88, edgecolor="white", linewidth=0.35)
    add_reference_lines(ax, rot_stats["median"], rot_stats["p90"], 45.0)
    ax.set(title="Rotation: full range", xlabel="Geodesic rotation error (deg)", ylabel="Count", xlim=(0, 180))
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    rot_zoom_max = float(rot_stats["p95"])
    ax.hist(rot_deg[rot_deg <= rot_zoom_max], bins=45, color="#d1495b", alpha=0.88, edgecolor="white", linewidth=0.35)
    add_reference_lines(ax, rot_stats["median"], rot_stats["p90"], 45.0)
    ax.set(title="Rotation: central 95%", xlabel="Geodesic rotation error (deg)", ylabel="Count", xlim=(0, rot_zoom_max))
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    x, y = ecdf(rot_deg)
    ax.plot(x, y, color="#d1495b", linewidth=2.2)
    ax.axvline(45.0, color="#7b2cbf", linestyle=":", linewidth=1.8)
    ax.axhline(0.9, color="#777777", linestyle="--", linewidth=1.0)
    ax.set(title="Rotation ECDF", xlabel="Geodesic rotation error (deg)", ylabel="Fraction <= error", xlim=(0, 180), ylim=(0, 1.01))
    ax.grid(alpha=0.25)

    fig.savefig(output_dir / "hcce_fit_error_distribution.png", dpi=180)
    plt.close(fig)


def plot_joint_tail(
    trans_mm: np.ndarray,
    rot_deg: np.ndarray,
    output_dir: Path,
    dataset_title: str,
    fit_label: str,
) -> None:
    normal = (trans_mm <= 30.0) & (rot_deg <= 45.0)
    catastrophic = (trans_mm > 50.0) | (rot_deg > 90.0)

    fig, ax = plt.subplots(figsize=(10.5, 7.5), constrained_layout=True)
    ax.scatter(trans_mm[normal], rot_deg[normal], s=10, alpha=0.28, color="#397367", label=f"within 30 mm and 45 deg ({normal.sum()})")
    tail = ~normal & ~catastrophic
    ax.scatter(trans_mm[tail], rot_deg[tail], s=18, alpha=0.55, color="#e09f3e", label=f"tail ({tail.sum()})")
    ax.scatter(trans_mm[catastrophic], rot_deg[catastrophic], s=24, alpha=0.72, color="#d1495b", label=f">50 mm or >90 deg ({catastrophic.sum()})")
    ax.axvline(30.0, color="#555555", linestyle="--", linewidth=1.2)
    ax.axhline(45.0, color="#555555", linestyle="--", linewidth=1.2)
    ax.axvline(50.0, color="#d1495b", linestyle=":", linewidth=1.2)
    ax.axhline(90.0, color="#d1495b", linestyle=":", linewidth=1.2)
    ax.set(
        title=f"{dataset_title}: {fit_label} translation/rotation tail structure",
        xlabel="L2 translation error (mm)",
        ylabel="Geodesic rotation error (deg)",
        ylim=(0, 180),
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="upper right")
    fig.savefig(output_dir / "hcce_fit_trans_rot_joint_distribution.png", dpi=180)
    plt.close(fig)


def plot_method_comparison(data: pd.DataFrame, output_dir: Path, dataset_title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for label, (trans_col, rot_col, color) in METHODS.items():
        trans_mm = finite(data[trans_col], 1000.0)
        rot_deg = finite(data[rot_col])
        x, y = ecdf(trans_mm)
        axes[0].plot(x, y, linewidth=2.0, color=color, label=f"{label} (median {np.median(trans_mm):.1f} mm)")
        x, y = ecdf(rot_deg)
        axes[1].plot(x, y, linewidth=2.0, color=color, label=f"{label} (median {np.median(rot_deg):.1f} deg)")

    axes[0].set(xlabel="L2 translation error (mm)", ylabel="Fraction <= error", title="Translation ECDF", xlim=(0, 100), ylim=(0, 1.01))
    axes[1].set(xlabel="Geodesic rotation error (deg)", ylabel="Fraction <= error", title="Rotation ECDF", xlim=(0, 180), ylim=(0, 1.01))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(f"{dataset_title} wrist-pose error comparison", fontsize=16)
    fig.savefig(output_dir / "hcce_fit_vs_direct_keypoint_pnp_ecdf.png", dpi=180)
    plt.close(fig)


def plot_fit_comparison(
    primary: pd.DataFrame,
    secondary: pd.DataFrame,
    output_dir: Path,
    dataset_title: str,
    primary_label: str,
    secondary_label: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    for data, label, color in (
        (primary, primary_label, "#d1495b"),
        (secondary, secondary_label, "#0077b6"),
    ):
        trans_mm = finite(data["hcce_fit_trans_err_m"], 1000.0)
        rot_deg = finite(data["hcce_fit_rot_err_deg"])
        x, y = ecdf(trans_mm)
        axes[0].plot(x, y, linewidth=2.2, color=color, label=f"{label} (median {np.median(trans_mm):.2f} mm)")
        x, y = ecdf(rot_deg)
        axes[1].plot(x, y, linewidth=2.2, color=color, label=f"{label} (median {np.median(rot_deg):.2f} deg)")

    axes[0].set(xlabel="L2 translation error (mm)", ylabel="Fraction <= error", title="Translation ECDF", ylim=(0, 1.01))
    axes[1].set(xlabel="Geodesic rotation error (deg)", ylabel="Fraction <= error", title="Rotation ECDF", xlim=(0, 180), ylim=(0, 1.01))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle(f"{dataset_title}: HCCE fit mode comparison", fontsize=16)
    fig.savefig(output_dir / "hcce_full_fit_vs_wrist_only_fit_ecdf.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    trans_mm = finite(data["hcce_fit_trans_err_m"], 1000.0)
    rot_deg = finite(data["hcce_fit_rot_err_deg"])
    paired = data[["hcce_fit_trans_err_m", "hcce_fit_rot_err_deg"]].replace([np.inf, -np.inf], np.nan).dropna()
    paired_trans_mm = paired["hcce_fit_trans_err_m"].to_numpy(dtype=np.float64) * 1000.0
    paired_rot_deg = paired["hcce_fit_rot_err_deg"].to_numpy(dtype=np.float64)

    plot_fit_distribution(trans_mm, rot_deg, args.output_dir, args.dataset_title, args.fit_label)
    plot_joint_tail(paired_trans_mm, paired_rot_deg, args.output_dir, args.dataset_title, args.fit_label)
    plot_method_comparison(data, args.output_dir, args.dataset_title)
    if args.secondary_input is not None:
        secondary = pd.read_csv(args.secondary_input)
        plot_fit_comparison(
            data,
            secondary,
            args.output_dir,
            args.dataset_title,
            args.fit_label,
            args.secondary_fit_label,
        )

    trans_stats = summarize(trans_mm)
    rot_stats = summarize(rot_deg)
    valid = len(paired)
    total = len(data)
    strict_tail = (paired_trans_mm > 30.0) | (paired_rot_deg > 45.0)
    catastrophic = (paired_trans_mm > 50.0) | (paired_rot_deg > 90.0)
    joint_thresholds = {
        "total_instances": total,
        "valid_fits": valid,
        "fit_failures": total - valid,
        "fit_failure_fraction": (total - valid) / total,
        "trans_gt_30mm_count": int((paired_trans_mm > 30.0).sum()),
        "trans_gt_30mm_fraction": float((paired_trans_mm > 30.0).mean()),
        "trans_gt_50mm_count": int((paired_trans_mm > 50.0).sum()),
        "trans_gt_50mm_fraction": float((paired_trans_mm > 50.0).mean()),
        "rot_gt_45deg_count": int((paired_rot_deg > 45.0).sum()),
        "rot_gt_45deg_fraction": float((paired_rot_deg > 45.0).mean()),
        "rot_gt_90deg_count": int((paired_rot_deg > 90.0).sum()),
        "rot_gt_90deg_fraction": float((paired_rot_deg > 90.0).mean()),
        "trans_gt_30mm_or_rot_gt_45deg_count": int(strict_tail.sum()),
        "trans_gt_30mm_or_rot_gt_45deg_fraction": float(strict_tail.mean()),
        "trans_gt_50mm_or_rot_gt_90deg_count": int(catastrophic.sum()),
        "trans_gt_50mm_or_rot_gt_90deg_fraction": float(catastrophic.mean()),
    }
    summary = {"input": str(args.input.resolve()), "translation_mm": trans_stats, "rotation_deg": rot_stats, "tail_counts": joint_thresholds}
    with (args.output_dir / "error_distribution_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    rows = []
    for metric, unit, stats in (("translation", "mm", trans_stats), ("rotation", "deg", rot_stats)):
        rows.append({"metric": metric, "unit": unit, **stats})
    with (args.output_dir / "error_distribution_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

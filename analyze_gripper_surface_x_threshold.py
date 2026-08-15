import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROBOPEPP_ROOT = Path(__file__).resolve().parent
DEFAULT_IN = ROBOPEPP_ROOT / "assets" / "instrument_surface_samples"
DEFAULT_OUT = DEFAULT_IN / "gripper_x_threshold_sweep_light"
DEFAULT_STATIC_THRESHOLD_MM = 2.13
PARTS = ("shaft", "wrist", "l_gripper", "r_gripper")
GRIPPERS = ("l_gripper", "r_gripper")
MOVING_RGB = (255, 90, 40)
STATIC_RGB = (35, 210, 255)
BG_RGB = (250, 250, 250)


def load_payloads(in_dir):
    in_dir = Path(in_dir)
    return {part: np.load(in_dir / f"{part}_surface_points.npy", allow_pickle=True).item() for part in PARTS}


def recommended_threshold(payloads):
    wrist_x_max = float(payloads["wrist"]["points_canon_world_m"][:, 0].max())
    offsets = [
        float(np.median(payloads[p]["points_canon_world_m"][:, 0] - payloads[p]["points_part_m"][:, 0]))
        for p in GRIPPERS
    ]
    return wrist_x_max - float(np.mean(offsets)), wrist_x_max, offsets


def safe_threshold_name(threshold_m):
    return f"xp{threshold_m * 1000.0:05.2f}mm".replace(".", "p")


def draw_panel(points, static_mask, threshold_m, axes, title, width=380, height=270):
    img = Image.new("RGB", (width, height), BG_RGB)
    draw = ImageDraw.Draw(img)
    ml, mt, mr, mb = 48, 30, 14, 28
    pw, ph = width - ml - mr, height - mt - mb
    x = points[:, axes[0]] * 1000.0
    y = points[:, axes[1]] * 1000.0
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx = max(1e-6, xmax - xmin)
    dy = max(1e-6, ymax - ymin)
    xmin, xmax = xmin - dx * 0.06, xmax + dx * 0.06
    ymin, ymax = ymin - dy * 0.10, ymax + dy * 0.10

    def map_xy(a, b):
        px = ml + (a - xmin) / (xmax - xmin) * pw
        py = mt + (ymax - b) / (ymax - ymin) * ph
        return int(round(px)), int(round(py))

    draw.rectangle([ml, mt, ml + pw, mt + ph], outline=(120, 120, 120), width=1)
    tx = threshold_m * 1000.0
    if xmin <= tx <= xmax:
        px, _ = map_xy(tx, ymin)
        draw.line([px, mt, px, mt + ph], fill=(0, 0, 0), width=2)

    idx = np.linspace(0, len(points) - 1, min(9000, len(points))).astype(np.int64)
    for val, color in ((False, MOVING_RGB), (True, STATIC_RGB)):
        ii = idx[static_mask[idx] == val]
        for a, b in zip(x[ii], y[ii]):
            px, py = map_xy(float(a), float(b))
            if 0 <= px < width and 0 <= py < height:
                img.putpixel((px, py), color)

    draw.text((8, 8), title, fill=(0, 0, 0))
    draw.text((ml, height - 20), "x part-frame (mm)", fill=(0, 0, 0))
    draw.text((ml + 4, mt + 4), f"cyan x < {threshold_m * 1000.0:.2f}mm", fill=(0, 100, 130))
    draw.text((ml + 4, mt + 18), "orange moving", fill=(170, 50, 20))
    return img


def write_visuals(payloads, thresholds, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for th in thresholds:
        panels = []
        for part in GRIPPERS:
            pts = payloads[part]["points_part_m"]
            static = pts[:, 0] < th
            panels.append(draw_panel(pts, static, th, (0, 1), f"{part} x-y | static {int(static.sum())}/{len(static)}"))
            panels.append(draw_panel(pts, static, th, (0, 2), f"{part} x-z | moving {int((~static).sum())}/{len(static)}"))
        row = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), BG_RGB)
        x0 = 0
        for panel in panels:
            row.paste(panel, (x0, 0))
            x0 += panel.width
        row.save(out_dir / f"gripper_split_{safe_threshold_name(th)}.png")
        rows.append(row)

    sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), BG_RGB)
    y0 = 0
    for row in rows:
        sheet.paste(row, (0, y0))
        y0 += row.height
    sheet.save(out_dir / "gripper_split_threshold_contact_sheet.png")


def write_stats(payloads, thresholds, out_dir):
    path = Path(out_dir) / "threshold_stats.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold_m", "threshold_mm", "part", "static_count", "moving_count", "static_fraction"])
        for th in thresholds:
            for part in GRIPPERS:
                x = payloads[part]["points_part_m"][:, 0]
                static = x < th
                writer.writerow([f"{th:.9f}", f"{th * 1000.0:.4f}", part, int(static.sum()), int((~static).sum()), f"{float(static.mean()):.6f}"])
    return path


def write_half_density_assets(payloads, threshold, out_dir, seed):
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))
    all_norm, all_part, all_canon, all_part_ids, all_eff, all_static, all_names = [], [], [], [], [], [], []
    summary = []
    for part in PARTS:
        payload = payloads[part]
        n = len(payload["points_norm"])
        keep = np.sort(rng.choice(n, n // 2, replace=False)) if part in GRIPPERS else np.arange(n)
        out = {}
        for key, value in payload.items():
            arr = np.asarray(value)
            out[key] = arr[keep] if arr.shape[:1] == (n,) else value
        n_keep = len(keep)
        original_id = int(np.asarray(payload["part_id"]).reshape(-1)[0])
        static = np.zeros((n_keep,), dtype=bool)
        effective = np.full((n_keep,), original_id, dtype=np.int64)
        if part in GRIPPERS:
            static = out["points_part_m"][:, 0] < threshold
            effective[static] = 2
        out["static_wrist_mask"] = static
        out["effective_part_ids"] = effective
        out["threshold_x_m"] = np.array(threshold, dtype=np.float32)
        np.save(out_dir / f"{part}_surface_points.npy", out, allow_pickle=True)

        all_norm.append(out["points_norm"].astype(np.float32))
        all_part.append(out["points_part_m"].astype(np.float32))
        all_canon.append(out["points_canon_world_m"].astype(np.float32))
        all_part_ids.append(np.full((n_keep,), original_id, dtype=np.int64))
        all_eff.append(effective)
        all_static.append(static)
        all_names.extend([part] * n_keep)
        summary.append(f"{part}: n={n_keep}, original_part_id={original_id}, static_as_wrist={int(static.sum())}, moving={int((~static).sum())}")

    combined = {
        "points_norm": np.concatenate(all_norm, axis=0).astype(np.float32),
        "points_part_m": np.concatenate(all_part, axis=0).astype(np.float32),
        "points_canon_world_m": np.concatenate(all_canon, axis=0).astype(np.float32),
        "part_ids": np.concatenate(all_part_ids, axis=0).astype(np.int64),
        "effective_part_ids": np.concatenate(all_eff, axis=0).astype(np.int64),
        "static_wrist_mask": np.concatenate(all_static, axis=0).astype(bool),
        "part_names": np.asarray(all_names),
        "canon_scale": np.asarray(payloads["wrist"]["canon_scale"], dtype=np.float32),
        "gripper_static_threshold_x_m": np.array(threshold, dtype=np.float32),
    }
    np.save(out_dir / "instrument_surface_points_all.npy", combined, allow_pickle=True)
    (out_dir / "README.txt").write_text(
        "Derived from assets/instrument_surface_samples.\n"
        f"Threshold: points_part_m[:,0] < {threshold:.9f} m ({threshold * 1000.0:.3f} mm).\n"
        "Gripper density is 0.5x. Static rear gripper points use effective_part_ids=2 and static_wrist_mask=True.\n"
        + "\n".join(summary)
        + "\n",
        encoding="utf-8",
    )
    return combined, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default=str(DEFAULT_IN))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--threshold_mm", type=float, default=DEFAULT_STATIC_THRESHOLD_MM)
    parser.add_argument("--thresholds_mm", type=str, default="0.0,1.5,2.13,2.33,3.0,4.0")
    parser.add_argument("--write_half_density", type=int, choices=[0, 1], default=1)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    payloads = load_payloads(args.in_dir)
    rec, wrist_x_max, offsets = recommended_threshold(payloads)
    thresholds = [float(v.strip()) / 1000.0 for v in args.thresholds_mm.split(",") if v.strip()]
    fixed_threshold = float(args.threshold_mm) / 1000.0
    if not any(abs(t - fixed_threshold) < 1e-9 for t in thresholds):
        thresholds.append(fixed_threshold)
    if not any(abs(t - rec) < 1e-6 for t in thresholds):
        thresholds.append(rec)
    thresholds = sorted(set(thresholds))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = write_stats(payloads, thresholds, out_dir)
    write_visuals(payloads, thresholds, out_dir)
    print(f"wrist_x_max_m={wrist_x_max:.9f} ({wrist_x_max * 1000.0:.3f} mm)")
    print("gripper_x_offsets_m=" + ",".join(f"{v:.9f}" for v in offsets))
    print(f"recommended_threshold_m={rec:.9f} ({rec * 1000.0:.3f} mm)")
    print(f"fixed_static_threshold_m={fixed_threshold:.9f} ({fixed_threshold * 1000.0:.3f} mm)")
    print(f"wrote stats: {stats_path}")
    print(f"wrote visualizations: {out_dir}")

    if int(args.write_half_density):
        asset_dir = Path(args.in_dir).parent / f"instrument_surface_samples_gripper_half_x{fixed_threshold * 1000.0:.2f}mm"
        combined, summary = write_half_density_assets(payloads, fixed_threshold, asset_dir, args.seed)
        print(f"wrote half-density asset: {asset_dir}")
        print(f"half-density combined counts: total={combined['points_norm'].shape[0]}, static_as_wrist={int(combined['static_wrist_mask'].sum())}")
        print("\n".join(summary))


if __name__ == "__main__":
    main()

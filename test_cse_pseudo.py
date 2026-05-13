"""
Step 1 sanity check for CSE pseudo-annotation generation.

Goal: pick one sample from RARPInstanceDataset, reload the matching
`memory_pool.pth` entry to recover the articulated pose
(rot quaternion, trans, alpha, theta_l, theta_r), then drive the GS
instrument's forward_kinematics and render a per-part silhouette
(R=shaft, G=wrist, B=gripper) side-by-side with the original frame.

Run:
    conda run -n multiipr python test_cse_pseudo.py --vis
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Make the user's RARPInstanceDataset importable.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# GMS submodule path — only used to locate the rendering helper script that
# is invoked as a subprocess. We never import GMS modules into this process,
# which sidesteps the multi-hmr/utils vs GMS/utils naming collision.
GMS_ROOT = REPO_ROOT / "submodules" / "gaussian-mesh-splatting"

from datasets.RarpInstanceDataset import RARPInstanceDataset  # noqa: E402


# ---------------------------------------------------------------------------
# memory_pool.pth was pickled with the GMS `Pose` class. We don't want to
# pull in the whole GMS dependency tree just to deserialize, so we register
# a stub that absorbs every attribute via __setstate__. After unpickling, the
# stub will carry .rot/.trans/.alpha/.theta_l/.theta_r as plain tensors
# (originally torch.nn.Parameter, but Parameter unpickles fine without a
# live optimizer).
# ---------------------------------------------------------------------------
class _PoseStub:
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _register_pose_stub():
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "Pose"):
        setattr(main_mod, "Pose", _PoseStub)


def _build_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """CPU/torch port of utils.graphics_utils.build_rotation (no cuda)."""
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    q = F.normalize(quat, p=2, dim=1)
    R = torch.zeros((q.size(0), 3, 3), device=q.device, dtype=q.dtype)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_pose_from_memory_pool(pose_root: str, video_name: str,
                               instance_id: int, frame_id: str):
    """
    Resolve the memory_pool.pth that backs the given (video, instance) and
    return the pose entry for `frame_id`.

    Returns a dict with:
        rot_quat:  (4,) float tensor (w, x, y, z) — raw stored quaternion
        rot_mat:   (3, 3) float tensor — quaternion turned into a rotation matrix
        trans:     (3,) float tensor (meters)
        alpha:     scalar tensor (raw, pre-activation)
        theta_l:   scalar tensor (raw, pre-activation)
        theta_r:   scalar tensor (raw, pre-activation)
        dice:      list[float] [wrist, shaft, gripper]
        mem_path:  Path to the memory_pool.pth that was loaded
    """
    _register_pose_stub()

    case_dir = Path(pose_root) / f"SARRARP502022_{video_name}_instance{instance_id}"
    mem_path = case_dir / "memory_pool.pth"
    if not mem_path.is_file():
        raise FileNotFoundError(f"memory_pool.pth not found: {mem_path}")

    memory_pool = torch.load(mem_path, map_location="cpu", weights_only=False)
    if frame_id not in memory_pool:
        raise KeyError(
            f"frame_id {frame_id!r} not in memory_pool {mem_path} "
            f"(have {len(memory_pool)} entries; sample keys: "
            f"{list(memory_pool.keys())[:3]})"
        )

    entry = memory_pool[frame_id]
    pose = entry["pose_info"]
    dice = entry["dice"]

    # Pose attributes are torch.nn.Parameter when pickled. Detach to plain tensors.
    rot_quat = pose.rot.detach().to(torch.float32).flatten()
    trans = pose.trans.detach().to(torch.float32).flatten()
    alpha = pose.alpha.detach().to(torch.float32).flatten()
    theta_l = pose.theta_l.detach().to(torch.float32).flatten()
    theta_r = pose.theta_r.detach().to(torch.float32).flatten()

    rot_mat = _build_rotation_matrix(rot_quat)[0]

    if isinstance(dice, torch.Tensor):
        dice_list = [float(x) for x in dice.flatten().tolist()]
    else:
        dice_list = [float(x) for x in dice]

    return dict(
        rot_quat=rot_quat,
        rot_mat=rot_mat,
        trans=trans,
        alpha=alpha,
        theta_l=theta_l,
        theta_r=theta_r,
        dice=dice_list,
        mem_path=mem_path,
    )


def render_part_silhouettes_subprocess(pose, K, img_h, img_w, work_dir):
    """
    Run the GS silhouette + canonical-coordinate rendering in a subprocess
    that lives entirely inside the GMS submodule. This keeps the multi-hmr
    `utils` package and the GMS `utils` package from ever sharing an
    interpreter.

    Returns:
        silhouette : (H, W, 3) float ndarray in [0, 1], channels = (shaft, wrist, gripper)
        coord_img  : (H, W, 4) float ndarray — channels 0-2 = xyz/scale (no offset),
                     channel 3 = part index (0=bg, 1=shaft, 2=wrist, 3=l_gripper, 4=r_gripper);
                     or None if the coord rendering step failed.
    """
    import subprocess

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pose_path  = work_dir / "pose.pt"
    out_path   = work_dir / "silhouette.npy"
    coord_path = work_dir / "coord_img.npy"

    payload = dict(
        rot_mat=pose["rot_mat"].cpu(),
        trans=pose["trans"].cpu(),
        alpha=pose["alpha"].cpu(),
        theta_l=pose["theta_l"].cpu(),
        theta_r=pose["theta_r"].cpu(),
        K=K.astype(np.float32),
        img_h=int(img_h),
        img_w=int(img_w),
    )
    torch.save(payload, pose_path)

    helper = GMS_ROOT / "_cse_render_helper.py"
    cmd = [sys.executable, str(helper),
           "--pose", str(pose_path),
           "--out", str(out_path),
           "--coord_out", str(coord_path)]
    print(f"[subproc] cwd={GMS_ROOT}\n[subproc] cmd={' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(GMS_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"GS render helper failed (rc={res.returncode})")

    silhouette = np.load(out_path)
    coord_img  = np.load(coord_path) if coord_path.exists() else None
    scale_path = work_dir / "coord_img_scale.npy"
    canon_scale = float(np.load(scale_path)) if scale_path.exists() else None
    canon_world_path = work_dir / "coord_img_canon_world.npz"
    canon_world = dict(np.load(canon_world_path)) if canon_world_path.exists() else None
    return silhouette, coord_img, canon_scale, canon_world


def make_visualization(rgb_np, silhouette_np, out_path, coord_np=None):
    """
    rgb_np:        (H, W, 3) uint8 — original frame
    silhouette_np: (H, W, 3) float [0,1] — channel 0=shaft, 1=wrist, 2=gripper
    coord_np:      (H, W, 4) float (optional) — channels 0-2 = xyz/scale, ch3 = part id

    Layout (2 rows × 3 cols, plus an optional 4th column when coord_np is given):
      row1: original | combined silhouette | overlay  [| coord xyz]
      row2: shaft    | wrist               | gripper  [| blank    ]
    """
    H, W = rgb_np.shape[:2]
    sil_uint = (silhouette_np * 255.0).astype(np.uint8)

    # Per-part silhouette panels (R/G/B tint)
    shaft   = np.zeros_like(rgb_np); shaft[..., 0]   = sil_uint[..., 0]
    wrist   = np.zeros_like(rgb_np); wrist[..., 1]   = sil_uint[..., 1]
    gripper = np.zeros_like(rgb_np); gripper[..., 2] = sil_uint[..., 2]

    combined = sil_uint

    mask    = silhouette_np.max(axis=-1, keepdims=True)
    overlay = ((1.0 - 0.55 * mask) * rgb_np + 0.55 * mask * sil_uint).astype(np.uint8)

    sep  = np.full((H, 4, 3), 255, dtype=np.uint8)
    hsep_w = None  # determined after rows are built

    if coord_np is not None:
        # CNOS-style: per-part normalise to [-1, 1] then map via (x+1)/2 → [0, 1]
        # Part ids: 1=shaft, 2=wrist, 3=l_gripper, 4=r_gripper
        coord_vis = np.zeros((H, W, 3), dtype=np.float32)
        for pid in [1, 2, 3, 4]:
            pmask = coord_np[:, :, 3] == pid
            if not pmask.any():
                continue
            xyz = coord_np[pmask, :3]                        # (N, 3)
            center = (xyz.max(axis=0) + xyz.min(axis=0)) * 0.5
            radius = float(np.linalg.norm(xyz - center, axis=1).max()) + 1e-8
            xyz_norm = (xyz - center) / radius               # ≈ [-1, 1]
            coord_vis[pmask] = np.clip(xyz_norm * 0.5 + 0.5, 0.0, 1.0)
        coord_uint = (coord_vis * 255.0).astype(np.uint8)
        blank = np.zeros((H, W, 3), dtype=np.uint8)
        row1 = np.concatenate([rgb_np, sep, combined, sep, overlay, sep, coord_uint], axis=1)
        row2 = np.concatenate([shaft,  sep, wrist,   sep, gripper, sep, blank      ], axis=1)
    else:
        row1 = np.concatenate([rgb_np, sep, combined, sep, overlay], axis=1)
        row2 = np.concatenate([shaft,  sep, wrist,   sep, gripper ], axis=1)

    hsep   = np.full((4, row1.shape[1], 3), 255, dtype=np.uint8)
    canvas = np.concatenate([row1, hsep, row2], axis=0)

    Image.fromarray(canvas).save(out_path)


def plot_nocs_pointcloud_world(canon_world: dict, out_path: str,
                               max_pts_per_part: int = 4000) -> None:
    """
    Plot canonical mesh vertices in a common world coordinate frame.

    canon_world : dict {part_name -> (N,3) float32 world-space vertices}
                  part names: 'shaft', 'wrist', 'l_gripper', 'r_gripper'
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    PART_COLOR = {
        'shaft':     '#e74c3c',
        'wrist':     '#2ecc71',
        'l_gripper': '#3498db',
        'r_gripper': '#f39c12',
    }
    VIEWS = [
        ('front', 10, -60),
        ('side',  10,  30),
        ('top',   80, -60),
    ]

    fig = plt.figure(figsize=(6 * len(VIEWS), 5))
    axes = [fig.add_subplot(1, len(VIEWS), i + 1, projection='3d')
            for i in range(len(VIEWS))]

    rng = np.random.default_rng(0)
    all_pts = []

    for part_name, verts in canon_world.items():
        color = PART_COLOR.get(part_name, '#888888')
        all_pts.append(verts)
        pts = verts
        if len(pts) > max_pts_per_part:
            pts = pts[rng.choice(len(pts), max_pts_per_part, replace=False)]
        for ax in axes:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=color, s=1.5, alpha=0.6, label=part_name, rasterized=True)

    if all_pts:
        combined = np.concatenate(all_pts, axis=0)
        lo, hi = combined.min(0), combined.max(0)
        ctr = (lo + hi) * 0.5
        half = (hi - lo).max() * 0.55

    for ax, (vname, elev, azim) in zip(axes, VIEWS):
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.set_title(vname)
        ax.view_init(elev=elev, azim=azim)
        if all_pts:
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            ax.set_zlim(ctr[2] - half, ctr[2] + half)

    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc='lower center', ncol=4, markerscale=6, frameon=False)
    fig.suptitle('Canonical mesh — world space (bias2world @ inv(bias2part))', fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] wrote point cloud (world) → {out_path}")


def plot_nocs_pointcloud(coord_np: np.ndarray, out_path: str,
                         canon_scale: float = None,
                         max_pts_per_part: int = 4000) -> None:
    """
    Draw canonical 3-D point cloud for each part in the same coordinate frame.

    coord_np    : (H, W, 4) float32 — ch 0-2 = xyz/canon_scale, ch 3 = part id
                  part ids: 0=bg, 1=shaft, 2=wrist, 3=l_gripper, 4=r_gripper
    out_path    : destination PNG
    canon_scale : if provided, multiply xyz by this to recover raw canonical coords
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    PART_META = {
        1: ('shaft',     '#e74c3c'),   # red
        2: ('wrist',     '#2ecc71'),   # green
        3: ('l_gripper', '#3498db'),   # blue
        4: ('r_gripper', '#f39c12'),   # orange
    }
    VIEWS = [
        ('front', 10,  -60),
        ('side',  10,   30),
        ('top',   80,  -60),
    ]

    fig = plt.figure(figsize=(6 * len(VIEWS), 5))
    axes = [fig.add_subplot(1, len(VIEWS), i + 1, projection='3d')
            for i in range(len(VIEWS))]

    rng = np.random.default_rng(0)
    all_xyz = []

    for pid, (pname, color) in PART_META.items():
        mask = coord_np[:, :, 3] == pid
        if not mask.any():
            continue
        xyz = coord_np[:, :, :3][mask].copy()   # (N, 3)
        if canon_scale is not None:
            xyz *= canon_scale                    # recover raw canonical coords
        all_xyz.append(xyz)
        pts = xyz
        if len(pts) > max_pts_per_part:
            idx = rng.choice(len(pts), max_pts_per_part, replace=False)
            pts = pts[idx]
        for ax in axes:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c=color, s=1.5, alpha=0.5, label=pname, rasterized=True)

    if all_xyz:
        all_pts = np.concatenate(all_xyz, axis=0)
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        ctr = (lo + hi) * 0.5
        half = (hi - lo).max() * 0.55

    for ax, (vname, elev, azim) in zip(axes, VIEWS):
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(vname)
        ax.view_init(elev=elev, azim=azim)
        if all_xyz:
            ax.set_xlim(ctr[0] - half, ctr[0] + half)
            ax.set_ylim(ctr[1] - half, ctr[1] + half)
            ax.set_zlim(ctr[2] - half, ctr[2] + half)

    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc='lower center', ncol=len(PART_META),
               markerscale=6, frameon=False)

    coord_desc = f'raw canonical (scale={canon_scale:.4f})' if canon_scale else 'normalised (xyz/scale)'
    fig.suptitle(f'Canonical point cloud — {coord_desc}', fontsize=11)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[ok] wrote point cloud → {out_path}")


def visualize_sample(pose_root, dataset_root, video_name, instance_id,
                     frame_id, pose, out_path, work_dir):
    # ── original frame at native resolution ──
    video_folder = Path(dataset_root) / f"SARRARP502022_{video_name}"
    frame_path = video_folder / "frames" / f"{frame_id}.png"
    if not frame_path.exists():
        frame_path = video_folder / "frames" / f"{frame_id}.jpg"
    rgb_pil = Image.open(frame_path).convert("RGB")
    rgb_np = np.asarray(rgb_pil)
    img_w, img_h = rgb_pil.size

    # ── camera intrinsics: same convention as the training pipeline ──
    rarp_fc = 587.54401824
    K = np.array([[rarp_fc, 0, img_w / 2.0],
                  [0, rarp_fc, img_h / 2.0],
                  [0, 0, 1.0]], dtype=np.float32)

    silhouette, coord_img, canon_scale, canon_world = render_part_silhouettes_subprocess(
        pose, K, img_h, img_w, work_dir=work_dir,
    )

    make_visualization(rgb_np, silhouette, out_path, coord_np=coord_img)
    print(f"[ok] wrote {out_path}")

    if canon_world is not None:
        pcd_path = str(Path(out_path).parent / (Path(out_path).stem + "_pcd3d.png"))
        plot_nocs_pointcloud_world(canon_world, pcd_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="/mnt/nas/share/shuojue/data/needlePuncture_videos",
    )
    parser.add_argument(
        "--pose_root",
        default="/mnt/nas/share/shuojue/data/needlePuncture_results",
    )
    parser.add_argument("--idx", type=int, default=0,
                        help="Sample index inside the dataset to inspect.")
    parser.add_argument("--vis", action="store_true",
                        help="Render the per-part silhouette and dump a "
                             "side-by-side image to --out.")
    parser.add_argument("--out", default="cse_pseudo_vis.png")
    parser.add_argument("--work_dir", default="/tmp/cse_pseudo",
                        help="Scratch dir for pose.pt / silhouette.npy "
                             "exchanged with the GS render subprocess.")
    args = parser.parse_args()

    # ── 1. build the dataset (uses memory_pool only for dice filtering) ──
    dataset = RARPInstanceDataset(
        split="train",
        training=False,
        dataset_root=args.dataset_root,
        pose_root=args.pose_root,
        train_ratio=1.0,           # keep everything in "train" so any idx works
    )
    print(f"[dataset] {dataset!r}")
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty; nothing to inspect.")

    idx = args.idx % len(dataset)
    video_name, frame_id, instances_list = dataset.samples[idx]
    print(f"[sample {idx}] video={video_name}  frame_id={frame_id}  "
          f"instances={[i for i, _ in instances_list]}")

    # also exercise __getitem__ to confirm the sample is well-formed
    img_array, annot = dataset[idx]
    print(f"[__getitem__] img shape={tuple(img_array.shape)}  "
          f"#instruments={len(annot['instruments'])}  "
          f"name={annot['imagename']}")

    # ── 2. for every (instance_id, instance_folder) reload the pose ──
    for instance_id, _ in instances_list:
        try:
            pose = load_pose_from_memory_pool(
                args.pose_root, video_name, instance_id, frame_id
            )
        except (FileNotFoundError, KeyError) as e:
            print(f"  [instance {instance_id}] skipped: {e}")
            continue

        print(f"\n  ── instance {instance_id} ── ({pose['mem_path']})")
        print(f"    rot quat (w,x,y,z): {pose['rot_quat'].tolist()}")
        print(f"    rot mat:\n{pose['rot_mat'].numpy()}")
        print(f"    trans (m):          {pose['trans'].tolist()}")
        print(f"    alpha   (raw):      {pose['alpha'].tolist()}")
        print(f"    theta_l (raw):      {pose['theta_l'].tolist()}")
        print(f"    theta_r (raw):      {pose['theta_r'].tolist()}")
        print(f"    dice [wrist,shaft,gripper]: {pose['dice']}")

        # quick sanity assertions
        assert pose["rot_quat"].numel() == 4, "rot quat must have 4 components"
        assert pose["trans"].numel() == 3, "trans must be 3-vector"
        for name in ("alpha", "theta_l", "theta_r"):
            assert pose[name].numel() == 1, f"{name} must be a scalar"
        # quaternion should be (close to) unit length after F.normalize
        q = F.normalize(pose["rot_quat"].unsqueeze(0), p=2, dim=1).squeeze(0)
        assert torch.allclose(q.norm(), torch.tensor(1.0), atol=1e-5)

    print("\n[ok] memory_pool pose loading verified.")

    # ── 3. optional: render and visualize the silhouette ──
    if True: #args.vis:
        # use the first instance for which we successfully loaded a pose
        for instance_id, _ in instances_list:
            try:
                pose = load_pose_from_memory_pool(
                    args.pose_root, video_name, instance_id, frame_id
                )
            except (FileNotFoundError, KeyError):
                continue
            visualize_sample(
                args.pose_root, args.dataset_root,
                video_name, instance_id, frame_id, pose, args.out,
                work_dir=args.work_dir,
            )
            break


if __name__ == "__main__":
    main()

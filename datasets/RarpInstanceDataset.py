import os
import sys
import glob
import warnings
import random
import numpy as np
import torch
import torch.nn.functional as _F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from utils import normalize_rgb
from datasets.rarp_augmentation import transform_rarp_sample
from datasets.rarp_pose_canonicalization import canonicalize_pose_symmetry as _canonicalize_pose_symmetry

ImageFile.LOAD_TRUNCATED_IMAGES = True

# memory_pool.pth files were pickled with a Pose class from the GMS codebase.
# We only use the 'dice' values, so a lightweight stub is enough for
# deserialization.  Register it under __main__ so pickle can find it.
class _PoseStub:
    """Stub that absorbs any pickle state without requiring the real GMS deps."""
    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {})

_main_mod = sys.modules.get('__main__')
if _main_mod is not None and not hasattr(_main_mod, 'Pose'):
    setattr(_main_mod, 'Pose', _PoseStub)

# ── On-the-fly CSE rendering ──────────────────────────────────────────────────
# GMS instrument singleton: lives in each DataLoader worker process after the
# first __getitem__ that needs on-the-fly rendering.  Because DataLoader workers
# are forked *after* this module is imported, each worker gets an independent
# copy of this variable (initially None) and initialises it lazily.
_gms_instrument = None
_gms_instrument_device = None
_GMS_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'gaussian-mesh-splatting'))
_GMS_PRETRAIN_PATH = '/mnt/iMVR/daiyun/shuojue-temp/code/Instrument-Splatting/pretrained_models'
_RARP_FC = 587.54401824   # focal length shared by all RARP sequences


def _get_render_cuda_device():
    if not torch.cuda.is_available():
        return None, 0
    idx = torch.cuda.current_device()
    return torch.device(f'cuda:{idx}'), idx


def _build_rot_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Unit quaternion [r, x, y, z] → 3×3 rotation matrix (CPU, float32)."""
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    q = _F.normalize(quat.float(), p=2, dim=1)
    r, x, y, z = q[0, 0], q[0, 1], q[0, 2], q[0, 3]
    R = torch.zeros(3, 3)
    R[0, 0] = 1 - 2*(y*y + z*z);  R[0, 1] = 2*(x*y - r*z);  R[0, 2] = 2*(x*z + r*y)
    R[1, 0] = 2*(x*y + r*z);      R[1, 1] = 1 - 2*(x*x + z*z); R[1, 2] = 2*(y*z - r*x)
    R[2, 0] = 2*(x*z - r*y);      R[2, 1] = 2*(y*z + r*x);  R[2, 2] = 1 - 2*(x*x + y*y)
    return R

NEEDLE_DATASET_ROOT = '/mnt/nas/share/shuojue/data/needlePuncture_videos'
NEEDLE_POSE_ROOT = '/mnt/nas/share/shuojue/data/needlePuncture_results'


def _resize_coord_img(arr, new_h, new_w):
    """Nearest-neighbour resize for (H, W, C) float32 coord array."""
    H, W = arr.shape[:2]
    ri = np.floor(np.arange(new_h) * H / new_h).astype(np.int32)
    ci = np.floor(np.arange(new_w) * W / new_w).astype(np.int32)
    return arr[ri[:, None], ci[None, :]]


def _resolve_mask_subfolder(instance_folder, part, v2_force):
    """
    Resolve the mask subfolder for a given part ('wrist', 'shaft', 'gripper').
    Priority: refined_masks_v2_{part} > refined_masks_{part} > masks_{part}
    """
    priority = [f'refined_masks_v2_{part}', f'refined_masks_{part}', f'masks_{part}'] \
        if v2_force else [f'refined_masks_{part}', f'masks_{part}']
    for variant in priority:
        candidate = os.path.join(instance_folder, variant)
        if os.path.isdir(candidate):
            return candidate
    return None


class RARPInstanceDataset(Dataset):
    """
    Dense RARP dataset for surgical instrument detection/segmentation.

    Returns the same (img_array, annot) format as SurgicalInstruments and is
    compatible with collate_fn_instrument without any changes to the training script.

    Data layout:
        dataset_root/
            SARRARP502022_{video_name}/
                frames/
                    00001.png, 00002.png, ...  (1-indexed, 5-digit)
                instance{N}/
                    masks_wrist/    00000.png, 00001.png, ...  (0-indexed, 5-digit)
                    masks_shaft/    ...
                    masks_gripper/  ...
                    [refined_masks_wrist/]      (higher-priority variant)
                    [refined_masks_wrist_v2/]   (highest-priority variant)

        pose_root/
            SARRARP502022_{video_name}_instance{N}/
                memory_pool.pth   {frame_id_str -> {'dice': [wrist, shaft, gripper]}}

    Part label encoding (matches SurgicalInstruments):
        0=background, 1=gripper, 2=wrist, 3=shaft
    """

    def __init__(
        self,
        split='train',
        training=False,
        img_size=630,
        dataset_root=None,
        pose_root=None,
        min_dice=None,   # [min_shaft_dice, min_wrist_dice, min_gripper_dice]
        train_ratio=0.8,
        subsample=1,
        n=-1,
        v2_force=False,
        cse_coord_root=None,
        render_on_the_fly=False,
        canonicalize_pose_symmetry=False,
        canonical_eps=0.08,
        aug_random_crop_rotate=False,
        aug_geom_prob=1.0,
        aug_crop_scale=1.2,
        aug_max_angle=np.pi,
        aug_offset_scale=1.0,
        aug_color_jitter=False,
        random_resample=True,
        dense_output_stride=4,
    ):
        super().__init__()
        if min_dice is None:
            min_dice = [0.8, 0.6, 0.6]

        self.name = 'rarp_LND'
        self.split = split
        self.training = training
        self.img_size = img_size
        self.dataset_root = dataset_root
        self.pose_root = pose_root
        self.min_dice = min_dice
        self.subsample = subsample
        self.v2_force = v2_force
        self.cse_coord_root = cse_coord_root
        self.render_on_the_fly = render_on_the_fly
        self.canonicalize_pose_symmetry = bool(canonicalize_pose_symmetry)
        self.canonical_eps = float(canonical_eps)
        self.aug_random_crop_rotate = bool(aug_random_crop_rotate)
        self.aug_geom_prob = float(aug_geom_prob)
        self.aug_crop_scale = float(aug_crop_scale)
        self.aug_max_angle = float(aug_max_angle)
        self.aug_offset_scale = float(aug_offset_scale)
        self.aug_color_jitter = bool(aug_color_jitter)
        self.random_resample = bool(random_resample)
        self.dense_output_stride = int(dense_output_stride)
        if self.canonicalize_pose_symmetry and self.cse_coord_root is not None and not self.render_on_the_fly:
            raise ValueError(
                "canonicalize_pose_symmetry requires render_on_the_fly=1 when "
                "cse_coord_root is set, because cached CSE labels were generated "
                "with the original pose convention."
            )
        # (video_name, frame_id, instance_id) → extracted pose tensors (CPU)
        # Populated only when render_on_the_fly=True; stays empty otherwise.
        self.pose_data: dict = {}
        # ------------------------------------------------------------------ #
        # Phase 1: scan pose_root and build per-frame index
        # ------------------------------------------------------------------ #
        # frame_dict[(video_name, frame_id)] -> list of (instance_id, mask_folder_base)
        # mask_folder_base is the instance folder (e.g. .../instance1); the
        # per-part subfolder is resolved lazily in __getitem__ using
        # _resolve_mask_subfolder.
        frame_dict = {}
        if pose_root is None or not os.path.isdir(pose_root):
            case_dirs = sorted(glob.glob(os.path.join(dataset_root, 'SARRARP502022_*')))
        else:
            case_dirs = sorted(glob.glob(os.path.join(pose_root, 'SARRARP502022_*_instance*')))

        for case_dir in case_dirs:
            case_name = os.path.basename(case_dir)

            if '_instance' not in case_name and pose_root is not None:
                warnings.warn(f"Skipping unexpected folder: {case_name}")
                continue
            elif '_instance' not in case_name and pose_root is None:
                # If pose_root is not provided, we assume all videos in dataset_root are valid.
                # Extract video_name from case_name for indexing.
                if not case_name.startswith('SARRARP502022_'):
                    warnings.warn(f"Unexpected folder name: {case_name}")
                    continue
                video_name = case_name[len('SARRARP502022_'):]
                frames_dir = os.path.join(dataset_root, case_name, 'frames_v2') \
                    if os.path.isdir(os.path.join(dataset_root, case_name, 'frames_v2')) and self.v2_force else os.path.join(dataset_root, case_name, 'frames')
                if not os.path.isdir(frames_dir):
                    warnings.warn(f"Frames folder missing: {frames_dir}")
                    continue
                frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.png')) +
                                     glob.glob(os.path.join(frames_dir, '*.jpg')))
                for frame_file in frame_files:
                    frame_id = os.path.splitext(os.path.basename(frame_file))[0]
                    key = (video_name, frame_id)
                    video_folder = os.path.join(dataset_root, f'SARRARP502022_{video_name}')
                    instance_folder1 = os.path.join(video_folder, 'instance1')  # dummy instance folder
                    instance_folder2 = os.path.join(video_folder, 'instance2')  # dummy instance folder
                    if key not in frame_dict:
                        frame_dict[key] = []
                    if os.path.isdir(instance_folder1):
                        frame_dict[key].append((1, instance_folder1))
                    if os.path.isdir(instance_folder2):
                        frame_dict[key].append((2, instance_folder2))
                continue
            
            prefix_and_video, inst_str = case_name.rsplit('_instance', 1)
            try:
                instance_id = int(inst_str.strip())
            except ValueError:
                warnings.warn(f"Could not parse instance id from {case_name}")
                continue

            if not prefix_and_video.startswith('SARRARP502022_'):
                warnings.warn(f"Unexpected prefix: {case_name}")
                continue

            video_name = prefix_and_video[len('SARRARP502022_'):]

            # Paths on the data side
            video_folder = os.path.join(dataset_root, f'SARRARP502022_{video_name}')
            instance_folder = os.path.join(video_folder, f'instance{instance_id}')
            frames_folder = os.path.join(video_folder, 'frames_v2') \
                if os.path.isdir(os.path.join(video_folder, 'frames_v2')) and self.v2_force else os.path.join(video_folder, 'frames')

            if not os.path.isdir(frames_folder):
                warnings.warn(f"Frames folder missing: {frames_folder}")
                continue
            if not os.path.isdir(instance_folder):
                warnings.warn(f"Instance folder missing: {instance_folder}")
                continue

            mem_path = os.path.join(case_dir, 'memory_pool.pth')
            if not os.path.isfile(mem_path):
                warnings.warn(f"memory_pool.pth not found in {case_dir}, skipping.")
                continue

            try:
                memory_pool = torch.load(mem_path, map_location='cpu', weights_only=False)
            except Exception as e:
                warnings.warn(f"Failed to load {mem_path}: {e}")
                continue

            for frame_id, val in memory_pool.items():
                dices = val['dice']
                # dice order in memory_pool: [wrist=0, shaft=1, gripper=2]
                wrist_dice = float(dices[0])
                shaft_dice = float(dices[1])
                gripper_dice = float(dices[2])

                if (shaft_dice < self.min_dice[0] or
                        wrist_dice < self.min_dice[1] or
                        gripper_dice < self.min_dice[2]):
                    continue

                key = (video_name, frame_id)
                if key not in frame_dict:
                    frame_dict[key] = []
                frame_dict[key].append((instance_id, instance_folder))

                if self.render_on_the_fly:
                    pose_info = val.get('pose_info')
                    if pose_info is not None:
                        try:
                            pose_dict = {
                                'rot':     pose_info.rot.detach().float().flatten(),
                                'trans':   pose_info.trans.detach().float().flatten(),
                                'alpha':   pose_info.alpha.detach().float().flatten(),
                                'theta_l': pose_info.theta_l.detach().float().flatten(),
                                'theta_r': pose_info.theta_r.detach().float().flatten(),
                            }
                            self.pose_data[(video_name, frame_id, instance_id)] = (
                                _canonicalize_pose_symmetry(
                                    pose_dict,
                                    eps=self.canonical_eps,
                                    enabled=self.canonicalize_pose_symmetry,
                                )
                            )
                        except AttributeError:
                            pass

        # ------------------------------------------------------------------ #
        # Phase 1.5: exclude videos that overlap with the RARP50 test split
        # to prevent data leakage when evaluating on SurgicalInstruments.
        #
        # RARP50 test folder names are like 'video_41', 'video_42', ...
        # RARPInstanceDataset video names embed the same ID as 'video41',
        # always as the last component after the final underscore, e.g.
        # 'needlePuncture_0_video41'.
        # ------------------------------------------------------------------ #
        _rarp50_test_dir = '/mnt/nas/haofeng/data/RARP50_0910/test/images'
        _exclude_video_ids: set = set()
        if os.path.isdir(_rarp50_test_dir):
            for _vfolder in os.listdir(_rarp50_test_dir):
                if _vfolder.startswith('video_'):
                    _num = int(_vfolder[len('video_'):])
                    _exclude_video_ids.add(f'video{_num}')

        if _exclude_video_ids:
            _before = len(frame_dict)
            frame_dict = {
                (vn, fid): insts
                for (vn, fid), insts in frame_dict.items()
                if not any(
                    vn == vid or vn.endswith(f'_{vid}')
                    for vid in _exclude_video_ids
                )
            }
            _removed = _before - len(frame_dict)
            if _removed:
                warnings.warn(
                    f"RARPInstanceDataset: removed {_removed} frames from "
                    f"{len(_exclude_video_ids)} RARP50 test videos to prevent data leakage."
                )

        # ------------------------------------------------------------------ #
        # Phase 2: train / val split by video name
        # ------------------------------------------------------------------ #
        all_videos = sorted({vn for vn, _ in frame_dict.keys()})
        n_train = max(1, int(len(all_videos) * train_ratio))
        if split == 'train':
            selected_videos = set(all_videos[:n_train])
        else:
            selected_videos = set(all_videos[n_train:])

        self.samples = [
            (vn, fid, insts)
            for (vn, fid), insts in frame_dict.items()
            if vn in selected_videos
        ]
        self.samples.sort(key=lambda s: (s[0], s[1]))

        if n >= 0:
            self.samples = self.samples[:n]
        if self.subsample > 1:
            self.samples = [self.samples[k]
                            for k in range(0, len(self.samples), self.subsample)]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"{self.name}: split={self.split} - N={len(self.samples)}"

    def __getitem__(self, idx):
        if self.training and self.random_resample:
            idx = random.choices(range(len(self.samples)))[0]

        video_name, frame_id, instances_list = self.samples[idx]

        # ------------------------------------------------------------------ #
        # Load RGB image
        # ------------------------------------------------------------------ #
        video_folder = os.path.join(self.dataset_root, f'SARRARP502022_{video_name}')
        if os.path.isdir(os.path.join(video_folder, 'frames_v2')) and self.v2_force:
            frames_folder = os.path.join(video_folder, 'frames_v2')
        else:
            frames_folder = os.path.join(video_folder, 'frames')

        img_path = os.path.join(frames_folder, f'{frame_id}.png')
        if not os.path.isfile(img_path):
            img_path = os.path.join(frames_folder, f'{frame_id}.jpg')

        img_pil = Image.open(img_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        real_width, real_height = img_pil.size
        rgb_np = np.asarray(img_pil)

        # mask files are 0-indexed
        mask_frame_id = f'{int(frame_id) - 1:05d}'

        # ------------------------------------------------------------------ #
        # Load per-instrument masks
        # ------------------------------------------------------------------ #
        instruments_raw = []
        for instance_id, instance_folder in instances_list:
            wrist_dir = _resolve_mask_subfolder(instance_folder, 'wrist', self.v2_force)
            shaft_dir = _resolve_mask_subfolder(instance_folder, 'shaft', self.v2_force)
            gripper_dir = _resolve_mask_subfolder(instance_folder, 'gripper', self.v2_force)

            if wrist_dir is None:
                continue

            wrist_path = os.path.join(wrist_dir, f'{mask_frame_id}.png')
            shaft_path = os.path.join(shaft_dir, f'{mask_frame_id}.png') if shaft_dir else None
            gripper_path = os.path.join(gripper_dir, f'{mask_frame_id}.png') if gripper_dir else None

            if not os.path.isfile(wrist_path):
                continue

            wrist_mask = np.array(Image.open(wrist_path).convert('L')) > 0  # [H, W] bool
            if wrist_mask.sum() == 0:
                continue  # no wrist pixels → can't compute localization point

            shaft_mask = (np.array(Image.open(shaft_path).convert('L')) > 0
                          if shaft_path and os.path.isfile(shaft_path)
                          else np.zeros_like(wrist_mask))
            gripper_mask = (np.array(Image.open(gripper_path).convert('L')) > 0
                            if gripper_path and os.path.isfile(gripper_path)
                            else np.zeros_like(wrist_mask))

            # Wrist center = centroid of wrist pixels (x, y) in original coords
            ys, xs = np.where(wrist_mask)
            wrist_center = np.array([xs.mean(), ys.mean()], dtype=np.float32)

            # Combine into a single part map: 0=bg, 1=gripper, 2=wrist, 3=shaft
            # Write in order shaft < wrist < gripper so gripper wins overlap
            part_mask = np.zeros(wrist_mask.shape, dtype=np.int64)
            part_mask[shaft_mask] = 3
            part_mask[wrist_mask] = 2
            part_mask[gripper_mask] = 1

            # Instance binary mask = union of all three parts
            inst_mask = (wrist_mask | shaft_mask | gripper_mask).astype(np.float32)

            # Optional: load pre-cached CSE canonical-coordinate image.
            # File layout: {cse_coord_root}/SARRARP502022_{video_name}_instance{id}/{frame_id}.npy
            # Array shape: (H_orig, W_orig, 4) float32
            #   channels 0-2 = xyz / canon_scale  (normalized canonical coords)
            #   channel  3   = part_id (0=bg, 1=shaft, 2=wrist, 3=l_gripper, 4=r_gripper)
            coord_img = None
            if self.cse_coord_root is not None:
                coord_file = os.path.join(
                    self.cse_coord_root,
                    f'SARRARP502022_{video_name}_instance{instance_id}',
                    f'{frame_id}.npy',
                )
                if os.path.isfile(coord_file):
                    coord_img = np.load(coord_file).astype(np.float32)

            if coord_img is None and self.render_on_the_fly:
                pose_key = (video_name, frame_id, instance_id)
                if pose_key in self.pose_data:
                    coord_img = self._render_cse_coord(
                        self.pose_data[pose_key], real_height, real_width
                    )

            instruments_raw.append({
                'instance_id': instance_id,
                'inst_mask': inst_mask,       # [H, W] float32
                'part_mask': part_mask,        # [H, W] int64
                'wrist_center': wrist_center,  # [2] (x, y)
                'coord_img': coord_img,        # [H, W, 4] float32 or None
                'pose_sym_flipped': bool(
                    self.pose_data.get(
                        (video_name, frame_id, instance_id), {}
                    ).get('pose_sym_flipped', torch.tensor(False)).item()
                ),
            })

        rgb_sq, instruments_raw, _ = transform_rarp_sample(
            rgb_np,
            instruments_raw,
            img_size=self.img_size,
            output_size=self.img_size // 4,
            dense_output_size=(
                self.img_size // self.dense_output_stride
                if self.dense_output_stride != 4
                else None
            ),
            training=self.training,
            aug_random_crop_rotate=self.aug_random_crop_rotate,
            aug_geom_prob=self.aug_geom_prob,
            aug_crop_scale=self.aug_crop_scale,
            aug_max_angle=self.aug_max_angle,
            aug_offset_scale=self.aug_offset_scale,
            aug_color_jitter=self.aug_color_jitter,
        )
        img_array = normalize_rgb(rgb_sq, imagenet_normalization=1)

        annot = {
            'imagename': f'{video_name}/{frame_id}',
            'instruments': instruments_raw,
            'dataset_source': 1,   # 1 = partially annotated (RARP-Instance)
        }

        return img_array, annot

    def _render_cse_coord(self, pose_params: dict, img_h: int, img_w: int):
        """
        Render a canonical-coordinate image on-the-fly using the GMS instrument.

        Initialises a per-rank singleton (module-level _gms_instrument) on
        first call.  In DDP, the GMS code uses plain .cuda() in several places,
        so we explicitly set/use this rank's current CUDA device before GMS init
        and FK to avoid mixing logical cuda:0 and cuda:1 tensors.

        The GMS submodule has its own 'utils' package that conflicts with
        multiipr's 'utils'.  We swap sys.modules temporarily during GMS
        import/init so both can coexist in the same interpreter.

        Returns (img_h, img_w, 4) float32 or None on failure.
        """
        global _gms_instrument, _gms_instrument_device
        cuda_device, egl_device_idx = _get_render_cuda_device()
        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)

        if _gms_instrument is None or _gms_instrument_device != cuda_device:
            if _GMS_ROOT not in sys.path:
                sys.path.insert(0, _GMS_ROOT)
            # Save and evict multiipr's 'utils' so GMS can import its own.
            _saved = {k: v for k, v in sys.modules.items()
                      if k == 'utils' or k.startswith('utils.')}
            for k in _saved:
                del sys.modules[k]
            try:
                if cuda_device is None:
                    from instrument_gaussian_wrapper import instrument_gaussian_wrapper
                    _gms_instrument = instrument_gaussian_wrapper(
                        pretrain_path=_GMS_PRETRAIN_PATH
                    ).get_instrument()
                else:
                    with torch.cuda.device(cuda_device):
                        from instrument_gaussian_wrapper import instrument_gaussian_wrapper
                        _gms_instrument = instrument_gaussian_wrapper(
                            pretrain_path=_GMS_PRETRAIN_PATH
                        ).get_instrument()
                _gms_instrument_device = cuda_device
            except Exception as e:
                warnings.warn(f"GMS instrument init failed: {e}")
                return None
            finally:
                # Evict GMS's utils and restore multiipr's utils.
                for k in [k for k in sys.modules if k == 'utils' or k.startswith('utils.')]:
                    del sys.modules[k]
                sys.modules.update(_saved)

        try:
            device = cuda_device if cuda_device is not None else torch.device('cpu')
            rot_mat = _build_rot_matrix(pose_params['rot']).to(device)
            trans   = pose_params['trans'].flatten().to(device)
            # Keep shape [1] — _rodrigues_rotation_matrix needs a 1-d tensor
            # so that torch.stack produces [3,1] rows, giving a [3,3] output.
            alpha   = pose_params['alpha'].flatten().to(device)
            theta_l = pose_params['theta_l'].flatten().to(device)
            theta_r = pose_params['theta_r'].flatten().to(device)

            with torch.no_grad():
                if cuda_device is not None:
                    with torch.cuda.device(cuda_device):
                        _gms_instrument.forward_kinematics(
                            rot_mat, trans, alpha, theta_l, theta_r,
                            device=device, pose_grad=False, with_mesh=True,
                        )
                else:
                    _gms_instrument.forward_kinematics(
                        rot_mat, trans, alpha, theta_l, theta_r,
                        device=device, pose_grad=False, with_mesh=True,
                    )
            K = np.array([[_RARP_FC, 0.0, img_w / 2.0],
                          [0.0, _RARP_FC, img_h / 2.0],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
            return _gms_instrument.render_canonical_coords(
                K, img_h, img_w, device_idx=egl_device_idx
            ).astype(np.float32)
        except Exception as e:
            warnings.warn(f"CSE on-the-fly render failed: {e}")
            return None

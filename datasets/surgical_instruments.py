import os
import json
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFile
from torch.utils.data import Dataset
from utils import normalize_rgb
import random
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

RARP50_DIR = '/mnt/nas/haofeng/data/RARP50'


class SurgicalInstruments(Dataset):
    """
    RARP50 dataset for surgical instrument detection and segmentation.

    Data structure:
        images/<video_id>/<frame>.jpg   - RGB images (1920x1080)
        masks/<video_id>/<frame>.png    - Instance segmentation (pixel=instrument_id)
        parts/<video_id>/<frame>.png    - Part segmentation, encoded as part_type*20+inst_id (part_type: 0=gripper,1=wrist,2=shaft)
        Meta/<video_id>.json            - Metadata with instrument categories

    Only instruments with a wrist part (part=1) are used for detection.
    """

    def __init__(self,
                 split='train',
                 training=False,
                 img_size=896,
                 root_dir=RARP50_DIR,
                 subsample=1,
                 n=-1):
        super().__init__()

        self.name = 'rarp50'
        self.split = split
        self.training = training
        self.img_size = img_size
        self.root_dir = root_dir
        self.subsample = subsample

        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        self.parts_dir = os.path.join(root_dir, split, 'parts')
        self.meta_dir = os.path.join(root_dir, split, 'Meta')

        # Build sample list: (video_id, frame_id)
        self.samples = []
        self.video_categories = {}  # video_id -> {inst_id: category_name}

        videos = sorted(os.listdir(self.image_dir))
        for video_id in videos:
            # Load metadata
            meta_file = os.path.join(self.meta_dir, f"{video_id}.json")
            if os.path.isfile(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                self.video_categories[video_id] = meta.get('info', {}).get('category', {})

            # List frames
            frames_dir = os.path.join(self.image_dir, video_id)
            frames = sorted(os.listdir(frames_dir))
            for frame_fn in frames:
                frame_id = frame_fn.replace('.jpg', '')
                self.samples.append((video_id, frame_id))

        if n >= 0:
            self.samples = self.samples[:n]

        if self.subsample > 1:
            self.samples = [self.samples[k] for k in np.arange(0, len(self.samples), self.subsample).tolist()]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"{self.name}: split={self.split} - N={len(self.samples)}"

    def __getitem__(self, idx):
        if self.training:
            idx = random.choices(range(len(self.samples)))[0]

        video_id, frame_id = self.samples[idx]

        # Load image
        img_path = os.path.join(self.image_dir, video_id, f"{frame_id}.jpg")
        img_pil = Image.open(img_path)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        real_width, real_height = img_pil.size  # (1920, 1080)

        # Load mask and parts
        mask_path = os.path.join(self.mask_dir, video_id, f"{frame_id}.png")
        parts_path = os.path.join(self.parts_dir, video_id, f"{frame_id}.png")
        mask_np = np.array(Image.open(mask_path))   # [H, W] uint8, pixel=instrument_id
        parts_np = np.array(Image.open(parts_path))  # [H, W] uint8, 1=wrist,2=shaft,3=body

        # Decode composite part encoding: value = part_type * 20 + inst_id
        # part_type: 0=gripper, 1=wrist, 2=shaft
        decoded_parts = np.where(parts_np > 0, parts_np // 20, 0)

        # Extract per-instrument annotations (only those with wrist part)
        instruments = []
        inst_ids = np.unique(mask_np)
        for inst_id in inst_ids:
            if inst_id == 0:
                continue
            inst_binary = (mask_np == inst_id)
            wrist_region = inst_binary & (decoded_parts == 1)  # part_type 1 = wrist
            if wrist_region.sum() == 0:
                continue  # skip instruments without wrist

            # Wrist center = centroid of wrist pixels
            ys, xs = np.where(wrist_region)
            wrist_center = np.array([xs.mean(), ys.mean()], dtype=np.float32)  # (x, y)

            # Per-instrument part labels: 0=bg, 1=gripper, 2=wrist, 3=shaft
            inst_parts = np.zeros_like(mask_np, dtype=np.int64)
            for part_type, label in [(0, 1), (1, 2), (2, 3)]:
                inst_parts[inst_binary & (decoded_parts == part_type)] = label

            instruments.append({
                'inst_mask': inst_binary.astype(np.float32),   # [H, W]
                'part_mask': inst_parts,                        # [H, W] int64
                'wrist_center': wrist_center,                   # [2] (x, y) in original pixel coords
            })

        # Resize image (aspect-ratio-preserving + pad to square)
        scale = self.img_size / max(real_width, real_height)
        new_w, new_h = int(real_width * scale), int(real_height * scale)

        img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size))
        img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size))

        # Compute padding offset (ImageOps.pad centers the image)
        pad_x = (self.img_size - new_w) // 2
        pad_y = (self.img_size - new_h) // 2

        # Apply same transform to masks and wrist centers
        quarter_size = self.img_size // 4
        for inst in instruments:
            # Resize mask to new_w x new_h, then pad to img_size x img_size
            m = Image.fromarray(inst['inst_mask']).resize((new_w, new_h), Image.NEAREST)
            m_padded = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            m_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(m)
            # Downsample to 1/4 resolution
            m_quarter = np.array(Image.fromarray(m_padded).resize(
                (quarter_size, quarter_size), Image.NEAREST))
            inst['inst_mask'] = m_quarter  # [H/4, W/4]

            # Part mask: same transform
            p = Image.fromarray(inst['part_mask'].astype(np.uint8)).resize((new_w, new_h), Image.NEAREST)
            p_padded = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            p_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(p)
            p_quarter = np.array(Image.fromarray(p_padded).resize(
                (quarter_size, quarter_size), Image.NEAREST))
            inst['part_mask'] = p_quarter.astype(np.int64)  # [H/4, W/4]

            # Transform wrist center
            wc = inst['wrist_center']
            wc[0] = wc[0] * scale + pad_x
            wc[1] = wc[1] * scale + pad_y
            inst['wrist_center'] = wc

        # Image to array
        img_array = np.asarray(img_pil)
        img_array = normalize_rgb(img_array, imagenet_normalization=1)

        annot = {
            'imagename': f"{video_id}/{frame_id}",
            'instruments': instruments,
            'dataset_source': 0,   # 0 = fully annotated (RARP50)
        }

        return img_array, annot


def collate_fn_instrument(x, *args, **kwargs):
    """
    Collate function for SurgicalInstruments dataset.
    Pads to max instruments per batch.
    """
    bs = len(x)

    # RGB images
    img_array = torch.from_numpy(np.stack([x[i][0] for i in range(bs)])).float()

    y = {}
    y['imagename'] = [x[i][1]['imagename'] for i in range(bs)]

    # Number of instruments per image
    n_insts = [len(x[i][1]['instruments']) for i in range(bs)]
    max_insts = max(n_insts) if max(n_insts) > 0 else 1

    # Validity mask
    valid = np.zeros((bs, max_insts), dtype=np.float32)
    for i in range(bs):
        valid[i, :n_insts[i]] = 1.0
    y['valid_instruments'] = torch.from_numpy(valid)

    # Wrist centers [bs, max_insts, 2]
    wrist_centers = np.zeros((bs, max_insts, 2), dtype=np.float32)
    for i in range(bs):
        for j, inst in enumerate(x[i][1]['instruments']):
            wrist_centers[i, j] = inst['wrist_center']
    y['wrist_centers'] = torch.from_numpy(wrist_centers)

    # Instance masks [bs, max_insts, H/4, W/4]
    # Find mask shape from the first image that has instruments
    h, w = None, None
    for i in range(bs):
        if len(x[i][1]['instruments']) > 0:
            h, w = x[i][1]['instruments'][0]['inst_mask'].shape
            break
    if h is None:
        quarter = img_array.shape[-1] // 4
        h, w = quarter, quarter

    inst_masks = np.zeros((bs, max_insts, h, w), dtype=np.float32)
    part_masks = np.zeros((bs, max_insts, h, w), dtype=np.int64)
    for i in range(bs):
        for j, inst in enumerate(x[i][1]['instruments']):
            inst_masks[i, j] = inst['inst_mask']
            part_masks[i, j] = inst['part_mask']
    y['inst_masks'] = torch.from_numpy(inst_masks)
    y['part_masks'] = torch.from_numpy(part_masks)

    y['n_instruments'] = torch.tensor(n_insts, dtype=torch.float32)

    if 'dataset_source' in x[0][1]:
        y['dataset_source'] = torch.tensor(
            [x[i][1]['dataset_source'] for i in range(bs)], dtype=torch.long)

    return img_array, y

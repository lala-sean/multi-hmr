import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageFile
from torch.utils.data import Dataset
from utils import normalize_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True

# EndoVis raw part value → our convention (0=bg, 1=gripper, 2=wrist, 3=shaft)
# Verified across E17/E18 train+valid: spatial order is always shaft→wrist→gripper
#   raw 1 = shaft   → our 3
#   raw 2 = gripper → our 1
#   raw 3 = wrist   → our 2  (used for detection center)
#   raw 4 = gripper (E18 scissors clasper tip) → our 1
# Values ≥5 (tissue: 5,10; ultrasound: 40) stay 0 → no wrist → filtered out.
_PART_REMAP = {1: 3, 2: 1, 3: 2, 4: 1}

class EndoVisDataset(Dataset):
    """
    EndoVis17 / EndoVis18 dataset for surgical instrument segmentation.
    Returns the same sample format as SurgicalInstruments so it can be used
    directly with collate_fn_instrument and train_instrument_rarp.py.

    Data structure:
        {split}/JPEGImages/<seq_N>/<00000>.png  - RGB images (1280×1024)
        {split}/Annotations/<seq_N>/<00000>.png - Instance masks (pixel=inst_id)
        {split}/PartAnnotations/<seq_N>/<00000>.png - Part masks
            EndoVis raw: 1=shaft, 2=gripper, 3=wrist, 4=gripper(E18)
            Our conv:    3=shaft, 1=gripper, 2=wrist

    Only instruments with at least one wrist pixel (EndoVis raw part==3) are kept.
    """

    def __init__(self, root_dir, split='train', training=False,
                 img_size=896, subsample=1):
        super().__init__()

        self.name = 'endovis17' if '17' in os.path.basename(root_dir.rstrip('/')) else 'endovis18'
        self.split = split
        self.training = training
        self.img_size = img_size

        self.img_root  = os.path.join(root_dir, split, 'JPEGImages')
        self.ann_root  = os.path.join(root_dir, split, 'Annotations')
        self.part_root = os.path.join(root_dir, split, 'PartAnnotations')

        self.samples = []   # [(seq, stem), ...]
        for seq in sorted(os.listdir(self.img_root)):
            seq_dir = os.path.join(self.img_root, seq)
            if not os.path.isdir(seq_dir):
                continue
            for fn in sorted(os.listdir(seq_dir)):
                stem = os.path.splitext(fn)[0]
                self.samples.append((seq, stem))

        if subsample > 1:
            self.samples = self.samples[::subsample]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"{self.name}: split={self.split} - N={len(self.samples)}"

    def __getitem__(self, idx):
        if self.training:
            idx = random.choices(range(len(self.samples)))[0]

        seq, stem = self.samples[idx]

        img_pil  = Image.open(os.path.join(self.img_root,  seq, stem + '.png')).convert('RGB')
        mask_np  = np.array(Image.open(os.path.join(self.ann_root,  seq, stem + '.png')))
        parts_np = np.array(Image.open(os.path.join(self.part_root, seq, stem + '.png')))

        real_w, real_h = img_pil.size   # 1280, 1024
        scale = self.img_size / max(real_w, real_h)
        new_w  = int(real_w * scale)
        new_h  = int(real_h * scale)
        pad_x  = (self.img_size - new_w) // 2
        pad_y  = (self.img_size - new_h) // 2
        quarter_size = self.img_size // 4

        # Collect per-instrument annotations (original resolution)
        instruments = []
        for inst_id in np.unique(mask_np):
            if inst_id == 0:
                continue
            inst_binary  = mask_np == inst_id
            wrist_region = inst_binary & (parts_np == 3)   # raw part 3 = wrist in EndoVis
            if wrist_region.sum() == 0:
                continue   # skip: ultrasound probe, tissue, etc.

            ys, xs = np.where(wrist_region)
            wrist_center = np.array([xs.mean(), ys.mean()], dtype=np.float32)  # (x, y)

            # Part mask: remap EndoVis encoding → our convention
            inst_parts = np.zeros_like(parts_np, dtype=np.int64)
            for endo_val, our_label in _PART_REMAP.items():
                inst_parts[inst_binary & (parts_np == endo_val)] = our_label

            instruments.append({
                'inst_mask': inst_binary.astype(np.float32),   # [H, W]
                'part_mask': inst_parts,                        # [H, W] int64
                'wrist_center': wrist_center,                   # [2] (x,y) original coords
            })

        # Resize image (aspect-ratio-preserving + centered pad to square)
        img_pil = ImageOps.contain(img_pil, (self.img_size, self.img_size))
        img_pil = ImageOps.pad(img_pil, size=(self.img_size, self.img_size))

        # Apply same spatial transform to masks and wrist centers
        for inst in instruments:
            # inst_mask: resize → pad → quarter downsample
            m = Image.fromarray(inst['inst_mask']).resize((new_w, new_h), Image.NEAREST)
            m_padded = np.zeros((self.img_size, self.img_size), dtype=np.float32)
            m_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(m)
            m_quarter = np.array(Image.fromarray(m_padded).resize(
                (quarter_size, quarter_size), Image.NEAREST))
            inst['inst_mask'] = m_quarter   # [H/4, W/4] float32

            # part_mask: same transform
            p = Image.fromarray(inst['part_mask'].astype(np.uint8)).resize(
                (new_w, new_h), Image.NEAREST)
            p_padded = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            p_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = np.array(p)
            p_quarter = np.array(Image.fromarray(p_padded).resize(
                (quarter_size, quarter_size), Image.NEAREST))
            inst['part_mask'] = p_quarter.astype(np.int64)   # [H/4, W/4]

            # wrist_center: scale + pad offset
            wc = inst['wrist_center']
            wc[0] = wc[0] * scale + pad_x
            wc[1] = wc[1] * scale + pad_y
            inst['wrist_center'] = wc

        img_array = normalize_rgb(np.asarray(img_pil), imagenet_normalization=1)

        

        annot = {
            'imagename': f'{seq}/{stem}',
            'instruments': instruments,
            'dataset_source': 0,   # 0 = fully annotated
        }
        return img_array, annot

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

from argparse import ArgumentParser
import torch
import numpy as np
from PIL import Image, ImageOps
from multi_instrument import Multi_Instrument
from utils import normalize_rgb

# part_id -> color: 0=background, 1=gripper=red, 2=wrist=green, 3=shaft=blue
PART_COLORS = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]


def preprocess(img_path, img_size):
    img_pil = Image.open(img_path).convert('RGB')
    img_pil = ImageOps.contain(img_pil, (img_size, img_size))
    img_pil = ImageOps.pad(img_pil, size=(img_size, img_size))
    img_np = np.array(img_pil)                                             # [H, W, 3] uint8
    img_tensor = torch.from_numpy(normalize_rgb(img_np)).unsqueeze(0)     # [1, 3, H, W]
    return img_tensor, img_np


def visualize_pred(img_np, pred_list, alpha=0.45):
    """
    Returns [original | pred overlay] side-by-side.
    Pred overlay: part segmentation colored by part id + wrist center crosshairs.
    """
    H, W = img_np.shape[:2]
    canvas = img_np.copy().astype(np.float32)

    for p in pred_list:
        inst_mask = (torch.sigmoid(p['inst_mask_logits']) > 0.5).cpu().numpy()  # [h, w]
        part_mask = p['part_mask_logits'].argmax(dim=0).cpu().numpy()            # [h, w]

        m_full = np.array(Image.fromarray(inst_mask.astype(np.uint8)).resize((W, H), Image.NEAREST))
        p_full = np.array(Image.fromarray(part_mask.astype(np.uint8)).resize((W, H), Image.NEAREST))

        region = m_full > 0
        if region.sum() == 0:
            continue

        color_map = np.zeros((H, W, 3), dtype=np.float32)
        for part_id, color in enumerate(PART_COLORS):
            part_region = region & (p_full == part_id)
            for c in range(3):
                color_map[:, :, c][part_region] = color[c]
        canvas[region] = canvas[region] * (1 - alpha) + color_map[region] * alpha

    canvas_uint8 = canvas.clip(0, 255).astype(np.uint8)

    # Draw wrist center crosshairs
    for p in pred_list:
        loc = p['loc']
        if torch.isnan(loc).any():
            continue
        cx, cy = int(loc[0].item()), int(loc[1].item())
        r = 8
        for dx in range(-r, r + 1):
            for dy in [-1, 0, 1]:
                for xx, yy in [(cx + dx, cy + dy), (cx + dy, cy + dx)]:
                    if 0 <= xx < W and 0 <= yy < H:
                        canvas_uint8[yy, xx] = [255, 255, 255]

    return np.concatenate([img_np, canvas_uint8], axis=1)


def run(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_kwargs = {k: v for k, v in vars(args).items()
                    if k in ('backbone', 'xat_dim', 'xat_depth', 'xat_heads',
                             'xat_dim_head', 'xat_mlp_dim', 'mask_dim', 'num_parts', 'img_size')}
    model = Multi_Instrument(pretrained_backbone=False, **model_kwargs)
    model = model.to(device).eval()

    if args.pretrained is not None and os.path.isfile(args.pretrained):
        print(f"Loading weights from {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        log = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(log)
    else:
        print(f"WARNING: checkpoint not found at {args.pretrained}")

    # Collect input images
    if os.path.isdir(args.input):
        img_paths = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.lower().endswith('.png') or f.lower().endswith('.jpg')
        ])
    else:
        img_paths = [args.input]

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Processing {len(img_paths)} image(s) -> {args.output_dir}")

    with torch.no_grad():
        for img_path in img_paths:
            img_tensor, img_np = preprocess(img_path, args.img_size)
            img_tensor = img_tensor.to(device)

            with torch.cuda.amp.autocast(enabled=True):
                pred_list = model(img_tensor, is_training=False,
                                  det_thresh=args.det_thresh,
                                  nms_kernel_size=args.nms_kernel_size)

            preds = [p for p in pred_list if p['batch_idx'] == 0] if isinstance(pred_list, list) else []
            print(f"  {os.path.basename(img_path)}: {len(preds)} instrument(s) detected")

            vis = visualize_pred(img_np, preds, alpha=args.alpha)

            stem = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(args.output_dir, f"{stem}_seg.jpg")
            Image.fromarray(vis).save(out_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--input', type=str, default= '/mnt/iMVR/daiyun/shuojue-temp/data/surgpose/000000/left_frames/', #'/mnt/nas/share/shuojue/data/jhu_suture_dataset/tissue_10/1_needle_pickup/20250428-223900-225949/left_img_dir',
                        help='Path to a single image or a folder of PNG/JPG images')
    parser.add_argument('--output_dir', type=str, default='demo_outputs/demo_output_surgpose')
    parser.add_argument('--pretrained', type=str,
                        default='logs/rarp_endovis/checkpoints/best.pt')

    # Inference
    parser.add_argument('--img_size', type=int, default=630) # 630
    parser.add_argument('--det_thresh', type=float, default=0.5)
    parser.add_argument('--nms_kernel_size', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.45,
                        help='Overlay transparency')

    # Model architecture (must match checkpoint)
    parser.add_argument('--backbone', type=str, default='dinov2_vits14',
                        choices=['dinov2_vitl14', 'dinov2_vitb14', 'dinov2_vits14'])
    parser.add_argument('--xat_dim', type=int, default=512)
    parser.add_argument('--xat_depth', type=int, default=4)
    parser.add_argument('--xat_heads', type=int, default=16)
    parser.add_argument('--xat_dim_head', type=int, default=32)
    parser.add_argument('--xat_mlp_dim', type=int, default=2048)
    parser.add_argument('--mask_dim', type=int, default=256)
    parser.add_argument('--num_parts', type=int, default=4)

    args = parser.parse_args()
    run(args)

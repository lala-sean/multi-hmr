"""
CSE-aware collate function that extends collate_fn_instrument with coord_img batching.

Adds to the y dict:
    coord_imgs: [bs, max_insts, H/4, W/4, 4] float32
                channels 0-2 = xyz / canon_scale, channel 3 = part_id
                zeros for instruments without coord_img
    has_cse:    [bs, max_insts] bool
                True only when a pre-cached coord_img was available
"""

import numpy as np
import torch

from datasets.surgical_instruments import collate_fn_instrument


def collate_fn_instrument_cse(batch):
    img_array, y = collate_fn_instrument(batch)

    bs = len(batch)
    n_insts = [len(batch[i][1]['instruments']) for i in range(bs)]
    max_inst = max(n_insts) if n_insts and max(n_insts) > 0 else 1

    # Infer spatial size from the first non-None coord_img
    h = w = img_array.shape[-1] // 4
    for i in range(bs):
        for inst in batch[i][1]['instruments']:
            ci = inst.get('coord_img')
            if ci is not None:
                h, w = ci.shape[:2]
                break
        else:
            continue
        break

    coord_imgs = np.zeros((bs, max_inst, h, w, 4), dtype=np.float32)
    has_cse    = np.zeros((bs, max_inst), dtype=bool)

    for i in range(bs):
        for j, inst in enumerate(batch[i][1]['instruments']):
            ci = inst.get('coord_img')
            if ci is not None:
                coord_imgs[i, j] = ci
                has_cse[i, j]    = True

    y['coord_imgs'] = torch.from_numpy(coord_imgs)   # [bs, max_inst, H/4, W/4, 4]
    y['has_cse']    = torch.from_numpy(has_cse)      # [bs, max_inst] bool

    dense_shape = None
    for i in range(bs):
        for inst in batch[i][1]['instruments']:
            ci = inst.get('coord_img_dense')
            if ci is not None:
                dense_shape = ci.shape[:2]
                break
        if dense_shape is not None:
            break
    if dense_shape is not None:
        dh, dw = dense_shape
        coord_imgs_dense = np.zeros((bs, max_inst, dh, dw, 4), dtype=np.float32)
        has_cse_dense = np.zeros((bs, max_inst), dtype=bool)
        for i in range(bs):
            for j, inst in enumerate(batch[i][1]['instruments']):
                ci = inst.get('coord_img_dense')
                if ci is not None:
                    coord_imgs_dense[i, j] = ci
                    has_cse_dense[i, j] = True
        y['coord_imgs_dense'] = torch.from_numpy(coord_imgs_dense)
        y['has_cse_dense'] = torch.from_numpy(has_cse_dense)
    return img_array, y

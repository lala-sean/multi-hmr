import numpy as np
import torch

from datasets.rarp_hcce_pose_dataset import collate_fn_instrument_hcce_pose


def collate_fn_instrument_multitask(batch):
    img_array, y = collate_fn_instrument_hcce_pose(batch)
    bs = len(batch)
    max_inst = y["valid_instruments"].shape[1]

    max_keypoints = 0
    for i in range(bs):
        for inst in batch[i][1]["instruments"]:
            if "keypoints" in inst:
                max_keypoints = max(max_keypoints, int(np.asarray(inst["keypoints"]).reshape(-1, 2).shape[0]))
    if max_keypoints > 0:
        keypoints = np.zeros((bs, max_inst, max_keypoints, 2), dtype=np.float32)
        keypoints_valid = np.zeros((bs, max_inst, max_keypoints), dtype=bool)
        has_keypoints = np.zeros((bs, max_inst), dtype=bool)
        for i in range(bs):
            for j, inst in enumerate(batch[i][1]["instruments"]):
                if "keypoints" not in inst:
                    continue
                pts = np.asarray(inst["keypoints"], dtype=np.float32).reshape(-1, 2)
                valid = np.asarray(
                    inst.get("keypoints_valid", np.ones((pts.shape[0],), dtype=bool)),
                    dtype=bool,
                ).reshape(-1)
                n = min(max_keypoints, pts.shape[0], valid.shape[0])
                if n <= 0:
                    continue
                keypoints[i, j, :n] = pts[:n]
                keypoints_valid[i, j, :n] = valid[:n]
                has_keypoints[i, j] = bool(keypoints_valid[i, j].any())
        y["keypoints"] = torch.from_numpy(keypoints)
        y["keypoints_valid"] = torch.from_numpy(keypoints_valid)
        y["has_keypoints"] = torch.from_numpy(has_keypoints)

    dense_shape = None
    for i in range(bs):
        for inst in batch[i][1]["instruments"]:
            dm = inst.get("depth_map_dense")
            if dm is not None:
                dense_shape = dm.shape
                break
        if dense_shape is not None:
            break
    if dense_shape is not None:
        dh, dw = dense_shape
        depth_maps = np.zeros((bs, max_inst, dh, dw), dtype=np.float32)
        depth_valid = np.zeros((bs, max_inst, dh, dw), dtype=np.float32)
        has_depth = np.zeros((bs, max_inst), dtype=bool)
        for i in range(bs):
            for j, inst in enumerate(batch[i][1]["instruments"]):
                dm = inst.get("depth_map_dense")
                dv = inst.get("depth_valid_dense")
                if dm is not None and dv is not None:
                    depth_maps[i, j] = dm
                    depth_valid[i, j] = dv
                    has_depth[i, j] = bool(inst.get("has_depth", np.any(dv > 0)))
        y["depth_maps_dense"] = torch.from_numpy(depth_maps)
        y["depth_valid_dense"] = torch.from_numpy(depth_valid)
        y["has_depth"] = torch.from_numpy(has_depth)

    return img_array, y

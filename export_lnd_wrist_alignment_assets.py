#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from instrument_opengl_renderer import _load_part_mesh


R_LND_FROM_REPO = np.array(
    [
        [-3.965562107231e-04, -9.999972555239e-01, -2.309044784927e-03],
        [-9.999996414411e-01, 3.948273259691e-04, 7.491522775273e-04],
        [-7.482385475190e-04, 2.309341037986e-03, -9.999970535372e-01],
    ],
    dtype=np.float64,
)
T_LND_FROM_REPO_MM = np.array(
    [3.687278000000e-04, -1.340887690810e-01, 2.029389171000e-03],
    dtype=np.float64,
)


def prepare_mesh(mesh, name, color):
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.metadata["name"] = name
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        face_colors=np.tile(np.asarray(color, dtype=np.uint8), (len(mesh.faces), 1)),
    )
    return mesh


def mesh_stats(mesh):
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_mm": np.asarray(mesh.bounds, dtype=np.float64).round(6).tolist(),
        "extents_mm": np.asarray(mesh.extents, dtype=np.float64).round(6).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lnd_mesh",
        default="/mnt/iMVR/daiyun/Dataset/LND/models/obj_000001.ply",
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).resolve().parent / "logs" / "lnd_wrist_alignment_viewer" / "assets"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lnd = trimesh.load(args.lnd_mesh, force="mesh", process=False)
    if not isinstance(lnd, trimesh.Trimesh):
        raise TypeError(f"Expected a mesh at {args.lnd_mesh}, got {type(lnd)}")
    lnd = prepare_mesh(lnd, "LND wrist", [39, 146, 225, 255])

    repo = _load_part_mesh("wrist")
    repo.vertices = (
        np.asarray(repo.vertices, dtype=np.float64) * 1000.0 @ R_LND_FROM_REPO.T
        + T_LND_FROM_REPO_MM
    )
    repo = prepare_mesh(repo, "Instrument-Splatting wrist (ICP registered)", [241, 116, 52, 255])

    lnd.export(output_dir / "lnd_wrist.glb")
    repo.export(output_dir / "instrument_splatting_wrist_registered.glb")
    metadata = {
        "coordinate_frame": "LND object-local",
        "units": "mm",
        "registration": {
            "R_lnd_from_repo": R_LND_FROM_REPO.tolist(),
            "t_lnd_from_repo_mm": T_LND_FROM_REPO_MM.tolist(),
            "origin_offset_mm": float(np.linalg.norm(T_LND_FROM_REPO_MM)),
        },
        "lnd": mesh_stats(lnd),
        "instrument_splatting_registered": mesh_stats(repo),
    }
    (output_dir / "alignment.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

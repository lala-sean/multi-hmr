# Noisy Label Learning Workspace

## Branch

- Development branch: `noisy-learning`
- Branch base: `master` at `9cf73b2`
- All noisy-label-learning changes should be developed and committed on this branch.
- Do not overwrite source labels or memory pools. Every generated noisy-label set must have an immutable version and manifest.

## Goal

Build a reproducible noisy-label-learning pipeline for articulated surgical instrument pose estimation.

The planned supervision and evaluation sources are:

1. **RARP:** generate multiple versions of noisy full pose/action labels and use them for first-round training. Evaluate on fixed RARP train and test subsets as the in-domain benchmark.
2. **SurgRIPE-LND:** use LND TRAIN as additional training data. The original LND pose is clean but supervises only the wrist 6D pose; the Instrument Splatting refined memory supplies pseudo wrist pose plus articulation (`alpha`, `theta_l`, `theta_r`). Evaluate LND TRAIN and TEST with the supervision boundary stated explicitly.
3. **SurgPose:** use its temporal 2D instrument keypoints as an external point-tracking/generalization benchmark. Do not assume that it provides compatible 6D pose GT.

Here, "LND original GT is wrist-only" means it does not provide reliable shaft and left/right gripper articulation labels. Refined pose/action must remain marked as pseudo GT rather than clean GT.

## Dataset Registry

### RARP

| Item | Path / definition |
| --- | --- |
| Dataset implementation | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/datasets/rarp_instrument.py` |
| Base frame/split implementation | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/datasets/RarpInstanceDataset.py` |
| Main pose trainer | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/train_instrument_pose.py` |
| Needle puncture RGB | `/mnt/nas/share/shuojue/data/needlePuncture_videos` |
| Needle puncture pose memory | `/mnt/nas/share/shuojue/data/needlePuncture_results` |
| Needle grasping RGB | `/mnt/nas/share/shuojue/data/needleGrasping_videos` |
| Needle grasping pose memory | `/mnt/nas/share/shuojue/data/needleGrasping_results` |
| Knotting RGB | `/mnt/nas/share/shuojue/data/knotting_videos` |
| Knotting pose memory | `/mnt/nas/share/shuojue/data/knotting_results` |
| RARP50 masks/metadata | `/mnt/nas/haofeng/data/RARP50` |
| Local new labels | `/mnt/iMVR/daiyun/shuojue-temp/data/RARP50_new_labels` |

The current pose record is normally loaded from:

```text
{pose_root}/SARRARP502022_{video_name}_instance{instance_id}/memory_pool.pth
```

Each frame record provides `rot`, `trans`, `alpha`, `theta_l`, and `theta_r`. The current dataset also applies the established pose-symmetry canonicalization.

**Split risk:** `RarpInstanceDataset.py` currently tries to exclude held-out test videos through `/mnt/nas/haofeng/data/RARP50_0910/test/images`, but that directory is absent. The exclusion can silently become a no-op. Before noisy-label experiments, replace this behavior with checked-in, explicit frame manifests and assert that train/test video and frame IDs are disjoint.

Useful evaluation entry points:

- `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/eval_rarp_hcce_pose_action_seg.py`
- `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/eval_robopepp_keypoint_trimesh.py`
- `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/eval_surfemb_articulated_rarp.py`

### SurgRIPE-LND

| Item | Path / definition |
| --- | --- |
| Dataset root | `/mnt/iMVR/daiyun/Dataset/LND` |
| Dataset implementation | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/datasets/surgripe_lnd_instrument.py` |
| TRAIN original wrist pose | `/mnt/iMVR/daiyun/Dataset/LND/TRAIN/pose/{frame_id}.npy` |
| TEST original wrist pose | `/mnt/iMVR/daiyun/Dataset/LND/TEST/pose/{frame_id}.npy` |
| Pre-refine action memory | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/gaussian-mesh-splatting/Results2/surgripe_lnd_action_gt_full/TRAIN/memory_pool.json` |
| Refined pseudo memory | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/gaussian-mesh-splatting/Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json` |
| Refine visual comparison | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/visualize_lnd_train_gt_vs_refined_fullmesh.py` |

Current frame counts:

- `TRAIN`: 1,147 frames
- `TEST`: 373 frames
- `TEST_occ`: 238 frames

The direct LND pose is converted into the repository/Instrument-Splatting wrist frame by `lnd_pose_to_repo_wrist_pose`. It supplies clean wrist rotation and translation only. The refined TRAIN memory supplies wrist pose and articulation pseudo labels.

Evaluation rules:

- **LND TRAIN clean metric:** compare wrist rotation/translation against direct LND GT.
- **LND TRAIN pseudo full-pose metric:** articulation/full-mesh evaluation may compare against refined memory, but must be named a pseudo-label agreement metric.
- **LND TEST:** report wrist-only rotation and translation against direct GT. Existing evaluations exclude confirmed outlier frame `210`; report both the 372-frame primary result and frame 210 separately.
- **LND TEST_occ:** use as a separately named occlusion stress test, never merge it silently with TEST.

Useful evaluation entry points:

- `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/eval_surgripe_lnd_batched.py`
- `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/eval_surfemb_wrist_lnd.py`
- `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/submodules/RoboPEPP/eval_robopepp_keypoint_trimesh.py`

### SurgPose

| Item | Path / definition |
| --- | --- |
| Complete dataset root | `/mnt/nas/share/shuojue/data/surgpose` |
| Partial local copy | `/mnt/iMVR/daiyun/shuojue-temp/data/surgpose` |
| Dataset implementation | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/datasets/surgpose_instruments.py` |
| Existing keypoint trainer | `/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr/train_instrument_hcce_crop_surgpose_keypoint_dpt.py` |

Expected episode data includes `processed_stereo_640/left_frames`, SAM instance/part masks, `depth_npy`, and `keypoints_left_rectified.yaml`. Each instrument has seven tracked image points. The current default split is episodes `000000`-`000027` for train and `000028`-`000033` for validation, corresponding to 28,028 and 6,006 frames respectively.

The existing code evaluates per-frame keypoints. A new temporal evaluator is needed for point tracking, with episode-preserving order and at least:

- visibility-gated pixel error and PCK;
- track survival rate;
- temporal jitter/acceleration;
- cumulative drift against YAML trajectories.

## Proposed Label Contract

Store generated labels outside source datasets, for example:

```text
noisy_labels/
  rarp/
    <label_version>/
      manifest.jsonl
      labels/
      generation_config.yaml
      validation_report.json
```

Each manifest row should contain at least:

```text
dataset, task, video_id, frame_id, instance_id, rgb_path, mask_path,
label_path, label_version, label_source, rot, trans, alpha, theta_l,
theta_r, symmetry_variant, confidence, fit_iou, fit_dice, valid
```

Required invariants:

- rotation/translation units and coordinate frames are explicit;
- source checkpoint, code commit, mesh asset version, intrinsics, and crop convention are recorded;
- symmetry canonicalization is identical between label generation, training, and evaluation;
- invalid labels remain visible in reports and are not silently replaced by another source;
- manifests are deterministic and immutable once used by an experiment.

## Experimental Plan

### Phase 0: Freeze data and metrics

1. Generate explicit RARP train/test manifests and assert zero video/frame overlap.
2. Generate LND TRAIN/TEST/TEST_occ manifests, recording direct-GT and refined-memory availability independently.
3. Freeze SurgPose episode splits and temporal evaluation points.
4. Add dataset-contract tests for pose frames, units, intrinsics, crop transforms, keypoint projections, symmetry, and sample counts.

### Phase 1: Noisy-label baseline

1. Add a model-independent RARP noisy-label adapter; do not modify the existing RARP loader behavior by default.
2. Train round one on one named RARP noisy-label version.
3. Save `best.pt` and `last.pt`, plus the exact manifests/config/commit.
4. Evaluate every checkpoint on fixed RARP TRAIN and TEST in-domain sets.

### Phase 2: LND training anchor

1. Add LND TRAIN without changing the RARP split or augmentation contract.
2. Supervise wrist pose with direct LND GT.
3. Supervise articulation with refined pseudo action only where available.
4. Keep separate loss masks and metrics for clean wrist GT and refined pseudo labels.
5. Compare noisy-RARP-only training with noisy-RARP plus LND training under matched model, crop, batch size, optimizer, schedule, and random seeds.

### Phase 3: Robust noisy-label methods

Start from the plain noisy-label baseline, then add one method at a time. Candidate experiments include confidence weighting, loss-based sample filtering, generalized cross entropy, bootstrapping/EMA teacher targets, and co-teaching. Every method must retain an unfiltered baseline and report performance by label-noise stratum.

### Phase 4: Fixed evaluation suite

For each experiment, report:

- RARP TRAIN and TEST: wrist/full pose, action, joint keypoints, and mesh projection metrics where GT permits;
- LND TRAIN: direct-GT wrist metrics and separately named refined-pseudo agreement;
- LND TEST: direct-GT wrist translation in millimeters and rotation in degrees, with frame 210 handled explicitly;
- LND TEST_occ: separate wrist robustness result;
- SurgPose validation: temporal point-tracking metrics by episode.

## Immediate Work Items

- [ ] Replace the missing `RARP50_0910` split dependency with explicit validated manifests.
- [ ] Define the first RARP noisy-label version and locate/generate its labels.
- [ ] Implement a versioned noisy-label adapter and schema validator.
- [ ] Add LND dual-supervision masks: direct wrist GT versus refined articulation pseudo GT.
- [ ] Add strict train/eval consistency tests for pose convention, symmetry, crop intrinsics, and keypoint projection.
- [ ] Run and archive a round-one noisy-label baseline.
- [ ] Implement the SurgPose temporal point-tracking evaluator.
- [ ] Produce a common report that compares RARP in-domain, LND cross-domain, and SurgPose tracking results.

## Decisions Needed Before Training

1. Which RARP noisy-label generators/versions form the initial noise series.
2. Which model is the first controlled baseline: RoboPEPP, HCCE, or SurfEmb.
3. Whether LND is introduced only after RARP round one or mixed into round one as a clean wrist anchor.
4. The confidence signal used for RARP noisy labels, such as silhouette IoU/Dice, correspondence confidence, reprojection error, or agreement between estimators.

Recommended first controlled run: train round one on a frozen RARP noisy-label manifest, then fine-tune/mix LND TRAIN using direct GT only for wrist supervision and refined memory only for articulation. This cleanly separates noisy-label learning from the value of LND supervision.

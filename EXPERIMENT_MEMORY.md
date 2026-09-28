# Surgical Instrument Pose / SurfEmb Experiment Memory

更新：2026-09-28。本文用于保存环境、实验路线、已测结果与排错经验，支持 `noisy-learning` 后续开发。
本文不是上游 Multi-HMR 的使用说明，也不是新的训练或测试报告。历史结果主要来自 2026-08 的本地实验，2026-09-08 已整理并核对逐帧 CSV。

## 1. 当前分支与发布边界

| 仓库 | 当前分支 | 当前提交 |
|---|---|---|
| 主仓库 multi-hmr | `noisy-learning` | 本文所在发布提交；上一提交为 `d1c1539` |
| RoboPEPP | `codex/workspace-snapshot-20260815-robopepp` | `a22a6fe` |
| gaussian-mesh-splatting | `codex/workspace-snapshot-20260815` | `4757090b` |
| 原版 SurfEmb 副本 | `master` | `53e1852` |

以 `noisy-learning` 为本次最新发布版本；`master` 保持在 `9cf73b2`。本次发布前已通过 `git ls-remote` 确认远端 `noisy-learning` 为 `d1c1539`，不强推或覆盖其他分支的历史。

本次纳入了原未提交的 `scripts/summarize_surfemb_experiments.py` 和 RoboPEPP 的 `visualize_lnd_train_large_refine_wrist_trimesh.py`，以及 memory 文档、导出脚本和轻量记录。

**本次已修复缺失的子模块映射：** 新增根目录 `.gitmodules`，并补齐 Instrument Splatting 的嵌套 `.gitmodules`。RoboPEPP、DINOv3 及嵌套依赖使用本人的 `lala-sean/multi-hmr` 快照分支；Instrument Splatting 使用 `lala-sean/instrument-splatting`，HCCEPose/SurfEmb 使用对应上游。新子仓库提交先推送，再更新主仓库 gitlink。递归 submodule 状态检查通过；未执行下载全部依赖的全新 clone 测试。

其余 gitlink：HCCEPose `b9fb63c`，dinov3 `9ec5f11`；GMS 内部 RoboPEPP `1307c348`、simple-knn `5bc92a00`。本次更新 RoboPEPP 和 GMS 指针，不修改它们的训练算法。

克隆需要有权限访问相应仓库的 SSH key；数据和预训练权重仍需单独准备：

```bash
git clone --branch noisy-learning --recurse-submodules git@github.com:lala-sean/multi-hmr.git
```

本次新增发布内容仅限代码、本文、子模块配置、小型汇总 CSV/JSON，没有新增权重或图片。**不要上传数据集、checkpoint、预训练权重、Results2、evaluation 原始输出、渲染图片或视频。** 根仓库 ignore 不会替代独立子仓库的 ignore；已被 Git 跟踪的文件也不会因为后来添加 ignore 自动消失。本次没有重写历史或清理早期已提交的资产。

## 2. 现有环境

### 实际使用的运行时

Augfix 历史日志指向 `/mnt/iMVR/daiyun/anaconda3/bin/python`，即 Anaconda base。以下为本次实测版本，不声称这些包从历史训练至今完全没更新。

| 项目 | 当前值 |
|---|---|
| OS | Ubuntu 20.04.6 LTS |
| Python | 3.11.4 |
| GPU | 8 x NVIDIA RTX A5000，24564 MiB/卡；是否空闲需实时检查 |
| NVIDIA driver | 570.153.02 |
| PyTorch / torchvision | 2.5.1 / 0.20.1 |
| PyTorch CUDA build | 12.4，不等于系统 nvcc 的版本声明 |
| numpy / scipy | 1.26.4 / 1.10.1 |
| opencv-python / Pillow | 4.11.0.86 / 9.4.0 |
| moderngl / glcontext | 5.12.0 / 3.0.0 |
| PyOpenGL / trimesh / pyrender | 3.1.0 / 4.12.2 / 0.1.45 |
| timm / einops | 1.0.27 / 0.8.2 |
| matplotlib / pytest / PyYAML | 3.7.1 / 7.4.0 / 6.0 |

机器可读记录：[environment_snapshot.json](docs/experiment_memory/environment_snapshot.json)。这是当前环境清单，**不是经过全新安装验证的 lockfile**。

- `submodules/RoboPEPP/requirements.txt` 仍写 torch 2.4.1 / torchvision 0.19.1，不是当前实测环境。
- `submodules/surfemb/environment.yml` 是上游旧环境（Python 3.8 / CUDA 10.2），不能当作本项目的训练环境快照。
- 当前 base 未安装 albumentations、pytorch-lightning、torch-scatter、pymeshlab。定制 Augfix 通过本地 NumPy/OpenCV augmentation 和直接引用 SurfEmb 的 U-Net/SIREN 工作，并不等于可以直接运行上游所有训练入口。
- 本次验证 Augfix 入口 `--help` 能正常导入并退出；未重新跑 CUDA 训练、EGL 渲染或全量 validation。
- 数据生成曾使用独立 `sam3`、`instrument_splatting` 环境。本次未验证它们的依赖版本，不要与 SurfEmb base 混为一套环境。

常用本机设置：

```bash
export REPO=/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr
export PYTHON=/mnt/iMVR/daiyun/anaconda3/bin/python
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
cd "$REPO"
"$PYTHON" submodules/RoboPEPP/train_surfemb_resnet_wrist_only_lnd_prerefine_augfix.py --help
```

OpenGL renderer 使用 standalone EGL context。不要在同一进程交替混用 pyrender 和 ModernGL；EGL device index 与 CUDA 设备可见性需要单独确认，不能假定 `CUDA_VISIBLE_DEVICES` 完全控制 EGL 枚举。

## 3. 数据与资产位置

文中代码路径相对仓库根目录；绝对数据路径是本机位置，需要使用有授权的数据副本。无数据或 memory 时应报错，不自动换成别的标签。

| 数据/资产 | 本机路径或仓库相对位置 |
|---|---|
| LND root | `/mnt/iMVR/daiyun/Dataset/LND` |
| LND RGB / 原始 wrist GT | `{root}/{TRAIN,TEST}/image/*.png` / `pose/*.npy` |
| LND intrinsics | `{root}/{split}/config.yaml` |
| SAM instance / part mask | `sam3_segmentation` / `sam3_segmentaion_part`，后者拼写不要自动改 |
| 外部遮挡 visible mask | split 下 `mask_original` 或 `mask visible`，按 loader 选择 |
| Pre-refine TRAIN memory | `submodules/gaussian-mesh-splatting/Results2/surgripe_lnd_action_gt_full/TRAIN/memory_pool.json` |
| Refined TRAIN memory | `submodules/gaussian-mesh-splatting/Results2/surgripe_lnd_refine_memory_train/TRAIN/refine_memory_pool.json` |
| RARP RGB | `/mnt/nas/share/shuojue/data/{needlePuncture,needleGrasping,knotting}_videos` |
| RARP pose memory | `/mnt/nas/share/shuojue/data/{needlePuncture,needleGrasping,knotting}_results` |
| RARP masks/metadata | `/mnt/nas/haofeng/data/RARP50`；另有 `/mnt/iMVR/daiyun/shuojue-temp/data/RARP50_new_labels` |
| SurgPose | `/mnt/nas/share/shuojue/data/surgpose`；本地 partial copy 见 NOISY_LEARNING.md |
| Instrument Splatting CAD | `submodules/gaussian-mesh-splatting/instrument_mesh/transformed_{shaft,wrist,gripper_left,gripper_right}.obj` |
| 当前训练 surface asset | `submodules/RoboPEPP/assets/instrument_surface_samples_surfemb_x2.13mm_wg1over3_shafttop30mm/instrument_surface_points_all.npy` |

LND 原始 TRAIN 1147、TEST 373，TEST_occ 238 是单独的集合，不能静默合并。Augfix 明确排除 TRAIN `340,408,779,1125`，剩 1143；TEST 排除 `210`，剩 372。历史的 373-frame 或剔除其他失败帧的结果必须单独标记。

**RARP split 风险仍存在：**历史 split 代码依赖 `/mnt/nas/haofeng/data/RARP50_0910/test/images`，本次确认不存在。不要假定已经排除了 held-out videos；新实验需显式 manifest 并检查 train/test 交集。

LND 原始 pose 只提供 wrist SE(3)。Pre-refine memory 固定转换后的原始 wrist pose，另拟合 articulation；refined memory 允许 wrist 小幅移动，是另一套 pseudo label。LND TEST 没有可靠的完整 articulation GT，不应据此计算所有 joint/part 的 pose 准确率。

## 4. 几何与监督约定

1. 使用 Instrument Splatting CAD，不是原始 LND CAD 直接训练。统一 FK 在 `submodules/RoboPEPP/instrument_geometry.py`。
2. **Gripper part-local x < 2.13 mm（0.00213 m）固定归 wrist**，随 wrist 运动，但不随 gripper opening angle 转动。这个阈值不是 2.13 cm，也不是 shaft 的裁剪阈值。
3. OpenGL 对跨阈值 triangle 做切分，static 部分用 wrist transform 和 wrist label。不是只给采样点换颜色。
4. 当前 positive 从 crop 中可见有效像素采样；用 triangle rasterization 的 barycentric XYZ，输出 canonical XYZ、effective part、depth、valid。保留整套 mesh 之间的 z-buffer 遮挡，过滤背面监督，再与真实 visible/SAM mask 求交。
5. 每个有效像素有一个 rasterized surface coordinate。不再用旧的多个投影点竞争同一像素、0.8 mm depth tolerance 等来产生 positive。残留参数不代表这条旧 positive 路径仍在使用。
6. Negative 仍来自离散 surface asset。当前 wrist-only 是整个 effective wrist surface，包括静态 gripper 根部，不局限于当前可见区域；旧 `visible` negative 实验必须分开记录。
7. Asset 中 wrist/每侧 gripper 各16667点，shaft 80000点。shaft 偏重 distal top 30 mm、后段稀疏；Full 采样再排除 shaft normalized x < -0.5。不要删掉其遮挡几何。
8. `canon_scale=0.028702 m`。`key_noise=1e-3` 加在 normalized XYZ 上，约对应每轴0.0287 mm，而不是1 mm。它不是图像高斯噪声。
9. LND 原始 translation 单位 mm，loader 转到 repo 的 m；quaternion 使用 wxyz，articulation 内部按弧度。报告 translation 用 mm、rotation geodesic angle 用 degree，不是欧拉角分量 MAE。
10. 当前 crop 保留 camera-frame pose，用 `K_crop = M3 @ K_orig` 表达图像 affine；RGB、mask、positive、joint projection 必须使用同一 M。随机旋转可能让 K_crop 非上三角、含负项，下游不可直接只提取 fx/fy/cx/cy。不要在此基础上又旋转 pose 导致重复变换。
11. Joint-keypoint 实验用 FK joint/axis/tip keypoints，不是 surface samples。当前简化 ResNet 没有 joint/action/pose regression head；不能把早期 DINO 多 head 版本的输出能力写到它身上。
12. 保留 repo 的 symmetry canonicalization。比较时声明 raw 或 canonical 指标；不能把任意180度误差都自动当作等价正确姿态。

代码：[OpenGL renderer](submodules/RoboPEPP/instrument_opengl_renderer.py)、[几何](submodules/RoboPEPP/instrument_geometry.py)、[LND转换](submodules/RoboPEPP/datasets/surgripe_lnd_instrument.py)、[triangle tests](submodules/RoboPEPP/tests/test_surfemb_triangle_rasterizer.py)。

## 5. 模型与训练 Recipe

### 两条模型路线

- 早期 Full DINO + 多 head：correspondence 与 joint/pose/action 等联合训练，不是当前简化模型。
- 当前 ResNet：引用本仓库 `submodules/surfemb/surfemb/dep/unet.py` 的 ImageNet-pretrained ResNet18 U-Net。224x224 输出一个 binary mask logit 和12维 dense query embedding；SIREN 将3D canonical XYZ映射到12维 key。没有独立 part segmentation head。
- ResNet loss = binary mask BCE + InfoNCE；每个 positive query 对应一个 positive key，与本张样本的1024个 surface negatives比较，非跨整个 batch 的负样本池。
- Raw-dot logit = q dot k；cosine logit = normalized(q) dot normalized(k) / temperature。Cosine同时归一化正负key与query，分割loss不变。它不保证归一化前的norm永远不增长。

### Augfix 的确定设置

| 设置 | 实际值 |
|---|---|
| 数据/标签 | LND-only，pre-refine TRAIN memory |
| Crop | wrist zoom-in，224x224，scale=1.2，训练角度范围 +/-pi |
| Positive / negative | 1024 / 1024，wrist-only / full effective wrist surface |
| Batch | 每卡56，4卡，总batch224；validation每卡56 |
| Workers | 每rank4，persistent_workers=False |
| Optimizer | Adam；CNN lr=1e-4，SIREN lr=3e-5，weight_decay=0 |
| LR schedule | 2000步线性warmup，之后常数；不是cosine LR decay |
| 精度/裁剪 | AMP开启，grad_clip=1 |
| Validation / checkpoint | 此10k实验val每250步，ckpt每1000步，先全量validation |
| Validation对象 | LND TEST direct wrist GT；mask、positive、negative仅wrist |
| best / last | best按validation mask BCE + NCE；last是最后保存点，不是pose最优 |
| Resume | 恢复optimizer/scheduler，不重置iteration |

参数名 `val_score_mode=rarp_total` 在这个 wrapper 中对应的实际 primary stream 被替换成 **LND**，日志打印 `lnd_total`。不要仅根据参数名字误判用了 RARP。

### Augfix 到底改了什么

| 增强项 | Legacy | Augfix `original_p30` |
|---|---|---|
| Gaussian noise | 标准差随机10..50 | 方差随机10..50，即标准差约3.16..7.07；p=0.5 |
| ColorJitter | 仅hue | brightness/contrast/saturation=0.2，hue=0.1；p=0.5 |
| ISO noise | 旧实现，p=0.5 | 对应SurfEmb/Albumentations的实现，p=0.3 |
| CLAHE | clip固定4，p=0.5 | clip随机1..4，p=0.3 |
| Debayer / Unsharpen | 各p=0.5 | 各p=0.3 |
| Crop mask retention | 0.985 | 0.70 |
| Crop offset multiplier | 1.0 | 2.5 |

保留两个GaussianBlur阶段、集中于instrument/wrist/gripper的CoarseDropout（p=0.5）。因此Augfix不是完全逐项等同原版BOP训练：30%概率、集中式遮挡、mask保留规则是定制的。

70%指保留原可见mask像素比例，不是wrist占patch面积比例。原始可见比例不足70%时，crop尽量包含全部可见wrist，并放宽render-IoU拒绝及crop后像素门槛；仍有原始最小像素/空监督检查，不等于所有坏样本无条件训练。该runner禁止换成另一个sample兜底，同帧augmentation可重试。

### Checkpoint 血缘

以下路径都在 `submodules/RoboPEPP/logs/` 下，本次只记录名字，不上传权重：

| ID | Run目录 | Checkpoint |
|---|---|---|
| A6 | `surfemb_debug_current_lnd_augfix_rawdot_resume4k_to6k_b56x4_20260811` | `checkpoints/last.pt`，6k |
| A10 | `surfemb_debug_current_lnd_augfix_rawdot_resume6k_to10k_b56x4_20260811` | `last.pt`，10k；`best_val_total.pt`，6250 |
| C10 | `surfemb_debug_current_lnd_augfix_cosine_t01_resume6k_to10k_b56x4_20260811` | `last.pt`，10k；`best_val_total.pt`，6750 |

**A10和C10都从A6继续训练4k步。C10不是从头训练10k cosine。** A10保持raw-dot、temperature=1；C10切换cosine、temperature=0.1，并在测试时保持一致。`cosine`指similarity，不是学习率schedule。

完整参数：[augfix_checkpoint_recipes.json](docs/experiment_memory/augfix_checkpoint_recipes.json)。其中iteration来自2026-09-08读取的checkpoint快照，路径叫last.pt并不保证今后还指向同一权重。

## 6. 训练与测试入口

| 用途 | 文件 |
|---|---|
| Augfix raw-dot | `submodules/RoboPEPP/train_surfemb_resnet_wrist_only_lnd_prerefine_augfix.py` |
| Augfix cosine | `submodules/RoboPEPP/train_surfemb_resnet_wrist_only_lnd_prerefine_augfix_cosine.py` |
| Wrist runner | `submodules/RoboPEPP/train_surfemb_resnet_wrist_only_rarp_lnd.py` |
| ResNet基础runner/loss | `submodules/RoboPEPP/train_surfemb_resnet_crop_rarp_lnd_refinemem.py` |
| Wrist dataset / augmentation | `submodules/RoboPEPP/datasets/surfemb_wrist_only_crop.py` / `surfemb_augment.py` |
| LND wrist pose评测 | `submodules/RoboPEPP/eval_surfemb_wrist_lnd.py` |
| RARP articulated评测 | `submodules/RoboPEPP/eval_surfemb_articulated_rarp.py` |
| Pose/correspondence实现 | `submodules/RoboPEPP/surfemb_articulated_pose.py` |

下面是复现 **A6 -> C10** 的命令模板，仅在确认四张空卡、数据与A6 checkpoint正确后手动执行。本次未启动训练。用新名字，不覆盖历史run。

```bash
export REPO=/mnt/iMVR/daiyun/shuojue-temp/code/multi-hmr
export PYTHON=/mnt/iMVR/daiyun/anaconda3/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 示例，必须先检查卡是否空闲
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1
cd "$REPO"
export A6="$REPO/submodules/RoboPEPP/logs/surfemb_debug_current_lnd_augfix_rawdot_resume4k_to6k_b56x4_20260811/checkpoints/last.pt"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  submodules/RoboPEPP/train_surfemb_resnet_wrist_only_lnd_prerefine_augfix_cosine.py \
  --name augfix_cosine_reproduction_new --resume "$A6" \
  --batch_size 56 --val_batch_size 56 --num_workers 4 \
  --max_iter 10000 --val_freq 250 --ckpt_freq 1000 \
  --validate_before_train 1 --surfemb_n_pos 1024 --surfemb_n_neg 1024 \
  --wrist_negative_source full_surface --surfemb_similarity cosine \
  --surfemb_temperature 0.1 --lr_cnn 1e-4 --lr_surfemb_mlp 3e-5 \
  --warmup_steps 2000 --amp 1 --grad_clip 1
```

可以在新的tmux会话中执行。Raw-dot对照使用raw入口、similarity=raw_dot、temperature=1，并从同一A6起点继续。独立从头训练属于新实验，不应套用本文10k结果。

### 固定的 LND 推理协议

- SAM wrist crop，scale=1.2，测试angle=0、offset=0；matching用GT/SAM wrist ROI。
- 50,000个候选surface keys，不是50,000个PnP对应点。每pixel搜索最相似key，按confidence选最多512个correspondence。
- EPNP RANSAC 2000次，reprojection threshold=3，confidence=0.999，再LM；保留既有AvgPool/down_sample_scale=3及配套坐标映射。
- min correspondences=12，min inliers=4，min inlier fraction=0。无BFGS，无rotation ensemble；失败仍须明确记录。
- `--model`的别名必须包含 **resnet**，当前解析器通过名字选择ResNet/DINO；避免旧辅助shell里只叫raw10k/cosine10k的命名。

```bash
export CKPT="$REPO/submodules/RoboPEPP/logs/surfemb_debug_current_lnd_augfix_cosine_t01_resume6k_to10k_b56x4_20260811/checkpoints/last.pt"
"$PYTHON" submodules/RoboPEPP/eval_surfemb_wrist_lnd.py \
  --model "resnet_cosine10k_last=$CKPT" --devices cuda:0 \
  --output_dir "$REPO/submodules/RoboPEPP/logs/new_eval_cosine10k" \
  --crop_mask_source wrist --surfemb_crop_scale 1.2 --pose_estimator topk_ransac \
  --wrist_roi_source gt_wrist --predicted_wrist_mask_mode binary_head \
  --surface_keys_per_part 50000 --mask_keys_per_part 512 --down_sample_scale 3 \
  --surface_seed 2026 --pose_seed 20260802 \
  --correspondence_similarity cosine --correspondence_temperature 0.1 \
  --topk_max_correspondences 512 --topk_min_correspondences 12 \
  --topk_ransac_iterations 2000 --topk_ransac_reprojection_error 3.0 \
  --topk_ransac_confidence 0.999 --topk_min_inliers 4 --topk_min_inlier_fraction 0 \
  --topk_bfgs_refine 0 --rotation_ensemble 0 --exclude_frame_ids 210 --amp 1 --fail_fast 1
```

还需准备该脚本引用的historical HCCE/RoboPEPP CSV，或显式传正确路径；不能默认为跨机器已有。此处为参数复现模板，尚未在全新环境端到端运行。

## 7. 已有实验结果

**所有 T/R 是逐帧误差的平均值（mm/degree），不是RMSE。** 有效帧数仅表示pose被接受/返回，不是误差低于阈值的准确率。不同有效帧、crop、matching ROI、label source、候选keys、temperature不可静默混比。

### Full patch 与 wrist zoom-in

均RARP+LND refined训练，LND TEST，GT wrist matching ROI，Top-K/LM。

| 模型 | Iter | T mean | R mean | 有效帧 |
|---|---:|---:|---:|---:|
| 旧Full DINO，point-projection positive | 22k | 4.644 | 12.210 | 372/372 |
| 旧Full ResNet，point-projection positive | 26k | 3.628 | 13.344 | 372/372 |
| 新Full ResNet，triangle positive | 21k | 4.709 | 11.307 | 372/372 |
| 新Full ResNet | 36k | 3.825 | 12.607 | 372/372 |
| 新Full ResNet | 43k | 4.042 | 11.654 | 372/372 |
| Wrist zoom-in ResNet | 22k | 5.144 | 6.740 | 372/372 |
| Wrist zoom-in ResNet | 38k | 4.509 | 6.405 | 372/372 |
| Wrist zoom-in ResNet | 43k | 4.095 | 6.073 | 358/372 |
| Wrist zoom-in ResNet | 46k | 4.175 | 5.838 | 363/372 |

Full为1024/1024且wrist份额614，另外有shaft/gripper negative；wrist-only为614/614 visible-wrist negative。两者都是4卡、每卡56，但不是严格的仅crop单变量实验。不同checkpoint也非同训练步对照。

### LND-only wrist

均GT wrist matching ROI、50k keys、Top-K/LM、372/372帧。

| 训练设置 | Iter | T mean | R mean |
|---|---:|---:|---:|
| Refined pose，614/614 visible negative | 14k | 4.023 | 7.148 |
| Refined pose，1024/1024 visible negative | 13k | 4.220 | 7.890 |
| Pre-refine pose，1024/1024 visible negative | 14k | 2.897 | 7.639 |
| Pre-refine，1024/1024 full-surface negative | 8k | 2.456 | 7.968 |
| 同上 | 28k | 2.321 | 6.208 |
| 同上 | 54k | 2.611 | 6.873 |
| Augfix raw-dot | 6k | 2.054 | 6.547 |
| Augfix raw-dot best(total) | 6250 | 2.322 | 7.696 |
| Augfix raw-dot last | 10k | 2.452 | 5.966 |
| Augfix cosine best(total) | 6750 | 2.168 | 6.867 |
| Augfix cosine last | 10k | 2.370 | 5.553 |

A6中位数1.473 mm / 4.992 degree；C10中位数1.981 mm / 4.008 degree。不能拿平均值与中位数互相比。Full-surface-negative 28k到54k，translation与rotation均变差，说明更久不保证更好。

### 推理消融：同一个 mixed-data wrist 38k

Predicted matching ROI；除标注外均372/372帧。

| 推理 | T mean | R mean | 有效帧 |
|---|---:|---:|---:|
| 4k keys，Top-K/LM | 4.357 | 6.399 | 372 |
| 50k keys，Top-K/LM，accept-all | 3.733 | 6.751 | 372 |
| 50k，额外projected ROI inlier条件 | 3.713 | 6.761 | 372 |
| 50k，spatial Top-K | 3.872 | 5.712 | 358 |
| Coarse4096 + fine20 neighbors | 3.979 | 6.069 | 372 |
| Top-K + LM + BFGS | 4.183 | 6.129 | 372 |
| 原版概率式SurfEmb，50k keys | 7.998 | 8.360 | 372 |
| 原版概率式 + mm-scaled BFGS | 4.606 | 6.022 | 372 |

取消AvgPool的wrist22k消融：baseline 5.206/6.627；no-pool 3px 5.407/6.610，9px 5.521/6.848，1px 5.293/6.387，均372帧。没有稳定收益，保留AvgPool。

### RARP 历史基线

同之前RoboPEPP的2301-instance testing manifest，各方法有效输出子集不同。

| 方法 | 有效帧 | T mean | R mean | Joint MAE |
|---|---:|---:|---:|---:|
| Full DINO 22k SurfEmb + Top-K | 2289/2301 | 2.780 | 9.904 | 12.530 |
| Full ResNet 26k SurfEmb + Top-K | 2295/2301 | 2.760 | 9.546 | 12.569 |
| RoboPEPP direct | 2301/2301 | 4.435 | 9.681 | 7.143 |
| RoboPEPP keypoint PnP | 2301/2301 | 8.560 | 21.726 | 7.143 |
| HCCE direct | 2301/2301 | 3.858 | 8.881 | 6.705 |
| HCCE keypoint PnP | 2301/2301 | 8.459 | 16.872 | 6.705 |
| HCCE fit | 2286/2301 | 13.210 | 16.056 | 10.160 |

共同有效2274 instances上，DINO T/R=2.754/9.712，ResNet=2.735/9.500。SurfEmb的translation收益不等于articulation收益。没有找到新mixed wrist zoom-in模型完整RARP pose测试，不能用Full模型仅fit wrist的消融冒充。

### Refine 标签改动本身

1147个LND TRAIN frame的pre/refine差异，不是预测误差：

| 改变量 | 平均 | 中位数 | p95 | 最大 |
|---|---:|---:|---:|---:|
| Wrist translation (mm) | 2.322 | 2.336 | 3.738 | 5.000 |
| Wrist rotation (degree) | 1.333 | 1.253 | 2.501 | 5.000 |
| Shaft alpha (degree) | 3.993 | 2.614 | 12.000 | 12.000 |

此版本theta_l/theta_r改变量均为0。达到上界意味着需检查约束，不代表refined pose更接近真实GT。可视化索引见下节。

## 8. 已知发现与未证实解释

- **已测：** zoom-in在多个checkpoint上改善rotation，但早期translation不随之改善。仍需固定label、negative pool、训练步数与seed才能做纯resolution因果结论。
- **已测：** pre-refine 14k比refined 13k的translation均值更低，约31%。这提示标签bias问题，但不是refine错误的充分证明，且这里测的是TEST，不是TRAIN。
- **已修的监督歧义：** triangle rasterization给每个有效pixel一个确定XYZ；这修复了旧point projection歧义，但不保证所有pose指标都优于旧模型。
- **解释而非已证实主因：** translation尤其深度受物体尺度、mesh/label边界、对应点分布、PnP条件数影响。大量局部inliers或较低reprojection error不能保证正确深度。
- **Loss不等于pose：** raw-dot从6k到10k，train NCE 0.394降到0.253，val NCE 2.033升到2.781，但rotation仍改善。Cosine10k train/val NCE=0.485/1.417；不同temperature/logit定义的NCE不能直接横比高低。
- **不能宣称RANSAC普遍优于原版：**一些原版SurfEmb对照使用detector crop、predicted ROI、373帧、ensemble/BFGS，与本地GT ROI/372帧并非唯一solver差异。
- **旧persistent-workers问题：**主进程set_epoch不会自动修改常驻worker的数据副本，可能使bbox-jitter schedule停在epoch0。当前ResNet loader为persistent_workers=False；修改worker设置后仍需确认augmentation实际生效。
- **可重复性风险：**LND TEST被用作validation选best及多轮debug，不是严格未触碰的最终hold-out。公开报告应声明，noisy-learning需另外冻结验证/测试策略。
- **记录风险：**历史结果引用的last/best路径可能已覆盖。2026-09-08 inventory发现10处引用iteration与现存checkpoint不同；应固定iter或权重hash。
- **不要误报最新结果：**原Full/wrist现存last在该inventory为60k/50k，但最高已找到LND pose评测仅43k/46k。本文不是60k/50k的新测试。
- **机器I/O风险：**曾出现/mnt RAID5 consistency check极慢、进程阻塞。不能直接归咎RAM不足；先看/proc/mdstat、I/O等待和进程状态，勿反复启动更多数据加载任务。

## 9. 可视化与测试索引

以下是本地生成目录，只记录路径，不随本memory上传图片：

- `submodules/RoboPEPP/logs/surfemb_triangle_coordinate_one_to_one_vis/contact_sheet.jpg`：triangle坐标一对一检查。
- `submodules/RoboPEPP/logs/surfemb_wrist_opengl_zbuffer_occlusion_rarp_lnd_current_20260728/contact_sheet.jpg`：wrist可见/遮挡点。
- `submodules/RoboPEPP/logs/rarp_unified_fk_joint_keypoint_diag/opengl_contact_sheet.jpg`：统一FK、joint与mesh。
- `submodules/RoboPEPP/logs/lnd_train_gt_vs_refined_large_differences/large_translation_difference_trimesh/contact_sheet_trimesh.jpg`：refine大改动case。
- `submodules/RoboPEPP/logs/surfemb_training_progress_lnd_20260805/wrist_last38k_source_gt_wristmesh_vis/`：source/GT对应与完整wrist mesh。
- `logs/lnd_refine_delta_summary/`：1147帧refine差异；小型聚合统计已随memory保存。

已有测试脚本：`submodules/RoboPEPP/tests/test_surfemb_triangle_rasterizer.py`、`test_surfemb_coarse_fine_pose.py`。Rasterizer测试需要CAD、surface asset、EGL上下文；本次未运行这些GPU相关测试，不能用`--help`通过代替。

每次新训练前至少检查：数据路径/样本数、两种crop的RGB与GT投影、2.13mm静态区域、背面和跨mesh遮挡、K_crop、单位、symmetry、negative source、全量validation。新worker配置要先smoke；不允许坏样本被静默替换。

## 10. 记录文件与 GitHub 上传

- [LND全部85条评测记录](docs/experiment_memory/lnd_all_evaluations.csv)：含不同推理口径和重复复测，不是85次独立训练。
- [RARP全部67条记录](docs/experiment_memory/rarp_all_evaluations.csv)：含smoke与分组统计，需看scope/group，不能全当full-test。
- [Train/validation loss末端记录](docs/experiment_memory/training_loss_endpoints.csv)。
- [原始汇总SHA256与CSV核对记录](docs/experiment_memory/source_audit.json)。
- [Refine标签改动统计](docs/experiment_memory/lnd_refine_delta_summary.json)。

85条LND结果在2026-09-08与逐帧CSV核对，mean/median/RMSE一致。原始评测JSON、CSV和图片仍在ignore目录，不包含在这些轻量索引中；CSV中的source/checkpoint路径是相对本仓库的本地资产引用。

从已有inventory重新导出记录（不启动推理）：

```bash
"$PYTHON" scripts/export_surfemb_memory_records.py
```

脚本只导出允许的CSV、选定checkpoint参数、当前包版本和GPU硬件信息，不导出环境变量、token、完整pip配置、数据或权重。`environment_snapshot.json`随导出时间更新；结果本身仍来自2026-09-08 inventory，不会变成最新评测。

后续上传继续遵循以下检查：

1. 确认仍在`noisy-learning`；将本文与`docs/experiment_memory/`、相关汇总脚本显式加入commit，不使用不审查的全量add。
2. 子仓库代码先单独commit/push；主仓库commit不会自动上传子仓库工作区。
3. 保持`.gitmodules`与remote可达性，确保被引用的子仓库commit已经发布。
4. 检查staged diff与文件大小；不得包含Results2、数据集、模型权重、图片视频或外部数据授权内容。
5. 再执行主仓库commit/push，并检查远端分支SHA与本地HEAD一致。

## 11. Noisy-learning 下一步

详细计划在 [NOISY_LEARNING.md](NOISY_LEARNING.md)。优先冻结RARP video/frame split与版本化噪声标签manifest，再做RARP noisy round-one与LND anchor对照。LND原始wrist GT与refined articulation pseudo GT必须分开标记。SurgPose用于外部point tracking，不假定有兼容的6D pose GT。

仍待完成：验证RARP无train/test overlap；固定label版本和样本过滤策略；固定模型/crop/negative数量/batch/optimizer/schedule/seed；建立独立于已反复调试TEST的最终评估。本文记录的是现状，不把这些待办写成已完成。

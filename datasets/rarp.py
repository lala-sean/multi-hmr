# RARP Dataset for Surgical Instrument Pose Estimation
# Based on the DREAM dataset structure

import copy
import os
import glob
import random
from typing import OrderedDict, List, Dict, Tuple, Optional
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from logging import getLogger
from augmentations import PadToSquare, PillowSharpness, PillowContrast, PillowBrightness, PillowColor, apply_color_jitter, apply_occlusion, apply_rgb_aug, occlusion_aug, ResizeLongerSide
from image_proc import create_belief_map
import sys
sys.path.append('/mnt/iMVR/shuojue/code/gaussian-mesh-splatting/')
from instrument_gaussian_wrapper import instrument_gaussian_wrapper
import torch
import torchvision
import sys
sys.path.append('/mnt/iMVR/shuojue/code/gaussian-mesh-splatting/')
# from pose_tracking_tipNet_rarp import Pose

_GLOBAL_SEED = 0
logger = getLogger()

class Part(object):
    def __init__(self, name, world2part=None):
        # self.mesh = mesh.copy()
        self.render_params = {}
        if isinstance(world2part, torch.Tensor):
            world2part = world2part.numpy()
        self.transformation = world2part if world2part is not None else np.eye(4)
        # self.mesh.apply_transform(world2part)
        self.name = name
        world2part = torch.tensor(world2part)
        trimesh2blender = torch.eye(4)
        trimesh2blender[:3, :3] = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        trimesh2blender = trimesh2blender.float()
        world2part =  world2part @torch.linalg.inv(trimesh2blender)
        if name == "shaft":
            #-0.009026, -1e-6, 0.004178
            self.keypoints = torch.tensor([[-0.010026, 0.0,0.0]]) @ world2part[:3,:3].t() + world2part[:3,3][None]
        elif name == "wrist":
            # the first keypoint is the wrist joint, the second keypoint is the wrist origin (i.e., world origin in canonical state)
            self.keypoints = torch.tensor([[0.009, 0.0, 0.0], [0.0, 0.0, 0.0],]) @ world2part[:3,:3].t() + world2part[:3,3][None]
        elif "l_gripper" == name:
            self.keypoints = torch.tensor([[0.018439, 0.0, 0.0]]) @ world2part[:3,:3].t() + world2part[:3,3][None]
        elif "r_gripper" == name:
            self.keypoints = torch.tensor([[0.018439, 0.0, 0.0]]) @ world2part[:3,:3].t() + world2part[:3,3][None]

    def apply_transformation_params(self, transformation, gs_grad = False):
        """
        Apply the transformation to the Gaussian models without changing the member variables.
        gaussian_grad = False while the transformation is differentiable.
        """
        keypoints =  self.keypoints.clone() @ transformation[:3,:3].t() + transformation[:3,3][None]

        # render_parameters = self.gaussian_model.apply_transform_params(transformation, gs_grad)

        self.render_params['keypoints'] = keypoints
        return self.render_params

class Instrument(object):
    def __init__(self,shaft_mesh=None, wrist_mesh=None, gripper_mesh_left=None, gripper_mesh_right=None):
        self._preComputeTransform()

        self.shaft = Part('shaft', self.bias2shaft)
        self.wrist = Part('wrist', self.bias2wrist)
        self.l_gripper = Part('l_gripper', self.bias2l_gripper)
        self.r_gripper = Part('r_gripper', self.bias2r_gripper)
        # self.new_mesh_dict = {}

        self.part_dict = {'shaft': self.shaft, 'wrist': self.wrist, 'l_gripper': self.l_gripper, 'r_gripper': self.r_gripper}

    def _preComputeTransform(self):
        # convert TriMesh coordinate system to Blender coordinate system
        # basis: trimesh coordinate system for manipulating mesh model
        # world: Blender coordinate system where we define keypoints and Gaussian Splatting primitives
        bias2world = torch.eye(4)
        bias2world[:3, :3] = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        self.bias2world = bias2world
        flip_wrist = torch.eye(4)
        flip_wrist[:3, :3] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).T
        shaft2world = torch.eye(4)
        wrist2world = torch.eye(4)
        l_gripper2world = torch.eye(4)
        r_gripper2world = torch.eye(4)

        shaft2world[:3, 3] = torch.tensor([-0.2159, 0, 0])  # unit: m
        l_gripper2world[:3, 3] = torch.tensor([0.009, 0, 0])  # unit: m
        r_gripper2world[:3, 3] = torch.tensor([0.009, 0, 0])  # unit: m

        self.shaft2world = shaft2world
        self.wrist2world = wrist2world
        self.l_gripper2world = l_gripper2world
        self.r_gripper2world = r_gripper2world

        self.bias2shaft = torch.inverse(shaft2world) @ bias2world
        self.bias2l_gripper = torch.inverse(l_gripper2world) @ bias2world
        self.bias2r_gripper = torch.inverse(r_gripper2world) @ bias2world
        self.bias2wrist = torch.inverse(wrist2world) @ bias2world

    def calculate_lateral_rotation(self, rot_angle, rot_wrist2camera=torch.eye(3),
                                   trans_wrist2camera=torch.zeros(3), alpha=torch.tensor(0)):

       # wrist to camera transformation (initialized)
        wrist2camera_transformation = torch.eye(4)
        wrist2camera_transformation[:3, :3] = rot_wrist2camera
        wrist2camera_transformation[:3, 3] = trans_wrist2camera

        # wrist transformation
        rot_wrist = self._rodrigues_rotation_matrix(torch.tensor([0, 1, 0]), alpha)

        wrist2shaft_transformation = torch.eye(4)
        wrist2shaft_transformation[:3, :3] = rot_wrist
        wrist2shaft_transformation[:3, 3] = torch.tensor([0.2159, 0, 0])
        shaft2camera_transformation = wrist2camera_transformation @ torch.inverse(wrist2shaft_transformation)

        # generate shaft lateral rotation
        lateral_shaft_transformation = torch.eye(4)
        lateral_shaft_transformation[:3, :3] = self._rodrigues_rotation_matrix(torch.tensor([1, 0, 0]), rot_angle.unsqueeze(0))

        wrist2camera_transformation = (shaft2camera_transformation @ lateral_shaft_transformation) @ wrist2shaft_transformation

        return wrist2camera_transformation[:3,:3]

    def _apply_transformation(self, part_name, transformation):
        # changing the mesh: now only for debug
        # if self.with_mesh:
        #     self.new_mesh_dict[part_name] = self.part_dict[part_name].apply_transformation_mesh(transformation)
        transformation = transformation.detach()
        render_params = self.part_dict[part_name].apply_transformation_params(transformation)
        return render_params

    def _rodrigues_rotation_matrix(self, axis, theta):
        """
        Calculate the rotation matrix using Rodrigues' rotation formula.

        Parameters:
        axis (torch tensor): The axis of rotation (must be a unit vector).
        theta (float): The angle of rotation in radians.

        Returns:
        torch tensor: The rotation matrix.
        """
        axis = axis / torch.norm(axis.float())
        a = torch.cos(theta / 2.0)
        b, c, d = -axis * torch.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        row1 = torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)])
        row2 = torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)])
        row3 = torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc])

        return torch.stack([row1, row2, row3])[...,0]
    def forward_kinematics(self, rot_wrist2camera=torch.eye(3), trans_wrist2camera=torch.zeros(3),
                            alpha=torch.tensor(0), theta_l=torch.tensor(0), theta_r=torch.tensor(0)):
       # wrist to camera transformation (initialized)
        wrist2camera_transformation = torch.eye(4)
        wrist2camera_transformation[:3, :3] = rot_wrist2camera
        wrist2camera_transformation[:3, 3] = trans_wrist2camera

        # wrist transformation
        rot_wrist = self._rodrigues_rotation_matrix(torch.tensor([0, 1, 0]), alpha)
        self.rot_wrist = rot_wrist
        wrist2shaft_transformation = torch.eye(4)
        wrist2shaft_transformation[:3, :3] = rot_wrist
        wrist2shaft_transformation[:3, 3] = torch.tensor([0.2159, 0, 0])
        shaft2camera_transformation = wrist2camera_transformation @ torch.inverse(wrist2shaft_transformation)

        # shaft transformation
        shaft_transformation = shaft2camera_transformation
        # wrist transformation
        wrist_transformation = wrist2shaft_transformation

        # left gripper transformation
        l_gripper_transformation = torch.eye(4)
        l_gripper_transformation[:3, :3] = self._rodrigues_rotation_matrix(torch.tensor([0, 0, 1]), theta_l)
        l_gripper_transformation[:3, 3] = torch.tensor([0.009, 0, 0])
        # right gripper transformation
        r_gripper_transformation = torch.eye(4)
        r_gripper_transformation[:3, :3] = self._rodrigues_rotation_matrix(torch.tensor([0, 0, 1]), -theta_r)
        r_gripper_transformation[:3, 3] = torch.tensor([0.009, 0, 0])
        # apply the transformation
        self._apply_transformation('shaft', shaft_transformation)
        self.shaft_transformation = shaft_transformation
        self.wrist_transform = shaft_transformation @ wrist_transformation
        self._apply_transformation('wrist', self.wrist_transform)
        self.l_gripper_transformation = self.wrist_transform @ l_gripper_transformation
        self._apply_transformation('l_gripper', self.l_gripper_transformation)
        self.r_gripper_transformation = self.wrist_transform @ r_gripper_transformation
        self._apply_transformation('r_gripper', self.r_gripper_transformation)

def matrix_to_quaternion(m):
    """

    Convert a rotation matrix to a quaternion.

    The quaternion will be in the form [w, x, y, z] where w is the real part.
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2.0 * torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return torch.stack([w, x, y, z])

def build_rotation(r):
    """

    Build rotation matrix from quaternion
    """
    if r.dim() == 1:
        r = r.unsqueeze(0)
    q = F.normalize(r, p=2, dim=1)
    R = torch.zeros((q.size(0), 3, 3))

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

class Pose(object):
    """
    Surgical robot pose class
    """
    def activate_alpha(self, alpha):

        # Scale alpha to range from -90 to 90 degrees
        return torch.pi * torch.sigmoid(alpha) - torch.pi / 2

    def activate_theta(self, theta):

        # Scale theta to range from -80 to 80 degrees
        return (160.0 / 180.0) * torch.pi * torch.sigmoid(theta) - (80.0 / 180.0) * torch.pi

    def __init__(self, rot, trans, alpha, theta_l, theta_r, init_optimizer=True):
        quat = matrix_to_quaternion(rot)
        quat = F.normalize(quat, p=2, dim=0)
        if init_optimizer:
            self.rot = torch.nn.Parameter(quat, requires_grad=True) # rotation of wrist
            self.trans = torch.nn.Parameter(trans, requires_grad=True) #  translation of wrist
            self.alpha = torch.nn.Parameter(alpha, requires_grad=True) # joint angle between wrist and shaft
            self.theta_l = torch.nn.Parameter(theta_l, requires_grad=True) # joint angle between wrist and left gripper
            self.theta_r = torch.nn.Parameter(theta_r, requires_grad=True) #  joint angle between wrist and right gripper
            self.optimizer = torch.optim.Adam([
                {'params': [self.rot], 'lr': 0.001, 'name': 'rot'},
                {'params': [self.trans], 'lr': 0.0001, 'name': 'trans'},
                {'params': [self.alpha], 'lr': 0.02, 'name': 'alpha'},
                {'params': [self.theta_l], 'lr': 0.1, 'name': 'theta_l'},
                {'params': [self.theta_r], 'lr': 0.1, 'name': 'theta_r'}
            ], lr=0.0, eps=1e-15)
        else:
            self.rot = quat.detach().clone()
            self.trans = trans.detach().clone()
            self.alpha = alpha.detach().clone()
            self.theta_l = theta_l.detach().clone()
            self.theta_r = theta_r.detach().clone()

    def get_pose(self):
        rotation_matrix = build_rotation(self.rot)
        return rotation_matrix[0], self.trans, self.alpha, self.theta_l, self.theta_r

    def step_optimizer(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)


def make_rarp(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    video_root_path=None,
    pose_root_path=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    crop_size=224
):
    dataset = RARP(
        video_root=video_root_path,
        pose_root=pose_root_path,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False,
        crop_size=crop_size
    )
    logger.info('RARP dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False
    )
    logger.info('RARP data loader created')

    return dataset, data_loader, dist_sampler


class RARP(torch.utils.data.Dataset):
    """
    RARP Dataset for surgical instrument pose estimation.

    Dataset structure:
    - video_root: /data/XXX_videos/
        - SARRARP52022_XXX_{id}_video{videoid}/
            - frames/  (00001.png, 00002.png, ...)
            - instance_1/
                - masks_overall/ (00000.png, ...)
                - masks_wrist/
                - masks_shaft/
                - masks_gripper/
            - instance_2/
                - (same structure as instance_1)

    - pose_root: /data/XXX_Results/
        - SARRARP52022_XXX_{id}_video{videoid}_instance{1 or 2}/
            - memory_pool.pth
                Format: {frame_id: {'dice': [wrist_dice, shaft_dice, gripper_dice], 'pose_info': Pose}}
    """

    def __init__(
        self,
        video_root: str,
        pose_root: str,
        transform=None,
        train: bool = True,
        job_id=None,
        local_rank=None,
        copy_data: bool = False,
        index_targets: bool = False,
        crop_size: int = 224,
        min_dice_threshold = [0.8,0.7, 0.5],  # Optional: filter samples by dice score
    ):
        """
        Initialize RARP dataset.

        Args:
            video_root: Path to video data folder (e.g., /data/XXX_videos)
            pose_root: Path to pose annotation folder (e.g., /data/XXX_Results)
            transform: Optional transform to apply
            train: Whether this is training data
            crop_size: Size to crop/resize images to
            min_dice_threshold: Minimum dice score to include a sample
        """
        self.video_root = video_root
        self.pose_root = pose_root
        self.transform = transform
        self.train = train
        self.crop_size = crop_size
        self.min_dice_threshold = min_dice_threshold
        self.rarp_fc = 587.54401824
        self.keypoint_names = ['shaft_base', 'wrist_origin', 'wrist_joint', 'gripper_left', 'gripper_right']
        # initilalize the instrument Gaussian

        # wrapper = instrument_gaussian_wrapper()
        # self.instrument = wrapper.get_instrument()
        self.instrument = Instrument()
        # Augmentation settings (same as DREAM)
        self.color_jitter = True
        self.rgb_augmentation = True
        self.occlusion_augmentation = True
        self.resize_transform = ResizeLongerSide(crop_size)
        self.pad_transform = PadToSquare(crop_size)
        self.occlu_p = 0.5

        # TODO: Define joint names and link names for surgical instrument
        # These should correspond to the instrument's kinematic chain
        self.joint_names = self._get_joint_names()
        self.link_names = self._get_link_names()

        # Cache for loaded memory pools
        self._memory_pool_cache: Dict[str, Dict] = {}

        # Build sample list: each sample is (video_folder, instance_id, frame_id)
        self.samples = self._build_sample_list()



        # Load camera intrinsics (or set default values)
        self.intrinsics = self._get_default_intrinsics()

        self.epoch = 0

        logger.info(f'Initialized RARP dataset with {len(self.samples)} samples')

    def _get_joint_names(self) -> List[str]:
        """
        TODO: Define joint names for the surgical instrument.
        These should match the joint names in your Pose class.

        Returns:
            List of joint names
        """
        # TODO: Replace with actual joint names
        # Example for a surgical instrument with wrist, shaft, and gripper joints
        return [
            'instrument_joint1',  # e.g., shaft rotation
            'instrument_joint2',  # e.g., wrist pitch
            'instrument_joint3',  # e.g., wrist yaw
            'instrument_joint4',  # e.g., gripper
        ]

    def _get_link_names(self) -> List[str]:
        """
        TODO: Define link names for the surgical instrument.
        These should correspond to keypoint locations.

        Returns:
            List of link names
        """
        # TODO: Replace with actual link names
        return [
            'shaft_base',
            'shaft_tip',
            'wrist',
            'gripper_left',
            'gripper_right',
        ]

    def _get_default_intrinsics(self) -> np.ndarray:
        """
        TODO: Set camera intrinsics for the RARP dataset.

        Returns:
            3x3 camera intrinsic matrix
        """
        # TODO: Replace with actual camera intrinsics from RARP dataset
        K = np.array([
                        [self.rarp_fc, 0, 320],
                        [0, self.rarp_fc, 352/2.],
                        [0, 0, 1]
                    ], dtype=np.float32)

        return K

    def _build_sample_list(self) -> List[Tuple[str, int, int]]:
        """
        Build list of valid samples based on available pose annotations.
        Only includes samples that have both video data and pose annotations.

        Returns:
            List of tuples: (video_folder_name, instance_id, frame_id)
        """
        samples = []

        # Find all pose annotation folders
        pose_folders = glob.glob(os.path.join(self.pose_root, 'SARRARP502022_*_instance*'))

        for pose_folder in pose_folders:
            folder_name = os.path.basename(pose_folder)

            # Parse folder name to get video info and instance id
            # Format: SARRARP52022_XXX_{id}_video{videoid}_instance{instance_id}
            try:
                # Extract instance_id from the folder name
                instance_part = folder_name.split('_instance')[-1]
                instance_id = int(instance_part)

                # Get the video folder name (without _instance{id})
                video_folder_name = folder_name.rsplit('_instance', 1)[0]

            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse folder name: {folder_name}, error: {e}")
                continue

            # Check if corresponding video folder exists

            video_folder_path = os.path.join(self.video_root, video_folder_name)

            frame_folder_name = 'frames_v2'
            if not os.path.exists(video_folder_path):
                logger.warning(f"Video folder not found: {video_folder_path}")
                continue
            if not os.path.exists(os.path.join(video_folder_path, frame_folder_name)):
                frame_folder_name = 'frames'
                if not os.path.exists(os.path.join(video_folder_path, frame_folder_name)):
                    logger.warning(f"Frame folder not found in video folder: {video_folder_path}")
                    continue
            if not os.path.exists(os.path.join(video_folder_path, f'instance{instance_id}')):
                logger.warning(f"Instance folder not found: {os.path.join(video_folder_path, f'instance{instance_id}')}")
                continue
            mask_folder_name = 'refined_masks_v2'
            if not os.path.exists(os.path.join(video_folder_path, f'instance{instance_id}', mask_folder_name+'_overall')):
                mask_folder_name = 'refined_masks'
                if not os.path.exists(os.path.join(video_folder_path, f'instance{instance_id}', mask_folder_name+'_overall')):
                    mask_folder_name = 'masks'
                    if not os.path.exists(os.path.join(video_folder_path, f'instance{instance_id}', mask_folder_name+'_overall')):
                        logger.warning(f"Mask folder not found: {os.path.join(video_folder_path, f'instance{instance_id}', mask_folder_name+'_overall')}")
                        continue
            # Load memory pool to get valid frame ids
            memory_pool_path = os.path.join(pose_folder, 'memory_pool.pth')
            if not os.path.exists(memory_pool_path):
                logger.warning(f"Memory pool not found: {memory_pool_path}")
                continue

            try:
                memory_pool = torch.load(memory_pool_path, map_location='cpu')
                self._memory_pool_cache[pose_folder] = memory_pool

                # Add each frame as a sample
                err_frames_num = 0
                full_case_usable = True
                valid_frame_ids_in_one_case = []
                for frame_id in memory_pool.keys():
                    # Optional: filter by dice score
                    if self.min_dice_threshold is not None:
                        dice_scores = memory_pool[frame_id].get('dice', [0, 0, 0])
                        shaft_dice = dice_scores[1].item()
                        wrist_dice = dice_scores[0].item()
                        gripper_dice = dice_scores[2].item()
                        # avg_dice = np.mean(dice_scores)
                        if shaft_dice < self.min_dice_threshold[0] or \
                            wrist_dice < self.min_dice_threshold[1] or \
                            gripper_dice < self.min_dice_threshold[2] or \
                            shaft_dice == 1.0 or wrist_dice == 1.0:

                            problematic_type = 'low_dice'
                            err_frames_num += 1
                            if err_frames_num > 50:
                                full_case_usable = False
                            continue
                        # reduce the number of consecutive frames to avoid redundancy
                        if len(valid_frame_ids_in_one_case) > 0:
                            if int(frame_id) < int(valid_frame_ids_in_one_case[-1]) + 4:
                                continue
                        valid_frame_ids_in_one_case.append(frame_id)

                    samples.append((video_folder_name, frame_folder_name, mask_folder_name,instance_id, frame_id))

            except Exception as e:
                logger.warning(f"Error loading memory pool {memory_pool_path}: {e}")
                continue

        logger.info(f"Found {len(samples)} valid samples from {len(pose_folders)} pose folders")


        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _get_frame_path(self, video_folder_name: str, frame_folder_name: str, frame_id: str) -> str:
        """
        Get the path to an RGB frame.
        Note: frames start from 00001.png
        """
        # Frame files start from 00001.png
        frame_filename = f"{frame_id}.png"
        return os.path.join(self.video_root, video_folder_name, frame_folder_name, frame_filename)

    def _get_mask_path(self, video_folder_name: str, instance_id: int,
                       mask_type: str, frame_id: str) -> str:
        """
        Get the path to a mask file.
        Note: masks start from 00000.png

        Args:
            mask_type: One of 'masks_overall', 'masks_wrist', 'masks_shaft', 'masks_gripper'
        """
        # Mask files start from 00000.png
        mask_filename = f"{int(frame_id)-1:05d}.png"
        return os.path.join(
            self.video_root, video_folder_name,
            f'instance{instance_id}', mask_type, mask_filename
        )

    def _load_memory_pool(self, video_folder_name: str, instance_id: int) -> Dict:
        """
        Load or retrieve cached memory pool for a video/instance.
        """
        pose_folder = os.path.join(
            self.pose_root,
            f"{video_folder_name}_instance{instance_id}"
        )

        if pose_folder not in self._memory_pool_cache:
            memory_pool_path = os.path.join(pose_folder, 'memory_pool.pth')
            self._memory_pool_cache[pose_folder] = torch.load(
                memory_pool_path, map_location='cpu'
            )

        return self._memory_pool_cache[pose_folder]

    def _get_bbox_from_mask(self, mask: np.ndarray, padding: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bounding box from mask with optional padding.

        Args:
            mask: Binary mask array
            padding: Padding to add around the bounding box

        Returns:
            bbox_min, bbox_max as numpy arrays
        """
        # Find non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            # Return full image bbox if mask is empty
            h, w = mask.shape[:2]
            return np.array([0, 0]), np.array([w, h])

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = mask.shape[:2]
        bbox_min = np.array([max(0, cmin - padding), max(0, rmin - padding)])
        bbox_max = np.array([min(w, cmax + padding), min(h, rmax + padding)])

        return bbox_min, bbox_max

    def _extract_joint_angles_from_pose(self, pose: Pose) -> torch.Tensor:
        rot, trans, alpha, theta_l, theta_r = pose.get_pose()
        rot = rot.detach().cpu()#.numpy()
        trans = trans.detach().cpu()#.numpy()
        alpha = alpha.detach().cpu()#.numpy()
        theta_l = theta_l.detach().cpu()#.numpy()
        theta_r = theta_r.detach().cpu()#.numpy()
        pose_dict = {'rot': rot, 'trans': trans, 'alpha': alpha, 'theta_l': theta_l, 'theta_r': theta_r}
        return pose_dict
    def project_3d_to_2d(self,points_3d: Dict[str, np.ndarray], K: np.ndarray) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        """

        Project 3D points to 2D image plane
        """
        points_2d = {}
        depths = {}
        for name, pt in points_3d.items():
            proj = K @ pt
            u, v = proj[0] / proj[2], proj[1] / proj[2]
            points_2d[name] = (u, v)
            depths[name] = pt[2]
        return points_2d, depths
    def rotation_matrix_x(self,angle_degrees):
        angle_radians = np.radians(angle_degrees)
        cos_angle = np.cos(angle_radians)
        sin_angle = np.sin(angle_radians)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_angle, -sin_angle],
            [0, sin_angle, cos_angle]
        ])
        return torch.tensor(rotation_matrix, dtype=torch.float32)

    def _compute_keypoints_from_pose(self, pose: Pose, intrinsics: np.ndarray, device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        rot, trans, alpha, theta_l, theta_r = pose.get_pose()
        device = alpha.device
        if rot[2,2] <= 0:
            try:
                self.instrument.forward_kinematics(rot, trans, alpha, theta_l, theta_r, device = device)
            except TypeError:
                self.instrument.forward_kinematics(rot, trans, alpha, theta_l, theta_r,)
        else:
            try:
                alpha = -alpha
                theta_tmp = theta_l
                theta_l = theta_r
                theta_r = theta_tmp
                self.instrument.forward_kinematics(rot @ self.rotation_matrix_x(180), trans, alpha, theta_l, theta_r, device = device)
            except TypeError:
                self.instrument.forward_kinematics(rot @ self.rotation_matrix_x(180), trans, alpha, theta_l, theta_r,)
        points_3d = {}
        points_3d['wrist_origin'] = self.instrument.wrist.render_params['keypoints'][1].detach().cpu().numpy()
        points_3d['shaft_base'] = self.instrument.shaft.render_params['keypoints'][0].detach().cpu().numpy()
        points_3d['wrist_joint'] = self.instrument.wrist.render_params['keypoints'][0].detach().cpu().numpy()
        points_3d['gripper_left'] = self.instrument.l_gripper.render_params['keypoints'][0].detach().cpu().numpy()
        points_3d['gripper_right'] = self.instrument.r_gripper.render_params['keypoints'][0].detach().cpu().numpy()
        points_2d, _ = self.project_3d_to_2d(points_3d, intrinsics)
        return points_2d, points_3d, alpha, theta_l, theta_r

    def _get_camera_to_robot_transform(self, pose) -> np.ndarray:
        # eye matrix
        cTr = np.eye(4, dtype=np.float32)
        return cTr

    def __getitem__(self, index: int):
        """
        Get a single sample.

        Returns:
            img: Normalized image tensor (C, H, W)
            jointpose: Joint angles tensor
            belief_maps: Belief maps tensor for keypoints
            metadata: Dictionary containing additional information
        """
        video_folder_name, frame_folder_name, mask_folder_name, instance_id, frame_id = self.samples[index]

        # Load RGB image
        frame_path = self._get_frame_path(video_folder_name,frame_folder_name, frame_id)
        img = Image.open(frame_path).convert('RGB')
        img_array = np.array(img)

        # Load overall mask for bounding box computation
        mask_path = self._get_mask_path(video_folder_name, instance_id, mask_folder_name+'_overall', frame_id)
        masks = self.load_masks(video_folder_name, instance_id, frame_id)
        wrist_mask = masks['wrist']
        gripper_mask = masks['gripper']
        shaft_mask = masks['shaft']
        wrist_gripper_mask = wrist_mask + gripper_mask
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))
            mask = (mask > 127).astype(np.uint8)
        # else:
        #     # If no mask available, use full image
        #     mask = np.ones(img_array.shape[:2], dtype=np.uint8)

        # Load pose annotation
        memory_pool = self._load_memory_pool(video_folder_name, instance_id)
        frame_data = memory_pool[frame_id]
        pose = frame_data['pose_info']
        dice_scores = frame_data.get('dice', [0, 0, 0])

        # Extract joint angles
        pose_dict = self._extract_joint_angles_from_pose(pose)

        # Compute keypoints from pose
        keypoints_dict, keypoints_3d_dict, alpha, theta_l, theta_r = self._compute_keypoints_from_pose(pose, self.intrinsics)

        # Convert keypoints dict to tensor (N, 2) and (N, 3) format like DREAM
        # Order: shaft_base, wrist_origin, wrist_joint, gripper_left, gripper_right

        kp_data = []
        kp3d_data = []
        for name in self.keypoint_names:
            if name in keypoints_dict:
                kp_data.append(list(keypoints_dict[name]))  # (u, v)
                kp3d_data.append(list(keypoints_3d_dict[name]))  # (x, y, z)
        keypoints = torch.as_tensor(kp_data, dtype=torch.float32)  # (N, 2)
        keypoints_3d = torch.as_tensor(kp3d_data, dtype=torch.float32)  # (N, 3)

        # Compute bounding box from mask
        bbox_min, bbox_max = self._get_bbox_from_mask(mask)
        bbox_wg_min, bbox_wg_max = self._get_bbox_from_mask(wrist_gripper_mask)

        # Apply epoch-based bbox augmentation (same as DREAM)
        if self.train:
            if self.epoch < 30:
                pass
            elif self.epoch < 50:
                bbox_min = bbox_min - (np.random.rand(2)) * 30
                bbox_max = bbox_max + (np.random.rand(2)) * 30
            elif self.epoch < 70:
                bbox_min = bbox_min - (np.random.rand(2)) * 50
                bbox_max = bbox_max + (np.random.rand(2)) * 50
            elif self.epoch < 90:
                bbox_min = bbox_min - (np.random.rand(2)) * 80
                bbox_max = bbox_max + (np.random.rand(2)) * 80
            elif self.epoch < 110:
                bbox_min = bbox_min - (np.random.rand(2)) * 100
                bbox_max = bbox_max + (np.random.rand(2)) * 100
            else:
                bbox_min = bbox_min - (np.random.rand(2)) * 120
                bbox_max = bbox_max + (np.random.rand(2)) * 120

        # Clip bbox to image bounds
        h, w = img_array.shape[:2]
        bbox_min = np.clip(bbox_min, [0.0, 0.0], [w, h])
        bbox_max = np.clip(bbox_max, [0.0, 0.0], [w, h])

        # Prepare metadata
        metadata = {}
        metadata['img_path'] = frame_path
        metadata['video_folder'] = video_folder_name
        metadata['instance_id'] = instance_id
        metadata['frame_id'] = frame_id
        metadata['dice_scores'] = dice_scores
        metadata['orig_img'] = copy.deepcopy(img_array)
        metadata['orig_keypoints'] = copy.deepcopy(keypoints.numpy())
        metadata['orig_keypoints_3d'] = copy.deepcopy(keypoints_3d.numpy())
        metadata['keypoint_names'] = self.keypoint_names  # Add keypoint names for visualization
        metadata['bbox_wg_min'] = bbox_wg_min
        metadata['bbox_wg_max'] = bbox_wg_max



        # Crop image
        img_cropped = img_array[int(bbox_min[1]):int(bbox_max[1]), int(bbox_min[0]):int(bbox_max[0])]

        # Adjust keypoints for cropping
        keypoints = keypoints.clone()
        keypoints[:, 0] -= bbox_min[0]
        keypoints[:, 1] -= bbox_min[1]
        metadata['bbox_min'] = bbox_min
        metadata['bbox_max'] = bbox_max

        # Resize
        img_resized, (new_w, new_h) = self.resize_transform(img_cropped)
        scale_x = new_w / (bbox_max[0] - bbox_min[0])
        scale_y = new_h / (bbox_max[1] - bbox_min[1])
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        metadata['scale'] = (scale_x, scale_y)

        # Pad
        img_padded, (pad_w, pad_h) = self.pad_transform(img_resized)
        keypoints[:, 0] += pad_w
        keypoints[:, 1] += pad_h
        metadata['pad'] = (pad_w, pad_h)

        # Valid indices mask
        valid_indices_mask = (keypoints[:, 0] > 0) & (keypoints[:, 0] < self.crop_size) & \
                             (keypoints[:, 1] > 0) & (keypoints[:, 1] < self.crop_size)
        metadata['valid_indices_mask'] = valid_indices_mask

        img = Image.fromarray(img_padded)

        # Create belief maps
        belief_maps = create_belief_map(
            image_resolution=(self.crop_size, self.crop_size),
            pointsBelief=keypoints.numpy(),
            sigma=2
        )
        belief_maps_as_tensor = torch.tensor(belief_maps).float()

        # Apply augmentations during training
        if self.train:
            rgb = np.array(img)
            if self.color_jitter and random.random() < 0.4:
                rgb = apply_color_jitter(rgb)
            if self.occlusion_augmentation and random.random() < self.occlu_p:
                rgb = apply_occlusion(rgb)
            if self.rgb_augmentation and random.random() < 0.2:
                rgb = apply_rgb_aug(rgb)
            img = rgb

        # Convert to tensor and normalize
        img = torchvision.transforms.ToTensor()(img)
        img = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )(img)

        # Store intrinsics
        metadata['K'] = self.intrinsics

        # Get camera to robot transformation
        cTr = self._get_camera_to_robot_transform(pose)
        metadata['cTr'] = cTr

        jointpose = torch.stack([alpha, theta_l, theta_r], dim=0)
        return img, jointpose, belief_maps_as_tensor, metadata

    def load_masks(self, video_folder_name: str, instance_id: int, frame_id: int) -> Dict[str, np.ndarray]:
        """
        Load all masks for a given frame.

        Returns:
            Dictionary with keys: 'overall', 'wrist', 'shaft', 'gripper'
        """
        mask_types = ['masks_overall', 'masks_wrist', 'masks_shaft', 'masks_gripper']
        masks = {}

        for mask_type in mask_types:
            mask_path = self._get_mask_path(video_folder_name, instance_id, mask_type, frame_id)

            key = mask_type.replace('masks_', '')

            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path).convert('L'))
                masks[key] = (mask > 127).astype(np.uint8)
            else:
                masks[key] = None

        return masks


# Optional: Collate function for batching
def rarp_collate_fn(batch):
    """
    Custom collate function for RARP dataset.
    """
    imgs, jointposes, belief_maps, metadatas = zip(*batch)

    imgs = torch.stack(imgs, dim=0)
    # jointposes is now a list of dicts, don't stack
    belief_maps = torch.stack(belief_maps, dim=0)

    return imgs, list(jointposes), belief_maps, list(metadatas)


def visualize_sample(img_tensor, pose_dict, belief_maps, metadata, save_path=None):
    """
    Visualize a single sample for debugging.

    Args:
        img_tensor: Normalized image tensor (C, H, W)
        pose_dict: Dictionary containing pose information
        belief_maps: Belief maps tensor
        metadata: Metadata dictionary
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_denorm = img_tensor * std + mean
    img_denorm = img_denorm.permute(1, 2, 0).numpy()
    img_denorm = np.clip(img_denorm, 0, 1)

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Original image with keypoints
    ax = axes[0, 0]
    orig_img = metadata['orig_img']
    ax.imshow(orig_img)
    orig_kps = metadata['orig_keypoints']  # shape (N, 2)
    keypoint_names = metadata.get('keypoint_names',
        ['shaft_base', 'wrist_origin', 'wrist_joint', 'gripper_left', 'gripper_right'])
    for i, (u, v) in enumerate(orig_kps):
        name = keypoint_names[i] if i < len(keypoint_names) else f'kp{i}'
        ax.plot(u, v, 'ro', markersize=8)
        ax.annotate(name, (u, v), fontsize=6, color='yellow')
    ax.set_title(f"Original Image\n{metadata['video_folder']}\nframe={metadata['frame_id']}, inst={metadata['instance_id']}")
    ax.axis('off')

    # 2. Cropped & processed image
    ax = axes[0, 1]
    ax.imshow(img_denorm)
    ax.set_title(f"Processed Image (crop_size={img_denorm.shape[0]})")
    ax.axis('off')

    # 3. Belief maps overlay
    ax = axes[0, 2]
    ax.imshow(img_denorm)
    belief_sum = belief_maps.sum(dim=0).numpy()
    ax.imshow(belief_sum, alpha=0.5, cmap='hot')
    ax.set_title("Belief Maps Overlay")
    ax.axis('off')

    # 4. Individual belief maps
    ax = axes[1, 0]
    n_keypoints = belief_maps.shape[0]
    belief_grid = belief_maps.numpy()
    # Show first few belief maps
    n_show = min(n_keypoints, 5)
    combined = np.zeros_like(belief_grid[0])
    for i in range(n_show):
        combined = np.maximum(combined, belief_grid[i] * (i + 1) / n_show)
    ax.imshow(combined, cmap='viridis')
    ax.set_title(f"Belief Maps ({n_keypoints} keypoints)")
    ax.axis('off')

    # 5. Pose info text
    ax = axes[1, 1]
    ax.axis('off')
    pose_text = "Pose Information:\n"
    pose_text += "-" * 30 + "\n"
    for key, value in pose_dict.items():
        if isinstance(value, torch.Tensor):
            if value.numel() <= 6:
                pose_text += f"{key}: {value.numpy()}\n"
            else:
                pose_text += f"{key}: shape={value.shape}\n"
        else:
            pose_text += f"{key}: {value}\n"
    pose_text += "-" * 30 + "\n"
    pose_text += f"Dice scores: {metadata['dice_scores']}\n"
    pose_text += f"bbox_min: {metadata['bbox_min']}\n"
    pose_text += f"bbox_max: {metadata['bbox_max']}\n"
    pose_text += f"scale: {metadata['scale']}\n"
    pose_text += f"pad: {metadata['pad']}\n"
    ax.text(0.1, 0.9, pose_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title("Pose & Metadata")

    # 6. 3D keypoints info
    ax = axes[1, 2]
    ax.axis('off')
    kp3d_text = "3D Keypoints (camera frame):\n"
    kp3d_text += "-" * 30 + "\n"
    orig_kps_3d = metadata['orig_keypoints_3d']  # shape (N, 3)
    keypoint_names = metadata.get('keypoint_names',
        ['shaft_base', 'wrist_origin', 'wrist_joint', 'gripper_left', 'gripper_right'])
    for i, pt in enumerate(orig_kps_3d):
        name = keypoint_names[i] if i < len(keypoint_names) else f'kp{i}'
        kp3d_text += f"{name}:\n  [{pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}]\n"
    ax.text(0.1, 0.9, kp3d_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.set_title("3D Keypoints")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        # Default save path if none provided
        default_path = 'tmp/rarp_debug_sample.png'
        if not os.path.exists('tmp'):
            os.makedirs('tmp', exist_ok=True)
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {default_path}")

    plt.close(fig)  # Close figure to free memory (no display)
    return fig


if __name__ == "__main__":
    import argparse
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Debug RARP Dataset')
    parser.add_argument('--video_root', type=str,
                        default='/mnt/nas/share/shuojue/data/needleGrasping_videos',
                        help='Path to video data folder')
    parser.add_argument('--pose_root', type=str,
                        default='/mnt/nas/share/shuojue/data/needleGrasping_results',
                        help='Path to pose annotation folder')
    parser.add_argument('--index', type=int, default=0,
                        help='Sample index to visualize')
    parser.add_argument('--num_samples', type=int, default=300,
                        help='Number of random samples to visualize')
    parser.add_argument('--save_dir', type=str, default='tmp/rarp_debug',
                        help='Directory to save visualizations')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Crop size for images')
    parser.add_argument('--no_train_aug', action='store_true',
                        help='Disable training augmentations')
    args = parser.parse_args()

    print("=" * 60)
    print("RARP Dataset Debug Script")
    print("=" * 60)

    # Create dataset
    print(f"\nLoading dataset...")
    print(f"  video_root: {args.video_root}")
    print(f"  pose_root: {args.pose_root}")
    print(f"  crop_size: {args.crop_size}")
    print(f"  train mode: {not args.no_train_aug}")

    dataset = RARP(
        video_root=args.video_root,
        pose_root=args.pose_root,
        train=not args.no_train_aug,
        crop_size=args.crop_size
    )

    print(f"\nDataset loaded successfully!")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Joint names: {dataset.joint_names}")
    print(f"  Link names: {dataset.link_names}")
    print(f"  Intrinsics:\n{dataset.intrinsics}")

    # Print sample distribution
    print(f"\nSample distribution:")
    video_counts = {}
    for sample in dataset.samples:
        video_name = sample[0]
        if video_name not in video_counts:
            video_counts[video_name] = 0
        video_counts[video_name] += 1
    for video_name, count in sorted(video_counts.items()):
        print(f"  {video_name}: {count} samples")

    # Test specific sample
    print(f"\n{'=' * 60}")
    print(f"Testing sample at index {args.index}...")
    print("=" * 60)

    try:
        img, pose_dict, belief_maps, metadata = dataset[args.index]

        print(f"\n[Output Format Check]")
        print(f"  img type: {type(img)}, shape: {img.shape}, dtype: {img.dtype}")
        print(f"  pose_dict type: {type(pose_dict)}, keys: {pose_dict.keys()}")
        for k, v in pose_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"    {k}: {type(v)}")
        print(f"  belief_maps type: {type(belief_maps)}, shape: {belief_maps.shape}, dtype: {belief_maps.dtype}")
        print(f"  metadata type: {type(metadata)}, keys: {metadata.keys()}")

        print(f"\n[Metadata Details]")
        print(f"  img_path: {metadata['img_path']}")
        print(f"  video_folder: {metadata['video_folder']}")
        print(f"  instance_id: {metadata['instance_id']}")
        print(f"  frame_id: {metadata['frame_id']}")
        print(f"  dice_scores: {metadata['dice_scores']}")
        print(f"  bbox_min: {metadata['bbox_min']}")
        print(f"  bbox_max: {metadata['bbox_max']}")
        print(f"  scale: {metadata['scale']}")
        print(f"  pad: {metadata['pad']}")
        print(f"  valid_indices_mask: {metadata['valid_indices_mask']}")
        print(f"  K (intrinsics):\n{metadata['K']}")
        print(f"  cTr (camera to robot):\n{metadata['cTr']}")

        print(f"\n[Keypoints]")
        print(f"  orig_keypoints (2D):")
        for name, pt in metadata['orig_keypoints'].items():
            print(f"    {name}: ({pt[0]:.2f}, {pt[1]:.2f})")
        print(f"  orig_keypoints_3d:")
        for name, pt in metadata['orig_keypoints_3d'].items():
            print(f"    {name}: [{pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f}]")

        # Visualize
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"sample_{args.index}.png")

        visualize_sample(img, pose_dict, belief_maps, metadata, save_path)

    except Exception as e:
        print(f"Error loading sample {args.index}: {e}")
        import traceback
        traceback.print_exc()

    # Test random samples
    if args.num_samples > 1:
        print(f"\n{'=' * 60}")
        print(f"Testing {args.num_samples} random samples...")
        print("=" * 60)

        random_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

        for i, idx in enumerate(random_indices):
            print(f"\n[Sample {i+1}/{args.num_samples}] index={idx}")
            try:
                img, pose_dict, belief_maps, metadata = dataset[idx]
                print(f"  ✓ Loaded successfully")
                print(f"    video: {metadata['video_folder']}")
                print(f"    frame_id: {metadata['frame_id']}, instance_id: {metadata['instance_id']}")
                print(f"    img shape: {img.shape}")
                print(f"    belief_maps shape: {belief_maps.shape}")
                print(f"    num keypoints (2D): {len(metadata['orig_keypoints'])}")
                print(f"    num keypoints (3D): {len(metadata['orig_keypoints_3d'])}")

                if args.save_dir:
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir, exist_ok=True)
                    save_path = os.path.join(args.save_dir, f"sample_random_{idx}.png")
                    visualize_sample(img, pose_dict, belief_maps, metadata, save_path)

            except Exception as e:
                print(f"  ✗ Error: {e}")

    # Test DataLoader
    print(f"\n{'=' * 60}")
    print("Testing DataLoader...")
    print("=" * 60)

    try:
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            collate_fn=rarp_collate_fn
        )

        batch = next(iter(dataloader))
        imgs, pose_dicts, belief_maps, metadatas = batch

        print(f"  ✓ DataLoader works!")
        print(f"    batch imgs shape: {imgs.shape}")
        print(f"    batch pose_dicts: list of {len(pose_dicts)} dicts")
        print(f"    batch belief_maps shape: {belief_maps.shape}")
        print(f"    batch metadatas: list of {len(metadatas)} dicts")

    except Exception as e:
        print(f"  ✗ DataLoader error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print("Debug complete!")
    print("=" * 60)

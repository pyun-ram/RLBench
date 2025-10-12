import os
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.backend.observation import Observation
from rlbench.backend import utils
from rlbench.backend.const import *
from rlbench.utils import write_image, write_pkl
from PIL import Image

class CameraMotion(object):
    def __init__(self, cam: VisionSensor):
        self.cam = cam

    def step(self):
        raise NotImplementedError()

    def save_pose(self):
        self._prev_pose = self.cam.get_pose()

    def restore_pose(self):
        self.cam.set_pose(self._prev_pose)


class CircleCameraMotion(CameraMotion):

    def __init__(self, cam: VisionSensor, origin: Dummy,
                 speed: float, init_rotation: float = np.deg2rad(180)):
        super().__init__(cam)
        self.origin = origin
        self.speed = speed  # in radians
        self.origin.rotate([0, 0, init_rotation])

    def step(self):
        self.origin.rotate([0, 0, self.speed])

class NeRFTaskRecorder(object):
    """
    for nerf data generation
    """

    def __init__(self, cam_list, cam_mask_list, cam_name_list):
        self._cam_list = cam_list
        self._cam_mask_list = cam_mask_list
        self._cam_name_list = cam_name_list

        self._snaps_episode = []
        self._depths_episode = []
        self._mask_episode = [] # add mask data
        self._poses_episode = []
        self._intrinsics_episode = []
        self._near_far_episode = []
        return

    def reset(self):
        self._snaps_episode = []
        self._depths_episode = []
        self._mask_episode = [] # add mask data
        self._poses_episode = []
        self._intrinsics_episode = []
        self._near_far_episode = []
        return

    def record_task_description(self, task_description):
        self._task_description = task_description
    
    def take_snap(self, scene=None, obs: Observation=None):
        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                mask = mask_fn(sensor.capture_rgb())
            return mask
        
        mask_fn = lambda x: x

        # get views
        all_views = []
        all_depths = []
        all_poses = []
        all_intrinsics = []
        all_near_far = []
        all_mask = []
        for i, (cam, cam_mask) in enumerate(zip(self._cam_list, self._cam_mask_list)):
            all_views.append((cam.capture_rgb() * 255.).astype(np.uint8))
            mask = get_mask(cam_mask, mask_fn)
            all_mask.append(mask)
            all_poses.append(cam.get_matrix())
            all_depths.append(cam.capture_depth(in_meters=False))
            all_intrinsics.append(cam.get_intrinsic_matrix())
            all_near_far.append((cam.get_near_clipping_plane(),
                                 cam.get_far_clipping_plane()))

        self._snaps_episode.append(all_views)
        self._depths_episode.append(all_depths)
        self._mask_episode.append(all_mask)
        self._poses_episode.append(all_poses)
        self._intrinsics_episode.append(all_intrinsics)
        self._near_far_episode.append(all_near_far)
        return


    def save_extrinsic_and_intrinsic(self, path, extrinsic, intrinsic, near, far):
        return write_pkl({
            "intrinsic": intrinsic,
            "extrinsic": extrinsic,
            "near": near,
            "far": far,
        }, path)


    def save(self, path_dir):
        """
        save imgs and poses for nerf
        """
        os.makedirs(path_dir, exist_ok=True)
        print('saving imgs, extrinsic, intrinsic to {}'.format(path_dir))

        assert len(self._snaps_episode) > 0, 'No imgs to save'
        
        for t, all_views in enumerate(self._snaps_episode):
            # save all views and poses of this time step in a folder
            timestep_dir = os.path.join(path_dir, str(t))
            timestep_img_dir = os.path.join(timestep_dir, 'images')
            timestep_mask_dir = os.path.join(timestep_dir, 'masks')
            timestep_depth_dir = os.path.join(timestep_dir, 'depths')
            timestep_pose_dir = os.path.join(timestep_dir, 'poses')

            os.makedirs(timestep_img_dir, exist_ok=True)
            os.makedirs(timestep_mask_dir, exist_ok=True)
            os.makedirs(timestep_depth_dir, exist_ok=True)
            os.makedirs(timestep_pose_dir, exist_ok=True)

            all_poses = self._poses_episode[t]
            all_intrinsics = self._intrinsics_episode[t]
            all_near_far = self._near_far_episode[t]
            for i, view in enumerate(all_views):
                # save the image
                cam_name = self._cam_name_list[i]
                img_path = os.path.join(timestep_img_dir, str(cam_name) + '.png')
                write_image(view, img_path)

                # save the depth
                depth_path = os.path.join(timestep_depth_dir, str(cam_name) + '.png')
                depth = self._depths_episode[t][i]
                depth = utils.float_array_to_rgb_image(depth, scale_factor=DEPTH_SCALE)
                depth.save(depth_path)
                
                # save the mask
                mask_path = os.path.join(timestep_mask_dir, str(cam_name) + '.png')
                mask = self._mask_episode[t][i]
                mask = Image.fromarray((mask * 255).astype(np.uint8))
                mask.save(mask_path)
                
                
                # save the pose and intrinsic
                pose_path = os.path.join(timestep_pose_dir, str(cam_name) + '.pkl')
                transformation_matrix =  all_poses[i]
                intrinsic_matrix = all_intrinsics[i]
                near, far = all_near_far[i]
                self.save_extrinsic_and_intrinsic(
                    pose_path,
                    extrinsic=transformation_matrix,
                    intrinsic=intrinsic_matrix,
                    near=near,
                    far=far,
                )
        self.reset()

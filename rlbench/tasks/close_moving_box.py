from typing import List, Tuple
import numpy as np
from pyrep.objects import CartesianPath, Dummy, Shape
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
import math
import torch
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError

def euler_to_quaternion_xyzw(euler):
    """
    Args:
        euler: List[float] = [roll_x, pitch_y, yaw_z] in radians
               Apply order: R = Rz(yaw) · Ry(pitch) · Rx(roll)  (ZYX)
    Return:
        List[float]: quaternion in [x, y, z, w] (xyzw)
    """
    rx, ry, rz = euler  # roll, pitch, yaw
    cr, sr = math.cos(rx * 0.5), math.sin(rx * 0.5)
    cp, sp = math.cos(ry * 0.5), math.sin(ry * 0.5)
    cz, sz = math.cos(rz * 0.5), math.sin(rz * 0.5)

    w = cz*cp*cr + sz*sp*sr
    x = cz*cp*sr - sz*sp*cr
    y = cz*sp*cr + sz*cp*sr
    z = cz*sp*sr - sz*cp*cr
    return [x, y, z, w]

def get_expert_info(task, bool_return_path=True):
    stage = task.stage
    tip_pose = task.robot.arm.get_tip().get_pose()
    wp0_pose = task.wp0.get_pose()
    wp1_pose = task.wp1.get_pose()
    wp2_inter0_pos_rot = task.wp2.get_pose_on_path(0.05)
    wp2_inter0_pose = np.concatenate([wp2_inter0_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter0_pos_rot[1])])
    wp2_inter0p1_pos_rot = task.wp2.get_pose_on_path(0.15)
    wp2_inter0p1_pose = np.concatenate([wp2_inter0p1_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter0p1_pos_rot[1])])
    wp2_inter1_pos_rot = task.wp2.get_pose_on_path(0.25)
    wp2_inter1_pose = np.concatenate([wp2_inter1_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter1_pos_rot[1])])
    wp2_inter2_pos_rot = task.wp2.get_pose_on_path(0.35)
    wp2_inter2_pose = np.concatenate([wp2_inter2_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter2_pos_rot[1])])
    wp2_inter2p1_pos_rot = task.wp2.get_pose_on_path(0.45)
    wp2_inter2p1_pose = np.concatenate([wp2_inter2p1_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter2p1_pos_rot[1])])
    wp2_inter2p2_pos_rot = task.wp2.get_pose_on_path(0.55)
    wp2_inter2p2_pose = np.concatenate([wp2_inter2p2_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter2p2_pos_rot[1])])
    wp2_inter3_pos_rot = task.wp2.get_pose_on_path(0.65)
    wp2_inter3_pose = np.concatenate([wp2_inter3_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter3_pos_rot[1])])
    wp2_inter4_pos_rot = task.wp2.get_pose_on_path(1.0)
    wp2_inter4_pose = np.concatenate([wp2_inter4_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter4_pos_rot[1])])
    wp3_pose = task.wp3.get_pose()
    wp4_pose = task.wp4.get_pose()

    dist_to_wp0 = np.linalg.norm(tip_pose[:3] - wp0_pose[:3])
    dist_to_wp1 = np.linalg.norm(tip_pose[:3] - wp1_pose[:3])
    dist_to_wp2_inter0 = np.linalg.norm(tip_pose[:3] - wp2_inter0_pose[:3])
    dist_to_wp2_inter0p1 = np.linalg.norm(tip_pose[:3] - wp2_inter0p1_pose[:3])
    dist_to_wp2_inter1 = np.linalg.norm(tip_pose[:3] - wp2_inter1_pose[:3])
    dist_to_wp2_inter2 = np.linalg.norm(tip_pose[:3] - wp2_inter2_pose[:3])
    dist_to_wp2_inter2p1 = np.linalg.norm(tip_pose[:3] - wp2_inter2p1_pose[:3])
    dist_to_wp2_inter2p2 = np.linalg.norm(tip_pose[:3] - wp2_inter2p2_pose[:3])
    dist_to_wp2_inter3 = np.linalg.norm(tip_pose[:3] - wp2_inter3_pose[:3])
    dist_to_wp2_inter4 = np.linalg.norm(tip_pose[:3] - wp2_inter4_pose[:3])
    dist_to_wp3 = np.linalg.norm(tip_pose[:3] - wp3_pose[:3])
    dist_to_wp4 = np.linalg.norm(tip_pose[:3] - wp4_pose[:3])

    th_wp0 = 0.03
    th_wp1 = 0.02
    th_wp3 = 0.02
    th_wp2_inter0 = 0.02 * 10
    th_wp2_inter0p1 = 0.02 * 10
    th_wp2_inter1 = 0.02 * 10
    th_wp2_inter2 = 0.02 * 10
    th_wp2_inter2p1 = 0.02 * 10
    th_wp2_inter2p2 = 0.02 * 10
    th_wp2_inter3 = 0.02
    th_wp2_inter4 = 0.03

    is_open = any(x > 0.95 for x in task.robot.gripper.get_open_amount())
    print('---------------------')
    print(f"task.step_id: {task.step_id}",
          f"stage: {stage}",
          f"is_open: {is_open}",
          f"dist_to_wp0: {dist_to_wp0:.2f}",
          f"dist_to_wp1: {dist_to_wp1:.2f}",
          f"dist_to_wp2_inter0: {dist_to_wp2_inter0:.2f}",
          f"dist_to_wp2_inter0p1: {dist_to_wp2_inter0p1:.2f}",
          f"dist_to_wp2_inter1: {dist_to_wp2_inter1:.2f}",
          f"dist_to_wp2_inter2: {dist_to_wp2_inter2:.2f}",
          f"dist_to_wp2_inter3: {dist_to_wp2_inter3:.2f}",
          f"dist_to_wp2_inter4: {dist_to_wp2_inter4:.2f}",
          f"dist_to_wp3: {dist_to_wp3:.2f}", f"dist_to_wp4: {dist_to_wp4:.2f}")
    
    if stage == 'wp0' and dist_to_wp0 > th_wp0:
        stage = 'wp0'
        open = 1
        eepose = wp0_pose
    elif stage == 'wp0' and dist_to_wp0 <= th_wp0:
        stage = 'wp1'
        open = 0
        eepose = wp1_pose
    elif stage == 'wp1' and dist_to_wp1 > th_wp1:
        stage = 'wp1'
        open = 0
        eepose = wp1_pose
    elif stage == 'wp1' and dist_to_wp1 <= th_wp1 and is_open:
        stage = 'wp1'
        open = 0
        eepose = wp1_pose
    elif stage == 'wp1' and dist_to_wp1 <= th_wp1 and not is_open:
        stage = 'wp2_inter0'
        open = 0
        eepose = wp2_inter0_pose
    elif stage == 'wp2_inter0' and dist_to_wp2_inter0 > th_wp2_inter0:
        stage = 'wp2_inter0'
        open = 0
        eepose = wp2_inter0_pose
    elif stage == 'wp2_inter0' and dist_to_wp2_inter0 <= th_wp2_inter0:
        stage = 'wp2_inter1'
        open = 0
        eepose = wp2_inter1_pose
    elif stage == 'wp2_inter0p1' and dist_to_wp2_inter0p1 > th_wp2_inter0p1:
        stage = 'wp2_inter0p1'
        open = 0
        eepose = wp2_inter0p1_pose
    elif stage == 'wp2_inter0p1' and dist_to_wp2_inter0p1 <= th_wp2_inter0p1:
        stage = 'wp2_inter1'
        open = 0
        eepose = wp2_inter1_pose
    elif stage == 'wp2_inter1' and dist_to_wp2_inter1 > th_wp2_inter1:
        stage = 'wp2_inter1'
        open = 0
        eepose = wp2_inter1_pose
    elif stage == 'wp2_inter1' and dist_to_wp2_inter1 <= th_wp2_inter1:
        stage = 'wp2_inter2'
        open = 0
        eepose = wp2_inter2_pose
    elif stage == 'wp2_inter2' and dist_to_wp2_inter2 > th_wp2_inter2:
        stage = 'wp2_inter2'
        open = 0
        eepose = wp2_inter2_pose
    elif stage == 'wp2_inter2' and dist_to_wp2_inter2 <= th_wp2_inter2:
        stage = 'wp2_inter2p2'
        open = 0
        eepose = wp2_inter2p2_pose
    elif stage == 'wp2_inter2p1' and dist_to_wp2_inter2p1 > th_wp2_inter2p1:
        stage = 'wp2_inter2p1'
        open = 0
        eepose = wp2_inter2p1_pose
    elif stage == 'wp2_inter2p1' and dist_to_wp2_inter2p1 <= th_wp2_inter2p1:
        stage = 'wp2_inter2p2'
        open = 0
        eepose = wp2_inter2p2_pose
    elif stage == 'wp2_inter2p2' and dist_to_wp2_inter2p2 > th_wp2_inter2p2:
        stage = 'wp2_inter2p2'
        open = 0
        eepose = wp2_inter2p2_pose
    elif stage == 'wp2_inter2p2' and dist_to_wp2_inter2p2 <= th_wp2_inter2p2:
        stage = 'wp2_inter3'
        open = 0
        eepose = wp2_inter3_pose
    elif stage == 'wp2_inter3' and dist_to_wp2_inter3 > th_wp2_inter3:
        stage = 'wp2_inter3'
        open = 0
        eepose = wp2_inter3_pose
    elif stage == 'wp2_inter3' and dist_to_wp2_inter3 <= th_wp2_inter3:
        stage = 'wp2_inter4'
        open = 0
        eepose = wp2_inter4_pose
    elif stage == 'wp2_inter4' and dist_to_wp2_inter4 > th_wp2_inter4:
        stage = 'wp2_inter4'
        open = 0
        eepose = wp2_inter4_pose
    elif stage == 'wp2_inter4' and dist_to_wp2_inter4 <= th_wp2_inter4:
        stage = 'wp3'
        open = 0
        eepose = wp3_pose
    elif stage == 'wp3' and dist_to_wp3 > th_wp3:
        stage = 'wp3'
        open = 0
        eepose = wp3_pose
    elif stage == 'wp3' and dist_to_wp3 <= th_wp3:
        stage = 'wp4'
        open = 0
        eepose = wp4_pose
    elif stage == 'wp4':
        stage = 'wp4'
        open = 1
        eepose = wp4_pose
    else:
        print("Unrecognized stage: ", stage)
        import pdb; pdb.set_trace()
    print(f"task.step_id: {task.step_id}",
          f"stage: {stage}",
          f"eepose: {eepose}",
          f"open: {open}",)
    print('---------------------------------')
    output = np.ones((1, 1, 8))
    output[0, 0, :7] = eepose
    output[0, 0, 7:] = open
    expert_info = {
        "trajectory": torch.from_numpy(output),
        "stage": stage,
        'open': open,
        "debug_info": {
            "tip_cur_position": tip_pose[:3],
            "tar_position": wp2_inter4_pose[:3],
            "t": task.t,
        }
    }
    if bool_return_path:
        path = task.get_path(eepose)
        expert_info["path"] = path
    return expert_info

class CloseMovingBox(Task):

    def init_task(self) -> None:
        self.register_success_conditions([
            JointCondition(Joint('box_joint'), 2.6)])
        self._frame_base = Shape('box_base')
        frame_base_z = self._frame_base.get_position()[2]
        self.area = [0.1, -0.4, frame_base_z, 0.4, 0.4, frame_base_z]
        self.var2target_state_list = {}
        for var_index in range(self.variation_count()):
            # yaw_speed = init_target_state(
            #     min_yaw=2.5,  # 5 degree/s
            #     max_yaw=7.5,  # 15 degree/s
            #     d_yaw=1,
            # )
            from .place_cups_on_rotating_frame import init_target_state
            yaw_speed = init_target_state(
                min_yaw=np.rad2deg(- np.pi * 20 / 1800),  # 5 degree/s
                max_yaw=np.rad2deg(- np.pi * 20 / 1800)*0.5,  # 15 degree/s
                d_yaw=1,
            )
            print(yaw_speed)
            frame_position = np.random.uniform(self.area[:3], self.area[3:])
            self.var2target_state_list[var_index] = {
                'yaw_speed': yaw_speed,
                'frame_position': frame_position,
            }

    def init_episode(self, index: int) -> List[str]:
        self.step_id = 0
        self.t = 0
        self.stage = 'wp0'
        self.target_state_list = []
        self.wp0 = Dummy('waypoint0')
        self.wp1 = Dummy('waypoint1')
        self.wp2 = CartesianPath('waypoint2')
        self.wp3 = Dummy('waypoint3')
        self.wp4 = Dummy('waypoint4')
        
        self._bool_expert = True
        self.var_index = index
        self.cleanup()
        self.yaw_speed = self.var2target_state_list[index]['yaw_speed']
        self.frame_position = self.var2target_state_list[index]['frame_position']
        self._frame_base.set_position(self.frame_position)
        return ['close box',
                'close the lid on the box',
                'shut the box',
                'shut the box lid']

    def variation_count(self) -> int:
        return 1
    
    def step(self) -> None:
        simulation_timestep = self.pyrep.get_simulation_timestep()
        rot = np.deg2rad(self.yaw_speed) * simulation_timestep
        self._frame_base.rotate([rot, 0, 0])
        if self._bool_expert:
            if self.step_id % 10 == 0:
                self._path, self._open = self.expert_plan()
                self._path_done = False
            if not self._path_done:
                self._path_done = self._path.step()
            if self._path_done:
                self.move_gripper_tip([self._open])

        self.step_id += 1
        self.t += simulation_timestep
        return

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 8], [0, 0, np.pi / 8]


    def disable_expert_plan(self):
        self._bool_expert = False
        return


    def expert_plan(self):
        expert_info = get_expert_info(self, bool_return_path=True)
        path = expert_info["path"]
        open = expert_info["open"]
        self.stage = expert_info["stage"]
        return path, open

    def move_gripper_tip(self, action):
        def _actuate(action):
            done = False
            while not done:
                done = self.robot.gripper.actuate(action, velocity=0.2)
                self.pyrep.step()
                # scene.task.step()
            return
        if 0.0 > action[0] > 1.0:
            raise InvalidActionError(
                'Gripper action expected to be within 0 and 1.')
        open_condition = all(
            x > 0.9 for x in self.robot.gripper.get_open_amount())
        current_ee = 1.0 if open_condition else 0.0
        action = float(action[0] > 0.5)

        if current_ee != action:
            detach_before_open = True
            attach_grasped_objects = True
            if not detach_before_open:
                _actuate(action)
            if action == 0.0 and attach_grasped_objects:
                # If gripper close action, the check for grasp.
                for g_obj in self.get_graspable_objects():
                    self.robot.gripper.grasp(g_obj)
            else:
                # If gripper open action, the check for un-grasp.
                self.robot.gripper.release()
            if detach_before_open:
                _actuate(action)
            if action == 1.0:
                # Step a few more times to allow objects to drop
                for _ in range(10):
                    self.pyrep.step()
                    # scene.task.step()
        return

    def get_path(self, action):
        ignore_collisions = True
        relative_to = None
        try:
            # try once with collision checking (if ignore_collisions is true)
            try:
                path = self.robot.arm.get_path(
                    action[:3],
                    quaternion=action[3:],
                    ignore_collisions=ignore_collisions,
                    relative_to=relative_to,
                    trials=100,
                    max_configs=10,
                    max_time_ms=10,
                    trials_per_goal=5,
                    algorithm=Algos.RRTConnect
                )
            except ConfigurationPathError as e:
                if ignore_collisions:
                    raise InvalidActionError(
                        'A path could not be found. Most likely due to the target '
                        'being inaccessible or a collison was detected.') from e
                else:
                    # try once more with collision checking disabled
                    path = self.robot.arm.get_path(
                        action[:3],
                        quaternion=action[3:],
                        ignore_collisions=True,
                        relative_to=relative_to,
                        trials=100,
                        max_configs=10,
                        max_time_ms=10,
                        trials_per_goal=5,
                        algorithm=Algos.RRTConnect
                    )
        except ConfigurationPathError as e:
            raise InvalidActionError(
                'A path could not be found. Most likely due to the target '
                'being inaccessible or a collison was detected.') from e
        # path = modify_path(path)
        return path

    def cleanup(self) -> None:
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self.stage = 'wp0'
        return

    def is_static_workspace(self):
        return True

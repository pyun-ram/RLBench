from typing import List, Tuple
import numpy as np
from pyrep.objects import CartesianPath, Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
import torch
from .place_cups_on_rotating_frame import init_target_state, compute_target_pose

def init_target_state(
    min_yaw: float,
    max_yaw: float,
    d_yaw: float = 1.0,
):
    '''
    Args:
        area: List[float], [xmin,ymin,zmin,xmax,ymax,zmax] in Fworld
        t_max: float, max time
        x0: List[float], initial position, 
        v0: float, initial velocity
        a0: float, initial acceleration
        dt: float
    Return:
        List[float], target state (x, v, a)
    '''
    yaw_range = [min_yaw, max_yaw]
    yaws = np.arange(yaw_range[0], yaw_range[1] + d_yaw/2, d_yaw)
    target_state = yaws
    idx = np.random.choice(target_state.shape[0])
    target_state = target_state[idx]
    return target_state

def compute_target_pose(
    frame_base,
    waypoint,
    t_delay,
    yaw_speed,
):
    org_matrix = frame_base.get_matrix()
    rot = np.deg2rad(yaw_speed) * t_delay
    frame_base.rotate([0, 0, rot])
    target_pose = waypoint.get_pose()
    frame_base.set_matrix(org_matrix)
    return target_pose

import math
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
    wp2_inter1_pos_rot = task.wp2.get_pose_on_path(0.25)
    wp2_inter0_pos_rot = task.wp2.get_pose_on_path(0.0)
    wp2_inter1_pose = np.concatenate([wp2_inter1_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter1_pos_rot[1])])
    wp2_inter2_pos_rot = task.wp2.get_pose_on_path(0.5)
    wp2_inter2_pose = np.concatenate([wp2_inter2_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter2_pos_rot[1])])
    wp2_inter3_pos_rot = task.wp2.get_pose_on_path(0.75)
    wp2_inter3_pose = np.concatenate([wp2_inter3_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter3_pos_rot[1])])
    wp2_inter4_pos_rot = task.wp2.get_pose_on_path(1.0)
    wp2_inter4_pose = np.concatenate([wp2_inter4_pos_rot[0], euler_to_quaternion_xyzw(wp2_inter4_pos_rot[1])])

    dist_to_wp0 = np.linalg.norm(tip_pose[:3] - wp0_pose[:3])
    dist_to_wp1 = np.linalg.norm(tip_pose[:3] - wp1_pose[:3])
    dist_to_wp2_inter1 = np.linalg.norm(tip_pose - wp2_inter1_pose[0])
    dist_to_wp2_inter2 = np.linalg.norm(tip_pose - wp2_inter2_pose[0])
    dist_to_wp2_inter3 = np.linalg.norm(tip_pose - wp2_inter3_pose[0])
    dist_to_wp2_inter4 = np.linalg.norm(tip_pose - wp2_inter4_pose[0])

    th_wp0 = 0.03
    th_wp1 = 0.03
    th_wp2_inter1 = 0.03
    th_wp2_inter2 = 0.03
    th_wp2_inter3 = 0.03
    th_wp2_inter4 = 0.03

    is_open = any(x > 0.95 for x in task.robot.gripper.get_open_amount())
    print('---------------------')
    print(f"task.step_id: {task.step_id}", f"is_open: {is_open}", f"dist_to_wp0: {dist_to_wp0:.2f}", f"dist_to_wp1: {dist_to_wp1:.2f}", f"dist_to_wp2_inter1: {dist_to_wp2_inter1:.2f}", f"dist_to_wp2_inter2: {dist_to_wp2_inter2:.2f}", f"dist_to_wp2_inter3: {dist_to_wp2_inter3:.2f}", f"dist_to_wp2_inter4: {dist_to_wp2_inter4:.2f}")
    print(f"stage: {stage}")
    if stage == 'wp0' and dist_to_wp0 > th_wp0:
        stage = 'wp0'
        eepose = wp0_pose
        open = 1
        delay = 0.0
    elif stage == 'wp0' and dist_to_wp0 <= th_wp0:
        stage = 'wp1'
        eepose = wp1_pose
        open = 0
        delay = 0.5
    elif stage == 'wp1' and is_open:
        stage = 'wp1'
        eepose = wp1_pose
        open = 0
        delay = 0
        import pdb; pdb.set_trace()
    elif stage == 'wp1' and not is_open:
        stage = 'wp2_inter1'
        eepose = wp2_inter1_pose
        open = 0
        delay = 0
        import pdb; pdb.set_trace()
    elif stage == 'wp2_inter1':
        stage = 'wp2_inter2'
        eepose = wp2_inter2_pose
        open = 0
        delay = 0
    elif stage == 'wp2_inter2':
        stage = 'wp2_inter3'
        eepose = wp2_inter3_pose
        open = 0
        delay = 0
    elif stage == 'wp2_inter3':
        stage = 'wp2_inter4'
        eepose = wp2_inter4_pose
        open = 0
        delay = 0
    elif stage == 'wp2_inter4':
        stage = 'wp2_inter4'
        eepose = wp2_inter4_pose
        open = 0
        delay = 0
    else:
        print("Unrecognized stage: ", stage)
        import pdb; pdb.set_trace()

    print(f"stage: {stage}, eepose: {eepose}, open: {open}, dist_to_wp0: {dist_to_wp0:.2f}, dist_to_wp1: {dist_to_wp1:.2f}, dist_to_wp2_inter1: {dist_to_wp2_inter1:.2f}, dist_to_wp2_inter2: {dist_to_wp2_inter2:.2f}, dist_to_wp2_inter3: {dist_to_wp2_inter3:.2f}, dist_to_wp2_inter4: {dist_to_wp2_inter4:.2f}")
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

class CloseMovingDoor(Task):

    def init_task(self) -> None:
        self.register_success_conditions([
            JointCondition(Joint('door_frame_joint'), np.deg2rad(25))
        ])
        self._frame_base = Shape('door_frame')
        frame_base_z = self._frame_base.get_position()[2]
        self.area = [0.2, -0.5, frame_base_z, 0.4, 0.5, frame_base_z]
        self.var2target_state_list = {}
        for var_index in range(self.variation_count()):
            yaw_speed = init_target_state(
                min_yaw=2.5,  # 5 degree/s
                max_yaw=7.5,  # 15 degree/s
                d_yaw=1,
            )
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

        self._bool_expert = True
        self.door_frame_joint = Joint('door_frame_joint')
        self.door_frame = Shape('door_frame')

        self.var_index = index
        self.cleanup()
        self.yaw_speed = self.var2target_state_list[index]['yaw_speed']
        self.target_state_list.append({
            'yaw_speed': self.yaw_speed,
            't0': 0,
        })
        self._frame_base.set_position(self.var2target_state_list[index]['frame_position'])
        return ['close the door',
                'shut the door',
                'grip the handle and pull the door shut',
                'use the handle to move the door closed']

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        self.yaw_speed = self.target_state_list[-1]['yaw_speed']
        simulation_timestep = self.pyrep.get_simulation_timestep()
        rot_speed = np.deg2rad(self.yaw_speed) * simulation_timestep
        self.door_frame.rotate([0, 0, rot_speed])
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
        ignore_collisions = False
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

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4.], [0, 0, np.pi / 4.]

    def boundary_root(self) -> Object:
        return Shape('boundary_root')

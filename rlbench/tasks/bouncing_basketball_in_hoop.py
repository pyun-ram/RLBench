from typing import List, Tuple

import numpy as np
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from pyrep.const import ConfigurationPathAlgorithms as Algos
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import ConfigurationPathError, IKError
from pyrep.backend import sim, utils
import torch
from pyrep.objects.dummy import Dummy

def get_expert_info(task, bool_return_path=True):
    wp0 = task.wp0.get_pose()
    wp1 = task.wp1.get_pose()
    wp2 = task.init_wp2_pose
    wp3 = task.wp3.get_pose()
    tip_pose = task.robot.arm.get_tip().get_pose()

    dist_to_wp0 = np.linalg.norm(tip_pose[:3] - wp0[:3])
    dist_to_wp1 = np.linalg.norm(tip_pose[:3] - wp1[:3])
    dist_to_wp2 = np.linalg.norm(tip_pose[:3] - wp2[:3])
    dist_to_wp3 = np.linalg.norm(tip_pose[:3] - wp3[:3])

    th_wp0 = 0.1
    th_wp1 = 0.05
    th_wp2 = 0.05
    th_wp3 = 0.05
    stage = task.stage
    is_grasping = len(task.robot.gripper.get_grasped_objects()) > 0

    if stage == 'wp0' and dist_to_wp0 > th_wp0:
        stage = 'wp0'
        eepose = wp0
        eepose[3:7] = [0, 1, 0, 0]
        open = 1
    elif stage == 'wp0' and dist_to_wp0 <= th_wp0:
        stage = 'wp1'
        eepose = wp1
        eepose[3:7] = [0, 1, 0, 0]
        open = 0
    elif stage == 'wp1' and dist_to_wp1 > th_wp1:
        stage = 'wp1'
        eepose = wp1
        eepose[3:7] = [0, 1, 0, 0]
        open = 0
    elif stage == 'wp1' and dist_to_wp1 <= th_wp1 and not is_grasping:
        stage = 'wp1'
        eepose = wp1
        eepose[3:7] = [0, 1, 0, 0]
        open = 0
    elif stage == 'wp1' and dist_to_wp1 <= th_wp1 and is_grasping:
        stage = 'wp2'
        eepose = wp2
        eepose[3:7] = [0, 1, 0, 0]
        open = 0
    elif stage == 'wp2' and dist_to_wp2 > th_wp2:
        stage = 'wp2'
        eepose = wp2
        eepose[3:7] = [0, 1, 0, 0]
        open = 0
    elif stage == 'wp2' and dist_to_wp2 <= th_wp2:
        stage = 'wp3'
        eepose = wp3
        eepose[3:7] = [0, 1, 0, 0]
        open = 1
    elif stage == 'wp3':
        stage = 'wp3'
        eepose = wp3
        eepose[3:7] = [0, 1, 0, 0]
        open = 1
    else:
        print("Unrecognized stage: ", stage)
        import pdb; pdb.set_trace()
    print(
        f"stage: {stage}, eepose: {eepose}, open: {open}, dist_to_wp0: {dist_to_wp0}")
    output = np.ones((1, 1, 8))
    output[0, 0, :7] = eepose
    output[0, 0, 7:] = open
    expert_info = {
        "trajectory": torch.from_numpy(output),
        "stage": stage,
        "open": open,
        "debug_info": {
            "tip_cur_position": tip_pose[:3],
            "tar_position": task.ball.get_position(),
            "t": task.t,
        }
    }
    if bool_return_path:
        path = task.get_path(eepose)
        expert_info["path"] = path
    return expert_info

class BouncingBasketballInHoop(Task):
    def init_task(self):
        ball = Shape('ball')
        self.register_graspable_objects([ball])
        self.register_success_conditions(
            [DetectedCondition(ball, ProximitySensor('success'))])
        self.ball = ball
        self.step_id = 0
        self.t = 0
        self.stage = 'wp0'
        self._bool_expert = True
        self.wp0 = Dummy('expert_wp0')
        self.wp1 = Dummy('expert_wp1')
        self.wp2 = Dummy('expert_wp2')
        self.wp3 = Dummy('expert_wp3')
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

    def get_path(self, action):
        def get_action_traj(path) -> np.ndarray:
            '''
            Args:
                path: ArmConfigurationPath
            Returns:
                np.ndarray, shape: (N, 7)
            '''
            def _set_joints(path, positions):
                [sim.simSetJointPosition(jh, p)  # type: ignore
                 for jh, p in zip(path._arm._joint_handles, positions)]
                [j.set_joint_target_position(p)  # type: ignore
                 for j, p in zip(path._arm.joints, positions)]
                return

            if len(path._path_points) <= 0:
                raise RuntimeError("Can't visualise a path with no points.")
            tip = path._arm.get_tip()
            init_angles = path._arm.get_joint_positions()
            action_traj = []
            joint_traj = []
            is_model = path._arm.is_model()
            if not is_model:
                path._arm.set_model(True)
            prior = sim.simGetModelProperty(path._arm.get_handle())
            p = prior | sim.sim_modelproperty_not_dynamic
            # Disable the dynamics
            sim.simSetModelProperty(path._arm._handle, p)
            with utils.step_lock:
                sim.simExtStep(True)  # Have to step for changes to take effect
            _set_joints(path, path._path_points[0: len(path._arm.joints)])
            for i in range(len(path._arm.joints), len(path._path_points),
                           len(path._arm.joints)):
                points = path._path_points[i:i + len(path._arm.joints)]
                _set_joints(path, points)
                p = list(tip.get_pose())  # x,y,z,qx,qy,qz,qw
                action_traj.append(p)
                joint_traj.append(points)
            _set_joints(path, init_angles)
            with utils.step_lock:
                sim.simExtStep(True)  # Have to step for changes to take effect
            # Re-enable the dynamics
            sim.simSetModelProperty(path._arm._handle, prior)
            path._arm.set_model(is_model)
            return np.array(action_traj), np.array(joint_traj)

        def modify_path(path, min_dist: float = 0.01, N: int = 10):
            # take 10 to 10+N action points
            cartesian_wp_array, joint_wp_array = get_action_traj(path)
            cartesian_wp_array = cartesian_wp_array[10:]
            joint_wp_array = joint_wp_array[10:]
            modified_cartesian_wp_array = []
            modified_joint_wp_array = []
            for i in range(len(cartesian_wp_array)):
                if i == 0:
                    modified_cartesian_wp_array.append(cartesian_wp_array[i])
                    modified_joint_wp_array.append(joint_wp_array[i])
                    continue
                dist = np.linalg.norm(
                    np.array(cartesian_wp_array[i][:3]) - np.array(modified_cartesian_wp_array[-1][:3]))
                if dist > min_dist:
                    add_noise = False
                    if add_noise:
                        try:
                            ee_noise_std = 0.02  # 2 cm for x/y/z dim
                            noise = np.random.normal(0, ee_noise_std, 7)
                            cartesian_wp = cartesian_wp_array[i] + noise
                            joint_wp = self.robot.arm.solve_ik_via_jacobian(
                                cartesian_wp[:3], quaternion=cartesian_wp[3:])
                        except IKError:
                            print(
                                f"Warning: Failed to add noise in execution. Using original cartesian_wp.")
                            cartesian_wp, joint_wp = cartesian_wp_array[i], joint_wp_array[i]
                    else:
                        cartesian_wp, joint_wp = cartesian_wp_array[i], joint_wp_array[i]
                    modified_cartesian_wp_array.append(cartesian_wp)
                    modified_joint_wp_array.append(joint_wp)
            cartesian_wp_array = np.array(modified_cartesian_wp_array)
            joint_wp_array = np.array(modified_joint_wp_array)
            if joint_wp_array.shape[0] < N:
                print(
                    f"Warning: joint_wp_array.shape[0] < N, {joint_wp_array.shape[0]} < {N}")
            path._path_points = np.asarray(joint_wp_array.reshape(-1))
            return path

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
        modify_path(path)
        return path

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

    def init_episode(self, index: int) -> List[str]:
        return ['put the ball in the hoop',
                'play basketball',
                'shoot the ball through the net',
                'pick up the basketball and put it in the hoop',
                'throw the basketball through the hoop',
                'place the basket ball through the hoop']
    
    def step(self):
        if self.step_id == 0:
            self.init_wp2_pose = self.wp2.get_pose()
            # 设置小球初始位置（高一些以便观察弹跳）
            ball_pose = self.ball.get_pose()
            ball_pose[2] += 0.5  # 提高到0.5米高度
            self.ball.set_pose(ball_pose)
        simulation_timestep = self.pyrep.get_simulation_timestep()
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
        np.set_printoptions(precision=3, suppress=True)
        print(self.step_id, self.ball.get_pose())
        import pdb; pdb.set_trace()
        return


    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4], [0, 0, np.pi / 4]

    def boundary_root(self) -> Object:
        return Shape('basket_boundary_root')

    def cleanup(self) -> None:
        self.target_state_list = []
        self.step_id = 0
        self.t = 0
        self.stage = 'wp0'
        return

from typing import List, Tuple

import numpy as np
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.task import Task
from .reach_single_moving_target_on_the_table import get_state_config, init_target_state, compute_target_position
from pyrep.const import ConfigurationPathAlgorithms as Algos
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import ConfigurationPathError, IKError
import torch
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.backend import sim, utils


def get_expert_info(task, bool_return_path=True):
    # reach -> pre-grasp -> grasp -> lift
    # reach: eepose: target_pose open: 1
    # pre-grasp: eepose: compute_target_position(t+t_delay 1.0s) (z+0.05 m)  rotation: target pose rotation open: 1
    # grasp: eepose: compute_target_position(t+t_delay 0.5s) (z+0.05 m) rotation: target pose rotation open: 0
    # lift: eepose: pick_and_lift_target.get_pose() position  open: 0

    tip_pose = task.robot.arm.get_tip().get_pose()
    w0_pose = task.wp0.get_pose()
    w1_pose = task.wp1.get_pose()
    w2_pose = task.wp2.get_pose()
    w3_pose = task.wp3.get_pose()
    w1_init_pose = task.wp1_init_pose
    w3_init_pose = task.wp3_init_pose
    dist_to_w0 = np.linalg.norm(tip_pose[:3] - w0_pose[:3])
    dist_to_w1 = np.linalg.norm(tip_pose[:3] - w1_pose[:3])
    dist_to_w2 = np.linalg.norm(tip_pose[:3] - w2_pose[:3])
    dist_to_w3 = np.linalg.norm(tip_pose[:3] - w3_pose[:3])
    th_w0 = 0.3  # m
    th_w1 = 0.03  # m
    th_w3 = 0.03  # m
    simulation_timestep = task.pyrep.get_simulation_timestep()
    target_state_dict = task.target_state_list[-1][0]
    hoop_target_state_dict = task.target_state_list[-1][1]
    stage = task.stage
    is_open = all(x > 0.9 for x in task.robot.gripper.get_open_amount())
    if stage == 'wp0' and dist_to_w0 > th_w0:
        stage = 'wp0'
        t_delay = 0.0
        eepose = w0_pose
        open = 1
    elif stage == 'wp0' and dist_to_w0 <= th_w0:
        stage = 'wp1'
        t_delay = 0.5
        eepose = w1_pose
        eepose[:3] = compute_target_position(
            t = task.t+t_delay,
            t0=0,
            x0=w1_init_pose[:3],
            v0=target_state_dict["v"],
            a0=target_state_dict["a"],
            dt=simulation_timestep,
        )
        open = 0
    elif stage == 'wp1' and dist_to_w1 > th_w1:
        stage = 'wp1'
        t_delay = 0.5
        eepose = w1_pose
        eepose[:3] = compute_target_position(
            t = task.t+t_delay,
            t0=0,
            x0=w1_init_pose[:3],
            v0=target_state_dict["v"],
            a0=target_state_dict["a"],
            dt=simulation_timestep,
        )
        open = 0
    elif stage == 'wp1' and dist_to_w1 <= th_w1 and is_open:
        stage = 'wp1'
        t_delay = 0.5
        eepose = w1_pose
        eepose[:3] = compute_target_position(
            t = task.t+t_delay,
            t0=0,
            x0=w1_init_pose[:3],
            v0=target_state_dict["v"],
            a0=target_state_dict["a"],
            dt=simulation_timestep,
        )
        open = 0
    elif stage == 'wp1' and not is_open:
        stage = 'wp2'
        t_delay = 0.5
        eepose = tip_pose
        eepose[2] += 0.3
        open = 0
    elif stage == 'wp2' and dist_to_w3 > th_w3:
        stage = 'wp3'
        t_delay = 0.5
        eepose = w3_pose
        eepose[:3] = compute_target_position(
            t = task.t+t_delay,
            t0=0,
            x0=w3_init_pose[:3],
            v0=hoop_target_state_dict["v"],
            a0=hoop_target_state_dict["a"],
            dt=simulation_timestep,
        )
        open = 0
    elif stage == 'wp3' and dist_to_w3 <= th_w3:
        stage = 'wp3'
        t_delay = 0.5
        eepose = w3_pose
        eepose[:3] = compute_target_position(
            t = task.t+t_delay,
            t0=0,
            x0=w3_init_pose[:3],
            v0=hoop_target_state_dict["v"],
            a0=hoop_target_state_dict["a"],
            dt=simulation_timestep,
        )
        open = 1
    else:
        print("Unrecognized stage: ", stage)
        import pdb; pdb.set_trace()
    print(
        f"stage: {stage}, eepose: {eepose}, open: {open}, dist_to_w0: {dist_to_w0:.2f}, dist_to_w1: {dist_to_w1:.2f}, dist_to_w2: {dist_to_w2:.2f}, dist_to_w3: {dist_to_w3:.2f}")
    output = np.ones((1, 1, 8))
    output[0, 0, :7] = eepose
    output[0, 0, 3:7] = w1_init_pose[3:7]
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

class MovingBasketballInHoop(Task):

    def init_task(self):
        ball = Shape('ball')
        hoop = Shape('basket_ball_hoop_respondable')
        self.register_graspable_objects([ball])
        self.register_success_conditions(
            [DetectedCondition(ball, ProximitySensor('success')),
             NothingGrasped(self.robot.gripper)])
        self.step_id = 0
        self.area = [0, -0.5, 0.8, 0.4, 0.5, 0.8]
        self.t_max = 6.5  # (s)
        self.t = 0
        self.target_state_list = []
        self._bool_expert = True
        self.var2target_state_list = {}
        self.stage = 'wp0'
        self.ball = ball
        self.hoop = hoop
        for var_index in range(self.variation_count()):
            self.var2target_state_list[var_index] = []
            bool_a = get_state_config(var_index)
            # ball target state
            x, v, a = init_target_state(
                t_max=self.t_max,
                area=self.area,
                x_range=self.area,
                v_range=[-0.2, -0.2, 0, 0.2, 0.2, 0],
                a_range=[-0.01, -0.01, 0, 0.01, 0.01, 0],
                x0=None,
                v0=None,
                a0=[0, 0, 0] if not bool_a else None,
                dx=[0.05, 0.05, 0.05],
                dv=[0.025, 0.025, 0.025],
                da=[0.001, 0.001, 0.001],
                min_velo_norm=0.03,
                min_acc_norm=0.01 if bool_a else 0,
            )
            self.var2target_state_list[var_index].append({
                "x": x,
                "v": v,
                "a": a,
            })
            # hoop target state
            x, v, a = init_target_state(
                t_max=self.t_max,
                area=self.area,
                x_range=self.area,
                v_range=[-0.2, -0.2, 0, 0.2, 0.2, 0],
                a_range=[-0.01, -0.01, 0, 0.01, 0.01, 0],
                x0=None,
                v0=None,
                a0=[0, 0, 0] if not bool_a else None,
                dx=[0.05, 0.05, 0.05],
                dv=[0.025, 0.025, 0.025],
                da=[0.001, 0.001, 0.001],
                min_velo_norm=0.03,
                min_acc_norm=0.01 if bool_a else 0,
            )
            self.var2target_state_list[var_index].append({
                "x": x,
                "v": v,
                "a": a,
            })

    def disable_expert_plan(self):
        self._bool_expert = False
        return

    def init_episode(self, index: int) -> List[str]:
        self.var_index = index
        ball_target_state = self.var2target_state_list[self.var_index][0]
        hoop_target_state = self.var2target_state_list[self.var_index][1]
        ball_target_state["x"][-1] = self.ball.get_position()[-1]
        hoop_target_state["x"][-1] = self.hoop.get_position()[-1]
        self.cleanup()
        self.ball.set_position(ball_target_state['x'])
        self.hoop.set_position(hoop_target_state['x'])
        self.target_state_list.append(({
            "x": ball_target_state['x'],
            "v": ball_target_state['v'],
            "a": ball_target_state['a'],
            "t0": 0,
        }, {
            "x": hoop_target_state['x'],
            "v": hoop_target_state['v'],
            "a": hoop_target_state['a'],
            "t0": 0,
        }))
        self.step_id = 0
        self.t = 0
        self.wp0 = Dummy('expertwp0')
        self.wp1 = Dummy('expertwp1')
        self.wp2 = Dummy('expertwp2')
        self.wp3 = Dummy('expertwp3')
        self.wp1_init_pose = self.wp1.get_pose()
        self.wp3_init_pose = self.wp3.get_pose()
        return ['put the ball in the hoop',
                'play basketball',
                'shoot the ball through the net',
                'pick up the basketball and put it in the hoop',
                'throw the basketball through the hoop',
                'place the basket ball through the hoop']

    def check_grasp_success(self):
        grasped_objects = self.robot.gripper.get_grasped_objects()
        return self.ball in grasped_objects
    
    def step(self) -> None:
        simulation_timestep = self.pyrep.get_simulation_timestep()
        ball_target_state_dict = self.target_state_list[-1][0]
        hoop_target_state_dict = self.target_state_list[-1][1]
        if not self.check_grasp_success():
            ball_target_position = compute_target_position(
                t=self.t,
                t0=ball_target_state_dict["t0"],
                x0=ball_target_state_dict["x"],
                v0=ball_target_state_dict["v"],
                a0=ball_target_state_dict["a"],
                dt=simulation_timestep,
            )
            self.ball.set_position(ball_target_position)
        hoop_target_position = compute_target_position(
            t=self.t,
            t0=hoop_target_state_dict["t0"],
            x0=hoop_target_state_dict["x"],
            v0=hoop_target_state_dict["v"],
            a0=hoop_target_state_dict["a"],
            dt=simulation_timestep,
        )
        self.hoop.set_position(hoop_target_position)
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

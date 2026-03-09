from typing import List
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from .reach_single_moving_target_on_the_table_high_speed import get_state_config, compute_target_position, init_target_state, cross_boundary, compute_target_position
from pyrep.const import ConfigurationPathAlgorithms as Algos
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import ConfigurationPathError
from pyrep.objects.dummy import Dummy
import torch
from copy import deepcopy

def get_unoverlapped_frame_dx_dy(n, x_range, y_range):
    min_dist_sq = 0.1  # 平方距离比较，避免开方
    frame_dx_dy = np.empty((n, 2))
    for i in range(n):
        while True:
            frame_dx_dy[i] = [
                np.random.uniform(x_range[0], x_range[1]),
                np.random.uniform(y_range[0], y_range[1]),
            ]
            overlap = False
            for j in range(i):
                if np.linalg.norm(frame_dx_dy[i] - frame_dx_dy[j]) < min_dist_sq:
                    overlap = True
                    break
            if not overlap:
                break
    return frame_dx_dy

def get_expert_info(task, bool_return_path=True):
    tip_pose = task.robot.arm.get_tip().get_pose()
    stage = task.stage
    wp0_pose = deepcopy(task.wp0_init_pose)
    wp1_pose = deepcopy(task.wp1_init_pose)
    wp1_pose[2] -= 0.02
    wp2_pose = deepcopy(task.wp2_init_pose)
    wp3_pose = deepcopy(task.wp3.get_pose())
    wp3_pose[2] -= 0.02

    dist_to_wp0 = np.linalg.norm(tip_pose[:3] - wp0_pose[:3])
    dist_to_wp1 = np.linalg.norm(tip_pose[:3] - wp1_pose[:3])
    dist_to_wp2 = np.linalg.norm(tip_pose[:3] - wp2_pose[:3])
    dist_to_wp3 = np.linalg.norm(tip_pose[:3] - wp3_pose[:3])

    th_wp0 = 0.1
    th_wp1 = 0.01
    th_wp2 = 0.05
    th_wp3 = 0.05
    is_grasping = len(task.robot.gripper.get_grasped_objects()) > 0
    t = task.t
    simulation_timestep = task.pyrep.get_simulation_timestep()
    target_state_dict = task.target_state_list[-1]
    print('---------------------------------')
    print(f'step_id:{task.step_id} ')
    print(f"stage: {stage}, dist_to_wp0: {dist_to_wp0:.2f}, dist_to_wp1: {dist_to_wp1:.2f}, dist_to_wp2: {dist_to_wp2:.2f}, dist_to_wp3: {dist_to_wp3:.2f}")
    print(f"is_grasping: {is_grasping}")
    if stage == 'wp0' and dist_to_wp0 > th_wp0:
        stage = 'wp0'
        t_delay = 0.0
        eepose = wp0_pose
        open = 1
    elif stage == 'wp0' and dist_to_wp0 <= th_wp0:
        stage = 'wp1'
        t_delay = 0
        eepose = wp1_pose
        open = 1
    elif stage == 'wp1' and dist_to_wp1 > th_wp1:
        stage = 'wp1'
        t_delay = 0.0
        eepose = wp1_pose
        open = 1
    elif stage == 'wp1' and dist_to_wp1 <= th_wp1 and not is_grasping:
        stage = 'wp1'
        t_delay = 0.0
        eepose = wp1_pose
        open = 0
    elif stage == 'wp1' and is_grasping:
        stage = 'wp2'
        t_delay = 0.0
        eepose = wp2_pose
        open = 0
    elif stage == 'wp2' and dist_to_wp2 > th_wp2:
        stage = 'wp2'
        t_delay = 0.0
        eepose = wp2_pose
        open = 0
    elif stage == 'wp2' and dist_to_wp2 <= th_wp2:
        stage = 'wp3'
        t_delay = 0.0
        eepose = wp3_pose
        open = 0
    elif stage  in ['wp2', 'wp3'] and dist_to_wp3 > th_wp3:
        stage = 'wp3'
        t_delay = 0.0
        eepose = wp3_pose
        open = 0
    elif stage  == 'wp3' and dist_to_wp3 <= th_wp3:
        stage = 'wp3'
        t_delay = 0.5
        wp3_pred_position = compute_target_position(
            t = t+t_delay,
            t0=target_state_dict["t0"],
            x0=target_state_dict["x"],
            v0=target_state_dict["v"],
            a0=target_state_dict["a"],
            dt=simulation_timestep,
        )
        eepose = wp3_pose.copy()
        eepose[:2] = wp3_pred_position[:2]
        open = 1
    else:
        print("Unrecognized stage: ", stage)
        import pdb; pdb.set_trace()
    print(f"stage: {stage}, eepose: {eepose}, open: {open}")
    output = np.ones((1,1,8))
    output[0,0,:7] = eepose
    output[0,0,7:] = open
    expert_info = {
        "trajectory": torch.from_numpy(output),
        "stage": stage,
        "open": open,
        "debug_info": {
            "tip_cur_position": tip_pose[:3],
            "tar_position": task.bin.get_position(),
            "t": task.t,
        }
    }
    if bool_return_path:
        try:
            path = task.get_path(eepose)
        except:
            path = None
            print(f"path is None")
        expert_info["path"] = path
    return expert_info
class PutRubbishInMovingBin(Task):

    def init_task(self):
        success_sensor = ProximitySensor('success')
        self.rubbish = Shape('rubbish')
        self.bin = Shape('bin')
        self.register_graspable_objects([self.rubbish])
        self.register_success_conditions(
            [DetectedCondition(self.rubbish, success_sensor)])
        self.wp0 = Dummy('waypoint0')
        self.wp1 = Dummy('waypoint1')
        self.wp2 = Dummy('waypoint2')
        self.wp3 = Dummy('waypoint3')
        self.step_id = 0
        self.area = [0.15, -0.5, 0.85, 0.4, 0.5, 0.85]
        self.t_max = 6.5 # (s)
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self.stage = 'wp0'
        self._bool_expert = True
        self.var2target_state_list = {}
        for var_index in range(self.variation_count()):
            self.var2target_state_list[var_index] = []
            bool_a = get_state_config(var_index)
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
            frame_dx_dy = get_unoverlapped_frame_dx_dy(n=3, x_range=[0.05, 0.12], y_range=[-0.4, 0.4])
            tomato1 = Shape('tomato1')
            tomato2 = Shape('tomato2')
            tomato1_position = np.array([frame_dx_dy[0][0], frame_dx_dy[0][1], 0.8])
            tomato2_position = np.array([frame_dx_dy[1][0], frame_dx_dy[1][1], 0.8])
            rubbish_position = np.array([frame_dx_dy[2][0], frame_dx_dy[2][1], 0.8])
            positions = np.array([tomato1_position, tomato2_position, rubbish_position])
            pos = np.random.randint(3)
            self.var2target_state_list[var_index] = {
                "x": x,
                "v": v,
                "a": a,
                "positions": positions,
                "pos": pos,
            }
        return

    def init_episode(self, index: int) -> List[str]:
        tomato1 = Shape('tomato1')
        tomato2 = Shape('tomato2')
        tomato1.set_position(self.var2target_state_list[index]['positions'][0])
        tomato2.set_position(self.var2target_state_list[index]['positions'][1])
        self.rubbish.set_position(self.var2target_state_list[index]['positions'][2])
        x1, y1, z1 = tomato2.get_position()
        x2, y2, z2 = self.rubbish.get_position()
        x3, y3, z3 = tomato1.get_position()
        pos = self.var2target_state_list[index]['pos']

        if pos == 0:
            self.rubbish.set_position([x1, y1, z2])
            tomato2.set_position([x2, y2, z1])
        elif pos == 2:
            self.rubbish.set_position([x3, y3, z2])
            tomato1.set_position([x2, y2, z3])

        if index > 0:
            err_msg = "Error: Only variation0 is supported."
            raise NotImplementedError(err_msg)
        self.var_index = index
        target_state = self.var2target_state_list[self.var_index]
        self.cleanup()
        self.target_state_list.append({
            "x": target_state['x'],
            "v": target_state['v'],
            "a": target_state['a'],
            "t0": 0,
        })
        self.bin.set_position(target_state['x'])
        self.wp0_init_pose = self.wp0.get_pose()
        self.wp1_init_pose = self.wp1.get_pose()
        self.wp2_init_pose = self.wp2.get_pose()

        return ['put rubbish in bin',
                'drop the rubbish into the bin',
                'pick up the rubbish and leave it in the trash can',
                'throw away the trash, leaving any other objects alone',
                'chuck way any rubbish on the table rubbish']

    def step(self) -> None:
        simulation_timestep = self.pyrep.get_simulation_timestep()
        target_state_dict = self.target_state_list[-1]
        target_position, target_velocity = compute_target_position(
            t=self.t,
            t0=target_state_dict["t0"],
            x0=target_state_dict["x"],
            v0=target_state_dict["v"],
            a0=target_state_dict["a"],
            dt=simulation_timestep,            bool_return_velocity=True,
        )
        bool_cross, boundary_index = cross_boundary(target_position, self.bin, self.area)
        if not bool_cross:
            self.bin.set_position(target_position)
        else:
            self.bin.set_position(target_position)
            new_target_velocity = deepcopy(target_velocity)
            for itm in boundary_index:
                new_target_velocity[itm] = - new_target_velocity[itm]
            target_state_dict = {
                "t0": self.t,
                "x": target_position,
                "v": new_target_velocity,
                "a": target_state_dict["a"],
            }
            self.target_state_list.append(target_state_dict)
        if self._bool_expert:
            if self.step_id % 10 == 0:
                self._path, self._open = self.expert_plan()
                self._path_done = False
            if self._path is not None and not self._path_done:
                self._path_done = self._path.step()
            if (self.step_id + 1) % 10 == 0:
                self.move_gripper_tip([self._open])
        self.step_id += 1
        self.t += simulation_timestep
        return
    
    def disable_expert_plan(self):
        self._bool_expert = False
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

    def variation_count(self) -> int:
        return 2

    def cleanup(self) -> None:
        self.target_state_list = []
        self.step_id = 0
        self.t = 0
        self.stage = 'wp0'
        return

    def is_static_workspace(self) -> bool:
        """Specify if the task should'nt be randomly placed in the workspace.

        :return: True if the task pose should not be sampled.
        """
        return True
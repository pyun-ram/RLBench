from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors
import torch
from rlbench.backend.exceptions import InvalidActionError
from .reach_single_moving_target_on_the_table_high_speed import get_state_config, compute_target_position, init_target_state, cross_boundary, handle_boundary
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from copy import deepcopy


def get_expert_info(task, th_reach=0.4, th_pre_grasp=0.2, bool_return_path=True):
        # reach -> pre-grasp -> grasp -> lift
        # reach: eepose: target_pose open: 1
        # pre-grasp: eepose: compute_target_position(t+t_delay 1.0s) (z+0.05 m)  rotation: target pose rotation open: 1
        # grasp: eepose: compute_target_position(t+t_delay 0.5s) (z+0.05 m) rotation: target pose rotation open: 0
        # lift: eepose: pick_and_lift_target.get_pose() position  open: 0
        target_pose = task.target_block.get_pose()
        tip_pose = task.robot.arm.get_tip().get_pose()
        dist = np.linalg.norm(target_pose[:3] - tip_pose[:3])
        th_reach = 0.4 # m
        th_pre_grasp = 0.2 # m
        bool_grasp_succ = task.check_grasp_success()
        t = task.t
        simulation_timestep = task.pyrep.get_simulation_timestep()
        target_state_dict = task.target_state_list[-1]
        
        if dist > th_reach:
            stage = 'reach'
            t_delay = 0.0 # s
        elif dist > th_pre_grasp:
            stage = 'pre-grasp'
            t_delay = 0.5 # s
        elif dist <= th_pre_grasp and not bool_grasp_succ:
            stage = 'grasp'
            t_delay = 0.5 # s
        elif bool_grasp_succ:
            stage = 'lift'
            t_delay = 0.0 # s
        else:
            stage = 'pre-grasp'
            t_delay = 0.5
            
        if stage == 'reach':
            eepose = target_pose
            open = 1 # open
        elif stage == 'pre-grasp':
            predicted_position = compute_target_position(
                t = t+t_delay,
                t0=target_state_dict["t0"],
                x0=target_state_dict["x"],
                v0=target_state_dict["v"],
                a0=target_state_dict["a"],
                dt=simulation_timestep,
            )
            # 构造eepose: position + rotation from target_pose; open = 1
            eepose = np.copy(target_pose)
            eepose[0:3] = predicted_position
            open = 1
            eepose = handle_boundary(eepose, task.area)
        elif stage == 'grasp':
            predicted_position = compute_target_position(
                t = t+t_delay,
                t0=target_state_dict["t0"],
                x0=target_state_dict["x"],
                v0=target_state_dict["v"],
                a0=target_state_dict["a"],
                dt=simulation_timestep,
            )
            # 构造eepose: position (z-0.02m) + rotation from target_pose; open = 0
            eepose = np.copy(target_pose)
            eepose[0:3] = predicted_position
            eepose[2] -= 0.015  # z-0.02 m
            eepose[3:7] = tip_pose[3:7]
            open = 0
            eepose = handle_boundary(eepose, task.area)
        elif stage == 'lift':
            # lift: 使用目标块的当前位置，保持夹爪关闭
            succ_position = task.success_detector.get_position()
            eepose = task.robot.arm.get_tip().get_pose()
            eepose[0:3] = succ_position
            eepose[3:7] = [0,1,0,0]
            open = 0
        # eepose = task.robot.arm.get_tip().get_pose()
        print(f"stage: {stage}, eepose: {eepose}, open: {open}, dist: {dist}")
        output = np.ones((1,1,8))
        output[0,0,:7] = eepose
        output[0,0,7:] = open
        expert_info = {
            "trajectory": torch.from_numpy(output),
            "stage": stage,
            "open": open,
            "debug_info": {
                "tip_cur_position": tip_pose[:3],
                "tar_position": target_pose[:3],
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

class PickMovingTargetOnTheTable(Task):

    def init_task(self) -> None:
        # This task does not use waypoints, so we need to remove them in case of failure in reset the environment.
        try:
            for i in range(4):
                waypoint = Dummy('waypoint%d' % i)
                waypoint.remove()
        except:
            print(f"Waypoints not found, will not remove them.")
        self.target_block = Shape('pick_and_lift_target')
        self.distractors = [
            Shape('stack_blocks_distractor%d' % i)
            for i in range(2)]
        self.register_graspable_objects([self.target_block])
        self.boundary = SpawnBoundary([Shape('pick_and_lift_boundary')])
        self.success_detector = ProximitySensor('pick_and_lift_success')

        cond_set = ConditionSet([
            GraspedCondition(self.robot.gripper, self.target_block),
            DetectedCondition(self.target_block, self.success_detector)
        ])
        self.register_success_conditions([cond_set])
        self.step_id = 0
        target_size_xy = [0.10, 0.10]
        self.area = [0, -0.5, 0.8, 0.4, 0.5, 0.8]
        self.area = [
            self.area[0]+target_size_xy[0]/2,
            self.area[1]+target_size_xy[1]/2,
            self.area[2],
            self.area[3]-target_size_xy[0]/2,
            self.area[4]-target_size_xy[1]/2,
            self.area[5],
        ]
        self.t_max = 0.5 # (s)
        self.t = 0
        self.target_state_list = []
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
            color_choices = np.random.choice(list(range(1, len(colors))), size=2, replace=False)
            self.boundary.clear()
            self.boundary.sample(
                self.success_detector, min_rotation=(0.0, 0.0, 0.0),
                max_rotation=(0.0, 0.0, 0.0))
            for block in self.distractors:
                self.boundary.sample(block, min_distance=0.1)
            self.var2target_state_list[var_index] = {
                "x": x,
                "v": v,   
                "a": a,
                "color_choices": color_choices,
                'success_detector_pose': self.success_detector.get_pose(),
                'distractors_poses': [block.get_pose() for block in self.distractors],
            }
        # import pickle
        # with open('var2target_state_list.pkl', 'wb') as f:
        #     pickle.dump(self.var2target_state_list, f)
        # with open('var2target_state_list.pkl', 'rb') as f:
        #     self.var2target_state_list = pickle.load(f)
        return

    def init_episode(self, index: int) -> List[str]:
        for i in range(len(self.var2target_state_list[index]["distractors_poses"])):
            self.var2target_state_list[index]["distractors_poses"][i][2] = 0.77
        block_color_name, block_rgb = colors[0]
        self.target_block.set_color(block_rgb)
        color_choices = self.var2target_state_list[index]["color_choices"]
        for i, ob in enumerate(self.distractors):
            name, rgb = colors[color_choices[int(i)]]
            ob.set_color(rgb)
        self.success_detector.set_pose(self.var2target_state_list[index]["success_detector_pose"])
        for block in self.distractors:
            block.set_pose(self.var2target_state_list[index]["distractors_poses"][i])
        self.var_index = index
        target_state = self.var2target_state_list[self.var_index]
        target_state["x"][-1] = self.target_block.get_position()[-1]
        self.cleanup()
        self.target_state_list.append({
            "x": target_state['x'],
            "v": target_state['v'],
            "a": target_state['a'],
            "t0": 0,
        })
        self.target_block.set_position(target_state['x'])
        self.step_id = 0
        self.t = 0
        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def step(self) -> None:
        simulation_timestep = self.pyrep.get_simulation_timestep()
        if not self.check_grasp_success():
            target_state_dict = self.target_state_list[-1]
            target_position, target_velocity = compute_target_position(
                t=self.t,
                t0=target_state_dict["t0"],
                x0=target_state_dict["x"],
                v0=target_state_dict["v"],
                a0=target_state_dict["a"],
                dt=simulation_timestep,
                bool_return_velocity=True,
            )
            bool_cross, boundary_index = cross_boundary(target_position, self.target_block, self.area)
            if not bool_cross:
                self.target_block.set_position(target_position)
            else:
                self.target_block.set_position(target_position)
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
    
    def check_grasp_success(self):
        grasped_objects = self.robot.gripper.get_grasped_objects()
        return self.target_block in grasped_objects
    
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
        return

    def is_static_workspace(self):
        return True
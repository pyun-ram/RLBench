from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, ConditionSet, \
    GraspedCondition
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.const import colors
from rlbench.backend.waypoints import Waypoint
from rlbench.backend.exceptions import InvalidActionError
from .reach_single_moving_target_on_the_table import get_state_config, init_target_state, compute_target_position
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError

class PickMovingTargetOnTheTable(Task):

    def init_task(self) -> None:
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
        self.area = [0, -0.5, 0.8, 0.4, 0.5, 0.8]
        self.t_max = 20 # (s)
        self.step_id = 0
        self.t = 0
        self.target_state_list = []

    def init_episode(self, index: int) -> List[str]:

        block_color_name, block_rgb = colors[index]
        self.target_block.set_color(block_rgb)

        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        for i, ob in enumerate(self.distractors):
            name, rgb = colors[color_choices[int(i)]]
            ob.set_color(rgb)

        self.boundary.clear()
        self.boundary.sample(
            self.success_detector, min_rotation=(0.0, 0.0, 0.0),
            max_rotation=(0.0, 0.0, 0.0))
        for block in [self.target_block] + self.distractors:
            self.boundary.sample(block, min_distance=0.1)

        var_index = index
        bool_a = get_state_config(var_index)
        x, v, a = init_target_state(
            self.area,
            self.t_max,
            x0=None,
            v0=None,
            a0=[0, 0, 0] if not bool_a else None,
        )
        print(v, np.linalg.norm(v))
        # save target_state
        self.target_state_list.append({
            "x": x,
            "v": v,
            "a": a,
            "t0": 0,
        })
        self.step_id = 0
        self.t = 0
        self.target_block.set_position(x)

        return ['pick up the %s block and lift it up to the target' %
                block_color_name,
                'grasp the %s block to the target' % block_color_name,
                'lift the %s block up to the target' % block_color_name]

    def _get_waypoints(self, validating=False) -> List[Waypoint]:
        return []

    def step(self) -> None:
        simulation_timestep = self.pyrep.get_simulation_timestep()
        if not self.check_grasp_success():
            target_state_dict = self.target_state_list[-1]
            target_position = compute_target_position(
                t=self.t,
                t0=target_state_dict["t0"],
                x0=target_state_dict["x"],
                v0=target_state_dict["v"],
                a0=target_state_dict["a"],
                dt=simulation_timestep,
            )
            self.target_block.set_position(target_position)

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
        # reach -> pre-grasp -> grasp -> lift
        # reach: eepose: target_pose open: 1
        # pre-grasp: eepose: compute_target_position(t+t_delay 1.0s) (z+0.05 m)  rotation: target pose rotation open: 1
        # grasp: eepose: compute_target_position(t+t_delay 0.5s) (z+0.05 m) rotation: target pose rotation open: 0
        # lift: eepose: pick_and_lift_target.get_pose() position  open: 0
        target_pose = self.target_block.get_pose()
        tip_pose = self.robot.arm.get_tip().get_pose()
        dist = np.linalg.norm(target_pose[:3] - tip_pose[:3])
        th_reach = 0.4 # m
        th_pre_grasp = 0.2 # m
        bool_grasp_succ = self.check_grasp_success()
        t = self.t
        simulation_timestep = self.pyrep.get_simulation_timestep()
        target_state_dict = self.target_state_list[-1]
        
        if dist > th_reach:
            stage = 'reach'
            t_delay = 0.0 # s
        elif dist > th_pre_grasp:
            stage = 'pre-grasp'
            t_delay = 1.0 # s
        elif dist <= th_pre_grasp and not bool_grasp_succ:
            stage = 'grasp'
            t_delay = 0.5 # s
        elif bool_grasp_succ:
            stage = 'lift'
            t_delay = 0.0 # s
        else:
            stage = 'pre-grasp'
            t_delay = 1.0
            
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
            eepose[2] -= 0.02  # z-0.02 m
            open = 0
        elif stage == 'lift':
            # lift: 使用目标块的当前位置，保持夹爪关闭
            succ_position = self.success_detector.get_position()
            eepose = self.robot.arm.get_tip().get_pose()
            eepose[0:3] = succ_position
            eepose[3:7] = [0,1,0,0]
            open = 0
        
        print(f"stage: {stage}, eepose: {eepose}, open: {open}, dist: {dist}")
        path = self.get_path(eepose)
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
        return len(colors)

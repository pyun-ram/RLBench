from rlbench.backend.task import Task
from typing import List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.conditions import GraspedCondition, DetectedCondition
from .reach_single_moving_target_on_the_table import get_state_config, init_target_state, compute_target_position
import numpy as np
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
from pyrep.objects.dummy import Dummy

class ScoopMovingTargetOnTheTable(Task):

    def init_task(self) -> None:
        spatula = Shape('scoop_with_spatula_spatula')
        self.register_graspable_objects([spatula])
        self.register_success_conditions([
            DetectedCondition(Shape('Cuboid'), ProximitySensor('success')),
            GraspedCondition(self.robot.gripper, spatula)
        ])
        self.target_block = Shape('Cuboid')
        self.step_id = 0
        self.area = [0, -0.5, 0.8, 0.4, 0.5, 0.8]
        self.t_max = 20 # (s)
        self.t = 0
        self.target_state_list = []
        self.stage = 'wp0'
        self.wp0 = Dummy('waypoint0')
        self.wp1 = Dummy('waypoint1')
        self.wp2 = Dummy('waypoint2')
        self.wp3 = Dummy('waypoint3')

    def init_episode(self, index: int) -> List[str]:
        var_index = 0
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
        self.wp2_init_pose = self.wp2.get_pose()

        return ['scoop up the cube and lift it with the spatula',
                'scoop up the block and lift it with the spatula',
                'use the spatula to scoop the cube and lift it',
                'use the spatula to scoop the block and lift it',
                'pick up the cube using the spatula',
                'pick up the block using the spatula']

    def check_scoop_success(self):
        # pos = self.target_block.get_position()
        # return pos[2] > 0.9
        return False
    
    def step(self) -> None:
        simulation_timestep = self.pyrep.get_simulation_timestep()
        target_state_dict = self.target_state_list[-1]
        if not self.check_scoop_success():
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
        tip_pose = self.robot.arm.get_tip().get_pose()
        stage = self.stage
        w0_pose = self.wp0.get_pose()
        w1_pose = self.wp1.get_pose()
        w2_pose = self.wp2.get_pose()
        dist_to_wp0 = np.linalg.norm(w0_pose[:3] - tip_pose[:3])
        dist_to_wp1 = np.linalg.norm(w1_pose[:3] - tip_pose[:3])
        dist_to_wp2 = np.linalg.norm(w2_pose[:3] - tip_pose[:3])
        th_w0 = 0.1 # m
        th_w1 = 0.05 # m
        th_w2 = 0.03 # m
        simulation_timestep = self.pyrep.get_simulation_timestep()
        is_grasping = len(self.robot.gripper.get_grasped_objects()) > 0
        target_state_dict = self.target_state_list[-1]
        print('---------------------------------')
        print(f'step: {self.step_id}')
        print(f"stage: {stage}, dist_to_wp0: {dist_to_wp0:.2f}, dist_to_wp1: {dist_to_wp1:.2f}, dist_to_wp2: {dist_to_wp2:.2f}, is_grasping: {is_grasping}")
        if stage == 'wp0' and dist_to_wp0 > th_w0:
            stage = 'wp0'
            t_delay = 0.0
            eepose = w0_pose
            open = 1
        elif stage == 'wp0' and dist_to_wp0 <= th_w0:
            stage = 'wp1'
            eepose = w1_pose
            open = 1
        elif stage == 'wp1' and dist_to_wp1 > th_w1:
            stage = 'wp1'
            t_delay = 0.0
            eepose = w1_pose
            open = 1
        elif stage == 'wp1' and dist_to_wp1 <= th_w1 and not is_grasping:
            stage = 'wp1'
            t_delay = 0.0
            eepose = w1_pose
            open = 0
        elif stage == 'wp1' and dist_to_wp1 <= th_w1 and is_grasping:
            stage = 'wp2'
            t_delay = 1.0
            eepose_position = compute_target_position(
                t=self.t+t_delay,
                t0=target_state_dict["t0"],
                x0=self.wp2_init_pose[:3],
                v0=target_state_dict["v"],
                a0=target_state_dict["a"],
                dt=simulation_timestep,
            )
            eepose = w2_pose
            eepose[:3] = eepose_position
            open = 0
        elif stage == 'wp2' and dist_to_wp2 > th_w2:
            stage = 'wp2'
            t_delay = 0.0
            eepose_position = compute_target_position(
                t=self.t+t_delay,
                t0=target_state_dict["t0"],
                x0=self.wp2_init_pose[:3],
                v0=target_state_dict["v"],
                a0=target_state_dict["a"],
                dt=simulation_timestep,
            )
            eepose = w2_pose
            eepose[:3] = eepose_position
            open = 0
        elif stage == 'wp2' and dist_to_wp2 <= th_w2:
            stage = 'wp3'
            t_dellay = 0.0
            eepose = tip_pose
            eepose[2] += 0.1
            open = 0
        elif stage == 'wp3':
            stage = 'wp0'
            t_delay = 0.0
            eepose = w0_pose
            open = 1
        else:
            print("Unrecognized stage: ", stage)
            import pdb; pdb.set_trace()
        print(f"stage: {stage}, eepose: {eepose}, open: {open}")
        path = self.get_path(eepose)
        self.stage = stage
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

    def cleanup(self) -> None:
        self.target_state_list = []
        self.t = 0
        self.step_id = 0
        self.stage = 'wp0'
        return

    def variation_count(self) -> int:
        return 1

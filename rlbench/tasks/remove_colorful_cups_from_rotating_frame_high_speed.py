from typing import List, Tuple
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, OrConditions
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
import numpy as np
from .place_cups_on_rotating_frame import init_target_state
import torch

MAX_CUPS_TO_REMOVE = 2

def get_expert_info(task, bool_return_path=True):
    tip_pose = task.robot.arm.get_tip().get_pose()
    target_cup = min(
        task.cups,
        key=lambda cup: np.linalg.norm(cup.get_position() - tip_pose[:3]))
    target_spoke = min(
        task.spokes,
        key=lambda spoke: np.linalg.norm(spoke.get_position() - target_cup.get_position()))

    # Save original parent of w1 and w5
    org_w1_parent = task.w1.get_parent()
    org_w2_parent = task.w2.get_parent()

    # self.w1.set_parent(target_cup)
    task.w1.set_position(task.w1_rel_pos, relative_to=target_cup, reset_dynamics=False)
    task.w1.set_orientation(task.w1_rel_ori, relative_to=target_cup, reset_dynamics=False)
    # self.w2.set_parent(target_spoke)
    task.w2.set_position(task.w2_rel_pos, relative_to=target_spoke, reset_dynamics=False)
    task.w2.set_orientation(task.w2_rel_ori, relative_to=target_spoke, reset_dynamics=False)
    wp0_pose = task.w0.get_pose()
    wp1_pose = task.w1.get_pose()
    wp2_pose = task.w2.get_pose()
    wp3_pose = task.w3.get_pose()
    wp4_pose = task.w4.get_pose()
    wp5_pose = task.w5.get_pose()
    dist_to_wp0 = np.linalg.norm(tip_pose[:3] - wp0_pose[:3])
    dist_to_wp1 = np.linalg.norm(tip_pose[:3] - wp1_pose[:3])
    dist_to_wp2 = np.linalg.norm(tip_pose[:3] - wp2_pose[:3])
    dist_to_wp3 = np.linalg.norm(tip_pose[:3] - wp3_pose[:3])
    dist_to_wp4 = np.linalg.norm(tip_pose[:3] - wp4_pose[:3])
    dist_to_wp5 = np.linalg.norm(tip_pose[:3] - wp5_pose[:3])
    th_w0 = 0.1
    th_w1 = 0.05
    th_w2 = 0.05
    th_w3 = 0.025
    th_w4 = 0.05
    th_w5 = 0.03
    is_grasping = len(task.robot.gripper.get_grasped_objects()) > 0

    from .place_cups_on_rotating_frame import compute_target_pose
    stage = task.stage
    print('---------------------------------')
    print(task.step_id)
    print(f"stage: {stage}, dist_to_wp0: {dist_to_wp0}, dist_to_wp1: {dist_to_wp1}, dist_to_wp2: {dist_to_wp2}, dist_to_wp3: {dist_to_wp3}, dist_to_wp4: {dist_to_wp4}, dist_to_wp5: {dist_to_wp5}, is_grasping: {is_grasping}")
    if stage == 'wp0' and dist_to_wp0 > th_w0:
        stage = 'wp0'
        t_delay = 0.0
        eepose = wp0_pose
        open = 1
    elif stage == 'wp0' and dist_to_wp0 <= th_w0:
        stage = 'wp1'
        t_delay = 1.0
        eepose = compute_target_pose(
            task._frame_base,
            task.w1,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 1
    elif stage == 'wp1' and dist_to_wp1 > th_w1:
        stage = 'wp1'
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w1,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 1
    elif stage == 'wp1' and dist_to_wp1 <= th_w1:
        stage = 'wp2'
        t_delay = 0.0
        eepose = compute_target_pose(
            task._frame_base,
            task.w2,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 0
    elif stage == 'wp2' and not is_grasping:
        stage = 'wp2'
        t_delay = 0.0
        eepose = compute_target_pose(
            task._frame_base,
            task.w2,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 0
    elif stage == 'wp2' and is_grasping:
        stage = 'wp3'
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w3,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 0
    elif stage == 'wp3' and dist_to_wp3 > th_w3:
        stage = 'wp3'
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w3,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 0
    elif stage == 'wp3' and dist_to_wp3 <= th_w3:
        stage = 'wp4'
        t_delay = 0
        eepose = wp4_pose
        open = 0
    elif stage == 'wp4' and dist_to_wp4 > th_w4:
        stage = 'wp4'
        t_delay = 0
        eepose = wp4_pose
        open = 0
    elif stage == 'wp4' and dist_to_wp4 <= th_w4:
        stage = 'wp5'
        t_delay = 0
        eepose = wp5_pose
        open = 0
    elif stage == 'wp5' and dist_to_wp5 > th_w5:
        stage = 'wp5'
        t_delay = 0
        eepose = wp5_pose
        open = 0
    elif stage == 'wp5' and dist_to_wp5 <= th_w5:
        stage = 'wp5'
        t_delay = 0
        eepose = wp5_pose
        open = 1
        task.cups_removed += 1
    else:
        print("Unrecognized stage: ", stage)
        import pdb; pdb.set_trace()
    
    print(f"stage: {stage}, eepose: {eepose}, open: {open}")
    print('---------------------------------')
    output = np.ones((1,1,8))
    output[0,0,:7] = eepose
    output[0,0,7:] = open
    expert_info = {
        "trajectory": torch.from_numpy(output),
        "stage": stage,
        "open": open,
        "debug_info": {
            "tip_cur_position": tip_pose[:3],
            "tar_position": target_cup.get_position(),
            "t": task.t,
        }
    }
    if bool_return_path:
        path = task.get_path(eepose)
        expert_info["path"] = path

    # Restore original parent of w1 and w2
    # self.w1.set_parent(target_cup)
    task.w1.set_position(task.w1_rel_pos, relative_to=org_w1_parent, reset_dynamics=False)
    task.w1.set_orientation(task.w1_rel_ori, relative_to=org_w1_parent, reset_dynamics=False)
    # self.w2.set_parent(target_spoke)
    task.w2.set_position(task.w2_rel_pos, relative_to=org_w2_parent, reset_dynamics=False)
    task.w2.set_orientation(task.w2_rel_ori, relative_to=org_w2_parent, reset_dynamics=False)
    return expert_info
class RemoveColorfulCupsFromRotatingFrameHighSpeed(Task):

    def init_task(self) -> None:
        self.cups_removed = -1
        self.cups = [Shape('mug%d' % i) for i in range(3)]
        self.spokes = [Shape('place_cups_holder_spoke%d' % i) for i in range(3)]
        self.holder = Shape('place_cups_holder_base')
        self.holder_boundary = Shape('tree_boundary')
        self.register_graspable_objects(self.cups)
        self.success_detectors = [ProximitySensor('success_detector%d' % i)
                                  for i in range(3)]
        self.w0 = Dummy('waypoint0')
        self.w3 = Dummy('waypoint3')
        self.w4 = Dummy('waypoint4')
        self.w1 = Dummy('waypoint1')
        self.w1_rel_pos = self.w1.get_position(relative_to=self.cups[0])
        self.w1_rel_ori = self.w1.get_orientation(relative_to=self.cups[0])
        self.w2 = Dummy('waypoint2')
        self.w2_rel_pos = self.w2.get_position(relative_to=self.spokes[0])
        self.w2_rel_ori = self.w2.get_orientation(relative_to=self.spokes[0])
        self.w5 = Dummy('waypoint5')
        self.w5_rel_pos = self.w5.get_orientation(
            relative_to=self.success_detectors[0])
        self.w5_new_pos = self.w5.get_position()
        self.w5_new_pos_saved = self.w5_new_pos
        self.success_conditions = [NothingGrasped(self.robot.gripper)]
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self.stage = 'wp0'
        self._bool_expert = True
        self.var2target_state_list = {}
        self._frame_base = Shape('place_cups_holder_base')
        for var_index in range(self.variation_count()):
            yaw_speed = [init_target_state(
                min_yaw=30, # 5 degree/s
                max_yaw=50, # 15 degree/s
                d_yaw=0.5,
            )]
            direction = np.random.choice([-1, 1])
            yaw_speed[0] = yaw_speed[0] * direction
            frame_dx_dy = np.random.uniform([-0.05, -0.05], [0.05, 0.05], size=2)
            frame_position = self._frame_base.get_position() + [frame_dx_dy[0], frame_dx_dy[1], 0]
            self.var2target_state_list[var_index] = {
                "yaw_speed": yaw_speed,
                "frame_position": frame_position,
            }
            
        return

    def init_episode(self, index: int) -> List[str]:
        self._frame_base.set_position(
            self.var2target_state_list[index]['frame_position']
        )
        self.cups_removed = -1
        self.cups_to_remove = 1 + index % MAX_CUPS_TO_REMOVE
        self.w5_new_pos = self.w5_new_pos_saved
        self.w1.set_position(self.w1_rel_pos,
                             relative_to=self.cups[0],
                             reset_dynamics=False)
        # for i in range(self.cups_to_remove):
        #     self.success_conditions.append(
        #         DetectedCondition(self.cups[i], self.success_detectors[i])
        #     )
        success_detectors = [
            ProximitySensor('success_detector%d' % i) for i in range(3)]
        self._on_table_conditions = OrConditions([OrConditions([
            DetectedCondition(self.cups[ci], success_detectors[sdi]) for sdi in
            range(3)]) for ci in range(3)])
        self.success_conditions.append(self._on_table_conditions)
        self.register_success_conditions(self.success_conditions)
        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)

        if index > 0:
            err_msg = "Error: Only variation0 is supported."
            raise NotImplementedError(err_msg)
        self.var_index = index
        self.cleanup()
        self.yaw_speed = self.var2target_state_list[index]['yaw_speed']
        self.target_state_list.append({
            "yaw_speed": self.yaw_speed,
            "t0": 0,
        })
        # init_joint_angles = np.array([
        # 0.884812593460083, 0.5623087882995605, -0.86405348777771,
        # -1.9321348667144775,0.1937483549118042, 2.1352787017822266, 2.0415501594543457])
        # self.robot.arm.set_joint_positions(init_joint_angles, disable_dynamics=True)
        if self.cups_to_remove == 1:
            return ['remove 1 cup from the high speed cup holder and place it on the '
                    'table',
                    'remove one cup from the high speed mug holder',
                    'pick up 1 cup from the high speed mug tree and place it on the table',
                    'slide 1 mug off of its spoke on the high speed cup holder and leave '
                    'it on the table top']
        else:
            return ['remove %d cups from the high speed cup holder and place it on the '
                    'table' % self.cups_to_remove,
                    'remove %d cups from the high speed cup holder' % self.cups_to_remove,
                    'pick up %d cups from the high speed mug tree and place them on the '
                    'table' % self.cups_to_remove,
                    'slide %d mugs off of their spokes on the high speed cup holder and '
                    'leave them on the high speed table top' % self.cups_to_remove]

    def step(self) -> None:
        self.yaw_speed = self.target_state_list[-1]['yaw_speed']
        simulation_timestep = self.pyrep.get_simulation_timestep()
        rot_speed = np.deg2rad(self.yaw_speed) * simulation_timestep
        self._frame_base.rotate([0, 0, rot_speed])
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

    def variation_count(self) -> int:
        return 1

    def _move_above_next_target(self, waypoint):
        if self.cups_removed > self.cups_to_remove:
            raise RuntimeError('Should not be here, all cups should have been '
                               'removed')
        move_index = self.cups_removed if self.cups_removed > -1 else 0
        next_move_index = self.cups_removed + 1 if self.cups_removed > -1 else 0
        if self.cups_removed > -1:
            self.w1.set_position(self.w1_rel_pos,
                                 relative_to=self.cups[next_move_index],
                                 reset_dynamics=False
                                 )
            self.w1.set_orientation(self.w1_rel_ori,
                                    relative_to=self.cups[next_move_index],
                                    reset_dynamics=False
                                    )
            self.w2.set_position(self.w2_rel_pos,
                                 relative_to=self.spokes[next_move_index],
                                 reset_dynamics=False)
            self.w2.set_orientation(self.w2_rel_ori,
                                    relative_to=self.spokes[next_move_index],
                                    reset_dynamics=False
                                    )
            new_x, new_y, _ = self.success_detectors[
                next_move_index].get_position()
            self.w5_new_pos[0] = new_x
            self.w5_new_pos[1] = new_y
            self.w5.set_position(self.w5_new_pos,
                                 reset_dynamics=False)

        ######
        self.cups_removed += 1
        ######

    def cleanup(self) -> None:
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self.cups_removed = -1
        self.stage = 'wp0'
        return

    def _repeat(self):
        return self.cups_removed < self.cups_to_remove - 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self):
        return True

    def set_target_color(self, color: List[List[float]]):
        for i, cup in enumerate([Shape('mug_visual%d' % i) for i in range(4)]):
            cup.set_color(color[i])
        return
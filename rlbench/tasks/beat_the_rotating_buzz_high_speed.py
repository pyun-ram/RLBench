from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from .place_cups_on_rotating_frame import init_target_state, compute_target_pose
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
import numpy as np
import torch

def get_expert_info(task, bool_return_path=True):
    stage = task.stage
    w0_pose = task.w0.get_pose()
    w1_pose = task.w1.get_pose()
    w2_pose = task.w2.get_pose()
    w3_pose = task.w3.get_pose()
    w3_pose_ = task.w3_.get_pose()
    tip_pose = task.robot.arm.get_tip().get_pose()
    dist_to_wp0 = np.linalg.norm(tip_pose[:3] - w0_pose[:3])
    dist_to_wp1 = np.linalg.norm(tip_pose[:3] - w1_pose[:3])
    dist_to_wp2 = np.linalg.norm(tip_pose[:3] - w2_pose[:3])
    dist_to_wp3 = np.linalg.norm(tip_pose[:3] - w3_pose[:3])
    dist_to_wp3_ = np.linalg.norm(tip_pose[:3] - w3_pose_[:3])
    th_wp0 = 0.1
    th_wp1 = 0.03
    th_wp2 = 0.03
    th_wp3_ = 0.03
    th_wp3 = 0.03
    is_grasping = len(task.robot.gripper.get_grasped_objects()) > 0
    print('---------------------------------')
    print(f"stage: {stage}, is_grasping: {is_grasping}, dist_to_wp0: {dist_to_wp0:.2f}, dist_to_wp1: {dist_to_wp1:.2f}, dist_to_wp2: {dist_to_wp2:.2f}, dist_to_wp3_: {dist_to_wp3_:.2f}, dist_to_wp3: {dist_to_wp3:.2f}")
    tip_pose = task.robot.arm.get_tip().get_pose()
    if stage == 'wp0' and dist_to_wp0 > th_wp0:
        stage = 'wp0'
        open = 1
        t_delay = 0.0
        eepose = w0_pose
    elif stage == 'wp0' and dist_to_wp0 <= th_wp0:
        stage = 'wp1'
        open = 1
        t_delay = 0.0
        eepose = w1_pose
    elif stage == 'wp1' and dist_to_wp1 > th_wp1:
        stage = 'wp1'
        open = 1
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w1,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    elif stage == 'wp1' and dist_to_wp1 <= th_wp1 and not is_grasping:
        stage = 'wp1'
        open = 0
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w1,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    elif stage == 'wp1' and dist_to_wp1 <= th_wp1 and is_grasping:
        stage = 'wp2'
        open = 0
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w2,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    elif stage == 'wp2' and dist_to_wp2 > th_wp2:
        stage = 'wp2'
        open = 0
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w2,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    elif stage == 'wp2' and dist_to_wp2 <= th_wp2:
        stage = 'wp3_'
        open = 0
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w3_,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    elif stage == 'wp3_' and dist_to_wp3_ > th_wp3_:
        stage = 'wp3_'
        open = 0
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w3_,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    elif stage == 'wp3_' and dist_to_wp3_ <= th_wp3_:
        stage = 'wp3'
        open = 0
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w3,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    elif stage == 'wp3' and dist_to_wp3 > th_wp3:
        stage = 'wp3'
        open = 0
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w3,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    elif stage == 'wp3' and dist_to_wp3 <= th_wp3:
        stage = 'wp3'
        open = 1
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w3,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
    else:
        print("Unrecognized stage: ", stage)
        import pdb; pdb.set_trace()

    print(f"stage: {stage}, eepose: {eepose}, open: {open}, dist_to_wp0: {dist_to_wp0:.2f}, dist_to_wp1: {dist_to_wp1:.2f}, dist_to_wp2: {dist_to_wp2:.2f}, dist_to_wp3_: {dist_to_wp3_:.2f}, dist_to_wp3: {dist_to_wp3:.2f}")
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
            "tar_position": task.wand.get_position(),
            "t": task.t,
        }
    }
    if bool_return_path:
        try:
            path = task.get_path(eepose)
        except:
            path = None
        expert_info["path"] = path
    return expert_info


class BeatTheRotatingBuzzHighSpeed(Task):

    def init_task(self) -> None:
        middle_sensor = ProximitySensor('middle_sensor')
        right_sensor = ProximitySensor('right_sensor')
        wand = Shape('wand')
        self.register_graspable_objects([wand])
        self.register_fail_conditions(
            [DetectedCondition(wand, middle_sensor)])
        self.register_success_conditions(
            [DetectedCondition(wand, right_sensor),
             NothingGrasped(self.robot.gripper)])
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self.stage = 'wp0'
        self._bool_expert = True
        self.var2target_state_list = {}
        self._frame_base = Shape('Cuboid')
        frame_base_z = self._frame_base.get_position()[2]
        self.wand = wand
        self.area = [0.1, -0.4, frame_base_z, 0.3, 0.4, frame_base_z]
        for var_index in range(self.variation_count()):
            yaw_speed = init_target_state(
                min_yaw=20,  # 5 degree/s
                max_yaw=25,  # 15 degree/s
                d_yaw=1,
            )
            frame_position = np.random.uniform(self.area[:3], self.area[3:])
            self.var2target_state_list[var_index] = {
                'yaw_speed': yaw_speed,
                'frame_position': frame_position,
            }
        return

    def init_episode(self, index: int) -> List[str]:
        self.var_index = index
        self.cleanup()
        self.yaw_speed = self.var2target_state_list[index]['yaw_speed']
        self.target_state_list.append({
            'yaw_speed': self.yaw_speed,
            't0': 0,
        })
        self._frame_base.set_position(self.var2target_state_list[index]['frame_position'])
        self.step_id = 0
        self.t = 0
        self.stage = 'wp0'
        self.w0 = Dummy('waypoint0')
        self.w1 = Dummy('waypoint1')
        self.w2 = Dummy('waypoint2')
        self.w3 = Dummy('waypoint3')
        self.w3_ = Dummy('waypoint3_')
        return ['beat the high speed rotating buzz',
                'slide the ring along the high speed rotating pole without allowing them to touch',
                'slide the ring along the high speed rotating metal pole, avoiding contact between '
                'them',
                'slide the ring from one end of the high speed rotating pole to the other without '
                'allowing them to touch',
                'slide the ring from one end of the high speed rotating pole to the other, '
                'avoiding contact between them',
                'move the ring from one end of the high speed rotating pole to the other, '
                'maintaining a gap between them',
                'move the ring from end of the high speed rotating pole to the other whilst '
                'maintaining separation between them']

    def step(self) -> None:
        print(self.step_id)
        self.yaw_speed = self.target_state_list[-1]['yaw_speed']
        simulation_timestep = self.pyrep.get_simulation_timestep()
        rot_speed = np.deg2rad(self.yaw_speed) * simulation_timestep
        self._frame_base.rotate([0, rot_speed, 0])
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

    def variation_count(self) -> int:
        return 1

    def is_static_workspace(self):
        return True
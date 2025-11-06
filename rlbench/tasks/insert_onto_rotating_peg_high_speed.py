from typing import List
import numpy as np
from pyrep.objects import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, ConditionSet
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.const import colors
from .place_cups_on_rotating_frame import init_target_state, compute_target_pose
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
import torch

def get_expert_info(task, bool_return_path=True):
    stage = task.stage
    tip_pose = task.robot.arm.get_tip().get_pose()
    w0_pose = task.w0.get_pose()
    w1_pose = task.w1.get_pose()
    w2_pose = task.w2.get_pose()
    w2_init_pose = task.w2_init_pose
    w3_pose = task.w3.get_pose()
    w4_pose = task.w4.get_pose()

    dist_to_wp0 = np.linalg.norm(tip_pose[:3] - w0_pose[:3])
    dist_to_wp1 = np.linalg.norm(tip_pose[:3] - w1_pose[:3])
    dist_to_wp2 = np.linalg.norm(tip_pose[:3] - w2_pose[:3])
    dist_to_wp2init = np.linalg.norm(tip_pose[:3] - w2_init_pose[:3])
    dist_to_wp3 = np.linalg.norm(tip_pose[:3] - w3_pose[:3])
    dist_to_wp4 = np.linalg.norm(tip_pose[:3] - w4_pose[:3])
    is_grasp = len(task.robot.gripper.get_grasped_objects()) > 0
    is_open = any(x > 0.95 for x in task.robot.gripper.get_open_amount())

    th_wp0 = 0.1
    th_wp1 = 0.05
    th_wp2 = 0.05
    th_wp3 = 0.03
    th_wp4 = 0.03

    # init wp2 pose
    if stage == 'wp0' and dist_to_wp0 > th_wp0:
        stage = 'wp0'
    elif stage == 'wp0' and dist_to_wp0 <= th_wp0:
        stage = 'wp1'
    elif stage == 'wp1' and dist_to_wp1 > th_wp1:
        stage = 'wp1'
    elif stage == 'wp1' and dist_to_wp1 <= th_wp1:
        stage = 'wp1-grasp'
    elif stage == 'wp1-grasp' and not is_grasp:
        stage = 'wp1-grasp'
    elif stage == 'wp1-grasp' and is_grasp:
        stage = 'wp2'
    elif stage == 'wp2' and dist_to_wp2init > th_wp2:
        stage = 'wp2'
    elif stage == 'wp2' and dist_to_wp2init <= th_wp2:
        stage = 'wp3'
    elif stage == 'wp3' and dist_to_wp3 > th_wp3:
        stage = 'wp3'
    elif stage == 'wp3' and dist_to_wp3 <= th_wp3:
        stage = 'wp4'
    elif stage == 'wp4' and dist_to_wp4 > th_wp4:
        stage = 'wp4'
    elif stage == 'wp4' and dist_to_wp4 <= th_wp4:
        stage = 'wp4-open'
    elif stage == 'wp4-open' and not is_grasp:
        stage = 'wp0'
    elif stage == 'wp4-open' and is_grasp:
        stage = 'wp4-open'
    else:
        print("Unrecognized stage: ", stage)
        import pdb
        pdb.set_trace()

    if stage == 'wp0':
        eepose = w0_pose
        open = 1
    elif stage == 'wp1':
        eepose = w1_pose
        open = 1
    elif stage == 'wp1-grasp':
        eepose = w1_pose
        open = 0
    elif stage == 'wp2':
        eepose = w2_init_pose
        open = 0
    elif stage == 'wp3':
        eepose = w3_pose
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w3,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 0
    elif stage == 'wp4':
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w4,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 0
    elif stage == 'wp4-open':
        t_delay = 0.5
        eepose = compute_target_pose(
            task._frame_base,
            task.w4,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        open = 1
    else:
        print("Unrecognized stage: ", stage)
        import pdb
        pdb.set_trace()

    print(f"stage: {stage}, eepose: {eepose}, open: {open}, dist_to_wp0: {dist_to_wp0:.2f}, dist_to_wp1: {dist_to_wp1:.2f}, dist_to_wp2: {dist_to_wp2:.2f}, dist_to_wp3: {dist_to_wp3:.2f}, dist_to_wp4: {dist_to_wp4:.2f}, is_open:{is_open}, ring_pos: {task._square_ring.get_position()}")
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
            "tar_position": task._square_ring.get_position(),
            "t": task.t,
        }
    }
    if bool_return_path:
        path = task.get_path(eepose)
        expert_info["path"] = path
    return expert_info


class InsertOntoRotatingPegHighSpeed(Task):

    def init_task(self) -> None:
        self._square_ring = Shape('square_ring')
        self._success_centre = Dummy('success_centre')
        success_detectors = [ProximitySensor(
            'success_detector%d' % i) for i in range(4)]
        self.register_graspable_objects([self._square_ring])
        success_condition = ConditionSet([DetectedCondition(
            self._square_ring, sd) for sd in success_detectors])
        self.register_success_conditions([success_condition])
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self.stage = 'wp0'
        self._bool_expert = True
        self.var2target_state_list = {}
        self._frame_base = Shape('square_base')
        for var_index in range(self.variation_count()):
            yaw_speed = init_target_state(
                min_yaw=30,  # 5 degree/s
                max_yaw=100,  # 15 degree/s
                d_yaw=1,
            )
            color_name, color_rgb = colors[var_index]
            target_spoke = np.random.choice(['pillar0', 'pillar2'])
            color_choices = np.random.choice(
                list(range(var_index)) + list(range(var_index + 1, len(colors))),
                size=2, replace=False)
            # b = SpawnBoundary([Shape('boundary0')])
            # b.sample(self._square_ring)
            frame_dx_dy = np.random.uniform([-0.05, -0.05], [0.05, 0.05], size=(2,2))
            ring_position = self._square_ring.get_position() + [frame_dx_dy[0][0], frame_dx_dy[0][1], 0]
            frame_position = self._frame_base.get_position() + [frame_dx_dy[1][0], frame_dx_dy[1][1], 0]
            self.var2target_state_list[var_index] = {
                'yaw_speed': yaw_speed,
                'color_name': color_name,
                'color_rgb': color_rgb,
                'target_spoke': target_spoke,
                'color_choices': color_choices,
                'ring_position': ring_position,
                'frame_position': frame_position,
            }

        return

    def init_episode(self, index: int, bool_random_place: bool = True) -> List[str]:
        if bool_random_place:
            for var_index in range(self.variation_count()):
                yaw_speed = init_target_state(
                    min_yaw=30,  # 5 degree/s
                    max_yaw=100,  # 15 degree/s
                    d_yaw=1,
                )
                color_name, color_rgb = colors[var_index]
                target_spoke = np.random.choice(['pillar0', 'pillar2'])
                color_choices = np.random.choice(
                    list(range(var_index)) + list(range(var_index + 1, len(colors))),
                    size=2, replace=False)
                # b = SpawnBoundary([Shape('boundary0')])
                # b.sample(self._square_ring)
                frame_dx_dy = np.random.uniform([-0.05, -0.05], [0.05, 0.05], size=(2,2))
                ring_position = self._square_ring.get_position() + [frame_dx_dy[0][0], frame_dx_dy[0][1], 0]
                frame_position = self._frame_base.get_position() + [frame_dx_dy[1][0], frame_dx_dy[1][1], 0]
                self.var2target_state_list[var_index] = {
                    'yaw_speed': yaw_speed,
                    'color_name': color_name,
                    'color_rgb': color_rgb,
                    'target_spoke': target_spoke,
                    'color_choices': color_choices,
                    'ring_position': ring_position,
                    'frame_position': frame_position,
                }
        self._square_ring.set_position(self.var2target_state_list[index]['ring_position'])
        self._frame_base.set_position(self.var2target_state_list[index]['frame_position'])
        # color_name, color_rgb = colors[index]
        color_name = self.var2target_state_list[index]['color_name']
        color_rgb = self.var2target_state_list[index]['color_rgb']
        spokes = [Shape('pillar0'), Shape('pillar1'), Shape('pillar2')]
        # chosen_pillar = np.random.choice(spokes)
        chosen_pillar = Shape(self.var2target_state_list[index]['target_spoke'])
        chosen_pillar.set_color(color_rgb)
        _, _, z = self._success_centre.get_position()
        x, y, _ = chosen_pillar.get_position()
        self._success_centre.set_position([x, y, z])

        # color_choices = np.random.choice(
        #     list(range(index)) + list(range(index + 1, len(colors))),
        #     size=2, replace=False)
        color_choices = self.var2target_state_list[index]['color_choices']
        spokes.remove(chosen_pillar)
        for spoke, i in zip(spokes, color_choices):
            name, rgb = colors[i]
            spoke.set_color(rgb)
        # b = SpawnBoundary([Shape('boundary0')])
        # b.sample(self._square_ring)
        self.var_index = index
        self.cleanup()
        self.yaw_speed = self.var2target_state_list[index]['yaw_speed']
        self.target_state_list.append({
            'yaw_speed': self.yaw_speed,
            't0': 0,
        })
        self.step_id = 0
        self.t = 0
        self.stage = 'wp0'
        self.w0 = Dummy('waypoint0')
        self.w1 = Dummy('waypoint1')
        self.w2 = Dummy('waypoint2')
        self.w3 = Dummy('waypoint3')
        self.w4 = Dummy('waypoint4')
        self.w2_init_pose = self.w2.get_pose()

        np.set_printoptions(precision=3, suppress=True)
        print("frame_pose", self._frame_base.get_pose())
        print("ring_pose", self._square_ring.get_pose())
        print("yaw_speed", self.yaw_speed)
        print("var2target_state_list", self.var2target_state_list)

        return ['put the ring on the high speed rotating %s spoke' % color_name,
                'slide the ring onto the high speed rotating %s colored spoke' % color_name,
                'place the ring onto the high speed rotating %s spoke' % color_name]

    def step(self) -> None:
        print(self.yaw_speed)
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

    def variation_count(self) -> int:
        return len(colors[:3])

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

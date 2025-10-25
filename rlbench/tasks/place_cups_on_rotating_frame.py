from typing import List, Tuple
import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition, NothingGrasped, \
    OrConditions
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError
import torch

def get_expert_info(task, bool_return_path=True):
    target_cup = task._cups[task._cups_placed]
    # target_spoke = self._spokes[self._cups_placed]
    tip_pose = task.robot.arm.get_tip().get_pose()
    # find nearest spoke as target spoke
    target_spoke = min(
        task._spokes,
        key=lambda spoke: np.linalg.norm(spoke.get_position() - tip_pose[:3]))
    
    task._w1.set_parent(target_cup)
    task._w5.set_pose(
        task._initial_relative_spoke,
        relative_to=target_spoke)
    task._w1.set_pose(
        task._initial_relative_cup,
        relative_to=target_cup)
    
    wp0_pose = task._w0.get_pose()
    wp1_pose = task._w1.get_pose()
    wp2_pose = task._w2.get_pose()
    wp5_pose = task._w5.get_pose()
    wp3_pose = task._w3.get_pose()
    wp4_pose = task._w4.get_pose()
    wp6_pose = task._w6.get_pose()
    stage = task.stage
    dist_to_wp0 = np.linalg.norm(tip_pose[:3] - wp0_pose[:3])
    dist_to_wp1 = np.linalg.norm(tip_pose[:3] - wp1_pose[:3])
    dist_to_wp2 = np.linalg.norm(tip_pose[:3] - wp2_pose[:3])
    dist_to_wp3 = np.linalg.norm(tip_pose[:3] - wp3_pose[:3])
    dist_to_wp4 = np.linalg.norm(tip_pose[:3] - wp4_pose[:3])
    dist_to_wp5 = np.linalg.norm(tip_pose[:3] - wp5_pose[:3])
    dist_to_wp6 = np.linalg.norm(tip_pose[:3] - wp6_pose[:3])
    th_w0 = 0.1
    th_w3 = 0.1
    th_w4 = 0.02
    th_w5= 0.03
    is_grasping = len(task.robot.gripper.get_grasped_objects()) > 0
    print('---------------------------------')
    print(task.step_id)
    print("stage: ", stage, "is_grasping: ", is_grasping)
    print(f"dist_to_wp0: {dist_to_wp0:.2f}, dist_to_wp1: {dist_to_wp1:.2f}, dist_to_wp2: {dist_to_wp2:.2f}, dist_to_wp3: {dist_to_wp3:.2f}, dist_to_wp4: {dist_to_wp4:.2f}, dist_to_wp5: {dist_to_wp5:.2f}, dist_to_wp6: {dist_to_wp6:.2f}")
    
    if stage == 'wp0' and dist_to_wp0 > th_w0:
        stage = 'wp0'
        eepose = wp0_pose
        open = 1
    elif stage == 'wp0' and dist_to_wp0 <= th_w0:
        stage = 'wp1'
        t_delay = 0.0
        eepose = wp1_pose
        open = 1
    elif stage == 'wp1' and not is_grasping:
        stage = 'wp1'
        t_delay = 0.0
        eepose = wp1_pose
        eepose[2] -= 0.02
        open = 0
    elif stage == 'wp1' and is_grasping:
        stage = 'wp2'
        t_delay = 0.0
        eepose = wp2_pose
        open = 0
    elif stage == 'wp2':
        stage = 'wp3'
        t_delay = 0.0
        eepose = wp3_pose
        open = 0
    elif stage == 'wp3' and dist_to_wp3 > th_w3:
        stage = 'wp3'
        t_delay = 0.0
        eepose = wp3_pose
        open = 0
    elif stage == 'wp3' and dist_to_wp3 <= th_w3:
        stage = 'wp4'
        t_delay = 0.5
        wp4_pose_pred = compute_target_pose(
            task._frame_base,
            task._w4,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        eepose = wp4_pose_pred
        open = 0
    elif stage == 'wp4' and dist_to_wp4 > th_w4:
        stage = 'wp4'
        t_delay = 0.5
        wp4_pose_pred = compute_target_pose(
            task._frame_base,
            task._w4,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        eepose = wp4_pose_pred
        open = 0
    elif stage == 'wp4' and dist_to_wp4 <= th_w4:
        stage = 'wp5'
        t_delay = 0.5
        wp5_pose_pred = compute_target_pose(
            task._frame_base,
            task._w5,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        eepose = wp5_pose_pred
        open = 0
    elif stage == 'wp5' and dist_to_wp5 > th_w5:
        stage = 'wp5'
        t_delay = 0.5
        wp5_pose_pred = compute_target_pose(
            task._frame_base,
            task._w5,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        eepose = wp5_pose_pred
        open = 0
    elif stage == 'wp5' and dist_to_wp5 < th_w5 and is_grasping:
        stage = 'wp5'
        t_delay = 0.5
        wp5_pose_pred = compute_target_pose(
            task._frame_base,
            task._w5,
            t_delay,
            yaw_speed=task.yaw_speed,
        )
        eepose = wp5_pose_pred
        open = 1
    elif stage == 'wp5' and dist_to_wp5 < th_w5 and not is_grasping:
        stage = 'wp0'
        t_delay = 0
        eepose = wp0_pose
        open = 1
        task_cup_placed = task._cups_placed + 1
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
        "task_cup_placed": task._cups_placed,
        "open": open,
        "eepose": eepose,
        "debug_info": {
            "tip_cur_position": tip_pose[:3],
            "tar_position": target_cup.get_position(),
            "t": task.t,
        }
    }
    if bool_return_path:
        path = task.get_path(eepose)
        expert_info["path"] = path
    return expert_info

def init_target_state(
    min_yaw: float,
    max_yaw: float,
    d_yaw: float,
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
    dyaw = 1
    yaws = np.arange(yaw_range[0], yaw_range[1] + dyaw/2, dyaw)
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

class PlaceCupsOnRotatingFrame(Task):

    def init_task(self) -> None:
        self._cups = [Shape('mug%d' % i) for i in range(3)]
        self._spokes = [Shape('place_cups_holder_spoke%d' % i) for i in
                        range(3)]
        self._cups_boundary = Shape('mug_boundary')
        self._w0 = Dummy('waypoint0')
        self._w1 = Dummy('waypoint1')
        self._w2 = Dummy('waypoint2')
        self._w3 = Dummy('waypoint3')
        self._w4 = Dummy('waypoint4')
        self._w5 = Dummy('waypoint5')
        self._w6 = Dummy('waypoint6')
        self.stage = 'wp0'
        success_detectors = [
            ProximitySensor('success_detector%d' % i) for i in range(3)]
        self._on_peg_conditions = [OrConditions([
            DetectedCondition(self._cups[ci], success_detectors[sdi]) for sdi in
            range(3)]) for ci in range(3)]
        self.register_graspable_objects(self._cups)
        self._initial_relative_cup = self._w1.get_pose(self._cups[0])
        self._initial_relative_spoke = self._w5.get_pose(self._spokes[0])
        self.step_id = 0
        self.t = 0
        self.stage = 'wp0'
        self.target_state_list = []
        self._bool_expert = True
        self.var2target_state_list = {}
        self._frame_base = Shape('place_cups_holder_base')
        for var_index in range(self.variation_count()):
            self.var2target_state_list[var_index] = [init_target_state(
                min_yaw=2.5, # 5 degree/s
                max_yaw=7.5, # 15 degree/s
                d_yaw=1,
            )]

    def init_episode(self, index: int) -> List[str]:
        self._cups_placed = 0
        self._index = index
        b = SpawnBoundary([self._cups_boundary])
        [b.sample(c, min_distance=0.10) for c in self._cups]
        success_conditions = [NothingGrasped(self.robot.gripper)
                              ] + self._on_peg_conditions[:index + 1]
        self.register_success_conditions(success_conditions)
        self.register_waypoint_ability_start(
            0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self.yaw_speed = self.var2target_state_list[index][0]
        self.target_state_list.append({
            "yaw_speed": self.yaw_speed,
            "t0": 0,
        })

        if index == 0:
            return ['place 1 cup on the cup holder',
                    'pick up one cup and put it on the mug tree',
                    'move 1 mug from the table to the cup holder',
                    'pick up one cup and slide its handle onto a spoke on the '
                    'mug holder']
        else:
            return ['place %d cups on the cup holder' % (index + 1),
                    'pick up %d cups and place them on the holder'
                    % (index + 1),
                    'move %d cups from the table to the mug tree'
                    % (index + 1),
                    'pick up %d mugs and slide their handles onto the cup '
                    'holder spokes' % (index + 1)]
    
    # def _get_waypoints(self, validating=False):
    #     return []

    def step(self) -> None:
        simulation_timestep = self.pyrep.get_simulation_timestep()
        rot_speed = np.deg2rad(self.yaw_speed) * simulation_timestep
        self._frame_base.rotate([0, 0, rot_speed])
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
        self._cups_placed = expert_info["task_cup_placed"]
        return path, open

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

    def variation_count(self) -> int:
        return 1

    def _move_above_next_target(self, waypoint):
        self._w1.set_parent(self._cups[self._cups_placed])
        self._w5.set_pose(
            self._initial_relative_spoke,
            relative_to=self._spokes[self._cups_placed])
        self._w1.set_pose(
            self._initial_relative_cup,
            relative_to=self._cups[self._cups_placed])
        self._cups_placed += 1

    def _repeat(self):
        return self._cups_placed < self._index + 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, -np.pi / 2], [0.0, 0.0, np.pi / 2]

    def cleanup(self) -> None:
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self._cups_placed = 0
        self._index = 0
        self.stage = 'wp0'
        self.yaw_speed = None
        return
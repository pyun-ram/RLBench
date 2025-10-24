import numpy as np
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import ConfigurationPathError
from pyrep.const import ConfigurationPathAlgorithms as Algos
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects import ProximitySensor, Shape, Dummy
import torch

def compute_target_position(
    t: float,
    x0: List[float],
    v0: List[float],
    a0: List[float],
    t0: float,
    dt: float = None,
) -> List[float]:
    '''
    Args:
        t: float, current time
        x0: List[float], initial position (x,y,z)
        v0: List[float], initial velocity (vx, vy, vz)
        a0: List[float], initial acceleration (ax, ay, az)
        t0: float, initial time
    Return:
        List[float], target position (x,y,z)
    '''
    t_rel = t - t0
    x0 = np.array(x0)
    v0 = np.array(v0)
    a0 = np.array(a0)
    pos = x0 + v0 * t_rel + 0.5 * a0 * t_rel**2
    return pos.tolist()

def get_expert_info(task, th_grasp=0.4, t_delay=1.0, bool_return_path=True):
    # compensate for grasping delay
    tar_position = task.target.get_position()
    tip_cur_pose = task.robot.arm.get_tip().get_pose()
    tip_cur_position = tip_cur_pose[:3]
    dist_tip_tar = np.linalg.norm(tip_cur_position - tar_position)
    simulation_timestep = task.pyrep.get_simulation_timestep()
    target_state_dict = task.target_state_list[-1]
    if dist_tip_tar >= th_grasp:
        stage = 'reach'
        wp_position = tar_position
    else:
        stage = 'grasp'
        wp_position = compute_target_position(
            t = task.t+t_delay,
            t0=task.t,
            x0=tar_position,
            v0=target_state_dict["v"],
            a0=target_state_dict["a"],
            dt = simulation_timestep,
        )
    eepose = np.ones((7))
    eepose[:7] = tip_cur_pose
    eepose[:3] = wp_position
    eepose[3:7] = np.array([0, 1, 0, 0])
    open = 1
    task.stage = stage
    output = np.ones((1,1,8))
    output[0,0,:7] = eepose
    output[0,0,7:] = open
    expert_info = {
        "trajectory": torch.from_numpy(output),
        "stage": stage,
        "debug_info": {
            "tip_cur_position": tip_cur_position,
            "tar_position": tar_position,
            "tip_cur_pose": tip_cur_pose,
            "dist_tip_tar": dist_tip_tar,
            "simulation_timestep": simulation_timestep,
            "target_state_dict": target_state_dict,
            "t": task.t,
        }
    }
    if bool_return_path:
        path = task.get_path(eepose)
        expert_info["path"] = path
    return expert_info

def get_state_config(var_index: int) -> bool:
    if var_index == 0:
        bool_a = False
    elif var_index == 1:
        bool_a = True
    else:
        raise ValueError("var_index must be 0, 1")
    return bool_a

def init_target_state(
    t_max: float,
    area: List[float],
    x_range: List[float],
    v_range: List[float],
    a_range: List[float],
    dx: List[float],
    dv: List[float],
    da: List[float],
    x0: List[float] = None,
    v0: List[float] = None,
    a0: List[float] = None,
    dt: float = 0.05,
    min_velo_norm: float = 0,
    min_acc_norm: float = 0,
) -> Tuple[List[float]]:
    '''
    Args:
        t_max: float, max time
        x_range: List[float], [xmin,ymin,zmin,xmax,ymax,zmax] in Fworld
        v_range: List[float], [vxmin,vymin,vzmin,vxmax,vymax,vzmax] in Fworld
        a_range: List[float], [axmin,aymin,azmin,axmax,aymax,azmax] in Fworld
        dx: List[float], [dx,dy,dz]
        dv: List[float], [dvx,dvy,dvz]
        da: List[float], [dvx,dvy,dvz]
        x0: List[float], initial position, 
        v0: List[float], initial velocity
        a0: List[float], initial acceleration
        dt: float
        min_velo_norm: float = 0,
        min_acc_norm: float = 0,
    Return:
        Tuple[List[float]], target state (x, v, a)
    '''
    x_range = x_range if x0 is None else [
        x0[0], x0[1], x0[2], x0[0], x0[1], x0[2]]
    v_range = v_range if v0 is None else [
        v0[0], v0[1], v0[2], v0[0], v0[1], v0[2]]
    a_range = a_range if a0 is None else [
        a0[0], a0[1], a0[2], a0[0], a0[1], a0[2]]
    args = x_range+v_range+a_range+dx+dv+da + \
        [area, t_max, dt, min_velo_norm, min_acc_norm]
    cache_name = "_".join([f'{itm}' for itm in args])
    cache_name = f"/tmp/{cache_name}.npy"
    if not Path(cache_name).exists():
        xs = np.arange(x_range[0], x_range[3] + dx[0]/2, dx[0])
        ys = np.arange(x_range[1], x_range[4] + dx[1]/2, dx[1])
        zs = np.arange(x_range[2], x_range[5] + dx[2]/2, dx[2])
        vxs = np.arange(v_range[0], v_range[3] + dv[0]/2, dv[0])
        vys = np.arange(v_range[1], v_range[4] + dv[1]/2, dv[1])
        vzs = np.arange(v_range[2], v_range[5] + dv[2]/2, dv[2])
        axs = np.arange(a_range[0], a_range[3] + da[0]/2, da[0])
        ays = np.arange(a_range[1], a_range[4] + da[1]/2, da[1])
        azs = np.arange(a_range[2], a_range[5] + da[2]/2, da[2])

        grid = np.array(np.meshgrid(xs, ys, zs, vxs, vys,
                        vzs, axs, ays, azs, indexing='ij'))
        points = grid.reshape(9, -1).T
        t_samples = np.arange(0, t_max + dt/2, dt)

        valid_mask = np.ones(points.shape[0], dtype=bool)
        for t in tqdm(t_samples):
            pos = points[:, 0:3] + points[:, 3:6] * \
                t + 0.5 * points[:, 6:9] * t**2
            in_box = (
                (area[0] <= pos[:, 0]) & (pos[:, 0] <= area[3]) &
                (area[1] <= pos[:, 1]) & (pos[:, 1] <= area[4]) &
                (area[2] <= pos[:, 2]) & (pos[:, 2] <= area[5])
            )
            valid_velocity = np.linalg.norm(
                points[:, 3:6], axis=-1) >= min_velo_norm
            valid_acc = np.linalg.norm(points[:, 6:9], axis=-1) >= min_acc_norm
            valid_mask &= in_box
            valid_mask &= valid_velocity
            valid_mask &= valid_acc
            if not valid_mask.any():
                break

        valid_points = points[valid_mask]
        Path(cache_name).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_name, valid_points)
    else:
        valid_points = np.load(cache_name)
    # sample one
    idx = np.random.choice(valid_points.shape[0])
    target_state = valid_points[idx]
    target_state = target_state.tolist()
    return target_state[:3], target_state[3:6], target_state[6:9]

class ReachSingleMovingTargetOnTheTable(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.waypoint0 = Dummy('waypoint0')
        self.success_sensor = ProximitySensor('success')
        self.var_index = None
        self.t = None
        self.step_id = None
        self.target_state_list = None
        self.t_max = 4  # (s)
        # area [xmin,ymin,zmin,xmax,ymax,zmax] in Fworld
        self.area = [0, -0.5, 0.8, 0.4, 0.5, 0.8]
        self.condition = DetectedCondition(
            self.robot.arm.get_tip(),
            self.success_sensor,
        )
        self.register_success_conditions([self.condition])
        self.var2target_state_list = {}
        self._bool_expert = True
        return
    
    def disable_expert_plan(self):
        self._bool_expert = False
        return

    def init_episode(self, index: int) -> List[str]:
        self.var_index = index
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
            self.var2target_state_list[var_index].append({
                "x": x,
                "v": v,
                "a": a,
            })
        target_state = self.var2target_state_list[self.var_index][0]
        # save target_state
        self.cleanup()
        self.target_state_list.append({
            "x": target_state["x"],
            "v": target_state["v"],
            "a": target_state["a"],
            "t0": 0,
        })
        self.target.set_position(target_state["x"])
        if index == 1:
            return [
                "reach single accelerated ball on the table"
            ]
        else:
            return [
                "reach single uniform-speed ball on the table"
            ]

    def variation_count(self) -> int:
        return 2

    def step(self) -> None:
        simulation_timestep = self.pyrep.get_simulation_timestep()
        target_state_dict = self.target_state_list[-1]
        target_position = compute_target_position(
            t=self.t,
            t0=target_state_dict["t0"],
            x0=target_state_dict["x"],
            v0=target_state_dict["v"],
            a0=target_state_dict["a"],
            dt=simulation_timestep,
        )
        self.target.set_position(target_position)
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

    def cleanup(self) -> None:
        self.target_state_list = []
        self.t = 0
        self.step_id = 0
        return

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True

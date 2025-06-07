from typing import List, Tuple
from rlbench.backend.task import Task
from pyrep.objects import ProximitySensor, Shape, Dummy
from rlbench.backend.conditions import DetectedCondition
import numpy as np
from tqdm import tqdm
from pathlib import Path

def get_state_config(var_index: int):
    if var_index == 0:
        return 'left'
    if var_index == 1:
        return 'right'
    else:
        raise ValueError("var_index must be 0, 1")

def init_target_state(
    area: List[float],
    t_max: float,
    x0: List[float] = None,
    v0: List[float] = None,
    a0: List[float] = None,
    dt: float = 0.05,
    direction: str = 'left',
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
    assert direction in ['left', 'right']

    dx = [0.05, 0.05, 0.05]
    dv = [0.01, 0.01, 0.01]
    da = [0.001, 0.001, 0.001]
    if direction == 'left':
        x_range = [area[0], area[1], area[5], area[3], area[1], area[5]] if x0 is None else [x0[0], x0[1], x0[2], x0[0], x0[1], x0[2]]
        v_range = [0, 0, 0, 0, 0.1, 0.1] if v0 is None else [v0[0], v0[1], v0[2], v0[0], v0[1], v0[2]]
    elif direction == 'right':
        x_range = [area[0], area[4], area[5], area[3], area[4], area[5]] if x0 is None else [x0[0], x0[1], x0[2], x0[0], x0[1], x0[2]]
        v_range = [0, -0.1, 0, 0, 0, 0.1] if v0 is None else [v0[0], v0[1], v0[2], v0[0], v0[1], v0[2]]
    a_range = [0, 0, -0.10, 0, 0, 0] if a0 is None else [a0[0], a0[1], a0[2], a0[0], a0[1], a0[2]]

    cache_name = "_".join([f'{itm}' for itm in x_range+v_range+a_range+dx+dv+da])
    cache_name = f"/tmp/{cache_name}.npy"
    if True:
        xs = np.arange(x_range[0], x_range[3] + dx[0]/2, dx[0])
        ys = np.arange(x_range[1], x_range[4] + dx[1]/2, dx[1])
        zs = np.arange(x_range[2], x_range[5] + dx[2]/2, dx[2])
        vxs = np.arange(v_range[0], v_range[3] + dv[0]/2, dv[0])
        vys = np.arange(v_range[1], v_range[4] + dv[1]/2, dv[1])
        vzs = np.arange(v_range[2], v_range[5] + dv[2]/2, dv[2])
        axs = np.arange(a_range[0], a_range[3] + da[0]/2, da[0])
        ays = np.arange(a_range[1], a_range[4] + da[1]/2, da[1])
        azs = np.arange(a_range[2], a_range[5] + da[2]/2, da[2])

        grid = np.array(np.meshgrid(xs, ys, zs, vxs, vys, vzs, axs, ays, azs, indexing='ij'))
        points = grid.reshape(9, -1).T
        t_samples = np.arange(0, t_max + dt/2, dt)

        valid_mask = np.ones(points.shape[0], dtype=bool)
        for t in tqdm(t_samples):
            pos = points[:, 0:3] + points[:, 3:6] * t + 0.5 * points[:, 6:9] * t**2
            in_box = (
                (area[0] <= pos[:, 0]) & (pos[:, 0] <= area[3]) &
                (area[1] <= pos[:, 1]) & (pos[:, 1] <= area[4]) &
                (area[2] <= pos[:, 2]) & (pos[:, 2] <= area[5])
            )
            valid_velocity = np.linalg.norm(points[:, 3:6], axis=-1) >= 0.03
            valid_acc = np.linalg.norm(points[:, 6:9], axis=-1) > 0.07
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


def compute_target_position(
        t: float,
        x0: List[float],
        v0: List[float],
        a0: List[float],
        t0: float,
        dt: float,
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

def compute_delay(cur_position, tar_position):
    '''
    Args:
        cur_position: List[float], current position (x,y,z)
        tar_position: List[float], target position (x,y,z)
    Return:
        float, delay
    '''
    avr_speed = 0.3 # (m/s)
    return np.linalg.norm(cur_position - tar_position) / avr_speed
    

class ReachSingleBouncingBall(Task):

    def init_task(self) -> None:
        self.success_sensor = ProximitySensor('success')
        self.waypoint0 = Dummy('waypoint0')
        self.target = Shape('target')
        # [xmin,ymin,zmin,xmax,ymax,zmax] in Fworld
        self.area = [0, -0.5, 0.8, 0.4, 0.5, 1.2]
        self.t_max = 1 # (s)
        self.step_id = 0
        self.t = 0
        self.target_state_list = []
        self.register_success_conditions([
            DetectedCondition(self.robot.arm.get_tip(), self.success_sensor)
        ])
        self.register_waypoint_ability_start(0, self._move_above_object)
        self.register_waypoints_should_repeat(self._repeat)
        return

    def init_episode(self, index: int) -> List[str]:
        var_index = index
        direction = get_state_config(var_index)
        x, v, a = init_target_state(
            self.area,
            self.t_max,
            x0=None,
            v0=None,
            a0=None,
            direction=direction
        )
        # save target_state
        self.target_state_list.append({
            "x": x,
            "v": v,
            "a": a,
            "t0": 0,
        })
        self.t = 0
        self.target.set_position(x)
        return [
            "reach single moving target",
            "reach single moving target on the table",
        ]

    def variation_count(self) -> int:
        return 2

    def step(self) -> None:
        if self.target.get_position()[-1] <= 0.8:
            target_state_dict = self.target_state_list[-1]
            v = np.array(target_state_dict['v']) + \
                np.array(target_state_dict['a']) * \
                (self.t - target_state_dict['t0'])
            v[2] *= -1
            v[2] *= 0.9
            x = self.target.get_position()
            x[2] = 0.8
            new_target_state_dict = {
                "x": x,
                "v": v,
                "a": target_state_dict['a'],
                "t0": self.t,
            }
            self.target_state_list.append(new_target_state_dict)
        self.step_id += 1
        simulation_timestep = self.pyrep.get_simulation_timestep()
        self.t += simulation_timestep
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
        return

    def cleanup(self) -> None:
        # Called during at the end of each episode. Remove this if not using.
        self.target_state_list = []

    def _move_above_object(self, waypoint):
        # compensate for grasping delay
        tip_cur_position = self.robot.arm.get_tip().get_position()
        tip_tar_position = self.target.get_position()
        simulation_timestep = self.pyrep.get_simulation_timestep()
        t_delay = compute_delay(
            tip_cur_position,
            tip_tar_position,
        )
        target_state_dict = self.target_state_list[-1]
        v = np.array(target_state_dict["v"]) + \
            np.array(target_state_dict["a"]) * \
            (self.t - target_state_dict['t0'])
        new_wp_position = compute_target_position(
            t = self.t+t_delay,
            t0=self.t,
            x0=tip_tar_position,
            v0=v,
            a0=target_state_dict["a"],
            dt = simulation_timestep,
        )
        way_obj = waypoint.get_waypoint_object()
        way_obj.set_position(new_wp_position)
        return

    def _repeat(self):
        return True

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True

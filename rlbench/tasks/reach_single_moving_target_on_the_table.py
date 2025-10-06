import numpy as np

from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple

from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects import ProximitySensor, Shape, Dummy


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
        return

    def init_episode(self, index: int) -> List[str]:
        self.var_index = index
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

    def compute_target_position(
        self,
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

    def variation_count(self) -> int:
        return 2

    def step(self) -> None:
        self.step_id += 1
        simulation_timestep = self.pyrep.get_simulation_timestep()
        self.t += simulation_timestep
        target_state_dict = self.target_state_list[-1]
        target_position = self.compute_target_position(
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
        self.target_state_list = []
        self.t = 0
        self.step_id = 0
        return

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def is_static_workspace(self) -> bool:
        return True

import numpy as np
from typing import List
from .reach_single_moving_target_with_gravity import ReachSingleMovingTargetWithGravity, get_state_config
from .reach_single_moving_target_on_the_table_cpst import compute_delay
from .reach_single_moving_target_on_the_table_nocpst import init_target_state
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects import ProximitySensor, Shape, Dummy
class ReachSingleBouncingBall(ReachSingleMovingTargetWithGravity):

    def init_task(self):
        self.target = Shape('target')
        self.waypoint0 = Dummy('waypoint0')
        self.success_sensor = ProximitySensor('success')
        self.var_index = None
        self.t = None
        self.step_id = None
        self.target_state_list = None
        # area [xmin,ymin,zmin,xmax,ymax,zmax] in Fworld
        self.condition = DetectedCondition(
            self.robot.arm.get_tip(),
            self.success_sensor,
        )
        self.register_success_conditions([self.condition])
        self.register_waypoint_ability_start(0, self._move_above_object)
        self.register_waypoints_should_repeat(self._repeat)
        self.area = [0, -0.5, 0.8, 0.4, 0.5, 1.2]
        self.t_max = 1 #(s)
        self.min_velo_norm = 0.03
        self.min_acc_norm = 0.07
        self.a_range = [0, 0, -0.1, 0, 0, 0]
        self.dx = [0.05, 0.05, 0.05]
        self.dv = [0.01, 0.01, 0.01]
        self.da = [0.01, 0.01, 0.01]
        self.var2target_state_list = {}
        for var_index in range(self.variation_count()):
            self.var2target_state_list[var_index] = []
            direction = get_state_config(var_index)
            area = self.area
            if direction == 'left':
                x_range = [area[0], area[1], area[5], area[3], area[1], area[5]]
                v_range = [0, 0, 0, 0, 0.1, 0.1]
            elif direction == 'right':
                x_range = [area[0], area[4], area[5], area[3], area[4], area[5]]
                v_range = [0, -0.1, 0, 0, 0, 0.1]
            x, v, a = init_target_state(
                t_max=self.t_max,
                area=area,
                x_range=x_range,
                v_range=v_range,
                a_range=self.a_range,
                x0=None,
                v0=None,
                a0=None,
                dx = self.dx,
                dv = self.dv,
                da = self.da,
                min_velo_norm=self.min_velo_norm,
                min_acc_norm=self.min_acc_norm,
            )
            self.var2target_state_list[var_index].append({
                "x": x,
                "v": v,
                "a": a,
            })
        return
    
    def init_episode(self, index: int) -> List[str]:
        super().init_episode(index)
        direction = get_state_config(index)
        return [
            f"reach single bouncing ball from {direction}",
        ]

    def compute_target_position(
        self,
        t: float,
        x0: List[float],
        v0: List[float],
        a0: List[float],
        t0: float,
        dt: float = 0.05,
        z_table: float = 0.8
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
        x0 = np.array(x0, dtype=float)
        v0 = np.array(v0, dtype=float)
        a0 = np.array(a0, dtype=float)
        t_curr = t0
        pos = x0.copy()
        vel = v0.copy()
        while t_curr < t:
            t_next = min(t_curr + dt, t)
            t_rel = t_next - t_curr
            pos_next = pos + vel * t_rel + 0.5 * a0 * t_rel**2
            vel_next = vel + a0 * t_rel
            # 检查是否穿过z_table
            if pos[2] > z_table and pos_next[2] < z_table:
                # 计算到碰撞的精确时间
                # z(t) = pos[2] + vel[2]*tau + 0.5*a0[2]*tau^2 = z_table
                # 0.5*a*tau^2 + v*tau + (pos[2]-z_table) = 0
                a = 0.5 * a0[2]
                b = vel[2]
                c = pos[2] - z_table
                tau = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a) if a != 0 else -c/b
                # 到达碰撞点
                pos = pos + vel * tau + 0.5 * a0 * tau**2
                vel = vel + a0 * tau
                vel[2] = -vel[2]  # 反向z速度
                t_curr += tau
            else:
                pos = pos_next
                vel = vel_next
                t_curr = t_next
        return pos.tolist()

    def _move_above_object(self, waypoint):
        T = 5 # s
        dt = 0.1 # s
        min_t_diff = np.inf
        new_wp_position = None
        target_state_dict = self.target_state_list[-1]
        v = np.array(target_state_dict['v']) + \
            np.array(target_state_dict['a']) * \
            (self.t - target_state_dict['t0'])
        for t in np.arange(0, T + dt/2, dt):
            x_tar = self.compute_target_position(
                t=self.t+t,
                t0=self.t,
                x0=self.target.get_position(),
                v0=v,
                a0=target_state_dict['a'],
                dt=self.pyrep.get_simulation_timestep(),
            )
            t_delay = compute_delay(
                self.robot.arm,
                x_tar,
            )
            t_diff = np.linalg.norm(t_delay - t)
            if t_diff < min_t_diff:
                min_t_diff = t_diff
                new_wp_position = x_tar
        way_obj = waypoint.get_waypoint_object()
        way_obj.set_position(new_wp_position)
        return
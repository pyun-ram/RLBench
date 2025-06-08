import numpy as np
from typing import List
from .reach_single_moving_target_with_gravity import ReachSingleMovingTargetWithGravity

class ReachSingleBouncingBall(ReachSingleMovingTargetWithGravity):

    def init_task(self):
        super().init_task()
        self.area = [0, -0.5, 0.8, 0.4, 0.5, 1.2]
        self.t_max = 1 #(s)
        self.min_velo_norm = 0.03
        self.min_acc_norm = 0.07
        self.a_range = [0, 0, -0.1, 0, 0, 0]
        self.dx = [0.05, 0.05, 0.05]
        self.dv = [0.01, 0.01, 0.01]
        self.da = [0.01, 0.01, 0.01]
        return
    
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


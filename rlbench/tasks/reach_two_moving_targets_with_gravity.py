from pyrep.objects import ProximitySensor, Shape, Dummy
import numpy as np
from .reach_single_moving_target_with_gravity import (
    ReachSingleMovingTargetWithGravity, init_target_state)
from typing import List

def get_state_config(var_index: int):
    if var_index == 0:
        return ['left', 'right']
    if var_index == 1:
        return ['right', 'left']
    else:
        raise ValueError("var_index must be 0, 1")

class ReachTwoMovingTargetsWithGravity(ReachSingleMovingTargetWithGravity):

    def step(self) -> None:
        if self.condition.condition_met()[0] and self.counter < 1:
            self.counter += 1
            direction = get_state_config(self.var_index)[self.counter]
            area = self.area
            min_velo_norm = 0.03
            min_acc_norm = 0.01
            if direction == 'left':
                x_range = [area[0], area[1], area[5], area[3], area[1], area[5]]
                v_range = [0, 0, 0, 0, 0.1, 0.1]
            elif direction == 'right':
                x_range = [area[0], area[4], area[5], area[3], area[4], area[5]]
                v_range = [0, -0.1, 0, 0, 0, 0.1]
            x = cur_x = self.target.get_position()
            while np.linalg.norm(x - cur_x) < 0.2:
                x, v, a = init_target_state(
                    t_max=self.t_max,
                    area=area,
                    x_range=x_range,
                    v_range=v_range,
                    a_range=[0, 0, -0.035, 0, 0, 0],
                    x0=None,
                    v0=None,
                    a0=None,
                    dx = [0.05, 0.05, 0.05],
                    dv = [0.01, 0.01, 0.01],
                    da = [0.001, 0.001, 0.001],
                    min_velo_norm=min_velo_norm,
                    min_acc_norm=min_acc_norm,
                )
            self.target_state_list.append({
                "x": x,
                "v": v,
                "a": a,
                "t0": self.t,
            })
            self.target.set_position(x)
        return super().step()

    def init_task(self):
        super().init_task()
        self.counter = 0
        return

    def cleanup(self):
        super().cleanup()
        self.counter = 0
        return

    def init_episode(self, index: int) -> List[str]:
        super().init_episode(index)
        direction = get_state_config(self.var_index)
        return [
            f"reach two falling balls, first from {direction[0]}",
        ]
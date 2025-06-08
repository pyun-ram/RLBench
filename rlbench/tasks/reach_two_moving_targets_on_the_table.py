import numpy as np

from .reach_single_moving_target_on_the_table_cpst import (
    ReachSingleMovingTargetOnTheTableCpst)
from .reach_single_moving_target_on_the_table_nocpst import (
    get_state_config, init_target_state)


class ReachTwoMovingTargetsOnTheTable(ReachSingleMovingTargetOnTheTableCpst):

    def step(self) -> None:
        if self.condition.condition_met()[0] and self.counter < 1:
            bool_a = get_state_config(self.var_index)
            x = cur_x = self.target.get_position()
            while np.linalg.norm(x - cur_x) < 0.2:
                x, v, a = init_target_state(
                    t_max=self.t_max,
                    x_range=self.area,
                    v_range=[-0.2, -0.2, 0, 0.2, 0.2, 0],
                    a_range=[-0.01, -0.01, 0, 0.01, 0.01, 0],
                    x0=None,
                    v0=None,
                    a0=[0, 0, 0] if not bool_a else None,
                    dx=[0.05, 0.05, 0.05],
                    dv=[0.025, 0.025, 0.025],
                    da=[0.001, 0.001, 0.001],
                )
            self.target_state_list.append({
                "x": x,
                "v": v,
                "a": a,
                "t0": self.t,
            })
            self.target.set_position(x)
            self.counter += 1
        return super().step()

    def init_task(self):
        super().init_task()
        self.counter = 0
        return

    def cleanup(self):
        super().cleanup()
        self.counter = 0
        return

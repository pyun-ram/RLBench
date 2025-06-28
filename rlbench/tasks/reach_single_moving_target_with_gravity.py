from typing import List
from .reach_single_moving_target_on_the_table_cpst import ReachSingleMovingTargetOnTheTableCpst
from .reach_single_moving_target_on_the_table_nocpst import init_target_state
from rlbench.backend.conditions import DetectedCondition
from pyrep.objects import ProximitySensor, Shape, Dummy

def get_state_config(var_index: int):
    if var_index == 0:
        return 'left'
    if var_index == 1:
        return 'right'
    else:
        raise ValueError("var_index must be 0, 1")

class ReachSingleMovingTargetWithGravity(ReachSingleMovingTargetOnTheTableCpst):

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
        self.t_max = 5 #(s)
        self.min_velo_norm = 0.03
        self.min_acc_norm = 0.03
        self.a_range = [0, 0, -0.035, 0, 0, 0]
        self.dx = [0.05, 0.05, 0.05]
        self.dv = [0.01, 0.01, 0.01]
        self.da = [0.001, 0.001, 0.001]
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
        self.var_index = index
        direction = get_state_config(index)
        target_state = self.var2target_state_list[index][0]
        # save target_state
        self.cleanup()
        self.target_state_list.append({
            "x": target_state["x"],
            "v": target_state["v"],
            "a": target_state["a"],
            "t0": 0,
        })
        self.target.set_position(target_state["x"])
        return [
            f"reach single falling ball from {direction}",
        ]

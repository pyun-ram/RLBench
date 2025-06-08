import numpy as np
from .reach_single_moving_target_on_the_table_nocpst import (
    ReachSingleMovingTargetOnTheTableNocpst, compute_target_position)
from pyrep.backend import sim
from pyrep.const import ConfigurationPathAlgorithms as Algos

def compute_path_duration(path) -> float:
    """
    计算执行 path 所需的总时间（秒）
    """
    # 确保轨迹从头开始
    path.set_to_start()
    # 获取仿真步长
    dt = sim.simGetSimulationTimeStep()
    steps = 0
    done = False
    # 必须先初始化 RML 句柄，否则 step() 会自动初始化
    if path._rml_handle is None:
        path._rml_handle = path._get_rml_handle()
    while not done:
        done = path.step()
        steps += 1
    duration = steps * dt
    return duration

def compute_delay(arm, tar_position):
    '''
     Return:
         float, delay
     '''
    path = arm.get_path(
        position=tar_position,
        euler=arm.get_tip().get_orientation(),
        ignore_collisions=True,
        trials=100,
        max_configs=10,
        max_time_ms=10,
        trials_per_goal=5,
        algorithm=Algos.RRTConnect,
    )
    return compute_path_duration(path)


class ReachSingleMovingTargetOnTheTableCpst(ReachSingleMovingTargetOnTheTableNocpst):

    def _move_above_object(self, waypoint):
        # compensate for grasping delay
        tip_tar_position = self.target.get_position()
        simulation_timestep = self.pyrep.get_simulation_timestep()
        t_delay = compute_delay(
            self.robot.arm,
            tip_tar_position,
        )
        target_state_dict = self.target_state_list[-1]
        v = np.array(target_state_dict['v']) + \
            np.array(target_state_dict['a']) * \
            (self.t - target_state_dict['t0'])
        new_wp_position = compute_target_position(
            t=self.t+t_delay,
            t0=self.t,
            x0=self.target.get_position(),
            v0=v,
            a0=target_state_dict["a"],
            dt=simulation_timestep,
        )
        way_obj = waypoint.get_waypoint_object()
        way_obj.set_position(new_wp_position)
        return
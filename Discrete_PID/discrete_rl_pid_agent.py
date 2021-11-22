from typing import Optional
from ROAR.agent_module.agent import Agent
from pathlib import Path

#from ROAR.control_module.pid_controller import PIDController
from Discrete_PID.discrete_rl_pid_controller import PIDController
#from Discrete_PID.wayline_planner import WayLinePlanner


from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner


from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
from Discrete_PID.wayline import WayLine
import logging


class RLPIDAgent(Agent):
    def __init__(self, target_speed=40, **kwargs):
        super().__init__(**kwargs)
        self.target_speed = target_speed
        self.logger = logging.getLogger("PID Agent")
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan

        self.behavior_planner = BehaviorPlanner(agent=self)
        
        
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1.5)

        self.logger.debug(
            f"Waypoint Following Agent Initiated. Reading f"
            f"rom {self.route_file_path.as_posix()}")

        # related to waylines
        self.plan_lst = list(self.mission_planner.produce_single_lap_mission_plan())

        self.kwargs = kwargs
        self.interval = self.kwargs.get('interval', 5)
        self.look_back = self.kwargs.get('look_back', 2)
        self.look_back_max = self.kwargs.get('look_back_max', 5)
        self.thres = self.kwargs.get('thres', 1e-3)

        self.int_counter = 0
        self.counter = 0
        self.finished = False
        self.has_crossed = False
        self.wayline: Optional[WayLine] = None
        self._get_next_wayline()

    def run_step(self, vehicle: Vehicle,
                 sensors_data: SensorsData) -> VehicleControl:
        super(RLPIDAgent, self).run_step(vehicle=vehicle,
                                         sensors_data=sensors_data)
        self.transform_history.append(self.vehicle.transform)

        self.has_crossed, _ = self.wayline_step()

        if self.is_done:
            control = VehicleControl()
            self.logger.debug("Path Following Agent is Done. Idling.")
        else:
            control = self.local_planner.run_in_series()
        return control

    ## adapted from the e2e agent
    def wayline_step(self):
        """
        This is the function that the line detection agent used

        Main function to use for detecting whether the vehicle reached a new strip in
        the current step. The old strip (represented as a bbox) will be gone forever
        return:
        crossed: a boolean value indicating whether a new strip is reached
        dist (optional): distance to the strip, value no specific meaning
        """
        self.counter += 1
        if not self.finished:
            crossed, dist = self.wayline.has_crossed(self.vehicle.transform)

            if crossed:
                self.int_counter += 1
                self._get_next_wayline()

            return crossed, dist
        return False, 0.0

    def _get_next_wayline(self):
        # make sure no index out of bound error
        curr_lb = self.look_back
        curr_idx = self.int_counter * self.interval
        while curr_idx + curr_lb < len(self.plan_lst):
            if curr_lb > self.look_back_max:
                self.int_counter += 1
                curr_lb = self.look_back
                curr_idx = self.int_counter * self.interval
                continue

            t1 = self.plan_lst[curr_idx]
            t2 = self.plan_lst[curr_idx + curr_lb]

            dx = t2.location.x - t1.location.x
            dz = t2.location.z - t1.location.z
            if abs(dx) < self.thres and abs(dz) < self.thres:
                curr_lb += 1
            else:
                self.wayline = WayLine(t1, t2)
                return
        # no next bbox
        print("finished all the iterations!")
        self.finished = True

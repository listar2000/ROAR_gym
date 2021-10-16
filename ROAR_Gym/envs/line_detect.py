from typing import Tuple
import cv2
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.utilities_module.data_structures_models import SensorsData, Transform, Location
from pathlib import Path
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.control_module.pid_controller import PIDController
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.agent import Agent
import numpy as np

class LineBBox(object):
    def __init__(self, transform1: Transform, transform2: Transform) -> None:
        self.x1, self.z1 = transform1.location.x, transform1.location.z
        self.x2, self.z2 = transform2.location.x, transform2.location.z
        print(self.x2, self.z2)
        self.pos_true = True
        self.thres = 1e-2
        self.eq = self._construct_eq()
        self.strip_list = None

        if self.eq(self.x1, self.z1) > 0:
            self.pos_true = False

    def _construct_eq(self):
        dz, dx = self.z2 - self.z1, self.x2 - self.x1

        if abs(dz) < self.thres:
            def vertical_eq(x, z):
                return x - self.x2
            return vertical_eq
        elif abs(dx) < self.thres:
            def horizontal_eq(x, z):
                return z - self.z2
            return horizontal_eq

        slope_ = dz / dx
        self.slope = -1 / slope_
        # print("tilted strip with slope {}".format(self.slope))
        self.intercept = -(self.slope * self.x2) + self.z2

        def linear_eq(x, z):
            return z - self.slope * x - self.intercept

        return linear_eq

    def has_crossed(self, transform: Transform):
        x, z = transform.location.x, transform.location.z
        dist = self.eq(x, z)
        return (dist > 0 if self.pos_true else dist < 0, dist)

    def get_visualize_locs(self, size=10):
        if self.strip_list is not None:
            return self.strip_list

        name = self.eq.__name__
        if name == 'vertical_eq':
            xs = np.repeat(self.x2, size)
            zs = np.arange(self.z2 - (size//2), self.z2 + (size//2))
        elif name == 'horizontal_eq':
            xs = np.arange(self.x2 - (size//2), self.x2 + (size//2))
            zs = np.repeat(self.z2, size)
        else:
            range_ = size * np.cos(np.arctan(self.slope))
            xs = np.linspace(self.x2 - range_/2, self.x2 + range_/2, num=size)
            zs = self.slope * xs + self.intercept

        self.strip_list = []
        for i in range(len(xs)):
            self.strip_list.append(Location(x=xs[i], y=0, z=zs[i]))

        return self.strip_list


"""""""""""""""""""""""""""""""""""""""""""""
You can ignore all the codes above this line
"""""""""""""""""""""""""""""""""""""""""""""

class LineDetectAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super().__init__(vehicle, agent_settings, **kwargs)
        self.agent_settings = agent_settings
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        self.plan_lst = list(self.mission_planner.produce_single_lap_mission_plan())

        self.kwargs = kwargs
        self.interval = self.kwargs.get('interval', 50)
        self.look_back  = self.kwargs.get('look_back', 5)
        self.look_back_max = self.kwargs.get('look_back_max', 10)
        self.thres = self.kwargs.get('thres', 1e-3)

        self.int_counter = 0
        self.counter = 0
        self.finished = False
        self._get_next_bbox()

        # the part about visualization
        self.occupancy_map = OccupancyGridMap(agent=self, threaded=True)

        occ_file_path = Path("../ROAR_Sim/data/easy_map_cleaned_global_occu_map.npy")
        self.occupancy_map.load_from_file(occ_file_path)

    def _get_next_bbox(self):
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
                self.bbox = LineBBox(t1, t2)
                return
        # no next bbox
        print("finished all the iterations!")
        self.finished = True

    def run_step(self, vehicle: Vehicle, sensors_data: SensorsData) -> Tuple[bool, float]:
        """
        Main function to use for detecting whether the vehicle reached a new strip in
        the current step. The old strip (represented as a bbox) will be gone forever
        return:
        crossed: a boolean value indicating whether a new strip is reached
        dist (optional): distance to the strip, value no specific meaning
        """
        self.counter += 1
        if not self.finished:
            crossed, dist = self.bbox.has_crossed(vehicle.transform)

            if crossed:
                self.int_counter += 1
                self._get_next_bbox()

            occ_map = self.occupancy_map.get_map(
                transform=vehicle.transform,
                view_size=(100, 100),
                arbitrary_locations=self.bbox.get_visualize_locs(size=20),
                arbitrary_point_value=0.5
            )

            cv2.imshow("", occ_map)
            cv2.waitKey(1)

            return crossed, dist

#         if self.kwargs.get("obstacle_coords", None) is not None:
#             # print("curr_transform", self.vehicle.transform)
#             points = self.kwargs["obstacle_coords"]
#             self.occupancy_map.update(points)
#             strip_list = self.bbox.get_visualize_locs(size=20)
#
#             # print(strip_list)
#             self.occupancy_map.visualize(self.vehicle.transform, view_size=(100, 100),
#                 strip_list=strip_list)

        return False, 0.0
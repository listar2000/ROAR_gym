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

occ_map =  OccupancyGridMap(agent=self, threaded=True)

occ_file_path = Path("../ROAR_Sim/data/berkeley_minor_global_occu_map.npy")
occ_map.load_from_file(occ_file_path)

print(occ_map)
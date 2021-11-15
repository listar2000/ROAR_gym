from ROAR.planning_module.mission_planner.mission_planner import (
    MissionPlanner,
)
from pathlib import Path
import logging
from typing import List, Optional
from ROAR.utilities_module.data_structures_models import Transform, Location, Rotation
from collections import deque
from ROAR.agent_module.agent import Agent
import numpy as np

from Discrete_PID.wayline import WayLine

def _read_data_file(file_path) -> List[List[float]]:
    """
    Read data file and generate a list of (x, y, z) where each of x, y, z is of type float
    Returns:
        List of waypoints in format of [x, y, z]
    """
    result = []
    with open(file_path.as_posix(), "r") as f:
        for line in f:
            result.append(_read_line(line=line))
    return result

def _raw_coord_to_transform(raw: List[float]) -> Optional[Transform]:
    """
    transform coordinate to Transform instance

    Args:
        raw: coordinate in form of [x, y, z, pitch, yaw, roll]

    Returns:
        Transform instance
    """
    if len(raw) == 3:
        return Transform(
            location=Location(x=raw[0], y=raw[1], z=raw[2]),
            rotation=Rotation(pitch=0, yaw=0, roll=0),
        )
    elif len(raw) == 6:
        return Transform(
            location=Location(x=raw[0], y=raw[1], z=raw[2]),
            rotation=Rotation(roll=raw[3], pitch=raw[4], yaw=raw[5]),
        )
    else:
        print(f"Point {raw} is invalid, skipping")
        return None

def _read_line(line: str) -> List[float]:
    """
    parse a line of string of "x,y,z" into [x,y,z]
    Args:
        line: comma delimetered line

    Returns:
        [x, y, z]
    """
    try:
        x, y, z = line.split(",")
        x, y, z = float(x), float(y), float(z)
        return [x, y, z]
    except:
        x, y, z, roll, pitch, yaw = line.split(",")
        return [float(x), float(y), float(z), float(roll), float(pitch), float(yaw)]


class WayLineMissionPlanner(MissionPlanner):
    """
    A mission planner that takes in 3 files that contains x,y,z coordinates, formulate into carla.Transform:
        1. left end of waylines
        2. right end of waylines
        3. calcualted waypoint
    """

    def run_in_series(self) -> deque:
        """
        Regenerate waypoints from file
        Find the waypoint that is closest to the current vehicle location.
        return a mission plan starting from that waypoint

        Args:
            vehicle: current state of the vehicle

        Returns:
            mission plan that start from the current vehicle location
        """
        super(WaypointFollowingMissionPlanner, self).run_in_series()
        return self.produce_mission_plan()

    def __init__(self, agent: Agent):
        super().__init__(agent=agent)
        self.logger = logging.getLogger(__name__)

        self.waypoints_path: Path = Path(self.agent.agent_setting.waypoint_file_path)
        self.left_waylines_path: Path = Path(self.agent.agent_setting.left_wayline_file_path)
        self.right_waylines_path: Path = Path(self.agent.agent_setting.right_wayline_file_path)

        self.num_laps = self.agent.agent_settings.num_laps

        self.waypoints_plan = self.produce_waypoints(laps = self.num_laps)
        self.waylines_plan = self.produce_waylines(laps = self.num_laps)

        self.waypoints_plan_backup = self.waypoints_plan.copy()
        self.waylines_plan_backup = self.waylines_plan.copy()

        self.logger.debug("Path Following Mission Planner Initiated.")

    def produce_waypoints(self, laps = 1) -> deque:
        """
        Generates a list of waypoints based on the input file path

        length of return list = (lenth of file) * laps

        :return a list of waypoint
        """
        raw_path: List[List[float]] = _read_data_file(self.waypoints_path)
        length = laps * len(raw_path)
        waypoints_list = deque(maxlen=length)
        
        if laps == 1:
            for coord in raw_path:
                if len(coord) == 3 or len(coord) == 6:
                    waypoints_list.append(_raw_coord_to_transform(coord)) 
        else:
            for coord in np.tile(raw_path, (laps, 1)):
                if len(coord) == 3 or len(coord) == 6:
                    waypoints_list.append(_raw_coord_to_transform(coord))
        self.logger.debug(f"Computed Mission path of length [{len(mission_plan)}]")
        return waypoints_list

    def produce_waylines(self, laps = 1):
        left_raw_path: List[List[float]] = _read_data_file(self.left_waylines_path)
        right_raw_path: List[List[float]] = _read_data_file(self.right_waylines_path)
        assert len(left_raw_path) == len(right_raw_path), "wayline dose not match"
        
        length = laps * len(raw_path)

        waylines_list = deque(maxlen=length)
        
        if laps == 1:
            for coord in raw_path:
                if len(coord) == 3 or len(coord) == 6:
                    mission_plan.append(self._raw_coord_to_transform(coord))
                self.logger.debug(f"Computed Mission path of length [{len(mission_plan)}]")
        else:
            for coord in np.tile(raw_path, (laps, 1)):
                if len(coord) == 3 or len(coord) == 6:
                    waypoints_list.append(self._raw_coord_to_transform(coord))
            self.logger.debug(f"Computed Mission path of length [{len(mission_plan)}]")

        return waypoints_list

    def restart(self):
       self.waylines_plan = self.waylines_plan_backup.copy()
       self.waypoints_plan = self.waypoints_plan_backup.copy()
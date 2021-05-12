try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.utilities_module.data_structures_models import Transform, Location, Rotation

from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any, Optional
import gym
import math
from collections import OrderedDict
import cv2


class OccuDebug(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        # action_space = target_speed, long_k, lat_k
        self.view_size = 100
        self.max_steering_angle = 0.5
        self.action_space = gym.spaces.Box(low=np.array([0.2, -self.max_steering_angle]),
                                           high=np.array([1, self.max_steering_angle]),
                                           dtype=np.float32)  # long (y) throttle, lat (x) steering,
        # observation_space = curr_speed, curr_throttle, curr_steering
        self.observation_space = gym.spaces.Box(low=-10, high=1,
                                                shape=(self.view_size, self.view_size, 1), dtype=np.float32)
        self._prev_speed = 0
        self._prev_waypoint = None
        self.debug_info: OrderedDict = OrderedDict()
        self.prev_location: Optional[Location] = None
        self.prev_next_waypoint: Optional[Location] = None
        self.dist_diff = 0

    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, dict]:
        """

        Args:
            action:

        Returns:

        """
        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)
        control = VehicleControl(throttle=action[0], steering=action[1])
        self.agent.kwargs["control"] = control

        before_dist = self.agent.vehicle.transform.location.distance(self.agent.local_planner.way_points_queue[
                                                                         self.agent.local_planner.get_curr_waypoint_index()].location)

        agent, reward, is_done, other_info = super(OccuDebug, self).step(action=action)
        after_dist = self.agent.vehicle.transform.location.distance(self.agent.local_planner.way_points_queue[
                                                                        self.agent.local_planner.get_curr_waypoint_index()].location)
        self.dist_diff = before_dist - after_dist
        obs = self._get_obs()
        return obs, reward, is_done, other_info

    def _get_obs(self) -> Any:
        if self.agent.occupancy_map is not None:
            occu_map = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                        view_size=(self.view_size, self.view_size),
                                                        vehicle_value=-10)
            occu_map = np.expand_dims(occu_map, axis=2)
        else:
            occu_map = np.zeros(shape=(self.view_size, self.view_size, 1))
        return occu_map

    def render(self, mode='ego'):
        super(OccuDebug, self).render(mode='ego')
        # if self.agent.front_depth_camera.data is not None:
        #     obs = self._get_obs()
        #     cv2.imshow("obs", obs.squeeze())
        #     cv2.waitKey(1)

    def get_reward(self) -> float:
        reward: float = 1.0
        if self.dist_diff <= 0:
            reward += 100
            self.debug_info["isGettingCloserToNextwaypoint"] = True
        else:
            reward -= 10
            self.debug_info["isGettingCloserToNextwaypoint"] = False

        if Vehicle.get_speed(self.agent.vehicle) < 10:
            reward -= 100
        if self.carla_runner.get_num_collision() > 0:
            reward -= 10000

        if abs(self.agent.vehicle.control.steering) > 0.3:
            reward -= 1000
        elif abs(self.agent.vehicle.control.steering) > 0.2:
            reward -= 500
        elif abs(self.agent.vehicle.control.steering) > 0.1:
            reward -= 100

        if abs(self.agent.vehicle.control.steering) < 0.1:
            reward += 1000
        self.debug_info["reward"] = reward
        return reward

    def _get_info(self) -> dict:
        self.debug_info["speed"] = Vehicle.get_speed(self.agent.vehicle)
        self.debug_info["throttle"] = self.agent.vehicle.control.throttle
        self.debug_info["steering"] = self.agent.vehicle.control.steering
        return self.debug_info

    def reset(self) -> Any:
        super(OccuDebug, self).reset()
        return self._get_obs()

    def _terminal(self) -> bool:
        is_done = super(OccuDebug, self)._terminal()
        if is_done:
            return True
        else:
            if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 1:
                return True
            else:
                return False

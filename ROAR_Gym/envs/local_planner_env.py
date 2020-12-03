from typing import Any, Tuple, Dict
from collections import OrderedDict
try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv
import gym
import numpy as np
from ROAR.utilities_module.data_structures_models import Transform, Location, Rotation
from pathlib import Path
import os
from ROAR.utilities_module.vehicle_models import Vehicle


class LocalPlannerEnv(ROAREnv):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # action space = next waypoint
        self.action_space = gym.spaces.Box(low=np.array([-1000, -1000, -1000]),
                                           high=np.array([1000, 1000, 1000]),
                                           dtype=float)
        # observation space = depth map, current throttle, current steering
        observation_space_dict: Dict[str, gym.spaces.Box] = {
            "depth": gym.spaces.Box(low=0, high=1, shape=(800, 600, 1), dtype=np.float64),
            "curr_throttle": gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float64),
            "curr_steering": gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float64)
        }

        size = 0
        for key in observation_space_dict.keys():
            shape = observation_space_dict[key].shape
            size += np.prod(shape)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float64)

        self._prev_speed = 0
        self._prev_waypoint = None

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        assert type(action) == list or type(action) == np.ndarray, f"Action is not recognizable"
        assert len(action) == 3, f"Action should be of length 3 but is of length [{len(action)}]."

        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)
        if len(self.agent.local_planner.way_points_queue) > 0:
            self._prev_waypoint = self.agent.local_planner.way_points_queue[-1].to_array()
        next_waypoint = Transform(location=Location(x=action[0], y=action[1], z=action[2]))
        self.agent.kwargs["next_waypoint"] = next_waypoint

        agent, reward, is_done, other_info = super(LocalPlannerEnv, self).step(action=action)
        obs = self._get_obs()
        return obs, reward, is_done, other_info

    def get_reward(self) -> float:
        reward: float = 0.0
        current_speed = Vehicle.get_speed(self.agent.vehicle)
        if self.carla_runner.get_num_collision() > self.max_collision_allowed:
            reward -= 1000000
        if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 1:
            # so that the agent is encouraged to move
            reward -= 100000
        if abs(self.agent.vehicle.control.steering) > 0.3:
            # prevent it from over steering
            reward -= 10
        if current_speed < 10:
            # again, agent is encouraged to move
            reward = 0
        else:
            reward += current_speed

        if current_speed < 80:
            # i know that you can drive around the track at least with this speed
            reward -= 100

        if current_speed > self._prev_speed:
            # agent is incentivized to go faster
            reward += 1

        return reward

    def _get_info(self) -> dict:
        curr_location = self.agent.vehicle.transform.location
        next_waypoint_location = self.agent.local_planner.way_points_queue[-1].location
        info = OrderedDict()
        info['speed'] = Vehicle.get_speed(self.agent.vehicle)
        info["Current Location"] = curr_location
        info["Next Waypoint Location"] = next_waypoint_location
        info['diff'] = curr_location.distance(next_waypoint_location)
        info['throttle'] = self.agent.vehicle.control.throttle
        info['steering'] = self.agent.vehicle.control.steering
        return info

    def _get_obs(self) -> Any:
        if self.agent.front_depth_camera.data is not None:
            depth_image = self.agent.front_depth_camera.data.copy()
        else:
            depth_image = np.zeros(shape=(800, 600, 1))

        curr_throttle = np.array([self.agent.vehicle.control.throttle])
        curr_steering = np.array([self.agent.vehicle.control.steering])
        obs = [np.ravel(depth_image), curr_throttle, curr_steering]
        return np.concatenate(obs)

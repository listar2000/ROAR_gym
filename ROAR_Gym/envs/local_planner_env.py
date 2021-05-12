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
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
import cv2
import time


class LocalPlannerEnv(ROAREnv):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # action space = next waypoint
        self.view_size = 100
        self.action_space = gym.spaces.Box(low=np.array([0, -1]),
                                           high=np.array([1, 1]),
                                           dtype=np.float16)  # throttle, steering
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.view_size, self.view_size, 3), dtype=np.uint8)

        self._prev_speed = 0
        self._prev_location = None
        self.correct_next_waypoint_world = None
        self.correct_next_waypoint_occu = None
        self.my_guess_next_waypoint_world = None
        self.my_guess_next_waypoint_occu = None
        self.reward = 0
        self.action = None
        self.info = OrderedDict()

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        assert type(action) == list or type(action) == np.ndarray, f"Action is not recognizable"
        assert len(action) == 2, f"Action should be of length 2 but is of length [{len(action)}]."
        self.action = action
        self.agent.kwargs["control"] = VehicleControl(throttle=action[0], steering=action[1])
        obs, reward, is_done, other_info = super(LocalPlannerEnv, self).step(action)

        # obs, reward, is_done, other_info = self._get_obs(), self.get_reward(), self._terminal(), self._get_info()

        return obs, reward, is_done, other_info

    def get_reward(self) -> float:
       return 0.0

    def _get_info(self) -> dict:
        self.info['speed'] = Vehicle.get_speed(self.agent.vehicle)
        self.info["steering"] = self.agent.vehicle.control.steering
        return self.info

    def _get_obs(self) -> Any:
        # obs = np.zeros(shape=(self.view_size, self.view_size, 3))
        obs = self.agent.get_obs()
        return obs

    def _terminal(self) -> bool:
        if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 10:
            return True
        elif self.carla_runner.get_num_collision() > 2:
            return True
        else:
            return False

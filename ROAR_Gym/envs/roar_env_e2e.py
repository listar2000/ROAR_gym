try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any
import gym
import math
from collections import OrderedDict
from gym.spaces import Discrete, Box
import cv2

# Define the discrete action space
DISCRETE_ACTIONS = {
    0: [0.0, 0.0],  # Coast
    1: [0.0, -0.5],  # Turn Left
    2: [0.0, 0.5],  # Turn Right
    3: [1.0, 0.0],  # Forward
    4: [-0.5, 0.0],  # Brake
    5: [1.0, -0.5],  # Bear Left & accelerate
    6: [1.0, 0.5],  # Bear Right & accelerate
    7: [-0.5, -0.5],  # Bear Left & decelerate
    8: [-0.5, 0.5],  # Bear Right & decelerate
}
FRAME_STACK = 10
CONFIG = {
    "x_res": 80,
    "y_res": 80
}


class ROAREnvE2E(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        self.action_space = Discrete(len(DISCRETE_ACTIONS))
        self.observation_space = Box(-1, 1, shape=(FRAME_STACK, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        print("Running step")
        obs = []
        rewards = []

        for i in range(FRAME_STACK):
            self.agent.kwargs["control"] = VehicleControl(throttle=DISCRETE_ACTIONS[action][0],
                                                          steering=DISCRETE_ACTIONS[action][1])
            ob, reward, is_done, info = super(ROAREnvE2E, self).step(action)
            obs.append(ob)
            rewards.append(reward)
            if is_done:
                break
        self.render()
        return np.array(obs), sum(rewards), False, dict()

    def get_reward(self) -> float:
        return 0.0

    def _get_obs(self) -> np.ndarray:
        depth_image = self.agent.front_depth_camera.data.copy()
        data = cv2.resize(depth_image, (CONFIG["x_res"], CONFIG["y_res"]), interpolation=cv2.INTER_AREA)
        # data = np.expand_dims(data, 2)
        return data  # height x width x 1 array

    def reset(self) -> Any:
        super(ROAREnvE2E, self).reset()
        return self.agent.front_depth_camera.data
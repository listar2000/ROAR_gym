try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import VehicleControl
from typing import Optional, Tuple
import numpy as np
from typing import List, Any
from PIL import Image
from collections import deque
from gym.spaces import Discrete, Box
import cv2

# Define the discrete action space
DISCRETE_ACTIONS = {
    0: [0.0, 0.0],  # Coast
    1: [0.0, -0.5],  # Turn Left
    2: [0.0, 0.5],  # Turn Right
    3: [1.0, 0.0],  # Forward
    4: [0.5, 0.0],  # Brake
    5: [1.0, -0.5],  # Bear Left & accelerate
    6: [1.0, 0.5],  # Bear Right & accelerate
    7: [0.5, -0.5],  # Bear Left & decelerate
    8: [0.5, 0.5],  # Bear Right & decelerate
}
FRAME_STACK = 5
FRAME_SKIP = 10
RENDER_MAP = True
# about 10 seconds of idle
DONE_THRESHOLD = 500
CONFIG = {
    "x_res": 80,
    "y_res": 80
}


class ROAREnvE2E(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        self.action_space = Discrete(len(DISCRETE_ACTIONS))
        self.observation_space = Box(-1, 1, shape=(FRAME_STACK, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        self.prev_speed = 0
        self.prev_dist_to_strip = 0
        self.min_prev_dist = 1e10
        self.obs_deque = None
        self.yaw_deque = None
        self.prev_collision_num = 0

        # a list of counters for early termination of current episode
        self.frame_counter = 0
        self.negative_reward_counter = 0

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        total_reward = 0

        self.agent.kwargs["control"] = VehicleControl(throttle=DISCRETE_ACTIONS[action][0],
                                                          steering=DISCRETE_ACTIONS[action][1])
        for _ in range(FRAME_SKIP):
            ob, reward, is_done, info = super(ROAREnvE2E, self).step(action)
            total_reward += reward
            self.obs_deque.append(ob)
            # well what a long sequence of dots...
            self.yaw_deque.append(self.agent.vehicle.transform.rotation.yaw)
            self.frame_counter += 1

            if RENDER_MAP:
                self.render()
            
            if is_done:
                break
        
        new_obs = self._stack_obs() 

        # "reward": total_reward, "action": DISCRETE_ACTIONS[action]
        return new_obs, total_reward, self._terminal(), {}

    def _terminal(self) -> bool:
        if self.carla_runner.get_num_collision() > self.max_collision_allowed:
            return True
        elif self.agent.finished:
            return True
        else:
            return self.negative_reward_counter > DONE_THRESHOLD

    def get_reward(self, action) -> float:
        # below are the original reward (legacy)
        # # prep for reward computation
        # reward = 0
        # curr_dist_to_strip = self.agent.curr_dist_to_strip

        # # reward computation
        # reward += 0.5 * (Vehicle.get_speed(self.agent.vehicle) - self.prev_speed)
        # reward += abs(self.agent.vehicle.control.steering)
        # reward += np.clip(self.prev_dist_to_strip - curr_dist_to_strip, -10, 10)
        # reward -= self.carla_runner.get_num_collision()

        # # log prev info for next reward computation
        # self.prev_speed = Vehicle.get_speed(self.agent.vehicle)
        # self.prev_dist_to_strip = curr_dist_to_strip
        # return reward

        # below are the reward imitating carracing-v0
        # if the car does nothing (e.g. not moving), we give a -0.1 penalty for inaction
        reward = -0.1

        # reward related to collision, we give a huge penalty to this
        num_collision = self.carla_runner.get_num_collision()
        if self.prev_collision_num < num_collision:
            reward -= 5
        self.prev_collision_num = num_collision

        # the agent has got closer to the next strip
        if self.min_prev_dist > self.agent.curr_dist_to_strip:
            reward += 0.1
        self.min_prev_dist = min(self.min_prev_dist, self.agent.curr_dist_to_strip)
        if self.agent.has_crossed:
            reward += 5
            self.min_prev_dist = 1e10

        # OPTIONAL: amplify the result of action at high speed
        throttle = DISCRETE_ACTIONS[action][0]
        if throttle > 0.9 and reward > 0:
            reward *= 1.5

        # if we are out of the `grace period` of 200 frames and 
        if self.frame_counter > 200 and reward < 0:
            self.negative_reward_counter += 1
        else:
            self.negative_reward_counter = 0

        return reward


    def _get_obs(self) -> np.ndarray:
        # star edited this: it's better to set the view_size directly instead of doing resize
        data = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                egocentric=False,
                                                view_size=(CONFIG["x_res"], CONFIG["y_res"]),
                                                arbitrary_locations=self.agent.bbox.get_visualize_locs(size=20),
                                                arbitrary_point_value=0.5
                                                )
        if RENDER_MAP:
            cv2.imshow("data", data) # uncomment to show occu map
            cv2.waitKey(1)
        return data  # height x width x 1 array
    
    # copy & paste from the carracing-v0 environment
    def _stack_obs(self) -> np.ndarray:
        raw_obs = np.array(self.obs_deque)
        stacked_obs = np.zeros_like(raw_obs)
        for i in range(FRAME_STACK):
            image = Image.fromarray(raw_obs[i])
            image = image.rotate(-self.yaw_deque[0])
            stacked_obs[i] = np.asarray(image)
        return stacked_obs

    def reset(self) -> Any:
        super(ROAREnvE2E, self).reset()
        self.obs_deque = deque(np.zeros((FRAME_STACK, 80, 80)).tolist(), maxlen=FRAME_STACK)
        self.yaw_deque = deque([0] * FRAME_STACK, maxlen=FRAME_STACK)

        self.frame_counter = 0
        self.negative_reward_counter = 0
        return self._stack_obs()

try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from math import sqrt
from numpy.core.numeric import cross
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any
import gym
from collections import OrderedDict

from Discrete_PID.valid_pid_action import VALID_ACTIONS, MAX_SPEED, TARGET_SPEED
from Discrete_PID.wayline import WayLine

class ROARPIDEnv(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        # action_space = speed, long_k, lat_k

        #self.action_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]),
        #                                  high=np.array([200, 1, 1, 1, 1, 1, 1]), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(len(VALID_ACTIONS))

        # observation_space = curr_speed, curr_transform, next_waypoint
        self.observation_space = gym.spaces.Box(low=np.array([-200,
                                                              -1000, -1000, -1000, -360, -360, -360,
                                                              -1000, -1000, -1000, -360, -360, -360,
                                                              ]),
                                                high=np.array([200,
                                                               1000, 1000, 1000, 360, 360, 360,
                                                               1000, 1000, 1000, 360, 360, 360]),
                                                dtype=np.float32)
        self._prev_speed = 0
        self.target_loc = None
        self.collisions = 0

    def step(self, action_num: int) -> Tuple[np.ndarray, float, bool, dict]:
        """

        Args:
            action_num: the index of action, which will give us  array of [target_speed,
                                                                          long_kp, long_kd, long_ki,
                                                                          lat_kp, lat_kd, lat_ki]

        Returns:

        """
        action = VALID_ACTIONS[int(action_num)]
        lat_k_p, lat_k_d, lat_k_i = action[0], action[1], action[2]

        self.agent.kwargs["lat_k_p"] = lat_k_p
        self.agent.kwargs["lat_k_d"] = lat_k_d
        self.agent.kwargs["lat_k_i"] = lat_k_i
        self.render()

        obs, reward, is_done, other_info = super(ROARPIDEnv, self).step(action=action)
        other_info["reward"] = reward
        return obs, reward, is_done, other_info

    def _get_obs(self) -> Any:
        curr_speed = np.array([Vehicle.get_speed(self.agent.vehicle)])
        curr_transform = self.agent.vehicle.transform.to_array()
        if len(self.agent.local_planner.way_points_queue) > 0:
            next_waypoint_transform = self.agent.local_planner.way_points_queue[0].to_array()
        else:
            next_waypoint_transform = curr_transform
        return np.append(np.append(curr_speed, curr_transform), next_waypoint_transform)

    def get_reward(self) -> float:
        # prep for reward computation
        reward = -15

        # reward computation
        # reward += 0.5 * (Vehicle.get_speed(self.agent.vehicle) - self._prev_speed)
        reward += self.agent.vehicle.control.throttle
        # reward += np.clip(self.prev_dist_to_strip - curr_dist_to_strip, -10, 10)
        # reward -= self.carla_runner.get_num_collision()
        
        # log prev info for next reward computation
        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)
        if self._prev_speed > TARGET_SPEED:
            reward += 10
        
        cross_rwd = self.wayline_reward()
        reward += cross_rwd
        if cross_rwd > 0:
            print("wayline reward: ", cross_rwd)
 
        if self.carla_runner.get_num_collision() > self.collisions:
            reward += -100000
            self.collisions = self.carla_runner.get_num_collision()
        if self.carla_runner.get_num_collision() > self.max_collision_allowed:
            reward += -100000000

        return reward

    # def get_reward(self, next_waypt = (0,0)) -> float:
    #     reward: float = 0.0
    #     #target_speed = self.agent.kwargs["target_speed"]
    #     current_speed = Vehicle.get_speed(self.agent.vehicle)
    #     current_steering = self.agent.vehicle.control.steering

    #     # 1. penalize for staying in the map
    #     reward -= 10

    #     # 2. big penalize for collision
    #     if self.carla_runner.get_num_collision() > self.collisions:
    #         reward += -10000
    #         self.collisions = self.carla_runner.get_num_collision()
    #     if self.carla_runner.get_num_collision() > self.max_collision_allowed:
    #         reward += -10000000
            
    #     # 3. reward fast, penalize slow
    #     # option 1: 
    #     # reward += (current_speed - TARGET_SPEED)
    #     # option 2:
    #     if current_speed < TARGET_SPEED:
    #         reward -= 20

    #     # 4. reward speeding up, penalize slowing down
    #     if current_speed > self._prev_speed:
    #         reward += 30
    #     elif current_speed <= self._prev_speed and current_steering <= 0.1:
    #         # slowing down while driving straight
    #         reward -= 30

    #     # 5. penalize steering:
    #     if abs(self.agent.vehicle.control.steering) >= 0.25:
    #         # prevent it from over steering
    #         reward -= 50
    #     elif abs(self.agent.vehicle.control.steering) >= 0.4:
    #         reward -= 120
    #     elif abs(self.agent.vehicle.control.steering) >= 0.6:
    #         reward -= 400


    #     current_transform = self.agent.vehicle.transform #.location.to_array()
        
    #     # 6. reward by wayline / circle
    #     reward += self.circle_wp_reward(current_transform , self.agent.local_planner.target_wp)
    #     # reward += self.wayline_reward()
        
    #     # if the agent is able to complete a lap, reward heavy
    #     if self.agent.is_done:
    #         reward += 1000000
        
    #     return reward

    def _get_info(self) -> dict:
        info_dict = OrderedDict()

        info_dict["speed"] = Vehicle.get_speed(self.agent.vehicle)
        #info_dict["target_speed"] = self.agent.kwargs["target_speed"]

        info_dict["throttle"] = self.agent.vehicle.control.throttle
        info_dict["steering"] = self.agent.vehicle.control.steering

        info_dict["lat_kp"] = self.agent.kwargs["lat_k_p"]
        info_dict["lat_kd"] = self.agent.kwargs["lat_k_d"]
        info_dict["lat_ki"] = self.agent.kwargs["lat_k_i"]
        return info_dict

    def reset(self) -> Any:
        super(ROARPIDEnv, self).reset()
        return self._get_obs()

    def circle_wp_reward(self, cur_wp, next_wp, thre = 0.5):
        reaching_reward = 0
        if disc_pt_to_pt(cur_wp, next_wp) <= thre:
            reaching_reward += 300
        if np.abs(cur_wp.rotation.roll - next_wp.rotation.roll) <= thre:
            reaching_reward += 100
        return reaching_reward

    def wayline_reward(self, max_reward = 100, clip_near = 0.5, clip_far = 3):
        """
        Args:
            max_reward: maximum reward achievable through passing wayline
            clip_near: the threshold (in m) that any distance smaller than it will receive max reward
            clip_far: the threshold (in m) that any distance larger than it will receive zero reward
        """
        if not self.agent.has_crossed:
            return 0

        print("crossing!")      
        target_waypoint: Transform = self.agent.prev_wp

        x1, z1 = self.agent.hit_loc
        x2, z2 = target_waypoint.location.x, target_waypoint.location.z
        #print("reward hit: ", (x1, z1))
        #print("reward taget: ", (x2, z2))
        dist = disc_pt_to_pt(x1, z1, x2, z2)
        
        #print(dist)
        frac = (clip_far - dist) / (clip_far - clip_near)
        frac = np.clip(frac, 0, 1)
        print(frac)
        #print(frac * max_reward)
        return frac * max_reward

# helper functions for linear reward
def disc_pt_to_line(wp, wl):
    return np.abs(wl.eq(wp.location.x, wp.location.z)) / np.power(wl.intercept**2 + wl.self.slope**2 , 1/2) 

def disc_pt_to_pt(x1, z1, x2, z2):
    return sqrt((x1 - x2) ** 2 + (z1 - z2) ** 2)
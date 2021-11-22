try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from numpy.core.numeric import cross
from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any
import gym
import math
from collections import OrderedDict

from Discrete_PID.valid_pid_action import VALID_ACTIONS, MAX_SPEED, TARGET_SPEED
from Discrete_PID.wayline import WayLine
from scipy.spatial import distance
from scipy.stats import beta

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

        action = VALID_ACTIONS[action_num]
        #assert type(action) == list or type(action) == np.ndarray, f"Action is of type {type(action)}"
        #assert len(action) == 7, f"Action of shape {np.shape(action)} is not correct"

        #target_speed = action[0]
        #long_k_p, long_k_d, long_k_i = action[1], action[2], action[3]
        lat_k_p, lat_k_d, lat_k_i = action[0], action[1], action[2]
        #self.agent.kwargs["long_k_p"] = long_k_p
        #self.agent.kwargs["long_k_d"] = long_k_d
        #self.agent.kwargs["long_k_i"] = long_k_i
        #self.agent.kwargs["target_speed"] = target_speed
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
        reward = -1
        crossed = self.agent.has_crossed

        # reward computation
        # reward += 0.5 * (Vehicle.get_speed(self.agent.vehicle) - self._prev_speed)
        reward += self.agent.vehicle.control.throttle
        # reward += np.clip(self.prev_dist_to_strip - curr_dist_to_strip, -10, 10)
        # reward -= self.carla_runner.get_num_collision()
        if crossed:
            print("reward crossed")
            reward += 100

        # log prev info for next reward computation
        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)
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

    def wayline_reward(self, cur_transform, target_wayline, target_waypoint,thre = 0.5):
        """
        Args:
            cur_loc : Transform : current vehicle location
            next_wl: WayLine
        """
        reaching_reward = 0

        assert target_wayline.has_crossed(target_waypoint), "optimal waypoint should lie on the wayline"

        fra_to_left = disc_pt_to_pt(cur_transform, target_wayline.left) / disc_pt_to_pt(target_wayline.left, target_wayline.right)
        fra_optimal = disc_pt_to_pt(target_waypoint, target_wayline.left) / disc_pt_to_pt(target_wayline.left, target_wayline.right)

        if not target_wayline.has_crossed(cur_transform):
            reaching_reward += 0
        else:
            reaching_reward += (1 - np.abs(fra_optimal - fra_to_left)) * 1000 

        #if disc_pt_to_line(cur_transform, target_wayline) <= thre : 
            #reaching_reward += 500

        #if disc_pt_to_pt(cur_transform, target_waypoint) <= thre:
            #reaching_reward += 300

        #if np.abs(cur_transform.rotation.roll - target_waypoint.rotation.roll) <= thre:
            #reaching_reward += 100

        return reching_reward

def disc_pt_to_line(wp, wl):
    return np.abs(wl.eq(wp.location.x, wp.location.z)) / np.power(wl.intercept**2 + wl.self.slope**2 , 1/2) 

def disc_pt_to_pt(pt1, pt2):
    return distance.euclidean(
            pt1.location.to_tuple(),
            pt2.location.to_tuple(),
        ) 
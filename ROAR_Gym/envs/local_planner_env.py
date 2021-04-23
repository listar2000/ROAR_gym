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
import cv2
import time


class LocalPlannerEnv(ROAREnv):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        # action space = next waypoint
        self.action_space = gym.spaces.Box(low=np.array([0, 0]),
                                           high=np.array([200, 200]),
                                           dtype=np.int)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(200, 200, 1), dtype=np.float32)

        self._prev_speed = 0
        self._prev_location = None
        self.correct_next_waypoint_world = None
        self.correct_next_waypoint_occu = None
        self.my_guess_next_waypoint_world = None
        self.my_guess_next_waypoint_occu = None
        self.reward = 0

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        assert type(action) == list or type(action) == np.ndarray, f"Action is not recognizable"
        assert len(action) == 2, f"Action should be of length 2 but is of length [{len(action)}]."
        start_init = time.time()
        # start = time.time()
        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)
        self._prev_location = self.agent.vehicle.transform.location
        curr_occu_map = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                         view_size=(200, 200))
        occu_map_vehicle_center = np.array(list(zip(*np.where(curr_occu_map == np.min(curr_occu_map))))[0])
        if len(self.agent.traditional_local_planner.way_points_queue) > 0:
            self.correct_next_waypoint_world = self.agent.traditional_local_planner.way_points_queue[0]

            diff = np.array([self.correct_next_waypoint_world.location.x,
                             self.correct_next_waypoint_world.location.z]) - \
                   np.array([self.agent.vehicle.transform.location.x,
                             self.agent.vehicle.transform.location.z])
            self.correct_next_waypoint_occu = occu_map_vehicle_center + diff

            self.my_guess_next_waypoint_occu = np.array(action)
            self.my_guess_next_waypoint_world = self.agent.occupancy_map.cropped_occu_to_world(
                cropped_occu_coord=self.my_guess_next_waypoint_occu,
                vehicle_transform=self.agent.vehicle.transform,
                occu_vehicle_center=occu_map_vehicle_center)

        self.agent.kwargs["next_waypoint"] = self.my_guess_next_waypoint_world
        # print(f"NN Setting took: {time.time() - start} -> {1/ (time.time() - start)}")
        # start = time.time()
        self.clock.tick()
        self.carla_runner.world.tick(self.clock)

        # print(f"Ticking: {time.time() - start} -> {1 / (time.time() - start)}")
        # start = time.time()

        self.carla_runner.fetch_data_async()

        # print(f"Convert data carla -> agent: {time.time() - start} -> {1 / (time.time() - start)}")
        # start = time.time()

        agent_control = self.agent.run_step(vehicle=self.carla_runner.vehicle_state,
                                            sensors_data=self.carla_runner.sensor_data)

        # print(f"Agent run step: {time.time() - start} -> {1 / (time.time() - start)}")
        # start = time.time()

        carla_control = self.carla_runner.carla_bridge.convert_control_from_agent_to_source(agent_control)

        # print(f"convert data agent -> carla: {time.time() - start} -> {1 / (time.time() - start)}")
        # start = time.time()

        self.carla_runner.world.player.apply_control(carla_control)
        # print(f"Carla Run took {time.time() - start} -> {1/ (time.time() - start)}")
        # start = time.time()
        obs, reward, is_done, other_info = self._get_obs(), self.get_reward(), self._terminal(), self._get_info()
        # print(f"Get Output took {time.time() - start} -> {1/ (time.time() - start)}")
        # print(f"Final overall time took {time.time() - start_init} -> {1/ (time.time() - start_init)}")
        # input("Enter to continue")
        return obs, reward, is_done, other_info

    def get_reward(self) -> float:
        """
        Reward policy:
            Surviving = +1
            Going forward (positive velocity) = +1
            Going toward a waypoint = +10
            Speed < 10 = -10
            Speed > 80 = +50
            Collision = -10000
            abs(steering) > 0.5 = -100
        Returns:
            reward according to the aforementioned policy
        """
        reward: float = 1.0

        if Vehicle.get_speed(self.agent.vehicle) > 10:
            reward += 1
        if len(self.agent.traditional_local_planner.way_points_queue) > 0 and self._prev_location is not None:
            next_waypoint: Transform = self.agent.traditional_local_planner.way_points_queue[0]
            curr_location: Location = self.agent.vehicle.transform.location
            if curr_location.distance(next_waypoint.location) < curr_location.distance(self._prev_location):
                print("Getting closer to next waypoint!!!")
                reward += 1000
        if Vehicle.get_speed(self.agent.vehicle) < 10:
            reward -= 10
        if Vehicle.get_speed(self.agent.vehicle) > 80:
            reward += 50
        if self.carla_runner.get_num_collision() > 0:
            reward -= 10000
        if abs(self.agent.vehicle.control.steering) > 0.5:
            reward -= 100

        self.reward = reward
        return reward

    def _get_info(self) -> dict:
        info = OrderedDict()
        info['speed'] = Vehicle.get_speed(self.agent.vehicle)
        info['reward'] = self.reward
        # info["Current Location"] = curr_location
        # info["Next Waypoint Location"] = next_waypoint_location
        info["num_collision"] = self.carla_runner.get_num_collision()
        info["correct_next_waypoint_world"] = self.correct_next_waypoint_world
        info["my_guess_next_waypoint_world"] = self.my_guess_next_waypoint_world
        info["correct_next_waypoint_occu"] = self.correct_next_waypoint_occu
        info["my_guess_next_waypoint_occu"] = self.my_guess_next_waypoint_occu
        return info

    def _get_obs(self) -> Any:
        if self.agent.occupancy_map is not None:
            occu_map = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                        view_size=(200, 200))
            occu_map = np.expand_dims(occu_map, axis=2)
        else:
            occu_map = np.zeros(shape=(200, 200, 1))
        return occu_map

    def _terminal(self) -> bool:
        if self.agent.time_counter > 100 and Vehicle.get_speed(self.agent.vehicle) < 10:
            return True
        elif self.carla_runner.get_num_collision() > 2:
            return True
        else:
            return False

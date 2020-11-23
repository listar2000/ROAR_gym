from ROAR_Gym.envs.roar_env import ROAREnv
from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from typing import Tuple


class ROARPIDEnv(ROAREnv):
    def step(self, action: VehicleControl) -> Tuple[Agent, float, bool, dict]:
        return super(ROARPIDEnv, self).step(action=action)

    def get_reward(self) -> float:
        return -1

import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())
import gym
import ROAR_Gym
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig


def main():
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config
    }
    env = gym.make('roar-v0', params=params)
    # obs = env.reset()
    #
    while True:
        action = [1.0, 0.0]
        obs, reward, is_done, info = env.step(action)
        env.render()
    #
    #     if is_done:
    #         print("IS DONE")
    #         obs = env.reset()


if __name__ == '__main__':
    main()

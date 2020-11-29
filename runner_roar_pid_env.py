import warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path
sys.path.append(Path(os.getcwd()).parent.as_posix())
import gym
import ROAR_Gym
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.agent import Agent
from ROAR.agent_module.rl_pid_agent import RLPIDAgent
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import DDPG



def main():
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": RLPIDAgent
    }
    env = gym.make('roar-pid-v0', params=params)
    env.reset()
    # TODO, i don't think this tensorboard thing is working lol
    model = DDPG(LnMlpPolicy, env=env, verbose=1, render=True,
                 tensorboard_log="./output/tensorboard/pid/",
                 full_tensorboard_log=True)  # full tensorboard log can take up space quickly

    model.learn(total_timesteps=int(2e5))


if __name__ == '__main__':
    main()

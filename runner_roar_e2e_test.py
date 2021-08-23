import argparse
import os
import sys
from pathlib import Path
from stable_baselines3.dqn import DQN

sys.path.append(Path(os.getcwd()).parent.as_posix())
import gym
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.rl_depth_e2e_agent import RLDepthE2EAgent

try:
    from ROAR_Gym.envs.roar_env import LoggingCallback
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import LoggingCallback

if __name__ == '__main__':

    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": RLDepthE2EAgent,
        "max_collision": 3,
    }

    parser = argparse.ArgumentParser(description='Play CarRacing by the trained model.')
    parser.add_argument('-m', '--model', required=True, help='The `.zip` file of the trained model.')
    parser.add_argument('-e', '--episodes', type=int, default=1, help='The number of episodes should the model plays.')
    parser.add_argument('-r', '--render', action='store_true', help='render the env or not')
    args = parser.parse_args()

    env = gym.make('roar-e2e-v0', params=params)
    env.reset()

    model = DQN.load(args.model)

    reward_list = [0 for _ in range(args.episodes)]
    for i in range(args.episodes):
        obs = env.reset()
        
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            reward_list[i] += reward
    
    print(f"rewards: {[round(reward, 2) for reward in reward_list]}")
    print(f"mean reward: {sum(reward_list) / args.episodes}")
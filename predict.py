import warnings
import logging
from typing import Optional, Dict

import numpy as np

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())
import gym
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig

#from ROAR.agent_module.rl_pid_agent import RLPIDAgent
from Discrete_PID.discrete_rl_pid_agent import RLPIDAgent
from Discrete_PID.valid_pid_action import init_actions_space
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3 import DQN


from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps, CallbackList

try:
    from ROAR_Gym.envs.roar_env import LoggingCallback
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import LoggingCallback


def main():
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": RLPIDAgent
    }

    A = init_actions_space()

    env = gym.make('roar-pid-v0', params=params)

    latest_model_path = find_latest_model()
    model = DQN.load(latest_model_path, print_system_info=True)   
    model.tensorboard_log = "./predict"

    env.reset()
    obs = np.array([0, -809.52062988, 75.64264679, -689.72875977, 0, 0, -90.00036907, -804.28, 75.04, -689.73, 0, 0, 0])
    
    i = 0
    while True:
        action, _states = model.predict(obs)
        #print(action)
        obs, rewards, is_done, info = env.step(action)
        if is_done:
            break
    print("10 lapses finished")

def find_latest_model(root_path: Path = Path(os.getcwd())) -> Optional[Path]:
    import os
    from pathlib import Path
    logs_path = (root_path / "output" / "discrete_pid_logs")
    if logs_path.exists() is False:
        print(f"No previous record found in {logs_path}")
        return None
    paths = sorted((root_path / "output" / "discrete_pid_logs").iterdir(), key=os.path.getmtime)
    
    paths_dict: Dict[int, Path] = {
        int(path.name.split("_")[2]): path for path in paths
    }
    
    if paths_dict is None:
        return None

    latest_model_file_path: Optional[Path] = paths_dict.get(max(paths_dict.keys()), None)
    print(latest_model_file_path)
    return latest_model_file_path


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.INFO)
    logging.getLogger("Controller").setLevel(logging.ERROR)
    logging.getLogger("SimplePathFollowingLocalPlanner").setLevel(logging.ERROR)
    main()



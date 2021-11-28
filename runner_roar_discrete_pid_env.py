import warnings
import logging
from typing import Optional, Dict

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

    print(A)

    env = gym.make('roar-pid-v0', params=params)
    env.reset()

    model_params = dict(
        verbose= 1,
        #"render": True,
        tensorboard_log =  "./output/tensorboard/discrete_pid/",
        # more arguments here 
        learning_rate = 0.001,
        buffer_size=5000,
        batch_size=64,
        learning_starts=10000,
        gamma=0.95,
        #train_freq=(4, "step"),
        gradient_steps=1,
        target_update_interval=1000,
        exploration_initial_eps=1,
        exploration_final_eps=0.1,
        exploration_fraction=0.2
    )

    latest_model_path = find_latest_model(Path(os.getcwd()))
    if latest_model_path is None:
        model = DQN(MlpPolicy, env=env, **model_params)  # full tensorboard log can take up space quickly
    else:
        # print(11111, latest_model_path)
        model = DQN.load(latest_model_path, env = env, print_system_info=True, **model_params)
        # model.get_env().reset()
        model.tensorboard_log = "./output/tensorboard/discrete_pid/"


    logging_callback = LoggingCallback(model=model)
    checkpoint_callback = CheckpointCallback(save_freq=1000, verbose=2, save_path='./output/discrete_pid_logs')
    event_callback = EveryNTimesteps(n_steps=1000, callback=logging_callback)
    callbacks = CallbackList([checkpoint_callback, event_callback])
    tot_t = int(1e6)
    model = model.learn(total_timesteps=tot_t, callback=callbacks, reset_num_timesteps=False)
    #path = f"output/pid_dqn_{datetime.now()}"
    #path = "output/pid_dqn_model" + datetime.now().isoformat(timespec='minutes') 
    path = "output/discrete_pid_logs/rl_model_" + str(tot_t + 1) + "_step"
    model.save(path)


def find_latest_model(root_path: Path) -> Optional[Path]:
    import os
    from pathlib import Path
    logs_path = (root_path / "output" / "discrete_pid_logs")
    if logs_path.exists() is False:
        print(f"No previous record found in {logs_path}")
        return None
    paths = sorted((root_path / "output" / "discrete_pid_logs").iterdir(), key=os.path.getmtime)
  
   
    #i = 1
    #for path in paths:
    #     print(i, path.name.split("_"))
    #     i += 1
    
    paths_dict: Dict[int, Path] = {
        int(path.name.split("_")[2]): path for path in paths
    }
    
    if paths_dict is None:
        return None

    latest_model_file_path: Optional[Path] = paths_dict.get(max(paths_dict.keys()), None)
    return latest_model_file_path


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt="%H:%M:%S", level=logging.INFO)
    logging.getLogger("Controller").setLevel(logging.ERROR)
    logging.getLogger("SimplePathFollowingLocalPlanner").setLevel(logging.ERROR)
    main()


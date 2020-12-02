from typing import Any, Tuple, Dict

try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv


class BehaviorPlannerEnv(ROAREnv):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # action space = current transform, depth image, current speed

        # observation space = next waypoint

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        pass

    def get_reward(self) -> float:
        pass

    def _get_info(self) -> dict:
        pass

    def _get_obs(self) -> Any:
        pass
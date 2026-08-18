import gymnasium as gym
from .controllers import RL_MineController
from .envs import MiningRLEnv

try:
    gym.register(
        id="MiningEnv-v0",
        entry_point="drs_mining.rl.environments.envs:MiningRLEnv",
    )
except Exception:
    pass

__all__ = [
    "RL_MineController",
    "MiningRLEnv",
]

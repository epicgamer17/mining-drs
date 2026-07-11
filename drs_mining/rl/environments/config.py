from dataclasses import dataclass
from drs_mining.components.config import ConcentratorConfig


@dataclass
class RLMineConfig:
    """RL-specific configuration wrapping the standard simulation config."""

    sim_config: ConcentratorConfig
    dense_reward_target_throughput: float = 5500.0
    sparse_reward_time_penalty_scale: float = 35.0
    sparse_reward_stock_penalty_weight: float = 0.05
    stockpile_scaling_factor: float = 1000.0
    time_scaling_factor: float = 1000.0

import math
import random
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from drs_mining.components.modes import RequireDecision
from drs import DRSEngine
from .models import RL_ConcentratorModel


class MiningRLEnv(gym.Env):
    """
    Gymnasium environment wrapping the DRS Mining Simulation.
    Follows standard Gymnasium API conventions.
    """

    def __init__(
        self,
        target_ore_stock_level: float = 60000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        replication_length: float = math.inf,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        dense_reward_target_throughput: float = 5500.0,
        sparse_reward_time_penalty_scale: float = 35.0,
        sparse_reward_stock_penalty_weight: float = 0.05,
        stockpile_scaling_factor: float = 1000.0,
        time_scaling_factor: float = 1000.0,
        max_steps: Optional[int] = None,
        enable_telemetry: bool = False,
        reward_type: str = "sparse",
        **kwargs,
    ):
        super().__init__()
        self.target_ore_stock_level = target_ore_stock_level
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
        )
        self.critical_ore2_level = critical_ore2_level
        self.duration_of_production_campaigns = duration_of_production_campaigns
        self.duration_of_shutdowns = duration_of_shutdowns
        self.duration_of_contingency_segments = duration_of_contingency_segments
        self.replication_length = replication_length
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.min_ore_mass = min_ore_mass
        self.max_ore_mass = max_ore_mass
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies

        self.dense_reward_target_throughput = dense_reward_target_throughput
        self.sparse_reward_time_penalty_scale = sparse_reward_time_penalty_scale
        self.sparse_reward_stock_penalty_weight = sparse_reward_stock_penalty_weight
        self.stockpile_scaling_factor = stockpile_scaling_factor
        self.time_scaling_factor = time_scaling_factor
        self.max_steps = max_steps
        self.enable_telemetry = enable_telemetry
        self.reward_type = reward_type

        # 0: Mode A, 1: Mode B
        self.action_space = spaces.Discrete(2)

        # [Ore1_Stock, Ore2_Stock, Total_Stock, Parcel_Ore_Fraction, Time]
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.sim = None
        self.engine = None
        self.last_extraction = 0.0
        self.last_time = 0.0
        self.current_step = 0

    def _get_current_time(self):
        """Helper to safely calculate total elapsed simulation days."""
        return self.sim.controller.total_duration

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.sim = RL_ConcentratorModel(
            mean_ore_fraction=self.mean_ore_fraction,
            std_dev_ore_fraction=self.std_dev_ore_fraction,
            target_ore_stock_level=self.target_ore_stock_level,
            total_ore_to_extract=self.total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=self.ore_to_be_extracted_during_warming_period,
            critical_ore2_level=self.critical_ore2_level,
            duration_of_production_campaigns=self.duration_of_production_campaigns,
            duration_of_shutdowns=self.duration_of_shutdowns,
            duration_of_contingency_segments=self.duration_of_contingency_segments,
            min_ore_mass=self.min_ore_mass,
            max_ore_mass=self.max_ore_mass,
            prob_new_facies=self.prob_new_facies,
            variation_same_facies=self.variation_same_facies,
            replication_length=self.replication_length,
            enable_telemetry=self.enable_telemetry,
        )
        self.engine = DRSEngine()
        self.engine.register(self.sim)
        self.engine.on_step(lambda t: self.sim.step_update())
        if self.enable_telemetry and hasattr(self.sim, "telemetry"):
            self.engine.attach_telemetry(self.sim.telemetry)
        self.last_extraction = 0.0
        self.last_time = 0.0
        self.current_step = 0

        try:
            self.engine.run(until=self.replication_length)
        except RequireDecision:
            pass

        self.last_time = self._get_current_time()
        self.last_extraction = self.sim.mine.cumulative_extracted_mass.value

        return self._get_obs(), {}

    def _calculate_dense_reward(self, dt: float, tons_processed: float) -> float:
        target_throughput = self.dense_reward_target_throughput
        return (
            tons_processed - (target_throughput * dt)
        ) / self.stockpile_scaling_factor

    def _calculate_sparse_reward(self, dt: float) -> float:
        reward_time_penalty = -(dt / self.sparse_reward_time_penalty_scale)
        stock_penalty_weight = self.sparse_reward_stock_penalty_weight
        total_stock = self.sim.total_stockpile_mass
        overstock = max(0.0, total_stock - self.target_ore_stock_level)
        overstock_scaled = overstock / self.stockpile_scaling_factor
        return reward_time_penalty - (stock_penalty_weight * overstock_scaled)

    def step(self, action):
        self.current_step += 1

        if (
            action == 0
            and self.sim.controller.total_system_ore_mass.value
            > self.target_ore_stock_level
        ):
            action = 2  # Mode A Mine Surging
        elif (
            action == 1
            and self.sim.controller.total_system_ore_mass.value
            > self.target_ore_stock_level
        ):
            action = 3  # Mode B Mine Surging

        self.sim.controller.pending_rl_action = action

        try:
            self.engine.run(until=self.replication_length)
        except RequireDecision:
            pass

        terminated = self.sim.is_terminating_condition_met()
        truncated = False
        if self.max_steps is not None and self.current_step >= self.max_steps:
            truncated = True

        current_time = self._get_current_time()
        current_extraction = self.sim.mine.cumulative_extracted_mass.value

        dt = current_time - self.last_time
        tons_processed = current_extraction - self.last_extraction

        self.last_time = current_time
        self.last_extraction = current_extraction

        if self.reward_type == "dense":
            reward = self._calculate_dense_reward(dt, tons_processed)
        else:
            reward = self._calculate_sparse_reward(dt)

        return self._get_obs(), float(reward), terminated, truncated, {}

    def _get_obs(self):
        target = self.target_ore_stock_level

        return np.array(
            [
                self.sim.ore1_mass / target,
                self.sim.ore2_mass / target,
                self.sim.total_stockpile_mass / target,
                self.sim.stockpile2_routing_fraction,
                self._get_current_time() / self.time_scaling_factor,
            ],
            dtype=np.float32,
        )

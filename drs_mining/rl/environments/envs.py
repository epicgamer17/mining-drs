import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from drs_mining.components.modes import RequireDecision
from drs import DRSEngine
from .config import RLMineConfig
from .models import RL_ConcentratorModel


# TODO: could i make this into a Generic DRSEnv instead? Maybe put it in the main drs/ folder?
class MiningRLEnv(gym.Env):
    """
    Gymnasium environment wrapping the DRS Mining Simulation.
    NOTE: This is a discrete action space, we could also make a continuous action space which sets threshold values and possibly the total stockpile value at the start of an episode
    NOTE: This is a terminating environment the agent is penalized for each step to encourage maximum throughput, it is also possible to make it an infinite horizon task where reward is throughput
    """

    def __init__(
        self,
        config: RLMineConfig,
        enable_telemetry: bool = False,
        reward_type: str = "sparse",
    ):
        super().__init__()
        self.rl_config = config
        self.config = config.sim_config
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
        self.last_time = 0.0  # <--- Add this

    def _get_current_time(self):
        """Helper to safely calculate total elapsed simulation days."""
        c = self.sim.controller
        return (
            c.cumulative_time_mode_a.value
            + c.cumulative_time_mode_a_contingency.value
            + c.cumulative_time_mode_a_surging.value
            + c.cumulative_time_mode_b.value
            + c.cumulative_time_mode_b_contingency.value
            + c.cumulative_time_mode_b_surging.value
            + c.cumulative_time_shutdown.value
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.sim = RL_ConcentratorModel(
            self.config, enable_telemetry=self.enable_telemetry
        )
        self.engine = DRSEngine(self.sim)
        if self.enable_telemetry and hasattr(self.sim, "telemetry"):
            self.engine.attach_telemetry(self.sim.telemetry)
        self.last_extraction = 0.0
        self.last_time = 0.0  # <--- Reset time

        try:
            # Step the engine until the first shutdown
            self.engine.run(max_time=self.config.replication_length)
        except RequireDecision:
            pass

        # Update time and extraction after the first initial run to the first decision point
        self.last_time = self._get_current_time()
        self.last_extraction = self.sim.mine.cumulative_extracted_mass.value

        return self._get_obs(), {}

    def _calculate_dense_reward(self, dt: float, tons_processed: float) -> float:
        target_throughput = self.rl_config.dense_reward_target_throughput
        return (
            tons_processed - (target_throughput * dt)
        ) / self.rl_config.stockpile_scaling_factor

    def _calculate_sparse_reward(self, dt: float) -> float:
        # --- 1. True Time Penalty Calculation ---
        # A normal campaign takes 35 days (34 prod + 1 shutdown).
        # We divide by 35 so the penalty stays around -1.0 for a standard step.
        reward_time_penalty = -(dt / self.rl_config.sparse_reward_time_penalty_scale)

        # 3. Stock Penalty
        stock_penalty_weight = self.rl_config.sparse_reward_stock_penalty_weight
        total_stock = (
            self.sim.ore1_stock.current_mass.value
            + self.sim.ore2_stock.current_mass.value
        )
        overstock = max(0.0, total_stock - self.config.target_ore_stock_level)
        overstock_scaled = overstock / self.rl_config.stockpile_scaling_factor

        # --- 3. Total Reward ---
        return reward_time_penalty - (stock_penalty_weight * overstock_scaled)

    def step(self, action):
        # 1. Queue the RL action
        # TODO: note to add action masking so if surging is needed only that mode can be selected
        if (
            action == 0
            and self.sim.controller.total_system_ore_mass.value
            > self.config.target_ore_stock_level
        ):
            action = 2  # Mode A Mine Surging
        elif (
            action == 1
            and self.sim.controller.total_system_ore_mass.value
            > self.config.target_ore_stock_level
        ):
            action = 3  # Mode B Mine Surging

        self.sim.controller.pending_rl_action = action

        # 2. Resume the simulation
        try:
            self.engine.run(max_time=self.config.replication_length)
        except RequireDecision:
            pass

        # 1. Did we hit the 6.6M target?
        terminated = self.sim.is_terminating_condition_met()

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

        # if terminated:
        # Terminal penalty: subtract the absolute distance from the 60,000 target
        # Scaled by 1000.0 to keep magnitudes comparable to the time penalties
        # TODO: scaling value for this, right now it is arbitrary.
        # TODO: should this be unified with the overstock penalty?
        # abs_distance = abs(
        #     self.sim.controller.total_system_ore_mass.value - self.config.target_ore_stock_level
        # )
        # reward -= abs_distance / 1000.0

        return self._get_obs(), float(reward), terminated, False, {}

    def _get_obs(self):
        target = self.config.target_ore_stock_level

        return np.array(
            [
                self.sim.ore1_stock.current_mass.value / target,
                self.sim.ore2_stock.current_mass.value / target,
                self.sim.controller.total_system_ore_mass.value / target,
                self.sim.fleet.stockpile2_routing_fraction.value,
                self._get_current_time() / self.rl_config.time_scaling_factor,
            ],
            dtype=np.float32,
        )

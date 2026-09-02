import random
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

import drs
from drs import DRSEngine, Telemetry, Processor
from drs_mining.components import (
    Flow,
    blend_flows,
    Stockpile,
    StochasticReserve,
    StochasticFaciesGenerator,
    RequireDecision,
)
from .controllers import RL_MineController


class MiningRLEnv(gym.Env):
    """Reinforcement Learning Environment for Tactical Blending Operations."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        target_ore_stock_level: float = 60000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        replication_length: float = 999999.0,
        max_steps: int = 100,
        reward_type: str = "dense",
        dense_reward_target_throughput: float = 5800.0,
        sparse_reward_time_penalty_scale: float = 1000.0,
        sparse_reward_stock_penalty_weight: float = 10.0,
        stockpile_scaling_factor: float = 1000.0,
        time_scaling_factor: float = 1000.0,
        enable_telemetry: bool = False,
    ):
        super().__init__()

        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.target_ore_stock_level = target_ore_stock_level
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
        )
        self.critical_ore2_level = critical_ore2_level
        self.duration_of_production_campaigns = duration_of_production_campaigns
        self.duration_of_shutdowns = duration_of_shutdowns
        self.duration_of_contingency_segments = duration_of_contingency_segments
        self.min_ore_mass = min_ore_mass
        self.max_ore_mass = max_ore_mass
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies
        self.replication_length = replication_length
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.dense_reward_target_throughput = dense_reward_target_throughput
        self.sparse_reward_time_penalty_scale = sparse_reward_time_penalty_scale
        self.sparse_reward_stock_penalty_weight = (
            sparse_reward_stock_penalty_weight
        )
        self.stockpile_scaling_factor = stockpile_scaling_factor
        self.time_scaling_factor = time_scaling_factor
        self.enable_telemetry = enable_telemetry

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(5,), dtype=np.float32
        )

        self.reserve = None
        self.mill = None
        self.mode_controller = None
        self.ore1_stock = None
        self.ore2_stock = None
        self.engine = None
        self.telemetry = None
        self.last_extraction = 0.0
        self.last_time = 0.0
        self.current_step = 0

    def _get_current_time(self) -> float:
        if self.engine is not None:
            return self.engine.current_time
        return 0.0

    def _setup_simulation(self):
        """Constructs lean simulation network with StochasticReserve, Stockpiles, and Processor."""
        gen = StochasticFaciesGenerator(
            mean_fraction=self.mean_ore_fraction,
            std_dev=self.std_dev_ore_fraction,
            prob_new_facies=self.prob_new_facies,
            variation_same_facies=self.variation_same_facies,
            attribute_name="ore2_fraction",
        )
        self.reserve = StochasticReserve(
            name="reserve",
            total_tonnes=self.total_ore_to_extract,
            generator=gen,
            min_parcel_mass=self.min_ore_mass,
            max_parcel_mass=self.max_ore_mass,
            warming_period=self.ore_to_be_extracted_during_warming_period,
        )

        init_mass1 = (1.0 - self.mean_ore_fraction) * self.target_ore_stock_level
        init_mass2 = self.mean_ore_fraction * self.target_ore_stock_level

        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            initial_mass=init_mass1,
            initial_attributes={"ore2_fraction": 0.0},
        )
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            initial_mass=init_mass2,
            initial_attributes={"ore2_fraction": 1.0},
        )

        self.mill = Processor(name="mill", max_rate=6000.0)

        self.mode_controller = RL_MineController(
            duration_of_production_campaigns=self.duration_of_production_campaigns,
            duration_of_shutdowns=self.duration_of_shutdowns,
            duration_of_contingency_segments=self.duration_of_contingency_segments,
            critical_ore2_level=self.critical_ore2_level,
            target_total_stock=self.target_ore_stock_level,
        )

        self.engine = DRSEngine()
        self.engine.register(
            self.reserve,
            self.mill,
            self.mode_controller,
            self.ore1_stock,
            self.ore2_stock,
        )

        @self.engine.on_step
        def _policy(t):
            self._step_policy(t)

    def _step_policy(self, time: float):
        """Top-level control policy invoked by the engine once per step."""
        campaign_mode = self.mode_controller.update_campaign(self.ore2_stock.level)
        active_mode = self.mode_controller.resolve_operating_mode(
            campaign_mode,
            ore1_level=self.ore1_stock.level,
            ore2_level=self.ore2_stock.level,
        )

        draw_rates = self.mode_controller.get_draw_rates(active_mode)
        ore1_target = draw_rates.get("Ore1Stock", 0.0)
        ore2_target = draw_rates.get("Ore2Stock", 0.0)

        ore2_frac = self.reserve.current_attributes.get("ore2_fraction", 0.0)
        mode_name = active_mode.name
        if "_MINE_SURGING" in mode_name:
            if mode_name == "MODE_A_MINE_SURGING":
                effective_fraction = max(1.0 - ore2_frac, 0.01)
                mine_target = ore1_target / effective_fraction
            else:
                effective_fraction = max(ore2_frac, 0.01)
                mine_target = ore2_target / effective_fraction
        else:
            mine_target = ore1_target + ore2_target

        extraction_flow = self.reserve.extract(mine_target)

        in_flow1 = Flow(
            rate=extraction_flow.rate * (1.0 - ore2_frac),
            attributes=extraction_flow.attributes,
        )
        in_flow2 = Flow(
            rate=extraction_flow.rate * ore2_frac,
            attributes=extraction_flow.attributes,
        )

        out1 = self.ore1_stock.feed_and_draw(in_flow1, ore1_target)
        out2 = self.ore2_stock.feed_and_draw(in_flow2, ore2_target)

        blended_feed = blend_flows([out1, out2])
        self.mill.rate = blended_feed.rate

    def _setup_telemetry(self):
        if not self.enable_telemetry:
            return
        self.telemetry = Telemetry(model=self.engine)
        self.telemetry.register_metric(
            "active_operating_mode",
            lambda t, m, s, _: self.mode_controller.active_operating_mode.value.name,
        )
        self.telemetry.register_metric(
            "Campaign_Shutdown",
            lambda t, m, s, _: self.mode_controller.current_campaign_duration.value,
        )
        self.telemetry.register_metric(
            "Contingency",
            lambda t, m, s, _: self.mode_controller.current_contingency_duration.value,
        )
        self.engine.attach_telemetry(self.telemetry)

    def _is_terminating_condition_met(self) -> bool:
        return self.reserve.is_exhausted

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._setup_simulation()
        self._setup_telemetry()
        self.last_extraction = 0.0
        self.last_time = 0.0
        self.current_step = 0

        try:
            self.engine.run(until=self.replication_length)
        except RequireDecision:
            pass

        self.last_time = self._get_current_time()
        self.last_extraction = self.reserve.cumulative_extracted_mass.value

        return self._get_obs(), {}

    def _calculate_dense_reward(self, dt: float, tons_processed: float) -> float:
        target_throughput = self.dense_reward_target_throughput
        return (
            tons_processed - (target_throughput * dt)
        ) / self.stockpile_scaling_factor

    def _calculate_sparse_reward(self, dt: float) -> float:
        reward_time_penalty = -(dt / self.sparse_reward_time_penalty_scale)
        stock_penalty_weight = self.sparse_reward_stock_penalty_weight
        total_stock = self.ore1_stock.level + self.ore2_stock.level
        overstock = max(0.0, total_stock - self.target_ore_stock_level)
        overstock_scaled = overstock / self.stockpile_scaling_factor
        return reward_time_penalty - (stock_penalty_weight * overstock_scaled)

    def step(self, action):
        self.current_step += 1

        total_stock = self.ore1_stock.level + self.ore2_stock.level
        if action == 0 and total_stock > self.target_ore_stock_level:
            action = 2  # Mode A Mine Surging
        elif action == 1 and total_stock > self.target_ore_stock_level:
            action = 3  # Mode B Mine Surging

        self.mode_controller.pending_rl_action = action

        try:
            self.engine.run(until=self.replication_length)
        except RequireDecision:
            pass

        terminated = self._is_terminating_condition_met()
        truncated = False
        if self.max_steps > 0 and self.current_step >= self.max_steps:
            truncated = True

        current_time = self._get_current_time()
        current_extraction = self.reserve.cumulative_extracted_mass.value

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
        ore1_mass = self.ore1_stock.level
        ore2_mass = self.ore2_stock.level
        ore2_frac = self.reserve.current_attributes.get("ore2_fraction", 0.0)

        return np.array(
            [
                ore1_mass / target,
                ore2_mass / target,
                (ore1_mass + ore2_mass) / target,
                ore2_frac,
                self._get_current_time() / self.time_scaling_factor,
            ],
            dtype=np.float32,
        )

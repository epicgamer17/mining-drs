import math
import random
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import drs
from drs.telemetry import Telemetry
from drs import DRSEngine
from drs_mining.components.modes import RequireDecision
from drs_mining.components.factories import build_mining_simulation
from .controllers import RL_MineController



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
        max_steps: int = 1000,
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

        self.mine = None
        self.fleet = None
        self.plant = None
        self.controller = None
        self.ore1_stock = None
        self.ore2_stock = None
        self.global_time = None
        self.engine = None
        self.telemetry = None
        self.last_extraction = 0.0
        self.last_time = 0.0
        self.current_step = 0

    def _get_current_time(self):
        """Helper to safely calculate total elapsed simulation days."""
        return self.controller.total_duration

    def _setup_simulation(self):
        """Builds the flat leaf components, registers them, and wires the policy."""
        faces, self.fleet, self.plant, self.controller, self.ore1_stock, self.ore2_stock = (
            build_mining_simulation(
                num_faces=1,
                controller_cls=RL_MineController,
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
            )
        )
        self.mine = faces[0]
        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)

        self.engine = DRSEngine()
        self.engine.register(
            self.mine,
            self.fleet,
            self.plant,
            self.controller,
            self.ore1_stock,
            self.ore2_stock,
        )

        @self.engine.on_step
        def _policy(time):
            self._step_policy(time)


    def _step_policy(self, time):
        """Top-level control policy invoked by the engine once per step."""
        self.global_time.rate = 1.0
        ctrl = self.controller

        mode = ctrl.update_mode(self.ore1_stock, self.ore2_stock)
        mine_target, stock1_target, stock2_target = ctrl.get_target_rates(
            mode, self.fleet
        )

        self.mine.target_rate = mine_target

        ore1_in, ore2_in = self.fleet.route(sources=[self.mine])
        out1 = self.ore1_stock.feed_and_draw(ore1_in, stock1_target)
        out2 = self.ore2_stock.feed_and_draw(ore2_in, stock2_target)

        self.plant.process(out1 + out2)

    def _setup_telemetry(self):
        if not self.enable_telemetry:
            return
        self.telemetry = Telemetry(model=self.engine)

        self.telemetry.register_metric(
            "MassOfCurrentParcel",
            lambda t, m, s, _: self.mine.active_parcel_initial_mass.value,
        )
        self.telemetry.register_metric(
            "CurrentParcelRoutingFraction",
            lambda t, m, s, _: self.fleet.stockpile2_routing_fraction.value,
        )
        self.telemetry.register_metric(
            "Campaign_Shutdown",
            lambda t, m, s, _: self.controller.current_campaign_duration.value,
        )
        self.telemetry.register_metric(
            "Contingency",
            lambda t, m, s, _: self.controller.current_contingency_duration.value,
        )

        self.engine.attach_telemetry(self.telemetry)

    def _is_terminating_condition_met(self) -> bool:
        return self.mine.cumulative_extracted_mass.value >= self.total_ore_to_extract

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
        self.last_extraction = self.mine.cumulative_extracted_mass.value

        return self._get_obs(), {}

    def _calculate_dense_reward(self, dt: float, tons_processed: float) -> float:
        target_throughput = self.dense_reward_target_throughput
        return (
            tons_processed - (target_throughput * dt)
        ) / self.stockpile_scaling_factor

    def _calculate_sparse_reward(self, dt: float) -> float:
        reward_time_penalty = -(dt / self.sparse_reward_time_penalty_scale)
        stock_penalty_weight = self.sparse_reward_stock_penalty_weight
        total_stock = self.ore1_stock.current_mass.value + self.ore2_stock.current_mass.value
        overstock = max(0.0, total_stock - self.target_ore_stock_level)
        overstock_scaled = overstock / self.stockpile_scaling_factor
        return reward_time_penalty - (stock_penalty_weight * overstock_scaled)

    def step(self, action):
        self.current_step += 1

        if (
            action == 0
            and self.controller.total_system_ore_mass.value
            > self.target_ore_stock_level
        ):
            action = 2  # Mode A Mine Surging
        elif (
            action == 1
            and self.controller.total_system_ore_mass.value
            > self.target_ore_stock_level
        ):
            action = 3  # Mode B Mine Surging

        self.controller.pending_rl_action = action

        try:
            self.engine.run(until=self.replication_length)
        except RequireDecision:
            pass

        terminated = self._is_terminating_condition_met()
        truncated = False
        if self.max_steps > 0 and self.current_step >= self.max_steps:
            truncated = True


        current_time = self._get_current_time()
        current_extraction = self.mine.cumulative_extracted_mass.value

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
        ore1_mass = self.ore1_stock.current_mass.value
        ore2_mass = self.ore2_stock.current_mass.value

        return np.array(
            [
                ore1_mass / target,
                ore2_mass / target,
                (ore1_mass + ore2_mass) / target,
                self.fleet.stockpile2_routing_fraction.value,
                self._get_current_time() / self.time_scaling_factor,
            ],
            dtype=np.float32,
        )

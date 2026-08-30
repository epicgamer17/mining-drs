"""Two-Area Multi-Stope Lifecycle Simulation: Turnaround Development, Waste Rock & Hierarchical Dispatch.

Implements the multi-stope operational underground environment:
  - Area 1 (Level 3): 3 active stopes (1A, 1B, 1C) with finite reserves (1.8M tonnes total).
  - Area 2 (Level 6): 3 deep stopes (2A, 2B, 2C) unlocked via 4,000 m capital decline development.
  - Stope Lifecycle:
      1. ORE_READY: Blasted ore mucked out by LHDs into 26.1t AD30 haul trucks.
      2. DEVELOPMENT_TURNAROUND: Ore round depleted. Requires waste rock extraction / development advance (30m)
         before the next ore round can be accessed.
      3. EXHAUSTED: Total stope reserve depleted; stope is permanently closed.
  - Two-Tier Hierarchical Closed-Loop Dispatch:
      - Tier 1 (Absolute Primary Requirement): Maintain 6,000 t/d total plant feed to prevent mill starvation.
      - Tier 2 (Secondary Blending Objective): Match analytical dispatch weights (w1, w2) for high-grade Mode A.
      - Tier 3 (Dynamic Constrained Fallback): If preferred stope is in turnaround or LHD-busy, immediately
        redirect haul truck to next available stope.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import drs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drs.plot import (
    Dashboard,
    plot_time_series,
    plot_dual_axis_step,
    plot_safety_margin,
)
from drs_mining.components.allocation import solve_face_allocation_rates
from drs_mining.components.modes import MODES, OperatingMode
from drs_mining.components.plant import MetallurgicalPlant, PlantDrawRates
from drs_mining.components.stockpiles import Stockpile
from drs_mining.components.controllers import OperatingModeController
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.stope import StopeFace, StopeState
from drs_mining.components.dispatch import TwoTierHierarchicalDispatchController
from drs_mining.components.planning import (
    AreaReadinessTarget,
    MiningPriority,
    StrategicYearTarget,
    strategic_target_for_year,
    trajectory_progress_ratio,
    select_mining_priority,
)
from drs_mining.components.plot import (
    MODE_PALETTE,
    prepare_history,
    plot_ore_with_modes,
    plot_mode_distribution,
    plot_truck_idle_and_utilization,
)


# ---------------------------------------------------------------------------
# Constants & Physical Parameters
# ---------------------------------------------------------------------------
DAYS_IN_YEAR = 365.0
NON_PRODUCTION_DAYS = 0

SHIFT_SECONDS = 12.0 * 3600.0
SHIFT_WORK_HOURS = 10.5
HAULAGE_SEAT_FRACTION = 0.5417
SEAT_PER_SHIFT_SEC = HAULAGE_SEAT_FRACTION * SHIFT_SECONDS

DECLINE_M = 2100.0
LEVEL_SPACING_M = 300.0
AREA1_LEVEL = 3  # Level 3: 900 m ramp climb (~28 min cycle)
AREA2_LEVEL = 6  # Level 6: 1,800 m ramp climb (~45 min cycle)
LEVEL_DRIFT_M = 60.0
SURFACE_M = 300.0

SPEEDS = {
    "surface": {"empty": 17.4, "loaded": 13.4},
    "decline": {"empty": 15.1, "loaded": 11.2},
    "ramp": {"empty": 12.9, "loaded": 9.2},
    "level": {"empty": 7.6, "loaded": 6.6},
}

ORE_PAYLOAD = 26.1
TRUCK_LOAD_SPOT_MIN = 0.50
LHD_ACQUISITION_MAX_MIN = 0.80
TRUCK_LOAD_DUR_MIN = 3.50
DUMP_SPOT_MIN = 0.57
DUMP_MIN = 0.88
SURFACE_TIP_SITES = 2

FUEL_BURN_PCT_PER_SEC = 100.0 / (7.5 * 3600.0)
REFUEL_DUR_MIN = 25.0
N_FUEL_PUMPS = 2
BASE_PASS_BAY_DELAY_SEC = 13.0
PER_TRUCK_PASS_BAY_DELAY_SEC = 1.0

DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY = 5.0
DT_MAX = 900.0


# ---------------------------------------------------------------------------
# Helpers & Enums
# ---------------------------------------------------------------------------
def _tri(rng: random.Random, mid: float, tol: float) -> float:
    return rng.triangular(mid * (1.0 - tol), mid * (1.0 + tol), mid)


def _in_shift_window(t: float) -> bool:
    hod = t % 86400.0
    return (0.0 <= hod < SHIFT_WORK_HOURS * 3600.0) or (
        12.0 * 3600.0 <= hod < 22.5 * 3600.0
    )


class TruckPhase(Enum):
    IDLE = "idle"
    EMPTY = "empty"
    WAIT_LOAD = "wait_load"
    SPOT_LOAD = "spot_load"
    ACQUIRE = "acquire"
    LOADING = "loading"
    LOADED = "loaded"
    WAIT_DUMP = "wait_dump"
    SPOT_DUMP = "spot_dump"
    DUMPING = "dumping"
    REFUELING = "refueling"


OPERATING_PHASES = {
    TruckPhase.EMPTY,
    TruckPhase.WAIT_LOAD,
    TruckPhase.SPOT_LOAD,
    TruckPhase.ACQUIRE,
    TruckPhase.LOADING,
    TruckPhase.LOADED,
    TruckPhase.WAIT_DUMP,
    TruckPhase.SPOT_DUMP,
    TruckPhase.DUMPING,
}

SEAT_PHASES = {
    TruckPhase.EMPTY,
    TruckPhase.SPOT_LOAD,
    TruckPhase.LOADING,
    TruckPhase.LOADED,
    TruckPhase.SPOT_DUMP,
    TruckPhase.DUMPING,
}

DUE_PHASES = {
    TruckPhase.EMPTY,
    TruckPhase.SPOT_LOAD,
    TruckPhase.ACQUIRE,
    TruckPhase.LOADING,
    TruckPhase.LOADED,
    TruckPhase.SPOT_DUMP,
    TruckPhase.DUMPING,
    TruckPhase.REFUELING,
}


@dataclass
class Operator:
    idx: int
    used_seat: float = 0.0
    free: bool = True


@dataclass
class Truck:
    truck_id: str
    timer: drs.Timer
    phase: TruckPhase = TruckPhase.IDLE
    target_stope_id: int = 1
    target_level: int = AREA1_LEVEL
    current_payload: float = 0.0
    payload_ore_fraction: float = 0.30
    seat_used: float = 0.0
    fuel: float = 100.0
    refuel_threshold: float = 30.0
    operator: int = -1
    trip_start: float = 0.0
    dump_dur: float = 0.0
    down_start: float = math.inf
    down_end: float = math.inf


@dataclass
class SurfaceDumpStation:
    name: str = "SURFACE_CRUSHER_HOPPER"
    capacity: int = SURFACE_TIP_SITES
    in_use: int = 0
    queue: list = field(default_factory=list)
    _active_ore1_rate: float = 0.0
    _active_ore2_rate: float = 0.0


# ---------------------------------------------------------------------------
# Multi-Stope Lifecycle Simulation Engine
# ---------------------------------------------------------------------------
class TwoAreaStopeLifecycleEngine(drs.Module):
    """Underground DES simulation module with multi-stope lifecycles, turnaround dev & two-tier dispatch."""

    def __init__(
        self,
        policy_name: str = "POLICY_2_VALUE_ORIENTED",  # "POLICY_1_MYOPIC" or "POLICY_2_VALUE_ORIENTED"
        use_two_tier_dispatch: bool = True,
        # Sizing haulage fleet to 18 AD30 trucks for deep Level 6 cycle times (~45 min)
        num_trucks: int = 18,
        num_operators: int = 18,
        num_lhds_per_stope: int = 1,
        availability: float = 0.85,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        strategic_period_days: float = 365.0,
        tactical_review_period_days: float = 30.0,
        tactical_progress_tolerance: float = 0.90,
        strategic_targets: Optional[Tuple[StrategicYearTarget, ...]] = None,
        area2_readiness_target: Optional[AreaReadinessTarget] = None,
        annual_discount_rate: float = 0.05,
        ore1_net_value_per_processed_tonne: float = 577.48,
        ore2_net_value_per_processed_tonne: float = 709.83,
        production_cost_per_tonne: float = 135.0,
        development_cost_per_unit: float = 15000.0,
        fixed_cost_per_day: float = 74460.0,
        mode_a_ore1_milling_rate: float = 3600.0,
        mode_a_ore2_milling_rate: float = 2400.0,
        mode_a_contingency_ore1_milling_rate: float = 3900.0,
        mode_b_ore1_milling_rate: float = 4600.0,
        mode_b_ore2_milling_rate: float = 800.0,
        mode_b_contingency_ore2_milling_rate: float = 2500.0,
        seed: int = 42,
    ):
        super().__init__()
        self.policy_name = policy_name
        self.use_two_tier_dispatch = use_two_tier_dispatch
        self.num_trucks = num_trucks
        self.num_operators = num_operators
        self.num_lhds_per_stope = num_lhds_per_stope
        self.availability = availability
        self.target_ore_stock_level = target_ore_stock_level
        self.critical_ore2_level = critical_ore2_level
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = ore_to_be_extracted_during_warming_period

        self.strategic_period_days = strategic_period_days
        self.tactical_review_period_days = tactical_review_period_days
        self.tactical_progress_tolerance = tactical_progress_tolerance
        self.strategic_targets = strategic_targets or (
            StrategicYearTarget(
                min_development=10000.0,
                min_ore1_production=1300000.0,
                min_ore2_production=850000.0,
            ),
        )
        self.area2_readiness_target = area2_readiness_target or AreaReadinessTarget(
            required_development=4000.0,
            ready_by_day=365.0,
        )

        self.annual_discount_rate = annual_discount_rate
        self.ore1_net_value_per_processed_tonne = ore1_net_value_per_processed_tonne
        self.ore2_net_value_per_processed_tonne = ore2_net_value_per_processed_tonne
        self.production_cost_per_tonne = production_cost_per_tonne
        self.development_cost_per_unit = development_cost_per_unit
        self.fixed_cost_per_day = fixed_cost_per_day

        self.rng = random.Random(seed)
        self.seed = seed

        self.truck_seat_credit = availability * SEAT_PER_SHIFT_SEC
        self._down_dur = max(0.0, (1.0 - availability) * SEAT_PER_SHIFT_SEC)

        # Calendar setup
        self._cur_day = -1
        self._shift_marker = -1
        self._holiday_today = False

        # Global time tracker
        self.gt = drs.Timer("gt", 0.0, rate=1.0)

        # 1. Multi-Stope Topology: 3 Stopes in Area 1 (Level 3), 3 Stopes in Area 2 (Level 6)
        self.stopes: List[StopeFace] = []

        # Area 1 Stopes (Level 3, Mean 70/30 ratio: ~30% Ore 2, Finite 600k t reserve each -> 1.8M t total)
        # 3 stopes centered around 0.30 (e.g. 0.28, 0.30, 0.32)
        a1_means = [0.28, 0.30, 0.32]
        for i in range(1, 4):
            mean_f = a1_means[i - 1]
            gen = StochasticFaciesGenerator(
                mean_fraction=mean_f,
                std_dev=0.03,
                prob_new_facies=0.3,
                variation_same_facies=0.01,
            )
            stope = StopeFace(
                name=f"stope_1{chr(64+i)}",
                face_id=i,
                area_id=1,
                level_index=AREA1_LEVEL,
                generator=gen,
                mean_ore_fraction=mean_f,
                std_dev_ore_fraction=0.03,
                total_stope_reserve=600000.0,
                min_parcel_ore_mass=25000.0,
                max_parcel_ore_mass=40000.0,
                waste_to_ore_ratio=0.15,
                turnaround_dev_per_parcel_m=5.0,
                seed=seed + i,
            )
            self.stopes.append(stope)

        # Area 2 Stopes (Level 6, Mean 65/35 ratio: ~35% Ore 2, 1.6M t reserve each -> 4.8M t total)
        # 3 stopes centered around 0.35 (e.g. 0.33, 0.35, 0.37)
        a2_means = [0.33, 0.35, 0.37]
        for i in range(4, 7):
            mean_f = a2_means[i - 4]
            gen = StochasticFaciesGenerator(
                mean_fraction=mean_f,
                std_dev=0.03,
                prob_new_facies=0.3,
                variation_same_facies=0.01,
            )
            stope = StopeFace(
                name=f"stope_2{chr(64+i-3)}",
                face_id=i,
                area_id=2,
                level_index=AREA2_LEVEL,
                generator=gen,
                mean_ore_fraction=mean_f,
                std_dev_ore_fraction=0.03,
                total_stope_reserve=1600000.0,
                min_parcel_ore_mass=25000.0,
                max_parcel_ore_mass=40000.0,
                waste_to_ore_ratio=0.20,
                turnaround_dev_per_parcel_m=5.0,
                seed=seed + i,
            )
            self.stopes.append(stope)

        # Two-Tier Hierarchical Dispatcher
        self.dispatcher = TwoTierHierarchicalDispatchController(
            stopes=self.stopes,
            target_daily_ore_tonnes=6000.0,
            target_stockpile_buffer_tonnes=target_ore_stock_level,
            seed=seed,
        )

        # 2. Continuous Surface Stockpiles
        init_ore1 = 0.70 * target_ore_stock_level
        init_ore2 = 0.30 * target_ore_stock_level
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=init_ore1,
            initial_attributes={"contained_ore_fraction_mass": init_ore1 * 0.30},
            attr_inflow=1.0,
        )
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=init_ore2,
            initial_attributes={"contained_ore_fraction_mass": init_ore2 * 0.30},
            attr_inflow=0.0,
        )
        self.total_extracted_ore = drs.Level("total_extracted_ore", 0.0)
        self.ore1_hauled = drs.Level("ore1_hauled", 0.0)
        self.ore2_hauled = drs.Level("ore2_hauled", 0.0)
        self.cumulative_mine_development = drs.Level("cumulative_mine_development", 0.0)
        self.area2_cumulative_development = drs.Level("area2_cumulative_development", 0.0)
        self.stope_turnaround_development = drs.Level("stope_turnaround_development", 0.0)

        # 3. Plant & Campaign Mode Controller
        self.mode_controller = OperatingModeController(
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            critical_ore2_level=critical_ore2_level,
            target_ore_stock_level=target_ore_stock_level,
            total_ore_to_extract=total_ore_to_extract,
        )
        self.plant = MetallurgicalPlant(
            stockpiles=[self.ore1_stock, self.ore2_stock],
            target_ore_stock_level=target_ore_stock_level,
            duration_of_contingency_segments=duration_of_contingency_segments,
            mode_a_ore1_milling_rate=mode_a_ore1_milling_rate,
            mode_a_ore2_milling_rate=mode_a_ore2_milling_rate,
            mode_a_contingency_ore1_milling_rate=mode_a_contingency_ore1_milling_rate,
            mode_b_ore1_milling_rate=mode_b_ore1_milling_rate,
            mode_b_ore2_milling_rate=mode_b_ore2_milling_rate,
            mode_b_contingency_ore2_milling_rate=mode_b_contingency_ore2_milling_rate,
        )

        # 4. Fleet & Operators
        self.trucks: List[Truck] = []
        for i in range(1, num_trucks + 1):
            timer = drs.Timer(f"tr_{i}_act", 0.0, rate=-1.0)
            timer.lower_threshold = 0.0
            tr = Truck(truck_id=f"T{i:02d}", timer=timer)
            tr.refuel_threshold = self.rng.uniform(15.0, 40.0)
            self.trucks.append(tr)

        self.operators = [Operator(i) for i in range(num_operators)]

        # 5. Stope Queues & Surface Dump Station
        self.stope_queues: Dict[int, List[Truck]] = {s.face_id: [] for s in self.stopes}
        self._stope_lhds_busy: Dict[int, int] = {s.face_id: 0 for s in self.stopes}
        self.dump_station = SurfaceDumpStation()
        self._pumps_free = N_FUEL_PUMPS

        # 6. Strategic & Tactical State Variables
        self.strategic_planning_started = False
        self.strategic_year_index = drs.Level("strategic_year_index", 0.0)
        self.strategic_year_timer = drs.Timer("strategic_year_timer", 0.0, rate=1.0)
        self.tactical_review_timer = drs.Timer("tactical_review_timer", 0.0, rate=1.0)
        self.tactical_review_count = drs.Level("tactical_review_count", 0.0)
        self.mining_priority = (
            MiningPriority.PRODUCTION
            if policy_name == "POLICY_1_MYOPIC"
            else MiningPriority.BALANCED
        )

        # 7. Area 2 Readiness Variables
        self.area2_ready = False
        self.area2_ready_day = drs.Level("area2_ready_day", -1.0)
        self.area2_readiness_fraction = drs.Level("area2_readiness_fraction", 0.0)
        self.area2_readiness_trajectory_ratio = drs.Level("area2_readiness_trajectory_ratio", 1.0)

        # 8. Economics Variables
        self.cumulative_processed_ore1 = drs.Level("cumulative_processed_ore1", 0.0)
        self.cumulative_processed_ore2 = drs.Level("cumulative_processed_ore2", 0.0)
        self.cumulative_cash_flow = drs.Level("cumulative_cash_flow", 0.0)
        self.cumulative_discounted_cash_flow = drs.Level("cumulative_discounted_cash_flow", 0.0)
        self.current_cash_flow_rate = drs.Level("current_cash_flow_rate", 0.0)
        self.current_discounted_cash_flow_rate = drs.Level("current_discounted_cash_flow_rate", 0.0)
        self.discount_factor = drs.Level("discount_factor", 1.0)
        self.operating_npv_proxy = drs.Level("operating_npv_proxy", 0.0)

        # Annual Progress Trackers
        self.annual_ore1_extracted = 0.0
        self.annual_ore2_extracted = 0.0
        self.annual_development_start = 0.0
        self.development_priority_reserved_trucks = drs.Level("development_priority_reserved_trucks", 0.0)
        self.development_rate_m_per_day = drs.Level("development_rate_m_per_day", 0.0)

        # Trajectory Ratios
        self.ore1_trajectory_ratio = drs.Level("ore1_trajectory_ratio", 1.0)
        self.ore2_trajectory_ratio = drs.Level("ore2_trajectory_ratio", 1.0)
        self.development_trajectory_ratio = drs.Level("development_trajectory_ratio", 1.0)

        # Analytical Blending & Fallback Metrics
        self.analytical_face1_weight = drs.Level("analytical_face1_weight", 1.0)
        self.analytical_face2_weight = drs.Level("analytical_face2_weight", 0.0)
        self.fallback_dispatch_count = drs.Level("fallback_dispatch_count", 0.0)

        # Operational metrics
        self.daily_target_ore = 6000.0
        self.daily_hauled_ore = 0.0
        self.trips = 0
        self._cycle_sum = 0.0
        self.traffic_delay_sum = 0.0
        self.horizon_sec = float("inf")

        self.history_records: List[dict] = []
        self._last_telemetry_time = -1.0

    # -- Area 2 Lock / Unlock Logic ------------------------------------------
    def is_area2_locked(self) -> bool:
        required = max(0.0, float(self.area2_readiness_target.required_development))
        if required <= 1e-12:
            return False
        return not (self.strategic_planning_started and self.area2_ready)

    def _update_area2_readiness(self):
        target = self.area2_readiness_target
        required = max(0.0, float(target.required_development))
        if required <= 1e-12:
            self.area2_ready = True
            self.area2_readiness_fraction.value = 1.0
            self.area2_readiness_trajectory_ratio.value = 1.0
            return

        if not self.strategic_planning_started:
            self.area2_readiness_fraction.value = 0.0
            self.area2_readiness_trajectory_ratio.value = 1.0
            return

        strategic_days = (
            float(self.strategic_year_index.value) * self.strategic_period_days
            + float(self.strategic_year_timer.value)
        )
        progress = float(self.area2_cumulative_development.value)
        fraction = min(1.0, progress / required)
        self.area2_readiness_fraction.value = fraction

        # Check for Physical Unlock
        if (not self.area2_ready) and progress >= required - 1e-6:
            self.area2_ready = True
            self.area2_ready_day.value = strategic_days
            print(
                f"\n >>> [{self.policy_name} UNLOCK] Area 2 (Deep Stopes 2A, 2B, 2C) UNLOCKED on Strategic Day {strategic_days:.2f}! <<<\n"
            )

        ready_by_day = target.ready_by_day
        if ready_by_day is not None and ready_by_day > 0.0:
            elapsed_fraction = max(1e-4, min(1.0, strategic_days / ready_by_day))
            self.area2_readiness_trajectory_ratio.value = trajectory_progress_ratio(
                actual=progress,
                annual_target=required,
                elapsed_fraction=elapsed_fraction,
            )

    # -- Strategic Economics -------------------------------------------------
    def _update_strategic_economics(self, out1_t_sec: float, out2_t_sec: float, dt_days: float):
        if not self.strategic_planning_started:
            self.discount_factor.value = 1.0
            self.current_cash_flow_rate.value = 0.0
            self.current_discounted_cash_flow_rate.value = 0.0
            return

        milled_ore1_t_day = out1_t_sec * 86400.0
        milled_ore2_t_day = out2_t_sec * 86400.0
        self.cumulative_processed_ore1.value += milled_ore1_t_day * dt_days
        self.cumulative_processed_ore2.value += milled_ore2_t_day * dt_days

        revenue_rate = (
            milled_ore1_t_day * self.ore1_net_value_per_processed_tonne
            + milled_ore2_t_day * self.ore2_net_value_per_processed_tonne
        )
        mined_ore_t_day = max(0.0, self.daily_hauled_ore)
        production_cost_rate = mined_ore_t_day * self.production_cost_per_tonne
        dev_rate_m_day = max(0.0, float(self.development_rate_m_per_day.value))
        development_cost_rate = dev_rate_m_day * self.development_cost_per_unit
        fixed_cost_rate = self.fixed_cost_per_day

        cash_flow_rate = revenue_rate - (production_cost_rate + development_cost_rate + fixed_cost_rate)
        self.current_cash_flow_rate.value = cash_flow_rate
        self.cumulative_cash_flow.value += cash_flow_rate * dt_days

        strategic_days = (
            float(self.strategic_year_index.value) * self.strategic_period_days
            + float(self.strategic_year_timer.value)
        )
        dfactor = 1.0 / ((1.0 + self.annual_discount_rate) ** (strategic_days / 365.0))
        self.discount_factor.value = dfactor

        discounted_rate = cash_flow_rate * dfactor
        self.current_discounted_cash_flow_rate.value = discounted_rate
        self.cumulative_discounted_cash_flow.value += discounted_rate * dt_days
        self.operating_npv_proxy.value = self.cumulative_discounted_cash_flow.value

    # -- Analytical Blending Computation -------------------------------------
    def _compute_analytical_face_allocation(self, plant_draw: PlantDrawRates):
        if self.is_area2_locked():
            self.analytical_face1_weight.value = 1.0
            self.analytical_face2_weight.value = 0.0
            return

        # Estimate average grades across active stopes in Area 1 vs Area 2
        a1_stopes = [s for s in self.stopes if s.area_id == 1 and not s.is_exhausted]
        a2_stopes = [s for s in self.stopes if s.area_id == 2 and not s.is_exhausted]

        avg_f1 = (
            np.mean([float(s.active_parcel_ore_fraction.value) for s in a1_stopes])
            if a1_stopes
            else 0.30
        )
        avg_f2 = (
            np.mean([float(s.active_parcel_ore_fraction.value) for s in a2_stopes])
            if a2_stopes
            else 0.35
        )

        f1_ore1 = 1.0 - avg_f1
        f2_ore1 = 1.0 - avg_f2

        alloc = solve_face_allocation_rates(
            target_ore1_rate=plant_draw.ore1,
            target_ore2_rate=plant_draw.ore2,
            face1_ore1_fraction=f1_ore1,
            face2_ore1_fraction=f2_ore1,
        )

        self.analytical_face1_weight.value = alloc.face1_weight
        self.analytical_face2_weight.value = alloc.face2_weight

    # -- DRS Engine Hooks ----------------------------------------------------
    def is_terminating_condition_met(self) -> bool:
        total_extracted = sum(s.cumulative_ore_extracted.value for s in self.stopes)
        if total_extracted >= self.total_ore_to_extract - 1e-6:
            return True
        accessible_stopes = [
            s for s in self.stopes
            if not s.is_exhausted and (s.area_id == 1 or not self.is_area2_locked())
        ]
        if not accessible_stopes:
            return True
        return self.gt.value >= self.horizon_sec - 1e-6

    def time_to_event(self) -> float:
        best = DT_MAX
        for tr in self.trucks:
            v = tr.timer.value
            if v > 1e-9:
                best = min(best, v)
        t = self.gt.value
        next_day = (math.floor(t / 86400.0) + 1.0) * 86400.0
        next_shift = (math.floor(t / SHIFT_SECONDS) + 1.0) * SHIFT_SECONDS
        best = min(best, next_day - t, next_shift - t)

        camp_thresh = (
            self.mode_controller.duration_of_shutdowns
            if self.mode_controller.active_campaign_mode.value.name == "SHUTDOWN"
            else self.mode_controller.duration_of_production_campaigns
        )
        rem_camp_days = max(
            0.0, camp_thresh - self.mode_controller.current_campaign_duration.value
        )
        if rem_camp_days > 1e-6:
            best = min(best, rem_camp_days * 86400.0)

        rem_tactical_days = max(
            0.0, self.tactical_review_period_days - self.tactical_review_timer.value
        )
        if rem_tactical_days > 1e-6:
            best = min(best, rem_tactical_days * 86400.0)

        return max(best, 1e-6)

    def step(self, dt: float):
        self.gt.step(dt)
        dt_days = dt / 86400.0

        self.mode_controller.current_campaign_duration.step(dt_days)
        active_mode_name = self.plant.active_operating_mode.value.name
        timer_attr = self.plant._MODE_TIMER_ATTRS.get(active_mode_name)
        if timer_attr:
            getattr(self.plant, timer_attr).step(dt_days)
        if active_mode_name in self.plant._CONTINGENCY_MODES:
            self.plant.current_contingency_duration.step(dt_days)

        if self.strategic_planning_started:
            self.strategic_year_timer.step(dt_days)
            self.tactical_review_timer.step(dt_days)

        for tr in self.trucks:
            if tr.timer.value > 0.0:
                tr.timer.step(dt)
            if tr.phase in SEAT_PHASES:
                tr.seat_used = min(self.truck_seat_credit, tr.seat_used + dt)
                if tr.phase in OPERATING_PHASES:
                    tr.fuel = max(0.0, tr.fuel - dt * FUEL_BURN_PCT_PER_SEC)
                if tr.operator >= 0:
                    op = self.operators[tr.operator]
                    op.used_seat = min(SEAT_PER_SHIFT_SEC, op.used_seat + dt)

        ore1_in_rate = self.dump_station._active_ore1_rate
        ore2_in_rate = self.dump_station._active_ore2_rate

        # Estimate current stock routing fraction
        a1_stopes = [s for s in self.stopes if s.area_id == 1 and not s.is_exhausted]
        a2_stopes = [s for s in self.stopes if s.area_id == 2 and not s.is_exhausted]
        if self.is_area2_locked() or not a2_stopes:
            f_blend = float(a1_stopes[0].active_parcel_ore_fraction.value) if a1_stopes else 0.15
        else:
            f1 = float(a1_stopes[0].active_parcel_ore_fraction.value) if a1_stopes else 0.15
            f2 = float(a2_stopes[0].active_parcel_ore_fraction.value) if a2_stopes else 0.45
            f_blend = (f1 + f2) / 2.0

        plant_draw, _ = self.plant.get_target_rates(
            self.mode_controller.active_campaign_mode.value,
            ore1_level=self.ore1_stock.level,
            ore2_level=self.ore2_stock.level,
            stockpile2_routing_fraction=f_blend,
        )

        self._compute_analytical_face_allocation(plant_draw)

        ore1_draw_rate_sec = plant_draw.ore1 / 86400.0
        ore2_draw_rate_sec = plant_draw.ore2 / 86400.0

        out1 = self.ore1_stock.feed_and_draw(ore1_in_rate, ore1_draw_rate_sec)
        out2 = self.ore2_stock.feed_and_draw(ore2_in_rate, ore2_draw_rate_sec)
        self.ore1_stock.step(dt)
        self.ore2_stock.step(dt)

        self.plant.process(out1 + out2)
        self.plant.cumulative_milled_mass.step(dt)

        # Policy-Driven Development Allocation & Stope Turnaround
        n_operating_trucks = sum(1 for tr in self.trucks if tr.phase in OPERATING_PHASES)
        total_trucks = len(self.trucks)
        available_extra = max(0, total_trucks - n_operating_trucks)

        if self.policy_name == "POLICY_1_MYOPIC":
            dev_trucks = max(2.0, float(available_extra))
            frac_to_area2 = 0.05
        else:
            reserved_trucks = float(self.development_priority_reserved_trucks.value)
            locked_boost = (total_trucks * 0.35) if self.is_area2_locked() else 0.0
            dev_trucks = max(reserved_trucks, float(available_extra)) + locked_boost
            prio = self.mining_priority
            frac_to_area2 = (
                0.85
                if prio == MiningPriority.DEVELOPMENT
                else (0.60 if prio == MiningPriority.BALANCED else 0.35)
            )

        self.development_rate_m_per_day.value = dev_trucks * DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY
        delta_dev = self.development_rate_m_per_day.value * dt_days

        # Two Physical Development Types Allocation:
        # 1. Capital Development: Area 2 Capital Decline (Ramp to Level 6)
        # 2. Stope Turnaround Development: In-stope waste & access advance (Unlocks next parcel)
        turnaround_stopes = [s for s in self.stopes if s.is_in_turnaround]

        if self.is_area2_locked() and self.strategic_planning_started:
            if self.policy_name == "POLICY_1_MYOPIC":
                capital_advance = delta_dev * frac_to_area2
                turnaround_advance = delta_dev * (1.0 - frac_to_area2)
            else:
                capital_advance = delta_dev * (frac_to_area2 if turnaround_stopes else 1.0)
                turnaround_advance = delta_dev * (1.0 - frac_to_area2) if turnaround_stopes else 0.0

            self.area2_cumulative_development.value += capital_advance
            if turnaround_stopes and turnaround_advance > 0.0:
                m_per_stope = turnaround_advance / len(turnaround_stopes)
                for stope in turnaround_stopes:
                    adv, _ = stope.advance_turnaround_development(m_per_stope)
                    self.stope_turnaround_development.value += adv
        else:
            # Area 2 unlocked: all active development capacity goes to locked stopes
            if turnaround_stopes:
                m_per_stope = delta_dev / len(turnaround_stopes)
                for stope in turnaround_stopes:
                    adv, _ = stope.advance_turnaround_development(m_per_stope)
                    self.stope_turnaround_development.value += adv

        self.cumulative_mine_development.value = (
            self.area2_cumulative_development.value + self.stope_turnaround_development.value
        )

        self._update_area2_readiness()
        self._update_strategic_economics(out1, out2, dt_days)

        self.total_extracted_ore.step(dt)
        self.ore1_hauled.step(dt)
        self.ore2_hauled.step(dt)
        self.fallback_dispatch_count.value = float(self.dispatcher.fallback_dispatches)
        self._record_telemetry(plant_draw)

    # -- Event Policy & Target Setting ----------------------------------------
    def on_event(self, t: float):
        self._calendar_update()
        self._update_strategic_tactical_review()
        self._update_operating_mode_and_targets()

        guard = 0
        changed = True
        while changed and guard < 200:
            changed = False
            guard += 1
            for tr in self.trucks:
                if tr.phase == TruckPhase.IDLE:
                    if self._try_dispatch(tr):
                        changed = True
                elif tr.phase in DUE_PHASES and tr.timer.value <= 1e-6:
                    if self._advance(tr):
                        changed = True

    def _update_strategic_tactical_review(self):
        total_extracted = sum(s.cumulative_ore_extracted.value for s in self.stopes)
        if (
            total_extracted >= self.ore_to_be_extracted_during_warming_period
            or self.horizon_sec < float("inf")
        ):
            if not self.strategic_planning_started:
                self.strategic_planning_started = True
                self.strategic_year_index.value = 0.0
                self.strategic_year_timer.reset()
                self.tactical_review_timer.reset()
                self.tactical_review_count.value = 0.0
                self.annual_ore1_extracted = 0.0
                self.annual_ore2_extracted = 0.0
                self.annual_development_start = float(self.cumulative_mine_development.value)

        if not self.strategic_planning_started:
            return

        if self.strategic_year_timer.value >= self.strategic_period_days - 1e-6:
            self.strategic_year_index.value += 1.0
            self.strategic_year_timer.reset()
            self.annual_ore1_extracted = 0.0
            self.annual_ore2_extracted = 0.0
            self.annual_development_start = float(self.cumulative_mine_development.value)

        elapsed_year_fraction = max(
            1e-4,
            min(1.0, self.strategic_year_timer.value / self.strategic_period_days),
        )
        current_target = strategic_target_for_year(
            self.strategic_targets, int(self.strategic_year_index.value)
        )

        annual_dev = float(self.cumulative_mine_development.value) - self.annual_development_start
        self.development_trajectory_ratio.value = trajectory_progress_ratio(
            actual=annual_dev,
            annual_target=current_target.min_development,
            elapsed_fraction=elapsed_year_fraction,
        )
        self.ore1_trajectory_ratio.value = trajectory_progress_ratio(
            actual=self.annual_ore1_extracted,
            annual_target=current_target.min_ore1_production,
            elapsed_fraction=elapsed_year_fraction,
        )
        self.ore2_trajectory_ratio.value = trajectory_progress_ratio(
            actual=self.annual_ore2_extracted,
            annual_target=current_target.min_ore2_production,
            elapsed_fraction=elapsed_year_fraction,
        )

        # Monthly Tactical Review
        if (
            self.tactical_review_timer.value >= self.tactical_review_period_days - 1e-6
            or self.tactical_review_count.value == 0.0
        ):
            self.tactical_review_timer.reset()
            self.tactical_review_count.value += 1.0

            if self.policy_name == "POLICY_1_MYOPIC":
                self.mining_priority = MiningPriority.PRODUCTION
                self.development_priority_reserved_trucks.value = 0.0
            else:
                effective_dev_ratio = min(
                    float(self.development_trajectory_ratio.value),
                    float(self.area2_readiness_trajectory_ratio.value),
                )
                selected = select_mining_priority(
                    development_ratio=effective_dev_ratio,
                    ore1_ratio=float(self.ore1_trajectory_ratio.value),
                    ore2_ratio=float(self.ore2_trajectory_ratio.value),
                    tolerance=self.tactical_progress_tolerance,
                )
                self.mining_priority = selected
                if selected == MiningPriority.DEVELOPMENT:
                    n_res = max(1, int(round(len(self.trucks) * 0.20)))
                    self.development_priority_reserved_trucks.value = float(n_res)
                else:
                    self.development_priority_reserved_trucks.value = 0.0

    def _update_operating_mode_and_targets(self):
        camp_mode = self.mode_controller.update(
            ore2_stock_level=self.ore2_stock.level,
            total_stock_level=self.ore1_stock.level + self.ore2_stock.level,
        )
        a1_stopes = [s for s in self.stopes if s.area_id == 1 and not s.is_exhausted]
        a2_stopes = [s for s in self.stopes if s.area_id == 2 and not s.is_exhausted]
        if self.is_area2_locked() or not a2_stopes:
            f_blend = float(a1_stopes[0].active_parcel_ore_fraction.value) if a1_stopes else 0.30
        else:
            f1 = float(a1_stopes[0].active_parcel_ore_fraction.value) if a1_stopes else 0.30
            f2 = float(a2_stopes[0].active_parcel_ore_fraction.value) if a2_stopes else 0.35
            f_blend = (f1 + f2) / 2.0

        plant_draw, _ = self.plant.get_target_rates(
            camp_mode,
            ore1_level=self.ore1_stock.level,
            ore2_level=self.ore2_stock.level,
            stockpile2_routing_fraction=f_blend,
        )

        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self.daily_target_ore = 0.0
        elif "_MINE_SURGING" in mode_name:
            self.daily_target_ore = plant_draw.total * 0.70
        else:
            self.daily_target_ore = plant_draw.total

    def _calendar_update(self):
        t = self.gt.value
        day = int(t // 86400.0)
        if self._cur_day != day:
            self._cur_day = day
            self._holiday_today = False
            self.daily_hauled_ore = 0.0
            self._update_operating_mode_and_targets()

        marker = int(t // SHIFT_SECONDS)
        if self._shift_marker != marker:
            self._shift_marker = marker
            bound = {tr.operator for tr in self.trucks if tr.operator >= 0}
            for op in self.operators:
                op.used_seat = 0.0
                op.free = op.idx not in bound
            for tr in self.trucks:
                tr.seat_used = 0.0
                self._schedule_down_window(tr, t)

    def _schedule_down_window(self, tr: Truck, t: float):
        if self._down_dur <= 1e-6:
            tr.down_start = math.inf
            tr.down_end = math.inf
            return
        shift_start = math.floor(t / SHIFT_SECONDS) * SHIFT_SECONDS
        offset = self.rng.uniform(0.0, max(1.0, SEAT_PER_SHIFT_SEC - self._down_dur))
        tr.down_start = shift_start + offset
        tr.down_end = tr.down_start + self._down_dur

    def _in_down_window(self, tr: Truck, t: float) -> bool:
        return tr.down_start <= t < tr.down_end

    def _try_dispatch(self, tr: Truck) -> bool:
        t = self.gt.value
        if (
            self._holiday_today
            or not _in_shift_window(t)
            or tr.seat_used >= self.truck_seat_credit
            or self._in_down_window(tr, t)
        ):
            self._release_operator(tr)
            return False

        if tr.fuel <= tr.refuel_threshold:
            if self._pumps_free > 0:
                if tr.operator < 0 and not self._acquire_operator(tr):
                    return False
                self._pumps_free -= 1
                tr.phase = TruckPhase.REFUELING
                tr.timer.value = _tri(self.rng, REFUEL_DUR_MIN * 60.0, 0.10)
                return True
            return False

        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self._release_operator(tr)
            return False

        reserved_trucks = int(self.development_priority_reserved_trucks.value)
        max_production_trucks = max(1, len(self.trucks) - reserved_trucks)
        active_prod_trucks = sum(1 for trk in self.trucks if trk.phase in OPERATING_PHASES)
        if active_prod_trucks >= max_production_trucks:
            self._release_operator(tr)
            return False

        # Two-Tier Hierarchical Dispatcher Evaluation
        day_progress = (t % 86400.0) / 86400.0
        total_stock = self.ore1_stock.level + self.ore2_stock.level

        # Pacing: when total stockpile is satisfied (>= 60,000 t), pace dispatches to match active plant draw rate
        expected_hauled_by_now = self.daily_target_ore * day_progress
        if (
            total_stock >= self.target_ore_stock_level
            and self.daily_hauled_ore >= expected_hauled_by_now
        ):
            self._release_operator(tr)
            return False

        w2 = float(self.analytical_face2_weight.value)
        lhd_q = {s.face_id: len(self.stope_queues[s.face_id]) for s in self.stopes}

        dispatch_result = self.dispatcher.select_stope_for_truck(
            current_total_stock=total_stock,
            daily_hauled_so_far=self.daily_hauled_ore,
            day_progress_fraction=day_progress,
            analytical_w2=w2,
            area2_locked=self.is_area2_locked(),
            lhd_queues=lhd_q,
            target_daily_ore_tonnes=self.daily_target_ore,
        )

        if dispatch_result is None:
            # Stockpiles satisfied -> surplus capacity
            self._release_operator(tr)
            return False

        selected_stope, is_fallback = dispatch_result
        if not self._acquire_operator(tr):
            self._release_operator(tr)
            return False

        tr.target_stope_id = selected_stope.face_id
        tr.target_level = selected_stope.level_index
        tr.trip_start = self.gt.value
        tr.phase = TruckPhase.EMPTY
        tr.timer.value = self._travel_time(tr, loaded=False)
        return True

    def _acquire_operator(self, tr: Truck) -> bool:
        if tr.operator >= 0 and not self.operators[tr.operator].free:
            return self.operators[tr.operator].used_seat < SEAT_PER_SHIFT_SEC
        for op in self.operators:
            if op.free and op.used_seat < SEAT_PER_SHIFT_SEC:
                op.free = False
                tr.operator = op.idx
                return True
        return False

    def _release_operator(self, tr: Truck):
        if tr.operator >= 0:
            self.operators[tr.operator].free = True
            tr.operator = -1

    # -- Transitions ---------------------------------------------------------
    def _advance(self, tr: Truck) -> bool:
        ph = tr.phase
        if ph == TruckPhase.EMPTY:
            self._enter_stope_loadout(tr)
            return True
        if ph == TruckPhase.SPOT_LOAD:
            tr.phase = TruckPhase.ACQUIRE
            tr.timer.value = self.rng.uniform(0.0, LHD_ACQUISITION_MAX_MIN) * 60.0
            return True
        if ph == TruckPhase.ACQUIRE:
            tr.phase = TruckPhase.LOADING
            tr.timer.value = _tri(self.rng, TRUCK_LOAD_DUR_MIN * 60.0, 0.20)
            return True
        if ph == TruckPhase.LOADING:
            self._finish_loading(tr)
            return True
        if ph == TruckPhase.LOADED:
            self._enter_dump(tr)
            return True
        if ph == TruckPhase.DUMPING:
            self._finish_dumping(tr)
            return True
        if ph == TruckPhase.REFUELING:
            self._finish_refuel(tr)
            return True
        return False

    def _enter_stope_loadout(self, tr: Truck):
        sid = tr.target_stope_id
        stope = next(s for s in self.stopes if s.face_id == sid)

        # Dynamic fallback if stope entered turnaround development while en-route
        if not stope.is_ore_available:
            alt_stopes = [s for s in self.stopes if s.area_id == stope.area_id and s.is_ore_available]
            if not alt_stopes:
                alt_stopes = [s for s in self.stopes if s.is_ore_available and (not self.is_area2_locked() or s.area_id == 1)]
            if alt_stopes:
                stope = min(alt_stopes, key=lambda s: len(self.stope_queues[s.face_id]))
                sid = stope.face_id
                tr.target_stope_id = sid
                tr.target_level = stope.level_index

        if self._stope_lhds_busy[sid] >= self.num_lhds_per_stope:
            self.stope_queues[sid].append(tr)
            tr.phase = TruckPhase.WAIT_LOAD
            tr.timer.value = 0.0
        else:
            self._stope_lhds_busy[sid] += 1
            tr.phase = TruckPhase.SPOT_LOAD
            tr.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)

    def _finish_loading(self, tr: Truck):
        sid = tr.target_stope_id
        stope = next(s for s in self.stopes if s.face_id == sid)
        payload = _tri(self.rng, ORE_PAYLOAD, 0.08)

        ext_t, o1_t, o2_t = stope.extract_ore(payload)
        # If partial payload, top off from adjacent stope if available
        if ext_t < payload - 1e-6:
            alt_stopes = [s for s in self.stopes if s.area_id == stope.area_id and s.is_ore_available and s.face_id != stope.face_id]
            if not alt_stopes:
                alt_stopes = [s for s in self.stopes if s.is_ore_available and (not self.is_area2_locked() or s.area_id == 1)]
            if alt_stopes:
                top_stope = alt_stopes[0]
                top_ext, _, _ = top_stope.extract_ore(payload - ext_t)
                ext_t += top_ext

        tr.current_payload = ext_t
        tr.payload_ore_fraction = float(stope.active_parcel_ore_fraction.value)

        self._stope_lhds_busy[sid] = max(0, self._stope_lhds_busy[sid] - 1)
        if self.stope_queues[sid]:
            nxt = self.stope_queues[sid].pop(0)
            self._stope_lhds_busy[sid] += 1
            nxt.phase = TruckPhase.SPOT_LOAD
            nxt.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)

        tr.phase = TruckPhase.LOADED
        tr.timer.value = self._travel_time(tr, loaded=True)

    def _enter_dump(self, tr: Truck):
        site = self.dump_station
        if site.in_use < site.capacity:
            self._start_dump(site, tr)
        else:
            site.queue.append(tr)
            tr.phase = TruckPhase.WAIT_DUMP
            tr.timer.value = 0.0

    def _start_dump(self, site: SurfaceDumpStation, tr: Truck):
        dur = _tri(self.rng, DUMP_SPOT_MIN * 60.0, 0.20) + _tri(self.rng, DUMP_MIN * 60.0, 0.10)
        site.in_use += 1
        tr.phase = TruckPhase.DUMPING
        tr.timer.value = dur
        tr.dump_dur = dur

        f = tr.payload_ore_fraction
        ore2_mass = tr.current_payload * f
        ore1_mass = tr.current_payload * (1.0 - f)

        site._active_ore1_rate += ore1_mass / dur
        site._active_ore2_rate += ore2_mass / dur

    def _finish_dumping(self, tr: Truck):
        site = self.dump_station
        f = tr.payload_ore_fraction
        ore2_mass = tr.current_payload * f
        ore1_mass = tr.current_payload * (1.0 - f)

        site._active_ore1_rate = max(0.0, site._active_ore1_rate - ore1_mass / tr.dump_dur)
        site._active_ore2_rate = max(0.0, site._active_ore2_rate - ore2_mass / tr.dump_dur)
        site.in_use -= 1

        self.daily_hauled_ore += tr.current_payload
        self.total_extracted_ore.value += tr.current_payload
        self.ore1_hauled.value += ore1_mass
        self.ore2_hauled.value += ore2_mass
        self.annual_ore1_extracted += ore1_mass
        self.annual_ore2_extracted += ore2_mass

        if site.queue:
            nxt = site.queue.pop(0)
            self._start_dump(site, nxt)

        self.trips += 1
        self._cycle_sum += self.gt.value - tr.trip_start
        tr.current_payload = 0.0
        tr.phase = TruckPhase.IDLE
        tr.timer.value = 0.0

    def _finish_refuel(self, tr: Truck):
        self._pumps_free += 1
        tr.fuel = 100.0
        tr.phase = TruckPhase.IDLE
        tr.timer.value = 0.0

    def _travel_time(self, tr: Truck, loaded: bool) -> float:
        load_key = "loaded" if loaded else "empty"
        lvl = tr.target_level

        def seg(dist: float, kind: str) -> float:
            return dist / (SPEEDS[kind][load_key] / 3.6)

        ramp_dist = (lvl - 1) * LEVEL_SPACING_M
        t = (
            seg(SURFACE_M, "surface")
            + seg(DECLINE_M, "decline")
            + seg(ramp_dist, "ramp")
            + seg(LEVEL_DRIFT_M, "level")
        )

        if not loaded:
            cong = sum(1 for trk in self.trucks if trk.phase in OPERATING_PHASES)
            delay = BASE_PASS_BAY_DELAY_SEC + PER_TRUCK_PASS_BAY_DELAY_SEC * max(0, cong - 3)
            t += delay
            self.traffic_delay_sum += delay
        return t

    # -- Telemetry Recording --------------------------------------------------
    def _record_telemetry(self, plant_draw: PlantDrawRates):
        t_days = self.gt.value / 86400.0
        if t_days - self._last_telemetry_time < 0.20 and t_days > 0.0:
            return
        self._last_telemetry_time = t_days

        act_mode = self.plant.active_operating_mode.value.name
        cap_dev = float(self.area2_cumulative_development.value)
        tot_dev = float(self.cumulative_mine_development.value)

        a1_rem_res = sum(s.remaining_reserve for s in self.stopes if s.area_id == 1)
        a2_rem_res = sum(s.remaining_reserve for s in self.stopes if s.area_id == 2)
        n_turnaround = sum(1 for s in self.stopes if s.is_in_turnaround)

        n_operating = sum(1 for tr in self.trucks if tr.phase in OPERATING_PHASES)
        n_refueling = sum(1 for tr in self.trucks if tr.phase == TruckPhase.REFUELING)
        n_idle = max(0, len(self.trucks) - (n_operating + n_refueling))
        n_dev_reserved = float(self.development_priority_reserved_trucks.value) if hasattr(self, "development_priority_reserved_trucks") else 0.0

        rec = {
            "time": t_days,
            "Ore1Stock_mass": float(self.ore1_stock.level),
            "Ore2Stock_mass": float(self.ore2_stock.level),
            "total_system_ore_mass": float(self.ore1_stock.level + self.ore2_stock.level),
            "active_operating_mode": self.plant.active_operating_mode.value,
            "active_operating_mode_name": act_mode,
            "cumulative_milled_mass": float(self.plant.cumulative_milled_mass.value),
            "cumulative_extracted_mass": float(self.total_extracted_ore.value),
            "area1_remaining_reserve": a1_rem_res,
            "area2_remaining_reserve": a2_rem_res,
            "stopes_in_turnaround": n_turnaround,
            "stope_turnaround_dev_m": float(self.stope_turnaround_development.value),
            "cumulative_mine_development": tot_dev,
            "area2_cumulative_development": cap_dev,
            "trucks_operating": n_operating,
            "trucks_refueling": n_refueling,
            "trucks_idle": n_idle,
            "trucks_dev_reserved": n_dev_reserved,
            "truck_idle_fraction": n_idle / max(1, len(self.trucks)),
            "area2_ready": self.area2_ready,
            "area2_ready_day": float(self.area2_ready_day.value),
            "mining_priority": self.mining_priority.value,
            "analytical_face1_weight": float(self.analytical_face1_weight.value),
            "analytical_face2_weight": float(self.analytical_face2_weight.value),
            "fallback_dispatch_count": float(self.fallback_dispatch_count.value),
            "operating_npv_proxy": float(self.operating_npv_proxy.value),
            "current_discounted_cash_flow_rate": float(self.current_discounted_cash_flow_rate.value),
            "discount_factor": float(self.discount_factor.value),
        }
        self.history_records.append(rec)


# ---------------------------------------------------------------------------
# Dashboard Visualization
# ---------------------------------------------------------------------------
def plot_stope_lifecycle_dashboard(
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    output_path: str = "plots/two_area_stope_lifecycle.png",
    palette: dict = None,
    figsize: Tuple[int, int] = (18, 63),
) -> Dashboard:
    """Renders 14-panel comprehensive multi-stope lifecycle visualization dashboard."""
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_p1 = prepare_history(df_p1)
    df_p2 = prepare_history(df_p2)

    unlock_rows_p2 = df_p2[df_p2["area2_ready"] == True]
    unlock_time_p2 = float(unlock_rows_p2["time"].iloc[0]) if not unlock_rows_p2.empty else None

    unlock_rows_p1 = df_p1[df_p1["area2_ready"] == True]
    unlock_time_p1 = float(unlock_rows_p1["time"].iloc[0]) if not unlock_rows_p1.empty else None

    dash = Dashboard(
        nrows=14,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title="Multi-Stope Underground Lifecycle, Waste Rock Requirements & Two-Tier Dispatch",
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # 0. Cumulative Operating NPV Comparison
    ax0 = dash[0]
    ax0.step(
        df_p2["time"],
        df_p2["operating_npv_proxy"] / 1e6,
        label="Policy 2: Value-Oriented + Multi-Stope Two-Tier Dispatch",
        color="#2e7d32",
        linewidth=2.4,
        where="post",
    )
    ax0.step(
        df_p1["time"],
        df_p1["operating_npv_proxy"] / 1e6,
        label="Policy 1: Myopic Baseline (Suffers Area 1 Depletion Collapse)",
        color="#c62828",
        linestyle="--",
        linewidth=2.0,
        where="post",
    )
    if unlock_time_p2 is not None:
        ax0.axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Policy 2 Area 2 Unlocked (Day {unlock_time_p2:.1f})",
        )
    if unlock_time_p1 is not None:
        ax0.axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle=":",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Policy 1 Area 2 Unlocked (Day {unlock_time_p1:.1f})",
        )
    ax0.set_title("Cumulative Operating NPV (@ 5% Discount Rate): Policy 2 vs Policy 1")
    ax0.set_ylabel("Operating NPV (M$)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower right", framealpha=0.90)

    # 1. Area 1 Finite Reserves Depletion Curve (Life-of-Mine Exhaustion)
    ax1 = dash[1]
    ax1.plot(
        df_p2["time"],
        df_p2["area1_remaining_reserve"] / 1e3,
        label="Policy 2: Area 1 Remaining Reserves (k tonnes)",
        color="#1565c0",
        linewidth=2.2,
    )
    ax1.plot(
        df_p1["time"],
        df_p1["area1_remaining_reserve"] / 1e3,
        label="Policy 1: Area 1 Remaining Reserves (k tonnes)",
        color="#c62828",
        linestyle=":",
        linewidth=2.0,
    )
    ax1.axhline(0.0, color="red", linestyle="--", label="Area 1 Complete Exhaustion (0 t)")
    if unlock_time_p1 is not None:
        ax1.axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle=":",
            linewidth=2.0,
            alpha=0.85,
            label=f"★ Policy 1 Unlocked (Day {unlock_time_p1:.1f})",
        )
    ax1.set_title("Area 1 Finite Reserves Depletion (1.8M t Initial Budget across Stopes 1A, 1B, 1C)")
    ax1.set_ylabel("Reserves (k tonnes)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", framealpha=0.90)

    # 2. Area 2 Remaining Reserves (Deep Level 6 Stopes 2A, 2B, 2C)
    ax2 = dash[2]
    ax2.plot(
        df_p2["time"],
        df_p2["area2_remaining_reserve"] / 1e3,
        label="Policy 2: Area 2 Deep Reserves (k tonnes)",
        color="#2e7d32",
        linewidth=2.2,
    )
    ax2.plot(
        df_p1["time"],
        df_p1["area2_remaining_reserve"] / 1e3,
        label="Policy 1: Area 2 Deep Reserves (Unmined / Locked)",
        color="#9e9e9e",
        linestyle="--",
        linewidth=1.8,
    )
    if unlock_time_p2 is not None:
        ax2.axvline(unlock_time_p2, color="#2e7d32", linestyle="-.", linewidth=2.0, label=f"★ Policy 2 Unlocked (Day {unlock_time_p2:.1f})")
    if unlock_time_p1 is not None:
        ax2.axvline(unlock_time_p1, color="#c62828", linestyle=":", linewidth=2.0, label=f"★ Policy 1 Unlocked (Day {unlock_time_p1:.1f})")
    ax2.set_title("Area 2 Deep Reserves (4.8M t Initial Budget across Stopes 2A, 2B, 2C)")
    ax2.set_ylabel("Reserves (k tonnes)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.90)

    # 3. Stacked Development Metres: Policy 2 (Capital Decline vs Stope Turnaround Development)
    ax3 = dash[3]
    ax3.fill_between(
        df_p2["time"],
        0,
        df_p2["area2_cumulative_development"],
        label="Area 2 Capital Decline (0 → 4,000 m)",
        color="#7b1fa2",
        alpha=0.60,
        step="post",
    )
    ax3.fill_between(
        df_p2["time"],
        df_p2["area2_cumulative_development"],
        df_p2["cumulative_mine_development"],
        label="Stope Turnaround & Access Development",
        color="#f57c00",
        alpha=0.65,
        step="post",
    )
    ax3.plot(
        df_p2["time"],
        df_p2["cumulative_mine_development"],
        label="Total Mine Development (Capital + Turnaround)",
        color="#212121",
        linewidth=2.0,
    )
    ax3.axhline(4000.0, color="#7b1fa2", linestyle=":", linewidth=1.8, label="Area 2 Unlock (4,000 m)")
    if unlock_time_p2 is not None:
        ax3.axvline(unlock_time_p2, color="#7b1fa2", linestyle="-.", linewidth=2.0)
    ax3.set_title("Policy 2: Underground Development Breakdown (Capital Decline & Stope Turnaround)")
    ax3.set_ylabel("Development (m)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", framealpha=0.90)

    # 4. Stockpiles: Policy 2
    plot_ore_with_modes(
        df_p2,
        time_col="time",
        ore_cols=["total_system_ore_mass", "Ore1Stock_mass", "Ore2Stock_mass"],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Stockpiles & Campaigns: Policy 2 (Two-Tier Hierarchical Dispatch Sustaining Mode A)",
        palette=palette,
        hlines=[
            {"y": 60000.0, "color": "black", "linestyle": "--", "label": "Target Total (60k)"},
            {"y": 20400.0, "color": "red", "linestyle": ":", "label": "Critical Ore 2 (20.4k)"},
        ],
        ax=dash[4],
    )
    if unlock_time_p2 is not None:
        dash[4].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Area 2 Unlocked (Day {unlock_time_p2:.1f})",
        )
        dash[4].legend(loc="upper right", framealpha=0.90)

    # 5. Stockpiles: Policy 1
    plot_ore_with_modes(
        df_p1,
        time_col="time",
        ore_cols=["total_system_ore_mass", "Ore1Stock_mass", "Ore2Stock_mass"],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Stockpiles & Campaigns: Policy 1 (Severe Ore 2 Starvation & Depletion Shutdown)",
        palette=palette,
        hlines=[
            {"y": 60000.0, "color": "black", "linestyle": "--", "label": "Target Total (60k)"},
            {"y": 20400.0, "color": "red", "linestyle": ":", "label": "Critical Ore 2 (20.4k)"},
        ],
        ax=dash[5],
    )
    if unlock_time_p1 is not None:
        dash[5].axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Area 2 Unlocked (Day {unlock_time_p1:.1f})",
        )
        dash[5].legend(loc="upper right", framealpha=0.90)

    # 6. Policy 2: Two-Tier Dispatch Weights & Dynamic Fallback Count
    ax6 = dash[6]
    ax6.step(
        df_p2["time"],
        df_p2["analytical_face1_weight"],
        label="Analytical Weight w1 (Area 1)",
        color="#1565c0",
        where="post",
    )
    ax6.step(
        df_p2["time"],
        df_p2["analytical_face2_weight"],
        label="Analytical Weight w2 (Area 2)",
        color="#2e7d32",
        where="post",
    )
    ax6_r = ax6.twinx()
    ax6_r.step(
        df_p2["time"],
        df_p2["fallback_dispatch_count"],
        label="Dynamic Constrained Fallback Events (Count)",
        color="#e65100",
        linestyle="--",
        where="post",
    )
    ax6.set_title("Policy 2: Two-Tier Dispatch Weights (w1, w2) & Dynamic Constrained Fallbacks")
    ax6.set_ylabel("Dispatch Weight")
    ax6_r.set_ylabel("Fallback Count")
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc="upper left")
    ax6_r.legend(loc="upper right")

    # 7. Operating Modes Timeline: Policy 2
    plot_time_series(
        df_p2,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes Timeline: Policy 2",
        is_step=True,
        ax=dash[7],
    )

    # 8. Operating Modes Timeline: Policy 1
    plot_time_series(
        df_p1,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes Timeline: Policy 1",
        is_step=True,
        ax=dash[8],
    )

    # 9. Stopes in Turnaround Development: Policy 2
    plot_time_series(
        df_p2,
        y_columns=["stopes_in_turnaround"],
        title="Policy 2: Number of Underground Stopes Simultaneously in Turnaround Development",
        is_step=True,
        ax=dash[9],
    )
    dash[9].set_ylabel("Stopes in Turnaround")

    # 10. Fleet Utilization & Idle Time: Policy 2
    plot_truck_idle_and_utilization(
        df_p2,
        title="Policy 2: Haul Fleet Utilization & Idle Time Breakdown",
        ax=dash[10],
    )

    # 11. Fleet Utilization & Idle Time: Policy 1
    plot_truck_idle_and_utilization(
        df_p1,
        title="Policy 1: Haul Fleet Utilization & Idle Time Breakdown",
        ax=dash[11],
    )

    # 12. Mode Distribution: Policy 2
    plot_mode_distribution(
        df_p2,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% Time Spent - Policy 2)",
        palette=palette,
        ax=dash[12],
    )

    # 13. Mode Distribution: Policy 1
    plot_mode_distribution(
        df_p1,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% Time Spent - Policy 1)",
        palette=palette,
        ax=dash[13],
    )

    dash.save(output_path)
    print(f"Saved multi-stope lifecycle benchmark dashboard to '{output_path}'.")
    return dash


# ---------------------------------------------------------------------------
# Execution & Benchmark Runner
# ---------------------------------------------------------------------------
def run_stope_lifecycle_study(
    total_ore_to_extract: float = 6600000.0,
    warmup_ore: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    availability: float = 0.85,
    stockpile_target: float = 60000.0,
    area2_required_dev: float = 4000.0,
    area2_ready_by_day: float = 365.0,
    discount_rate: float = 0.05,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the comparative benchmark with multi-stope lifecycles and two-tier dispatch."""
    strategic_target = StrategicYearTarget(
        min_development=10000.0,
        min_ore1_production=1300000.0,
        min_ore2_production=850000.0,
    )
    area2_target = AreaReadinessTarget(
        required_development=area2_required_dev,
        ready_by_day=area2_ready_by_day,
    )

    # 1. Run Policy 2
    print("\n" + "=" * 80)
    print(" 1/2 RUNNING POLICY 2: VALUE-ORIENTED + MULTI-STOPE TWO-TIER DISPATCH")
    print("=" * 80)
    sim_p2 = TwoAreaStopeLifecycleEngine(
        policy_name="POLICY_2_VALUE_ORIENTED",
        use_two_tier_dispatch=True,
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=stockpile_target,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warmup_ore,
        strategic_targets=(strategic_target,),
        area2_readiness_target=area2_target,
        annual_discount_rate=discount_rate,
        seed=seed,
    )
    eng_p2 = drs.DRSEngine(max_step_size=DT_MAX)
    eng_p2.register(sim_p2)
    eng_p2.on_step(sim_p2.on_event)

    if total_days is not None:
        sim_p2.horizon_sec = total_days * 86400.0
        eng_p2.run(until=sim_p2.horizon_sec)
    else:
        sim_p2.total_ore_to_extract = warmup_ore
        eng_p2.run(until=float("inf"))

        sim_p2.plant.reset_mode_timers()
        sim_p2.plant.cumulative_milled_mass.value = 0.0

        sim_p2.total_ore_to_extract = total_ore_to_extract
        eng_p2.run(until=float("inf"))

    df_p2 = pd.DataFrame(sim_p2.history_records)

    # 2. Run Policy 1
    print("\n" + "=" * 80)
    print(" 2/2 RUNNING POLICY 1: LOCAL MYOPIC BASELINE (AREA 1 DEPLETION RISK)")
    print("=" * 80)
    sim_p1 = TwoAreaStopeLifecycleEngine(
        policy_name="POLICY_1_MYOPIC",
        use_two_tier_dispatch=False,
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=stockpile_target,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warmup_ore,
        strategic_targets=(strategic_target,),
        area2_readiness_target=area2_target,
        annual_discount_rate=discount_rate,
        seed=seed,
    )
    eng_p1 = drs.DRSEngine(max_step_size=DT_MAX)
    eng_p1.register(sim_p1)
    eng_p1.on_step(sim_p1.on_event)

    if total_days is not None:
        sim_p1.horizon_sec = total_days * 86400.0
        eng_p1.run(until=sim_p1.horizon_sec)
    else:
        sim_p1.total_ore_to_extract = warmup_ore
        eng_p1.run(until=float("inf"))

        sim_p1.plant.reset_mode_timers()
        sim_p1.plant.cumulative_milled_mass.value = 0.0

        sim_p1.total_ore_to_extract = total_ore_to_extract
        eng_p1.run(until=float("inf"))

    df_p1 = pd.DataFrame(sim_p1.history_records)

    # Summary Statistics
    final_p2 = df_p2.iloc[-1]
    final_p1 = df_p1.iloc[-1]
    npv_p2 = float(final_p2["operating_npv_proxy"])
    npv_p1 = float(final_p1["operating_npv_proxy"])
    value_gain = npv_p2 - npv_p1

    print("\n" + "=" * 80)
    print(" MULTI-STOPE UNDERGROUND BENCHMARK RESULTS")
    print("=" * 80)
    print("Policy 2 (Value-Oriented + Multi-Stope Two-Tier Dispatch):")
    print(f"  Area 2 Unlocked:             {sim_p2.area2_ready} (Day: {float(final_p2['area2_ready_day']):.2f})")
    print(f"  Total Ore Milled:            {float(final_p2['cumulative_milled_mass']):,.1f} t")
    print(f"  Capital Dev (Area 2):        {float(final_p2['area2_cumulative_development']):,.1f} / {area2_required_dev:.1f} m")
    print(f"  Stope Turnaround Dev:        {float(final_p2['stope_turnaround_dev_m']):,.1f} m")
    print(f"  Dynamic Fallback Dispatches: {float(final_p2['fallback_dispatch_count']):,.0f}")
    print(f"  Operating Net Present Value: ${npv_p2:,.2f}")

    print("\nPolicy 1 (Local-Objective Myopic Baseline):")
    print(f"  Area 2 Unlocked:             {sim_p1.area2_ready} (Day: {float(final_p1['area2_ready_day']):.2f})")
    print(f"  Total Ore Milled:            {float(final_p1['cumulative_milled_mass']):,.1f} t")
    print(f"  Capital Dev (Area 2):        {float(final_p1['area2_cumulative_development']):,.1f} / {area2_required_dev:.1f} m")
    print(f"  Operating Net Present Value: ${npv_p1:,.2f}")

    print("\n" + "-" * 80)
    print(f" >>> TOTAL VALUE CREATED BY HIERARCHICAL MULTI-STOPE CONTROL: ${value_gain:,.2f} <<<")
    print("-" * 80)
    print("=" * 80 + "\n")

    if plot and len(df_p2) > 0 and len(df_p1) > 0:
        plot_stope_lifecycle_dashboard(df_p1, df_p2)

    return df_p1, df_p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Stope Underground Lifecycle Simulation Benchmark")
    parser.add_argument("--total_ore_to_extract", type=float, default=6600000.0)
    parser.add_argument("--warmup_ore", type=float, default=600000.0)
    parser.add_argument("--total_days", type=float, default=None)
    parser.add_argument("--trucks", type=int, default=18)
    parser.add_argument("--operators", type=int, default=18)
    parser.add_argument("--availability", type=float, default=0.85)
    parser.add_argument("--stockpile_target", type=float, default=60000.0)
    parser.add_argument("--area2_required_dev", type=float, default=4000.0)
    parser.add_argument("--area2_ready_by_day", type=float, default=365.0)
    parser.add_argument("--discount_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    run_stope_lifecycle_study(
        total_ore_to_extract=args.total_ore_to_extract,
        warmup_ore=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        availability=args.availability,
        stockpile_target=args.stockpile_target,
        area2_required_dev=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        discount_rate=args.discount_rate,
        seed=args.seed,
        plot=not args.no_plot,
    )

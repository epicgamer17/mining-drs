"""Two-Area Full Hierarchy Simulation: Strategic Planning, Tactical Control, and Analytical Operational Blending.

Implements the unified three-level hierarchical decision framework described across
Slides 6-30 of the research presentation (copy0813.pptx):

  - Level 1 (Strategic): Discounted Cash Flow (DCF) NPV valuation & multi-year trajectory targets.
  - Level 2 (Tactical): Monthly progress reviews, trajectory ratios (R_dev, R_area2, R_ore1, R_ore2),
    and adaptive mining priority selection (DEVELOPMENT, PRODUCTION, BALANCED).
  - Level 3 (Operational): Closed-form analytical face-allocation equations (Appendix A & B, Slide 29)
    solving exact mass rates (r1, r2) and dispatch weights (w1, w2) from active parcel grades.

Comparative Policy Benchmark:
  - Policy 1 (Myopic Baseline - Slide 22): Local shift-level tonnage maximization without capital
    development reservation or trajectory feedback. Area 2 never unlocks; trapped in Mode B.
  - Policy 2 (Hierarchical Value-Oriented Control with Analytical Blending - Slide 23 & 29):
    Proactive fleet reservation, on-time Area 2 physical unlocking (~Day 80-84), and analytical
    operational face-allocation blending maximizing high-grade Mode A throughput and whole-mine NPV.
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
from drs_mining.config import MILL_MODES, FLEET_MODES
from drs_mining.components.modes import OperatingMode
from drs_mining.components.plant import MetallurgicalPlant, PlantDrawRates
from drs_mining.components.stockpiles import Stockpile
from drs_mining.components.controllers import OperatingModeController
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.mine_face import MineFace
from drs_mining.components.planning import (
    AreaReadinessTarget,
    StrategicYearTarget,
    strategic_target_for_year,
    trajectory_progress_ratio,
    select_fleet_mode,
)
from drs_mining.components.plot import (
    MODE_PALETTE,
    prepare_history,
    plot_ore_with_modes,
    plot_mode_distribution,
    plot_mode_dwell_times,
    plot_attributed_deficit,
    plot_deficit_disparity,
    plot_deficit_breakdown_bar,
    plot_truck_idle_and_utilization,
    print_transition_log,
    print_deficit_by_mode,
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
    target_face_id: int = 1
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
# Full Hierarchy Simulation Engine
# ---------------------------------------------------------------------------
class TwoAreaFullHierarchyEngine(drs.Module):
    """Underground DES simulation module implementing full 3-level hierarchical control."""

    def __init__(
        self,
        policy_name: str = "POLICY_2_VALUE_ORIENTED",  # "POLICY_1_MYOPIC" or "POLICY_2_VALUE_ORIENTED"
        use_analytical_blending: bool = True,
        # NOTE: Sizing the haulage fleet to 18 AD30 trucks (18 operators) rather than
        # a standard ~14-truck single-level fleet is specifically chosen to account for
        # the deep Level 6 haulage cycle time penalty (+60% longer cycle, ~45 min vs ~28 min).
        # In Mode A (40% Ore 2 demand), metallurgical mass-balance dictates that 83.3% of
        # all ore must be hauled from deep Level 6 (Face 2). An 18-truck fleet ensures
        # sufficient haulage throughput (~6,000 t/d) from depth without starving the mill,
        # while non-hauling/surplus capacity is dynamically redirected into development.
        num_trucks: int = 18,
        num_operators: int = 18,
        num_lhds_per_face: int = 2,
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
        self.use_analytical_blending = use_analytical_blending
        self.num_trucks = num_trucks
        self.num_operators = num_operators
        self.num_lhds_per_face = num_lhds_per_face
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
        self.holidays: Set[int] = set()
        self._cur_day = -1
        self._shift_marker = -1
        self._holiday_today = False

        # Global time tracker
        self.gt = drs.Timer("gt", 0.0, rate=1.0)

        # 1. Dual Mine Faces (Area 1 70/30: 30% Ore 2, Area 2 65/35: 35% Ore 2)
        self.gen1 = StochasticFaciesGenerator(
            mean_fraction=0.30,
            std_dev=0.05,
            prob_new_facies=0.3,
            variation_same_facies=0.01,
        )
        face_ore_capacity = total_ore_to_extract / 2.0
        face_warmup = ore_to_be_extracted_during_warming_period / 2.0
        self.face1 = MineFace(
            name="mine_face_1",
            face_id=1,
            generator=self.gen1,
            min_ore_mass=30000.0,
            max_ore_mass=50000.0,
            total_ore_to_extract=face_ore_capacity,
            ore_to_be_extracted_during_warming_period=face_warmup,
            mean_ore_fraction=0.30,
            std_dev_ore_fraction=0.05,
            prob_new_facies=0.3,
            variation_same_facies=0.01,
            initial_parcel_mass=40000.0,
        )

        self.gen2 = StochasticFaciesGenerator(
            mean_fraction=0.35,
            std_dev=0.05,
            prob_new_facies=0.3,
            variation_same_facies=0.01,
        )
        self.face2 = MineFace(
            name="mine_face_2",
            face_id=2,
            generator=self.gen2,
            min_ore_mass=30000.0,
            max_ore_mass=50000.0,
            total_ore_to_extract=face_ore_capacity,
            ore_to_be_extracted_during_warming_period=face_warmup,
            mean_ore_fraction=0.35,
            std_dev_ore_fraction=0.05,
            prob_new_facies=0.3,
            variation_same_facies=0.01,
            initial_parcel_mass=40000.0,
        )
        self.faces = [self.face1, self.face2]

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
        self.sustaining_cumulative_development = drs.Level("sustaining_cumulative_development", 0.0)

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

        # 5. Face Loadouts & Surface Dump Station
        self.face_queues = {1: [], 2: []}
        self._face_lhds_busy = {1: 0, 2: 0}
        self.dump_station = SurfaceDumpStation()
        self._pumps_free = N_FUEL_PUMPS

        # 6. Strategic & Tactical State Variables
        self.strategic_planning_started = False
        self.strategic_year_index = drs.Level("strategic_year_index", 0.0)
        self.strategic_year_timer = drs.Timer("strategic_year_timer", 0.0, rate=1.0)
        self.tactical_review_timer = drs.Timer("tactical_review_timer", 0.0, rate=1.0)
        self.tactical_review_count = drs.Level("tactical_review_count", 0.0)
        self.fleet_mode = (
            FLEET_MODES["PRODUCTION"]
            if policy_name == "POLICY_1_MYOPIC"
            else FLEET_MODES["BALANCED"]
        )
        self.mining_priority = self.fleet_mode

        # 7. Area 2 Readiness Variables
        self.area2_ready = False
        self.area2_ready_day = drs.Level("area2_ready_day", -1.0)
        self.area2_deadline_missed = False
        self.area2_currently_late = False
        self.area2_completed_late = False
        self.area2_readiness_fraction = drs.Level("area2_readiness_fraction", 0.0)
        self.area2_readiness_trajectory_ratio = drs.Level(
            "area2_readiness_trajectory_ratio", 1.0
        )

        # 8. Economics Variables
        self.cumulative_processed_ore1 = drs.Level("cumulative_processed_ore1", 0.0)
        self.cumulative_processed_ore2 = drs.Level("cumulative_processed_ore2", 0.0)
        self.cumulative_cash_flow = drs.Level("cumulative_cash_flow", 0.0)
        self.cumulative_discounted_cash_flow = drs.Level(
            "cumulative_discounted_cash_flow", 0.0
        )
        self.current_cash_flow_rate = drs.Level("current_cash_flow_rate", 0.0)
        self.current_discounted_cash_flow_rate = drs.Level(
            "current_discounted_cash_flow_rate", 0.0
        )
        self.discount_factor = drs.Level("discount_factor", 1.0)
        self.operating_npv_proxy = drs.Level("operating_npv_proxy", 0.0)

        # Annual Progress Trackers
        self.annual_ore1_extracted = 0.0
        self.annual_ore2_extracted = 0.0
        self.annual_development_start = 0.0
        self.development_priority_reserved_trucks = drs.Level(
            "development_priority_reserved_trucks", 0.0
        )
        self.development_rate_m_per_day = drs.Level("development_rate_m_per_day", 0.0)

        # Trajectory Ratios
        self.ore1_trajectory_ratio = drs.Level("ore1_trajectory_ratio", 1.0)
        self.ore2_trajectory_ratio = drs.Level("ore2_trajectory_ratio", 1.0)
        self.development_trajectory_ratio = drs.Level(
            "development_trajectory_ratio", 1.0
        )

        # Operational Analytical Blending Trackers (Appendix A & B, Slide 29)
        self.analytical_face1_weight = drs.Level("analytical_face1_weight", 1.0)
        self.analytical_face2_weight = drs.Level("analytical_face2_weight", 0.0)
        self.analytical_face1_rate_target = drs.Level("analytical_face1_rate_target", 0.0)
        self.analytical_face2_rate_target = drs.Level("analytical_face2_rate_target", 0.0)
        self.analytical_blend_feasible = drs.Level("analytical_blend_feasible", 1.0)

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

    def is_face1_exhausted(self) -> bool:
        return self.face1.cumulative_extracted_mass.value >= self.face1.total_ore_to_extract - 1e-6

    def is_face2_exhausted(self) -> bool:
        return self.face2.cumulative_extracted_mass.value >= self.face2.total_ore_to_extract - 1e-6

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
                f"\n >>> [{self.policy_name} UNLOCK] Area 2 (Face 2) UNLOCKED on Strategic Day {strategic_days:.2f}! <<<\n"
            )

        # Deadline Tracking
        ready_by_day = target.ready_by_day
        if ready_by_day is not None and ready_by_day > 0.0:
            elapsed_fraction = max(1e-4, min(1.0, strategic_days / ready_by_day))
            self.area2_readiness_trajectory_ratio.value = trajectory_progress_ratio(
                actual=progress,
                annual_target=required,
                elapsed_fraction=elapsed_fraction,
            )

            deadline_exceeded = strategic_days > ready_by_day
            if not self.area2_ready:
                if deadline_exceeded:
                    self.area2_deadline_missed = True
                    self.area2_currently_late = True
                    self.area2_completed_late = False
                else:
                    self.area2_deadline_missed = False
                    self.area2_currently_late = False
                    self.area2_completed_late = False
            else:
                self.area2_currently_late = False
                if float(self.area2_ready_day.value) > ready_by_day + 1e-6:
                    self.area2_deadline_missed = True
                    self.area2_completed_late = True
                else:
                    self.area2_deadline_missed = False
                    self.area2_completed_late = False
        else:
            self.area2_readiness_trajectory_ratio.value = 1.0

    # -- Strategic Economics -------------------------------------------------
    def _update_strategic_economics(
        self, out1_t_sec: float, out2_t_sec: float, dt_days: float
    ):
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

        cash_flow_rate = revenue_rate - (
            production_cost_rate + development_cost_rate + fixed_cost_rate
        )
        self.current_cash_flow_rate.value = cash_flow_rate
        self.cumulative_cash_flow.value += cash_flow_rate * dt_days

        strategic_days = (
            float(self.strategic_year_index.value) * self.strategic_period_days
            + float(self.strategic_year_timer.value)
        )
        dfactor = 1.0 / (
            (1.0 + self.annual_discount_rate) ** (strategic_days / 365.0)
        )
        self.discount_factor.value = dfactor

        discounted_rate = cash_flow_rate * dfactor
        self.current_discounted_cash_flow_rate.value = discounted_rate
        self.cumulative_discounted_cash_flow.value += discounted_rate * dt_days
        self.operating_npv_proxy.value = (
            self.cumulative_discounted_cash_flow.value
        )

    # -- Analytical Operational Face Allocation (Appendix A & B, Slide 29) ----
    def _compute_analytical_face_allocation(self, plant_draw: PlantDrawRates):
        if self.is_area2_locked():
            self.analytical_face1_weight.value = 1.0
            self.analytical_face2_weight.value = 0.0
            self.analytical_face1_rate_target.value = plant_draw.total
            self.analytical_face2_rate_target.value = 0.0
            self.analytical_blend_feasible.value = 1.0
            return

        f1_done = self.is_face1_exhausted()
        f2_done = self.is_face2_exhausted()
        if f1_done and not f2_done:
            self.analytical_face1_weight.value = 0.0
            self.analytical_face2_weight.value = 1.0
            self.analytical_face1_rate_target.value = 0.0
            self.analytical_face2_rate_target.value = plant_draw.total
            self.analytical_blend_feasible.value = 1.0
            return
        if f2_done and not f1_done:
            self.analytical_face1_weight.value = 1.0
            self.analytical_face2_weight.value = 0.0
            self.analytical_face1_rate_target.value = plant_draw.total
            self.analytical_face2_rate_target.value = 0.0
            self.analytical_blend_feasible.value = 1.0
            return

        f1_ore1 = 1.0 - float(self.face1.active_parcel_ore_fraction.value)
        f2_ore1 = 1.0 - float(self.face2.active_parcel_ore_fraction.value)

        alloc = solve_face_allocation_rates(
            target_ore1_rate=plant_draw.ore1,
            target_ore2_rate=plant_draw.ore2,
            face1_ore1_fraction=f1_ore1,
            face2_ore1_fraction=f2_ore1,
        )

        self.analytical_face1_weight.value = alloc.face1_weight
        self.analytical_face2_weight.value = alloc.face2_weight
        self.analytical_face1_rate_target.value = alloc.face1_rate
        self.analytical_face2_rate_target.value = alloc.face2_rate
        self.analytical_blend_feasible.value = 1.0 if alloc.is_feasible else 0.0

    # -- DRS Engine Hooks ----------------------------------------------------
    def is_terminating_condition_met(self) -> bool:
        total_extracted = sum(
            f.cumulative_extracted_mass.value for f in self.faces
        )
        if total_extracted >= self.total_ore_to_extract - 1e-6:
            return True
        if self.is_face1_exhausted() and self.is_face2_exhausted():
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
            if self.mode_controller.active_campaign_mode.value.name
            == "SHUTDOWN"
            else self.mode_controller.duration_of_production_campaigns
        )
        rem_camp_days = max(
            0.0,
            camp_thresh
            - self.mode_controller.current_campaign_duration.value,
        )
        if rem_camp_days > 1e-6:
            best = min(best, rem_camp_days * 86400.0)

        rem_tactical_days = max(
            0.0,
            self.tactical_review_period_days
            - self.tactical_review_timer.value,
        )
        if rem_tactical_days > 1e-6:
            best = min(best, rem_tactical_days * 86400.0)

        if "_CONTINGENCY" in self.plant.active_operating_mode.value.name:
            c_thresh = self.plant.duration_of_contingency_segments
            rem_c_days = max(
                0.0,
                c_thresh - self.plant.current_contingency_duration.value,
            )
            if rem_c_days > 1e-6:
                best = min(best, rem_c_days * 86400.0)

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

        if self.is_area2_locked():
            f_blend = self.face1.active_parcel_ore_fraction.value
        else:
            f_blend = (
                self.face1.active_parcel_ore_fraction.value
                + self.face2.active_parcel_ore_fraction.value
            ) / 2.0

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

        # Policy-Driven Development Calculation (Slide 22 vs Slide 23)
        if self.policy_name == "POLICY_1_MYOPIC":
            n_operating_trucks = sum(
                1 for tr in self.trucks if tr.phase in OPERATING_PHASES
            )
            total_trucks = len(self.trucks)
            available_extra = max(0, total_trucks - n_operating_trucks)
            dev_trucks = max(2.0, float(available_extra))
            if self.is_area2_locked() and self.is_face1_exhausted():
                # Once Area 1 is depleted, redirect all development into Area 2 decline to unlock it!
                frac_to_area2 = 1.0
            else:
                frac_to_area2 = 0.05
        else:
            reserved_trucks = float(
                self.development_priority_reserved_trucks.value
            )
            n_operating_trucks = sum(
                1 for tr in self.trucks if tr.phase in OPERATING_PHASES
            )
            total_trucks = len(self.trucks)
            available_extra = max(0, total_trucks - n_operating_trucks)
            locked_boost = (
                (total_trucks * 0.35) if self.is_area2_locked() else 0.0
            )
            dev_trucks = (
                max(reserved_trucks, float(available_extra)) + locked_boost
            )
            prio = self.fleet_mode
            frac_to_area2 = (
                0.85
                if prio == FLEET_MODES["DEVELOPMENT"]
                else (0.60 if prio == FLEET_MODES["BALANCED"] else 0.35)
            )

        self.development_rate_m_per_day.value = (
            dev_trucks * DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY
        )
        delta_dev = self.development_rate_m_per_day.value * dt_days
        self.cumulative_mine_development.value += delta_dev

        # Dual Development Accounting: Capital Decline vs Sustaining Mine Dev
        if self.is_area2_locked() and self.strategic_planning_started:
            capital_advance = delta_dev * frac_to_area2
            sustaining_advance = delta_dev * (1.0 - frac_to_area2)
            self.area2_cumulative_development.value += capital_advance
            self.sustaining_cumulative_development.value += sustaining_advance
        else:
            self.sustaining_cumulative_development.value += delta_dev

        self._update_area2_readiness()
        self._update_strategic_economics(out1, out2, dt_days)

        self.total_extracted_ore.step(dt)
        self.ore1_hauled.step(dt)
        self.ore2_hauled.step(dt)
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
        total_extracted = sum(
            f.cumulative_extracted_mass.value for f in self.faces
        )
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
                self.annual_development_start = float(
                    self.cumulative_mine_development.value
                )

        if not self.strategic_planning_started:
            return

        if self.strategic_year_timer.value >= self.strategic_period_days - 1e-6:
            self.strategic_year_index.value += 1.0
            self.strategic_year_timer.reset()
            self.annual_ore1_extracted = 0.0
            self.annual_ore2_extracted = 0.0
            self.annual_development_start = float(
                self.cumulative_mine_development.value
            )

        elapsed_year_fraction = max(
            1e-4,
            min(1.0, self.strategic_year_timer.value / self.strategic_period_days),
        )
        current_target = strategic_target_for_year(
            self.strategic_targets, int(self.strategic_year_index.value)
        )

        annual_dev = (
            float(self.cumulative_mine_development.value)
            - self.annual_development_start
        )
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
            self.tactical_review_timer.value
            >= self.tactical_review_period_days - 1e-6
            or self.tactical_review_count.value == 0.0
        ):
            self.tactical_review_timer.reset()
            self.tactical_review_count.value += 1.0

            if self.policy_name == "POLICY_1_MYOPIC":
                self.fleet_mode = FLEET_MODES["PRODUCTION"]
                self.mining_priority = self.fleet_mode
                self.development_priority_reserved_trucks.value = 0.0
            else:
                effective_dev_ratio = min(
                    float(self.development_trajectory_ratio.value),
                    float(self.area2_readiness_trajectory_ratio.value),
                )
                selected = select_fleet_mode(
                    development_ratio=effective_dev_ratio,
                    ore1_ratio=float(self.ore1_trajectory_ratio.value),
                    ore2_ratio=float(self.ore2_trajectory_ratio.value),
                    tolerance=self.tactical_progress_tolerance,
                )
                self.fleet_mode = selected
                self.mining_priority = self.fleet_mode
                if selected == FLEET_MODES["DEVELOPMENT"]:
                    n_res = max(1, int(round(len(self.trucks) * 0.20)))
                    self.development_priority_reserved_trucks.value = float(
                        n_res
                    )
                else:
                    self.development_priority_reserved_trucks.value = 0.0

    def _update_operating_mode_and_targets(self):
        """Updates campaign mode, plant operating mode, and extraction target matching shelswell_single_face."""
        # 1. Update Campaign Mode (Mode A, Mode B, Shutdown)
        camp_mode = self.mode_controller.update(
            ore2_stock_level=self.ore2_stock.level,
            total_stock_level=self.ore1_stock.level + self.ore2_stock.level,
        )

        if self.is_area2_locked() or self.is_face2_exhausted():
            f_blend = self.face1.active_parcel_ore_fraction.value
        elif self.is_face1_exhausted():
            f_blend = self.face2.active_parcel_ore_fraction.value
        else:
            f_blend = (
                self.face1.active_parcel_ore_fraction.value
                + self.face2.active_parcel_ore_fraction.value
            ) / 2.0

        # 2. Update Plant Operational State & Draw Rates
        plant_draw, _ = self.plant.get_target_rates(
            camp_mode,
            ore1_level=self.ore1_stock.level,
            ore2_level=self.ore2_stock.level,
            stockpile2_routing_fraction=f_blend,
        )

        # 3. Derive Daily Extraction Target directly from plant demand
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
        offset = self.rng.uniform(
            0.0, max(1.0, SEAT_PER_SHIFT_SEC - self._down_dur)
        )
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

        # If all accessible faces are exhausted, trucks cannot dispatch for ore
        if self.is_area2_locked():
            if self.is_face1_exhausted():
                self._release_operator(tr)
                return False
        else:
            if self.is_face1_exhausted() and self.is_face2_exhausted():
                self._release_operator(tr)
                return False

        total_stock = self.ore1_stock.level + self.ore2_stock.level
        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self._release_operator(tr)
            return False

        reserved_trucks = int(self.development_priority_reserved_trucks.value)
        max_production_trucks = max(1, len(self.trucks) - reserved_trucks)
        active_prod_trucks = sum(
            1 for trk in self.trucks if trk.phase in OPERATING_PHASES
        )
        if active_prod_trucks >= max_production_trucks:
            self._release_operator(tr)
            return False

        day_progress = (t % 86400.0) / 86400.0
        expected_hauled_by_now = self.daily_target_ore * day_progress
        if (
            total_stock >= self.target_ore_stock_level
            and self.daily_hauled_ore > expected_hauled_by_now + 100.0
        ):
            self._release_operator(tr)
            return False

        if not self._acquire_operator(tr):
            self._release_operator(tr)
            return False

        # Dispatch Face Selection: Analytical (Slide 29) or Fixed Heuristic
        target_face_id = self._select_face()
        tr.target_face_id = target_face_id
        tr.target_level = AREA1_LEVEL if target_face_id == 1 else AREA2_LEVEL

        tr.trip_start = self.gt.value
        tr.phase = TruckPhase.EMPTY
        tr.timer.value = self._travel_time(tr, loaded=False)
        return True

    def _select_face(self) -> int:
        if self.is_area2_locked():
            return 1

        f1_done = self.is_face1_exhausted()
        f2_done = self.is_face2_exhausted()
        if f1_done and not f2_done:
            return 2
        if f2_done and not f1_done:
            return 1
        if f1_done and f2_done:
            return 1

        if (
            self.use_analytical_blending
            and self.policy_name != "POLICY_1_MYOPIC"
        ):
            # Policy 2: Analytical Operational Face-Allocation (Slide 29)
            w2 = float(self.analytical_face2_weight.value)
            return 2 if self.rng.random() < w2 else 1
        else:
            # Policy 1: Fixed heuristic dispatch
            mode_name = self.plant.active_operating_mode.value.name
            p_face2 = 0.65 if "MODE_A" in mode_name else 0.35
            return 2 if self.rng.random() < p_face2 else 1

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
            self._enter_face_loadout(tr)
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

    def _enter_face_loadout(self, tr: Truck):
        fid = tr.target_face_id
        if self._face_lhds_busy[fid] >= self.num_lhds_per_face:
            self.face_queues[fid].append(tr)
            tr.phase = TruckPhase.WAIT_LOAD
            tr.timer.value = 0.0
        else:
            self._face_lhds_busy[fid] += 1
            tr.phase = TruckPhase.SPOT_LOAD
            tr.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)

    def _finish_loading(self, tr: Truck):
        payload = _tri(self.rng, ORE_PAYLOAD, 0.08)
        tr.current_payload = payload

        fid = tr.target_face_id
        face = self.face1 if fid == 1 else self.face2
        tr.payload_ore_fraction = face.active_parcel_ore_fraction.value

        face.parcel_extracted_mass.value += payload
        face.cumulative_extracted_mass.value += payload
        if face.parcel_extracted_mass.value >= face.active_parcel_initial_mass.value:
            face._load_next_batch()
            face.parcel_extracted_mass.value = 0.0

        self._face_lhds_busy[fid] -= 1
        if self.face_queues[fid]:
            nxt = self.face_queues[fid].pop(0)
            self._face_lhds_busy[fid] += 1
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
        dur = _tri(self.rng, DUMP_SPOT_MIN * 60.0, 0.20) + _tri(
            self.rng, DUMP_MIN * 60.0, 0.10
        )
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

        site._active_ore1_rate = max(
            0.0, site._active_ore1_rate - ore1_mass / tr.dump_dur
        )
        site._active_ore2_rate = max(
            0.0, site._active_ore2_rate - ore2_mass / tr.dump_dur
        )
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
            delay = BASE_PASS_BAY_DELAY_SEC + PER_TRUCK_PASS_BAY_DELAY_SEC * max(
                0, cong - 3
            )
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
        sus_dev = float(self.sustaining_cumulative_development.value)
        tot_dev = float(self.cumulative_mine_development.value)

        n_operating = sum(1 for tr in self.trucks if tr.phase in OPERATING_PHASES)
        n_refueling = sum(1 for tr in self.trucks if tr.phase == TruckPhase.REFUELING)
        n_idle = max(0, len(self.trucks) - (n_operating + n_refueling))
        n_dev_reserved = float(self.development_priority_reserved_trucks.value) if hasattr(self, "development_priority_reserved_trucks") else 0.0

        rec = {
            "time": t_days,
            "Ore1Stock_mass": float(self.ore1_stock.level),
            "Ore2Stock_mass": float(self.ore2_stock.level),
            "total_system_ore_mass": float(
                self.ore1_stock.level + self.ore2_stock.level
            ),
            "active_operating_mode": self.plant.active_operating_mode.value,
            "active_operating_mode_name": act_mode,
            "cumulative_milled_mass": float(
                self.plant.cumulative_milled_mass.value
            ),
            "cumulative_extracted_mass": float(
                self.total_extracted_ore.value
            ),
            "cumulative_mine_development": tot_dev,
            "area2_cumulative_development": cap_dev,
            "sustaining_cumulative_development": sus_dev,
            "trucks_operating": n_operating,
            "trucks_refueling": n_refueling,
            "trucks_idle": n_idle,
            "trucks_dev_reserved": n_dev_reserved,
            "truck_idle_fraction": n_idle / max(1, len(self.trucks)),
            "area2_readiness_fraction": float(
                self.area2_readiness_fraction.value
            ),
            "area2_ready": self.area2_ready,
            "area2_ready_day": float(self.area2_ready_day.value),
            "mining_priority": self.mining_priority.value,
            "development_priority_reserved_trucks": float(
                self.development_priority_reserved_trucks.value
            ),
            "development_trajectory_ratio": float(
                self.development_trajectory_ratio.value
            ),
            "area2_readiness_trajectory_ratio": float(
                self.area2_readiness_trajectory_ratio.value
            ),
            "ore1_trajectory_ratio": float(self.ore1_trajectory_ratio.value),
            "ore2_trajectory_ratio": float(self.ore2_trajectory_ratio.value),
            "analytical_face1_weight": float(
                self.analytical_face1_weight.value
            ),
            "analytical_face2_weight": float(
                self.analytical_face2_weight.value
            ),
            "analytical_blend_feasible": float(
                self.analytical_blend_feasible.value
            ),
            "operating_npv_proxy": float(self.operating_npv_proxy.value),
            "current_discounted_cash_flow_rate": float(
                self.current_discounted_cash_flow_rate.value
            ),
            "discount_factor": float(self.discount_factor.value),
            "tactical_review_count": float(self.tactical_review_count.value),
            "strategic_year_index": float(self.strategic_year_index.value),
        }
        self.history_records.append(rec)


# ---------------------------------------------------------------------------
# Data Preparation & Visualization Dashboard
# ---------------------------------------------------------------------------
def plot_full_hierarchy_dashboard(
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    output_path: str = "plots/two_area_full_hierarchy.png",
    palette: dict = None,
    figsize: Tuple[int, int] = (18, 63),
) -> Dashboard:
    """Renders 14-panel comprehensive comparative visualization dashboard."""
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_p1 = prepare_history(df_p1)
    df_p2 = prepare_history(df_p2)

    unlock_rows_p2 = df_p2[df_p2["area2_ready"] == True]
    unlock_time_p2 = (
        float(unlock_rows_p2["time"].iloc[0])
        if not unlock_rows_p2.empty
        else None
    )

    unlock_rows_p1 = df_p1[df_p1["area2_ready"] == True]
    unlock_time_p1 = (
        float(unlock_rows_p1["time"].iloc[0])
        if not unlock_rows_p1.empty
        else None
    )

    dash = Dashboard(
        nrows=14,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title="Three-Level Strategic, Tactical & Analytical Blending Mining Benchmark",
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # 0. Cumulative Operating NPV Comparison (Policy 2 vs Policy 1)
    ax0 = dash[0]
    ax0.step(
        df_p2["time"],
        df_p2["operating_npv_proxy"] / 1e6,
        label="Policy 2: Value-Oriented Control + Analytical Blending",
        color="#2e7d32",
        linewidth=2.4,
        where="post",
    )
    ax0.step(
        df_p1["time"],
        df_p1["operating_npv_proxy"] / 1e6,
        label="Policy 1: Local-Objective Myopic Baseline",
        color="#c62828",
        linestyle="--",
        linewidth=2.0,
        where="post",
    )
    if unlock_time_p2 is not None:
        ax0.axvspan(
            df_p2["time"].min(),
            unlock_time_p2,
            color="#ffebee",
            alpha=0.35,
            label="Policy 2: Area 2 Locked (Capital Phase)",
        )
        ax0.axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Policy 2 Area 2 Unlocked (Day {unlock_time_p2:.1f})",
        )
        t_max = max(df_p2["time"].max(), df_p1["time"].max())
        text_x = (
            unlock_time_p2 + (t_max * 0.03)
            if (unlock_time_p2 < t_max * 0.80)
            else unlock_time_p2 - (t_max * 0.18)
        )
        y_pos = float(df_p2["operating_npv_proxy"].max() / 1e6) * 0.55
        ax0.annotate(
            f"★ P2 AREA 2 UNLOCKED\nDay {unlock_time_p2:.1f}",
            xy=(unlock_time_p2, y_pos),
            xytext=(text_x, y_pos * 1.15),
            arrowprops=dict(
                facecolor="#2e7d32",
                edgecolor="#2e7d32",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#e8f5e9",
                edgecolor="#2e7d32",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#2e7d32",
            zorder=10,
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
        t_max = max(df_p2["time"].max(), df_p1["time"].max())
        text_x_p1 = (
            unlock_time_p1 + (t_max * 0.03)
            if (unlock_time_p1 < t_max * 0.80)
            else unlock_time_p1 - (t_max * 0.18)
        )
        y_pos_p1 = float(df_p1["operating_npv_proxy"].max() / 1e6) * 0.45
        ax0.annotate(
            f"★ P1 AREA 2 UNLOCKED\nDay {unlock_time_p1:.1f}",
            xy=(unlock_time_p1, y_pos_p1),
            xytext=(text_x_p1, y_pos_p1 * 0.85),
            arrowprops=dict(
                facecolor="#c62828",
                edgecolor="#c62828",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffebee",
                edgecolor="#c62828",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#c62828",
            zorder=10,
        )

    ax0.set_title(
        "Cumulative Operating NPV (@ 5% Discount Rate): Policy 2 vs Policy 1"
    )
    ax0.set_ylabel("Operating NPV (M$)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower right", framealpha=0.90)

    # 1. Daily Discounted Cash Flow Rates Comparison
    ax1 = dash[1]
    ax1.plot(
        df_p2["time"],
        df_p2["current_discounted_cash_flow_rate"] / 1e3,
        label="Discounted CF Rate: Policy 2 ($k/day)",
        color="#2e7d32",
        alpha=0.85,
    )
    ax1.plot(
        df_p1["time"],
        df_p1["current_discounted_cash_flow_rate"] / 1e3,
        label="Discounted CF Rate: Policy 1 ($k/day)",
        color="#c62828",
        linestyle=":",
        alpha=0.75,
    )
    if unlock_time_p2 is not None:
        ax1.axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    if unlock_time_p1 is not None:
        ax1.axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle=":",
            linewidth=2.0,
            alpha=0.85,
        )
    ax1.set_title("Daily Discounted Cash Flow Rate ($k/day)")
    ax1.set_ylabel("Rate ($k/day)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", framealpha=0.90)

    # 2. Stacked Development Metres: Policy 2 (Capital vs Sustaining)
    ax2 = dash[2]
    ax2.fill_between(
        df_p2["time"],
        0,
        df_p2["area2_cumulative_development"],
        label="Area 2 Capital Decline (0 → 4,000 m Target)",
        color="#7b1fa2",
        alpha=0.60,
        step="post",
    )
    ax2.fill_between(
        df_p2["time"],
        df_p2["area2_cumulative_development"],
        df_p2["cumulative_mine_development"],
        label="General Sustaining Mine Development",
        color="#2e7d32",
        alpha=0.35,
        step="post",
    )
    ax2.step(
        df_p2["time"],
        df_p2["cumulative_mine_development"],
        label="Total Combined Development",
        color="#1b5e20",
        linewidth=2.0,
        where="post",
    )
    ax2.axhline(
        4000.0,
        color="#7b1fa2",
        linestyle=":",
        linewidth=1.8,
        label="Area 2 Unlock Threshold (4,000 m)",
    )
    if unlock_time_p2 is not None:
        ax2.axvline(
            unlock_time_p2,
            color="#7b1fa2",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    ax2.set_title(
        "Policy 2: Stacked Underground Development (Rapid Capital Advance & Timely Unlock)"
    )
    ax2.set_ylabel("Development (m)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", framealpha=0.90)

    # 3. Stacked Development Metres: Policy 1 (Myopic Neglect & Emergency Finish)
    ax3 = dash[3]
    ax3.fill_between(
        df_p1["time"],
        0,
        df_p1["area2_cumulative_development"],
        label="Area 2 Capital Decline (Emergency Finish on Depletion)",
        color="#c2185b",
        alpha=0.60,
        step="post",
    )
    ax3.fill_between(
        df_p1["time"],
        df_p1["area2_cumulative_development"],
        df_p1["cumulative_mine_development"],
        label="General Sustaining Mine Development",
        color="#e65100",
        alpha=0.35,
        step="post",
    )
    ax3.step(
        df_p1["time"],
        df_p1["cumulative_mine_development"],
        label="Total Combined Development",
        color="#b71c1c",
        linewidth=2.0,
        where="post",
    )
    ax3.axhline(
        4000.0,
        color="#c2185b",
        linestyle=":",
        linewidth=1.8,
        label="Area 2 Unlock Threshold (4,000 m)",
    )
    if unlock_time_p1 is not None:
        ax3.axvline(
            unlock_time_p1,
            color="#c2185b",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Policy 1 Area 2 Unlocked (Day {unlock_time_p1:.1f})",
        )
        t_max_p1 = df_p1["time"].max()
        text_x = (
            unlock_time_p1 + (t_max_p1 * 0.03)
            if (unlock_time_p1 < t_max_p1 * 0.80)
            else unlock_time_p1 - (t_max_p1 * 0.18)
        )
        ax3.annotate(
            f"★ AREA 2 UNLOCKED\nDay {unlock_time_p1:.1f}",
            xy=(unlock_time_p1, 4000.0),
            xytext=(text_x, 15000.0),
            arrowprops=dict(
                facecolor="#c2185b",
                edgecolor="#c2185b",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffebee",
                edgecolor="#c2185b",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#c2185b",
            zorder=10,
        )
    ax3.set_title(
        "Policy 1: Stacked Underground Development (Emergency Decline Boring upon Area 1 Depletion)"
    )
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
        title="Stockpiles & Campaigns: Policy 2 (Dual-Area Supply with Analytical Blending)",
        palette=palette,
        hlines=[
            {
                "y": 60000.0,
                "color": "black",
                "linestyle": "--",
                "label": "Target Total (60k)",
            },
            {
                "y": 20400.0,
                "color": "red",
                "linestyle": ":",
                "label": "Critical Ore 2 (20.4k)",
            },
        ],
        ax=dash[4],
    )
    if unlock_time_p2 is not None:
        dash[4].axvspan(
            df_p2["time"].min(),
            unlock_time_p2,
            color="#ffebee",
            alpha=0.35,
            label="Mine 2 Locked",
        )
        dash[4].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Mine 2 Unlocked (Day {unlock_time_p2:.1f})",
        )
        t_max = df_p2["time"].max()
        text_x = (
            unlock_time_p2 + (t_max * 0.03)
            if (unlock_time_p2 < t_max * 0.80)
            else unlock_time_p2 - (t_max * 0.18)
        )
        dash[4].annotate(
            f"★ MINE 2 UNLOCKED\nDay {unlock_time_p2:.1f}",
            xy=(unlock_time_p2, 48000.0),
            xytext=(text_x, 52000.0),
            arrowprops=dict(
                facecolor="#2e7d32",
                edgecolor="#2e7d32",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#e8f5e9",
                edgecolor="#2e7d32",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#2e7d32",
            zorder=10,
        )
        dash[4].legend(loc="upper right", framealpha=0.90)

    # 5. Stockpiles: Policy 1
    plot_ore_with_modes(
        df_p1,
        time_col="time",
        ore_cols=["total_system_ore_mass", "Ore1Stock_mass", "Ore2Stock_mass"],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Stockpiles & Campaigns: Policy 1 (Severe Ore 2 Starvation & Mode B Trapping)",
        palette=palette,
        hlines=[
            {
                "y": 60000.0,
                "color": "black",
                "linestyle": "--",
                "label": "Target Total (60k)",
            },
            {
                "y": 20400.0,
                "color": "red",
                "linestyle": ":",
                "label": "Critical Ore 2 (20.4k)",
            },
        ],
        ax=dash[5],
    )
    if unlock_time_p1 is not None:
        dash[5].axvspan(
            df_p1["time"].min(),
            unlock_time_p1,
            color="#ffebee",
            alpha=0.35,
            label="Mine 2 Locked",
        )
        dash[5].axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Mine 2 Unlocked (Day {unlock_time_p1:.1f})",
        )
        t_max_p1 = df_p1["time"].max()
        text_x_p1 = (
            unlock_time_p1 + (t_max_p1 * 0.03)
            if (unlock_time_p1 < t_max_p1 * 0.80)
            else unlock_time_p1 - (t_max_p1 * 0.18)
        )
        dash[5].annotate(
            f"★ MINE 2 UNLOCKED\nDay {unlock_time_p1:.1f}",
            xy=(unlock_time_p1, 48000.0),
            xytext=(text_x_p1, 52000.0),
            arrowprops=dict(
                facecolor="#c62828",
                edgecolor="#c62828",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffebee",
                edgecolor="#c62828",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#c62828",
            zorder=10,
        )
        dash[5].legend(loc="upper right", framealpha=0.90)

    # 6. Policy 2: Analytical Operational Face Allocation Weights (Appendix A & B)
    plot_time_series(
        df_p2,
        y_columns=["analytical_face1_weight", "analytical_face2_weight"],
        title="Policy 2: Analytical Face Allocation Dispatch Weights w1 (Face 1) & w2 (Face 2) [Slide 29]",
        is_step=True,
        ax=dash[6],
    )
    if unlock_time_p2 is not None:
        dash[6].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    dash[6].set_ylabel("Dispatch Weight (Fraction)")
    dash[6].set_ylim(-0.05, 1.05)
    dash[6].legend(loc="upper right")

    # 7. Operating Modes Timeline: Policy 2
    plot_time_series(
        df_p2,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes Timeline: Policy 2 (Balanced High-Grade Campaigns)",
        is_step=True,
        ax=dash[7],
    )
    if unlock_time_p2 is not None:
        dash[7].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
            label=f"★ Policy 2 Unlocked (Day {unlock_time_p2:.1f})",
        )
        dash[7].legend(loc="upper right", framealpha=0.90)

    # 8. Operating Modes Timeline: Policy 1
    plot_time_series(
        df_p1,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes Timeline: Policy 1 (Permanently Trapped in Low-Throughput Mode B)",
        is_step=True,
        ax=dash[8],
    )
    if unlock_time_p1 is not None:
        dash[8].axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
            label=f"★ Policy 1 Unlocked (Day {unlock_time_p1:.1f})",
        )
        dash[8].legend(loc="upper right", framealpha=0.90)

    # 9. Strategic Trajectory Ratios: Policy 2
    plot_time_series(
        df_p2,
        y_columns=[
            "development_trajectory_ratio",
            "area2_readiness_trajectory_ratio",
            "ore1_trajectory_ratio",
            "ore2_trajectory_ratio",
        ],
        title="Policy 2: Strategic & Area 2 Trajectory Progress Ratios (Level 2 Tactical Reviews)",
        is_step=True,
        ax=dash[9],
    )
    if unlock_time_p2 is not None:
        dash[9].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    dash[9].axhline(
        0.90, color="red", linestyle=":", label="Tolerance Threshold (0.90)"
    )
    dash[9].axhline(
        1.00, color="gray", linestyle="--", label="Target Trajectory (1.00)"
    )
    dash[9].set_ylabel("Trajectory Ratio")
    dash[9].legend(loc="upper right")

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
    if unlock_time_p1 is not None:
        dash[11].axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )

    # 12. Mode Distribution: Policy 2
    plot_mode_distribution(
        df_p2,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% Time Spent - Policy 2 Hierarchical Value-Oriented Control)",
        palette=palette,
        ax=dash[12],
    )

    # 13. Mode Distribution: Policy 1
    plot_mode_distribution(
        df_p1,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% Time Spent - Policy 1 Local Myopic Baseline)",
        palette=palette,
        ax=dash[13],
    )

    dash.save(output_path)
    print(f"Saved full hierarchy benchmark dashboard to '{output_path}'.")
    return dash


# ---------------------------------------------------------------------------
# Execution & Benchmark Runner
# ---------------------------------------------------------------------------
def run_full_hierarchy_study(
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
    """Runs the complete comparative benchmark between Policy 1 and Policy 2 with Analytical Blending."""
    strategic_target = StrategicYearTarget(
        min_development=10000.0,
        min_ore1_production=1300000.0,
        min_ore2_production=850000.0,
    )
    area2_target = AreaReadinessTarget(
        required_development=area2_required_dev,
        ready_by_day=area2_ready_by_day,
    )

    # 1. Run Policy 2: Hierarchical Control with Analytical Blending
    print("\n" + "=" * 80)
    print(
        " 1/2 RUNNING POLICY 2: HIERARCHICAL VALUE-ORIENTED CONTROL + ANALYTICAL BLENDING (Slides 23, 29)"
    )
    print("=" * 80)
    sim_p2 = TwoAreaFullHierarchyEngine(
        policy_name="POLICY_2_VALUE_ORIENTED",
        use_analytical_blending=True,
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
        sim_p2.face1.total_ore_to_extract = total_ore_to_extract / 2.0
        sim_p2.face2.total_ore_to_extract = total_ore_to_extract / 2.0
        eng_p2.run(until=float("inf"))

        sim_p2.plant.reset_mode_timers()
        sim_p2.plant.cumulative_milled_mass.value = 0.0

        sim_p2.total_ore_to_extract = total_ore_to_extract
        sim_p2.face1.total_ore_to_extract = total_ore_to_extract / 2.0
        sim_p2.face2.total_ore_to_extract = total_ore_to_extract / 2.0
        eng_p2.run(until=float("inf"))

    df_p2 = pd.DataFrame(sim_p2.history_records)

    # 2. Run Policy 1: Local Myopic Baseline
    print("\n" + "=" * 80)
    print(
        " 2/2 RUNNING POLICY 1: LOCAL-OBJECTIVE MYOPIC BASELINE (Slide 22)"
    )
    print("=" * 80)
    sim_p1 = TwoAreaFullHierarchyEngine(
        policy_name="POLICY_1_MYOPIC",
        use_analytical_blending=False,
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
        sim_p1.face1.total_ore_to_extract = total_ore_to_extract / 2.0
        sim_p1.face2.total_ore_to_extract = total_ore_to_extract / 2.0
        eng_p1.run(until=float("inf"))

        sim_p1.plant.reset_mode_timers()
        sim_p1.plant.cumulative_milled_mass.value = 0.0

        sim_p1.total_ore_to_extract = total_ore_to_extract
        sim_p1.face1.total_ore_to_extract = total_ore_to_extract / 2.0
        sim_p1.face2.total_ore_to_extract = total_ore_to_extract / 2.0
        eng_p1.run(until=float("inf"))

    df_p1 = pd.DataFrame(sim_p1.history_records)

    # Summary Statistics
    final_p2 = df_p2.iloc[-1]
    final_p1 = df_p1.iloc[-1]
    npv_p2 = float(final_p2["operating_npv_proxy"])
    npv_p1 = float(final_p1["operating_npv_proxy"])
    value_gain = npv_p2 - npv_p1

    print("\n" + "=" * 80)
    print(
        " FULL HIERARCHY THREE-LEVEL MINING BENCHMARK RESULTS"
    )
    print("=" * 80)
    print(
        "Policy 2 (Hierarchical Value-Oriented Control + Analytical Blending):"
    )
    print(
        f"  Area 2 Unlocked:             {sim_p2.area2_ready} (Strategic Day: {float(final_p2['area2_ready_day']):.2f})"
    )
    print(
        f"  Total Ore Milled:            {float(final_p2['cumulative_milled_mass']):,.1f} t"
    )
    print(
        f"  Capital Dev (Area 2):        {float(final_p2['area2_cumulative_development']):,.1f} / {area2_required_dev:.1f} m"
    )
    print(
        f"  Sustaining Mine Dev:         {float(final_p2['sustaining_cumulative_development']):,.1f} metres"
    )
    print(
        f"  Total Combined Dev:          {float(final_p2['cumulative_mine_development']):,.1f} metres"
    )
    print(f"  Operating Net Present Value: ${npv_p2:,.2f}")

    print("\nPolicy 1 (Local-Objective Myopic Baseline):")
    print(
        f"  Area 2 Unlocked:             {sim_p1.area2_ready} (Strategic Day: {float(final_p1['area2_ready_day']):.2f})"
    )
    print(
        f"  Total Ore Milled:            {float(final_p1['cumulative_milled_mass']):,.1f} t"
    )
    print(
        f"  Capital Dev (Area 2):        {float(final_p1['area2_cumulative_development']):,.1f} / {area2_required_dev:.1f} m"
    )
    print(
        f"  Sustaining Mine Dev:         {float(final_p1['sustaining_cumulative_development']):,.1f} metres"
    )
    print(
        f"  Total Combined Dev:          {float(final_p1['cumulative_mine_development']):,.1f} metres"
    )
    print(f"  Operating Net Present Value: ${npv_p1:,.2f}")

    print("\n" + "-" * 80)
    print(
        f" >>> TOTAL INCREMENTAL VALUE FROM HIERARCHICAL CONTROL + ANALYTICAL BLENDING: ${value_gain:,.2f} <<<"
    )
    print("-" * 80)
    print("=" * 80 + "\n")

    if plot and len(df_p2) > 0 and len(df_p1) > 0:
        plot_full_hierarchy_dashboard(df_p1, df_p2)

    return df_p1, df_p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Three-Level Hierarchical Mining Simulation Benchmark"
    )
    parser.add_argument("--total_ore_to_extract", type=float, default=6600000.0)
    parser.add_argument("--warmup_ore", type=float, default=600000.0)
    parser.add_argument("--total_days", type=float, default=None)
    parser.add_argument(
        "--trucks",
        type=int,
        default=18,
        help="Number of AD30 haulage trucks (default: 18 for deep Level 6 cycle times)",
    )
    parser.add_argument(
        "--operators",
        type=int,
        default=18,
        help="Number of operators per shift (default: 18)",
    )
    parser.add_argument("--availability", type=float, default=0.85)
    parser.add_argument("--stockpile_target", type=float, default=60000.0)
    parser.add_argument("--area2_required_dev", type=float, default=4000.0)
    parser.add_argument("--area2_ready_by_day", type=float, default=365.0)
    parser.add_argument("--discount_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    run_full_hierarchy_study(
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

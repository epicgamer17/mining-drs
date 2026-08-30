"""Two-Area Strategic Planning, Area 2 Readiness, and Counterfactual Incremental NPV.

Combines:
1. Two Distinct Mining Areas / Faces with Stochastic Geological Facies:
   - Area 1 (Face 1, Level 3): Low Ore 2 grade (mean: 0.15, std: 0.05). Active from Day 0.
   - Area 2 (Face 2, Level 6): High Ore 2 grade (mean: 0.45, std: 0.05).
     * Physically locked until cumulative development reaches readiness target (e.g. 4,000 m).
     * Once unlocked, Face 2 enters active production and supplies high-grade ore.
2. Area 2 Readiness & Physical Unlock Mechanics:
   - AreaReadinessTarget (required_development: 4,000 m, ready_by_day: 365.0 days).
   - While Area 2 is locked, trucks allocated to Area 2 are redeployed to development.
   - Development advance is simulated continuously (5.0 m/truck-day).
   - Persistent deadline-miss and lateness tracking.
3. Strategic & Tactical Planning Framework:
   - StrategicYearTarget: Annual commitments (min development, min Ore 1, min Ore 2).
   - Monthly Tactical Review (every 30 days): Evaluates trajectory ratios including Area 2.
   - Dynamic MiningPriority selection (BALANCED, PRODUCTION, DEVELOPMENT).
   - Fleet Reservation: When in DEVELOPMENT priority, reserves extra haulage capacity.
4. Strategic Economics & Counterfactual Incremental NPV Evaluation:
   - DCF Accounting: Revenue per processed tonne, mining cost/t, development cost/m, fixed cost/day, discount factor.
   - Incremental NPV: Evaluates WITH-Area2 vs WITHOUT-Area2 counterfactual using identical random seeds:
     Incremental NPV = NPV(With Area 2) - NPV(Without Area 2)
5. Shelswell (2017) DES Haulage Engine & DRS Blending Modes:
   - Rate and buffer-aware dispatch throttling maintaining total stockpile at target buffer (60,000 t).
   - Disabled statutory holidays (NON_PRODUCTION_DAYS = 0 with NOTE:) for continuous supply.
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
from drs_mining.components.modes import MODES, OperatingMode
from drs_mining.components.plant import MetallurgicalPlant, PlantDrawRates
from drs_mining.components.stockpiles import Stockpile
from drs_mining.components.controllers import OperatingModeController
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.mine_face import MineFace
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

# NOTE: Statutory non-production holidays (NON_PRODUCTION_DAYS = 11 in Shelswell 2017)
# have been disabled (set to 0) in this baseline to maintain uninterrupted fleet supply
# and keep the total stockpile tightly centered around the 60,000 t target buffer.
NON_PRODUCTION_DAYS = 0

SHIFT_SECONDS = 12.0 * 3600.0  # 12 h calendar slot per shift
SHIFT_WORK_HOURS = 10.5  # 10.5 h shift duration
HAULAGE_SEAT_FRACTION = 0.5417  # 54.17 % workable seat availability
SEAT_PER_SHIFT_SEC = HAULAGE_SEAT_FRACTION * SHIFT_SECONDS  # ~6.5 h

# Mine Geometry (Two Distinct Extraction Areas)
DECLINE_M = 2100.0
LEVEL_SPACING_M = 300.0
AREA1_LEVEL = 3  # Area 1 / Face 1 at Level 3
AREA2_LEVEL = 6  # Area 2 / Face 2 at Level 6
LEVEL_DRIFT_M = 60.0
SURFACE_M = 300.0

# Speeds (Table 1, kph)
SPEEDS = {
    "surface": {"empty": 17.4, "loaded": 13.4},
    "decline": {"empty": 15.1, "loaded": 11.2},
    "ramp": {"empty": 12.9, "loaded": 9.2},
    "level": {"empty": 7.6, "loaded": 6.6},
}

# Equipment Durations
ORE_PAYLOAD = 26.1  # AD30 rated capacity (tonnes)
TRUCK_LOAD_SPOT_MIN = 0.50
LHD_ACQUISITION_MAX_MIN = 0.80
TRUCK_LOAD_DUR_MIN = 3.50
DUMP_SPOT_MIN = 0.57
DUMP_MIN = 0.88
SURFACE_TIP_SITES = 2

# Refuelling & Delays
FUEL_BURN_PCT_PER_SEC = 100.0 / (7.5 * 3600.0)
REFUEL_DUR_MIN = 25.0
N_FUEL_PUMPS = 2
BASE_PASS_BAY_DELAY_SEC = 13.0
PER_TRUCK_PASS_BAY_DELAY_SEC = 1.0

# Development Rate Calibration
DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY = 5.0
DT_MAX = 900.0  # max engine drift step (sec)


# ---------------------------------------------------------------------------
# Sampling Helpers
# ---------------------------------------------------------------------------
def _tri(rng: random.Random, mid: float, tol: float) -> float:
    """Symmetric triangular distribution around ``mid`` with width ``tol``."""
    return rng.triangular(mid * (1.0 - tol), mid * (1.0 + tol), mid)


def _in_shift_window(t: float) -> bool:
    """Two 10.5 h shifts per day separated by two 1.5 h off-shift gaps."""
    hod = t % 86400.0
    return (0.0 <= hod < SHIFT_WORK_HOURS * 3600.0) or (
        12.0 * 3600.0 <= hod < 22.5 * 3600.0
    )


# ---------------------------------------------------------------------------
# Discrete State Entities
# ---------------------------------------------------------------------------
class TruckPhase(Enum):
    IDLE = "idle"
    EMPTY = "empty"
    WAIT_LOAD = "wait_load"
    SPOT_LOAD = "spot_load"
    ACQUIRE = "acquire"
    LOADING = "loading"
    LOADED = "loaded"
    WAIT_DUMP = "wait_dump"
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
    TruckPhase.DUMPING,
}

SEAT_PHASES = OPERATING_PHASES | {TruckPhase.REFUELING}

DUE_PHASES = {
    TruckPhase.EMPTY,
    TruckPhase.SPOT_LOAD,
    TruckPhase.ACQUIRE,
    TruckPhase.LOADING,
    TruckPhase.LOADED,
    TruckPhase.DUMPING,
    TruckPhase.REFUELING,
}


@dataclass
class Operator:
    idx: int
    free: bool = True
    used_seat: float = 0.0


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
# Two-Area Strategic Planning & Economic Simulation Module
# ---------------------------------------------------------------------------
class TwoAreaEconomicSimulation(drs.Module):
    """Two-Area Strategic DES Simulation with Area 2 Readiness & Discounted Cash Flow Economics."""

    def __init__(
        self,
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
        area2_physical_unlock_enabled: bool = True,
        area2_counterfactual_disable: bool = False,
        area2_redeploy_locked_face_trucks_to_development: bool = True,
        development_priority_truck_reservation_fraction: float = 0.20,
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
        self.area2_physical_unlock_enabled = area2_physical_unlock_enabled
        self.area2_counterfactual_disable = area2_counterfactual_disable
        self.area2_redeploy_locked_face_trucks_to_development = (
            area2_redeploy_locked_face_trucks_to_development
        )
        self.development_priority_truck_reservation_fraction = (
            development_priority_truck_reservation_fraction
        )

        # Economic Assumptions
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
        self.holidays = set()
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
        self.face1 = MineFace(
            name="mine_face_1",
            face_id=1,
            generator=self.gen1,
            min_ore_mass=30000.0,
            max_ore_mass=50000.0,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
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
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
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
            initial_attributes={
                "contained_ore_fraction_mass": init_ore1 * 0.30
            },
            attr_inflow=1.0,
        )
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=init_ore2,
            initial_attributes={
                "contained_ore_fraction_mass": init_ore2 * 0.30
            },
            attr_inflow=0.0,
        )
        self.total_extracted_ore = drs.Level("total_extracted_ore", 0.0)
        self.ore1_hauled = drs.Level("ore1_hauled", 0.0)
        self.ore2_hauled = drs.Level("ore2_hauled", 0.0)
        self.cumulative_mine_development = drs.Level("cumulative_mine_development", 0.0)
        self.area2_cumulative_development = drs.Level("area2_cumulative_development", 0.0)

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
        self.mining_priority = MiningPriority.BALANCED

        # 7. Area 2 Readiness Variables
        self.area2_ready = False
        self.area2_ready_day = drs.Level("area2_ready_day", -1.0)
        self.area2_deadline_missed = False
        self.area2_currently_late = False
        self.area2_completed_late = False
        self.area2_readiness_fraction = drs.Level("area2_readiness_fraction", 0.0)
        self.area2_readiness_trajectory_ratio = drs.Level("area2_readiness_trajectory_ratio", 1.0)

        # 8. Strategic Economics State Variables
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

        # Operational metrics
        self.daily_target_ore = 6000.0
        self.daily_hauled_ore = 0.0
        self.trips = 0
        self._cycle_sum = 0.0
        self.traffic_delay_sum = 0.0
        self.horizon_sec = float("inf")

        # Telemetry records
        self.history_records: List[dict] = []

    # -- Area 2 Lock / Unlock Logic ------------------------------------------
    def is_area2_locked(self) -> bool:
        """Returns True if Area 2 is physically locked due to development readiness."""
        if self.area2_counterfactual_disable:
            return True
        if not self.area2_physical_unlock_enabled:
            return False
        required = max(0.0, float(self.area2_readiness_target.required_development))
        if required <= 1e-12:
            return False
        return not (self.strategic_planning_started and self.area2_ready)

    def _update_area2_readiness(self):
        """Updates Area 2 readiness metrics, unlocks face if threshold met, and tracks deadlines."""
        if self.area2_counterfactual_disable:
            self.area2_ready = False
            self.area2_readiness_fraction.value = 0.0
            self.area2_readiness_trajectory_ratio.value = 1.0
            return

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
            print(f"\n >>> [PHYSICAL UNLOCK] Area 2 (Face 2) is READY and UNLOCKED on Strategic Day {strategic_days:.2f}! <<<\n")

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
    def _update_strategic_economics(self, out1_t_sec: float, out2_t_sec: float, dt_days: float):
        """Calculates revenue, operating costs, discounted cash flow, and operating NPV."""
        if not self.strategic_planning_started:
            self.discount_factor.value = 1.0
            self.current_cash_flow_rate.value = 0.0
            self.current_discounted_cash_flow_rate.value = 0.0
            return

        # Daily Milling Outflows (t/day)
        milled_ore1_t_day = out1_t_sec * 86400.0
        milled_ore2_t_day = out2_t_sec * 86400.0
        self.cumulative_processed_ore1.value += milled_ore1_t_day * dt_days
        self.cumulative_processed_ore2.value += milled_ore2_t_day * dt_days

        # Revenue Rate ($/day)
        revenue_rate = (
            milled_ore1_t_day * self.ore1_net_value_per_processed_tonne
            + milled_ore2_t_day * self.ore2_net_value_per_processed_tonne
        )

        # Mining & Haulage Production Cost Rate ($/day)
        mined_ore_t_day = max(0.0, self.daily_hauled_ore)
        production_cost_rate = mined_ore_t_day * self.production_cost_per_tonne

        # Underground Development Cost Rate ($/day)
        # Note: in WITHOUT_AREA2 counterfactual, Area 2 development is not incurred
        dev_rate_m_day = max(0.0, float(self.development_rate_m_per_day.value))
        development_cost_rate = dev_rate_m_day * self.development_cost_per_unit

        # Fixed Overhead Cost Rate ($/day)
        fixed_cost_rate = self.fixed_cost_per_day

        # Net Daily Cash Flow Rate ($/day)
        cash_flow_rate = revenue_rate - (
            production_cost_rate + development_cost_rate + fixed_cost_rate
        )
        self.current_cash_flow_rate.value = cash_flow_rate
        self.cumulative_cash_flow.value += cash_flow_rate * dt_days

        # Discount Factor
        strategic_days = (
            float(self.strategic_year_index.value) * self.strategic_period_days
            + float(self.strategic_year_timer.value)
        )
        dfactor = 1.0 / ((1.0 + self.annual_discount_rate) ** (strategic_days / 365.0))
        self.discount_factor.value = dfactor

        # Discounted Cash Flow ($)
        discounted_rate = cash_flow_rate * dfactor
        self.current_discounted_cash_flow_rate.value = discounted_rate
        self.cumulative_discounted_cash_flow.value += discounted_rate * dt_days
        self.operating_npv_proxy.value = self.cumulative_discounted_cash_flow.value

    # -- DRS Engine Hooks ----------------------------------------------------
    def is_terminating_condition_met(self) -> bool:
        total_extracted = sum(f.cumulative_extracted_mass.value for f in self.faces)
        if total_extracted >= self.total_ore_to_extract - 1e-6:
            return True
        return self.gt.value >= self.horizon_sec - 1e-6

    def time_to_event(self) -> float:
        """Find the earliest discrete event time boundary."""
        best = DT_MAX
        for tr in self.trucks:
            v = tr.timer.value
            if v > 1e-9:
                best = min(best, v)
        t = self.gt.value
        next_day = (math.floor(t / 86400.0) + 1.0) * 86400.0
        next_shift = (math.floor(t / SHIFT_SECONDS) + 1.0) * SHIFT_SECONDS
        best = min(best, next_day - t, next_shift - t)

        # Campaign timer boundary
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

        # Tactical review timer boundary
        rem_tactical_days = max(
            0.0, self.tactical_review_period_days - self.tactical_review_timer.value
        )
        if rem_tactical_days > 1e-6:
            best = min(best, rem_tactical_days * 86400.0)

        # Contingency timer boundary
        if "_CONTINGENCY" in self.plant.active_operating_mode.value.name:
            c_thresh = self.plant.duration_of_contingency_segments
            rem_c_days = max(
                0.0, c_thresh - self.plant.current_contingency_duration.value
            )
            if rem_c_days > 1e-6:
                best = min(best, rem_c_days * 86400.0)

        return max(best, 1e-6)

    def step(self, dt: float):
        """Continuous integration between discrete event boundaries."""
        self.gt.step(dt)
        dt_days = dt / 86400.0

        # Step timers
        self.mode_controller.current_campaign_duration.step(dt_days)
        active_mode_name = self.plant.active_operating_mode.value.name
        timer_attr = self.plant._MODE_TIMER_ATTRS.get(active_mode_name)
        if timer_attr:
            getattr(self.plant, timer_attr).step(dt_days)
        if active_mode_name in self.plant._CONTINGENCY_MODES:
            self.plant.current_contingency_duration.step(dt_days)

        # Step strategic & tactical timers
        if self.strategic_planning_started:
            self.strategic_year_timer.step(dt_days)
            self.tactical_review_timer.step(dt_days)

        # Step trucks
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

        # Inflow rates from surface dumping
        ore1_in_rate = self.dump_station._active_ore1_rate
        ore2_in_rate = self.dump_station._active_ore2_rate

        # Blended routing fraction across active available faces
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

        ore1_draw_rate_sec = plant_draw.ore1 / 86400.0
        ore2_draw_rate_sec = plant_draw.ore2 / 86400.0

        out1 = self.ore1_stock.feed_and_draw(ore1_in_rate, ore1_draw_rate_sec)
        out2 = self.ore2_stock.feed_and_draw(ore2_in_rate, ore2_draw_rate_sec)
        self.ore1_stock.step(dt)
        self.ore2_stock.step(dt)

        self.plant.process(out1 + out2)
        self.plant.cumulative_milled_mass.step(dt)

        # Development advance step
        reserved_trucks = float(self.development_priority_reserved_trucks.value)
        n_operating_trucks = sum(
            1 for tr in self.trucks if tr.phase in OPERATING_PHASES
        )
        total_trucks = len(self.trucks)
        available_extra = max(0, total_trucks - n_operating_trucks)

        locked_boost = (
            (total_trucks * 0.35)
            if (
                self.is_area2_locked()
                and self.area2_redeploy_locked_face_trucks_to_development
                and not self.area2_counterfactual_disable
            )
            else 0.0
        )
        dev_trucks = max(reserved_trucks, float(available_extra)) + locked_boost
        self.development_rate_m_per_day.value = (
            dev_trucks * DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY
        )
        delta_dev = self.development_rate_m_per_day.value * dt_days
        self.cumulative_mine_development.value += delta_dev

        # Allocate development metres to Area 2 project
        if self.is_area2_locked() and self.strategic_planning_started and not self.area2_counterfactual_disable:
            prio = self.mining_priority
            if prio == MiningPriority.DEVELOPMENT:
                frac = 0.85
            elif prio == MiningPriority.BALANCED:
                frac = 0.60
            else:
                frac = 0.35
            self.area2_cumulative_development.value += delta_dev * frac

        self._update_area2_readiness()
        self._update_strategic_economics(out1, out2, dt_days)

        # Accumulators
        self.total_extracted_ore.step(dt)
        self.ore1_hauled.step(dt)
        self.ore2_hauled.step(dt)

        self._record_telemetry(plant_draw)

    # -- Event Policy & Target Setting ----------------------------------------
    def on_event(self, t: float):
        """Engine step policy: calendar updates, strategic review, truck transitions."""
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
        """Conducts strategic annual rollovers and monthly tactical progress reviews."""
        t_days = self.gt.value / 86400.0

        total_extracted = sum(f.cumulative_extracted_mass.value for f in self.faces)
        if total_extracted >= self.ore_to_be_extracted_during_warming_period or self.horizon_sec < float("inf"):
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

        # Annual Rollover Check
        if self.strategic_year_timer.value >= self.strategic_period_days - 1e-6:
            self.strategic_year_index.value += 1.0
            self.strategic_year_timer.reset()
            self.annual_ore1_extracted = 0.0
            self.annual_ore2_extracted = 0.0
            self.annual_development_start = float(
                self.cumulative_mine_development.value
            )

        # Compute Annual Trajectory Progress Ratios
        elapsed_year_fraction = max(
            1e-4, min(1.0, self.strategic_year_timer.value / self.strategic_period_days)
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

        # Monthly Tactical Review (every 30 days)
        if (
            self.tactical_review_timer.value
            >= self.tactical_review_period_days - 1e-6
            or self.tactical_review_count.value == 0.0
        ):
            self.tactical_review_timer.reset()
            self.tactical_review_count.value += 1.0

            selected = select_mining_priority(
                development_ratio=float(self.development_trajectory_ratio.value),
                ore1_ratio=float(self.ore1_trajectory_ratio.value),
                ore2_ratio=float(self.ore2_trajectory_ratio.value),
                tolerance=self.tactical_progress_tolerance,
                area2_readiness_trajectory_ratio=float(self.area2_readiness_trajectory_ratio.value),
            )
            self.mining_priority = selected

            if selected == MiningPriority.DEVELOPMENT:
                reserved = math.ceil(
                    len(self.trucks)
                    * self.development_priority_truck_reservation_fraction
                )
                self.development_priority_reserved_trucks.value = float(reserved)
            else:
                self.development_priority_reserved_trucks.value = 0.0

    def _update_operating_mode_and_targets(self):
        """Updates campaign mode, plant operating mode, and extraction targets."""
        camp_mode = self.mode_controller.update(
            ore2_stock_level=self.ore2_stock.level,
            total_stock_level=self.ore1_stock.level + self.ore2_stock.level,
        )

        if self.is_area2_locked():
            f_blend = self.face1.active_parcel_ore_fraction.value
        else:
            f_blend = (
                self.face1.active_parcel_ore_fraction.value
                + self.face2.active_parcel_ore_fraction.value
            ) / 2.0

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

    # -- Dispatch Policy -----------------------------------------------------
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
        if total_stock >= self.target_ore_stock_level and self.daily_hauled_ore > expected_hauled_by_now + 100.0:
            self._release_operator(tr)
            return False

        if not self._acquire_operator(tr):
            self._release_operator(tr)
            return False

        target_face_id = self._select_face_by_blend_need()
        tr.target_face_id = target_face_id
        tr.target_level = AREA1_LEVEL if target_face_id == 1 else AREA2_LEVEL

        tr.trip_start = self.gt.value
        tr.phase = TruckPhase.EMPTY
        tr.timer.value = self._travel_time(tr, loaded=False)
        return True

    def _select_face_by_blend_need(self) -> int:
        if self.is_area2_locked():
            return 1

        mode_name = self.plant.active_operating_mode.value.name
        if "MODE_A" in mode_name:
            p_face2 = 0.65
        elif "MODE_B" in mode_name:
            p_face2 = 0.35
        else:
            p_face2 = 0.50
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
            delay = BASE_PASS_BAY_DELAY_SEC + PER_TRUCK_PASS_BAY_DELAY_SEC * max(
                0, cong - 3
            )
            t += delay
            self.traffic_delay_sum += delay
        return t

    # -- Telemetry & History -------------------------------------------------
    def _record_telemetry(self, plant_draw: PlantDrawRates):
        t_days = self.gt.value / 86400.0
        active_mode = self.plant.active_operating_mode.value.name
        camp_mode = self.mode_controller.active_campaign_mode.value.name
        mining_prio = self.mining_priority.name

        n_waiting_load = sum(len(q) for q in self.face_queues.values())
        n_waiting_dump = len(self.dump_station.queue)
        n_refueling = sum(
            1 for tr in self.trucks if tr.phase == TruckPhase.REFUELING
        )
        n_operating = sum(
            1 for tr in self.trucks if tr.phase in OPERATING_PHASES
        )

        total_extracted = sum(f.cumulative_extracted_mass.value for f in self.faces)
        target = strategic_target_for_year(
            self.strategic_targets, int(self.strategic_year_index.value)
        )

        if self.is_area2_locked():
            f_blend = self.face1.active_parcel_ore_fraction.value
        else:
            f_blend = (
                self.face1.active_parcel_ore_fraction.value
                + self.face2.active_parcel_ore_fraction.value
            ) / 2.0

        n_idle = max(0, len(self.trucks) - (n_operating + n_refueling))

        self.history_records.append(
            {
                "time": t_days,
                "ore1_stock": self.ore1_stock.level,
                "ore2_stock": self.ore2_stock.level,
                "Ore1Stock_mass": self.ore1_stock.level,
                "Ore2Stock_mass": self.ore2_stock.level,
                "total_system_ore_mass": self.ore1_stock.level
                + self.ore2_stock.level,
                "active_operating_mode": self.plant.active_operating_mode.value,
                "active_operating_mode_name": active_mode,
                "campaign_mode": camp_mode,
                "trucks_operating": n_operating,
                "trucks_refueling": n_refueling,
                "trucks_idle": n_idle,
                "truck_idle_fraction": n_idle / max(1, len(self.trucks)),
                "current_campaign_duration": self.mode_controller.current_campaign_duration.value,
                "current_contingency_duration": self.plant.current_contingency_duration.value,
                "strategic_year_index": self.strategic_year_index.value,
                "tactical_review_count": self.tactical_review_count.value,
                "mining_priority": mining_prio,
                "annual_target_development": target.min_development,
                "annual_target_ore1": target.min_ore1_production,
                "annual_target_ore2": target.min_ore2_production,
                "development_trajectory_ratio": self.development_trajectory_ratio.value,
                "ore1_trajectory_ratio": self.ore1_trajectory_ratio.value,
                "ore2_trajectory_ratio": self.ore2_trajectory_ratio.value,
                "area2_required_development": self.area2_readiness_target.required_development,
                "area2_ready_by_day": self.area2_readiness_target.ready_by_day or 0.0,
                "area2_development_progress": self.area2_cumulative_development.value,
                "area2_readiness_fraction": self.area2_readiness_fraction.value,
                "area2_readiness_trajectory_ratio": self.area2_readiness_trajectory_ratio.value,
                "area2_ready": self.area2_ready,
                "area2_deadline_missed": self.area2_deadline_missed,
                "area2_currently_late": self.area2_currently_late,
                "area2_completed_late": self.area2_completed_late,
                "area2_ready_day": self.area2_ready_day.value,
                "cumulative_mine_development": self.cumulative_mine_development.value,
                "area2_cumulative_development": self.area2_cumulative_development.value,
                "development_rate_m_per_day": self.development_rate_m_per_day.value,
                "development_priority_reserved_trucks": self.development_priority_reserved_trucks.value,
                "daily_target_ore": self.daily_target_ore,
                "daily_hauled_ore": self.daily_hauled_ore,
                "ore1_hauled_total": self.ore1_hauled.value,
                "ore2_hauled_total": self.ore2_hauled.value,
                "cumulative_extracted_mass": total_extracted,
                "milled_ore1_rate": plant_draw.ore1,
                "milled_ore2_rate": plant_draw.ore2,
                "cumulative_milled_mass": self.plant.cumulative_milled_mass.value,
                "discount_factor": self.discount_factor.value,
                "current_cash_flow_rate": self.current_cash_flow_rate.value,
                "current_discounted_cash_flow_rate": self.current_discounted_cash_flow_rate.value,
                "cumulative_cash_flow": self.cumulative_cash_flow.value,
                "cumulative_discounted_cash_flow": self.cumulative_discounted_cash_flow.value,
                "operating_npv_proxy": self.operating_npv_proxy.value,
                "face1_active_fraction": self.face1.active_parcel_ore_fraction.value,
                "face2_active_fraction": self.face2.active_parcel_ore_fraction.value,
                "blended_active_fraction": f_blend,
                "MassOfCurrentParcel": self.face1.active_parcel_initial_mass.value,
                "Grade (% Ore 2)": f_blend * 100.0,
                "active_trucks": n_operating,
                "trucks_waiting_load": n_waiting_load,
                "trucks_waiting_dump": n_waiting_dump,
                "trucks_refueling": n_refueling,
                "traffic_delay_min": self.traffic_delay_sum / 60.0,
            }
        )


# ---------------------------------------------------------------------------
# Statistics & Dashboards
# ---------------------------------------------------------------------------
def print_statistics(plant, sim: TwoAreaEconomicSimulation):
    """Print operating-mode time-shares and throughput matching blending_modes format."""
    print("\n--- Output Statistics ---")
    total_time = plant.total_duration

    if total_time > 0:
        for attr, label in [
            ("cumulative_time_mode_a", "PortionOfTimeInModeA"),
            ("cumulative_time_mode_a_contingency", "PortionOfTimeInModeAContingency"),
            ("cumulative_time_mode_a_surging", "PortionOfTimeInModeAMineSurging"),
            ("cumulative_time_mode_b", "PortionOfTimeInModeB"),
            ("cumulative_time_mode_b_contingency", "PortionOfTimeInModeBContingency"),
            ("cumulative_time_mode_b_surging", "PortionOfTimeInModeBMineSurging"),
            ("cumulative_time_shutdown", "PortionOfTimeInShutdown"),
        ]:
            print(f"{label}: {getattr(plant, attr).value / total_time:.4f}")
    else:
        print("Total time is 0. Cannot calculate mode portions.")

    active_time = plant.active_duration(total_time)
    if active_time > 0:
        total_ore_processed = plant.cumulative_milled_mass.value
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


def print_strategic_economic_summary(
    with_df: pd.DataFrame, without_df: Optional[pd.DataFrame] = None
):
    """Prints comprehensive whole-mine economic results and incremental NPV."""
    final_with = with_df.iloc[-1]
    npv_with = float(final_with.get("operating_npv_proxy", 0.0))
    cf_with = float(final_with.get("cumulative_cash_flow", 0.0))
    milled_with = float(final_with.get("cumulative_milled_mass", 0.0))
    dev_with = float(final_with.get("cumulative_mine_development", 0.0))

    print("\n" + "=" * 80)
    print(" STRATEGIC ECONOMICS & INCREMENTAL NPV SUMMARY")
    print("=" * 80)
    print(f"WITH Area 2 (Base Case):")
    print(f"  Total Ore Milled:            {milled_with:,.1f} t")
    print(f"  Total Mine Development:      {dev_with:,.1f} metres")
    print(f"  Cumulative Undiscounted CF:  ${cf_with:,.2f}")
    print(f"  Operating Net Present Value: ${npv_with:,.2f}")

    if without_df is not None and not without_df.empty:
        final_without = without_df.iloc[-1]
        npv_without = float(final_without.get("operating_npv_proxy", 0.0))
        cf_without = float(final_without.get("cumulative_cash_flow", 0.0))
        milled_without = float(final_without.get("cumulative_milled_mass", 0.0))
        dev_without = float(final_without.get("cumulative_mine_development", 0.0))
        incremental_npv = npv_with - npv_without

        print(f"\nWITHOUT Area 2 (Counterfactual):")
        print(f"  Total Ore Milled:            {milled_without:,.1f} t")
        print(f"  Total Mine Development:      {dev_without:,.1f} metres")
        print(f"  Cumulative Undiscounted CF:  ${cf_without:,.2f}")
        print(f"  Operating Net Present Value: ${npv_without:,.2f}")

        print(f"\n--------------------------------------------------------------------------------")
        print(f" >>> TRUE INCREMENTAL NPV OF AREA 2 CAPITAL PROJECT: ${incremental_npv:,.2f} <<<")
        print(f"--------------------------------------------------------------------------------")
    print("=" * 80 + "\n")


def plot_two_area_economic_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/two_area_economic_dashboard.png",
    palette: dict = None,
    figsize: Tuple[int, int] = (16, 56),
):
    """Builds and saves the 14-panel comprehensive economic & operational diagnostics dashboard."""
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if "active_operating_mode_name" not in df.columns or "Mode A" not in df.columns:
        df = prepare_history(df)

    dash = Dashboard(
        nrows=14,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title="Two-Area Strategic Planning, Area 2 Readiness & Discounted Cash Flow Economics",
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13])

    # Detect unlock
    unlock_rows = df[df["area2_ready"] == True]
    unlock_time = float(unlock_rows["time"].iloc[0]) if not unlock_rows.empty else None

    # 0. Operating Modes Step Timeline
    plot_time_series(
        df,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes (Step Timeline)",
        is_step=True,
        ax=dash[0],
    )
    if unlock_time is not None:
        dash[0].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
            label=f"★ Mine 2 Unlocked (Day {unlock_time:.1f})",
        )
        dash[0].legend(loc="upper right", framealpha=0.90)

    # 1. Stockpiles & Operating Modes
    plot_ore_with_modes(
        df,
        time_col="time",
        ore_cols=["total_system_ore_mass", "Ore1Stock_mass", "Ore2Stock_mass"],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Ore Stockpiles & Operating Campaigns",
        palette=palette,
        hlines=[
            {
                "y": 60000.0,
                "color": "black",
                "linestyle": "--",
                "linewidth": 1.5,
                "alpha": 0.7,
                "label": "Target Total (60k)",
            },
            {
                "y": 20400.0,
                "color": "red",
                "linestyle": ":",
                "linewidth": 2.0,
                "alpha": 0.8,
                "label": "Critical Ore 2 (20.4k)",
            },
        ],
        ax=dash[1],
    )

    if unlock_time is not None:
        dash[1].axvspan(
            df["time"].min(),
            unlock_time,
            color="#ffebee",
            alpha=0.35,
            label="Mine 2 Locked (Dev Phase)",
        )
        dash[1].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Mine 2 Unlocked (Day {unlock_time:.1f})",
        )
        t_max = df["time"].max()
        text_x = unlock_time + (t_max * 0.03) if (unlock_time < t_max * 0.80) else unlock_time - (t_max * 0.18)
        dash[1].annotate(
            f"★ MINE 2 UNLOCKED\nDay {unlock_time:.1f}",
            xy=(unlock_time, 48000.0),
            xytext=(text_x, 52000.0),
            arrowprops=dict(facecolor="#880e4f", edgecolor="#880e4f", shrink=0.08, width=2.0, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fce4ec", edgecolor="#880e4f", linewidth=1.8, alpha=0.95),
            fontsize=10,
            fontweight="bold",
            color="#880e4f",
            zorder=10,
        )
        dash[1].legend(loc="upper right", framealpha=0.90)

    # 2. Cumulative Discounted Cash Flow (Operating NPV)
    plot_dual_axis_step(
        df,
        y1_col="operating_npv_proxy",
        y2_col="current_discounted_cash_flow_rate",
        y1_label="Cumulative Discounted CF ($)",
        y2_label="Discounted CF Rate ($/day)",
        title="Whole-Mine Cumulative Operating NPV & Discounted Cash Flow Rates",
        ax=dash[2],
    )
    if unlock_time is not None:
        dash[2].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )

    # 3. Area 2 Readiness Progress & Development
    plot_dual_axis_step(
        df,
        y1_col="area2_cumulative_development",
        y2_col="area2_readiness_fraction",
        y1_label="Area 2 Dev (m)",
        y2_label="Readiness Fraction (1.0 = Ready)",
        title="Area 2 Development Progress & Readiness Fraction",
        ax=dash[3],
    )
    if unlock_time is not None:
        dash[3].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )

    # 4. Strategic Trajectory Progress Ratios
    plot_time_series(
        df,
        y_columns=[
            "development_trajectory_ratio",
            "area2_readiness_trajectory_ratio",
            "ore1_trajectory_ratio",
            "ore2_trajectory_ratio",
        ],
        title="Strategic Trajectory Progress Ratios",
        is_step=True,
        ax=dash[4],
    )
    if unlock_time is not None:
        dash[4].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    dash[4].axhline(0.90, color="red", linestyle=":", label="Tolerance Threshold (0.90)")
    dash[4].axhline(1.00, color="gray", linestyle="--", label="Target Trajectory (1.00)")
    dash[4].set_ylabel("Trajectory Ratio")
    dash[4].legend(loc="upper right")

    # 5. Total Development & Truck Reservations
    plot_dual_axis_step(
        df,
        y1_col="cumulative_mine_development",
        y2_col="development_priority_reserved_trucks",
        y1_label="Total Dev (m)",
        y2_label="Reserved Dev Trucks",
        title="Total Underground Development & Tactical Truck Reservations",
        ax=dash[5],
    )
    if unlock_time is not None:
        dash[5].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )

    # 6. Safety Margin: Ore 1
    plot_safety_margin(
        df,
        level_col="Ore1Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 1 Distance to Starvation Floor",
        danger_threshold=5000.0,
        ax=dash[6],
    )

    # 7. Safety Margin: Ore 2
    plot_safety_margin(
        df,
        level_col="Ore2Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 2 Distance to Starvation Floor",
        danger_threshold=3000.0,
        ax=dash[7],
    )

    # 8. Fleet Activity & Queuing
    plot_time_series(
        df,
        y_columns=[
            "active_trucks",
            "trucks_waiting_load",
            "trucks_waiting_dump",
            "trucks_refueling",
        ],
        title="Underground Haulage Fleet Activity & Queues (AD30 Trucks)",
        is_step=True,
        ax=dash[8],
    )

    # 9. Mode Distribution
    plot_mode_distribution(
        df,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% of Time Spent)",
        palette=palette,
        ax=dash[9],
    )

    # 10. Mode Dwell Times
    plot_mode_dwell_times(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        title="Mode Stability & Campaign Dwell Times",
        ax=dash[10],
    )

    # 11. Cumulative Deficit
    plot_attributed_deficit(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        extraction_col="cumulative_extracted_mass",
        ideal_rate_per_day=6000.0,
        title="Cumulative Production Deficit by Operating Mode",
        palette=palette,
        ax=dash[11],
    )

    # 12. Deficit Bar
    plot_deficit_breakdown_bar(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[12],
    )

    # 13. Fleet Utilization & Idle Time Breakdown
    plot_truck_idle_and_utilization(
        df,
        title="Haul Fleet Utilization & Idle Time Breakdown",
        ax=dash[13],
    )

    dash.save(output_path)
    print(f"Saved dashboard visualization to '{output_path}'.")
    return dash


# ---------------------------------------------------------------------------
# Counterfactual Incremental NPV Runner
# ---------------------------------------------------------------------------
def run_two_area_economic_simulation(
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    availability: float = 0.85,
    target_ore_stock_level: float = 60000.0,
    area2_required_development: float = 4000.0,
    area2_ready_by_day: float = 365.0,
    annual_discount_rate: float = 0.05,
    ore1_net_value: float = 577.48,
    ore2_net_value: float = 709.83,
    production_cost: float = 135.0,
    development_cost: float = 15000.0,
    fixed_cost: float = 74460.0,
    seed: int = 42,
    run_counterfactual: bool = True,
    plot: bool = True,
) -> Tuple[TwoAreaEconomicSimulation, pd.DataFrame]:
    """Runs the base simulation (WITH Area 2) and optionally the counterfactual (WITHOUT Area 2)."""
    strategic_target = StrategicYearTarget(
        min_development=10000.0,
        min_ore1_production=1300000.0,
        min_ore2_production=850000.0,
    )
    area2_target = AreaReadinessTarget(
        required_development=area2_required_development,
        ready_by_day=area2_ready_by_day,
    )

    # 1. Base Case: WITH Area 2
    print("\n" + "=" * 70)
    print(" RUNNING BASE CASE: WITH AREA 2 CAPITAL EXPANSION")
    print("=" * 70)
    sim_with = TwoAreaEconomicSimulation(
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        strategic_targets=(strategic_target,),
        area2_readiness_target=area2_target,
        area2_counterfactual_disable=False,
        annual_discount_rate=annual_discount_rate,
        ore1_net_value_per_processed_tonne=ore1_net_value,
        ore2_net_value_per_processed_tonne=ore2_net_value,
        production_cost_per_tonne=production_cost,
        development_cost_per_unit=development_cost,
        fixed_cost_per_day=fixed_cost,
        seed=seed,
    )

    engine_with = drs.DRSEngine(max_step_size=DT_MAX)
    engine_with.register(sim_with)
    engine_with.on_step(sim_with.on_event)

    if total_days is not None:
        sim_with.horizon_sec = total_days * 86400.0
        engine_with.run(until=sim_with.horizon_sec)
    else:
        sim_with.total_ore_to_extract = ore_to_be_extracted_during_warming_period
        sim_with.face1.total_ore_to_extract = ore_to_be_extracted_during_warming_period
        sim_with.face2.total_ore_to_extract = ore_to_be_extracted_during_warming_period
        engine_with.run(until=float("inf"))

        sim_with.plant.reset_mode_timers()
        sim_with.plant.cumulative_milled_mass.value = 0.0

        sim_with.total_ore_to_extract = total_ore_to_extract
        sim_with.face1.total_ore_to_extract = total_ore_to_extract
        sim_with.face2.total_ore_to_extract = total_ore_to_extract
        engine_with.run(until=float("inf"))

    df_with = pd.DataFrame(sim_with.history_records)
    print_statistics(sim_with.plant, sim_with)

    # 2. Counterfactual Case: WITHOUT Area 2
    df_without = None
    if run_counterfactual:
        print("\n" + "=" * 70)
        print(" RUNNING COUNTERFACTUAL CASE: WITHOUT AREA 2 (PERMANENTLY LOCKED)")
        print("=" * 70)
        sim_without = TwoAreaEconomicSimulation(
            num_trucks=num_trucks,
            num_operators=num_operators,
            availability=availability,
            target_ore_stock_level=target_ore_stock_level,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            strategic_targets=(strategic_target,),
            area2_readiness_target=area2_target,
            area2_counterfactual_disable=True,  # Disables Area 2 and project costs
            annual_discount_rate=annual_discount_rate,
            ore1_net_value_per_processed_tonne=ore1_net_value,
            ore2_net_value_per_processed_tonne=ore2_net_value,
            production_cost_per_tonne=production_cost,
            development_cost_per_unit=development_cost,
            fixed_cost_per_day=fixed_cost,
            seed=seed,
        )

        engine_without = drs.DRSEngine(max_step_size=DT_MAX)
        engine_without.register(sim_without)
        engine_without.on_step(sim_without.on_event)

        if total_days is not None:
            sim_without.horizon_sec = total_days * 86400.0
            engine_without.run(until=sim_without.horizon_sec)
        else:
            sim_without.total_ore_to_extract = ore_to_be_extracted_during_warming_period
            sim_without.face1.total_ore_to_extract = ore_to_be_extracted_during_warming_period
            sim_without.face2.total_ore_to_extract = ore_to_be_extracted_during_warming_period
            engine_without.run(until=float("inf"))

            sim_without.plant.reset_mode_timers()
            sim_without.plant.cumulative_milled_mass.value = 0.0

            sim_without.total_ore_to_extract = total_ore_to_extract
            sim_without.face1.total_ore_to_extract = total_ore_to_extract
            sim_without.face2.total_ore_to_extract = total_ore_to_extract
            engine_without.run(until=float("inf"))

        df_without = pd.DataFrame(sim_without.history_records)

    print_strategic_economic_summary(df_with, df_without)

    df_prepared = prepare_history(df_with)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim_with.critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        label="Two-Area Economic Blending",
    )

    if plot and len(df_prepared) > 0:
        plot_two_area_economic_dashboard(df_prepared)

    return sim_with, df_prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Area Strategic Planning & Counterfactual Incremental NPV Simulation"
    )
    parser.add_argument(
        "--total_ore_to_extract",
        type=float,
        default=6600000.0,
        help="Total production ore tonnage to extract (default: 6,600,000.0 t)",
    )
    parser.add_argument(
        "--warmup_ore",
        type=float,
        default=600000.0,
        help="Warmup period ore tonnage to extract (default: 600,000.0 t)",
    )
    parser.add_argument(
        "--total_days",
        type=float,
        default=None,
        help="Total simulation duration in days (optional)",
    )
    parser.add_argument(
        "--trucks",
        type=int,
        default=18,
        help="Number of AD30 haulage trucks (default: 18)",
    )
    parser.add_argument(
        "--operators",
        type=int,
        default=18,
        help="Number of operators per shift (default: 18)",
    )
    parser.add_argument(
        "--availability",
        type=float,
        default=0.85,
        help="Mechanical availability fraction (default: 0.85)",
    )
    parser.add_argument(
        "--stockpile_target",
        type=float,
        default=60000.0,
        help="Target total ore stockpile buffer (default: 60000.0 t)",
    )
    parser.add_argument(
        "--area2_required_dev",
        type=float,
        default=4000.0,
        help="Required development metres to unlock Area 2 (default: 4,000.0 m)",
    )
    parser.add_argument(
        "--area2_ready_by_day",
        type=float,
        default=365.0,
        help="Target schedule deadline for Area 2 (default: 365.0 d)",
    )
    parser.add_argument(
        "--discount_rate",
        type=float,
        default=0.05,
        help="Annual discount rate (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip_counterfactual",
        action="store_true",
        help="Skip the WITHOUT Area 2 counterfactual run",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable dashboard plot generation",
    )
    args = parser.parse_args()

    run_two_area_economic_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        availability=args.availability,
        target_ore_stock_level=args.stockpile_target,
        area2_required_development=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        annual_discount_rate=args.discount_rate,
        seed=args.seed,
        run_counterfactual=not args.skip_counterfactual,
        plot=not args.no_plot,
    )

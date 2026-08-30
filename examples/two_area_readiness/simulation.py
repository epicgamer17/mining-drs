"""Two-Area Strategic & Tactical Planning with Area 2 Physical Unlocking & Readiness.

Combines:
1. Two Distinct Mining Areas / Faces with Stochastic Geological Facies:
   - Area 1 (Face 1, Level 3): Low Ore 2 grade (mean ore fraction: 0.15, std: 0.05). Active from Day 0.
   - Area 2 (Face 2, Level 6): High Ore 2 grade (mean ore fraction: 0.45, std: 0.05).
     * Initially physically locked until cumulative development reaches readiness target (e.g. 4,000 m).
     * Once unlocked, Face 2 enters active production and supplies high-grade ore.
2. Area 2 Readiness & Physical Unlock Mechanics:
   - AreaReadinessTarget (required_development: 4,000 m, ready_by_day: 365.0 days).
   - While Area 2 is locked, trucks allocated to Area 2 are redeployed to development.
   - Development metres accumulate based on development truck fleet (5.0 m/truck-day).
   - Persistent deadline-miss and lateness tracking (deadline_missed, currently_late, completed_late).
3. Strategic & Tactical Planning Framework:
   - StrategicYearTarget: Annual commitments (min development metres, min Ore 1, min Ore 2).
   - Monthly Tactical Review (every 30 days): Evaluates trajectory ratios including Area 2 readiness.
   - Dynamic MiningPriority selection (BALANCED, PRODUCTION, DEVELOPMENT).
   - Fleet Reservation: When in DEVELOPMENT priority, reserves extra haulage capacity for development.
4. Shelswell (2017) DES Haulage Engine & DRS Blending Modes:
   - Rate and buffer-aware dispatch throttling maintaining total stockpile at target buffer (60,000 t).
   - Disabled statutory holidays (NON_PRODUCTION_DAYS = 0 with NOTE:) for continuous supply.
   - 10.5 h workable shifts, 85% mechanical availability, AD30 payload cycles.
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
# Constants & Physical Parameters (Shelswell 2017 & Blending Modes)
# ---------------------------------------------------------------------------
DAYS_IN_YEAR = 365.0

# NOTE: Statutory non-production holidays (NON_PRODUCTION_DAYS = 11 in Shelswell 2017)
# have been disabled (set to 0) in this two-area readiness baseline. In the original
# Shelswell DES, 11 unworked holidays cause the surface stockpile to absorb 1-day
# continuous mill draws (dipping by 5.4k-6.0k t per holiday). Disabling holidays
# ensures uninterrupted fleet supply, keeping the total stockpile tightly centered
# at the target 60,000 t buffer matching blending_modes/simulation.py.
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
LEVEL_DRIFT_M = 60.0  # 40 m loadout + 20 m air door
SURFACE_M = 300.0  # Surface dump hopper from portal

# Speeds (Table 1, kph)
SPEEDS = {
    "surface": {"empty": 17.4, "loaded": 13.4},
    "decline": {"empty": 15.1, "loaded": 11.2},
    "ramp": {"empty": 12.9, "loaded": 9.2},
    "level": {"empty": 7.6, "loaded": 6.6},
}

# Payloads & Equipment Durations
ORE_PAYLOAD = 26.1  # AD30 rated capacity (tonnes)
TRUCK_LOAD_SPOT_MIN = 0.50
LHD_ACQUISITION_MAX_MIN = 0.80
TRUCK_LOAD_DUR_MIN = 3.50  # 2 bucket passes with high-capacity loader
DUMP_SPOT_MIN = 0.57
DUMP_MIN = 0.88

# Surface Dump Tip Capacities
SURFACE_TIP_SITES = 2

# Refuelling
FUEL_BURN_PCT_PER_SEC = 100.0 / (7.5 * 3600.0)
REFUEL_DUR_MIN = 25.0
N_FUEL_PUMPS = 2

# Traffic Congestion Delays
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
    IDLE = "idle"  # parked on surface, awaiting dispatch or shift
    EMPTY = "empty"  # empty travel surface -> underground loadout
    WAIT_LOAD = "wait_load"  # queued at face LHD
    SPOT_LOAD = "spot_load"  # spotting at loadout bay
    ACQUIRE = "acquire"  # waiting for LHD
    LOADING = "loading"  # active loading
    LOADED = "loaded"  # loaded travel face -> surface tip
    WAIT_DUMP = "wait_dump"  # queued at surface dump station
    DUMPING = "dumping"  # active dumping into stockpiles
    REFUELING = "refueling"  # fuel depot


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
    target_face_id: int = 1  # 1 for Area 1, 2 for Area 2
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
    _active_ore1_rate: float = 0.0  # t/sec into Ore1Stock
    _active_ore2_rate: float = 0.0  # t/sec into Ore2Stock


# ---------------------------------------------------------------------------
# Two-Area Strategic & Tactical Planning with Area 2 Physical Unlocking
# ---------------------------------------------------------------------------
class TwoAreaReadinessSimulation(drs.Module):
    """Two-Area Strategic Planning DES Simulation with Area 2 Physical Readiness Unlocking.

    - Area 1 (Face 1 at Level 3): mean ore fraction f1 = 0.15 (low Ore 2). Active from Day 0.
    - Area 2 (Face 2 at Level 6): mean ore fraction f2 = 0.45 (high Ore 2).
      * Initially LOCKED.
      * Unlocks when cumulative development reaches area2_readiness_target.required_development.
    - Tracks Area 2 readiness fraction, delivery deadlines, and persistent late-completion states.
    - Rate and Buffer-aware DES haulage keeping total stockpile stable at ~60,000 t.
    """

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
        area2_redeploy_locked_face_trucks_to_development: bool = True,
        development_priority_truck_reservation_fraction: float = 0.20,
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
        self.area2_redeploy_locked_face_trucks_to_development = (
            area2_redeploy_locked_face_trucks_to_development
        )
        self.development_priority_truck_reservation_fraction = (
            development_priority_truck_reservation_fraction
        )

        self.rng = random.Random(seed)
        self.seed = seed

        self.truck_seat_credit = availability * SEAT_PER_SHIFT_SEC
        self._down_dur = max(0.0, (1.0 - availability) * SEAT_PER_SHIFT_SEC)

        # Calendar setup
        # NOTE: Non-production holidays are disabled so total stockpile remains constant at ~60k buffer.
        self.holidays = set()
        self._cur_day = -1
        self._shift_marker = -1
        self._holiday_today = False

        # Global time tracker
        self.gt = drs.Timer("gt", 0.0, rate=1.0)

        # 1. Dual Mine Faces
        # Area 1 (Face 1, Level 3: 70/30 ratio, 30% Ore 2, active from Day 0)
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

        # Area 2 (Face 2, Level 6: 65/35 ratio, 35% Ore 2, physically locked until readiness target met)
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
        init_ore1 = 0.70 * target_ore_stock_level  # 42,000 t
        init_ore2 = 0.30 * target_ore_stock_level  # 18,000 t
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

        # 7. Area 2 Readiness & Physical Unlock Variables
        self.area2_ready = False
        self.area2_ready_day = drs.Level("area2_ready_day", -1.0)
        self.area2_deadline_missed = False
        self.area2_currently_late = False
        self.area2_completed_late = False
        self.area2_readiness_fraction = drs.Level("area2_readiness_fraction", 0.0)
        self.area2_readiness_trajectory_ratio = drs.Level("area2_readiness_trajectory_ratio", 1.0)

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
        if not self.area2_physical_unlock_enabled:
            return False
        required = max(0.0, float(self.area2_readiness_target.required_development))
        if required <= 1e-12:
            return False
        return not (self.strategic_planning_started and self.area2_ready)

    def _update_area2_readiness(self):
        """Updates Area 2 readiness metrics, unlocks face if threshold met, and tracks deadlines."""
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
            self.area2_deadline_missed = False
            self.area2_currently_late = False
            self.area2_completed_late = False

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

        # Campaign timer boundary (converted to seconds)
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

        # Extra development boost while Area 2 is locked and redeployment is enabled
        locked_boost = (
            (total_trucks * 0.35)
            if (
                self.is_area2_locked()
                and self.area2_redeploy_locked_face_trucks_to_development
            )
            else 0.0
        )
        dev_trucks = max(reserved_trucks, float(available_extra)) + locked_boost
        self.development_rate_m_per_day.value = (
            dev_trucks * DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY
        )
        delta_dev = self.development_rate_m_per_day.value * dt_days
        self.cumulative_mine_development.value += delta_dev

        # Allocate development metres to Area 2 project (measured from start of strategic planning)
        if self.is_area2_locked() and self.strategic_planning_started:
            prio = self.mining_priority
            if prio == MiningPriority.DEVELOPMENT:
                frac = 0.85
            elif prio == MiningPriority.BALANCED:
                frac = 0.60
            else:
                frac = 0.35
            self.area2_cumulative_development.value += delta_dev * frac

        self._update_area2_readiness()

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

        # Check if strategic planning is started (after warmup or if no warmup specified)
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

            # Select Mining Priority based on Trajectory Deficits (incorporating Area 2 readiness ratio)
            selected = select_mining_priority(
                development_ratio=float(self.development_trajectory_ratio.value),
                ore1_ratio=float(self.ore1_trajectory_ratio.value),
                ore2_ratio=float(self.ore2_trajectory_ratio.value),
                tolerance=self.tactical_progress_tolerance,
                area2_readiness_trajectory_ratio=float(self.area2_readiness_trajectory_ratio.value),
            )
            self.mining_priority = selected

            # Update Fleet Reservation for Development
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

    # -- Dispatch Policy (Target & Buffer Aware) ------------------------------
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

        # Target and Buffer dispatch regulation
        total_stock = self.ore1_stock.level + self.ore2_stock.level
        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self._release_operator(tr)
            return False

        # Development reservation: check if production trucks limit is reached
        reserved_trucks = int(self.development_priority_reserved_trucks.value)
        max_production_trucks = max(1, len(self.trucks) - reserved_trucks)
        active_prod_trucks = sum(
            1 for trk in self.trucks if trk.phase in OPERATING_PHASES
        )
        if active_prod_trucks >= max_production_trucks:
            self._release_operator(tr)
            return False

        # Progress of current 24-hr day
        day_progress = (t % 86400.0) / 86400.0
        expected_hauled_by_now = self.daily_target_ore * day_progress
        if total_stock >= self.target_ore_stock_level and self.daily_hauled_ore > expected_hauled_by_now + 100.0:
            self._release_operator(tr)
            return False

        if not self._acquire_operator(tr):
            self._release_operator(tr)
            return False

        # Choose Face (Area 1 vs Area 2) based on Blending Needs and Area 2 lock status
        target_face_id = self._select_face_by_blend_need()
        tr.target_face_id = target_face_id
        tr.target_level = AREA1_LEVEL if target_face_id == 1 else AREA2_LEVEL

        tr.trip_start = self.gt.value
        tr.phase = TruckPhase.EMPTY
        tr.timer.value = self._travel_time(tr, loaded=False)
        return True

    def _select_face_by_blend_need(self) -> int:
        """Selects between Face 1 (low Ore 2) and Face 2 (high Ore 2)."""
        # If Area 2 is locked, all trucks MUST go to Face 1 (Area 1)
        if self.is_area2_locked():
            return 1

        mode_name = self.plant.active_operating_mode.value.name
        if "MODE_A" in mode_name:
            # Mode A needs more Ore 2 -> bias towards Face 2
            p_face2 = 0.65
        elif "MODE_B" in mode_name:
            # Mode B needs less Ore 2 -> bias towards Face 1
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

    # -- State Transitions ---------------------------------------------------
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

        # Capture active parcel ore fraction from selected face
        tr.payload_ore_fraction = face.active_parcel_ore_fraction.value

        # Advance face parcel state
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

        # Split payload into continuous inflow rates
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

        # Bookkeeping
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

    # -- Travel Times & Congestion -------------------------------------------
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
# Statistics & Visual Dashboards
# ---------------------------------------------------------------------------
def print_statistics(plant, sim: TwoAreaReadinessSimulation):
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


def print_strategic_tactical_review_table(df: pd.DataFrame):
    """Prints tabular summary of monthly tactical reviews and Area 2 readiness trajectory."""
    print("\n" + "=" * 115)
    print(" STRATEGIC & TACTICAL PLANNING + AREA 2 READINESS REVIEW LOG")
    print("=" * 115)
    if "tactical_review_count" not in df.columns:
        print("No tactical review data available.")
        return

    review_df = df[
        df["tactical_review_count"].ne(df["tactical_review_count"].shift())
        & (df["tactical_review_count"] > 0)
    ]

    cols_to_print = [
        "time",
        "strategic_year_index",
        "tactical_review_count",
        "mining_priority",
        "development_trajectory_ratio",
        "area2_readiness_trajectory_ratio",
        "area2_development_progress",
        "area2_readiness_fraction",
        "area2_ready",
        "development_priority_reserved_trucks",
    ]
    available_cols = [c for c in cols_to_print if c in review_df.columns]

    if not review_df.empty and available_cols:
        print(review_df[available_cols].to_string(index=False))
    else:
        print("No tactical reviews occurred.")
    print("=" * 115 + "\n")


def print_simulation_statistics(
    sim: TwoAreaReadinessSimulation, df: pd.DataFrame
):
    """Prints operational summary statistics."""
    total_days = sim.gt.value / 86400.0
    total_ore_hauled = sum(f.cumulative_extracted_mass.value for f in sim.faces)
    total_milled = sim.plant.cumulative_milled_mass.value
    active_days = sim.plant.active_duration(sim.plant.total_duration)

    print("\n" + "=" * 70)
    print(" TWO-AREA STRATEGIC DES + AREA 2 READINESS RESULTS")
    print("=" * 70)
    print(f"Simulation Horizon:        {total_days:.1f} days")
    print(f"Total Trips Completed:     {sim.trips}")
    avg_cycle = (sim._cycle_sum / max(1, sim.trips)) / 60.0
    print(f"Average Truck Cycle Time:  {avg_cycle:.2f} min")
    print(f"Total Ore Hauled:          {total_ore_hauled:,.1f} t ({total_ore_hauled / max(1e-3, total_days):.1f} t/d)")
    print(f"  ↳ Face 1 (Area 1, 15%):  {sim.face1.cumulative_extracted_mass.value:,.1f} t")
    print(f"  ↳ Face 2 (Area 2, 45%):  {sim.face2.cumulative_extracted_mass.value:,.1f} t")
    print(f"Total Ore Milled:          {total_milled:,.1f} t ({total_milled / max(1e-3, active_days):.1f} t/active-day)")
    print(f"Final Ore 1 Stockpile:     {sim.ore1_stock.level:,.1f} t")
    print(f"Final Ore 2 Stockpile:     {sim.ore2_stock.level:,.1f} t")
    print(f"Final Total Stockpile:     {sim.ore1_stock.level + sim.ore2_stock.level:,.1f} t")
    print(f"Cumulative Mine Dev:       {sim.cumulative_mine_development.value:,.1f} metres")
    print(f"Area 2 Dev Progress:       {sim.area2_cumulative_development.value:,.1f} / {sim.area2_readiness_target.required_development:.1f} metres")
    print(f"Area 2 Unlocked:           {sim.area2_ready} (Unlocked Day: {sim.area2_ready_day.value:.2f})")
    print(f"Area 2 Deadline Missed:    {sim.area2_deadline_missed}")
    print(f"Total Traffic Delay:       {sim.traffic_delay_sum / 60.0:,.1f} truck-min")

    print("\n--- Operating Mode Time-Shares ---")
    tot_dur = sim.plant.total_duration
    if tot_dur > 0:
        for attr, label in [
            ("cumulative_time_mode_a", "Mode A (Normal)"),
            ("cumulative_time_mode_a_contingency", "Mode A (Contingency)"),
            ("cumulative_time_mode_a_surging", "Mode A (Surging)"),
            ("cumulative_time_mode_b", "Mode B (Normal)"),
            ("cumulative_time_mode_b_contingency", "Mode B (Contingency)"),
            ("cumulative_time_mode_b_surging", "Mode B (Surging)"),
            ("cumulative_time_shutdown", "Shutdown"),
        ]:
            val = getattr(sim.plant, attr).value
            print(f"  {label:<25}: {val:.2f} days ({100.0 * val / tot_dur:5.1f} %)")
    print("=" * 70 + "\n")


def plot_two_area_readiness_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/two_area_readiness_dashboard.png",
    palette: dict = None,
    figsize: Tuple[int, int] = (16, 52),
):
    """Builds and saves the 13-panel comprehensive diagnostics dashboard."""
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if "active_operating_mode_name" not in df.columns or "Mode A" not in df.columns:
        df = prepare_history(df)

    dash = Dashboard(
        nrows=13,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title="Two-Area Strategic Planning with Area 2 Physical Readiness Diagnostics",
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 7, 10, 12])

    # 0. Operating Modes Step Timeline
    plot_time_series(
        df,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes (Step Timeline)",
        is_step=True,
        ax=dash[0],
    )

    # Detect first physical unlock event
    unlock_rows = df[df["area2_ready"] == True]
    unlock_time = (
        float(unlock_rows["time"].iloc[0]) if not unlock_rows.empty else None
    )

    hlines_stockpile = [
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
    ]

    # 1. Stockpiles & Operating Modes
    plot_ore_with_modes(
        df,
        time_col="time",
        ore_cols=["total_system_ore_mass", "Ore1Stock_mass", "Ore2Stock_mass"],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Ore Stockpiles & Operating Campaigns (Mine 2 Unlock Demarcation)",
        palette=palette,
        hlines=hlines_stockpile,
        ax=dash[1],
    )

    if unlock_time is not None:
        # Shaded locked region (pre-unlock)
        dash[1].axvspan(
            df["time"].min(),
            unlock_time,
            color="#ffebee",
            alpha=0.35,
            label="Mine 2 Locked (Dev Phase)",
        )
        # Prominent vertical line for Mine 2 Unlock
        dash[1].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"★ Mine 2 Unlocked (Day {unlock_time:.1f})",
        )
        # Annotation Callout Box
        t_max = df["time"].max()
        text_x = (
            unlock_time + (t_max * 0.03)
            if (unlock_time < t_max * 0.80)
            else unlock_time - (t_max * 0.18)
        )
        dash[1].annotate(
            f"★ MINE 2 UNLOCKED\nDay {unlock_time:.1f}",
            xy=(unlock_time, 48000.0),
            xytext=(text_x, 52000.0),
            arrowprops=dict(
                facecolor="#880e4f",
                edgecolor="#880e4f",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#fce4ec",
                edgecolor="#880e4f",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#880e4f",
            zorder=10,
        )
        dash[1].legend(loc="upper right", framealpha=0.90)

        # Also add vertical indicator to Panel 0 (Operating Modes) and Panel 2 (Readiness)
        dash[0].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.8,
            label=f"Mine 2 Unlocked (Day {unlock_time:.1f})",
        )
        dash[0].legend(loc="upper right", framealpha=0.85)

        dash[2].axvline(
            unlock_time,
            color="#880e4f",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.8,
            label=f"Mine 2 Unlocked (Day {unlock_time:.1f})",
        )

    # 2. Area 2 Readiness Progress (Metres) & Unlock State
    plot_dual_axis_step(
        df,
        y1_col="area2_cumulative_development",
        y2_col="area2_readiness_fraction",
        y1_label="Area 2 Dev (m)",
        y2_label="Readiness Fraction (1.0 = Ready)",
        title="Area 2 Development Progress & Readiness Fraction",
        ax=dash[2],
    )

    # 3. Strategic & Area 2 Trajectory Progress Ratios
    plot_time_series(
        df,
        y_columns=[
            "development_trajectory_ratio",
            "area2_readiness_trajectory_ratio",
            "ore1_trajectory_ratio",
            "ore2_trajectory_ratio",
        ],
        title="Strategic & Area 2 Trajectory Progress Ratios (1.0 = On Schedule, <0.90 = Deficit)",
        is_step=True,
        ax=dash[3],
    )
    dash[3].axhline(0.90, color="red", linestyle=":", label="Tolerance Threshold (0.90)")
    dash[3].axhline(1.00, color="gray", linestyle="--", label="Target Trajectory (1.00)")
    dash[3].set_ylabel("Trajectory Ratio")
    dash[3].legend(loc="upper right")

    # 4. Cumulative Mine Development & Tactical Truck Reservations
    plot_dual_axis_step(
        df,
        y1_col="cumulative_mine_development",
        y2_col="development_priority_reserved_trucks",
        y1_label="Total Mine Dev (m)",
        y2_label="Reserved Dev Trucks",
        title="Total Underground Development & Tactical Truck Reservations",
        ax=dash[4],
    )

    # 5. Safety Margin: Ore 1
    plot_safety_margin(
        df,
        level_col="Ore1Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 1 Distance to Starvation Floor",
        danger_threshold=5000.0,
        ax=dash[5],
    )

    # 6. Safety Margin: Ore 2
    plot_safety_margin(
        df,
        level_col="Ore2Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 2 Distance to Starvation Floor",
        danger_threshold=3000.0,
        ax=dash[6],
    )

    # 7. Underground Haulage Fleet Activity & Queuing
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
        ax=dash[7],
    )

    # 8. Mode Distribution (% of Time Spent)
    plot_mode_distribution(
        df,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% of Time Spent)",
        palette=palette,
        ax=dash[8],
    )

    # 9. Mode Stability (Dwell Times)
    plot_mode_dwell_times(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        title="Mode Stability & Campaign Dwell Times",
        ax=dash[9],
    )

    # 10. Cumulative Production Deficit by Mode
    plot_attributed_deficit(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        extraction_col="cumulative_extracted_mass",
        ideal_rate_per_day=6000.0,
        title="Cumulative Production Deficit by Operating Mode",
        palette=palette,
        ax=dash[10],
    )

    # 11. Deficit Breakdown Bar
    plot_deficit_breakdown_bar(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[11],
    )

    # 12. Fleet Utilization & Idle Time Breakdown
    plot_truck_idle_and_utilization(
        df,
        title="Haul Fleet Utilization & Idle Time Breakdown",
        ax=dash[12],
    )

    dash.save(output_path)
    print(f"Saved dashboard visualization to '{output_path}'.")
    return dash


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------
def run_two_area_readiness_simulation(
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    availability: float = 0.85,
    target_ore_stock_level: float = 60000.0,
    min_development: float = 10000.0,
    min_ore1_production: float = 1300000.0,
    min_ore2_production: float = 850000.0,
    area2_required_development: float = 4000.0,
    area2_ready_by_day: float = 365.0,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[TwoAreaReadinessSimulation, pd.DataFrame]:
    """Builds and runs the Two-Area strategic simulation with Area 2 physical readiness."""
    strategic_target = StrategicYearTarget(
        min_development=min_development,
        min_ore1_production=min_ore1_production,
        min_ore2_production=min_ore2_production,
    )
    area2_target = AreaReadinessTarget(
        required_development=area2_required_development,
        ready_by_day=area2_ready_by_day,
    )

    sim = TwoAreaReadinessSimulation(
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        strategic_targets=(strategic_target,),
        area2_readiness_target=area2_target,
        seed=seed,
    )

    engine = drs.DRSEngine(max_step_size=DT_MAX)
    engine.register(sim)
    engine.on_step(sim.on_event)

    if total_days is not None:
        # Time-horizon mode
        sim.horizon_sec = total_days * 86400.0
        engine.run(until=sim.horizon_sec)
    else:
        # Phase 1: Warmup Phase (extract initial burn-in tonnage)
        sim.total_ore_to_extract = ore_to_be_extracted_during_warming_period
        sim.face1.total_ore_to_extract = ore_to_be_extracted_during_warming_period
        sim.face2.total_ore_to_extract = ore_to_be_extracted_during_warming_period
        engine.run(until=float("inf"))

        # Reset plant operating mode duration timers and milled mass for production metrics
        sim.plant.reset_mode_timers()
        sim.plant.cumulative_milled_mass.value = 0.0

        # Phase 2: Production Measurement Phase
        sim.total_ore_to_extract = total_ore_to_extract
        sim.face1.total_ore_to_extract = total_ore_to_extract
        sim.face2.total_ore_to_extract = total_ore_to_extract
        engine.run(until=float("inf"))

    df = pd.DataFrame(sim.history_records)
    print_simulation_statistics(sim, df)
    print_statistics(sim.plant, sim)
    print_strategic_tactical_review_table(df)

    df_prepared = prepare_history(df)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim.critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        label="Two-Area Readiness Blending",
    )
    print_deficit_by_mode(
        df_prepared,
        extraction_cols=["cumulative_extracted_mass"],
        ideal_rate=6000.0,
    )

    if plot and len(df_prepared) > 0:
        plot_two_area_readiness_dashboard(df_prepared)
    return sim, df_prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Two-Area Strategic Planning & Area 2 Readiness Simulation"
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
        help="Total simulation duration in days (optional, overrides tonnage termination)",
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
        help="Overall mechanical availability fraction (default: 0.85)",
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
        help="Target schedule deadline in days for Area 2 readiness (default: 365.0 d)",
    )
    parser.add_argument(
        "--min_development",
        type=float,
        default=10000.0,
        help="Annual minimum development metres target (default: 10,000.0 m)",
    )
    parser.add_argument(
        "--min_ore1",
        type=float,
        default=1300000.0,
        help="Annual minimum Ore 1 extraction target (default: 1,300,000.0 t)",
    )
    parser.add_argument(
        "--min_ore2",
        type=float,
        default=850000.0,
        help="Annual minimum Ore 2 extraction target (default: 850,000.0 t)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable dashboard plot generation",
    )
    args = parser.parse_args()

    run_two_area_readiness_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        availability=args.availability,
        target_ore_stock_level=args.stockpile_target,
        area2_required_development=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        min_development=args.min_development,
        min_ore1_production=args.min_ore1,
        min_ore2_production=args.min_ore2,
        seed=args.seed,
        plot=not args.no_plot,
    )

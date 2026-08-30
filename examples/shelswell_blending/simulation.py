"""Shelswell (2017) Discrete Event Simulation with DRS Blending Modes & Plant Operations.

Combines:
1. DES Underground Truck-Loader Haulage (Shelswell & Labrecque 2017):
   - 7-level spiral ramp + 2100 m decline access.
   - Shared level LHDs, AD30 haulage trucks, operator pooling, shift seat-time.
   - Mechanical availability downtime windows, refuelling, traffic congestion.
2. DRS Metallurgical Plant & Operating Mode Controller:
   - Campaign supervisory controller (Mode A: 34 days, Mode B: 34 days, Shutdown: 1 day).
   - Metallurgical plant with dual-ore blending (Mode A, Mode B, Contingency, Surging, Shutdown).
   - Continuous surface stockpiles (Ore 1 Stockpile, Ore 2 Stockpile, Waste Stockpile).
3. Dynamic Target-Driven Fleet Dispatch & Buffer Regulation:
   - Daily production targets (Ore 1, Ore 2, Waste) generated dynamically from plant operating modes.
   - Target deficit-weighted payload dispatch with Shelswell highest-unclaimed-muck routing.
   - Rate and buffer-aware dispatch throttling maintaining total stockpile at target buffer (60,000 t).
4. Two-Phase Lifecycle & Milestone-Based Execution:
   - Phase 1 Warmup (600k t) + reset mode timers + Phase 2 Production (6.6 Mt).
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
from drs_mining.config import (
    MILL_MODES,
    CalendarConfig,
    TopologyConfig,
    HaulageFleetConfig,
    PlantConfig,
    GeologyConfig,
    SimulationConfig,
)
from drs_mining.components import (
    OperatingMode,
    MetallurgicalPlant,
    PlantDrawRates,
    Stockpile,
    OperatingModeController,
    TruckPhase,
    Operator,
    OPERATING_PHASES,
    SEAT_PHASES,
    DUE_PHASES,
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
# Centralized Config Constants
# ---------------------------------------------------------------------------
_CFG = SimulationConfig()
DAYS_IN_YEAR = _CFG.calendar.days_in_year
NON_PRODUCTION_DAYS = _CFG.calendar.non_production_days
SHIFT_SECONDS = _CFG.calendar.shift_seconds
SHIFT_WORK_HOURS = _CFG.calendar.shift_work_hours
HAULAGE_SEAT_FRACTION = _CFG.calendar.haulage_seat_fraction
SEAT_PER_SHIFT_SEC = _CFG.calendar.seat_per_shift_sec

DECLINE_M = _CFG.topology.decline_m
LEVEL_SPACING_M = _CFG.topology.level_spacing_m
ORE_LEVEL_DRIFT_M = _CFG.topology.level_drift_m
WASTE_LEVEL_DRIFT_M = 75.0
ROM_SURFACE_M = _CFG.topology.surface_m
WASTE_SURFACE_M = 440.0
N_LEVELS = 7
SPEEDS = _CFG.topology.speeds

ORE_PAYLOAD = _CFG.fleet.truck_payload
WASTE_PAYLOAD = 24.6
TRUCK_LOAD_SPOT_MIN = 0.82
LHD_ACQUISITION_MAX_MIN = 3.0
TRUCK_LOAD_DUR_MIN = 6.69
DUMP_SPOT_MIN = _CFG.fleet.dump_spot_min
DUMP_MIN = _CFG.fleet.dump_dur_min

ROM_TIP_SITES_ORE1 = 2
ROM_TIP_SITES_ORE2 = 2
WASTE_TIP_SITES = 2

FUEL_BURN_PCT_PER_SEC = _CFG.fleet.fuel_burn_pct_per_sec
REFUEL_DUR_MIN = _CFG.fleet.refuel_dur_min
N_FUEL_PUMPS = _CFG.fleet.num_fuel_pumps
BASE_PASS_BAY_DELAY_SEC = _CFG.fleet.base_pass_bay_delay_sec
PER_TRUCK_PASS_BAY_DELAY_SEC = _CFG.fleet.per_truck_pass_bay_delay_sec
DT_MAX = _CFG.dt_max


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
@dataclass
class Truck:
    truck_id: str
    timer: drs.Timer
    phase: TruckPhase = TruckPhase.IDLE
    target_loadout: int = -1
    target_level: int = 4
    payload_type: str = "ORE_1"  # "ORE_1", "ORE_2", or "WASTE"
    current_payload: float = 0.0
    seat_used: float = 0.0
    fuel: float = 100.0
    refuel_threshold: float = 30.0
    operator: int = -1
    trip_start: float = 0.0
    dump_dur: float = 0.0
    down_start: float = math.inf
    down_end: float = math.inf


@dataclass
class Loadout:
    idx: int
    level: int
    bay_type: str  # "ORE_1", "ORE_2", or "WASTE"
    muck_remaining: drs.Level
    queue: list = field(default_factory=list)
    last_assigned_seq: int = 0
    active: bool = True


@dataclass
class DumpSite:
    name: str
    capacity: int
    in_use: int = 0
    queue: list = field(default_factory=list)
    stockpile: Optional[Stockpile] = None
    level_accumulator: Optional[drs.Level] = None
    _active_rate: float = 0.0  # continuous inflow rate (t/sec) into stockpile during dumping


# ---------------------------------------------------------------------------
# Hybrid Simulation Module: Shelswell DES + DRS Blending Modes
# ---------------------------------------------------------------------------
class ShelswellBlendingHaulage(drs.Module):
    """Hybrid simulation combining Shelswell DES Truck-Loader Haulage with
    Continuous DRS Stockpiles and Metallurgical Plant Operations.
    """

    def __init__(
        self,
        num_trucks: int = 18,
        num_operators: int = 18,
        availability: float = 0.85,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        seed: int = 42,
        waste_daily_target: float = 500.0,
        mode_a_ore1_milling_rate: float = 3600.0,
        mode_a_ore2_milling_rate: float = 2400.0,
        mode_a_contingency_ore1_milling_rate: float = 3900.0,
        mode_b_ore1_milling_rate: float = 4600.0,
        mode_b_ore2_milling_rate: float = 800.0,
        mode_b_contingency_ore2_milling_rate: float = 2500.0,
    ):
        super().__init__()
        self.num_trucks = num_trucks
        self.num_operators = num_operators
        self.availability = availability
        self.target_ore_stock_level = target_ore_stock_level
        self.critical_ore2_level = critical_ore2_level
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = ore_to_be_extracted_during_warming_period
        self.waste_daily_target = waste_daily_target

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

        # Continuous Stockpiles (Tonnes)
        init_ore1 = 0.70 * target_ore_stock_level
        init_ore2 = 0.30 * target_ore_stock_level
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["grade"],
            initial_mass=init_ore1,
            initial_attributes={"grade": 1.0},
            attr_inflow=1.0,
        )
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["grade"],
            initial_mass=init_ore2,
            initial_attributes={"grade": 1.0},
            attr_inflow=1.0,
        )
        self.waste_hauled = drs.Level("waste_hauled", 0.0)
        self.ore1_hauled = drs.Level("ore1_hauled", 0.0)
        self.ore2_hauled = drs.Level("ore2_hauled", 0.0)
        self.cumulative_extracted_ore = drs.Level("cumulative_extracted_ore", 0.0)

        # Plant & Mode Controller (operating in continuous days/seconds)
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

        # Fleet & Operators
        self.trucks: List[Truck] = []
        for i in range(1, num_trucks + 1):
            timer = drs.Timer(f"tr_{i}_act", 0.0, rate=-1.0)
            timer.lower_threshold = 0.0
            tr = Truck(truck_id=f"T{i:02d}", timer=timer)
            tr.refuel_threshold = self.rng.uniform(15.0, 40.0)
            self.trucks.append(tr)

        self.operators = [Operator(i) for i in range(num_operators)]

        # Underground Loadouts: Ore 1, Ore 2, and Waste per level
        self.loadouts: List[Loadout] = []
        lo_idx = 0
        for level in range(1, N_LEVELS + 1):
            for bay_type in ("ORE_1", "ORE_2", "WASTE"):
                self.loadouts.append(
                    Loadout(
                        idx=lo_idx,
                        level=level,
                        bay_type=bay_type,
                        muck_remaining=drs.Level(f"muck_L{level}_{bay_type}", 0.0),
                    )
                )
                lo_idx += 1

        # Active levels: ~1 level per 2 trucks, centered at L4
        k = max(1, int(math.ceil(num_trucks / 2.0)))
        lvls = []
        if k % 2 == 1:
            lvls.append(4)
        for d in range(1, k // 2 + 1):
            lvls += [4 - d, 4 + d]
        self.active_levels = set(lvls)
        for lo in self.loadouts:
            lo.active = lo.level in self.active_levels

        # Surface Dump Sites
        self.dump_sites = {
            "ORE_1": DumpSite(
                "ROM_PAD_1", ROM_TIP_SITES_ORE1, stockpile=self.ore1_stock
            ),
            "ORE_2": DumpSite(
                "ROM_PAD_2", ROM_TIP_SITES_ORE2, stockpile=self.ore2_stock
            ),
            "WASTE": DumpSite(
                "WASTE_STOCKPILE",
                WASTE_TIP_SITES,
                level_accumulator=self.waste_hauled,
            ),
        }

        # Daily Target Tracking
        self.daily_target_ore1 = 0.0
        self.daily_target_ore2 = 0.0
        self.daily_target_waste = self.waste_daily_target
        self.daily_hauled_ore1 = 0.0
        self.daily_hauled_ore2 = 0.0
        self.daily_hauled_waste = 0.0

        # LHD and Shared Resources
        self._lhd_busy = {level: False for level in range(1, N_LEVELS + 1)}
        self._pumps_free = N_FUEL_PUMPS
        self._dispatch_counter = 0
        self.trips = 0
        self._cycle_sum = 0.0
        self.traffic_delay_sum = 0.0
        self.horizon_sec = float("inf")

        # Telemetry / History records
        self.history_records: List[dict] = []

    # -- DRS Hooks -----------------------------------------------------------
    def is_terminating_condition_met(self) -> bool:
        if (self.ore1_hauled.value + self.ore2_hauled.value) >= self.total_ore_to_extract - 1e-6:
            return True
        return self.gt.value >= self.horizon_sec - 1e-6

    def time_to_event(self) -> float:
        """Next discrete event boundary (truck timer, shift, day, mode timer)."""
        best = DT_MAX
        for tr in self.trucks:
            v = tr.timer.value
            if v > 1e-9:
                best = min(best, v)
        t = self.gt.value
        next_day = (math.floor(t / 86400.0) + 1.0) * 86400.0
        next_shift = (math.floor(t / SHIFT_SECONDS) + 1.0) * SHIFT_SECONDS
        best = min(best, next_day - t, next_shift - t)

        # Operating mode / campaign timer boundaries (scaled to seconds)
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

        if "_CONTINGENCY" in self.plant.active_operating_mode.value.name:
            c_thresh = self.plant.duration_of_contingency_segments
            rem_c_days = max(
                0.0, c_thresh - self.plant.current_contingency_duration.value
            )
            if rem_c_days > 1e-6:
                best = min(best, rem_c_days * 86400.0)

        return max(best, 1e-6)

    def step(self, dt: float):
        """Continuous integration between events."""
        self.gt.step(dt)
        dt_days = dt / 86400.0

        # Step campaign and plant mode timers
        self.mode_controller.current_campaign_duration.step(dt_days)
        active_mode_name = self.plant.active_operating_mode.value.name
        timer_attr = self.plant._MODE_TIMER_ATTRS.get(active_mode_name)
        if timer_attr:
            getattr(self.plant, timer_attr).step(dt_days)
        if active_mode_name in self.plant._CONTINGENCY_MODES:
            self.plant.current_contingency_duration.step(dt_days)

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

        # Continuous Stockpiles Inflow / Outflow balance
        ore1_in_rate = self.dump_sites["ORE_1"]._active_rate  # t/sec
        ore2_in_rate = self.dump_sites["ORE_2"]._active_rate  # t/sec

        plant_draw, _ = self.plant.get_target_rates(
            self.mode_controller.active_campaign_mode.value,
            ore1_level=self.ore1_stock.level,
            ore2_level=self.ore2_stock.level,
        )

        ore1_draw_rate_sec = plant_draw.ore1 / 86400.0
        ore2_draw_rate_sec = plant_draw.ore2 / 86400.0

        out1 = self.ore1_stock.feed_and_draw(ore1_in_rate, ore1_draw_rate_sec)
        out2 = self.ore2_stock.feed_and_draw(ore2_in_rate, ore2_draw_rate_sec)
        self.ore1_stock.step(dt)
        self.ore2_stock.step(dt)

        self.plant.process(out1 + out2)
        self.plant.cumulative_milled_mass.step(dt)

        # Waste accumulator step
        self.waste_hauled.step(dt)
        self.ore1_hauled.step(dt)
        self.ore2_hauled.step(dt)
        self.cumulative_extracted_ore.step(dt)

        # Record telemetry snapshot periodically
        self._record_telemetry(plant_draw)

    # -- Event Policy & Target Setting ----------------------------------------
    def on_event(self, t: float):
        """Engine step policy: calendar updates, mode updates, truck transitions."""
        self._calendar_update()
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

    def _update_operating_mode_and_targets(self):
        """Updates campaign mode, plant operating mode, and daily haulage targets."""
        # 1. Update Campaign Mode (Mode A, Mode B, Shutdown)
        camp_mode = self.mode_controller.update(
            ore2_stock_level=self.ore2_stock.level,
            total_stock_level=self.ore1_stock.level + self.ore2_stock.level,
        )

        # 2. Update Plant Operational State & Draw Rates
        plant_draw, _ = self.plant.get_target_rates(
            camp_mode,
            ore1_level=self.ore1_stock.level,
            ore2_level=self.ore2_stock.level,
        )

        # 3. Derive Daily Haulage Targets
        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self.daily_target_ore1 = 0.0
            self.daily_target_ore2 = 0.0
            self.daily_target_waste = self.waste_daily_target * 1.5  # boost dev during shutdown
        elif "_CONTINGENCY" in mode_name:
            self.daily_target_ore1 = plant_draw.ore1
            self.daily_target_ore2 = plant_draw.ore2
            self.daily_target_waste = self.waste_daily_target
        elif "_MINE_SURGING" in mode_name:
            # Surging draw down: scale targets down to allow stockpile reduction
            self.daily_target_ore1 = plant_draw.ore1 * 0.70
            self.daily_target_ore2 = plant_draw.ore2 * 0.70
            self.daily_target_waste = self.waste_daily_target * 1.2
        else:
            self.daily_target_ore1 = plant_draw.ore1
            self.daily_target_ore2 = plant_draw.ore2
            self.daily_target_waste = self.waste_daily_target

    def _calendar_update(self):
        t = self.gt.value
        day = int(t // 86400.0)
        if self._cur_day != day:
            self._cur_day = day
            self._holiday_today = False
            self.daily_hauled_ore1 = 0.0
            self.daily_hauled_ore2 = 0.0
            self.daily_hauled_waste = 0.0
            self._update_operating_mode_and_targets()
            self._schedule_daily_muck()

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

    def _schedule_daily_muck(self):
        """Replenishes underground loadouts with daily muck calls."""
        n_act = max(1, len(self.active_levels))
        ore1_call_per_lvl = max(1200.0, (self.daily_target_ore1 * 1.5) / n_act)
        ore2_call_per_lvl = max(900.0, (self.daily_target_ore2 * 1.5) / n_act)
        waste_call_per_lvl = max(400.0, (self.daily_target_waste * 1.5) / n_act)

        for lo in self.loadouts:
            if not lo.active:
                continue
            if lo.bay_type == "ORE_1":
                lo.muck_remaining.value += _tri(self.rng, ore1_call_per_lvl, 0.25)
            elif lo.bay_type == "ORE_2":
                lo.muck_remaining.value += _tri(self.rng, ore2_call_per_lvl, 0.25)
            else:
                lo.muck_remaining.value += _tri(self.rng, waste_call_per_lvl, 0.40)

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

    # -- Dispatch Policy (Target & Buffer Driven) -----------------------------
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
        # NOTE: If the surface total stockpile is at or above target buffer (60k) and daily
        # ore haulage is ahead of target schedule, dispatch is throttled to track mill draw.
        total_stock = self.ore1_stock.level + self.ore2_stock.level
        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self._release_operator(tr)
            return False

        day_progress = (t % 86400.0) / 86400.0
        expected_ore_hauled_by_now = (self.daily_target_ore1 + self.daily_target_ore2) * day_progress
        if (
            total_stock >= self.target_ore_stock_level
            and (self.daily_hauled_ore1 + self.daily_hauled_ore2) > expected_ore_hauled_by_now + 100.0
        ):
            self._release_operator(tr)
            return False

        # Select Payload Type (ORE_1, ORE_2, WASTE) based on Daily Target Deficits
        ptype = self._select_payload_by_target_deficit()
        tr.payload_type = ptype

        cands = [
            lo
            for lo in self.loadouts
            if lo.active and lo.bay_type == ptype and lo.muck_remaining.value > 5.0
        ]
        if not cands:
            # Fallback to any active muck bay with muck
            cands = [
                lo
                for lo in self.loadouts
                if lo.active and lo.muck_remaining.value > 5.0
            ]
            if not cands:
                self._release_operator(tr)
                return False
            # Update truck payload to fallback bay type
            target = max(
                cands,
                key=lambda lo: (lo.muck_remaining.value, -lo.last_assigned_seq),
            )
            tr.payload_type = target.bay_type
            ptype = target.bay_type
        else:
            target = max(
                cands,
                key=lambda lo: (lo.muck_remaining.value, -lo.last_assigned_seq),
            )

        if not self._acquire_operator(tr):
            self._release_operator(tr)
            return False

        self._dispatch_counter += 1
        target.last_assigned_seq = self._dispatch_counter
        claim = ORE_PAYLOAD if "ORE" in ptype else WASTE_PAYLOAD
        target.muck_remaining.value = max(0.0, target.muck_remaining.value - claim)
        tr.target_loadout = target.idx
        tr.target_level = target.level
        tr.trip_start = self.gt.value
        tr.phase = TruckPhase.EMPTY
        tr.timer.value = self._travel_time(tr, loaded=False)
        return True

    def _select_payload_by_target_deficit(self) -> str:
        """Chooses ORE_1, ORE_2, or WASTE proportionally to remaining daily target deficit."""
        def_ore1 = max(0.0, self.daily_target_ore1 - self.daily_hauled_ore1)
        def_ore2 = max(0.0, self.daily_target_ore2 - self.daily_hauled_ore2)
        def_waste = max(0.0, self.daily_target_waste - self.daily_hauled_waste)

        # Proportional stockpile starvation buffer feedback
        if self.ore2_stock.level < self.critical_ore2_level:
            urgency2 = 1.0 + (self.critical_ore2_level - self.ore2_stock.level) / max(1.0, self.critical_ore2_level)
            def_ore2 *= urgency2
        if self.ore1_stock.level < 0.35 * self.target_ore_stock_level:
            urgency1 = 1.0 + (0.35 * self.target_ore_stock_level - self.ore1_stock.level) / max(1.0, 0.35 * self.target_ore_stock_level)
            def_ore1 *= urgency1

        total_def = def_ore1 + def_ore2 + def_waste
        if total_def > 1e-3:
            p_ore1 = def_ore1 / total_def
            p_ore2 = def_ore2 / total_def
            r = self.rng.random()
            if r < p_ore1:
                return "ORE_1"
            elif r < p_ore1 + p_ore2:
                return "ORE_2"
            else:
                return "WASTE"

        w_ore1 = max(1.0, self.daily_target_ore1)
        w_ore2 = max(1.0, self.daily_target_ore2)
        w_waste = max(1.0, self.daily_target_waste)
        w_tot = w_ore1 + w_ore2 + w_waste
        r = self.rng.random()
        if r < w_ore1 / w_tot:
            return "ORE_1"
        elif r < (w_ore1 + w_ore2) / w_tot:
            return "ORE_2"
        return "WASTE"

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
            self._enter_loadout(tr)
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

    def _enter_loadout(self, tr: Truck):
        lo = self.loadouts[tr.target_loadout]
        lvl = lo.level
        if self._lhd_busy[lvl]:
            lo.queue.append(tr)
            tr.phase = TruckPhase.WAIT_LOAD
            tr.timer.value = 0.0
        else:
            self._lhd_busy[lvl] = True
            tr.phase = TruckPhase.SPOT_LOAD
            tr.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)

    def _finish_loading(self, tr: Truck):
        lvl = tr.target_level
        is_ore = "ORE" in tr.payload_type
        payload = _tri(self.rng, ORE_PAYLOAD if is_ore else WASTE_PAYLOAD, 0.08)
        tr.current_payload = payload

        self._lhd_busy[lvl] = False
        candidates = []
        for lo in self.loadouts:
            if lo.level == lvl and lo.queue:
                candidates.append(lo)
        if candidates:
            chosen_lo = max(candidates, key=lambda l: len(l.queue))
            nxt = chosen_lo.queue.pop(0)
            self._lhd_busy[lvl] = True
            nxt.phase = TruckPhase.SPOT_LOAD
            nxt.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)

        tr.phase = TruckPhase.LOADED
        tr.timer.value = self._travel_time(tr, loaded=True)

    def _enter_dump(self, tr: Truck):
        site = self.dump_sites[tr.payload_type]
        if site.in_use < site.capacity:
            self._start_dump(site, tr)
        else:
            site.queue.append(tr)
            tr.phase = TruckPhase.WAIT_DUMP
            tr.timer.value = 0.0

    def _start_dump(self, site: DumpSite, tr: Truck):
        dur = _tri(self.rng, DUMP_SPOT_MIN * 60.0, 0.20) + _tri(
            self.rng, DUMP_MIN * 60.0, 0.10
        )
        site.in_use += 1
        tr.phase = TruckPhase.DUMPING
        tr.timer.value = dur
        tr.dump_dur = dur

        # Inject continuous feed rate into target Stockpile during dumping
        if site.stockpile is not None:
            rate_val = tr.current_payload / dur
            site._active_rate += rate_val

    def _finish_dumping(self, tr: Truck):
        site = self.dump_sites[tr.payload_type]
        if site.stockpile is not None:
            rate_val = tr.current_payload / tr.dump_dur
            site._active_rate = max(0.0, site._active_rate - rate_val)
        elif site.level_accumulator is not None:
            site.level_accumulator.value += tr.current_payload

        site.in_use -= 1

        if tr.payload_type == "ORE_1":
            self.daily_hauled_ore1 += tr.current_payload
            self.ore1_hauled.value += tr.current_payload
            self.cumulative_extracted_ore.value += tr.current_payload
        elif tr.payload_type == "ORE_2":
            self.daily_hauled_ore2 += tr.current_payload
            self.ore2_hauled.value += tr.current_payload
            self.cumulative_extracted_ore.value += tr.current_payload
        else:
            self.daily_hauled_waste += tr.current_payload

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
        lvl = tr.target_level
        is_ore = "ORE" in tr.payload_type
        load_key = "loaded" if loaded else "empty"

        def seg(dist: float, kind: str) -> float:
            return dist / (SPEEDS[kind][load_key] / 3.6)

        surface_dist = ROM_SURFACE_M if is_ore else WASTE_SURFACE_M
        ramp_dist = (lvl - 1) * LEVEL_SPACING_M
        drift_dist = ORE_LEVEL_DRIFT_M if is_ore else WASTE_LEVEL_DRIFT_M

        t = (
            seg(surface_dist, "surface")
            + seg(DECLINE_M, "decline")
            + seg(ramp_dist, "ramp")
            + seg(drift_dist, "level")
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

        n_waiting_load = sum(len(lo.queue) for lo in self.loadouts)
        n_waiting_dump = sum(len(s.queue) for s in self.dump_sites.values())
        n_refueling = sum(
            1 for tr in self.trucks if tr.phase == TruckPhase.REFUELING
        )
        n_operating = sum(
            1 for tr in self.trucks if tr.phase in OPERATING_PHASES
        )

        total_ore_hauled = self.ore1_hauled.value + self.ore2_hauled.value

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
                "current_campaign_duration": self.mode_controller.current_campaign_duration.value,
                "current_contingency_duration": self.plant.current_contingency_duration.value,
                "daily_target_ore1": self.daily_target_ore1,
                "daily_target_ore2": self.daily_target_ore2,
                "daily_target_total_ore": self.daily_target_ore1
                + self.daily_target_ore2,
                "daily_hauled_ore1": self.daily_hauled_ore1,
                "daily_hauled_ore2": self.daily_hauled_ore2,
                "daily_hauled_total_ore": self.daily_hauled_ore1
                + self.daily_hauled_ore2,
                "daily_target_waste": self.daily_target_waste,
                "daily_hauled_waste": self.daily_hauled_waste,
                "ore1_hauled_total": self.ore1_hauled.value,
                "ore2_hauled_total": self.ore2_hauled.value,
                "cumulative_extracted_mass": total_ore_hauled,
                "waste_hauled_total": self.waste_hauled.value,
                "milled_ore1_rate": plant_draw.ore1,
                "milled_ore2_rate": plant_draw.ore2,
                "cumulative_milled_mass": self.plant.cumulative_milled_mass.value,
                "active_trucks": n_operating,
                "trucks_operating": n_operating,
                "trucks_waiting_load": n_waiting_load,
                "trucks_waiting_dump": n_waiting_dump,
                "trucks_refueling": n_refueling,
                "trucks_idle": max(0, len(self.trucks) - (n_operating + n_refueling)),
                "truck_idle_fraction": max(0, len(self.trucks) - (n_operating + n_refueling)) / max(1, len(self.trucks)),
                "traffic_delay_min": self.traffic_delay_sum / 60.0,
            }
        )

    def run(self, total_days: float = 365.0) -> pd.DataFrame:
        self.horizon_sec = total_days * 86400.0
        engine = drs.DRSEngine(max_step_size=DT_MAX)
        engine.register(self)
        engine.on_step(self.on_event)
        engine.run(until=self.horizon_sec)
        return pd.DataFrame(self.history_records)


# ---------------------------------------------------------------------------
# Statistics & Visual Dashboards
# ---------------------------------------------------------------------------
def print_statistics(plant, sim: ShelswellBlendingHaulage):
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
            print(
                f"{label}: {getattr(plant, attr).value / total_time:.4f}"
            )
    else:
        print("Total time is 0. Cannot calculate mode portions.")

    active_time = plant.active_duration(total_time)
    if active_time > 0:
        total_ore_processed = plant.cumulative_milled_mass.value
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


def print_simulation_statistics(sim: ShelswellBlendingHaulage, df: pd.DataFrame):
    """Prints operational summary statistics."""
    total_days = sim.gt.value / 86400.0
    total_ore_hauled = sim.ore1_hauled.value + sim.ore2_hauled.value
    total_milled = sim.plant.cumulative_milled_mass.value
    active_days = sim.plant.active_duration(sim.plant.total_duration)

    print("\n" + "=" * 70)
    print(" SHELSWELL DES + DRS BLENDING MODES SIMULATION RESULTS")
    print("=" * 70)
    print(f"Simulation Horizon:        {total_days:.1f} days")
    print(f"Total Trips Completed:     {sim.trips}")
    avg_cycle = (sim._cycle_sum / max(1, sim.trips)) / 60.0
    print(f"Average Truck Cycle Time:  {avg_cycle:.2f} min")
    print(f"Total Ore 1 Hauled:        {sim.ore1_hauled.value:,.1f} t ({sim.ore1_hauled.value / max(1e-3, total_days):.1f} t/d)")
    print(f"Total Ore 2 Hauled:        {sim.ore2_hauled.value:,.1f} t ({sim.ore2_hauled.value / max(1e-3, total_days):.1f} t/d)")
    print(f"Total Combined Ore Hauled: {total_ore_hauled:,.1f} t ({total_ore_hauled / max(1e-3, total_days):.1f} t/d)")
    print(f"Total Waste Hauled:        {sim.waste_hauled.value:,.1f} t ({sim.waste_hauled.value / max(1e-3, total_days):.1f} t/d)")
    print(f"Total Ore Milled:          {total_milled:,.1f} t ({total_milled / max(1e-3, active_days):.1f} t/active-day)")
    print(f"Final Ore 1 Stockpile:     {sim.ore1_stock.level:,.1f} t")
    print(f"Final Ore 2 Stockpile:     {sim.ore2_stock.level:,.1f} t")
    print(f"Final Total Stockpile:     {sim.ore1_stock.level + sim.ore2_stock.level:,.1f} t")
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


def plot_shelswell_blending_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/shelswell_blending_dashboard.png",
    palette: dict = None,
    figsize: Tuple[int, int] = (16, 48),
):
    """Builds and saves the 12-panel comprehensive diagnostics dashboard."""
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if "active_operating_mode_name" not in df.columns or "Mode A" not in df.columns:
        df = prepare_history(df)

    dash = Dashboard(
        nrows=12,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title="Shelswell (2017) DES Haulage + DRS Blending Modes Diagnostics",
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 9, 11])

    # 0. Operating Modes Step Timeline
    plot_time_series(
        df,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes (Step Timeline)",
        is_step=True,
        ax=dash[0],
    )

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

    # 2. Daily Targets vs Hauled Production (Ore 1 & Ore 2)
    plot_time_series(
        df,
        y_columns=[
            "daily_target_ore1",
            "daily_hauled_ore1",
            "daily_target_ore2",
            "daily_hauled_ore2",
        ],
        title="Dynamic Target Deficit-Driven Underground Haulage Rates (t/d)",
        is_step=True,
        ax=dash[2],
    )

    # 3. Total Daily Ore Target vs Combined Hauled
    plot_time_series(
        df,
        y_columns=[
            "daily_target_total_ore",
            "daily_hauled_total_ore",
        ],
        title="Total Daily Ore Extraction Target vs Combined Haulage (t/d)",
        is_step=True,
        ax=dash[3],
    )

    # 4. Safety Margin: Ore 1
    plot_safety_margin(
        df,
        level_col="Ore1Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 1 Distance to Starvation Floor",
        danger_threshold=5000.0,
        ax=dash[4],
    )

    # 5. Safety Margin: Ore 2
    plot_safety_margin(
        df,
        level_col="Ore2Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 2 Distance to Starvation Floor",
        danger_threshold=3000.0,
        ax=dash[5],
    )

    # 6. Underground Haulage Fleet Activity & Queuing
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
        ax=dash[6],
    )

    # 7. Mode Distribution (% of Time Spent)
    plot_mode_distribution(
        df,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% of Time Spent)",
        palette=palette,
        ax=dash[7],
    )

    # 8. Mode Stability (Dwell Times)
    plot_mode_dwell_times(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        title="Mode Stability & Campaign Dwell Times",
        ax=dash[8],
    )

    # 9. Cumulative Production Deficit by Mode
    plot_attributed_deficit(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        extraction_col="cumulative_extracted_mass",
        ideal_rate_per_day=6000.0,
        title="Cumulative Production Deficit by Operating Mode",
        palette=palette,
        ax=dash[9],
    )

    # 10. Deficit Breakdown Bar
    plot_deficit_breakdown_bar(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[10],
    )

    # 11. Fleet Utilization & Idle Time Breakdown
    plot_truck_idle_and_utilization(
        df,
        title="Haul Fleet Utilization & Idle Time Breakdown",
        ax=dash[11],
    )

    dash.save(output_path)
    print(f"Saved dashboard visualization to '{output_path}'.")
    return dash


# ---------------------------------------------------------------------------
# CLI & Execution
# ---------------------------------------------------------------------------
def run_shelswell_blending_simulation(
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    availability: float = 0.85,
    target_ore_stock_level: float = 60000.0,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[ShelswellBlendingHaulage, pd.DataFrame]:
    """Builds and runs the hybrid simulation with two-phase lifecycle support."""
    sim = ShelswellBlendingHaulage(
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
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
        engine.run(until=float("inf"))

        # Reset plant operating mode duration timers for production metrics
        sim.plant.reset_mode_timers()

        # Phase 2: Production Measurement Phase
        sim.total_ore_to_extract = total_ore_to_extract
        engine.run(until=float("inf"))

    df = pd.DataFrame(sim.history_records)
    print_simulation_statistics(sim, df)
    print_statistics(sim.plant, sim)

    df_prepared = prepare_history(df)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim.critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        label="Shelswell Blending",
    )
    print_deficit_by_mode(
        df_prepared,
        extraction_cols=["cumulative_extracted_mass"],
        ideal_rate=6000.0,
    )

    if plot and len(df_prepared) > 0:
        plot_shelswell_blending_dashboard(df_prepared)
    return sim, df_prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Shelswell DES + DRS Blending Modes Hybrid Simulation"
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
        help="Number of AD30 haulage trucks (default: 18 for 6,000 t/d mill)",
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

    run_shelswell_blending_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        availability=args.availability,
        target_ore_stock_level=args.stockpile_target,
        seed=args.seed,
        plot=not args.no_plot,
    )

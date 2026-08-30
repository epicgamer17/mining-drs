"""Shelswell (2017) DES Haulage + Single-Face DRS Blending Modes Simulation.

Combines:
1. Single Mine Face with Stochastic Geological Facies (blending_modes paradigm):
   - 1 Active mining face with StochasticFaciesGenerator (mean: 0.30, std: 0.05).
   - Stochastic parcel progression (30,000 - 50,000 tonnes per parcel).
   - Each parcel carries an ore fraction f representing the split into Ore 2 and Ore 1.
2. Discrete Event Simulation (DES) Underground Haulage (Shelswell & Labrecque 2017):
   - AD30 haulage trucks, 2 shared face LHD loaders, operator pooling, shift seat-time.
   - 10.5 h shift calendar with 1.5 h off-shift gaps, 85% mechanical availability.
   - Passing bay traffic congestion, surface fuel depot, surface dump station.
3. Surface Stockpile Coupling & DRS Plant Operations:
   - When a truck dumps at the surface hopper, its payload is split into Ore 1 and Ore 2
     inflow rates according to the loaded parcel's ore fraction f.
   - Dual continuous stockpiles (Ore 1 & Ore 2, buffer target: 60,000 t).
   - Supervisory campaign controller (Mode A: 34d, Mode B: 34d, Shutdown: 1d).
   - Metallurgical plant with dynamic mode resolution (Mode A, Mode B, Contingency, Surging).
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
    plot_safety_margin,
    plot_dual_axis_step,
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
    StochasticFaciesGenerator,
    MineFace,
    TruckPhase,
    Operator,
    DESTruck as Truck,
    SurfaceDumpStation,
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
FACE_LEVEL = 4
RAMP_M = (FACE_LEVEL - 1) * LEVEL_SPACING_M
LEVEL_DRIFT_M = _CFG.topology.level_drift_m
SURFACE_M = _CFG.topology.surface_m
SPEEDS = _CFG.topology.speeds

ORE_PAYLOAD = _CFG.fleet.truck_payload
TRUCK_LOAD_SPOT_MIN = _CFG.fleet.load_spot_min
LHD_ACQUISITION_MAX_MIN = _CFG.fleet.lhd_acquisition_max_min
TRUCK_LOAD_DUR_MIN = _CFG.fleet.load_dur_min
DUMP_SPOT_MIN = _CFG.fleet.dump_spot_min
DUMP_MIN = _CFG.fleet.dump_dur_min
SURFACE_TIP_SITES = _CFG.fleet.surface_tip_sites

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
# Single-Face Blending Modes Simulation Module
# ---------------------------------------------------------------------------
class ShelswellSingleFaceBlending(drs.Module):
    """Hybrid simulation combining Shelswell DES Truck-Loader Haulage with
    Single-Face Stochastic Parcel Geology and DRS Blending Modes.

    - Single MineFace produces geological parcels of 30k-50k tonnes.
    - Trucks haul ore from the face; during dumping, each payload is split
      into Ore 1 and Ore 2 stockpiles based on the active parcel's ore fraction.
    - MetallurgicalPlant & OperatingModeController run continuously on surface stockpiles.
    """

    def __init__(
        self,
        num_trucks: int = 18,
        num_operators: int = 18,
        num_lhds: int = 2,
        availability: float = 0.85,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
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
        self.num_lhds = num_lhds
        self.availability = availability
        self.target_ore_stock_level = target_ore_stock_level
        self.critical_ore2_level = critical_ore2_level
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = ore_to_be_extracted_during_warming_period

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

        # Global time tracker (in seconds)
        self.gt = drs.Timer("gt", 0.0, rate=1.0)

        # 1. Single Underground Mine Face with Stochastic Geology
        self.facies_gen = StochasticFaciesGenerator(
            mean_fraction=mean_ore_fraction,
            std_dev=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
        )
        self.mine_face = MineFace(
            name="mine_face",
            face_id=1,
            generator=self.facies_gen,
            min_ore_mass=min_ore_mass,
            max_ore_mass=max_ore_mass,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            mean_ore_fraction=mean_ore_fraction,
            std_dev_ore_fraction=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
            initial_parcel_mass=40000.0,
        )

        # 2. Continuous Surface Stockpiles (Tonnes)
        init_ore1 = (1.0 - mean_ore_fraction) * target_ore_stock_level  # 42,000 t
        init_ore2 = mean_ore_fraction * target_ore_stock_level  # 18,000 t
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=init_ore1,
            initial_attributes={
                "contained_ore_fraction_mass": init_ore1 * mean_ore_fraction
            },
            attr_inflow=1.0,
        )
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=init_ore2,
            initial_attributes={
                "contained_ore_fraction_mass": init_ore2 * mean_ore_fraction
            },
            attr_inflow=0.0,
        )
        self.total_extracted_ore = drs.Level("total_extracted_ore", 0.0)
        self.ore1_hauled = drs.Level("ore1_hauled", 0.0)
        self.ore2_hauled = drs.Level("ore2_hauled", 0.0)

        # 3. Plant & Campaign Controller
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

        # 5. Face Loadout & Surface Dump Station
        self.face_queue: List[Truck] = []
        self._lhds_busy = 0
        self.dump_station = SurfaceDumpStation()
        self._pumps_free = N_FUEL_PUMPS

        # Operational metrics
        self.daily_target_ore = 6000.0
        self.daily_hauled_ore = 0.0
        self.trips = 0
        self._cycle_sum = 0.0
        self.traffic_delay_sum = 0.0
        self.horizon_sec = float("inf")

        # History telemetry
        self.history_records: List[dict] = []

    # -- DRS Engine Hooks ----------------------------------------------------
    def is_terminating_condition_met(self) -> bool:
        if self.mine_face.cumulative_extracted_mass.value >= self.mine_face.total_ore_to_extract - 1e-6:
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

        # Step campaign and plant mode timers
        self.mode_controller.current_campaign_duration.step(dt_days)
        active_mode_name = self.plant.active_operating_mode.value.name
        timer_attr = self.plant._MODE_TIMER_ATTRS.get(active_mode_name)
        if timer_attr:
            getattr(self.plant, timer_attr).step(dt_days)
        if active_mode_name in self.plant._CONTINGENCY_MODES:
            self.plant.current_contingency_duration.step(dt_days)

        # Step trucks (timers, seat time, fuel)
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
        ore1_in_rate = self.dump_station._active_ore1_rate  # t/sec
        ore2_in_rate = self.dump_station._active_ore2_rate  # t/sec

        # Calculate plant draw rates based on campaign mode & stockpile levels
        plant_draw, _ = self.plant.get_target_rates(
            self.mode_controller.active_campaign_mode.value,
            ore1_level=self.ore1_stock.level,
            ore2_level=self.ore2_stock.level,
            stockpile2_routing_fraction=self.mine_face.active_parcel_ore_fraction.value,
        )

        ore1_draw_rate_sec = plant_draw.ore1 / 86400.0
        ore2_draw_rate_sec = plant_draw.ore2 / 86400.0

        # Feed stockpiles and draw into metallurgical plant
        out1 = self.ore1_stock.feed_and_draw(ore1_in_rate, ore1_draw_rate_sec)
        out2 = self.ore2_stock.feed_and_draw(ore2_in_rate, ore2_draw_rate_sec)
        self.ore1_stock.step(dt)
        self.ore2_stock.step(dt)

        self.plant.process(out1 + out2)
        self.plant.cumulative_milled_mass.step(dt)

        # Accumulators
        self.total_extracted_ore.step(dt)
        self.ore1_hauled.step(dt)
        self.ore2_hauled.step(dt)

        # Record telemetry record
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
        """Updates campaign mode, plant operating mode, and extraction target."""
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
            stockpile2_routing_fraction=self.mine_face.active_parcel_ore_fraction.value,
        )

        # 3. Derive Daily Extraction Target
        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self.daily_target_ore = 0.0
        elif "_MINE_SURGING" in mode_name:
            # Surging draw down: scale extraction down to allow stockpile reduction
            self.daily_target_ore = plant_draw.total * 0.70
        else:
            self.daily_target_ore = plant_draw.total

    def _calendar_update(self):
        t = self.gt.value
        day = int(t // 86400.0)
        if self._cur_day != day:
            self._cur_day = day
            self._holiday_today = day in self.holidays
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

        # Refuelling check
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

        # Progress of current 24-hr day
        day_progress = (t % 86400.0) / 86400.0
        expected_hauled_by_now = self.daily_target_ore * day_progress
        if total_stock >= self.target_ore_stock_level and self.daily_hauled_ore > expected_hauled_by_now + 100.0:
            self._release_operator(tr)
            return False

        if not self._acquire_operator(tr):
            self._release_operator(tr)
            return False

        tr.trip_start = self.gt.value
        tr.phase = TruckPhase.EMPTY
        tr.timer.value = self._travel_time(loaded=False)
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
        if self._lhds_busy >= self.num_lhds:
            self.face_queue.append(tr)
            tr.phase = TruckPhase.WAIT_LOAD
            tr.timer.value = 0.0
        else:
            self._lhds_busy += 1
            tr.phase = TruckPhase.SPOT_LOAD
            tr.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)

    def _finish_loading(self, tr: Truck):
        payload = _tri(self.rng, ORE_PAYLOAD, 0.08)
        tr.current_payload = payload

        # Capture active parcel ore fraction at loading
        tr.payload_ore_fraction = self.mine_face.active_parcel_ore_fraction.value

        # Advance MineFace parcel state
        self.mine_face.parcel_extracted_mass.value += payload
        self.mine_face.cumulative_extracted_mass.value += payload
        if (
            self.mine_face.parcel_extracted_mass.value
            >= self.mine_face.active_parcel_initial_mass.value
        ):
            self.mine_face._load_next_batch()
            self.mine_face.parcel_extracted_mass.value = 0.0

        self._lhds_busy -= 1
        if self.face_queue:
            nxt = self.face_queue.pop(0)
            self._lhds_busy += 1
            nxt.phase = TruckPhase.SPOT_LOAD
            nxt.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)

        tr.phase = TruckPhase.LOADED
        tr.timer.value = self._travel_time(loaded=True)

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

        # Split payload into continuous inflow rates for Ore 1 and Ore 2
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
    def _travel_time(self, loaded: bool) -> float:
        load_key = "loaded" if loaded else "empty"

        def seg(dist: float, kind: str) -> float:
            return dist / (SPEEDS[kind][load_key] / 3.6)

        t = (
            seg(SURFACE_M, "surface")
            + seg(DECLINE_M, "decline")
            + seg(RAMP_M, "ramp")
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

        n_waiting_load = len(self.face_queue)
        n_waiting_dump = len(self.dump_station.queue)
        n_refueling = sum(
            1 for tr in self.trucks if tr.phase == TruckPhase.REFUELING
        )
        n_operating = sum(
            1 for tr in self.trucks if tr.phase in OPERATING_PHASES
        )

        self.history_records.append(
            {
                "time": t_days,
                "ore1_stock": self.ore1_stock.level,
                "ore2_stock": self.ore2_stock.level,
                "Ore1Stock_mass": self.ore1_stock.level,
                "Ore2Stock_mass": self.ore2_stock.level,
                "total_system_ore_mass": self.ore1_stock.level
                + self.ore2_stock.level,
                "MassOfCurrentParcel": self.mine_face.active_parcel_initial_mass.value,
                "active_parcel_initial_mass": self.mine_face.active_parcel_initial_mass.value,
                "active_parcel_ore_fraction": self.mine_face.active_parcel_ore_fraction.value,
                "CurrentParcelRoutingFraction": self.mine_face.active_parcel_ore_fraction.value * 100.0,
                "Grade (% Ore 2)": self.mine_face.active_parcel_ore_fraction.value * 100.0,
                "active_operating_mode": self.plant.active_operating_mode.value,
                "active_operating_mode_name": active_mode,
                "campaign_mode": camp_mode,
                "current_campaign_duration": self.mode_controller.current_campaign_duration.value,
                "current_contingency_duration": self.plant.current_contingency_duration.value,
                "daily_target_ore": self.daily_target_ore,
                "daily_hauled_ore": self.daily_hauled_ore,
                "ore1_hauled_total": self.ore1_hauled.value,
                "ore2_hauled_total": self.ore2_hauled.value,
                "cumulative_extracted_mass": self.mine_face.cumulative_extracted_mass.value,
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


# ---------------------------------------------------------------------------
# Statistics & Visual Dashboards
# ---------------------------------------------------------------------------
def print_statistics(plant, mine):
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
        if hasattr(plant, "cumulative_milled_mass"):
            total_ore_processed = plant.cumulative_milled_mass.value
        else:
            total_ore_processed = mine.cumulative_extracted_mass.value
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


def print_simulation_statistics(sim: ShelswellSingleFaceBlending, df: pd.DataFrame):
    """Prints operational summary statistics."""
    total_days = sim.gt.value / 86400.0
    total_ore_hauled = sim.mine_face.cumulative_extracted_mass.value
    total_milled = sim.plant.cumulative_milled_mass.value
    active_days = sim.plant.active_duration(sim.plant.total_duration)

    print("\n" + "=" * 70)
    print(" SHELSWELL SINGLE-FACE DES + DRS BLENDING MODES SIMULATION RESULTS")
    print("=" * 70)
    print(f"Simulation Horizon:        {total_days:.1f} days")
    print(f"Total Trips Completed:     {sim.trips}")
    avg_cycle = (sim._cycle_sum / max(1, sim.trips)) / 60.0
    print(f"Average Truck Cycle Time:  {avg_cycle:.2f} min")
    print(f"Total Ore Hauled:          {total_ore_hauled:,.1f} t ({total_ore_hauled / max(1e-3, total_days):.1f} t/d)")
    print(f"  ↳ Ore 1 Equivalent:      {sim.ore1_hauled.value:,.1f} t ({sim.ore1_hauled.value / max(1e-3, total_days):.1f} t/d)")
    print(f"  ↳ Ore 2 Equivalent:      {sim.ore2_hauled.value:,.1f} t ({sim.ore2_hauled.value / max(1e-3, total_days):.1f} t/d)")
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


def plot_single_face_shelswell_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/shelswell_single_face_dashboard.png",
    palette: dict = None,
    figsize: Tuple[int, int] = (16, 48),
):
    """Builds and saves the 12-panel diagnostics dashboard."""
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if "active_operating_mode_name" not in df.columns or "Mode A" not in df.columns:
        df = prepare_history(df)

    dash = Dashboard(
        nrows=12,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title="Shelswell Single-Face DES + DRS Blending Modes Diagnostics",
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

    # 2. Current Parcel Properties: Mass (tons) and Grade (% Ore 2)
    plot_dual_axis_step(
        df,
        y1_col="MassOfCurrentParcel",
        y2_col="Grade (% Ore 2)",
        y1_label="Parcel Mass (tons)",
        y2_label="Grade (% Ore 2)",
        title="Current Parcel Properties",
        ax=dash[2],
    )

    # 3. Daily Targets vs Hauled Production
    plot_time_series(
        df,
        y_columns=[
            "daily_target_ore",
            "daily_hauled_ore",
        ],
        title="Daily Extraction Target vs Underground Hauled Production (t/d)",
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
# CLI & Runner
# ---------------------------------------------------------------------------
def run_shelswell_single_face_simulation(
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    num_lhds: int = 2,
    availability: float = 0.85,
    target_ore_stock_level: float = 60000.0,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[ShelswellSingleFaceBlending, pd.DataFrame]:
    """Builds and runs the single face hybrid simulation with two-phase execution."""
    sim = ShelswellSingleFaceBlending(
        num_trucks=num_trucks,
        num_operators=num_operators,
        num_lhds=num_lhds,
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
        sim.mine_face.total_ore_to_extract = ore_to_be_extracted_during_warming_period
        engine.run(until=float("inf"))

        # Reset plant operating mode duration timers for production metrics
        sim.plant.reset_mode_timers()

        # Phase 2: Production Measurement Phase
        sim.mine_face.total_ore_to_extract = total_ore_to_extract
        engine.run(until=float("inf"))

    df = pd.DataFrame(sim.history_records)
    print_simulation_statistics(sim, df)
    print_statistics(sim.plant, sim.mine_face)

    df_prepared = prepare_history(df)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim.critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        label="Shelswell Single-Face Blending",
    )
    print_deficit_by_mode(
        df_prepared,
        extraction_cols=["cumulative_extracted_mass"],
        ideal_rate=6000.0,
    )

    if plot and len(df_prepared) > 0:
        plot_single_face_shelswell_dashboard(df_prepared)
    return sim, df_prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Shelswell Single-Face DES + DRS Blending Modes Simulation"
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
        help="Number of AD30 haulage trucks (default: 18 for ~6,000 t/d mill)",
    )
    parser.add_argument(
        "--operators",
        type=int,
        default=18,
        help="Number of operators per shift (default: 18)",
    )
    parser.add_argument(
        "--lhds",
        type=int,
        default=2,
        help="Number of LHD loaders at the single face (default: 2)",
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

    run_shelswell_single_face_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        num_lhds=args.lhds,
        availability=args.availability,
        target_ore_stock_level=args.stockpile_target,
        seed=args.seed,
        plot=not args.no_plot,
    )

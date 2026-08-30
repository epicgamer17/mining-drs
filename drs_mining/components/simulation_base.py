"""Base Orchestration Engine for Two-Area Underground Mining Simulations.

Encapsulates discrete-event haulage state transitions, vehicle queues,
operator shift scheduling, maintenance windows, plant processing,
and telemetry recording, while delegating physical travel dynamics to MineTopology,
muck extraction & readiness to MineFace, and metallurgical recovery to MetallurgicalPlant.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import drs
import pandas as pd

from drs_mining.config import (
    MILL_MODES,
    FLEET_MODES,
    EconomicParameters,
    CalendarConfig,
    TopologyConfig,
    HaulageFleetConfig,
    PlantConfig,
    GeologyConfig,
    StrategicPlanningConfig,
    SimulationConfig,
)
from drs_mining.components.modes import OperatingMode
from drs_mining.components.plant import MetallurgicalPlant, PlantDrawRates
from drs_mining.components.stockpiles import Stockpile
from drs_mining.components.controllers import OperatingModeController
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.mine_face import MineFace, FaceState
from drs_mining.components.topology import MineTopology, DEFAULT_SPEEDS
from drs_mining.components.planning import (
    AreaReadinessTarget,
    StrategicYearTarget,
    strategic_target_for_year,
    trajectory_progress_ratio,
    select_fleet_mode,
    TacticalReviewController,
)
from drs_mining.components.discrete_fleet import (
    TruckPhase,
    Operator,
    DESTruck as Truck,
    SurfaceDumpStation,
    OPERATING_PHASES,
    SEAT_PHASES,
    DUE_PHASES,
)

# ---------------------------------------------------------------------------
# Default Constants (Mapped to Centralized Configs)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = SimulationConfig()
DAYS_IN_YEAR = _DEFAULT_CONFIG.calendar.days_in_year
NON_PRODUCTION_DAYS = _DEFAULT_CONFIG.calendar.non_production_days
SHIFT_SECONDS = _DEFAULT_CONFIG.calendar.shift_seconds
SHIFT_WORK_HOURS = _DEFAULT_CONFIG.calendar.shift_work_hours
HAULAGE_SEAT_FRACTION = _DEFAULT_CONFIG.calendar.haulage_seat_fraction
SEAT_PER_SHIFT_SEC = _DEFAULT_CONFIG.calendar.seat_per_shift_sec

DECLINE_M = _DEFAULT_CONFIG.topology.decline_m
LEVEL_SPACING_M = _DEFAULT_CONFIG.topology.level_spacing_m
AREA1_LEVEL = _DEFAULT_CONFIG.topology.area1_level
AREA2_LEVEL = _DEFAULT_CONFIG.topology.area2_level
LEVEL_DRIFT_M = _DEFAULT_CONFIG.topology.level_drift_m
SURFACE_M = _DEFAULT_CONFIG.topology.surface_m

ORE_PAYLOAD = _DEFAULT_CONFIG.fleet.truck_payload
TRUCK_LOAD_SPOT_MIN = _DEFAULT_CONFIG.fleet.load_spot_min
LHD_ACQUISITION_MAX_MIN = _DEFAULT_CONFIG.fleet.lhd_acquisition_max_min
TRUCK_LOAD_DUR_MIN = _DEFAULT_CONFIG.fleet.load_dur_min
DUMP_SPOT_MIN = _DEFAULT_CONFIG.fleet.dump_spot_min
DUMP_MIN = _DEFAULT_CONFIG.fleet.dump_dur_min
SURFACE_TIP_SITES = _DEFAULT_CONFIG.fleet.surface_tip_sites

FUEL_BURN_PCT_PER_SEC = _DEFAULT_CONFIG.fleet.fuel_burn_pct_per_sec
REFUEL_DUR_MIN = _DEFAULT_CONFIG.fleet.refuel_dur_min
N_FUEL_PUMPS = _DEFAULT_CONFIG.fleet.num_fuel_pumps
BASE_PASS_BAY_DELAY_SEC = _DEFAULT_CONFIG.fleet.base_pass_bay_delay_sec
PER_TRUCK_PASS_BAY_DELAY_SEC = _DEFAULT_CONFIG.fleet.per_truck_pass_bay_delay_sec

DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY = _DEFAULT_CONFIG.fleet.dev_m_per_extra_truck_day
DT_MAX = _DEFAULT_CONFIG.dt_max


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
# Base Two-Area Simulation Engine
# ---------------------------------------------------------------------------
class TwoAreaSimulationBase(drs.Module):
    """Orchestrator for two-area discrete event haulage, plant processing,
    tactical fleet modes, and economic cash-flow tracking.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
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
        self.config = config or _DEFAULT_CONFIG
        self.num_trucks = num_trucks
        self.num_operators = num_operators
        self.num_lhds_per_face = num_lhds_per_face
        self.availability = availability
        self.target_ore_stock_level = target_ore_stock_level
        self.critical_ore2_level = critical_ore2_level
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
        )

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

        # 1. Physical Topology
        self.topology = MineTopology(
            decline_m=DECLINE_M,
            level_spacing_m=LEVEL_SPACING_M,
            level_drift_m=LEVEL_DRIFT_M,
            surface_m=SURFACE_M,
            level_depths={AREA1_LEVEL: 900.0, AREA2_LEVEL: 1800.0},
            speeds=DEFAULT_SPEEDS,
            base_pass_bay_delay_sec=BASE_PASS_BAY_DELAY_SEC,
            per_truck_pass_bay_delay_sec=PER_TRUCK_PASS_BAY_DELAY_SEC,
        )

        # 2. Dual Mine Faces
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
            required_development=0.0,
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
            required_development=self.area2_readiness_target.required_development,
            ready_by_day=self.area2_readiness_target.ready_by_day,
            counterfactual_disable=self.area2_counterfactual_disable,
            on_unlock_callback=self._on_area2_unlocked,
        )

        # 3. Stockpiles
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["ore_grade"],
            initial_mass=self.target_ore_stock_level * 0.50,
            initial_attributes={"ore_grade": 0.0},
            capacity=120000.0,
        )
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["ore_grade"],
            initial_mass=self.target_ore_stock_level * 0.50,
            initial_attributes={"ore_grade": 1.0},
            capacity=120000.0,
        )

        # 4. Plant & Campaign Mode Controller
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
            duration_of_contingency_segments=duration_of_contingency_segments / 24.0,
            mode_a_ore1_milling_rate=mode_a_ore1_milling_rate,
            mode_a_ore2_milling_rate=mode_a_ore2_milling_rate,
            mode_a_contingency_ore1_milling_rate=mode_a_contingency_ore1_milling_rate,
            mode_b_ore1_milling_rate=mode_b_ore1_milling_rate,
            mode_b_ore2_milling_rate=mode_b_ore2_milling_rate,
            mode_b_contingency_ore2_milling_rate=mode_b_contingency_ore2_milling_rate,
            economic_params=EconomicParameters(
                annual_discount_rate=annual_discount_rate,
                ore1_net_value_per_processed_tonne=ore1_net_value_per_processed_tonne,
                ore2_net_value_per_processed_tonne=ore2_net_value_per_processed_tonne,
                production_cost_per_tonne=production_cost_per_tonne,
                development_cost_per_unit=development_cost_per_unit,
                fixed_cost_per_day=fixed_cost_per_day,
            ),
        )

        # 5. Discrete Entities
        self.dump_station = SurfaceDumpStation(capacity=SURFACE_TIP_SITES)
        self.face_queues: Dict[int, list] = {1: [], 2: []}
        self.face_lhds_busy: Dict[int, int] = {1: 0, 2: 0}

        self.trucks = [
            Truck(f"T{i+1:02d}", drs.Timer(f"tr_tmr_{i}", 0.0, rate=-1.0))
            for i in range(num_trucks)
        ]
        self.operators = [Operator(idx=i) for i in range(num_operators)]
        self._refuel_pumps_in_use = 0
        self._refuel_queue: list = []

        # 6. Strategic / Tactical Planning Controller
        self.tactical_controller = TacticalReviewController(
            strategic_targets=self.strategic_targets,
            strategic_period_days=strategic_period_days,
            tactical_review_period_days=tactical_review_period_days,
            tactical_progress_tolerance=tactical_progress_tolerance,
            development_priority_truck_reservation_fraction=development_priority_truck_reservation_fraction,
        )

        # Component References
        self.readiness_tracker = self.face2
        self.economics = self.plant

        # Counters & Telemetry
        self.ore1_dumped_total = drs.Level("ore1_dumped_total", 0.0)
        self.ore2_dumped_total = drs.Level("ore2_dumped_total", 0.0)
        self.tonnes_hauled_by_face = {1: 0.0, 2: 0.0}
        self.ore1_hauled_by_face = {1: 0.0, 2: 0.0}
        self.ore2_hauled_by_face = {1: 0.0, 2: 0.0}

        self.telemetry_history: List[Dict[str, Any]] = []
        self._telemetry_dt = 1800.0  # Log every 30 minutes
        self._next_telemetry_t = 0.0

    def _on_area2_unlocked(self, day: float = 0.0) -> None:
        """Callback triggered when Area 2 physical development reaches 100%."""
        pass

    def is_area2_locked(self, current_day: float = 0.0) -> bool:
        """Evaluates whether Area 2 / Face 2 is currently locked."""
        if not self.area2_physical_unlock_enabled:
            return False
        return self.face2.is_locked(current_day)

    # -----------------------------------------------------------------------
    # Operator & Shift Mechanics
    # -----------------------------------------------------------------------
    def _acquire_operator(self, tr: Truck) -> bool:
        if tr.operator >= 0:
            return True
        for op in self.operators:
            if op.free and op.used_seat < self.truck_seat_credit:
                op.free = False
                tr.operator = op.idx
                return True
        return False

    def _release_operator(self, tr: Truck) -> None:
        if tr.operator >= 0:
            self.operators[tr.operator].free = True
            tr.operator = -1

    def _schedule_down_window(self, tr: Truck) -> None:
        if self._down_dur <= 0.0:
            tr.down_start = math.inf
            tr.down_end = math.inf
            return
        shift_num = int(self.gt.value // SHIFT_SECONDS)
        base = shift_num * SHIFT_SECONDS
        window_open = base
        window_close = max(
            window_open, base + SHIFT_WORK_HOURS * 3600.0 - self._down_dur
        )
        tr.down_start = self.rng.uniform(window_open, window_close)
        tr.down_end = tr.down_start + self._down_dur

    def _in_down_window(self, tr: Truck, t: float) -> bool:
        return tr.down_start <= t < tr.down_end

    def _travel_time(self, tr: Truck, loaded: bool) -> float:
        active_count = sum(
            1 for x in self.trucks if x.phase in OPERATING_PHASES and x.phase != TruckPhase.IDLE
        )
        return self.topology.calculate_travel_time_sec(
            level=tr.target_level,
            loaded=loaded,
            active_truck_count=active_count,
            rng=self.rng,
        )

    # -----------------------------------------------------------------------
    # Pluggable Dispatch Hook
    # -----------------------------------------------------------------------
    def select_face_for_truck(self, tr: Truck) -> int:
        """Default heuristic dispatch hook: balance ore supply against plant setpoint."""
        a2_locked = self.is_area2_locked(self.gt.value / 86400.0)
        if a2_locked:
            return 1

        # Heuristic dispatch based on mode requirement
        target_mode = self.plant.active_operating_mode.value.name
        o1_draw = getattr(self.plant, "mode_a_ore1_milling_rate", 3600.0)
        o2_draw = getattr(self.plant, "mode_a_ore2_milling_rate", 2400.0)

        # Mode A targets 40% Ore 2 (hauled 83.3% from Face 2)
        if "MODE_A" in target_mode:
            return 2 if self.rng.random() < 0.833 else 1
        # Mode B targets ~15% Ore 2 (hauled primarily from Face 1)
        elif "MODE_B" in target_mode:
            return 1 if self.rng.random() < 0.70 else 2
        return 1

    # -----------------------------------------------------------------------
    # Discrete State Machine Flow
    # -----------------------------------------------------------------------
    def _try_dispatch(self, tr: Truck) -> bool:
        t = self.gt.value
        if not _in_shift_window(t) or self._holiday_today or self._in_down_window(tr, t):
            return False
        if not self._acquire_operator(tr):
            return False

        # Development reservation check
        active_fleet_mode = self.tactical_controller.active_fleet_mode
        if active_fleet_mode == FLEET_MODES["DEVELOPMENT"] and self.is_area2_locked(t / 86400.0):
            pass

        target_face_id = self.select_face_for_truck(tr)
        tr.target_face_id = target_face_id
        tr.target_level = AREA2_LEVEL if target_face_id == 2 else AREA1_LEVEL
        tr.phase = TruckPhase.EMPTY
        dur = self._travel_time(tr, loaded=False)
        tr.trip_start = t
        tr.timer.value = dur
        tr.timer.rate = -1.0
        return True

    def _enter_face_loadout(self, tr: Truck, face_id: int) -> None:
        tr.phase = TruckPhase.WAIT_LOAD
        tr.timer.rate = 0.0
        self.face_queues[face_id].append(tr)
        self._service_face_queue(face_id)

    def _service_face_queue(self, face_id: int) -> None:
        q = self.face_queues[face_id]
        while q and self.face_lhds_busy[face_id] < self.num_lhds_per_face:
            nxt = q.pop(0)
            self.face_lhds_busy[face_id] += 1
            nxt.phase = TruckPhase.SPOT_LOAD
            spot = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.15)
            nxt.timer.value = spot
            nxt.timer.rate = -1.0

    def _finish_loading(self, tr: Truck) -> None:
        t = self.gt.value
        face = self.face2 if tr.target_face_id == 2 else self.face1
        ore2_frac = face.current_ore_grade
        tr.current_payload = ORE_PAYLOAD
        tr.payload_ore_fraction = ore2_frac

        # Update face extraction counters
        face.cumulative_extracted_mass.value += ORE_PAYLOAD
        self.tonnes_hauled_by_face[tr.target_face_id] += ORE_PAYLOAD
        self.ore1_hauled_by_face[tr.target_face_id] += ORE_PAYLOAD * (1.0 - ore2_frac)
        self.ore2_hauled_by_face[tr.target_face_id] += ORE_PAYLOAD * ore2_frac

        # Free loader
        self.face_lhds_busy[tr.target_face_id] = max(
            0, self.face_lhds_busy[tr.target_face_id] - 1
        )
        self._service_face_queue(tr.target_face_id)

        # Dispatch loaded haul
        tr.phase = TruckPhase.LOADED
        dur = self._travel_time(tr, loaded=True)
        tr.timer.value = dur
        tr.timer.rate = -1.0

    def _enter_dump(self, tr: Truck) -> None:
        tr.phase = TruckPhase.WAIT_DUMP
        tr.timer.rate = 0.0
        self.dump_station.queue.append(tr)
        self._service_dump_queue()

    def _service_dump_queue(self) -> None:
        while (
            self.dump_station.queue
            and self.dump_station.in_use < self.dump_station.capacity
        ):
            nxt = self.dump_station.queue.pop(0)
            self._start_dump(nxt)

    def _start_dump(self, tr: Truck) -> None:
        self.dump_station.in_use += 1
        tr.phase = TruckPhase.DUMPING
        spot = _tri(self.rng, DUMP_SPOT_MIN * 60.0, 0.15)
        dump = _tri(self.rng, DUMP_MIN * 60.0, 0.15)
        tr.dump_dur = spot + dump
        tr.timer.value = tr.dump_dur
        tr.timer.rate = -1.0

        # Continuous dump flow rates into stockpiles
        o2_frac = tr.payload_ore_fraction
        r_ore1 = (tr.current_payload * (1.0 - o2_frac)) / tr.dump_dur
        r_ore2 = (tr.current_payload * o2_frac) / tr.dump_dur
        self.dump_station._active_ore1_rate += r_ore1
        self.dump_station._active_ore2_rate += r_ore2

    def _finish_dumping(self, tr: Truck) -> None:
        t = self.gt.value
        o2_frac = tr.payload_ore_fraction
        r_ore1 = (tr.current_payload * (1.0 - o2_frac)) / tr.dump_dur
        r_ore2 = (tr.current_payload * o2_frac) / tr.dump_dur
        self.dump_station._active_ore1_rate = max(
            0.0, self.dump_station._active_ore1_rate - r_ore1
        )
        self.dump_station._active_ore2_rate = max(
            0.0, self.dump_station._active_ore2_rate - r_ore2
        )

        self.ore1_dumped_total.value += tr.current_payload * (1.0 - o2_frac)
        self.ore2_dumped_total.value += tr.current_payload * o2_frac
        self.ore1_stock.current_mass.value += tr.current_payload * (1.0 - o2_frac)
        self.ore2_stock.current_mass.value += tr.current_payload * o2_frac

        self.dump_station.in_use = max(0, self.dump_station.in_use - 1)
        self._service_dump_queue()

        tr.current_payload = 0.0
        self._release_operator(tr)

        # Check refueling requirement
        if tr.fuel < tr.refuel_threshold:
            self._enter_refuel(tr)
        else:
            tr.phase = TruckPhase.IDLE
            tr.timer.rate = 0.0
            self._try_dispatch(tr)

    def _enter_refuel(self, tr: Truck) -> None:
        tr.phase = TruckPhase.REFUELING
        if self._refuel_pumps_in_use < N_FUEL_PUMPS:
            self._refuel_pumps_in_use += 1
            dur = _tri(self.rng, REFUEL_DUR_MIN * 60.0, 0.15)
            tr.timer.value = dur
            tr.timer.rate = -1.0
        else:
            tr.timer.rate = 0.0
            self._refuel_queue.append(tr)

    def _finish_refuel(self, tr: Truck) -> None:
        tr.fuel = 100.0
        self._refuel_pumps_in_use = max(0, self._refuel_pumps_in_use - 1)
        if self._refuel_queue:
            nxt = self._refuel_queue.pop(0)
            self._refuel_pumps_in_use += 1
            dur = _tri(self.rng, REFUEL_DUR_MIN * 60.0, 0.15)
            nxt.timer.value = dur
            nxt.timer.rate = -1.0

        tr.phase = TruckPhase.IDLE
        tr.timer.rate = 0.0
        self._try_dispatch(tr)

    # -----------------------------------------------------------------------
    # Calendar & Tactical Progress Updates
    # -----------------------------------------------------------------------
    def _calendar_update(self, t: float) -> None:
        day = int(t // 86400.0)
        if day != self._cur_day:
            self._cur_day = day
            self._holiday_today = (day % 365) in self.holidays

            # Tactical review update
            daily_dev = self._compute_daily_development_meters()
            a2_locked = self.is_area2_locked(day)
            area2_dev = daily_dev if a2_locked else 0.0

            # Advance face 2 physical development
            self.face2.advance_development(area2_dev, current_day=float(day))

            # Step economics daily on MetallurgicalPlant
            self.plant.step_daily_economics(
                current_day=float(day),
                ore1_mined_t=self.ore1_dumped_total.value,
                ore2_mined_t=self.ore2_dumped_total.value,
                development_units=float(self.face2.cumulative_development.value),
            )

            # Tactical review monthly check
            self.tactical_controller.step_daily_tactical_review(
                current_day=float(day),
                cum_development=float(self.face2.cumulative_development.value),
                cum_ore1=float(self.ore1_dumped_total.value),
                cum_ore2=float(self.ore2_dumped_total.value),
                area2_readiness_tracker=self.face2,
                total_trucks=self.num_trucks,
            )

        shift = int(t // SHIFT_SECONDS)
        if shift != self._shift_marker:
            self._shift_marker = shift
            for tr in self.trucks:
                tr.seat_used = 0.0
                self._schedule_down_window(tr)
            for op in self.operators:
                op.used_seat = 0.0

    def _compute_daily_development_meters(self) -> float:
        """Calculates development meters advanced per day based on active fleet allocation."""
        base_dev_per_day = 10.0
        active_mode = self.tactical_controller.active_fleet_mode

        if active_mode == FLEET_MODES["DEVELOPMENT"]:
            extra_trucks = int(self.num_trucks * self.development_priority_truck_reservation_fraction)
            return base_dev_per_day + extra_trucks * DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY
        elif active_mode == FLEET_MODES["BALANCED"]:
            return base_dev_per_day
        else:  # PRODUCTION
            return base_dev_per_day * 0.50

    # -----------------------------------------------------------------------
    # Continuous Integration & DES Stepping
    # -----------------------------------------------------------------------
    def time_to_event(self) -> float:
        min_dt = DT_MAX
        t = self.gt.value

        for tr in self.trucks:
            if tr.timer.rate < 0 and tr.timer.value > 0:
                min_dt = min(min_dt, tr.timer.value)
            if tr.down_start > t and tr.down_start != math.inf:
                min_dt = min(min_dt, tr.down_start - t)
            if tr.down_end > t and tr.down_end != math.inf:
                min_dt = min(min_dt, tr.down_end - t)

        # Telemetry interval
        if self._next_telemetry_t > t:
            min_dt = min(min_dt, self._next_telemetry_t - t)

        return max(1e-6, min_dt)

    def _advance(self, t_target: float) -> None:
        dt = t_target - self.gt.value
        if dt <= 0:
            return

        self._calendar_update(self.gt.value)

        # Update plant campaign mode & draw rates
        campaign_mode = self.mode_controller.update(
            ore2_stock_level=self.ore2_stock.current_mass.value,
            total_stock_level=self.ore1_stock.current_mass.value + self.ore2_stock.current_mass.value,
        )
        draw_rates, _ = self.plant.determine_operating_mode(
            campaign_mode=campaign_mode,
            ore1_level=self.ore1_stock.current_mass.value,
            ore2_level=self.ore2_stock.current_mass.value,
        )

        # Draw rates into plant (tonnes/day -> tonnes/sec)
        o1_rate_sec = draw_rates.ore1 / 86400.0
        o2_rate_sec = draw_rates.ore2 / 86400.0

        actual_o1_draw = min(self.ore1_stock.current_mass.value, o1_rate_sec * dt)
        actual_o2_draw = min(self.ore2_stock.current_mass.value, o2_rate_sec * dt)

        self.ore1_stock.current_mass.value -= actual_o1_draw
        self.ore2_stock.current_mass.value -= actual_o2_draw
        self.plant.step_metallurgical_accounting(actual_o1_draw, actual_o2_draw)

        # Fuel burn & seat hours & timers
        for tr in self.trucks:
            if tr.timer.rate < 0 and tr.timer.value > 0.0:
                tr.timer.value = max(0.0, tr.timer.value - dt)
            if tr.phase in OPERATING_PHASES:
                tr.fuel = max(0.0, tr.fuel - FUEL_BURN_PCT_PER_SEC * dt)
            if tr.phase in SEAT_PHASES and tr.operator >= 0:
                tr.seat_used += dt
                self.operators[tr.operator].used_seat += dt

        self.gt.value = t_target

    def step(self, dt: float) -> None:
        t_end = self.gt.value + dt
        while self.gt.value < t_end:
            tte = min(self.time_to_event(), t_end - self.gt.value)
            t_next = self.gt.value + tte
            self._advance(t_next)
            self.on_event(t_next)

    def on_event(self, t: float) -> None:
        # Check truck phase transitions
        for tr in self.trucks:
            if tr.timer.rate < 0 and tr.timer.value <= 1e-6:
                tr.timer.value = 0.0
                tr.timer.rate = 0.0

                if tr.phase == TruckPhase.EMPTY:
                    self._enter_face_loadout(tr, tr.target_face_id)
                elif tr.phase == TruckPhase.SPOT_LOAD:
                    tr.phase = TruckPhase.ACQUIRE
                    acq = _tri(self.rng, LHD_ACQUISITION_MAX_MIN * 30.0, 0.20)
                    tr.timer.value = acq
                    tr.timer.rate = -1.0
                elif tr.phase == TruckPhase.ACQUIRE:
                    tr.phase = TruckPhase.LOADING
                    ld = _tri(self.rng, TRUCK_LOAD_DUR_MIN * 60.0, 0.15)
                    tr.timer.value = ld
                    tr.timer.rate = -1.0
                elif tr.phase == TruckPhase.LOADING:
                    self._finish_loading(tr)
                elif tr.phase == TruckPhase.LOADED:
                    self._enter_dump(tr)
                elif tr.phase == TruckPhase.DUMPING:
                    self._finish_dumping(tr)
                elif tr.phase == TruckPhase.REFUELING:
                    self._finish_refuel(tr)

        # Dispatch idle trucks
        for tr in self.trucks:
            if tr.phase == TruckPhase.IDLE:
                self._try_dispatch(tr)

        # Record periodic telemetry
        if t >= self._next_telemetry_t:
            self._record_telemetry(t)
            self._next_telemetry_t = t + self._telemetry_dt

    @property
    def area2_ready(self) -> bool:
        """Compatibility property: True if Area 2 is unlocked and ready for extraction."""
        return not self.is_area2_locked(self.gt.value / 86400.0)

    @property
    def area2_ready_day(self) -> drs.Level:
        """Compatibility property: Level holding the day Area 2 was unlocked."""
        return self.face2.ready_day

    @property
    def area2_cumulative_development(self) -> drs.Level:
        """Compatibility property: Level holding cumulative development metres."""
        return self.face2.cumulative_development

    @property
    def area2_readiness_fraction(self) -> drs.Level:
        """Compatibility property: Level holding readiness fraction."""
        return self.face2.readiness_fraction

    @property
    def strategic_year_timer(self) -> drs.Timer:
        """Compatibility property: Timer tracking the annual strategic cycle."""
        return self.tactical_controller.strategic_year_timer

    @property
    def area2_deadline_missed(self) -> bool:
        return self.face2.deadline_missed

    @property
    def area2_currently_late(self) -> bool:
        return self.face2.currently_late

    @property
    def area2_completed_late(self) -> bool:
        return self.face2.completed_late

    @property
    def strategic_planning_started(self) -> bool:
        return self.tactical_controller.planning_started

    @strategic_planning_started.setter
    def strategic_planning_started(self, value: bool) -> None:
        self.tactical_controller.planning_started = value

    def _update_area2_readiness(self) -> bool:
        """Helper to update readiness status at current strategic timer day."""
        t_day = self.tactical_controller.strategic_year_timer.value
        return self.face2.update_status(t_day)

    def _select_face_by_blend_need(self) -> int:
        """Compatibility method for tests."""
        return self.select_face_for_truck(self.trucks[0])

    def _record_telemetry(self, t: float) -> None:
        day = t / 86400.0
        mode_name = self.plant.active_operating_mode.value.name
        o1_stock = self.ore1_stock.current_mass.value
        o2_stock = self.ore2_stock.current_mass.value
        tot_stock = o1_stock + o2_stock
        tot_mined = self.ore1_dumped_total.value + self.ore2_dumped_total.value
        p_o1 = self.plant.cumulative_processed_ore1.value
        p_o2 = self.plant.cumulative_processed_ore2.value
        tot_processed = p_o1 + p_o2
        a2_locked = self.is_area2_locked(day)

        record = {
            "time_sec": t,
            "time": day,
            "day": day,
            "ore1_stockpile": o1_stock,
            "ore2_stockpile": o2_stock,
            "total_stockpile": tot_stock,
            "Ore1Stock_mass": o1_stock,
            "Ore2Stock_mass": o2_stock,
            "total_system_ore_mass": tot_stock,
            "mill_mode": mode_name,
            "active_operating_mode_name": mode_name,
            "fleet_mode": self.tactical_controller.active_fleet_mode.name,
            "Mode A": 1.0 if "MODE_A" in mode_name else 0.0,
            "Mode B": 1.0 if "MODE_B" in mode_name else 0.0,
            "Shutdown": 1.0 if "SHUTDOWN" in mode_name else 0.0,
            "ore1_mined": self.ore1_dumped_total.value,
            "ore2_mined": self.ore2_dumped_total.value,
            "total_mined": tot_mined,
            "Ore1_Mined": self.ore1_dumped_total.value,
            "Ore2_Mined": self.ore2_dumped_total.value,
            "ore1_processed": p_o1,
            "ore2_processed": p_o2,
            "total_processed": tot_processed,
            "cumulative_milled_mass": tot_processed,
            "cumulative_development": float(self.face2.cumulative_development.value),
            "cumulative_mine_development": float(self.face2.cumulative_development.value),
            "area2_cumulative_development": float(self.face2.cumulative_development.value),
            "area2_readiness_fraction": float(self.face2.readiness_fraction.value),
            "area2_trajectory_ratio": float(self.face2.readiness_trajectory_ratio.value),
            "area2_ready_day": float(self.face2.ready_day.value),
            "area2_is_locked": float(a2_locked),
            "area2_ready": not a2_locked,
            "cumulative_cash_flow": self.plant.cumulative_cash_flow,
            "cumulative_npv": self.plant.cumulative_npv.value,
            "operating_npv_proxy": self.plant.cumulative_npv.value,
            "cumulative_discounted_cash_flow": self.plant.cumulative_npv.value,
            "discount_factor": self.plant.discount_factor,
            "current_cash_flow_rate": self.plant.daily_net_cash_flow,
            "daily_revenue": self.plant.daily_revenue,
            "daily_cost": self.plant.daily_cost,
            "daily_net_cash_flow": self.plant.daily_net_cash_flow,
            "current_campaign_duration": self.mode_controller.current_campaign_duration.value,
            "current_contingency_duration": getattr(self.plant, "current_contingency_duration", drs.Timer("cc", 0)).value if hasattr(getattr(self.plant, "current_contingency_duration", None), "value") else 0.0,
            "cumulative_time_shutdown": self.plant.cumulative_time_shutdown.value,
            "cumulative_time_mode_a": self.plant.cumulative_time_mode_a.value,
            "cumulative_time_mode_b": self.plant.cumulative_time_mode_b.value,
            "trucks_idle": sum(1 for tr in self.trucks if tr.phase == TruckPhase.IDLE),
            "trucks_operating": sum(
                1 for tr in self.trucks if tr.phase in OPERATING_PHASES and tr.phase != TruckPhase.IDLE
            ),
        }
        self.telemetry_history.append(record)

    def results(self) -> Dict[str, Any]:
        """Returns structured simulation summary outputs and time-series dataframe."""
        df = pd.DataFrame(self.telemetry_history)
        return {
            "df": df,
            "total_mined_tonnes": self.ore1_dumped_total.value + self.ore2_dumped_total.value,
            "ore1_mined_tonnes": self.ore1_dumped_total.value,
            "ore2_mined_tonnes": self.ore2_dumped_total.value,
            "cumulative_development_m": self.face2.cumulative_development,
            "cumulative_npv": self.plant.cumulative_npv,
            "area2_ready_day": self.face2.ready_day,
            "area2_completed_late": self.face2.completed_late,
        }

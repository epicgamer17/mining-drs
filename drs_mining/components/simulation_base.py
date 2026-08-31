"""Base Orchestration Engine for Two-Area Underground Mining Simulations.

Encapsulates discrete-event haulage state transitions, vehicle queues,
operator shift scheduling, maintenance windows, plant processing,
and telemetry recording, while delegating physical travel dynamics to MineTopology,
muck extraction & readiness to MineFace, and metallurgical recovery to MetallurgicalPlant.
"""

from collections import defaultdict
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
    DEFAULT_CONFIG,
)
from drs_mining.components.modes import OperatingMode
from drs_mining.components.plant import MetallurgicalPlant, PlantDrawRates
from drs_mining.components.stockpiles import Stockpile
from drs_mining.components.controllers import OperatingModeController
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.mine_face import MineFace, FaceState
from drs_mining.components.topology import MineTopology, RoadSegment, PassingBay, DEFAULT_SPEEDS
from drs_mining.components.planning import (
    AreaReadinessTarget,
    StrategicYearTarget,
    strategic_target_for_year,
    trajectory_progress_ratio,
    select_fleet_mode,
    TacticalReviewController,
)
from drs_mining.components.fleet_controller import (
    FleetOperatingMode,
    FleetModeController,
)
from drs_mining.components.fleet import (
    TruckPhase,
    MissionType,
    Operator,
    Truck,
    SurfaceDumpStation,
    SurfaceWasteDumpStation,
    OPERATING_PHASES,
    SEAT_PHASES,
    DUE_PHASES,
)

# ---------------------------------------------------------------------------
# Default Constants (Mapped to Centralized Configs)
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = DEFAULT_CONFIG
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
BASE_PASS_BAY_DELAY_SEC = _DEFAULT_CONFIG.topology.base_pass_bay_delay_sec
PER_TRUCK_PASS_BAY_DELAY_SEC = _DEFAULT_CONFIG.topology.per_truck_pass_bay_delay_sec

DEVELOPMENT_METRES_PER_EXTRA_TRUCK_PER_DAY = _DEFAULT_CONFIG.fleet.dev_m_per_extra_truck_day
DT_MAX = _DEFAULT_CONFIG.dt_max


def _tri(rng: random.Random, mid: float, tol: float) -> float:
    """Symmetric triangular distribution around ``mid`` with width ``tol``."""
    return rng.triangular(mid * (1.0 - tol), mid * (1.0 + tol), mid)


def _in_shift_window(t: float) -> bool:
    """Continuous 12.0 h shifts across 24/7 operations without artificial off-shift gaps."""
    return True


# ---------------------------------------------------------------------------
# Base Mining Simulation Engine
# ---------------------------------------------------------------------------
class MiningSimulationBase(drs.Module):
    """General discrete-event mining simulation orchestrator.
    
    Coordinates multi-face haulage, discrete truck fleets, physical decline topology,
    stockpiles, processing plant campaigns, and financial accounting.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        faces: Optional[Sequence[MineFace]] = None,
        topology: Optional[MineTopology] = None,
        num_trucks: Optional[int] = None,
        num_operators: Optional[int] = None,
        num_lhds_per_face: Optional[int] = None,
        availability: Optional[float] = None,
        target_ore_stock_level: Optional[float] = None,
        critical_ore2_level: Optional[float] = None,
        total_ore_to_extract: Optional[float] = None,
        ore_to_be_extracted_during_warming_period: Optional[float] = None,
        duration_of_production_campaigns: Optional[float] = None,
        duration_of_shutdowns: Optional[float] = None,
        duration_of_contingency_segments: Optional[float] = None,
        strategic_period_days: Optional[float] = None,
        tactical_review_period_days: Optional[float] = None,
        tactical_progress_tolerance: Optional[float] = None,
        strategic_targets: Optional[Tuple[StrategicYearTarget, ...]] = None,
        area2_readiness_target: Optional[AreaReadinessTarget] = None,
        area2_physical_unlock_enabled: bool = True,
        area2_counterfactual_disable: bool = False,
        area2_redeploy_locked_face_trucks_to_development: bool = True,
        development_priority_truck_reservation_fraction: Optional[float] = None,
        annual_discount_rate: Optional[float] = None,
        ore1_net_value_per_processed_tonne: Optional[float] = None,
        ore2_net_value_per_processed_tonne: Optional[float] = None,
        production_cost_per_tonne: Optional[float] = None,
        development_cost_per_unit: Optional[float] = None,
        fixed_cost_per_day: Optional[float] = None,
        mode_a_ore1_milling_rate: Optional[float] = None,
        mode_a_ore2_milling_rate: Optional[float] = None,
        mode_a_contingency_ore1_milling_rate: Optional[float] = None,
        mode_b_ore1_milling_rate: Optional[float] = None,
        mode_b_ore2_milling_rate: Optional[float] = None,
        mode_b_contingency_ore2_milling_rate: Optional[float] = None,
        policy: int = 2,
        policy_name: Optional[str] = None,
        enable_analytical_blending: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        cfg = config or _DEFAULT_CONFIG
        self.config = cfg
        self.policy = policy
        if policy_name is not None:
            self.policy_name = policy_name
        else:
            self.policy_name = (
                "POLICY_1_MYOPIC" if policy == 1 else "POLICY_2_VALUE_ORIENTED"
            )
        self.enable_analytical_blending = enable_analytical_blending

        self.num_trucks = num_trucks if num_trucks is not None else cfg.fleet.num_trucks
        self.num_operators = num_operators if num_operators is not None else cfg.fleet.num_operators
        self.num_lhds_per_face = num_lhds_per_face if num_lhds_per_face is not None else cfg.fleet.num_lhds_per_face
        self.availability = availability if availability is not None else cfg.fleet.availability
        self.target_ore_stock_level = target_ore_stock_level if target_ore_stock_level is not None else cfg.plant.target_ore_stock_level
        self.critical_ore2_level = critical_ore2_level if critical_ore2_level is not None else cfg.plant.critical_ore2_level
        self.total_ore_to_extract = total_ore_to_extract if total_ore_to_extract is not None else cfg.plant.total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
            if ore_to_be_extracted_during_warming_period is not None
            else cfg.plant.ore_to_be_extracted_during_warming_period
        )

        self.strategic_period_days = (
            strategic_period_days
            if strategic_period_days is not None
            else cfg.planning.strategic_period_days
        )
        self.tactical_review_period_days = (
            tactical_review_period_days
            if tactical_review_period_days is not None
            else cfg.planning.tactical_review_period_days
        )
        self.tactical_progress_tolerance = (
            tactical_progress_tolerance
            if tactical_progress_tolerance is not None
            else cfg.planning.tactical_progress_tolerance
        )
        self.strategic_targets = strategic_targets or (
            StrategicYearTarget(
                min_development=cfg.planning.annual_min_development_m,
                min_ore1_production=cfg.planning.annual_min_ore1_production_t,
                min_ore2_production=cfg.planning.annual_min_ore2_production_t,
            ),
        )
        self.area2_readiness_target = area2_readiness_target or AreaReadinessTarget(
            required_development=cfg.planning.area2_required_development,
            ready_by_day=cfg.planning.area2_ready_by_day,
        )
        self.area2_physical_unlock_enabled = area2_physical_unlock_enabled
        self.area2_counterfactual_disable = area2_counterfactual_disable
        self.area2_redeploy_locked_face_trucks_to_development = (
            area2_redeploy_locked_face_trucks_to_development
        )
        self.development_priority_truck_reservation_fraction = (
            development_priority_truck_reservation_fraction
            if development_priority_truck_reservation_fraction is not None
            else cfg.planning.development_priority_truck_reservation_fraction
        )

        self.annual_discount_rate = (
            annual_discount_rate
            if annual_discount_rate is not None
            else cfg.economics.annual_discount_rate
        )
        self.ore1_net_value_per_processed_tonne = (
            ore1_net_value_per_processed_tonne
            if ore1_net_value_per_processed_tonne is not None
            else cfg.economics.ore1_net_value_per_processed_tonne
        )
        self.ore2_net_value_per_processed_tonne = (
            ore2_net_value_per_processed_tonne
            if ore2_net_value_per_processed_tonne is not None
            else cfg.economics.ore2_net_value_per_processed_tonne
        )
        self.production_cost_per_tonne = (
            production_cost_per_tonne
            if production_cost_per_tonne is not None
            else cfg.economics.production_cost_per_tonne
        )
        self.development_cost_per_unit = (
            development_cost_per_unit
            if development_cost_per_unit is not None
            else cfg.economics.development_cost_per_unit
        )
        self.fixed_cost_per_day = (
            fixed_cost_per_day
            if fixed_cost_per_day is not None
            else cfg.economics.fixed_cost_per_day
        )

        self.seed = seed if seed is not None else cfg.seed
        self.rng = random.Random(self.seed)

        self.truck_seat_credit = cfg.calendar.shift_seconds
        self._down_dur = 0.0

        # Calendar setup
        self.holidays: Set[int] = set(cfg.calendar.holidays)
        self._cur_day = -1
        self._shift_marker = -1
        self._holiday_today = False

        # Global time tracker
        self.gt = drs.Timer("gt", 0.0, rate=1.0)

        # 1. Physical Topology
        if topology is not None:
            self.topology = topology
        else:
            self.topology = MineTopology(
                decline_m=cfg.topology.decline_m,
                level_spacing_m=cfg.topology.level_spacing_m,
                level_drift_m=cfg.topology.level_drift_m,
                surface_m=cfg.topology.surface_m,
                level_depths={
                    cfg.topology.area1_level: cfg.topology.level_spacing_m * cfg.topology.area1_level,
                    cfg.topology.area2_level: cfg.topology.level_spacing_m * cfg.topology.area2_level,
                },
                speeds=cfg.topology.speeds,
                base_pass_bay_delay_sec=cfg.topology.base_pass_bay_delay_sec,
                per_truck_pass_bay_delay_sec=cfg.topology.per_truck_pass_bay_delay_sec,
                traffic_variation_tol=cfg.topology.traffic_variation_tol,
            )

        # 2. Mine Faces
        if faces is not None:
            self.faces = list(faces)
            self.face1 = self.faces[0]
            self.face2 = self.faces[1] if len(self.faces) > 1 else self.faces[0]
            self.gen1 = getattr(self.face1, "generator", None)
            self.gen2 = getattr(self.face2, "generator", None)
        else:
            self.gen1 = StochasticFaciesGenerator(
                mean_fraction=cfg.geology.area1_mean_fraction,
                std_dev=cfg.geology.area1_std_dev,
                prob_new_facies=cfg.geology.prob_new_facies,
                variation_same_facies=cfg.geology.variation_same_facies,
            )
            face_capacity = (
                self.total_ore_to_extract / 2.0
                if self.total_ore_to_extract is not None
                else 3300000.0
            )
            warmup_cap = (
                self.ore_to_be_extracted_during_warming_period / 2.0
                if self.ore_to_be_extracted_during_warming_period is not None
                else 0.0
            )
            self.face1 = MineFace(
                name="mine_face_1",
                face_id=1,
                area_id=1,
                level_index=cfg.topology.area1_level,
                generator=self.gen1,
                min_ore_mass=cfg.geology.min_parcel_mass,
                max_ore_mass=cfg.geology.max_parcel_mass,
                total_ore_to_extract=face_capacity,
                ore_to_be_extracted_during_warming_period=warmup_cap,
                mean_ore_fraction=cfg.geology.area1_mean_fraction,
                std_dev_ore_fraction=cfg.geology.area1_std_dev,
                prob_new_facies=cfg.geology.prob_new_facies,
                variation_same_facies=cfg.geology.variation_same_facies,
                initial_parcel_mass=cfg.geology.initial_parcel_mass,
                required_development=0.0,
                waste_to_ore_ratio=cfg.geology.waste_to_ore_ratio,
                turnaround_dev_per_parcel_m=cfg.geology.turnaround_dev_per_parcel_m,
                heading_cross_section_m2=cfg.topology.stope_cross_section_m2,
                rock_density_t_per_m3=cfg.topology.rock_density_t_per_m3,
            )

            self.gen2 = StochasticFaciesGenerator(
                mean_fraction=cfg.geology.area2_mean_fraction,
                std_dev=cfg.geology.area2_std_dev,
                prob_new_facies=cfg.geology.prob_new_facies,
                variation_same_facies=cfg.geology.variation_same_facies,
            )
            self.face2 = MineFace(
                name="mine_face_2",
                face_id=2,
                area_id=2,
                level_index=cfg.topology.area2_level,
                generator=self.gen2,
                min_ore_mass=cfg.geology.min_parcel_mass,
                max_ore_mass=cfg.geology.max_parcel_mass,
                total_ore_to_extract=face_capacity,
                ore_to_be_extracted_during_warming_period=warmup_cap,
                mean_ore_fraction=cfg.geology.area2_mean_fraction,
                std_dev_ore_fraction=cfg.geology.area2_std_dev,
                prob_new_facies=cfg.geology.prob_new_facies,
                variation_same_facies=cfg.geology.variation_same_facies,
                initial_parcel_mass=cfg.geology.initial_parcel_mass,
                required_development=self.area2_readiness_target.required_development,
                ready_by_day=self.area2_readiness_target.ready_by_day,
                counterfactual_disable=self.area2_counterfactual_disable,
                on_unlock_callback=self._on_area2_unlocked,
                waste_to_ore_ratio=cfg.geology.waste_to_ore_ratio,
                turnaround_dev_per_parcel_m=cfg.geology.turnaround_dev_per_parcel_m,
                heading_cross_section_m2=cfg.topology.stope_cross_section_m2,
                rock_density_t_per_m3=cfg.topology.rock_density_t_per_m3,
            )
            self.faces = [self.face1, self.face2]

        self.face_levels = {1: cfg.topology.area1_level, 2: cfg.topology.area2_level}

        # 3. Stockpiles
        init_fill = self.target_ore_stock_level * cfg.plant.initial_stock_fraction
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["ore_grade"],
            initial_mass=init_fill,
            initial_attributes={"ore_grade": 0.0},
            capacity=cfg.plant.stockpile_capacity,
        )
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["ore_grade"],
            initial_mass=init_fill,
            initial_attributes={"ore_grade": 1.0},
            capacity=cfg.plant.stockpile_capacity,
        )

        # 4. Plant & Campaign Mode Controller
        campaign_dur = (
            duration_of_production_campaigns
            if duration_of_production_campaigns is not None
            else cfg.plant.duration_of_production_campaigns
        )
        shutdown_dur = (
            duration_of_shutdowns
            if duration_of_shutdowns is not None
            else cfg.plant.duration_of_shutdowns
        )
        contingency_dur = (
            duration_of_contingency_segments
            if duration_of_contingency_segments is not None
            else cfg.plant.duration_of_contingency_segments
        )

        self.mode_controller = OperatingModeController(
            duration_of_production_campaigns=campaign_dur,
            duration_of_shutdowns=shutdown_dur,
            critical_ore2_level=self.critical_ore2_level,
            target_ore_stock_level=self.target_ore_stock_level,
            total_ore_to_extract=self.total_ore_to_extract,
        )
        self.plant = MetallurgicalPlant(
            stockpiles=[self.ore1_stock, self.ore2_stock],
            target_ore_stock_level=self.target_ore_stock_level,
            duration_of_contingency_segments=contingency_dur,
            mode_a_ore1_milling_rate=(
                mode_a_ore1_milling_rate
                if mode_a_ore1_milling_rate is not None
                else cfg.plant.mode_a_ore1_milling_rate
            ),
            mode_a_ore2_milling_rate=(
                mode_a_ore2_milling_rate
                if mode_a_ore2_milling_rate is not None
                else cfg.plant.mode_a_ore2_milling_rate
            ),
            mode_a_contingency_ore1_milling_rate=(
                mode_a_contingency_ore1_milling_rate
                if mode_a_contingency_ore1_milling_rate is not None
                else cfg.plant.mode_a_contingency_ore1_milling_rate
            ),
            mode_b_ore1_milling_rate=(
                mode_b_ore1_milling_rate
                if mode_b_ore1_milling_rate is not None
                else cfg.plant.mode_b_ore1_milling_rate
            ),
            mode_b_ore2_milling_rate=(
                mode_b_ore2_milling_rate
                if mode_b_ore2_milling_rate is not None
                else cfg.plant.mode_b_ore2_milling_rate
            ),
            mode_b_contingency_ore2_milling_rate=(
                mode_b_contingency_ore2_milling_rate
                if mode_b_contingency_ore2_milling_rate is not None
                else cfg.plant.mode_b_contingency_ore2_milling_rate
            ),
            economic_params=EconomicParameters(
                annual_discount_rate=self.annual_discount_rate,
                ore1_net_value_per_processed_tonne=self.ore1_net_value_per_processed_tonne,
                ore2_net_value_per_processed_tonne=self.ore2_net_value_per_processed_tonne,
                production_cost_per_tonne=self.production_cost_per_tonne,
                development_cost_per_unit=self.development_cost_per_unit,
                fixed_cost_per_day=self.fixed_cost_per_day,
            ),
        )

        # 5. Discrete Entities
        self.dump_station = SurfaceDumpStation(capacity=cfg.fleet.surface_tip_sites)
        self.waste_dump_station = SurfaceWasteDumpStation(capacity=cfg.fleet.surface_tip_sites)
        self.face_queues: Dict[int, list] = defaultdict(list)
        self.face_lhds_busy: Dict[int, int] = defaultdict(int)
        self.decline_heading_queue: list = []
        self.decline_heading_lhds_busy: int = 0
        self.num_lhds_per_decline: int = cfg.fleet.num_lhds_per_decline

        # Physical Heading Advance Parameters (Geometry Independence)
        self.capital_decline_cross_section_m2: float = cfg.topology.capital_decline_cross_section_m2
        self.stope_cross_section_m2: float = cfg.topology.stope_cross_section_m2
        self.rock_density_t_per_m3: float = cfg.topology.rock_density_t_per_m3
        self.heading_cross_section_m2: float = self.stope_cross_section_m2
        self.decline_dev_m_per_tonne_waste: float = 1.0 / (
            self.capital_decline_cross_section_m2 * self.rock_density_t_per_m3
        )
        self.stope_dev_m_per_tonne_waste: float = 1.0 / (
            self.stope_cross_section_m2 * self.rock_density_t_per_m3
        )
        self.dev_m_per_tonne_waste: float = self.stope_dev_m_per_tonne_waste
        self._delta_dev_accum: float = 0.0

        self.trucks = [
            Truck(f"T{i+1:02d}", drs.Timer(f"tr_tmr_{i}", 0.0, rate=-1.0))
            for i in range(self.num_trucks)
        ]
        self.operators = [Operator(idx=i) for i in range(self.num_operators)]
        self._refuel_pumps_in_use = 0
        self._refuel_queue: list = []

        # 6. Strategic / Tactical Planning Controller
        self.tactical_controller = TacticalReviewController(
            strategic_targets=self.strategic_targets,
            strategic_period_days=self.strategic_period_days,
            tactical_review_period_days=self.tactical_review_period_days,
            tactical_progress_tolerance=self.tactical_progress_tolerance,
            development_priority_truck_reservation_fraction=self.development_priority_truck_reservation_fraction,
        )

        # Readiness & Development Levels
        self.sustaining_cumulative_development = drs.Level(
            "sustaining_cumulative_development", 0.0
        )
        self.area2_cumulative_development = drs.Level(
            "area2_cumulative_development", 0.0
        )
        self.cumulative_mine_development = drs.Level(
            "cumulative_mine_development", 0.0
        )
        self.development_rate_m_per_day = drs.Level(
            "development_rate_m_per_day", 0.0
        )
        self.development_priority_reserved_trucks = drs.Level(
            "development_priority_reserved_trucks", 0.0
        )

        self.development_trajectory_ratio = drs.Level(
            "development_trajectory_ratio", 1.0
        )
        self.ore1_trajectory_ratio = drs.Level("ore1_trajectory_ratio", 1.0)
        self.ore2_trajectory_ratio = drs.Level("ore2_trajectory_ratio", 1.0)
        self.area2_readiness_trajectory_ratio = drs.Level(
            "area2_readiness_trajectory_ratio", 1.0
        )
        self.area2_readiness_fraction = drs.Level(
            "area2_readiness_fraction", 0.0
        )
        self.area2_ready_day = drs.Level("area2_ready_day", -1.0)
        self.area1_depleted_day = drs.Level("area1_depleted_day", -1.0)
        self.area2_depleted_day = drs.Level("area2_depleted_day", -1.0)
        self._area2_unlocked = False
        self._face1_depleted = False
        self._face2_depleted = False

        # Fleet Operating Mode Controller (Production vs Development)
        self.fleet_controller = FleetModeController(
            initial_mode=FleetOperatingMode.PRODUCTION,
            target_stockpile_buffer_tonnes=self.target_ore_stock_level,
        )

        # Analytical Blending Levels (Level 3 Operational Control)
        self.analytical_face1_weight = drs.Level("analytical_face1_weight", 1.0)
        self.analytical_face2_weight = drs.Level("analytical_face2_weight", 0.0)
        self.analytical_face1_rate_target = drs.Level(
            "analytical_face1_rate_target", 0.0
        )
        self.analytical_face2_rate_target = drs.Level(
            "analytical_face2_rate_target", 0.0
        )
        self.analytical_blend_feasible = drs.Level(
            "analytical_blend_feasible", 1.0
        )

        # Operational metrics & targets
        self.daily_target_ore = 6000.0
        self.daily_hauled_ore = 0.0
        self.trips = 0
        self._cycle_sum = 0.0
        self.traffic_delay_sum = 0.0
        self.horizon_sec = float("inf")

        self.annual_ore1_extracted = 0.0
        self.annual_ore2_extracted = 0.0
        self.annual_development_start = 0.0

        # Component References
        self.readiness_tracker = self.face2
        self.economics = self.plant

        # Counters & Telemetry
        self.total_extracted_ore = drs.Level("total_extracted_ore", 0.0)
        self.ore1_hauled = drs.Level("ore1_hauled", 0.0)
        self.ore2_hauled = drs.Level("ore2_hauled", 0.0)
        self.ore1_dumped_total = drs.Level("ore1_dumped_total", 0.0)
        self.ore2_dumped_total = drs.Level("ore2_dumped_total", 0.0)
        self.tonnes_hauled_by_face: Dict[int, float] = defaultdict(float)
        self.ore1_hauled_by_face: Dict[int, float] = defaultdict(float)
        self.ore2_hauled_by_face: Dict[int, float] = defaultdict(float)

        self.telemetry_history: List[Dict[str, Any]] = []
        self._telemetry_dt = cfg.telemetry_dt  # Log interval from config
        self._next_telemetry_t = 0.0

    def _on_area2_unlocked(self, day: float = 0.0) -> None:
        """Callback triggered when Area 2 physical development reaches 100%."""
        if not self._area2_unlocked:
            self._area2_unlocked = True
            self.area2_ready_day.value = day
            self.face2.ready_day.value = day
            self.face2.state = FaceState.ORE_READY
            print(
                f"\n >>> [{self.policy_name} UNLOCK] Area 2 (Face 2) UNLOCKED on Strategic Day {day:.2f}! <<<\n"
            )

    def _on_face1_exhausted(self, day: float = 0.0) -> None:
        """Callback triggered when Area 1 (Face 1) is fully exhausted."""
        if not self._face1_depleted:
            self._face1_depleted = True
            self.area1_depleted_day.value = day
            if hasattr(self, "face1"):
                self.face1.state = FaceState.EXHAUSTED
            print(
                f"\n >>> [{self.policy_name} DEPLETED] Area 1 (Face 1) DEPLETED on Strategic Day {day:.2f}! <<<\n"
            )

    def _on_face2_exhausted(self, day: float = 0.0) -> None:
        """Callback triggered when Area 2 (Face 2) is fully exhausted."""
        if not self._face2_depleted:
            self._face2_depleted = True
            self.area2_depleted_day.value = day
            if hasattr(self, "face2"):
                self.face2.state = FaceState.EXHAUSTED
            print(
                f"\n >>> [{self.policy_name} DEPLETED] Area 2 (Face 2) DEPLETED on Strategic Day {day:.2f}! <<<\n"
            )


    def is_area2_locked(self, current_day: float = 0.0) -> bool:
        """Evaluates whether Area 2 / Face 2 is currently locked."""
        if not self.area2_physical_unlock_enabled:
            return False
        required = max(0.0, float(self.area2_readiness_target.required_development))
        if required <= 1e-12:
            return False
        return not (self.tactical_controller.planning_started and self.area2_ready)

    def is_face1_exhausted(self) -> bool:
        return (
            self.face1.cumulative_extracted_mass.value
            >= self.face1.total_ore_to_extract - 1e-6
        )

    def is_face2_exhausted(self) -> bool:
        if not hasattr(self, "face2") or self.face2 is self.face1:
            return True
        return (
            self.face2.cumulative_extracted_mass.value
            >= self.face2.total_ore_to_extract - 1e-6
        )

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Operator & Shift Mechanics
    # -----------------------------------------------------------------------
    def _acquire_operator(self, tr: Truck) -> bool:
        if tr.operator >= 0 and not self.operators[tr.operator].free:
            return self.operators[tr.operator].used_seat < self.truck_seat_credit
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
        return self.topology.calculate_travel_time_sec(
            level=tr.target_level,
            loaded=loaded,
            active_truck_count=len(self.trucks),
            rng=self.rng,
        )

    def _advance_truck_corridor(self, tr: Truck) -> None:
        """Advances a truck through its assigned route of discrete single-lane segments and passing bays."""
        direction = "DOWN" if tr.phase == TruckPhase.EMPTY else "UP"

        # 1. Reached destination at end of corridor route
        if tr.route_segment_idx >= len(tr.route_segments):
            if tr.in_passing_bay is not None:
                tr.in_passing_bay.remove_truck(tr)
            tr.current_segment = None
            if tr.phase == TruckPhase.EMPTY:
                if tr.is_waste and tr.mission_type == MissionType.CAPITAL_DECLINE_DEV:
                    self._enter_decline_loadout(tr)
                else:
                    self._enter_face_loadout(tr, tr.target_face_id)
            elif tr.phase == TruckPhase.LOADED:
                if tr.is_waste:
                    self._enter_waste_dump(tr)
                else:
                    self._enter_dump(tr)
            return

        # 2. Try entering next segment on route
        seg = tr.route_segments[tr.route_segment_idx]
        if seg.can_enter(tr, direction):
            trav_time = seg.occupy(tr, direction)
            tr.current_segment = seg
            if tr.in_passing_bay is not None:
                tr.in_passing_bay.remove_truck(tr)
            tr.timer.value = trav_time
            tr.timer.rate = -1.0
        else:
            # Yield and wait in passing bay at the entry of this segment
            entry_bay = seg.upstream_bay if direction == "DOWN" else seg.downstream_bay
            if entry_bay is not None:
                entry_bay.queue_truck(tr)
            tr.current_segment = None
            tr.timer.value = 0.0
            tr.timer.rate = 0.0

    def _wake_waiting_traffic_at_segment(self, seg: RoadSegment) -> None:
        """Wakes up yielding trucks waiting in passing bays for this cleared segment."""
        # 1. Loaded trucks heading UP have right-of-way
        if seg.downstream_bay is not None:
            loaded_trk = seg.downstream_bay.get_waiting_loaded_truck()
            if loaded_trk is not None and seg.can_enter(loaded_trk, "UP"):
                self._advance_truck_corridor(loaded_trk)
                return

        # 2. Empty trucks heading DOWN
        if seg.upstream_bay is not None:
            empty_trk = seg.upstream_bay.get_waiting_empty_truck()
            if empty_trk is not None and seg.can_enter(empty_trk, "DOWN"):
                self._advance_truck_corridor(empty_trk)
                return

    # -----------------------------------------------------------------------
    # Dispatch Face Selection
    # -----------------------------------------------------------------------
    def select_face_for_truck(self, tr: Optional[Truck] = None) -> int:
        """Selects target face for haulage dispatch using mode-based routing matching Shelswell baseline."""
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

        mode_name = self.plant.active_operating_mode.value.name
        p_face2 = 0.65 if "MODE_A" in mode_name else 0.35
        return 2 if self.rng.random() < p_face2 else 1

    # -----------------------------------------------------------------------
    # Discrete State Machine Flow & Haulage Dispatch
    # -----------------------------------------------------------------------
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
            if self._refuel_pumps_in_use < N_FUEL_PUMPS:
                if tr.operator < 0 and not self._acquire_operator(tr):
                    return False
                self._enter_refuel(tr)
                return True
            return False

        mode_name = self.plant.active_operating_mode.value.name
        is_shutdown = mode_name == "SHUTDOWN"
        is_surging = "SURGING" in mode_name

        total_stock = float(self.ore1_stock.level + self.ore2_stock.level)
        active_capital_dev = sum(
            1
            for trk in self.trucks
            if getattr(trk, "mission_type", None) == MissionType.CAPITAL_DECLINE_DEV
            and trk.phase in OPERATING_PHASES
        )
        active_prod = sum(
            1
            for trk in self.trucks
            if getattr(trk, "mission_type", None) == MissionType.ORE_HAUL
            and trk.phase in OPERATING_PHASES
        )

        can_mine_ore = False
        if not is_shutdown:
            if self.is_area2_locked():
                if not self.is_face1_exhausted():
                    can_mine_ore = True
            else:
                if not (self.is_face1_exhausted() and self.is_face2_exhausted()):
                    can_mine_ore = True

        mission_res = self.fleet_controller.select_mission(
            truck=tr,
            current_total_stock=total_stock,
            is_plant_shutdown=is_shutdown,
            can_mine_ore=can_mine_ore,
            active_prod_trucks=active_prod,
            active_capital_dev_trucks=active_capital_dev,
            faces=self.faces if hasattr(self, "faces") else [self.face1, self.face2],
            area2_locked=self.is_area2_locked(),
            preferred_face_id=self.select_face_for_truck(tr),
            face_levels=self.face_levels,
            default_area1_level=AREA1_LEVEL,
            default_area2_level=AREA2_LEVEL,
            daily_target_ore=self.daily_target_ore,
            daily_hauled_ore=self.daily_hauled_ore,
            is_surging=is_surging,
            area1_exhausted=self.is_face1_exhausted(),
            reserved_dev_trucks=int(max(2, round(len(self.trucks) * 0.33))),
        )

        if mission_res is None:
            self._release_operator(tr)
            return False

        mission_type, target_face_id, target_level, is_waste = mission_res

        if not self._acquire_operator(tr):
            self._release_operator(tr)
            return False

        tr.mission_type = mission_type
        tr.target_face_id = target_face_id
        tr.target_level = target_level
        tr.is_waste = is_waste
        tr.trip_start = t
        tr.phase = TruckPhase.EMPTY
        is_cap_dev = (mission_type == MissionType.CAPITAL_DECLINE_DEV)
        tr.route_segments = self.topology.get_route(
            target_level=target_level, direction="DOWN", is_capital_dev=is_cap_dev
        )
        tr.route_segment_idx = 0
        self._advance_truck_corridor(tr)
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

    def _enter_decline_loadout(self, tr: Truck) -> None:
        tr.phase = TruckPhase.WAIT_LOAD
        tr.timer.rate = 0.0
        self.decline_heading_queue.append(tr)
        self._service_decline_queue()

    def _service_decline_queue(self) -> None:
        while (
            self.decline_heading_queue
            and self.decline_heading_lhds_busy < self.num_lhds_per_decline
        ):
            nxt = self.decline_heading_queue.pop(0)
            self.decline_heading_lhds_busy += 1
            nxt.phase = TruckPhase.SPOT_LOAD
            spot = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.15)
            nxt.timer.value = spot
            nxt.timer.rate = -1.0

    def get_face_by_id(self, face_id: int) -> MineFace:
        """Resolves a MineFace component by its numeric face_id."""
        if hasattr(self, "faces"):
            faces_attr = getattr(self, "faces")
            if isinstance(faces_attr, dict) and face_id in faces_attr:
                return faces_attr[face_id]
            if isinstance(faces_attr, (list, tuple)):
                for f in faces_attr:
                    if getattr(f, "face_id", None) == face_id:
                        return f
        if face_id == 2:
            return self.face2
        return self.face1

    def _finish_loading(self, tr: Truck) -> None:
        t = self.gt.value
        if tr.mission_type == MissionType.CAPITAL_DECLINE_DEV:
            tr.current_payload = ORE_PAYLOAD  # 54t development waste
            tr.payload_ore_fraction = 0.0
            tr.is_waste = True
            self.decline_heading_lhds_busy = max(
                0, self.decline_heading_lhds_busy - 1
            )
            self._service_decline_queue()
            tr.phase = TruckPhase.LOADED
            tr.route_segments = self.topology.get_route(
                target_level=tr.target_level, direction="UP", is_capital_dev=True
            )
            tr.route_segment_idx = 0
            self._advance_truck_corridor(tr)
            return

        if tr.mission_type == MissionType.STOPE_TURNAROUND_DEV:
            face = self.get_face_by_id(tr.target_face_id)
            w_mass, d_m, is_complete = face.extract_turnaround_waste(ORE_PAYLOAD)
            tr.current_payload = w_mass if w_mass > 0 else ORE_PAYLOAD
            tr.payload_ore_fraction = 0.0
            tr.is_waste = True
            self.face_lhds_busy[tr.target_face_id] = max(
                0, self.face_lhds_busy[tr.target_face_id] - 1
            )
            self._service_face_queue(tr.target_face_id)
            tr.phase = TruckPhase.LOADED
            tr.route_segments = self.topology.get_route(
                target_level=tr.target_level, direction="UP", is_capital_dev=False
            )
            tr.route_segment_idx = 0
            self._advance_truck_corridor(tr)
            return

        face = self.get_face_by_id(tr.target_face_id)
        ore2_frac = float(face.active_parcel_ore_fraction.value)
        tr.current_payload = ORE_PAYLOAD
        tr.payload_ore_fraction = ore2_frac
        tr.is_waste = False

        # Update face extraction counters
        face.cumulative_extracted_mass.value += ORE_PAYLOAD
        face.parcel_extracted_mass.value += ORE_PAYLOAD
        if hasattr(face, "advance_parcel_state"):
            face.advance_parcel_state()

        self.tonnes_hauled_by_face[tr.target_face_id] += ORE_PAYLOAD
        self.ore1_hauled_by_face[tr.target_face_id] += ORE_PAYLOAD * (
            1.0 - ore2_frac
        )
        self.ore2_hauled_by_face[tr.target_face_id] += ORE_PAYLOAD * ore2_frac

        if (
            tr.target_face_id == 1
            and not self._face1_depleted
            and self.is_face1_exhausted()
        ):
            day = (
                float(self.tactical_controller.strategic_year_index.value)
                * self.strategic_period_days
                + float(self.tactical_controller.strategic_year_timer.value)
            ) if hasattr(self, "tactical_controller") else (self.gt.value / 86400.0)
            self._on_face1_exhausted(day)
        elif (
            tr.target_face_id == 2
            and not self._face2_depleted
            and self.is_face2_exhausted()
        ):
            day = (
                float(self.tactical_controller.strategic_year_index.value)
                * self.strategic_period_days
                + float(self.tactical_controller.strategic_year_timer.value)
            ) if hasattr(self, "tactical_controller") else (self.gt.value / 86400.0)
            self._on_face2_exhausted(day)

        # Free loader
        self.face_lhds_busy[tr.target_face_id] = max(
            0, self.face_lhds_busy[tr.target_face_id] - 1
        )
        self._service_face_queue(tr.target_face_id)

        # Dispatch loaded haul
        tr.phase = TruckPhase.LOADED
        is_cap_dev = (tr.mission_type == MissionType.CAPITAL_DECLINE_DEV)
        tr.route_segments = self.topology.get_route(
            target_level=tr.target_level, direction="UP", is_capital_dev=is_cap_dev
        )
        tr.route_segment_idx = 0
        self._advance_truck_corridor(tr)

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
        dur = tr.dump_dur if tr.dump_dur > 0.0 else 1.0
        o2_frac = tr.payload_ore_fraction
        ore2_mass = tr.current_payload * o2_frac
        ore1_mass = tr.current_payload * (1.0 - o2_frac)

        self.dump_station._active_ore1_rate = max(
            0.0, self.dump_station._active_ore1_rate - ore1_mass / dur
        )
        self.dump_station._active_ore2_rate = max(
            0.0, self.dump_station._active_ore2_rate - ore2_mass / dur
        )

        self.daily_hauled_ore += tr.current_payload
        self.total_extracted_ore.value += tr.current_payload
        self.ore1_hauled.value += ore1_mass
        self.ore2_hauled.value += ore2_mass
        self.ore1_dumped_total.value += ore1_mass
        self.ore2_dumped_total.value += ore2_mass
        self.annual_ore1_extracted += ore1_mass
        self.annual_ore2_extracted += ore2_mass

        self.dump_station.in_use = max(0, self.dump_station.in_use - 1)
        self._service_dump_queue()

        self.trips += 1
        self._cycle_sum += t - tr.trip_start
        tr.current_payload = 0.0
        self._release_operator(tr)

        # Check refueling requirement
        if tr.fuel <= tr.refuel_threshold:
            self._enter_refuel(tr)
        else:
            tr.phase = TruckPhase.IDLE
            tr.timer.value = 0.0
            tr.timer.rate = 0.0

    def _enter_waste_dump(self, tr: Truck) -> None:
        tr.phase = TruckPhase.WAIT_DUMP
        tr.timer.rate = 0.0
        self.waste_dump_station.queue.append(tr)
        self._service_waste_dump_queue()

    def _service_waste_dump_queue(self) -> None:
        while (
            self.waste_dump_station.queue
            and self.waste_dump_station.in_use < self.waste_dump_station.capacity
        ):
            nxt = self.waste_dump_station.queue.pop(0)
            self._start_waste_dump(nxt)

    def _start_waste_dump(self, tr: Truck) -> None:
        self.waste_dump_station.in_use += 1
        tr.phase = TruckPhase.DUMPING
        spot = _tri(self.rng, DUMP_SPOT_MIN * 60.0, 0.15)
        dump = _tri(self.rng, DUMP_MIN * 60.0, 0.15)
        tr.dump_dur = spot + dump
        tr.timer.value = tr.dump_dur
        tr.timer.rate = -1.0

    def _finish_waste_dumping(self, tr: Truck) -> None:
        t = self.gt.value

        if tr.mission_type == MissionType.CAPITAL_DECLINE_DEV:
            dev_m = tr.current_payload * self.decline_dev_m_per_tonne_waste
            self._delta_dev_accum += dev_m
            self.area2_cumulative_development.value += dev_m
            self.face2.cumulative_development.value = float(
                self.area2_cumulative_development.value
            )
            self.cumulative_mine_development.value += dev_m
            req_dev = (
                self.area2_readiness_target.required_development
                if self.area2_readiness_target
                else 6000.0
            )
            if (
                float(self.area2_cumulative_development.value) >= req_dev - 1e-6
                and not self._area2_unlocked
            ):
                day = (
                    float(self.tactical_controller.strategic_year_index.value)
                    * self.strategic_period_days
                    + float(self.tactical_controller.strategic_year_timer.value)
                ) if hasattr(self, "tactical_controller") else (self.gt.value / 86400.0)
                self._on_area2_unlocked(day)
        elif tr.mission_type == MissionType.STOPE_TURNAROUND_DEV:
            face = self.get_face_by_id(tr.target_face_id)
            face_cs = getattr(face, "heading_cross_section_m2", self.stope_cross_section_m2)
            dev_m = tr.current_payload / (face_cs * self.rock_density_t_per_m3)
            self._delta_dev_accum += dev_m
            self.sustaining_cumulative_development.value += dev_m
            self.cumulative_mine_development.value += dev_m
            face.advance_turnaround(dev_advance_m=dev_m, waste_tonnes=tr.current_payload)
        else:
            dev_m = tr.current_payload * self.stope_dev_m_per_tonne_waste
            self._delta_dev_accum += dev_m
            self.sustaining_cumulative_development.value += dev_m
            self.cumulative_mine_development.value += dev_m

        self.waste_dump_station.in_use = max(
            0, self.waste_dump_station.in_use - 1
        )
        self._service_waste_dump_queue()

        self.trips += 1
        self._cycle_sum += t - tr.trip_start
        tr.current_payload = 0.0
        tr.is_waste = False
        self._release_operator(tr)

        # Check refueling requirement
        if tr.fuel <= tr.refuel_threshold:
            self._enter_refuel(tr)
        else:
            tr.phase = TruckPhase.IDLE
            tr.timer.value = 0.0
            tr.timer.rate = 0.0

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
        tr.timer.value = 0.0
        tr.timer.rate = 0.0

    # -----------------------------------------------------------------------
    # Campaign Mode & Plant Target Updating
    # -----------------------------------------------------------------------
    def _update_operating_mode_and_targets(self):
        """Updates campaign mode, plant operating mode, and daily extraction target."""
        camp_mode = self.mode_controller.update(
            ore2_stock_level=float(self.ore2_stock.level),
            total_stock_level=float(self.ore1_stock.level + self.ore2_stock.level),
        )

        if self.is_area2_locked() or self.is_face2_exhausted():
            f_blend = float(self.face1.active_parcel_ore_fraction.value)
        elif self.is_face1_exhausted():
            f_blend = float(self.face2.active_parcel_ore_fraction.value)
        else:
            f_blend = (
                float(self.face1.active_parcel_ore_fraction.value)
                + float(self.face2.active_parcel_ore_fraction.value)
            ) / 2.0

        plant_draw, _ = self.plant.get_target_rates(
            camp_mode,
            ore1_level=float(self.ore1_stock.level),
            ore2_level=float(self.ore2_stock.level),
            stockpile2_routing_fraction=f_blend,
        )

        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self.daily_target_ore = 0.0
        elif "_MINE_SURGING" in mode_name:
            self.daily_target_ore = plant_draw.total * 0.70
        else:
            self.daily_target_ore = plant_draw.total

        return plant_draw

    # -----------------------------------------------------------------------
    # Strategic & Tactical Progress Reviews (Level 1 & 2 Hierarchy)
    # -----------------------------------------------------------------------
    def _update_strategic_tactical_review(self):
        total_extracted = sum(
            float(f.cumulative_extracted_mass.value) for f in self.faces
        )
        if (
            total_extracted >= self.ore_to_be_extracted_during_warming_period
            or self.horizon_sec < float("inf")
        ):
            if not self.tactical_controller.planning_started:
                self.tactical_controller.planning_started = True
                self.tactical_controller.strategic_year_index.value = 0.0
                self.tactical_controller.strategic_year_timer.reset()
                self.tactical_controller.tactical_review_timer.reset()
                self.tactical_controller.tactical_review_count.value = 0.0
                self.annual_ore1_extracted = 0.0
                self.annual_ore2_extracted = 0.0
                self.annual_development_start = float(
                    self.cumulative_mine_development.value
                )

        if not self.tactical_controller.planning_started:
            return

        # Strategic Annual Rollover
        if (
            self.tactical_controller.strategic_year_timer.value
            >= self.strategic_period_days - 1e-6
        ):
            self.tactical_controller.strategic_year_index.value += 1.0
            self.tactical_controller.strategic_year_timer.reset()
            self.annual_ore1_extracted = 0.0
            self.annual_ore2_extracted = 0.0
            self.annual_development_start = float(
                self.cumulative_mine_development.value
            )

        elapsed_year_fraction = max(
            1e-4,
            min(
                1.0,
                self.tactical_controller.strategic_year_timer.value
                / self.strategic_period_days,
            ),
        )
        current_target = strategic_target_for_year(
            self.strategic_targets,
            int(self.tactical_controller.strategic_year_index.value),
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

        # Monthly Tactical Review (Adaptive Priority Selection - Slide 18)
        if (
            self.tactical_controller.tactical_review_timer.value
            >= self.tactical_review_period_days - 1e-6
        ):
            self.tactical_controller.tactical_review_timer.reset()
            self.tactical_controller.tactical_review_count.value += 1.0

            r_dev = float(self.development_trajectory_ratio.value)
            r_area2 = float(self.area2_readiness_trajectory_ratio.value)
            r_ore1 = float(self.ore1_trajectory_ratio.value)
            r_ore2 = float(self.ore2_trajectory_ratio.value)
            tol = self.tactical_progress_tolerance

            prio_mode = select_fleet_mode(
                development_ratio=r_dev,
                ore1_ratio=r_ore1,
                ore2_ratio=r_ore2,
                tolerance=tol,
                area2_readiness_trajectory_ratio=r_area2,
            )
            self.tactical_controller.active_fleet_mode = prio_mode

            if prio_mode == FLEET_MODES["DEVELOPMENT"]:
                frac = self.development_priority_truck_reservation_fraction
                n_res = max(1, int(math.ceil(len(self.trucks) * frac)))
                self.development_priority_reserved_trucks.value = float(n_res)
            else:
                self.development_priority_reserved_trucks.value = 0.0

    # -----------------------------------------------------------------------
    # Area 2 Readiness & Deadline Tracking
    # -----------------------------------------------------------------------
    def _update_area2_readiness(self):
        target = self.area2_readiness_target
        required = max(0.0, float(target.required_development))
        if required <= 1e-12:
            self._area2_unlocked = True
            self.area2_readiness_fraction.value = 1.0
            self.area2_readiness_trajectory_ratio.value = 1.0
            return

        if not self.tactical_controller.planning_started:
            self.area2_readiness_fraction.value = 0.0
            self.area2_readiness_trajectory_ratio.value = 1.0
            return

        strategic_days = (
            float(self.tactical_controller.strategic_year_index.value)
            * self.strategic_period_days
            + float(self.tactical_controller.strategic_year_timer.value)
        )
        progress = float(self.area2_cumulative_development.value)
        fraction = min(1.0, progress / required)
        self.area2_readiness_fraction.value = fraction

        # Check for Physical Unlock
        if (not self._area2_unlocked) and progress >= required - 1e-6:
            self._on_area2_unlocked(strategic_days)

        # Deadline Tracking
        ready_by_day = target.ready_by_day
        if ready_by_day is not None and ready_by_day > 0.0:
            elapsed_fraction = max(1e-4, min(1.0, strategic_days / ready_by_day))
            self.area2_readiness_trajectory_ratio.value = (
                trajectory_progress_ratio(
                    actual=progress,
                    annual_target=required,
                    elapsed_fraction=elapsed_fraction,
                )
            )
            deadline_exceeded = strategic_days > ready_by_day
            if not self._area2_unlocked:
                self.face2.deadline_missed = deadline_exceeded
                self.face2.currently_late = deadline_exceeded
                self.face2.completed_late = False
            else:
                self.face2.currently_late = False
                if float(self.area2_ready_day.value) > ready_by_day + 1e-6:
                    self.face2.deadline_missed = True
                    self.face2.completed_late = True
                else:
                    self.face2.deadline_missed = False
                    self.face2.completed_late = False
        else:
            self.area2_readiness_trajectory_ratio.value = 1.0

        # Update Fleet Operating Mode Controller (Production vs Development)
        self.fleet_controller.evaluate_mode(
            policy=self.policy,
            current_day=strategic_days,
            dev_progress_m=progress,
            required_dev_m=required,
            deadline_day=ready_by_day,
            area2_locked=not self._area2_unlocked,
            plant_operating_mode=self.plant.active_operating_mode.value.name,
        )

    # -----------------------------------------------------------------------
    # Calendar & Shift Updates
    # -----------------------------------------------------------------------
    def _calendar_update(self, t: float) -> None:
        day = int(t // 86400.0)
        if day != self._cur_day:
            self._cur_day = day
            self._holiday_today = (day % 365) in self.holidays
            self.daily_hauled_ore = 0.0
            self._update_operating_mode_and_targets()

        shift = int(t // SHIFT_SECONDS)
        if shift != self._shift_marker:
            self._shift_marker = shift
            bound = {tr.operator for tr in self.trucks if tr.operator >= 0}
            for op in self.operators:
                op.used_seat = 0.0
                op.free = op.idx not in bound
            for tr in self.trucks:
                tr.seat_used = 0.0
                self._schedule_down_window(tr)

    # -----------------------------------------------------------------------
    # Continuous Integration & DES Stepping
    # -----------------------------------------------------------------------
    def time_to_event(self) -> float:
        min_dt = DT_MAX
        t = self.gt.value

        for tr in self.trucks:
            if tr.timer.rate < 0 and tr.timer.value > 0.0:
                min_dt = min(min_dt, tr.timer.value)
            if tr.down_start > t and tr.down_start != math.inf:
                min_dt = min(min_dt, tr.down_start - t)
            if tr.down_end > t and tr.down_end != math.inf:
                min_dt = min(min_dt, tr.down_end - t)

        # Calendar and shift boundaries
        next_day = (math.floor(t / 86400.0) + 1.0) * 86400.0
        next_shift = (math.floor(t / SHIFT_SECONDS) + 1.0) * SHIFT_SECONDS
        min_dt = min(min_dt, max(1.0, next_day - t), max(1.0, next_shift - t))

        # Remaining campaign duration
        c_thresh = (
            self.mode_controller.duration_of_shutdowns
            if self.mode_controller.active_campaign_mode.value.name == "SHUTDOWN"
            else self.mode_controller.duration_of_production_campaigns
        )
        rem_c_days = max(
            0.0,
            c_thresh - self.mode_controller.current_campaign_duration.value,
        )
        if rem_c_days > 1e-6:
            min_dt = min(min_dt, rem_c_days * 86400.0)

        # Remaining contingency duration
        if (
            self.plant.active_operating_mode.value.name
            in self.plant._CONTINGENCY_MODES
        ):
            rem_cont = max(
                0.0,
                self.plant.duration_of_contingency_segments
                - self.plant.current_contingency_duration.value,
            )
            if rem_cont > 1e-6:
                min_dt = min(min_dt, rem_cont * 86400.0)

        # Telemetry interval
        if self._next_telemetry_t > t:
            min_dt = min(min_dt, self._next_telemetry_t - t)

        return max(1e-6, min_dt)

    def _advance(self, t_target: float) -> None:
        dt = t_target - self.gt.value
        if dt <= 0.0:
            return
        dt_days = dt / 86400.0

        # 1. Step Global & Campaign Timers
        self.gt.step(dt)
        self.mode_controller.current_campaign_duration.step(dt_days)
        active_mode_name = self.plant.active_operating_mode.value.name
        if active_mode_name == "MODE_A":
            self.plant.cumulative_time_mode_a.step(dt_days)
        elif active_mode_name == "MODE_B":
            self.plant.cumulative_time_mode_b.step(dt_days)
        elif active_mode_name == "SHUTDOWN":
            self.plant.cumulative_time_shutdown.step(dt_days)

        if active_mode_name in self.plant._CONTINGENCY_MODES:
            self.plant.current_contingency_duration.step(dt_days)

        if self.tactical_controller.planning_started:
            self.tactical_controller.step_timers(dt_days)

        # 2. Step Fleet & Operator Timers
        for tr in self.trucks:
            if tr.timer.rate < 0.0 and tr.timer.value > 0.0:
                tr.timer.value = max(0.0, tr.timer.value - dt)
            if tr.phase in SEAT_PHASES:
                tr.seat_used = min(self.truck_seat_credit, tr.seat_used + dt)
                if tr.phase in OPERATING_PHASES:
                    tr.fuel = max(0.0, tr.fuel - dt * FUEL_BURN_PCT_PER_SEC)
                if tr.operator >= 0:
                    op = self.operators[tr.operator]
                    op.used_seat = min(SEAT_PER_SHIFT_SEC, op.used_seat + dt)

        # 3. Mill Draw and Stockpile Integration
        ore1_in_rate = self.dump_station._active_ore1_rate
        ore2_in_rate = self.dump_station._active_ore2_rate

        if self.is_area2_locked() or self.is_face2_exhausted():
            f_blend = float(self.face1.active_parcel_ore_fraction.value)
        elif self.is_face1_exhausted():
            f_blend = float(self.face2.active_parcel_ore_fraction.value)
        else:
            f_blend = (
                float(self.face1.active_parcel_ore_fraction.value)
                + float(self.face2.active_parcel_ore_fraction.value)
            ) / 2.0

        plant_draw, _ = self.plant.get_target_rates(
            self.mode_controller.active_campaign_mode.value,
            ore1_level=float(self.ore1_stock.level),
            ore2_level=float(self.ore2_stock.level),
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

        # 4. Discrete Development Capex Stepping
        delta_dev = self._delta_dev_accum
        self._delta_dev_accum = 0.0

        self._update_area2_readiness()

        # Step plant financial economics
        t_days = self.gt.value / 86400.0
        self.plant.step_economics(
            out1_t_sec=out1,
            out2_t_sec=out2,
            delta_dev_meters=delta_dev,
            dt_days=dt_days,
            t_days=t_days,
        )

        self.total_extracted_ore.step(dt)
        self.ore1_hauled.step(dt)
        self.ore2_hauled.step(dt)

    def _advance_truck_state(self, tr: Truck) -> bool:
        ph = tr.phase
        if ph in (TruckPhase.EMPTY, TruckPhase.LOADED):
            if tr.current_segment is not None:
                prev_seg = tr.current_segment
                prev_seg.release(tr)
                tr.current_segment = None
                self._wake_waiting_traffic_at_segment(prev_seg)
                tr.route_segment_idx += 1
            self._advance_truck_corridor(tr)
            return True
        if ph == TruckPhase.SPOT_LOAD:
            tr.phase = TruckPhase.ACQUIRE
            tr.timer.value = (
                self.rng.uniform(0.0, LHD_ACQUISITION_MAX_MIN) * 60.0
            )
            tr.timer.rate = -1.0
            return True
        if ph == TruckPhase.ACQUIRE:
            tr.phase = TruckPhase.LOADING
            tr.timer.value = _tri(self.rng, TRUCK_LOAD_DUR_MIN * 60.0, 0.20)
            tr.timer.rate = -1.0
            return True
        if ph == TruckPhase.LOADING:
            self._finish_loading(tr)
            return True
        if ph == TruckPhase.DUMPING:
            if tr.is_waste:
                self._finish_waste_dumping(tr)
            else:
                self._finish_dumping(tr)
            return True
        if ph == TruckPhase.REFUELING:
            self._finish_refuel(tr)
            return True
        return False

    def is_terminating_condition_met(self) -> bool:
        """Evaluates whether all mining reserves are fully exhausted or simulation horizon is reached."""
        if hasattr(self, "faces") and len(self.faces) > 1:
            if self.is_face1_exhausted() and self.is_face2_exhausted():
                return True
        elif hasattr(self, "face1"):
            if self.is_face1_exhausted():
                return True
        return self.gt.value >= self.horizon_sec - 1e-6

    def step(self, dt: float) -> None:
        t_end = self.gt.value + dt
        while self.gt.value < t_end:
            if self.is_terminating_condition_met():
                break
            tte = min(self.time_to_event(), t_end - self.gt.value)
            t_next = self.gt.value + tte
            self._advance(t_next)
            self.on_event(t_next)

    def on_event(self, t: float) -> None:
        self._calendar_update(t)
        self._update_strategic_tactical_review()
        self._update_operating_mode_and_targets()

        # Multi-pass cascade event resolution
        guard = 0
        changed = True
        while changed and guard < 200:
            changed = False
            guard += 1
            for tr in self.trucks:
                if tr.phase in (TruckPhase.IDLE, TruckPhase.PARKED):
                    if self._try_dispatch(tr):
                        changed = True
                elif (
                    tr.phase in DUE_PHASES
                    and tr.timer.rate < 0.0
                    and tr.timer.value <= 1e-6
                ):
                    if self._advance_truck_state(tr):
                        changed = True

        # Periodic telemetry logging
        if t >= self._next_telemetry_t:
            self._record_telemetry(t)
            self._next_telemetry_t = t + self._telemetry_dt

    @property
    def area2_ready(self) -> bool:
        """True if Area 2 is unlocked and ready for extraction."""
        return (
            self._area2_unlocked
            or float(self.area2_cumulative_development.value)
            >= float(self.area2_readiness_target.required_development) - 1e-6
        )

    @property
    def mining_priority(self) -> Any:
        return self.tactical_controller.active_fleet_mode

    @property
    def tactical_review_count(self) -> drs.Level:
        return self.tactical_controller.tactical_review_count

    @property
    def strategic_year_index(self) -> drs.Level:
        return self.tactical_controller.strategic_year_index

    @property
    def strategic_year_timer(self) -> drs.Timer:
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
    def operating_npv_proxy(self) -> drs.Level:
        return self.plant.cumulative_npv

    @property
    def current_cash_flow_rate(self) -> drs.Variable:
        return self.plant.cash_flow_rate_per_day

    @property
    def current_discounted_cash_flow_rate(self) -> drs.Variable:
        return self.plant.discounted_cash_flow_rate_per_day

    @property
    def cumulative_cash_flow(self) -> drs.Level:
        return self.plant.cumulative_net_cash_flow

    @property
    def cumulative_discounted_cash_flow(self) -> drs.Level:
        return self.plant.cumulative_npv

    @property
    def discount_factor(self) -> drs.Variable:
        return self.plant._discount_factor

    @property
    def strategic_planning_started(self) -> bool:
        return self.tactical_controller.planning_started

    @strategic_planning_started.setter
    def strategic_planning_started(self, value: bool) -> None:
        self.tactical_controller.planning_started = value

    def _select_face_by_blend_need(self) -> int:
        return self.select_face_for_truck(self.trucks[0])

    def _record_telemetry(self, t: float) -> None:
        day = t / 86400.0
        mode_name = self.plant.active_operating_mode.value.name
        o1_stock = float(self.ore1_stock.level)
        o2_stock = float(self.ore2_stock.level)
        tot_stock = o1_stock + o2_stock
        tot_mined = float(self.total_extracted_ore.value)
        p_o1 = float(self.plant.cumulative_processed_ore1.value)
        p_o2 = float(self.plant.cumulative_processed_ore2.value)
        tot_processed = float(self.plant.cumulative_milled_mass.value)
        a2_locked = self.is_area2_locked(day)

        n_operating_f1 = sum(
            1
            for tr in self.trucks
            if tr.phase in OPERATING_PHASES
            and not tr.is_waste
            and getattr(tr, "target_face_id", 1) == 1
        )
        n_operating_f2 = sum(
            1
            for tr in self.trucks
            if tr.phase in OPERATING_PHASES
            and not tr.is_waste
            and getattr(tr, "target_face_id", 1) == 2
        )
        n_capital_dev = sum(
            1
            for tr in self.trucks
            if tr.phase in OPERATING_PHASES
            and getattr(tr, "mission_type", None) == MissionType.CAPITAL_DECLINE_DEV
        )
        n_stope_dev = sum(
            1
            for tr in self.trucks
            if tr.phase in OPERATING_PHASES
            and getattr(tr, "mission_type", None) == MissionType.STOPE_TURNAROUND_DEV
        )
        n_dev_total = n_capital_dev + n_stope_dev
        n_operating = n_operating_f1 + n_operating_f2
        n_refueling = sum(
            1 for tr in self.trucks if tr.phase == TruckPhase.REFUELING
        )
        n_idle = max(0, len(self.trucks) - (n_operating + n_dev_total + n_refueling))
        n_dev_reserved = float(self.development_priority_reserved_trucks.value)

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
            "active_operating_mode": self.plant.active_operating_mode.value,
            "active_operating_mode_name": mode_name,
            "fleet_mode": self.tactical_controller.active_fleet_mode.name,
            "mining_priority": self.tactical_controller.active_fleet_mode.name,
            "Mode A": 1.0 if "MODE_A" in mode_name else 0.0,
            "Mode B": 1.0 if "MODE_B" in mode_name else 0.0,
            "Shutdown": 1.0 if "SHUTDOWN" in mode_name else 0.0,
            "ore1_mined": float(self.ore1_dumped_total.value),
            "ore2_mined": float(self.ore2_dumped_total.value),
            "face1_mined": float(self.tonnes_hauled_by_face[1]),
            "face2_mined": float(self.tonnes_hauled_by_face[2]),
            "area1_mined": float(self.tonnes_hauled_by_face[1]),
            "area2_mined": float(self.tonnes_hauled_by_face[2]),
            "total_mined": tot_mined,

            "Ore1_Mined": float(self.ore1_dumped_total.value),
            "Ore2_Mined": float(self.ore2_dumped_total.value),
            "ore1_processed": p_o1,
            "ore2_processed": p_o2,
            "total_processed": tot_processed,
            "cumulative_milled_mass": tot_processed,
            "cumulative_extracted_mass": tot_mined,
            "cumulative_development": float(
                self.cumulative_mine_development.value
            ),
            "cumulative_mine_development": float(
                self.cumulative_mine_development.value
            ),
            "area2_cumulative_development": float(
                self.area2_cumulative_development.value
            ),
            "sustaining_cumulative_development": float(
                self.sustaining_cumulative_development.value
            ),
            "area2_readiness_fraction": float(
                self.area2_readiness_fraction.value
            ),
            "area2_trajectory_ratio": float(
                self.area2_readiness_trajectory_ratio.value
            ),
            "area2_readiness_trajectory_ratio": float(
                self.area2_readiness_trajectory_ratio.value
            ),
            "development_trajectory_ratio": float(
                self.development_trajectory_ratio.value
            ),
            "ore1_trajectory_ratio": float(self.ore1_trajectory_ratio.value),
            "ore2_trajectory_ratio": float(self.ore2_trajectory_ratio.value),
            "area2_ready_day": float(self.area2_ready_day.value),
            "area1_depleted_day": float(self.area1_depleted_day.value),
            "area2_depleted_day": float(self.area2_depleted_day.value),
            "area2_is_locked": float(a2_locked),
            "area2_ready": not a2_locked,
            "area1_exhausted": bool(self.is_face1_exhausted()),
            "area2_exhausted": bool(self.is_face2_exhausted()),
            "face1_exhausted": bool(self.is_face1_exhausted()),
            "face2_exhausted": bool(self.is_face2_exhausted()),
            "analytical_face1_weight": float(
                self.analytical_face1_weight.value
            ),
            "analytical_face2_weight": float(
                self.analytical_face2_weight.value
            ),
            "analytical_blend_feasible": float(
                self.analytical_blend_feasible.value
            ),
            "cumulative_cash_flow": float(
                self.plant.cumulative_net_cash_flow.value
            ),
            "cumulative_npv": float(self.plant.cumulative_npv.value),
            "operating_npv_proxy": float(self.plant.cumulative_npv.value),
            "cumulative_discounted_cash_flow": float(
                self.plant.cumulative_npv.value
            ),
            "discount_factor": float(self.plant._discount_factor.value),
            "current_cash_flow_rate": float(
                self.plant.cash_flow_rate_per_day.value
            ),
            "current_discounted_cash_flow_rate": float(
                self.plant.discounted_cash_flow_rate_per_day.value
            ),
            "daily_revenue": float(self.plant.daily_revenue_rate.value),
            "daily_cost": float(self.plant.daily_cost_rate.value),
            "daily_net_cash_flow": float(
                self.plant.cash_flow_rate_per_day.value
            ),
            "current_campaign_duration": float(
                self.mode_controller.current_campaign_duration.value
            ),
            "current_contingency_duration": float(
                self.plant.current_contingency_duration.value
            ),
            "cumulative_time_shutdown": float(
                self.plant.cumulative_time_shutdown.value
            ),
            "cumulative_time_mode_a": float(
                self.plant.cumulative_time_mode_a.value
            ),
            "cumulative_time_mode_b": float(
                self.plant.cumulative_time_mode_b.value
            ),
            "trucks_idle": n_idle,
            "trucks_operating": n_operating,
            "trucks_operating_face1": n_operating_f1,
            "trucks_operating_face2": n_operating_f2,
            "trucks_area1_operating": n_operating_f1,
            "trucks_area2_operating": n_operating_f2,
            "trucks_capital_dev": n_capital_dev,
            "trucks_stope_dev": n_stope_dev,
            "trucks_dev_total": n_dev_total,
            "trucks_refueling": n_refueling,
            "trucks_dev_reserved": n_dev_reserved,
            "truck_idle_fraction": n_idle / max(1, len(self.trucks)),
            "fleet_operating_mode": self.fleet_controller.mode.name,
            "fleet_mode": self.fleet_controller.mode.name,
            "tactical_review_count": float(
                self.tactical_controller.tactical_review_count.value
            ),
            "strategic_year_index": float(
                self.tactical_controller.strategic_year_index.value
            ),
        }
        self.telemetry_history.append(record)

    def results(self) -> Dict[str, Any]:
        """Returns structured simulation summary outputs and time-series dataframe."""
        if not self.telemetry_history or self.telemetry_history[-1]["time_sec"] < self.gt.value - 1e-6:
            self._record_telemetry(self.gt.value)
        df = pd.DataFrame(self.telemetry_history)
        return {
            "df": df,
            "total_mined_tonnes": float(self.total_extracted_ore.value),
            "ore1_mined_tonnes": float(self.ore1_dumped_total.value),
            "ore2_mined_tonnes": float(self.ore2_dumped_total.value),
            "cumulative_development_m": float(
                self.cumulative_mine_development.value
            ),
            "area2_cumulative_development_m": float(
                self.area2_cumulative_development.value
            ),
            "sustaining_cumulative_development_m": float(
                self.sustaining_cumulative_development.value
            ),
            "cumulative_npv": float(self.plant.cumulative_npv.value),
            "area2_ready_day": float(self.area2_ready_day.value),
            "area2_completed_late": self.face2.completed_late,
        }


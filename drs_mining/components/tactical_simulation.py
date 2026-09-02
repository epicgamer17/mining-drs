"""Pure DRS Tactical Mining Simulation.

Continuous rate-based integration with:
- N MineFace ore bodies (main + satellites)
- Dual stockpiles (Ore 1, Ore 2)
- Metallurgical plant with operating modes
- Simple routing from faces to stockpiles
- Single total throughput metric
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import drs
import pandas as pd

from drs_mining.config import (
    MILL_MODES,
    FLEET_MODES,
    CalendarConfig,
    PlantConfig,
    GeologyConfig,
    StrategicPlanningConfig,
    SimulationConfig,
    DEFAULT_CONFIG,
)
from drs_mining.components.modes import OperatingMode
from drs_mining.components.plant import MetallurgicalPlant
from drs_mining.components.stockpiles import Stockpile
from drs_mining.components.controllers import OperatingModeController
from drs_mining.components.mine_face import MineFace
from drs_mining.components.geology import StochasticReserve
from drs_mining.components.haulage import HaulRoute
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.planning import (
    AreaReadinessTarget,
    select_fleet_mode,
    TacticalReviewController,
)


_DEFAULT_CONFIG = DEFAULT_CONFIG
DAYS_IN_YEAR = _DEFAULT_CONFIG.calendar.days_in_year
DT_MAX = _DEFAULT_CONFIG.dt_max


class TacticalMiningSimulation(drs.Module):
    """Pure DRS tactical mining simulation with continuous rate-based integration.

    Coordinates N mine faces (main + satellites), dual stockpiles,
    metallurgical plant campaigns, and tactical planning reviews.
    """

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        faces: Optional[Sequence[MineFace]] = None,
        target_ore_stock_level: Optional[float] = None,
        critical_ore2_level: Optional[float] = None,
        total_ore_to_extract: Optional[float] = None,
        ore_to_be_extracted_during_warming_period: Optional[float] = None,
        duration_of_production_campaigns: Optional[float] = None,
        duration_of_shutdowns: Optional[float] = None,
        duration_of_contingency_segments: Optional[float] = None,
        tactical_review_period_days: Optional[float] = None,
        tactical_progress_tolerance: Optional[float] = None,
        area2_readiness_target: Optional[AreaReadinessTarget] = None,
        development_rate_m_per_day: float = 5.0,
        mode_a_ore1_milling_rate: Optional[float] = None,
        mode_a_ore2_milling_rate: Optional[float] = None,
        mode_a_contingency_ore1_milling_rate: Optional[float] = None,
        mode_b_ore1_milling_rate: Optional[float] = None,
        mode_b_ore2_milling_rate: Optional[float] = None,
        mode_b_contingency_ore2_milling_rate: Optional[float] = None,
        enable_analytical_blending: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        cfg = config or _DEFAULT_CONFIG
        self.config = cfg
        self.enable_analytical_blending = enable_analytical_blending

        self.target_ore_stock_level = (
            target_ore_stock_level
            if target_ore_stock_level is not None
            else cfg.plant.target_ore_stock_level
        )
        self.critical_ore2_level = (
            critical_ore2_level
            if critical_ore2_level is not None
            else cfg.plant.critical_ore2_level
        )
        self.total_ore_to_extract = (
            total_ore_to_extract
            if total_ore_to_extract is not None
            else cfg.plant.total_ore_to_extract
        )
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
            if ore_to_be_extracted_during_warming_period is not None
            else cfg.plant.ore_to_be_extracted_during_warming_period
        )
        self.development_rate_m_per_day = development_rate_m_per_day

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
        self.area2_readiness_target = area2_readiness_target or AreaReadinessTarget(
            required_development=cfg.planning.area2_required_development,
            ready_by_day=cfg.planning.area2_ready_by_day,
        )

        self.seed = seed if seed is not None else cfg.seed
        self.rng = random.Random(self.seed)

        # Global time tracker
        self.gt = drs.Timer("gt", 0.0, rate=1.0)

        # 1. Mine Faces
        if faces is not None:
            self.faces = list(faces)
        else:
            gen1 = StochasticFaciesGenerator(
                mean_fraction=cfg.geology.area1_mean_fraction,
                std_dev=cfg.geology.area1_std_dev,
                prob_new_facies=cfg.geology.prob_new_facies,
                variation_same_facies=cfg.geology.variation_same_facies,
            )
            face_capacity = self.total_ore_to_extract / 2.0
            warmup_cap = self.ore_to_be_extracted_during_warming_period / 2.0

            res1 = StochasticReserve(
                name="mine_face_1_reserve",
                total_tonnes=face_capacity,
                generator=gen1,
                min_parcel_mass=cfg.geology.min_parcel_mass,
                max_parcel_mass=cfg.geology.max_parcel_mass,
                warming_period=warmup_cap,
                seed=self.seed,
            )
            self.face1 = MineFace(
                name="mine_face_1",
                geology=res1,
                haulage=HaulRoute(distance_km=1.0),
            )

            gen2 = StochasticFaciesGenerator(
                mean_fraction=cfg.geology.area2_mean_fraction,
                std_dev=cfg.geology.area2_std_dev,
                prob_new_facies=cfg.geology.prob_new_facies,
                variation_same_facies=cfg.geology.variation_same_facies,
            )
            res2 = StochasticReserve(
                name="mine_face_2_reserve",
                total_tonnes=face_capacity,
                generator=gen2,
                min_parcel_mass=cfg.geology.min_parcel_mass,
                max_parcel_mass=cfg.geology.max_parcel_mass,
                warming_period=warmup_cap,
                seed=self.seed,
            )
            self.face2 = MineFace(
                name="mine_face_2",
                geology=res2,
                haulage=HaulRoute(distance_km=3.0),
            )
            self.faces = [self.face1, self.face2]

        # 2. Stockpiles
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

        # 3. Plant & Campaign Mode Controller
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
        )

        # 4. Strategic / Tactical Planning Controller
        self.tactical_controller = TacticalReviewController(
            tactical_review_period_days=self.tactical_review_period_days,
            tactical_progress_tolerance=self.tactical_progress_tolerance,
        )

        # Development tracking
        self.sustaining_cumulative_development = drs.Level(
            "sustaining_cumulative_development", 0.0
        )
        self.area2_cumulative_development = drs.Level(
            "area2_cumulative_development", 0.0
        )
        self.cumulative_mine_development = drs.Level("cumulative_mine_development", 0.0)

        # Throughput (single total metric)
        self.cumulative_throughput = drs.Level("cumulative_throughput", 0.0)

        # Analytical Blending Levels
        self.analytical_face1_weight = drs.Level("analytical_face1_weight", 1.0)
        self.analytical_face2_weight = drs.Level("analytical_face2_weight", 0.0)

        # Area 2 readiness
        self.area2_readiness_trajectory_ratio = drs.Level(
            "area2_readiness_trajectory_ratio", 1.0
        )
        self.area2_readiness_fraction = drs.Level("area2_readiness_fraction", 0.0)
        self.area2_ready_day = drs.Level("area2_ready_day", -1.0)
        self._area2_unlocked = False

        # Operational metrics
        self.daily_target_ore = 6000.0
        self._telemetry_dt = cfg.telemetry_dt
        self._next_telemetry_t = 0.0
        self.telemetry_history: List[Dict[str, Any]] = []

    def levels(self) -> Sequence[drs.Level]:
        base_levels: List[drs.Level] = [
            self.sustaining_cumulative_development,
            self.area2_cumulative_development,
            self.cumulative_mine_development,
            self.cumulative_throughput,
            self.analytical_face1_weight,
            self.analytical_face2_weight,
            self.area2_readiness_trajectory_ratio,
            self.area2_readiness_fraction,
            self.area2_ready_day,
        ]

        base_levels.extend(self.ore1_stock.levels())
        base_levels.extend(self.ore2_stock.levels())
        base_levels.extend(self.mode_controller.levels())
        base_levels.extend(self.plant.levels())
        base_levels.extend(self.tactical_controller.levels())

        for face in self.faces:
            base_levels.extend(face.levels())

        base_levels.append(self.gt)

        return tuple(base_levels)

    # -----------------------------------------------------------------------
    # Area 2 Unlock Callback
    # -----------------------------------------------------------------------
    def _on_area2_unlocked(self, day: float = 0.0) -> None:
        if not self._area2_unlocked:
            self._area2_unlocked = True
            self.area2_ready_day.value = day

    def is_area2_locked(self) -> bool:
        required = max(0.0, float(self.area2_readiness_target.required_development))
        if required <= 1e-12:
            return False
        return not self._area2_unlocked

    # -----------------------------------------------------------------------
    # Face Selection for Routing
    # -----------------------------------------------------------------------
    def _select_face_for_routing(self) -> int:
        """Selects which face to route ore from based on mode and availability."""
        if self.is_area2_locked():
            return 1

        f1_done = self._is_face_exhausted(1)
        f2_done = self._is_face_exhausted(2)

        if f1_done and not f2_done:
            return 2
        if f2_done and not f1_done:
            return 1
        if f1_done and f2_done:
            return 1

        # Check availability for face 2
        face2 = self._get_face(2)
        if not face2.is_ore_available:
            return 1

        mode_name = self.plant.active_operating_mode.value.name
        p_face2 = 0.65 if "MODE_A" in mode_name else 0.35
        return 2 if self.rng.random() < p_face2 else 1

    def _get_face(self, face_id: int) -> MineFace:
        return self.faces[face_id - 1]

    def _is_face_exhausted(self, face_id: int) -> bool:
        return self._get_face(face_id).is_exhausted

    # -----------------------------------------------------------------------
    # DRS Engine Interface
    # -----------------------------------------------------------------------
    def is_terminating_condition_met(self) -> bool:
        if len(self.faces) > 1:
            if self._is_face_exhausted(1) and self._is_face_exhausted(2):
                return True
        elif self._is_face_exhausted(1):
            return True
        return False

    def time_to_event(self) -> float:
        min_dt = DT_MAX
        t = self.gt.value

        # Next day boundary
        next_day = (math.floor(t / 86400.0) + 1.0) * 86400.0
        min_dt = min(min_dt, next_day - t)

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
        if self.plant.active_operating_mode.value.name in self.plant._CONTINGENCY_MODES:
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

    def step(self, dt: float) -> None:
        t_end = self.gt.value + dt
        while self.gt.value < t_end:
            if self.is_terminating_condition_met():
                break
            tte = min(self.time_to_event(), t_end - self.gt.value)
            t_next = self.gt.value + tte
            self._advance(t_next)
            self.on_event(t_next)

    def _advance(self, t_target: float) -> None:
        dt = t_target - self.gt.value
        if dt <= 0.0:
            return
        dt_days = dt / 86400.0

        # 1. Step Global & Campaign Timers
        self.gt.step(dt)
        self.mode_controller.current_campaign_duration.step(dt_days)
        active_mode_name = self.plant.active_operating_mode.value.name
        timer_attr = self.plant._MODE_TIMER_ATTRS[active_mode_name]
        getattr(self.plant, timer_attr).step(dt_days)

        if active_mode_name in self.plant._CONTINGENCY_MODES:
            self.plant.current_contingency_duration.step(dt_days)

        if self.tactical_controller.planning_started:
            self.tactical_controller.step_timers(dt_days)

        # 2. Compute blend fraction from active faces
        available_faces = [f for f in self.faces if f.is_ore_available]
        if available_faces:
            f_blend = sum(
                (
                    f.geology.active_parcel.ore2_fraction
                    if f.geology.active_parcel
                    else 0.0
                )
                for f in available_faces
            ) / len(available_faces)
        else:
            # No ore available
            f_blend = 0.0

        # 3. Plant target rates
        ore1_rate, ore2_rate, mine_target = self.plant.get_target_rates(
            self.mode_controller.active_campaign_mode.value,
            ore1_level=float(self.ore1_stock.level),
            ore2_level=float(self.ore2_stock.level),
            stockpile2_routing_fraction=f_blend,
        )

        mode_name = self.plant.active_operating_mode.value.name
        if mode_name == "SHUTDOWN":
            self.daily_target_ore = 0.0
        else:
            self.daily_target_ore = mine_target

        # 4. Route ore from faces to stockpiles (continuous rates)
        ore1_in_rate = 0.0
        ore2_in_rate = 0.0

        for face in self.faces:
            if not face.is_ore_available:
                continue

            # Allocate extraction rate proportional to face weight
            n_available = len(available_faces)
            if n_available > 0:
                face_target = self.daily_target_ore / n_available
            else:
                face_target = 0.0

            # Set face target rate (tonnes/day -> tonnes/sec)
            face.target_rate = face_target / 86400.0

            # Get actual extraction from face
            ore2_frac = (
                face.geology.active_parcel.ore2_fraction
                if face.geology.active_parcel
                else 0.0
            )
            ore1_frac = 1.0 - ore2_frac
            actual_rate = face.actual_rate

            ore1_in_rate += actual_rate * ore1_frac
            ore2_in_rate += actual_rate * ore2_frac

        # 5. Feed stockpiles and draw into plant
        ore1_draw_rate_sec = ore1_rate / 86400.0
        ore2_draw_rate_sec = ore2_rate / 86400.0

        out1 = self.ore1_stock.feed_and_draw(ore1_in_rate, ore1_draw_rate_sec)
        out2 = self.ore2_stock.feed_and_draw(ore2_in_rate, ore2_draw_rate_sec)
        self.ore1_stock.step(dt)
        self.ore2_stock.step(dt)

        self.plant.process(out1 + out2)
        self.plant.cumulative_milled_mass.step(dt)

        # 6. Update throughput
        throughput_tonnes = (out1 + out2) * dt
        self.cumulative_throughput.value += throughput_tonnes

        # 7. Advance development for area 2
        if self.tactical_controller.planning_started:
            fleet_mode = self.tactical_controller.fleet_mode
            if fleet_mode == FLEET_MODES["DEVELOPMENT"]:
                dev_delta = self.development_rate_m_per_day * dt_days
                self.area2_cumulative_development.value += dev_delta

        # 8. Update area 2 readiness
        self._update_area2_readiness()

        # 9. Step face levels
        for face in self.faces:
            face.step(dt)

        # 10. Periodic telemetry
        if t_target >= self._next_telemetry_t:
            self._record_telemetry(t_target)
            self._next_telemetry_t = t_target + self._telemetry_dt

    def on_event(self, t: float) -> None:
        day = int(t // 86400.0)

        # Update operating mode
        self.mode_controller.update(
            ore2_stock_level=float(self.ore2_stock.level),
            total_stock_level=float(self.ore1_stock.level + self.ore2_stock.level),
        )

        # Update tactical review
        self._update_tactical_review()

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

        current_day = self.gt.value / 86400.0
        progress = float(self.area2_cumulative_development.value)
        fraction = min(1.0, progress / required)
        self.area2_readiness_fraction.value = fraction

        if (not self._area2_unlocked) and progress >= required - 1e-6:
            self._on_area2_unlocked(current_day)

        ready_by_day = target.ready_by_day
        if ready_by_day is not None and ready_by_day > 0.0:
            elapsed_fraction = max(1e-4, min(1.0, current_day / ready_by_day))
            expected = required * elapsed_fraction
            if expected > 1e-12:
                self.area2_readiness_trajectory_ratio.value = (
                    max(0.0, progress) / expected
                )
            else:
                self.area2_readiness_trajectory_ratio.value = 1.0
        else:
            self.area2_readiness_trajectory_ratio.value = 1.0

    def _update_tactical_review(self):
        if not self.tactical_controller.planning_started:
            return

        r_area2 = float(self.area2_readiness_trajectory_ratio.value)
        self.tactical_controller.update_mode(
            area2_readiness_trajectory_ratio=r_area2,
        )

    # -----------------------------------------------------------------------
    # Telemetry
    # -----------------------------------------------------------------------
    def _record_telemetry(self, t: float) -> None:
        day = t / 86400.0
        mode_name = self.plant.active_operating_mode.value.name
        o1_stock = float(self.ore1_stock.level)
        o2_stock = float(self.ore2_stock.level)
        tot_stock = o1_stock + o2_stock
        tot_throughput = float(self.cumulative_throughput.value)

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
            "fleet_mode": self.tactical_controller.fleet_mode.name,
            "mining_priority": self.tactical_controller.fleet_mode.name,
            "Mode A": 1.0 if "MODE_A" in mode_name else 0.0,
            "Mode B": 1.0 if "MODE_B" in mode_name else 0.0,
            "Shutdown": 1.0 if "SHUTDOWN" in mode_name else 0.0,
            "total_throughput": tot_throughput,
            "cumulative_milled_mass": float(self.plant.cumulative_milled_mass.value),
            "cumulative_development": float(self.cumulative_mine_development.value),
            "area2_cumulative_development": float(
                self.area2_cumulative_development.value
            ),
            "area2_readiness_fraction": float(self.area2_readiness_fraction.value),
            "area2_trajectory_ratio": float(
                self.area2_readiness_trajectory_ratio.value
            ),
            "area2_ready_day": float(self.area2_ready_day.value),
            "area2_is_locked": float(self.is_area2_locked()),
            "area2_ready": not self.is_area2_locked(),
        }

        # Per-face metrics
        for face in self.faces:
            record[f"face{face.face_id}_mined"] = float(
                face.cumulative_extracted_mass.value
            )
            record[f"face{face.face_id}_available"] = (
                float(face.sporadic_available.value) if face.sporadic else 1.0
            )
            record[f"face{face.face_id}_ore_fraction"] = float(
                face.active_parcel_ore_fraction.value
            )

        self.telemetry_history.append(record)

    def results(self) -> Dict[str, Any]:
        if (
            not self.telemetry_history
            or self.telemetry_history[-1]["time_sec"] < self.gt.value - 1e-6
        ):
            self._record_telemetry(self.gt.value)
        df = pd.DataFrame(self.telemetry_history)
        return {
            "df": df,
            "total_throughput": float(self.cumulative_throughput.value),
            "cumulative_milled_mass": float(self.plant.cumulative_milled_mass.value),
            "cumulative_development_m": float(self.cumulative_mine_development.value),
            "area2_cumulative_development_m": float(
                self.area2_cumulative_development.value
            ),
            "area2_ready_day": float(self.area2_ready_day.value),
        }

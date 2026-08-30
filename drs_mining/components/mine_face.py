"""Unified underground mine face & stope component with parcel geology, readiness & lifecycle mechanics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union, Dict, Any, Tuple, Callable

import drs
from drs import Processor
from .generators import StochasticFaciesGenerator


class FaceState(Enum):
    """Operational lifecycle state of an underground mining stope / face."""

    LOCKED = "LOCKED"  # Still undergoing capital development; not yet accessible
    ORE_READY = "ORE_READY"  # Blasted ore available for mucking & haulage
    DEVELOPMENT_TURNAROUND = (
        "DEVELOPMENT_TURNAROUND"  # Waste rock mucking & turnaround heading advance
    )
    EXHAUSTED = "EXHAUSTED"  # Total reserve depleted; permanently decommissioned


# Compatibility alias
StopeState = FaceState


@dataclass
class StopeParcel:
    """Represents a discrete extraction round in a face/stope containing ore and waste rock."""

    parcel_index: int
    ore_mass: float  # Tonnes of ore in this round
    ore1_fraction: float  # Ore 1 grade fraction (0.0 to 1.0)
    ore2_fraction: float  # Ore 2 grade fraction (0.0 to 1.0)
    waste_rock_mass: float  # Tonnes of waste rock that must be extracted before next round
    required_dev_m: float  # Equivalent development advance in metres


class MineFace(Processor):
    """Unified mine face & stope component with parcel geology, capital readiness & lifecycle mechanics."""

    def __init__(
        self,
        name: str,
        face_id: int = 1,
        area_id: int = 1,
        level_index: int = 3,
        generator: Optional[StochasticFaciesGenerator] = None,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        initial_parcel_mass: float = 40000.0,
        max_rate: float = math.inf,
        # Capital development & physical readiness
        required_development: float = 0.0,
        ready_by_day: Optional[float] = None,
        on_unlock_callback: Optional[Callable[[], None]] = None,
        counterfactual_disable: bool = False,
        # Stope lifecycle & turnaround parameters
        total_stope_reserve: Optional[float] = None,
        min_parcel_ore_mass: Optional[float] = None,
        max_parcel_ore_mass: Optional[float] = None,
        waste_to_ore_ratio: float = 0.0,
        turnaround_dev_per_parcel_m: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.face_id = face_id
        self.area_id = area_id
        self.level_index = level_index
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies

        self.min_ore_mass = min_parcel_ore_mass or min_ore_mass
        self.max_ore_mass = max_parcel_ore_mass or max_ore_mass
        self.min_parcel_ore_mass = self.min_ore_mass
        self.max_parcel_ore_mass = self.max_ore_mass

        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
        )
        self.total_stope_reserve = total_stope_reserve or total_ore_to_extract

        self.waste_to_ore_ratio = waste_to_ore_ratio
        self.turnaround_dev_per_parcel_m = turnaround_dev_per_parcel_m

        self.required_development = required_development
        self.ready_by_day = ready_by_day
        self.on_unlock_callback = on_unlock_callback
        self.counterfactual_disable = counterfactual_disable

        self.rng = random.Random((seed or 42) + face_id * 100)
        self.generator = generator or StochasticFaciesGenerator(
            mean_fraction=mean_ore_fraction,
            std_dev=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
        )

        # Dynamic State Variables
        if counterfactual_disable or (required_development > 0.0):
            self.state = FaceState.LOCKED
        else:
            self.state = FaceState.ORE_READY
        self.parcel_index = 0

        # DRS Observable Levels & Variables
        var_name = (
            f"face{face_id}_ore_fraction"
            if face_id != 1
            else "active_parcel_ore_fraction"
        )
        self.active_parcel_ore_fraction = drs.Variable(var_name, self.mean_ore_fraction)
        self.active_parcel_initial_mass = drs.Variable(
            f"{name}_active_parcel_initial_mass", initial_parcel_mass
        )
        self.active_parcel_ore_mass = self.active_parcel_initial_mass
        self.active_parcel_waste_mass = drs.Variable(f"{name}_parcel_waste_mass", 0.0)
        self.required_turnaround_dev_m = drs.Variable(f"{name}_req_dev_m", 0.0)

        # Extraction Counters
        self.cumulative_extracted_mass = drs.Level(
            f"{name}_cumulative_extracted_mass", initial_value=0.0
        )
        self.parcel_extracted_mass = drs.Level(
            f"{name}_parcel_extracted_mass", initial_value=0.0
        )
        self.parcel_ore_extracted = self.parcel_extracted_mass
        self.parcel_waste_extracted = drs.Level(
            f"{name}_parcel_waste_extracted", initial_value=0.0
        )
        self.cumulative_ore_extracted = self.cumulative_extracted_mass
        self.cumulative_waste_extracted = drs.Level(
            f"{name}_cum_waste_extracted", initial_value=0.0
        )
        self.cumulative_stope_dev_m = drs.Level(
            f"{name}_cum_stope_dev_m", initial_value=0.0
        )

        # Physical capital development & readiness tracking
        self.cumulative_development = drs.Level(
            f"face{face_id}_cumulative_development", initial_value=0.0
        )
        initial_readiness = 1.0 if (required_development <= 0.0 and not counterfactual_disable) else 0.0
        self.readiness_fraction = drs.Level(
            f"face{face_id}_readiness_fraction", initial_value=initial_readiness
        )
        self.readiness_trajectory_ratio = drs.Level(
            f"face{face_id}_readiness_trajectory_ratio", initial_value=1.0
        )
        self.ready_day = drs.Level(
            f"face{face_id}_ready_day",
            initial_value=0.0 if (required_development <= 0.0 and not counterfactual_disable) else -1.0,
        )
        self.deadline_missed: bool = False
        self.currently_late: bool = False
        self.completed_late: bool = False

        # Load initial batch/parcel
        self._load_next_batch()

    # -----------------------------------------------------------------------
    # Readiness & Physical Unlock Mechanics
    # -----------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """Returns True if face development requirement is fully satisfied."""
        if self.counterfactual_disable:
            return False
        if self.required_development <= 0.0:
            return True
        return self.cumulative_development.value >= self.required_development - 1e-6

    @is_ready.setter
    def is_ready(self, value: bool) -> None:
        if value:
            self.state = FaceState.ORE_READY


    def is_locked(self, current_day: float = 0.0) -> bool:
        """Returns True if the face is still undergoing capital development and cannot be mined."""
        if self.counterfactual_disable:
            return True
        return not self.is_ready

    def update_status(self, current_day: float) -> bool:
        """Update readiness fraction, trajectory ratio, and check unlock triggers."""
        if self.counterfactual_disable:
            return False

        req_dev = self.required_development
        deadline = self.ready_by_day

        if req_dev <= 1e-6:
            if self.state == FaceState.LOCKED:
                self.state = FaceState.ORE_READY
                self.ready_day.value = current_day
                self.readiness_fraction.value = 1.0
                self.readiness_trajectory_ratio.value = 1.0
                if self.on_unlock_callback:
                    self.on_unlock_callback()
                return True
            return False

        dev = float(self.cumulative_development.value)
        frac = min(1.0, dev / req_dev)
        self.readiness_fraction.value = frac

        # Unlocked condition
        if dev >= req_dev - 1e-6:
            just_unlocked = False
            if self.state == FaceState.LOCKED:
                self.state = FaceState.ORE_READY
                just_unlocked = True
                self.ready_day.value = current_day
                if deadline is not None and current_day > deadline:
                    self.completed_late = True
                    self.deadline_missed = True
                if self.on_unlock_callback:
                    self.on_unlock_callback()
            self.currently_late = False
            self.readiness_trajectory_ratio.value = 1.0
            return just_unlocked

        # Still locked - evaluate schedule trajectory
        if deadline is not None and deadline > 0:
            if current_day > deadline:
                self.currently_late = True
                self.deadline_missed = True

            elapsed_frac = max(1e-4, min(1.0, current_day / deadline))
            expected = req_dev * elapsed_frac
            self.readiness_trajectory_ratio.value = (
                max(0.0, dev) / expected if expected > 1e-9 else 1.0
            )
        else:
            self.readiness_trajectory_ratio.value = 1.0

        return False

    def advance_development(
        self, delta_meters: float, current_day: float = 0.0
    ) -> bool:
        """Advances physical capital development and unlocks the face when complete."""
        if self.counterfactual_disable or self.required_development <= 0.0:
            return False

        if not self.is_ready:
            self.cumulative_development.value += max(0.0, delta_meters)

        return self.update_status(current_day)

    # -----------------------------------------------------------------------
    # Stope Lifecycle & Turnaround Mechanics
    # -----------------------------------------------------------------------
    @property
    def is_ore_available(self) -> bool:
        """True if the face is currently in ORE_READY state with remaining ore."""
        return (
            self.state == FaceState.ORE_READY
            and (
                float(self.parcel_extracted_mass.value)
                < float(self.active_parcel_initial_mass.value) - 1e-6
            )
            and not self.is_exhausted
        )

    @property
    def is_in_turnaround(self) -> bool:
        """True if face is undergoing development / waste rock turnaround."""
        return self.state == FaceState.DEVELOPMENT_TURNAROUND

    @property
    def is_exhausted(self) -> bool:
        """True if the face has extracted its full life-of-mine reserve."""
        return self.state == FaceState.EXHAUSTED or (
            float(self.cumulative_extracted_mass.value)
            >= self.total_stope_reserve - 1e-6
        )

    @property
    def remaining_reserve(self) -> float:
        """Remaining unextracted ore reserve in this face/stope."""
        return max(
            0.0,
            self.total_stope_reserve - float(self.cumulative_extracted_mass.value),
        )

    @property
    def remaining_parcel_ore(self) -> float:
        """Remaining ore in the active blasted parcel."""
        if self.state != FaceState.ORE_READY:
            return 0.0
        return max(
            0.0,
            float(self.active_parcel_initial_mass.value)
            - float(self.parcel_extracted_mass.value),
        )

    @property
    def remaining_turnaround_dev(self) -> float:
        """Remaining development advance metres required to complete turnaround."""
        if self.state != FaceState.DEVELOPMENT_TURNAROUND:
            return 0.0
        return max(
            0.0,
            float(self.required_turnaround_dev_m.value)
            - float(self.parcel_waste_extracted.value),
        )

    def extract_ore(self, payload_tonnes: float) -> Tuple[float, float, float]:
        """Extracts ore from the active parcel. Returns (extracted_tonnes, ore1_mass, ore2_mass)."""
        if not self.is_ore_available:
            return 0.0, 0.0, 0.0

        rem_parcel = self.remaining_parcel_ore
        rem_res = self.remaining_reserve
        actual_tonnes = min(payload_tonnes, rem_parcel, rem_res)

        if actual_tonnes <= 1e-6:
            self._check_ore_depletion()
            return 0.0, 0.0, 0.0

        self.parcel_extracted_mass.value += actual_tonnes
        self.cumulative_extracted_mass.value += actual_tonnes

        f = float(self.active_parcel_ore_fraction.value)
        ore2_mass = actual_tonnes * f
        ore1_mass = actual_tonnes * (1.0 - f)

        self._check_ore_depletion()
        return actual_tonnes, ore1_mass, ore2_mass

    def _check_ore_depletion(self) -> None:
        """Transitions to DEVELOPMENT_TURNAROUND or EXHAUSTED when parcel ore is depleted."""
        if float(self.cumulative_extracted_mass.value) >= self.total_stope_reserve - 1e-6:
            self.state = FaceState.EXHAUSTED
            return

        if float(self.parcel_extracted_mass.value) >= float(self.active_parcel_initial_mass.value) - 1e-6:
            if self.turnaround_dev_per_parcel_m > 0 or self.waste_to_ore_ratio > 0:
                self.state = FaceState.DEVELOPMENT_TURNAROUND
                self.parcel_waste_extracted.value = 0.0
            else:
                self._load_next_batch()

    def advance_turnaround(
        self, dev_advance_m: float = 0.0, waste_tonnes: float = 0.0
    ) -> bool:
        """Advances turnaround waste rock / slot raise development."""
        if self.state != FaceState.DEVELOPMENT_TURNAROUND:
            return False

        if dev_advance_m > 0:
            self.parcel_waste_extracted.value += dev_advance_m
            self.cumulative_stope_dev_m.value += dev_advance_m

        if waste_tonnes > 0:
            self.cumulative_waste_extracted.value += waste_tonnes

        req = float(self.required_turnaround_dev_m.value)
        if float(self.parcel_waste_extracted.value) >= req - 1e-6:
            self._load_next_batch()
            return True

        return False

    def advance_turnaround_development(
        self, dev_advance_m: float
    ) -> Tuple[float, bool]:
        """Advances turnaround development meters. Returns (dev_advanced, is_complete)."""
        is_complete = self.advance_turnaround(dev_advance_m=dev_advance_m)
        return dev_advance_m, is_complete

    # -----------------------------------------------------------------------
    # Parcel Generation
    # -----------------------------------------------------------------------
    def _load_next_batch(self):
        """Generates next parcel of blasted ore and associated waste rock turnaround."""
        self.parcel_index += 1
        rem_res = self.remaining_reserve
        if rem_res <= 1e-6:
            self.state = FaceState.EXHAUSTED
            return

        p_mass = self.rng.uniform(self.min_ore_mass, self.max_ore_mass)
        p_mass = min(p_mass, rem_res)

        self.active_parcel_initial_mass.value = p_mass
        self.parcel_extracted_mass.value = 0.0

        parcel = self.generator.generate_next()
        if isinstance(parcel, dict):
            self.active_parcel_ore_fraction.value = float(parcel.get("ore1_frac", self.mean_ore_fraction))
        elif hasattr(parcel, "ore1_frac"):
            self.active_parcel_ore_fraction.value = float(parcel.ore1_frac)
        elif hasattr(parcel, "value"):
            self.active_parcel_ore_fraction.value = float(parcel.value)
        else:
            self.active_parcel_ore_fraction.value = float(parcel)

        self.active_parcel_waste_mass.value = p_mass * self.waste_to_ore_ratio
        self.required_turnaround_dev_m.value = self.turnaround_dev_per_parcel_m

        if not self.is_locked():
            self.state = FaceState.ORE_READY

    @property
    def current_ore_grade(self) -> float:
        """Ore 2 (high-grade) fraction of the currently active parcel."""
        return float(self.active_parcel_ore_fraction.value)

    @property
    def net_extracted_mass(self) -> float:
        return (
            self.cumulative_extracted_mass.value
            - self.ore_to_be_extracted_during_warming_period
        )

    def is_terminating_condition_met(self) -> bool:
        return self.cumulative_extracted_mass.value >= self.total_ore_to_extract

    def advance_parcel_state(self):
        """Advance parcel mechanics and set continuous level thresholds."""
        if (
            self.parcel_extracted_mass.value
            >= self.active_parcel_initial_mass.value - 1e-6
        ):
            self._check_ore_depletion()

        if (
            self.cumulative_extracted_mass.value
            < self.ore_to_be_extracted_during_warming_period
        ):
            self.cumulative_extracted_mass.upper_threshold = (
                self.ore_to_be_extracted_during_warming_period
            )
        else:
            self.cumulative_extracted_mass.upper_threshold = self.total_ore_to_extract

        self.parcel_extracted_mass.upper_threshold = (
            self.active_parcel_initial_mass.value
        )

    def step(self, dt: float) -> None:
        """Apply the face's continuous mechanics for one engine step."""
        self.advance_parcel_state()
        actual = self.actual_rate
        self.cumulative_extracted_mass.rate = actual
        self.parcel_extracted_mass.rate = actual
        super().step(dt)

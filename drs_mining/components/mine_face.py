"""Mine face component with parcel geology, readiness & lifecycle mechanics."""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import Optional, Callable, Sequence

import drs
from drs import Processor
from .generators import StochasticFaciesGenerator


class FaceState(Enum):
    """Operational lifecycle state of an underground mining face."""

    LOCKED = "LOCKED"
    ORE_READY = "ORE_READY"
    EXHAUSTED = "EXHAUSTED"


class MineFace(Processor):
    """Mine face with parcel geology, capital readiness & lifecycle mechanics.

    Supports both main ore bodies and satellite ore bodies:
    - Main faces: always available, no development required
    - Satellite faces: may require development, sporadic availability, variable waste fractions
    """

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
        initial_parcel_mass: Optional[float] = None,
        max_rate: float = math.inf,
        required_development: float = 0.0,
        ready_by_day: Optional[float] = None,
        on_unlock_callback: Optional[Callable[[], None]] = None,
        counterfactual_disable: bool = False,
        waste_to_ore_ratio: float = 0.0,
        seed: Optional[int] = None,
        sporadic: bool = False,
        availability_probability: float = 1.0,
        min_waste_fraction: float = 0.0,
        max_waste_fraction: float = 0.3,
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.face_id = face_id
        self.area_id = area_id
        self.level_index = level_index
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies

        self.min_ore_mass = min_ore_mass
        self.max_ore_mass = max_ore_mass

        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
        )

        self.waste_to_ore_ratio = waste_to_ore_ratio

        self.required_development = required_development
        self.ready_by_day = ready_by_day
        self.on_unlock_callback = on_unlock_callback
        self.counterfactual_disable = counterfactual_disable

        self.sporadic = sporadic
        self.availability_probability = availability_probability
        self.min_waste_fraction = min_waste_fraction
        self.max_waste_fraction = max_waste_fraction

        self.rng = random.Random(seed + face_id * 100) if seed is not None else random
        self.generator = generator or StochasticFaciesGenerator(
            mean_fraction=mean_ore_fraction,
            std_dev=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
        )

        if counterfactual_disable or (required_development > 0.0):
            self.state = FaceState.LOCKED
        else:
            self.state = FaceState.ORE_READY
        self.parcel_index = 0

        self._sporadic_available = True
        if sporadic:
            self._sporadic_available = random.random() < availability_probability
        self.sporadic_available = drs.Variable(
            f"face{face_id}_sporadic_available",
            1.0 if self._sporadic_available else 0.0,
        )

        self.active_parcel_ore_fraction = drs.Variable(
            f"face{face_id}_ore_fraction"
            if face_id != 1
            else "active_parcel_ore_fraction",
            self.mean_ore_fraction,
        )
        self.active_parcel_initial_mass = drs.Variable(
            f"{name}_active_parcel_initial_mass",
            initial_parcel_mass if initial_parcel_mass is not None else 0.0,
        )

        self.cumulative_extracted_mass = drs.Level(
            f"{name}_cumulative_extracted_mass", initial_value=0.0
        )
        self.parcel_extracted_mass = drs.Level(
            f"{name}_parcel_extracted_mass", initial_value=0.0
        )

        self.cumulative_development = drs.Level(
            f"face{face_id}_cumulative_development", initial_value=0.0
        )
        initial_readiness = (
            1.0 if (required_development <= 0.0 and not counterfactual_disable) else 0.0
        )
        self.readiness_fraction = drs.Level(
            f"face{face_id}_readiness_fraction", initial_value=initial_readiness
        )
        self.ready_day = drs.Level(
            f"face{face_id}_ready_day",
            initial_value=(
                0.0
                if (required_development <= 0.0 and not counterfactual_disable)
                else -1.0
            ),
        )

        if initial_parcel_mass is None:
            self._load_next_batch()

    def check_sporadic_availability(self, rng: Optional[random.Random] = None) -> bool:
        if not self.sporadic:
            self._sporadic_available = True
            self.sporadic_available.value = 1.0
            return True

        r = (rng or self.rng).random()
        self._sporadic_available = r < self.availability_probability
        self.sporadic_available.value = 1.0 if self._sporadic_available else 0.0
        return self._sporadic_available

    @property
    def is_sporadic_available(self) -> bool:
        return self._sporadic_available

    @property
    def is_ready(self) -> bool:
        if self.counterfactual_disable:
            return False
        if self.required_development <= 0.0:
            return True
        return self.cumulative_development.value >= self.required_development - 1e-6

    @is_ready.setter
    def is_ready(self, value: bool) -> None:
        if value:
            self.state = FaceState.ORE_READY

    def is_locked(self) -> bool:
        if self.counterfactual_disable:
            return True
        return not self.is_ready

    def advance_development(self, delta_meters: float) -> bool:
        if self.counterfactual_disable or self.required_development <= 0.0:
            return False

        if not self.is_ready:
            self.cumulative_development.value += max(0.0, delta_meters)

        if self.is_ready and self.state == FaceState.LOCKED:
            self.state = FaceState.ORE_READY
            if self.on_unlock_callback:
                self.on_unlock_callback()
            return True
        return False

    @property
    def is_ore_available(self) -> bool:
        if self.sporadic and not self._sporadic_available:
            return False
        return (
            self.state == FaceState.ORE_READY
            and (
                float(self.parcel_extracted_mass.value)
                < float(self.active_parcel_initial_mass.value) - 1e-6
            )
            and not self.is_exhausted
        )

    @property
    def is_exhausted(self) -> bool:
        return (
            float(self.cumulative_extracted_mass.value)
            >= self.total_ore_to_extract - 1e-6
        )

    @property
    def remaining_reserve(self) -> float:
        return max(
            0.0,
            self.total_ore_to_extract - float(self.cumulative_extracted_mass.value),
        )

    @property
    def net_extracted_mass(self) -> float:
        return (
            self.cumulative_extracted_mass.value
            - self.ore_to_be_extracted_during_warming_period
        )

    def _check_ore_depletion(self) -> None:
        if (
            float(self.parcel_extracted_mass.value)
            >= float(self.active_parcel_initial_mass.value) - 1e-6
        ):
            self._load_next_batch()

    def _load_next_batch(self):
        self.parcel_index += 1
        if self.remaining_reserve <= 1e-6:
            return

        p_mass = self.rng.uniform(self.min_ore_mass, self.max_ore_mass)
        p_mass = min(p_mass, self.remaining_reserve)

        self.active_parcel_initial_mass.value = p_mass
        self.parcel_extracted_mass.value = 0.0

        parcel = self.generator.generate_next()
        if isinstance(parcel, dict):
            self.active_parcel_ore_fraction.value = float(
                parcel.get("ore1_frac", self.mean_ore_fraction)
            )
        elif hasattr(parcel, "ore1_frac"):
            self.active_parcel_ore_fraction.value = float(parcel.ore1_frac)
        elif hasattr(parcel, "value"):
            self.active_parcel_ore_fraction.value = float(parcel.value)
        else:
            self.active_parcel_ore_fraction.value = float(parcel)

        if not self.is_locked():
            self.state = FaceState.ORE_READY

    def advance_parcel_state(self):
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

    def levels(self) -> Sequence[drs.Level]:
        return (
            self.cumulative_extracted_mass,
            self.parcel_extracted_mass,
            self.cumulative_development,
            self.readiness_fraction,
            self.ready_day,
        )

    def time_to_event(self) -> float:
        min_dt = math.inf
        for lvl in self.levels():
            dt = lvl.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt

    def is_terminating_condition_met(self) -> bool:
        return self.cumulative_extracted_mass.value >= self.total_ore_to_extract

    def step(self, dt: float) -> None:
        self.advance_parcel_state()
        actual = self.actual_rate
        self.cumulative_extracted_mass.rate = actual
        self.parcel_extracted_mass.rate = actual
        for lvl in self.levels():
            lvl.step(dt)

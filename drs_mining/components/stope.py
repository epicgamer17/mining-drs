"""Underground Mining Stope Component with Multi-Phase Lifecycle, Waste Rock & Depletion.

Implements realistic underground stope cycle mechanics:
  1. ORE_READY: Stope round has blasted ore ready for LHD mucking and truck haulage to surface.
  2. DEVELOPMENT_TURNAROUND: Ore is fully mucked out. Stope requires waste rock extraction,
     slot raising, ground support, and drilling/blasting before the next ore round can be accessed.
  3. EXHAUSTED: Stope has extracted its total allocated life-of-mine reserve and is permanently closed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Dict, Any

import drs
from drs import Processor
from drs_mining.components.generators import StochasticFaciesGenerator


class StopeState(Enum):
    """Operational lifecycle state of an underground mining stope / face."""

    ORE_READY = "ORE_READY"  # Blasted ore available for mucking & haulage
    DEVELOPMENT_TURNAROUND = "DEVELOPMENT_TURNAROUND"  # Waste rock mucking & turnaround heading advance
    EXHAUSTED = "EXHAUSTED"  # Total reserve depleted; permanently decommissioned


@dataclass
class StopeParcel:
    """Represents a discrete extraction round in a stope containing ore and waste rock."""

    parcel_index: int
    ore_mass: float  # Tonnes of ore in this round
    ore1_fraction: float  # Ore 1 grade fraction (0.0 to 1.0)
    ore2_fraction: float  # Ore 2 grade fraction (0.0 to 1.0)
    waste_rock_mass: float  # Tonnes of waste rock that must be extracted before next round
    required_dev_m: float  # Equivalent development advance in metres (e.g. 20-50m)


class StopeFace(Processor):
    """An underground stope face with multi-phase lifecycle, waste rock, and finite reserves."""

    def __init__(
        self,
        name: str,
        face_id: int,
        area_id: int,
        level_index: int,
        generator: StochasticFaciesGenerator,
        mean_ore_fraction: float,
        std_dev_ore_fraction: float,
        total_stope_reserve: float = 600000.0,
        min_parcel_ore_mass: float = 25000.0,
        max_parcel_ore_mass: float = 45000.0,
        waste_to_ore_ratio: float = 0.20,
        turnaround_dev_per_parcel_m: float = 30.0,
        seed: int = 42,
    ):
        super().__init__(name=name, max_rate=math.inf)
        self.face_id = face_id
        self.area_id = area_id
        self.level_index = level_index
        self.generator = generator
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.total_stope_reserve = total_stope_reserve
        self.min_parcel_ore_mass = min_parcel_ore_mass
        self.max_parcel_ore_mass = max_parcel_ore_mass
        self.waste_to_ore_ratio = waste_to_ore_ratio
        self.turnaround_dev_per_parcel_m = turnaround_dev_per_parcel_m
        self.rng = random.Random(seed + face_id * 100)

        # Dynamic State Variables
        self.state = StopeState.ORE_READY
        self.parcel_index = 0

        # DRS Observable Levels & Variables
        self.active_parcel_ore_fraction = drs.Variable(
            f"{name}_ore_fraction", mean_ore_fraction
        )
        self.active_parcel_ore_mass = drs.Variable(
            f"{name}_parcel_ore_mass", 0.0
        )
        self.active_parcel_waste_mass = drs.Variable(
            f"{name}_parcel_waste_mass", 0.0
        )
        self.required_turnaround_dev_m = drs.Variable(
            f"{name}_req_dev_m", 0.0
        )

        self.parcel_ore_extracted = drs.Level(f"{name}_parcel_ore_extracted", 0.0)
        self.parcel_waste_extracted = drs.Level(
            f"{name}_parcel_waste_extracted", 0.0
        )
        self.cumulative_ore_extracted = drs.Level(
            f"{name}_cum_ore_extracted", 0.0
        )
        self.cumulative_waste_extracted = drs.Level(
            f"{name}_cum_waste_extracted", 0.0
        )
        self.cumulative_stope_dev_m = drs.Level(
            f"{name}_cum_stope_dev_m", 0.0
        )

        # Generate the initial parcel
        self._generate_next_parcel()

    @property
    def is_ore_available(self) -> bool:
        """True if the stope is currently in ORE_READY state with remaining ore."""
        return (
            self.state == StopeState.ORE_READY
            and (
                float(self.parcel_ore_extracted.value)
                < float(self.active_parcel_ore_mass.value) - 1e-6
            )
            and not self.is_exhausted
        )

    @property
    def is_in_turnaround(self) -> bool:
        """True if stope is undergoing development/waste rock turnaround."""
        return self.state == StopeState.DEVELOPMENT_TURNAROUND

    @property
    def is_exhausted(self) -> bool:
        """True if the stope has extracted its full life-of-mine reserve."""
        return self.state == StopeState.EXHAUSTED or (
            float(self.cumulative_ore_extracted.value)
            >= self.total_stope_reserve - 1e-6
        )

    @property
    def remaining_reserve(self) -> float:
        """Remaining unextracted ore reserve in this stope."""
        return max(
            0.0,
            self.total_stope_reserve
            - float(self.cumulative_ore_extracted.value),
        )

    @property
    def remaining_parcel_ore(self) -> float:
        """Remaining ore in the active blasted parcel."""
        if self.state != StopeState.ORE_READY:
            return 0.0
        return max(
            0.0,
            float(self.active_parcel_ore_mass.value)
            - float(self.parcel_ore_extracted.value),
        )

    @property
    def remaining_turnaround_dev(self) -> float:
        """Remaining development advance metres required to complete turnaround."""
        if self.state != StopeState.DEVELOPMENT_TURNAROUND:
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

        self.parcel_ore_extracted.value += actual_tonnes
        self.cumulative_ore_extracted.value += actual_tonnes

        f = float(self.active_parcel_ore_fraction.value)
        ore2_mass = actual_tonnes * f
        ore1_mass = actual_tonnes * (1.0 - f)

        self._check_ore_depletion()
        return actual_tonnes, ore1_mass, ore2_mass

    def advance_turnaround_development(
        self, dev_metres: float
    ) -> Tuple[float, bool]:
        """Advances stope turnaround development heading. Returns (actual_dev_m, is_turnaround_complete)."""
        if self.state != StopeState.DEVELOPMENT_TURNAROUND:
            return 0.0, False

        needed_m = self.remaining_turnaround_dev
        actual_m = min(dev_metres, needed_m)
        self.parcel_waste_extracted.value += actual_m
        self.cumulative_stope_dev_m.value += actual_m

        # Estimate waste rock mass from meters (approx 25 tonnes/meter for 4.5x4.5m heading)
        waste_t = actual_m * 25.0
        self.cumulative_waste_extracted.value += waste_t

        if (
            self.parcel_waste_extracted.value
            >= self.required_turnaround_dev_m.value - 1e-6
        ):
            # Turnaround complete! Transition to next ore parcel
            self._transition_to_next_ore_parcel()
            return actual_m, True

        return actual_m, False

    def _check_ore_depletion(self):
        """Checks if current ore round or stope reserve is depleted."""
        if (
            self.cumulative_ore_extracted.value
            >= self.total_stope_reserve - 1e-6
        ):
            self.state = StopeState.EXHAUSTED
            return

        if (
            self.parcel_ore_extracted.value
            >= self.active_parcel_ore_mass.value - 1e-6
        ):
            # Ore round depleted: transition to turnaround development
            self.state = StopeState.DEVELOPMENT_TURNAROUND
            self.parcel_waste_extracted.value = 0.0

    def _generate_next_parcel(self):
        """Generates a new stope parcel with stochastic ore and waste rock components."""
        self.parcel_index += 1
        rem_reserve = self.remaining_reserve
        if rem_reserve <= 1e-6:
            self.state = StopeState.EXHAUSTED
            return

        ore_tonnes = self.rng.uniform(
            self.min_parcel_ore_mass, self.max_parcel_ore_mass
        )
        ore_tonnes = min(ore_tonnes, rem_reserve)
        self.active_parcel_ore_mass.value = ore_tonnes

        # Sample grade from facies generator
        parcel_val = self.generator.generate_next()
        if isinstance(parcel_val, dict):
            f = float(parcel_val.get("ore1_frac", self.mean_ore_fraction))
        elif hasattr(parcel_val, "value"):
            f = float(parcel_val.value)
        else:
            f = float(parcel_val)
        self.active_parcel_ore_fraction.value = f

        # Waste rock and turnaround development component
        waste_tonnes = ore_tonnes * self.waste_to_ore_ratio
        self.active_parcel_waste_mass.value = waste_tonnes
        self.required_turnaround_dev_m.value = (
            self.turnaround_dev_per_parcel_m
            * (ore_tonnes / self.max_parcel_ore_mass)
        )

        self.parcel_ore_extracted.value = 0.0
        self.parcel_waste_extracted.value = 0.0
        self.state = StopeState.ORE_READY

    def _transition_to_next_ore_parcel(self):
        """Called upon completing turnaround development to blast the next ore round."""
        self._generate_next_parcel()

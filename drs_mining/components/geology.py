"""Geological sources, parcels, and reserve models for mining faces."""

from __future__ import annotations

import abc
import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import drs
from .generators import StochasticFaciesGenerator


@dataclass
class Parcel:
    """A discrete geological parcel extracted from a face."""

    mass: float
    ore1_fraction: float
    ore2_fraction: float
    waste_fraction: float = 0.0


class GeologySource(abc.ABC):
    """Abstract base class for face geology and reserve depletion."""

    @property
    @abc.abstractmethod
    def remaining_reserve(self) -> float:
        """Remaining unextracted reserve mass (tonnes)."""
        ...

    @property
    @abc.abstractmethod
    def is_exhausted(self) -> bool:
        """Whether the total reserve has been fully extracted."""
        ...

    @property
    @abc.abstractmethod
    def is_parcel_exhausted(self) -> bool:
        """Whether the currently loaded parcel has been fully extracted."""
        ...

    @abc.abstractmethod
    def advance_parcel_state(self) -> None:
        """Check for parcel depletion and update thresholds."""
        ...

    @abc.abstractmethod
    def load_next_parcel(self) -> Optional[Parcel]:
        """Draw the next parcel from the reserve."""
        ...

    @abc.abstractmethod
    def levels(self) -> Sequence[drs.Level]:
        """Return stateful DRS levels for reserve and parcel tracking."""
        ...

    @abc.abstractmethod
    def step(self, dt: float) -> None:
        """Advance internal levels forward by dt."""
        ...

    def time_to_event(self) -> float:
        """Time to next threshold crossing event."""
        min_dt = math.inf
        for lvl in self.levels():
            dt = lvl.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt


class StochasticReserve(GeologySource):
    """Stochastic geological reserve with parcel batching and facies variation."""

    def __init__(
        self,
        name: str,
        total_tonnes: float,
        generator: StochasticFaciesGenerator,
        min_parcel_mass: float = 30000.0,
        max_parcel_mass: float = 50000.0,
        initial_parcel_mass: Optional[float] = None,
        warming_period: float = 0.0,
        min_waste_fraction: float = 0.0,
        max_waste_fraction: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.name = name
        self._total_tonnes = total_tonnes
        self.generator = generator
        self.min_parcel_mass = min_parcel_mass
        self.max_parcel_mass = max_parcel_mass
        self.warming_period = warming_period
        self.min_waste_fraction = min_waste_fraction
        self.max_waste_fraction = max_waste_fraction

        self.rng = random.Random(seed) if seed is not None else random
        self.active_parcel: Optional[Parcel] = None

        self.cumulative_extracted_mass = drs.Level(
            f"{name}_cumulative_extracted_mass", initial_value=0.0
        )
        self.cumulative_extracted_mass.upper_threshold = (
            warming_period if warming_period > 0.0 else total_tonnes
        )

        self.parcel_extracted_mass = drs.Level(
            f"{name}_parcel_extracted_mass", initial_value=0.0
        )

        if initial_parcel_mass is not None:
            ore2_frac = generator.mean_fraction
            self.active_parcel = Parcel(
                mass=min(initial_parcel_mass, self.remaining_reserve),
                ore1_fraction=1.0 - ore2_frac,
                ore2_fraction=ore2_frac,
            )
            self.parcel_extracted_mass.upper_threshold = self.active_parcel.mass
        else:
            self.load_next_parcel()

    @property
    def total_tonnes(self) -> float:
        return self._total_tonnes

    @total_tonnes.setter
    def total_tonnes(self, value: float) -> None:
        self._total_tonnes = value
        self.cumulative_extracted_mass.upper_threshold = value
        if self.active_parcel is None and not self.is_exhausted:
            self.load_next_parcel()

    @property
    def remaining_reserve(self) -> float:
        return max(0.0, self._total_tonnes - float(self.cumulative_extracted_mass.value))

    @property
    def is_exhausted(self) -> bool:
        return float(self.cumulative_extracted_mass.value) >= self._total_tonnes - 1e-6

    @property
    def is_parcel_exhausted(self) -> bool:
        if self.active_parcel is None:
            return True
        return float(self.parcel_extracted_mass.value) >= self.active_parcel.mass - 1e-6

    @property
    def net_extracted_mass(self) -> float:
        return max(0.0, float(self.cumulative_extracted_mass.value) - self.warming_period)

    def load_next_parcel(self) -> Optional[Parcel]:
        if self.remaining_reserve <= 1e-6:
            self.active_parcel = None
            return None

        p_mass = self.rng.uniform(self.min_parcel_mass, self.max_parcel_mass)
        p_mass = min(p_mass, self.remaining_reserve)

        self.parcel_extracted_mass.value = 0.0
        self.parcel_extracted_mass.upper_threshold = p_mass

        facies = self.generator.generate_next()
        ore2_frac = float(facies["ore1_frac"])
        ore1_frac = 1.0 - ore2_frac

        waste_frac = 0.0
        if self.max_waste_fraction > self.min_waste_fraction:
            waste_frac = self.rng.uniform(self.min_waste_fraction, self.max_waste_fraction)

        self.active_parcel = Parcel(
            mass=p_mass,
            ore1_fraction=ore1_frac,
            ore2_fraction=ore2_frac,
            waste_fraction=waste_frac,
        )
        return self.active_parcel

    def advance_parcel_state(self) -> None:
        if self.is_parcel_exhausted:
            self.load_next_parcel()

        if self.cumulative_extracted_mass.value < self.warming_period:
            self.cumulative_extracted_mass.upper_threshold = self.warming_period
        else:
            self.cumulative_extracted_mass.upper_threshold = self._total_tonnes

        if self.active_parcel is not None:
            self.parcel_extracted_mass.upper_threshold = self.active_parcel.mass

    def levels(self) -> Sequence[drs.Level]:
        return (self.cumulative_extracted_mass, self.parcel_extracted_mass)

    def step(self, dt: float) -> None:
        self.cumulative_extracted_mass.step(dt)
        self.parcel_extracted_mass.step(dt)

"""Geological sources, discrete entities, and reserve depletion models."""

from __future__ import annotations

import abc
import math
import random
from typing import Mapping, Optional, Sequence

import drs
from .generators import StochasticFaciesGenerator
from .material import Entity, Flow


class GeologySource(abc.ABC, drs.Module):
    """Abstract base class for material sources and reserve depletion."""

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
    def active_entity(self) -> Optional[Entity]:
        """Currently loaded discrete material entity (parcel / block)."""
        ...

    @property
    @abc.abstractmethod
    def current_attributes(self) -> Mapping[str, float]:
        """Current attribute concentrations of the material being extracted."""
        ...

    @abc.abstractmethod
    def advance_state(self) -> None:
        """Check for current entity depletion and advance to the next entity."""
        ...

    @abc.abstractmethod
    def extract(self, rate: float) -> Flow:
        """Extract material at the given rate and return the continuous flow."""
        ...

    @abc.abstractmethod
    def levels(self) -> Sequence[drs.Level]:
        """Return stateful DRS levels for reserve and entity tracking."""
        ...

    def time_to_event(self) -> float:
        """Time to next threshold crossing event."""
        min_dt = math.inf
        for lvl in self.levels():
            dt = lvl.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt

    @abc.abstractmethod
    def step(self, dt: float) -> None:
        """Advance internal levels forward by dt."""
        ...


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
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.name = name
        self._total_tonnes = float(total_tonnes)
        self.generator = generator
        self.min_parcel_mass = float(min_parcel_mass)
        self.max_parcel_mass = float(max_parcel_mass)
        self.warming_period = float(warming_period)

        self.rng = random.Random(seed) if seed is not None else random
        self._active_entity: Optional[Entity] = None

        self.cumulative_extracted_mass = drs.Level(
            f"{name}_cumulative_extracted_mass", initial_value=0.0
        )
        self.cumulative_extracted_mass.upper_threshold = (
            warming_period if warming_period > 0.0 else total_tonnes
        )

        self.entity_extracted_mass = drs.Level(
            f"{name}_entity_extracted_mass", initial_value=0.0
        )

        if initial_parcel_mass is not None:
            attrs = {generator.attribute_name: generator.mean_fraction}
            self._active_entity = Entity(
                mass=min(initial_parcel_mass, self.remaining_reserve),
                attributes=attrs,
            )
            self.entity_extracted_mass.upper_threshold = self._active_entity.mass
        else:
            self.load_next_entity()

    @property
    def total_tonnes(self) -> float:
        return self._total_tonnes

    @total_tonnes.setter
    def total_tonnes(self, value: float) -> None:
        self._total_tonnes = float(value)
        self.cumulative_extracted_mass.upper_threshold = float(value)
        if self._active_entity is None and not self.is_exhausted:
            self.load_next_entity()

    @property
    def remaining_reserve(self) -> float:
        return max(0.0, self._total_tonnes - float(self.cumulative_extracted_mass.value))

    @property
    def is_exhausted(self) -> bool:
        return float(self.cumulative_extracted_mass.value) >= self._total_tonnes - 1e-6

    @property
    def is_entity_exhausted(self) -> bool:
        if self._active_entity is None:
            return True
        return float(self.entity_extracted_mass.value) >= self._active_entity.mass - 1e-6

    @property
    def active_entity(self) -> Optional[Entity]:
        return self._active_entity

    @property
    def current_attributes(self) -> Mapping[str, float]:
        if self._active_entity is not None:
            return self._active_entity.attributes
        return {}

    @property
    def net_extracted_mass(self) -> float:
        return max(0.0, float(self.cumulative_extracted_mass.value) - self.warming_period)

    def load_next_entity(self) -> Optional[Entity]:
        if self.remaining_reserve <= 1e-6:
            self._active_entity = None
            return None

        p_mass = self.rng.uniform(self.min_parcel_mass, self.max_parcel_mass)
        p_mass = min(p_mass, self.remaining_reserve)

        self.entity_extracted_mass.value = 0.0
        self.entity_extracted_mass.upper_threshold = p_mass

        attrs = self.generator.generate_next()
        self._active_entity = Entity(mass=p_mass, attributes=attrs)
        return self._active_entity

    def advance_state(self) -> None:
        if self.is_entity_exhausted:
            self.load_next_entity()

        if self.cumulative_extracted_mass.value < self.warming_period:
            self.cumulative_extracted_mass.upper_threshold = self.warming_period
        else:
            self.cumulative_extracted_mass.upper_threshold = self._total_tonnes

        if self._active_entity is not None:
            self.entity_extracted_mass.upper_threshold = self._active_entity.mass

    def extract(self, rate: float) -> Flow:
        """Extract material from the reserve at the requested rate.

        Sets level rates and returns the resulting Flow with current quality attributes.
        """
        self.advance_state()
        if self.is_exhausted:
            actual_rate = 0.0
        else:
            actual_rate = max(0.0, rate)

        self.cumulative_extracted_mass.rate = actual_rate
        self.entity_extracted_mass.rate = actual_rate
        return Flow(rate=actual_rate, attributes=self.current_attributes)

    def levels(self) -> Sequence[drs.Level]:
        return (self.cumulative_extracted_mass, self.entity_extracted_mass)

    def is_terminating_condition_met(self) -> bool:
        return self.is_exhausted

    def step(self, dt: float) -> None:
        self.cumulative_extracted_mass.step(dt)
        self.entity_extracted_mass.step(dt)

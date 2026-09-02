"""Material sources, continuous flow extraction, and discrete entity generation."""

from __future__ import annotations

import math
import random
from typing import Iterable, Iterator, Mapping, Optional

import drs
from drs import Flow, Entity


def autocorrelated_generator(
    mean_fraction: float = 0.30,
    std_dev: float = 0.05,
    prob_new_facies: float = 0.30,
    variation_step: float = 0.01,
    min_mass: float = 30000.0,
    max_mass: float = 50000.0,
    initial_mass: Optional[float] = None,
    attribute_name: str = "ore2_fraction",
    seed: Optional[int] = None,
) -> Iterator[Entity]:
    """Generates an infinite sequence of Entity batches with an autocorrelated random walk."""
    rng = random.Random(seed) if seed is not None else random
    is_new = True
    curr_frac = float(mean_fraction)

    if initial_mass is not None:
        yield Entity(mass=float(initial_mass), attributes={attribute_name: curr_frac})

    while True:
        mass = rng.uniform(min_mass, max_mass)
        if is_new:
            curr_frac = rng.gauss(mean_fraction, std_dev) if std_dev != 0.0 else mean_fraction
        else:
            curr_frac += variation_step * rng.uniform(-1.0, 1.0)
        curr_frac = max(0.0, min(1.0, curr_frac))
        is_new = rng.random() <= prob_new_facies

        yield Entity(mass=mass, attributes={attribute_name: curr_frac})


class MaterialSource(drs.Module):
    """Source of material with reserve depletion and continuous/discrete extraction.

    Consumes an arbitrary stream (Iterable[Entity]) representing blocks, parcels,
    or geostatistical realizations, and exposes continuous Flow extraction.
    """

    def __init__(
        self,
        name: str = "source",
        total_tonnes: float = math.inf,
        stream: Optional[Iterable[Entity]] = None,
        warming_period: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self._total_tonnes = float(total_tonnes)
        self.warming_period = float(warming_period)
        self._stream = iter(stream) if stream is not None else autocorrelated_generator()

        self.cumulative_extracted_mass = drs.Level(
            f"{name}_cumulative_extracted_mass", initial_value=0.0, owner=self
        )
        self.cumulative_extracted_mass.upper_threshold = (
            warming_period if warming_period > 0.0 else total_tonnes
        )

        self.entity_extracted_mass = drs.Level(
            f"{name}_entity_extracted_mass", initial_value=0.0, owner=self
        )

        self._active_entity: Optional[Entity] = None
        self.next_entity()

    @property
    def total_tonnes(self) -> float:
        return self._total_tonnes

    @total_tonnes.setter
    def total_tonnes(self, value: float) -> None:
        self._total_tonnes = float(value)
        self.cumulative_extracted_mass.upper_threshold = float(value)
        if self._active_entity is None and not self.is_exhausted:
            self.next_entity()

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

    def next_entity(self) -> Optional[Entity]:
        """Draw the next discrete entity batch from the stream."""
        if self.remaining_reserve <= 1e-6:
            self._active_entity = None
            return None

        try:
            entity = next(self._stream)
        except StopIteration:
            self._active_entity = None
            return None

        actual_mass = min(entity.mass, self.remaining_reserve)
        self.entity_extracted_mass.value = 0.0
        self.entity_extracted_mass.upper_threshold = actual_mass

        self._active_entity = Entity(mass=actual_mass, attributes=dict(entity.attributes))
        return self._active_entity

    def advance_state(self) -> None:
        if self.is_entity_exhausted:
            self.next_entity()

        if self.cumulative_extracted_mass.value < self.warming_period:
            self.cumulative_extracted_mass.upper_threshold = self.warming_period
        else:
            self.cumulative_extracted_mass.upper_threshold = self._total_tonnes

        if self._active_entity is not None:
            self.entity_extracted_mass.upper_threshold = self._active_entity.mass

    def extract(self, rate: float) -> Flow:
        """Extract continuous flow from the source at the requested rate."""
        self.advance_state()
        actual_rate = 0.0 if self.is_exhausted else max(0.0, float(rate))

        self.cumulative_extracted_mass.rate = actual_rate
        self.entity_extracted_mass.rate = actual_rate
        return Flow(rate=actual_rate, attributes=dict(self.current_attributes))

    def is_terminating_condition_met(self) -> bool:
        return self.is_exhausted

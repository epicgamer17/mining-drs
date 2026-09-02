"""Material sources, continuous flow extraction, and discrete entity generation."""

from __future__ import annotations

import math
import random
from typing import Callable, Mapping, Optional

import drs
from drs import Flow, Entity


class MaterialSource(drs.Module):
    """Source of material with reserve depletion and attribute generation.

    Generates quality attributes either from statistical parameters (autocorrelated
    random walk), a block model iterator, or a custom attribute generator function.
    Seamlessly extracts continuous Flow (DRS) or discrete Entity batches (DES).
    """

    def __init__(
        self,
        name: str = "source",
        total_tonnes: float = math.inf,
        mean_attributes: Optional[Mapping[str, float]] = None,
        attribute_std_dev: float = 0.05,
        variation_autocorrelation: float = 0.30,
        variation_step: float = 0.01,
        min_parcel_mass: float = 30000.0,
        max_parcel_mass: float = 50000.0,
        initial_parcel_mass: Optional[float] = None,
        warming_period: float = 0.0,
        attribute_generator: Optional[Callable[[], Mapping[str, float]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.name = name
        self._total_tonnes = float(total_tonnes)
        self.mean_attributes = dict(mean_attributes or {"ore2_fraction": 0.30})
        self.attribute_std_dev = float(attribute_std_dev)
        self.variation_autocorrelation = float(variation_autocorrelation)
        self.variation_step = float(variation_step)
        self.min_parcel_mass = float(min_parcel_mass)
        self.max_parcel_mass = float(max_parcel_mass)
        self.warming_period = float(warming_period)
        self.attribute_generator = attribute_generator

        self.rng = random.Random(seed) if seed is not None else random
        self._active_entity: Optional[Entity] = None

        # Autocorrelation Markov state
        self._next_is_new: bool = True
        self._current_fraction: float = float(self.mean_attributes.get("ore2_fraction", 0.30))

        self.cumulative_extracted_mass = drs.Level(
            f"{name}_cumulative_extracted_mass", initial_value=0.0, owner=self
        )
        self.cumulative_extracted_mass.upper_threshold = (
            warming_period if warming_period > 0.0 else total_tonnes
        )

        self.entity_extracted_mass = drs.Level(
            f"{name}_entity_extracted_mass", initial_value=0.0, owner=self
        )

        if initial_parcel_mass is not None:
            attrs = dict(self.mean_attributes)
            self._active_entity = Entity(
                mass=min(initial_parcel_mass, self.remaining_reserve),
                attributes=attrs,
            )
            self.entity_extracted_mass.upper_threshold = self._active_entity.mass
        else:
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
        return self.mean_attributes

    def next_entity(self) -> Optional[Entity]:
        """Draw the next discrete entity batch from the source."""
        if self.remaining_reserve <= 1e-6:
            self._active_entity = None
            return None

        p_mass = self.rng.uniform(self.min_parcel_mass, self.max_parcel_mass)
        p_mass = min(p_mass, self.remaining_reserve)

        self.entity_extracted_mass.value = 0.0
        self.entity_extracted_mass.upper_threshold = p_mass

        if self.attribute_generator is not None:
            attrs = self.attribute_generator()
        else:
            if self._next_is_new:
                if self.attribute_std_dev != 0.0:
                    fraction = self.rng.gauss(
                        self.mean_attributes.get("ore2_fraction", 0.30),
                        self.attribute_std_dev,
                    )
                else:
                    fraction = self.mean_attributes.get("ore2_fraction", 0.30)
            else:
                fraction = (
                    self._current_fraction
                    + self.variation_step * self.rng.uniform(-1.0, 1.0)
                )

            self._current_fraction = max(0.0, min(1.0, fraction))
            self._next_is_new = self.rng.random() <= self.variation_autocorrelation
            attrs = {"ore2_fraction": self._current_fraction}

        self._active_entity = Entity(mass=p_mass, attributes=attrs)
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

"""Multi-attribute stockpile component specializing drs.Storage."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import drs
from drs import Storage


class Stockpile(Storage):
    """Multi-attribute stockpile component specializing drs.Storage.

    Tracks total ore mass while maintaining balances for discrete quality
    attributes (such as contained metal mass, grades, or deleterious elements).
    """

    def __init__(
        self,
        name: str,
        expected_attributes: Sequence[str] = (),
        initial_mass: float = 0.0,
        initial_attributes: Optional[Mapping[str, float]] = None,
        capacity: float = math.inf,
        attr_inflow: float = 1.0,
    ):
        super().__init__(
            name=f"{name}_mass", capacity=capacity, initial_level=initial_mass
        )
        self.name = name
        self.expected_attributes = list(expected_attributes)
        self.attr_inflow = float(attr_inflow)

        self.actual_outflow_rate = drs.Variable(f"{name}_actual_outflow_rate", 0.0)

        attrs = dict(initial_attributes or {})
        self.attributes: dict[str, drs.Level] = {}
        for attr in self.expected_attributes:
            attr_lvl = drs.Level(f"{name}_{attr}", initial_value=attrs.get(attr, 0.0))
            attr_lvl.lower_threshold = 0.0
            self.attributes[attr] = attr_lvl

    def __getattr__(self, name: str) -> Any:
        if "attributes" in self.__dict__ and name in self.attributes:
            return self.attributes[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def current_concentration(self, attr: str) -> float:
        """Calculates current concentration (e.g. grade) of an attribute."""
        level = self.attributes[attr]
        return level.value / max(1e-6, self.level)

    def feed_and_draw(self, inflow_rate: float, outflow_rate: float) -> float:
        """Feed stockpile from routing inflow and draw into plant."""
        return self.set_inout(inflow_rate, outflow_rate, attr_inflow=self.attr_inflow)

    def set_inout(
        self,
        inflow_rate: float,
        outflow_rate: float,
        attr_inflow: float = 1.0,
    ) -> float:
        """Set net inflow/outflow rates for one engine step."""
        actual_outflow = outflow_rate
        if self.is_empty or self.level <= 1e-6:
            actual_outflow = min(actual_outflow, inflow_rate)

        self.rate = inflow_rate - actual_outflow

        for attr, level in self.attributes.items():
            level.rate = (
                inflow_rate * attr_inflow
                - actual_outflow * self.current_concentration(attr)
            )

        self.actual_outflow_rate.value = actual_outflow
        return actual_outflow

    def levels(self) -> Sequence[drs.Level]:
        """Return the stateful levels owned by this stockpile."""
        return (self._level, *self.attributes.values())

    def time_to_event(self) -> float:
        """Time until this stockpile or any attribute hits a state boundary."""
        min_dt = math.inf
        for lvl in self.levels():
            dt = lvl.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt

    def step(self, dt: float) -> None:
        """Advance all owned levels forward by dt."""
        for lvl in self.levels():
            lvl.step(dt)

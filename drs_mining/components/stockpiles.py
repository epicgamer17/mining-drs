"""Multi-attribute stockpile component specializing drs.Storage."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import drs
from drs import Storage
from .material import Flow, blend_flows


class Stockpile(Storage):
    """Multi-attribute stockpile component specializing drs.Storage.

    Tracks total ore mass while maintaining exact CSTR conservation of mass
    and dynamic grade/attribute balances for arbitrary quality attributes.
    """

    def __init__(
        self,
        name: str,
        capacity: float = math.inf,
        initial_mass: float = 0.0,
        initial_attributes: Optional[Mapping[str, float]] = None,
    ):
        super().__init__(name=f"{name}_mass", capacity=capacity, initial_level=initial_mass)
        self.name = name
        self.attribute_masses: dict[str, drs.Level] = {}

        if initial_attributes:
            for attr, concentration in initial_attributes.items():
                attr_lvl = drs.Level(
                    f"{name}_{attr}_mass",
                    initial_value=float(initial_mass) * float(concentration),
                )
                attr_lvl.lower_threshold = 0.0
                self.attribute_masses[attr] = attr_lvl

    def grade(self, attr: str) -> float:
        """Calculate current concentration / grade of an attribute in the stockpile."""
        lvl = self.attribute_masses.get(attr)
        if lvl is None:
            return 0.0
        m = self.level
        if m <= 1e-6:
            return 0.0
        return max(0.0, float(lvl.value) / m)

    @property
    def current_attributes(self) -> dict[str, float]:
        """Return a dictionary of current concentrations for all tracked attributes."""
        return {attr: self.grade(attr) for attr in self.attribute_masses}

    def _ensure_attribute(self, attr: str) -> drs.Level:
        if attr not in self.attribute_masses:
            lvl = drs.Level(f"{self.name}_{attr}_mass", initial_value=0.0)
            lvl.lower_threshold = 0.0
            self.attribute_masses[attr] = lvl
        return self.attribute_masses[attr]

    def feed_and_draw(
        self,
        inflow: Flow | Sequence[Flow],
        draw_rate: float,
    ) -> Flow:
        """Feed stockpile from incoming flow(s) and draw material to downstream processors.

        Applies mass balance, empty buffer starvation limits, and dynamic grade dilution.
        Returns the outgoing Flow with instantaneous stockpile attributes.
        """
        if isinstance(inflow, Sequence):
            net_inflow = blend_flows(inflow)
        else:
            net_inflow = inflow

        actual_draw = max(0.0, float(draw_rate))
        if self.is_empty or self.level <= 1e-6:
            actual_draw = min(actual_draw, net_inflow.rate)

        # Net mass rate of change
        self.rate = net_inflow.rate - actual_draw

        # Track outgoing attribute concentrations before updating rates
        out_attrs = self.current_attributes

        # Update rates for all incoming and existing attributes
        all_attrs = set(self.attribute_masses.keys()) | set(net_inflow.attributes.keys())
        for attr in all_attrs:
            attr_lvl = self._ensure_attribute(attr)
            g_in = net_inflow.attributes.get(attr, 0.0)
            g_out = out_attrs.get(attr, 0.0)
            attr_lvl.rate = (net_inflow.rate * g_in) - (actual_draw * g_out)

        return Flow(rate=actual_draw, attributes=out_attrs)

    def levels(self) -> Sequence[drs.Level]:
        """Return all stateful levels owned by this stockpile (mass + attributes)."""
        return (self._level, *self.attribute_masses.values())

    def time_to_event(self) -> float:
        """Time until this stockpile mass or any attribute hits a threshold."""
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

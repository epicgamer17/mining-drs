import math
from typing import List, Dict, Optional

import drs
from drs import Storage
from drs.flow import Flow


class Stockpile(Storage):
    """Stockpile component inheriting from drs.Storage."""

    def __init__(
        self,
        name: str,
        expected_attributes: List[str],
        initial_mass: float = 0.0,
        initial_attributes: Optional[Dict[str, float]] = None,
        capacity: float = math.inf,
    ):
        super().__init__(name=f"{name}_mass", capacity=capacity, initial_level=initial_mass)
        self.name = name
        self.expected_attributes = expected_attributes

        # Bind current_mass to Storage's underlying Level for compatibility
        self.current_mass = self._level
        self.actual_outflow_rate = drs.Variable(f"{name}_actual_outflow_rate", 0.0)

        initial_attributes = initial_attributes or {}
        for attr in expected_attributes:
            setattr(
                self,
                attr,
                drs.Level(
                    f"{name}_{attr}", initial_value=initial_attributes.get(attr, 0.0)
                ),
            )

    def current_concentration(self, attr: str) -> float:
        level = getattr(self, attr, None)
        if level is None:
            return 0.0
        return level.value / max(1e-6, self.level)

    def forward(self, requested_outflow_rate, inflow=None) -> "Flow":
        if inflow is not None:
            material = inflow.value
            inflow_rate = material.extraction_rate
            inflow_attr = material.attr_value
        else:
            inflow_rate = 0.0
            inflow_attr = 0.0

        self.current_mass.rate = inflow_rate
        for attr in self.expected_attributes:
            getattr(self, attr).rate = inflow_rate * inflow_attr

        current_inflow = self.current_mass.rate

        actual_outflow = requested_outflow_rate.value
        if self.level <= 1e-6:
            actual_outflow = min(actual_outflow, current_inflow)

        for attr in self.expected_attributes:
            level = getattr(self, attr)
            level.rate = level.rate - actual_outflow * self.current_concentration(attr)

        self.current_mass.rate = self.current_mass.rate - actual_outflow

        if self.current_mass.rate < 0:
            self.current_mass.lower_threshold = 0.0
            for attr in self.expected_attributes:
                getattr(self, attr).lower_threshold = 0.0

        self.actual_outflow_rate.value = actual_outflow
        return Flow(value=actual_outflow)


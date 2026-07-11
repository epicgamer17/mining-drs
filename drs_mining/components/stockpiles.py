from typing import List, Dict, Optional

import drs
from drs.flow import Flow


class Stockpile(drs.Module):
    def __init__(
        self,
        name: str,
        expected_attributes: List[str],
        initial_mass: float = 0.0,
        initial_attributes: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.name = name
        self.expected_attributes = expected_attributes

        self.current_mass = drs.Level(f"{name}_mass", initial_value=initial_mass)
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
        return level.value / max(1e-6, self.current_mass.value)

    def forward(self, requested_outflow_rate, inflow=None) -> "Flow":
        if inflow is not None:
            material = inflow.value
            self.current_mass.rate = material.extraction_rate
            for attr in self.expected_attributes:
                getattr(self, attr).rate = (
                    material.extraction_rate * material.attr_value
                )

        current_inflow = self.current_mass.rate

        actual_outflow = requested_outflow_rate.value
        if self.current_mass.value <= 1e-6:
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

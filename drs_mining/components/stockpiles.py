import math
from typing import List, Dict, Optional

import drs
from drs import Storage


class Stockpile(Storage):
    """Stockpile component inheriting from drs.Storage."""

    def __init__(
        self,
        name: str,
        expected_attributes: List[str],
        initial_mass: float = 0.0,
        initial_attributes: Optional[Dict[str, float]] = None,
        capacity: float = math.inf,
        attr_inflow: float = 1.0,
    ):
        super().__init__(name=f"{name}_mass", capacity=capacity, initial_level=initial_mass)
        self.name = name
        self.expected_attributes = expected_attributes
        self.attr_inflow = float(attr_inflow)

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

    def feed_and_draw(self, inflow_rate: float, outflow_rate: float) -> float:
        """Feed the stockpile from a routing inflow and draw into the plant.

        High-level alias for :meth:`set_inout` using this stockpile's stored
        ``attr_inflow`` (Ore1 = 1.0, Ore2 = 0.0). Returns the realised outflow.
        """
        return self.set_inout(
            inflow_rate, outflow_rate, attr_inflow=self.attr_inflow
        )

    def set_inout(
        self,
        inflow_rate: float,
        outflow_rate: float,
        attr_inflow: float = 1.0,
    ) -> float:
        """Set the net inflow/outflow for this stockpile for one engine step.

        Converts flow into rate signals on the stockpile mass level and each
        expected attribute. ``attr_inflow`` is the fraction of the incoming
        material's attribute value (e.g. grade) applied to the stockpile;
        outflow always removes the current stockpile concentration. When the
        stockpile would run dry the outflow is capped at the inflow so mass
        never dips below zero. Returns the realised outflow rate.
        """
        actual_outflow = outflow_rate
        if self.level <= 1e-6:
            actual_outflow = min(actual_outflow, inflow_rate)

        net = inflow_rate - actual_outflow
        self.current_mass.rate = net
        for attr in self.expected_attributes:
            level = getattr(self, attr)
            level.rate = (
                inflow_rate * attr_inflow
                - actual_outflow * self.current_concentration(attr)
            )

        if net < 0:
            self.current_mass.lower_threshold = 0.0
            for attr in self.expected_attributes:
                getattr(self, attr).lower_threshold = 0.0

        self.actual_outflow_rate.value = actual_outflow
        return actual_outflow



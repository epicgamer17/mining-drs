import math
from typing import Mapping, Sequence, List, Optional

import drs
from drs import Storage


class Stockpile(Storage):
    """Multi-attribute stockpile component specializing drs.Storage.

    Extends basic single-level Storage with:
    - Multi-attribute tracking (ore grades, moisture, deleterious elements) via dynamic Levels.
    - Coupled mass-concentration balances and dilution dynamics.
    - Physical starvation bounds and actual outflow rate telemetry.
    """

    def __init__(
        self,
        name: str,
        expected_attributes: Sequence[str],
        initial_mass: float,
        initial_attributes: Mapping[str, float],
        capacity: float = math.inf,
        attr_inflow: float = 1.0,
    ):
        super().__init__(
            name=f"{name}_mass", capacity=capacity, initial_level=initial_mass
        )
        self.name = name
        self.expected_attributes = list(expected_attributes)
        self.attr_inflow = float(attr_inflow)

        # Bind current_mass and mass to Storage's underlying Level for compatibility
        self.current_mass = self._level
        self.actual_outflow_rate = drs.Variable(f"{name}_actual_outflow_rate", 0.0)

        attrs = dict(initial_attributes)
        for attr in self.expected_attributes:
            attr_lvl = drs.Level(f"{name}_{attr}", initial_value=attrs.get(attr, 0.0))
            attr_lvl.lower_threshold = 0.0
            setattr(self, attr, attr_lvl)

    def current_concentration(self, attr: str) -> float:
        """Calculates current concentration (e.g. grade) of an attribute.

        Returns 0.0 if the attribute level is not present or if stockpile is empty.
        """
        level = getattr(self, attr, None)
        if level is None:
            return 0.0
        return level.value / max(1e-6, self.level)

    def feed_and_draw(self, inflow_rate: float, outflow_rate: float) -> float:
        """Feed the stockpile from a routing inflow and draw into the plant.

        High-level interface for :meth:`set_inout` using this stockpile's stored
        ``attr_inflow``. Returns the realised outflow rate.
        """
        return self.set_inout(inflow_rate, outflow_rate, attr_inflow=self.attr_inflow)

    def set_inout(
        self,
        inflow_rate: float,
        outflow_rate: float,
        attr_inflow: float = 1.0,
    ) -> float:
        """Set the net inflow/outflow for this stockpile for one engine step.

        Converts flow into rate signals on the stockpile mass level and each
        expected attribute. Outflow is capped when the stockpile is empty to prevent
        negative mass levels. Returns the realised outflow rate.
        """
        actual_outflow = outflow_rate
        if self.is_empty or self.level <= 1e-6:
            actual_outflow = min(actual_outflow, inflow_rate)

        net = inflow_rate - actual_outflow
        self.rate = net

        for attr in self.expected_attributes:
            level = getattr(self, attr, None)
            if level is not None:
                level.rate = (
                    inflow_rate * attr_inflow
                    - actual_outflow * self.current_concentration(attr)
                )

        self.actual_outflow_rate.value = actual_outflow
        return actual_outflow

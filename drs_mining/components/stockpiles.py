import math
from typing import Mapping, Sequence, List

import drs
from drs import Storage


class Stockpile(Storage):
    """Stockpile component inheriting from drs.Storage."""

    # TODO: what does this do that python-drs Storage does not do? And why can't we use drs.Storage directly? I'm not saying we should use drs.Storage directly, but lets make sure this class only adds to drs.Storage what it specifically does additionally, and have drs.Storage handle the things it should handle (and not reimplmentent or alias them here)
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

        # Bind current_mass to Storage's underlying Level for compatibility
        self.current_mass = self._level
        self.actual_outflow_rate = drs.Variable(f"{name}_actual_outflow_rate", 0.0)

        attrs = dict(initial_attributes)
        for attr in self.expected_attributes:
            setattr(
                self,
                attr,
                drs.Level(f"{name}_{attr}", initial_value=attrs.get(attr, 0.0)),
            )

    # TODO: why can level ever be None? what is current concentration used for?
    def current_concentration(self, attr: str) -> float:
        level = getattr(self, attr, None)
        if level is None:
            return 0.0
        return level.value / max(1e-6, self.level)

    # TODO: should this be on Storage of python-drs?
    # TODO: should we remove this alias to improve readability? or remove setinout?
    def feed_and_draw(self, inflow_rate: float, outflow_rate: float) -> float:
        """Feed the stockpile from a routing inflow and draw into the plant.

        High-level alias for :meth:`set_inout` using this stockpile's stored
        ``attr_inflow`` (Ore1 = 1.0, Ore2 = 0.0). Returns the realised outflow.
        """
        return self.set_inout(inflow_rate, outflow_rate, attr_inflow=self.attr_inflow)

    # TODO: should this be on Storage from python-drs?
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
            level = getattr(self, attr, None)
            if level is not None:
                level.rate = (
                    inflow_rate * attr_inflow
                    - actual_outflow * self.current_concentration(attr)
                )

        if net < 0:
            self.current_mass.lower_threshold = 0.0
            for attr in self.expected_attributes:
                attr_level = getattr(self, attr, None)
                if attr_level is not None:
                    attr_level.lower_threshold = 0.0

        self.actual_outflow_rate.value = actual_outflow
        return actual_outflow

    # TODO: needed?
    def is_terminating_condition_met(self) -> bool:
        return False

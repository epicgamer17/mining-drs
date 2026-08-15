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



from typing import Sequence
import math
import drs
from drs import Processor
from .stockpiles import Stockpile


class MetallurgicalPlant(Processor):
    """Represents a metallurgical plant / concentrator processing mined ore."""

    def __init__(
        self,
        stockpiles: Sequence[Stockpile],
        max_rate: float = math.inf,
        name: str = "metallurgical_plant",
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.stockpiles = list(stockpiles)
        if len(self.stockpiles) < 2:
            raise ValueError(
                f"MetallurgicalPlant requires at least 2 stockpiles (Ore1 and Ore2), got {len(self.stockpiles)}"
            )
        self.ore1_stock = self.stockpiles[0]
        self.ore2_stock = self.stockpiles[1]
        self.cumulative_milled_mass = drs.Level(
            "cumulative_milled_mass", initial_value=0.0
        )

    def process(self, mass_rate: float) -> None:
        """Draw ``mass_rate`` into the plant for one engine step.

        Sets the target rate and stamps the milled-mass rate from the
        realised rate; the milled-mass level is advanced in ``step``.
        """
        self.target_rate = mass_rate
        self.cumulative_milled_mass.rate = self.actual_rate

from typing import Optional, Sequence
import math
import drs
from drs import Processor
from .fleet import ContinuousFleetLogistics


class BaseMetallurgicalPlant(Processor):
    def __init__(
        self,
        mine=None,
        fleet: Optional[ContinuousFleetLogistics] = None,
        ore1_stock=None,
        ore2_stock=None,
        max_rate: float = math.inf,
        name: str = "metallurgical_plant",
        stockpiles: Optional[Sequence] = None,
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.mine = mine
        self.fleet = fleet

        if stockpiles is not None:
            self.stockpiles = list(stockpiles)
            self.ore1_stock = self.stockpiles[0] if len(self.stockpiles) > 0 else ore1_stock
            self.ore2_stock = self.stockpiles[1] if len(self.stockpiles) > 1 else ore2_stock
        else:
            self.ore1_stock = ore1_stock
            self.ore2_stock = ore2_stock
            self.stockpiles = [s for s in [ore1_stock, ore2_stock] if s is not None]

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


class ConcentratorPlant(BaseMetallurgicalPlant):
    def __init__(
        self,
        mine=None,
        fleet: Optional[ContinuousFleetLogistics] = None,
        ore1_stock=None,
        ore2_stock=None,
        max_rate: float = math.inf,
        name: str = "concentrator_plant",
        stockpiles: Optional[Sequence] = None,
    ):
        super().__init__(
            mine=mine,
            fleet=fleet,
            ore1_stock=ore1_stock,
            ore2_stock=ore2_stock,
            max_rate=max_rate,
            name=name,
            stockpiles=stockpiles,
        )




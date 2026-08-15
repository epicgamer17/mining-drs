import math
import drs
from drs import Processor
from .fleet import ContinuousFleetLogistics


class BaseMetallurgicalPlant(Processor):
    def __init__(
        self,
        mine,
        fleet: ContinuousFleetLogistics,
        ore1_stock,
        ore2_stock,
        max_rate: float = math.inf,
        name: str = "metallurgical_plant",
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.mine = mine
        self.fleet = fleet

        self.cumulative_milled_mass = drs.Level(
            "cumulative_milled_mass", initial_value=0.0
        )

class ConcentratorPlant(BaseMetallurgicalPlant):
    def __init__(
        self,
        mine,
        fleet: ContinuousFleetLogistics,
        ore1_stock,
        ore2_stock,
        max_rate: float = math.inf,
        name: str = "concentrator_plant",
    ):
        super().__init__(mine, fleet, ore1_stock, ore2_stock, max_rate=max_rate, name=name)



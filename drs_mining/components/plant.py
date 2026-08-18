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
        mine,
        fleet: ContinuousFleetLogistics,
        ore1_stock,
        ore2_stock,
        max_rate: float = math.inf,
        name: str = "concentrator_plant",
    ):
        super().__init__(mine, fleet, ore1_stock, ore2_stock, max_rate=max_rate, name=name)



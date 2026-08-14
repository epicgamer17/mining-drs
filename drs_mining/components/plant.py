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

    def update_milling_rate(self, ore1_outflow: float, ore2_outflow: float) -> float:
        """Sets target milling rate and updates cumulative milled mass accumulation rate."""
        o1 = ore1_outflow.value if hasattr(ore1_outflow, "value") else float(ore1_outflow)
        o2 = ore2_outflow.value if hasattr(ore2_outflow, "value") else float(ore2_outflow)

        total_inflow = o1 + o2
        self.target_rate = total_inflow
        self.cumulative_milled_mass.rate = self.actual_rate
        return self.actual_rate


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



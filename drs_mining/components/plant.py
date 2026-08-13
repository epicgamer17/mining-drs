import drs
from drs.flow import Flow
from .fleet import ContinuousFleetLogistics


class BaseMetallurgicalPlant(drs.Module):
    def __init__(
        self, mine, fleet: ContinuousFleetLogistics, ore1_stock, ore2_stock
    ):
        super().__init__()
        self.mine = mine
        self.fleet = fleet

        self.cumulative_milled_mass = drs.Level(
            "cumulative_milled_mass", initial_value=0.0
        )

    def forward(self, ore1_outflow, ore2_outflow):
        o1 = ore1_outflow.value if isinstance(ore1_outflow, Flow) else ore1_outflow
        o2 = ore2_outflow.value if isinstance(ore2_outflow, Flow) else ore2_outflow

        total_inflow = o1 + o2
        self.cumulative_milled_mass.rate = total_inflow


class ConcentratorPlant(BaseMetallurgicalPlant):
    def __init__(
        self,
        mine,
        fleet: ContinuousFleetLogistics,
        ore1_stock,
        ore2_stock,
    ):
        super().__init__(mine, fleet, ore1_stock, ore2_stock)

    def forward(self, ore1_outflow, ore2_outflow):
        super().forward(ore1_outflow, ore2_outflow)

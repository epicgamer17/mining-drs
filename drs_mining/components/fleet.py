from typing import Tuple
import math
import drs
from drs.flow import Flow
from .data import MineOutput


class ContinuousFleetLogistics(drs.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.stockpile2_routing_fraction = drs.Variable(
            "stockpile2_routing_fraction", 0.0
        )

    def forward(self, *mine_flows):
        total_ore1_rate = 0.0
        total_ore2_rate = 0.0
        total_rate = 0.0
        for flow in mine_flows:
            if flow is not None:
                out = flow.value
                ore1_frac = out.attr_value
                total_ore1_rate += out.extraction_rate * ore1_frac
                total_ore2_rate += out.extraction_rate * (1.0 - ore1_frac)
                total_rate += out.extraction_rate

        if total_rate > 1e-6:
            self.stockpile2_routing_fraction.value = total_ore2_rate / total_rate
        else:
            self.stockpile2_routing_fraction.value = 0.0

        # Output pure Ore 1 rate and pure Ore 2 rate
        return Flow(value=MineOutput(extraction_rate=total_ore1_rate, attr_value=1.0)), Flow(
            value=MineOutput(extraction_rate=total_ore2_rate, attr_value=0.0)
        )

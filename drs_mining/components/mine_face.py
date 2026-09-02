"""Mine face component composed of geology and haulage mechanics."""

from __future__ import annotations

import math
from typing import Sequence

import drs
from drs import Processor

from .geology import GeologySource
from .haulage import HaulRoute


class MineFace(Processor):
    """Mine face processor that extracts ore from a geology source and connects to haulage."""

    def __init__(
        self,
        name: str,
        geology: GeologySource,
        haulage: HaulRoute,
        max_rate: float = math.inf,
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.geology = geology
        self.haulage = haulage

    @property
    def is_exhausted(self) -> bool:
        return self.geology.is_exhausted

    @property
    def is_ore_available(self) -> bool:
        return not self.geology.is_parcel_exhausted and not self.is_exhausted

    def advance_parcel_state(self) -> None:
        self.geology.advance_parcel_state()

    def is_terminating_condition_met(self) -> bool:
        return self.is_exhausted

    def levels(self) -> Sequence[drs.Level]:
        return self.geology.levels()

    def time_to_event(self) -> float:
        return self.geology.time_to_event()

    def step(self, dt: float) -> None:
        self.advance_parcel_state()
        actual = self.actual_rate
        self.geology.cumulative_extracted_mass.rate = actual
        self.geology.parcel_extracted_mass.rate = actual
        self.geology.step(dt)

"""Metallurgical plant (concentrator) processing mined ore from stockpiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import drs
from drs import Processor
from .stockpiles import Stockpile
from .modes import OperatingMode
from drs_mining.config import MILL_MODES


@dataclass(frozen=True)
class MillingSetpoints:
    """Milling rate setpoints (tonnes/day) for operating modes."""

    mode_a_ore1: float = 3600.0
    mode_a_ore2: float = 2400.0
    mode_a_contingency_ore1: float = 3900.0
    mode_b_ore1: float = 4600.0
    mode_b_ore2: float = 800.0
    mode_b_contingency_ore2: float = 2500.0


class MetallurgicalPlant(Processor):
    """Represents a metallurgical plant / concentrator processing mined ore.

    Encapsulates plant milling rate setpoints, stockpile starvation detection,
    contingency milling, and surging extraction requirements.
    """

    _CONTINGENCY_MODES = {"MODE_A_CONTINGENCY", "MODE_B_CONTINGENCY"}

    def __init__(
        self,
        stockpiles: Sequence[Stockpile],
        setpoints: Optional[MillingSetpoints] = None,
        name: str = "metallurgical_plant",
        max_rate: float = math.inf,
        target_ore_stock_level: float = 60000.0,
        duration_of_contingency_segments: float = 1.0,
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.stockpiles = list(stockpiles)
        if len(self.stockpiles) < 2:
            raise ValueError(
                f"MetallurgicalPlant requires at least 2 stockpiles (Ore1 and Ore2), got {len(self.stockpiles)}"
            )
        self.ore1_stock = self.stockpiles[0]
        self.ore2_stock = self.stockpiles[1]

        self.setpoints = setpoints or MillingSetpoints()
        self.target_ore_stock_level = target_ore_stock_level
        self.duration_of_contingency_segments = duration_of_contingency_segments

        # Levels and State
        self.cumulative_milled_mass = drs.Level(
            "cumulative_milled_mass", initial_value=0.0
        )
        self.active_operating_mode = drs.Variable(
            "active_operating_mode", MILL_MODES["MODE_A"]
        )
        self.current_contingency_duration = drs.Timer(
            "current_contingency_duration", initial_value=0.0
        )

        # Mode Timers
        self.cumulative_time_mode_a = drs.Timer(
            "cumulative_time_mode_a", initial_value=0.0
        )
        self.cumulative_time_mode_a_contingency = drs.Timer(
            "cumulative_time_mode_a_contingency", initial_value=0.0
        )
        self.cumulative_time_mode_a_surging = drs.Timer(
            "cumulative_time_mode_a_surging", initial_value=0.0
        )
        self.cumulative_time_mode_b = drs.Timer(
            "cumulative_time_mode_b", initial_value=0.0
        )
        self.cumulative_time_mode_b_contingency = drs.Timer(
            "cumulative_time_mode_b_contingency", initial_value=0.0
        )
        self.cumulative_time_mode_b_surging = drs.Timer(
            "cumulative_time_mode_b_surging", initial_value=0.0
        )
        self.cumulative_time_shutdown = drs.Timer(
            "cumulative_time_shutdown", initial_value=0.0
        )

        self.mode_timers = {
            "MODE_A": self.cumulative_time_mode_a,
            "MODE_A_CONTINGENCY": self.cumulative_time_mode_a_contingency,
            "MODE_A_MINE_SURGING": self.cumulative_time_mode_a_surging,
            "MODE_B": self.cumulative_time_mode_b,
            "MODE_B_CONTINGENCY": self.cumulative_time_mode_b_contingency,
            "MODE_B_MINE_SURGING": self.cumulative_time_mode_b_surging,
            "SHUTDOWN": self.cumulative_time_shutdown,
        }

        self.total_system_ore_mass = drs.Level(
            "total_system_ore_mass", initial_value=self.target_ore_stock_level
        )

        self.target_stock1_outflow_rate = drs.Variable(
            "target_stock1_outflow_rate", 0.0
        )
        self.target_stock2_outflow_rate = drs.Variable(
            "target_stock2_outflow_rate", 0.0
        )
        self.target_mine_mass_rate = drs.Variable("target_mine_mass_rate", 0.0)

    @property
    def total_duration(self) -> float:
        """Returns total accumulated duration across all operating modes."""
        return sum(t.value for t in self.mode_timers.values())

    def active_duration(self, current_time: float = -1.0) -> float:
        """Operational duration excluding shutdown time."""
        if current_time < 0.0:
            current_time = self.total_duration
        return max(0.0, current_time - self.cumulative_time_shutdown.value)

    def reset_mode_timers(self) -> None:
        """Reset all operating mode timers (e.g. at the end of warmup)."""
        for timer in self.mode_timers.values():
            timer.reset()

    def get_target_rates(
        self,
        campaign_mode: OperatingMode,
        ore1_level: float,
        ore2_level: float,
        stockpile2_routing_fraction: float,
    ) -> Tuple[float, float, float]:
        """Determines active operational state and computes draw rates in tonnes/day."""
        resolved_mode = self._resolve_operating_mode(
            campaign_mode, ore1_level, ore2_level
        )
        self.active_operating_mode.value = resolved_mode

        mode_name = resolved_mode.name
        self._update_mode_timers(mode_name)

        ore1_rate, ore2_rate = self._read_milling_rates(mode_name)

        if "_MINE_SURGING" in mode_name:
            self.total_system_ore_mass.lower_threshold = self.target_ore_stock_level
            p = stockpile2_routing_fraction
            if mode_name == "MODE_A_MINE_SURGING":
                effective_fraction = max(1.0 - p, 0.01)
                mine_target = ore1_rate / effective_fraction
            else:
                effective_fraction = max(p, 0.01)
                mine_target = ore2_rate / effective_fraction
        else:
            self.total_system_ore_mass.lower_threshold = -math.inf
            mine_target = ore1_rate + ore2_rate

        self.total_system_ore_mass.value = ore1_level + ore2_level
        self.total_system_ore_mass.rate = self.ore1_stock.rate + self.ore2_stock.rate

        self.target_stock1_outflow_rate.value = ore1_rate
        self.target_stock2_outflow_rate.value = ore2_rate
        self.target_mine_mass_rate.value = mine_target

        return ore1_rate, ore2_rate, mine_target

    def _resolve_operating_mode(
        self, campaign_mode: OperatingMode, ore1: float, ore2: float
    ) -> OperatingMode:
        c_name = campaign_mode.name
        if c_name == "SHUTDOWN":
            return MILL_MODES["SHUTDOWN"]

        current_name = self.active_operating_mode.value.name
        eps = 1e-9
        total_stock = ore1 + ore2

        if not current_name.startswith(c_name):
            current_name = c_name
            if total_stock > self.target_ore_stock_level + 1e-6:
                return MILL_MODES[c_name + "_MINE_SURGING"]
            return MILL_MODES[c_name]

        if "_CONTINGENCY" in current_name:
            if self._contingency_complete():
                self.current_contingency_duration.reset()
                return MILL_MODES[c_name]
            if c_name == "MODE_A" and ore1 <= eps:
                return MILL_MODES["MODE_A_MINE_SURGING"]
            if c_name == "MODE_B" and ore2 <= eps:
                return MILL_MODES["MODE_B_MINE_SURGING"]
            return MILL_MODES[current_name]

        if "_MINE_SURGING" in current_name:
            if total_stock <= self.target_ore_stock_level + 1e-6:
                return MILL_MODES[c_name]
            return MILL_MODES[current_name]

        if c_name == "MODE_A":
            if ore1 <= eps:
                return MILL_MODES["MODE_A_MINE_SURGING"]
            if ore2 <= eps:
                self.current_contingency_duration.reset()
                return MILL_MODES["MODE_A_CONTINGENCY"]
            return MILL_MODES["MODE_A"]

        if c_name == "MODE_B":
            if ore1 <= eps:
                self.current_contingency_duration.reset()
                return MILL_MODES["MODE_B_CONTINGENCY"]
            if ore2 <= eps:
                return MILL_MODES["MODE_B_MINE_SURGING"]
            return MILL_MODES["MODE_B"]

        return MILL_MODES[c_name]

    def _contingency_complete(self) -> bool:
        threshold = self.duration_of_contingency_segments
        self.current_contingency_duration.upper_threshold = threshold
        return self.current_contingency_duration.value >= (threshold - 1e-6)

    def _read_milling_rates(self, mode_name: str) -> Tuple[float, float]:
        sp = self.setpoints
        if mode_name in ("MODE_A", "MODE_A_MINE_SURGING"):
            return sp.mode_a_ore1, sp.mode_a_ore2
        elif mode_name == "MODE_A_CONTINGENCY":
            return sp.mode_a_contingency_ore1, 0.0
        elif mode_name in ("MODE_B", "MODE_B_MINE_SURGING"):
            return sp.mode_b_ore1, sp.mode_b_ore2
        elif mode_name == "MODE_B_CONTINGENCY":
            return 0.0, sp.mode_b_contingency_ore2
        else:  # SHUTDOWN
            return 0.0, 0.0

    def _update_mode_timers(self, mode_name: str) -> None:
        for name, timer in self.mode_timers.items():
            timer.rate = 1.0 if name == mode_name else 0.0

        if mode_name in self._CONTINGENCY_MODES:
            self.current_contingency_duration.rate = 1.0
            self.current_contingency_duration.lower_threshold = -math.inf
            self.current_contingency_duration.upper_threshold = (
                self.duration_of_contingency_segments
            )
        else:
            self.current_contingency_duration.rate = 0.0

    def process(self, mass_rate: float) -> None:
        """Draw ``mass_rate`` into the plant for one engine step."""
        self.rate = mass_rate
        self.cumulative_milled_mass.rate = self.actual_rate

    def levels(self) -> Sequence[drs.Level]:
        """Return all stateful Level instances owned by the metallurgical plant."""
        return (
            self.cumulative_milled_mass,
            self.current_contingency_duration,
            *self.mode_timers.values(),
            self.total_system_ore_mass,
        )

    def time_to_event(self) -> float:
        """Time until any owned level reaches a threshold."""
        min_dt = math.inf
        for lvl in self.levels():
            dt = lvl.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt

    def step(self, dt: float) -> None:
        """Advance all owned levels forward by dt."""
        for lvl in self.levels():
            lvl.step(dt)

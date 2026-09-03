"""Self-contained RL controller managing campaign timers and RL actions for MiningRLEnv."""

from __future__ import annotations

from typing import Iterator, Mapping, Optional, Sequence

import drs
from drs_mining.components.modes import OperatingMode, RequireDecision


class RL_MineController(drs.Module):
    """Supervisory controller managing campaign timers and RL actions for MiningRLEnv."""

    def __init__(
        self,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        critical_ore2_level: float = 20400.0,
        target_total_stock: float = 60000.0,
        modes: Optional[Mapping[str, OperatingMode]] = None,
    ):
        super().__init__()
        self.duration_of_production_campaigns = float(duration_of_production_campaigns)
        self.duration_of_shutdowns = float(duration_of_shutdowns)
        self.duration_of_contingency_segments = float(duration_of_contingency_segments)
        self.critical_ore2_level = float(critical_ore2_level)
        self.target_total_stock = float(target_total_stock)

        self.modes = dict(
            modes
            or {
                "MODE_A": OperatingMode(
                    "MODE_A", id=0, draw_rates={"Ore1Stock": 3600.0, "Ore2Stock": 2400.0}
                ),
                "MODE_A_CONTINGENCY": OperatingMode(
                    "MODE_A_CONTINGENCY",
                    id=1,
                    draw_rates={"Ore1Stock": 3900.0, "Ore2Stock": 0.0},
                ),
                "MODE_A_MINE_SURGING": OperatingMode(
                    "MODE_A_MINE_SURGING",
                    id=2,
                    draw_rates={"Ore1Stock": 3600.0, "Ore2Stock": 2400.0},
                ),
                "MODE_B": OperatingMode(
                    "MODE_B", id=3, draw_rates={"Ore1Stock": 4600.0, "Ore2Stock": 800.0}
                ),
                "MODE_B_CONTINGENCY": OperatingMode(
                    "MODE_B_CONTINGENCY",
                    id=4,
                    draw_rates={"Ore1Stock": 0.0, "Ore2Stock": 2500.0},
                ),
                "MODE_B_MINE_SURGING": OperatingMode(
                    "MODE_B_MINE_SURGING",
                    id=5,
                    draw_rates={"Ore1Stock": 4600.0, "Ore2Stock": 800.0},
                ),
                "SHUTDOWN": OperatingMode(
                    "SHUTDOWN", id=6, draw_rates={"Ore1Stock": 0.0, "Ore2Stock": 0.0}
                ),
            }
        )

        self.active_campaign_mode = drs.Variable(
            "active_campaign_mode", self.modes["MODE_A"]
        )
        self.active_operating_mode = drs.Variable(
            "active_operating_mode", self.modes["MODE_A"]
        )
        self.modes["MODE_A"].activate()

        self.current_campaign_duration = drs.Timer(
            "current_campaign_duration", initial_value=0.0
        )
        self.current_campaign_duration.rate = 1.0
        self.current_campaign_duration.upper_threshold = (
            self.duration_of_production_campaigns
        )

        self.current_contingency_duration = drs.Timer(
            "current_contingency_duration", initial_value=0.0
        )
        self.current_contingency_duration.rate = 0.0
        self.current_contingency_duration.upper_threshold = (
            self.duration_of_contingency_segments
        )

        self.pending_rl_action: Optional[int] = None

    def get_draw_rates(self, mode: OperatingMode) -> Mapping[str, float]:
        return mode.draw_rates

    def update_campaign(self, ore2_stock_level: float = 0.0) -> OperatingMode:
        thresh = (
            self.duration_of_shutdowns
            if self.active_campaign_mode.value.name == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        self.current_campaign_duration.upper_threshold = thresh
        if self.current_campaign_duration.value >= (thresh - 1e-6):
            self.current_campaign_duration.reset()
            if self.active_campaign_mode.value.name == "SHUTDOWN":
                if self.pending_rl_action is not None:
                    action = self.pending_rl_action
                    self.pending_rl_action = None
                    action_modes = [
                        self.modes["MODE_A"],
                        self.modes["MODE_B"],
                        self.modes["MODE_A_MINE_SURGING"],
                        self.modes["MODE_B_MINE_SURGING"],
                    ]
                    self.active_campaign_mode.value = action_modes[action]
                else:
                    raise RequireDecision()
            else:
                self.active_campaign_mode.value = self.modes["SHUTDOWN"]

        thresh = (
            self.duration_of_shutdowns
            if self.active_campaign_mode.value.name == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        self.current_campaign_duration.upper_threshold = thresh
        return self.active_campaign_mode.value

    def resolve_operating_mode(
        self,
        campaign_mode: OperatingMode,
        ore1_level: float,
        ore2_level: float,
    ) -> OperatingMode:
        c_name = campaign_mode.name
        if c_name == "SHUTDOWN":
            resolved = self.modes["SHUTDOWN"]
            self._set_active_mode(resolved)
            return resolved

        current_name = self.active_operating_mode.value.name
        total_stock = ore1_level + ore2_level

        if not current_name.startswith(c_name):
            resolved = (
                self.modes[f"{c_name}_MINE_SURGING"]
                if total_stock > self.target_total_stock + 1e-6
                else self.modes[c_name]
            )
            self._set_active_mode(resolved)
            return resolved

        if "_CONTINGENCY" in current_name:
            if self.current_contingency_duration.value >= (
                self.duration_of_contingency_segments - 1e-6
            ):
                self.current_contingency_duration.reset()
                resolved = self.modes[c_name]
                self._set_active_mode(resolved)
                return resolved
            return self.modes[current_name]

        if c_name == "MODE_A" and ore2_level <= 1e-6:
            self.current_contingency_duration.reset()
            self.current_contingency_duration.upper_threshold = (
                self.duration_of_contingency_segments
            )
            resolved = self.modes["MODE_A_CONTINGENCY"]
            self._set_active_mode(resolved)
            return resolved
        elif c_name == "MODE_B" and ore1_level <= 1e-6:
            self.current_contingency_duration.reset()
            self.current_contingency_duration.upper_threshold = (
                self.duration_of_contingency_segments
            )
            resolved = self.modes["MODE_B_CONTINGENCY"]
            self._set_active_mode(resolved)
            return resolved

        resolved = (
            self.modes[f"{c_name}_MINE_SURGING"]
            if total_stock > self.target_total_stock + 1e-6
            else self.modes[c_name]
        )
        self._set_active_mode(resolved)
        return resolved

    def _set_active_mode(self, mode: OperatingMode) -> None:
        if self.active_operating_mode.value != mode:
            self.active_operating_mode.value.deactivate()
            mode.activate()
            self.active_operating_mode.value = mode
        if "_CONTINGENCY" in mode.name:
            self.current_contingency_duration.rate = 1.0
        else:
            self.current_contingency_duration.rate = 0.0

    def levels(self) -> Sequence[drs.Level]:
        return (
            self.current_campaign_duration,
            self.current_contingency_duration,
            *(m.timer for m in self.modes.values()),
        )

    def variables(self) -> Iterator[drs.Variable]:
        yield self.current_campaign_duration
        yield self.current_contingency_duration
        for m in self.modes.values():
            yield m.timer

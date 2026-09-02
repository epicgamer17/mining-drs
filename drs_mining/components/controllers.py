"""Supervisory controllers managing campaign timers, operating modes, and draw setpoints."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import drs
from .modes import OperatingMode, RequireDecision
from drs_mining.config import MILL_MODES


@dataclass
class ModeSetpoints:
    """Milling draw setpoints and mine extraction targets for an operating mode."""

    draw_rates: Mapping[str, float] = field(default_factory=dict)
    mine_target: Optional[float] = None


class OperatingModeController(drs.Module):
    """Supervisory controller managing campaign timers, mode transitions, and setpoints."""

    def __init__(
        self,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        critical_ore2_level: float = 20400.0,
        target_total_stock: float = 60000.0,
        mode_setpoints: Optional[Mapping[str, ModeSetpoints]] = None,
        initial_mode: OperatingMode = MILL_MODES["MODE_A"],
    ):
        super().__init__()
        self.duration_of_production_campaigns = float(duration_of_production_campaigns)
        self.duration_of_shutdowns = float(duration_of_shutdowns)
        self.duration_of_contingency_segments = float(duration_of_contingency_segments)
        self.critical_ore2_level = float(critical_ore2_level)
        self.target_total_stock = float(target_total_stock)

        self.mode_setpoints: dict[str, ModeSetpoints] = dict(mode_setpoints or {})

        self.active_campaign_mode = drs.Variable("active_campaign_mode", initial_mode)
        self.active_operating_mode = drs.Variable("active_operating_mode", initial_mode)

        self.current_campaign_duration = drs.Timer(
            "current_campaign_duration", initial_value=0.0
        )
        self.current_contingency_duration = drs.Timer(
            "current_contingency_duration", initial_value=0.0
        )

        # Dynamic mode timers to record total time spent in each mode
        self.mode_timers: dict[str, drs.Timer] = {}
        for mode_name in MILL_MODES:
            timer = drs.Timer(f"cumulative_time_{mode_name.lower()}", initial_value=0.0)
            timer.rate = 0.0
            self.mode_timers[mode_name] = timer

        self.is_rl_controlled: bool = False
        self.pending_rl_action: Optional[int] = None
        self._update_campaign_timer()

    @property
    def current_target_duration(self) -> float:
        if self.active_campaign_mode.value.name == "SHUTDOWN":
            return self.duration_of_shutdowns
        return self.duration_of_production_campaigns

    def get_draw_rates(self, mode: OperatingMode) -> Mapping[str, float]:
        """Get planned draw rates for the given operating mode."""
        if mode.name in self.mode_setpoints:
            return self.mode_setpoints[mode.name].draw_rates
        from drs_mining.config import MILL_MODE_CONFIGS
        if mode.name in MILL_MODE_CONFIGS:
            return MILL_MODE_CONFIGS[mode.name].draw_rates
        return {}

    def update_campaign(self, ore2_stock_level: float) -> OperatingMode:
        """Check campaign timer expiry and transition between production and shutdown."""
        if self._is_campaign_complete():
            self.current_campaign_duration.reset()

            if self.active_campaign_mode.value.name == "SHUTDOWN":
                if self.is_rl_controlled:
                    if self.pending_rl_action is not None:
                        action = self.pending_rl_action
                        self.pending_rl_action = None
                        action_modes = [
                            MILL_MODES["MODE_A"],
                            MILL_MODES["MODE_B"],
                            MILL_MODES["MODE_A_MINE_SURGING"],
                            MILL_MODES["MODE_B_MINE_SURGING"],
                        ]
                        next_mode = action_modes[action]
                        self.active_campaign_mode.value = next_mode
                    else:
                        raise RequireDecision()
                else:
                    if ore2_stock_level > self.critical_ore2_level:
                        next_mode = MILL_MODES["MODE_A"]
                    else:
                        next_mode = MILL_MODES["MODE_B"]
                    self.active_campaign_mode.value = next_mode
            else:
                self.active_campaign_mode.value = MILL_MODES["SHUTDOWN"]

        self._update_campaign_timer()
        return self.active_campaign_mode.value

    def resolve_operating_mode(
        self,
        campaign_mode: OperatingMode,
        ore1_level: float,
        ore2_level: float,
    ) -> OperatingMode:
        """Resolve active operating mode, handling contingencies and stockpile surging."""
        c_name = campaign_mode.name
        if c_name == "SHUTDOWN":
            resolved = MILL_MODES["SHUTDOWN"]
            self._set_active_mode(resolved)
            return resolved

        current_name = self.active_operating_mode.value.name
        total_stock = ore1_level + ore2_level

        if not current_name.startswith(c_name):
            current_name = c_name
            if total_stock > self.target_total_stock + 1e-6:
                resolved = MILL_MODES[f"{c_name}_MINE_SURGING"]
            else:
                resolved = MILL_MODES[c_name]
            self._set_active_mode(resolved)
            return resolved

        if "_CONTINGENCY" in current_name:
            if self.current_contingency_duration.value >= (self.duration_of_contingency_segments - 1e-6):
                self.current_contingency_duration.reset()
                resolved = MILL_MODES[c_name]
                self._set_active_mode(resolved)
                return resolved
            self._set_active_mode(MILL_MODES[current_name])
            return MILL_MODES[current_name]

        # Check starvation triggers for contingency
        if c_name == "MODE_A" and ore2_level <= 1e-6:
            self.current_contingency_duration.reset()
            self.current_contingency_duration.upper_threshold = self.duration_of_contingency_segments
            resolved = MILL_MODES["MODE_A_CONTINGENCY"]
            self._set_active_mode(resolved)
            return resolved
        elif c_name == "MODE_B" and ore1_level <= 1e-6:
            self.current_contingency_duration.reset()
            self.current_contingency_duration.upper_threshold = self.duration_of_contingency_segments
            resolved = MILL_MODES["MODE_B_CONTINGENCY"]
            self._set_active_mode(resolved)
            return resolved

        # Surging checks
        if total_stock > self.target_total_stock + 1e-6:
            resolved = MILL_MODES[f"{c_name}_MINE_SURGING"]
        else:
            resolved = MILL_MODES[c_name]

        self._set_active_mode(resolved)
        return resolved

    def _set_active_mode(self, mode: OperatingMode) -> None:
        self.active_operating_mode.value = mode
        mode_name = mode.name

        # Update mode timer integration rates
        for name, timer in self.mode_timers.items():
            timer.rate = 1.0 if name == mode_name else 0.0

        if "_CONTINGENCY" in mode_name:
            self.current_contingency_duration.rate = 1.0
            self.current_contingency_duration.upper_threshold = self.duration_of_contingency_segments
        else:
            self.current_contingency_duration.rate = 0.0

    def _is_campaign_complete(self) -> bool:
        thresh = self.current_target_duration
        self.current_campaign_duration.upper_threshold = thresh
        return self.current_campaign_duration.value >= (thresh - 1e-6)

    def _update_campaign_timer(self) -> None:
        thresh = self.current_target_duration
        self.current_campaign_duration.rate = 1.0
        self.current_campaign_duration.lower_threshold = -math.inf
        self.current_campaign_duration.upper_threshold = thresh

    def reset_mode_timers(self) -> None:
        """Reset all cumulative mode timers to zero."""
        for timer in self.mode_timers.values():
            timer.reset()

    def levels(self) -> Sequence[drs.Level]:
        return (
            self.current_campaign_duration,
            self.current_contingency_duration,
            *self.mode_timers.values(),
        )

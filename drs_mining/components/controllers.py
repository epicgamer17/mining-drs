"""High-level supervisory controller managing campaign timers and operating mode transitions."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import drs
from .modes import OperatingMode, RequireDecision
from drs_mining.config import MILL_MODES


class OperatingModeController(drs.Module):
    """High-level supervisory controller managing campaign timers and operating mode transitions."""

    ACTION_MODES: Sequence[OperatingMode] = (
        MILL_MODES["MODE_A"],
        MILL_MODES["MODE_B"],
        MILL_MODES["MODE_A_MINE_SURGING"],
        MILL_MODES["MODE_B_MINE_SURGING"],
    )

    def __init__(
        self,
        duration_of_production_campaigns: float,
        duration_of_shutdowns: float,
        critical_ore2_level: float,
        initial_mode: OperatingMode = MILL_MODES["MODE_A"],
    ):
        super().__init__()
        self.duration_of_production_campaigns = duration_of_production_campaigns
        self.duration_of_shutdowns = duration_of_shutdowns
        self.critical_ore2_level = critical_ore2_level

        self.active_campaign_mode = drs.Variable("active_campaign_mode", initial_mode)
        self.current_campaign_duration = drs.Timer(
            "current_campaign_duration", initial_value=0.0
        )

        self.is_rl_controlled: bool = False
        self.pending_rl_action: Optional[int] = None
        self._update_campaign_timer()

    @property
    def current_target_duration(self) -> float:
        """Duration target for active campaign segment (production vs shutdown)."""
        if self.active_campaign_mode.value.name == "SHUTDOWN":
            return self.duration_of_shutdowns
        return self.duration_of_production_campaigns

    def update(
        self,
        ore2_stock_level: float,
        total_stock_level: Optional[float] = None,
    ) -> OperatingMode:
        """Check campaign timer expiry and transition between production and shutdown."""
        if self._is_campaign_complete():
            self.current_campaign_duration.reset()

            if self.active_campaign_mode.value.name == "SHUTDOWN":
                if self.is_rl_controlled:
                    if self.pending_rl_action is not None:
                        action = self.pending_rl_action
                        self.pending_rl_action = None
                        next_mode = self.ACTION_MODES[action]
                        self.active_campaign_mode.value = next_mode
                    else:
                        raise RequireDecision()
                else:
                    next_mode = self._choose_next_campaign_mode(ore2_stock_level)
                    self.active_campaign_mode.value = next_mode
            else:
                self.active_campaign_mode.value = MILL_MODES["SHUTDOWN"]

        self._update_campaign_timer()
        return self.active_campaign_mode.value

    def _choose_next_campaign_mode(self, ore2_stock_level: float) -> OperatingMode:
        if ore2_stock_level > self.critical_ore2_level:
            return MILL_MODES["MODE_A"]
        return MILL_MODES["MODE_B"]

    def _is_campaign_complete(self) -> bool:
        thresh = self.current_target_duration
        self.current_campaign_duration.upper_threshold = thresh
        return self.current_campaign_duration.value >= (thresh - 1e-6)

    def _update_campaign_timer(self) -> None:
        thresh = self.current_target_duration
        self.current_campaign_duration.rate = 1.0
        self.current_campaign_duration.lower_threshold = -math.inf
        self.current_campaign_duration.upper_threshold = thresh

    def levels(self) -> tuple[drs.Level, ...]:
        return (self.current_campaign_duration,)

    def is_terminating_condition_met(self) -> bool:
        return False

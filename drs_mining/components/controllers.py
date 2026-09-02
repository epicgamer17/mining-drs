import math
from typing import List, Mapping, Sequence, Optional, Tuple
import drs
from .modes import OperatingMode
from drs_mining.config import MILL_MODES


class OperatingModeController(drs.Module):
    """High-level supervisory controller that manages campaign timers and operating mode transitions."""

    def __init__(
        self,
        duration_of_production_campaigns: float,
        duration_of_shutdowns: float,
        critical_ore2_level: float,
        target_ore_stock_level: float = 60000.0,
        total_ore_to_extract: float = 6600000.0,
        initial_mode: OperatingMode = MILL_MODES["MODE_A"],
    ):
        super().__init__()
        self.duration_of_production_campaigns = duration_of_production_campaigns
        self.duration_of_shutdowns = duration_of_shutdowns
        self.critical_ore2_level = critical_ore2_level
        self.target_ore_stock_level = target_ore_stock_level
        self.total_ore_to_extract = total_ore_to_extract

        self.active_campaign_mode = drs.Variable("active_campaign_mode", initial_mode)

        self.current_campaign_duration = drs.Timer(
            "current_campaign_duration", initial_value=0.0
        )

    def update(
        self,
        ore2_stock_level: float,
        total_stock_level: Optional[float] = None,
    ) -> OperatingMode:
        """Advance campaign timer and determine active campaign mode (MODE_A, MODE_B, SHUTDOWN)."""
        name = self.active_campaign_mode.value.name

        if self._campaign_complete():
            if name == "SHUTDOWN":
                rl_action = getattr(self, "pending_rl_action", None)
                if rl_action is not None:
                    self.current_campaign_duration.reset()
                    self.pending_rl_action = None
                    action_modes = [
                        MILL_MODES["MODE_A"],
                        MILL_MODES["MODE_B"],
                        MILL_MODES["MODE_A_MINE_SURGING"],
                        MILL_MODES["MODE_B_MINE_SURGING"],
                    ]
                    next_mode = action_modes[rl_action]
                    self.active_campaign_mode.value = next_mode
                    self._update_campaign_timers(next_mode.name)
                    return next_mode

                if hasattr(self, "pending_rl_action"):
                    from .modes import RequireDecision
                    raise RequireDecision()

                self.current_campaign_duration.reset()
                next_mode = self._choose_next_campaign_mode(
                    ore2_stock_level, total_stock_level
                )
                self.active_campaign_mode.value = next_mode
            else:
                self.current_campaign_duration.reset()
                self.active_campaign_mode.value = MILL_MODES["SHUTDOWN"]

        active_name = self.active_campaign_mode.value.name
        self._update_campaign_timers(active_name)
        return self.active_campaign_mode.value

    def _choose_next_campaign_mode(
        self, ore2_stock_level: float, total_stock_level: Optional[float] = None
    ) -> OperatingMode:
        if ore2_stock_level > self.critical_ore2_level:
            return MILL_MODES["MODE_A"]
        return MILL_MODES["MODE_B"]

    def _campaign_complete(self) -> bool:
        threshold = (
            self.duration_of_shutdowns
            if self.active_campaign_mode.value.name == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        self.current_campaign_duration.upper_threshold = threshold
        return self.current_campaign_duration.value >= (threshold - 1e-6)

    def _update_campaign_timers(self, name: str):
        threshold = (
            self.duration_of_shutdowns
            if name == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        self.current_campaign_duration.rate = 1.0
        self.current_campaign_duration.lower_threshold = -math.inf
        self.current_campaign_duration.upper_threshold = threshold

    def levels(self) -> tuple[drs.Level, ...]:
        return (self.current_campaign_duration,)

    def is_terminating_condition_met(self) -> bool:
        return False

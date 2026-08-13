import drs
from drs_mining.components.modes import RequireDecision
from drs_mining.components.controllers import ConcentratorController
from drs_mining.components.modes import MODES


class RL_MineController(ConcentratorController):
    """A modified controller that yields to the RL agent using RequireDecision."""

    def __init__(
        self,
        mine,
        fleet,
        plant,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
    ):
        super().__init__(
            mine=mine,
            fleet=fleet,
            plant=plant,
            target_ore_stock_level=target_ore_stock_level,
            critical_ore2_level=critical_ore2_level,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        )
        self.pending_rl_action = None

    def controller_decision(self):
        m = self.active_operating_mode.value.name

        if self.is_campaign_complete():
            if m == "SHUTDOWN":
                if self.pending_rl_action is not None:
                    self.reset_campaign_timer()
                    action_map = [
                        MODES["MODE_A"],
                        MODES["MODE_B"],
                        MODES["MODE_A_MINE_SURGING"],
                        MODES["MODE_B_MINE_SURGING"],
                    ]
                    chosen = action_map[self.pending_rl_action]
                    self.pending_rl_action = None
                    return chosen
                else:
                    raise RequireDecision()
            else:
                self.reset_campaign_timer()
                return MODES["SHUTDOWN"]

        if m.endswith("_CONTINGENCY") and self.is_contingency_complete():
            self.reset_contingency_timer()
            return MODES[m.replace("_CONTINGENCY", "")]

        if (
            m.endswith("_MINE_SURGING")
            and self.total_system_ore_mass.value <= self.target_ore_stock_level
        ):
            return MODES[m.replace("_MINE_SURGING", "")]

        return None

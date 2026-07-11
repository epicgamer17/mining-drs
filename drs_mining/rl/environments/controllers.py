import drs
from drs_mining.components.modes import RequireDecision
from drs_mining.components.config import ConcentratorConfig
from drs_mining.components.controllers import ConcentratorController
from drs_mining.components.modes import MODES


class RL_MineController(ConcentratorController):
    """A modified controller that yields to the RL agent using RequireDecision."""

    def __init__(self, config: ConcentratorConfig, mine, fleet, plant):
        super().__init__(config, mine, fleet, plant)
        self.pending_rl_action = None  # Stores the action passed in by env.step()

    def controller_decision(self):
        m = self.current_mode.value.name

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
            and self.plant.ore_stock.value <= self.config.target_ore_stock_level
        ):
            return MODES[m.replace("_MINE_SURGING", "")]

        return None

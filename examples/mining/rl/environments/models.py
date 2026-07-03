import drs
from examples.mining.components.models import ConcentratorModel
from examples.mining.components.config import ConcentratorConfig
from .controllers import RL_MineController


class RL_ConcentratorModel(ConcentratorModel):
    def __init__(self, config: ConcentratorConfig, enable_telemetry: bool = False):
        super().__init__(config, enable_telemetry=enable_telemetry)

        # Replace standard controller with RL Controller
        self.controller = RL_MineController(
            self.config,
            self.mine,
            self.fleet,
            self.plant,
        )

        if not enable_telemetry:
            # Remove telemetry to prevent memory leaks during RL training
            if getattr(self, "telemetry", None) is not None:
                self.telemetry = None

    def is_terminating_condition_met(self) -> bool:
        # In RL, we terminate strictly when the extraction limit is reached.
        # The agent receives a terminal penalty for the stockpile offset.
        c = self.config
        return self.mine.cumulative_extracted_mass.value >= c.total_ore_to_extract

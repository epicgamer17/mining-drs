import math
import drs
from drs_mining.components.models import ConcentratorModel
from .controllers import RL_MineController


class RL_ConcentratorModel(ConcentratorModel):
    def __init__(
        self,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        target_ore_stock_level: float = 60000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        replication_length: float = math.inf,
        enable_telemetry: bool = False,
    ):
        super().__init__(
            mean_ore_fraction=mean_ore_fraction,
            std_dev_ore_fraction=std_dev_ore_fraction,
            target_ore_stock_level=target_ore_stock_level,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            critical_ore2_level=critical_ore2_level,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            min_ore_mass=min_ore_mass,
            max_ore_mass=max_ore_mass,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
            replication_length=replication_length,
            enable_telemetry=enable_telemetry,
        )

        # Replace standard controller with RL Controller
        self.controller = RL_MineController(
            mine=self.mine,
            fleet=self.fleet,
            plant=self.plant,
            target_ore_stock_level=target_ore_stock_level,
            critical_ore2_level=critical_ore2_level,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        )

        if not enable_telemetry:
            # Remove telemetry to prevent memory leaks during RL training
            if getattr(self, "telemetry", None) is not None:
                self.telemetry = None

    def is_terminating_condition_met(self) -> bool:
        return self.mine.cumulative_extracted_mass.value >= self.total_ore_to_extract

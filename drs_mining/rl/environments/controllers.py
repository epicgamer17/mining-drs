from drs_mining.components.controllers import BlendingController


class RL_MineController(BlendingController):
    """State container that exposes pending_rl_action to the operating policy.

    The top-level policy (MiningRLEnv._step_policy) reads pending_rl_action when
    a campaign decision is required and raises RequireDecision to yield to the
    RL agent otherwise.
    """

    def __init__(
        self,
        faces=None,
        mine=None,
        fleet=None,
        plant=None,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        total_ore_to_extract: float = 6600000.0,
        **kwargs,
    ):
        super().__init__(
            faces=faces,
            mine=mine,
            fleet=fleet,
            plant=plant,
            target_ore_stock_level=target_ore_stock_level,
            critical_ore2_level=critical_ore2_level,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            total_ore_to_extract=total_ore_to_extract,
            **kwargs,
        )
        self.pending_rl_action = None

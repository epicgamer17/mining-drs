from drs_mining.components.controllers import BlendingController


class RL_MineController(BlendingController):
    """State container that exposes pending_rl_action to the operating policy.

    The top-level policy (MiningRLEnv._step_policy) reads pending_rl_action when
    a campaign decision is required and raises RequireDecision to yield to the
    RL agent otherwise.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_rl_action = None

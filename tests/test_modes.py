import pytest
import drs
from drs_mining.components import OperatingMode
from drs_mining.rl.environments.controllers import RL_MineController
from examples.blending_modes.simulation import (
    create_blending_modes,
    update_campaign_mode,
    resolve_operating_mode,
)


def test_operating_mode_self_timing():
    mode = OperatingMode("TEST_MODE", id=99, draw_rates={"Ore1": 1000.0})
    assert mode.cumulative_time == 0.0
    assert mode.timer.rate == 0.0
    assert mode.draw_rates["Ore1"] == 1000.0

    mode.activate()
    assert mode.timer.rate == 1.0

    # Simulate 5.0 time units
    mode.timer.value = 5.0
    assert mode.cumulative_time == 5.0

    mode.deactivate()
    assert mode.timer.rate == 0.0
    assert mode.cumulative_time == 5.0

    mode.reset_timer()
    assert mode.cumulative_time == 0.0


def test_operating_mode_drs_engine_integration():
    mode = OperatingMode("MODE_A")
    mode.activate()

    engine = drs.DRSEngine()
    engine.register(mode)
    engine.run(until=5.0)

    assert mode.cumulative_time == pytest.approx(5.0)


def test_inlined_campaign_transition():
    modes = create_blending_modes()
    campaign_timer = drs.Timer("campaign_timer", initial_value=0.0)
    campaign_timer.rate = 1.0
    active_campaign = drs.Variable("active_campaign", modes["MODE_A"])

    # Before completion: stays in MODE_A
    campaign_timer.value = 20.0
    c_mode = update_campaign_mode(
        campaign_timer,
        active_campaign,
        ore2_stock_level=25000.0,
        modes=modes,
        campaign_duration=34.0,
        shutdown_duration=1.0,
        critical_ore2_level=20000.0,
    )
    assert c_mode.name == "MODE_A"

    # End of campaign (34.0 days) -> transitions to SHUTDOWN
    campaign_timer.value = 34.0
    c_mode = update_campaign_mode(
        campaign_timer,
        active_campaign,
        ore2_stock_level=25000.0,
        modes=modes,
        campaign_duration=34.0,
        shutdown_duration=1.0,
        critical_ore2_level=20000.0,
    )
    assert c_mode.name == "SHUTDOWN"
    assert active_campaign.value.name == "SHUTDOWN"

    # End of shutdown (1.0 day) -> above critical level -> transitions back to MODE_A
    campaign_timer.value = 1.0
    c_mode = update_campaign_mode(
        campaign_timer,
        active_campaign,
        ore2_stock_level=25000.0,
        modes=modes,
        campaign_duration=34.0,
        shutdown_duration=1.0,
        critical_ore2_level=20000.0,
    )
    assert c_mode.name == "MODE_A"
    assert active_campaign.value.name == "MODE_A"


def test_inlined_operating_mode_resolution():
    modes = create_blending_modes()
    contingency_timer = drs.Timer("contingency_timer", initial_value=0.0)
    contingency_timer.rate = 0.0

    mode_a = modes["MODE_A"]
    mode_a.activate()

    # Standard Mode A
    mode = resolve_operating_mode(
        campaign_mode=mode_a,
        current_mode=mode_a,
        ore1_level=30000.0,
        ore2_level=20000.0,
        contingency_timer=contingency_timer,
        modes=modes,
        target_total_stock=60000.0,
    )
    assert mode.name == "MODE_A"

    # Ore 2 starvation -> triggers contingency
    mode_contingency = resolve_operating_mode(
        campaign_mode=mode_a,
        current_mode=mode_a,
        ore1_level=30000.0,
        ore2_level=0.0,
        contingency_timer=contingency_timer,
        modes=modes,
        target_total_stock=60000.0,
    )
    assert mode_contingency.name == "MODE_A_CONTINGENCY"
    assert contingency_timer.rate == 1.0

    # Advance contingency timer -> contingency completes
    contingency_timer.value = 1.0
    mode_resumed = resolve_operating_mode(
        campaign_mode=mode_a,
        current_mode=mode_contingency,
        ore1_level=30000.0,
        ore2_level=20000.0,
        contingency_timer=contingency_timer,
        modes=modes,
        target_total_stock=60000.0,
    )
    assert mode_resumed.name == "MODE_A"
    assert contingency_timer.rate == 0.0

    # Surging: total stock > 60k
    mode_surging = resolve_operating_mode(
        campaign_mode=mode_a,
        current_mode=mode_a,
        ore1_level=40000.0,
        ore2_level=25000.0,
        contingency_timer=contingency_timer,
        modes=modes,
        target_total_stock=60000.0,
    )
    assert mode_surging.name == "MODE_A_MINE_SURGING"


def test_rl_mine_controller():
    ctrl = RL_MineController(
        duration_of_production_campaigns=34.0,
        duration_of_shutdowns=1.0,
        critical_ore2_level=20000.0,
    )
    assert ctrl.active_campaign_mode.value.name == "MODE_A"
    ctrl.current_campaign_duration.value = 34.0
    next_mode = ctrl.update_campaign(ore2_stock_level=25000.0)
    assert next_mode.name == "SHUTDOWN"

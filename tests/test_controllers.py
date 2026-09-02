import pytest
from drs_mining.components import OperatingModeController
from drs_mining.config import MILL_MODES


def test_operating_mode_controller_campaign_transition():
    ctrl = OperatingModeController(
        duration_of_production_campaigns=34.0,
        duration_of_shutdowns=1.0,
        critical_ore2_level=20000.0,
    )

    # Initial campaign mode is MODE_A
    assert ctrl.active_campaign_mode.value == MILL_MODES["MODE_A"]

    # Step timer by 34 days (end of campaign)
    ctrl.current_campaign_duration.value = 34.0
    next_mode = ctrl.update_campaign(ore2_stock_level=25000.0)

    # Should transition to SHUTDOWN
    assert next_mode == MILL_MODES["SHUTDOWN"]
    assert ctrl.active_campaign_mode.value == MILL_MODES["SHUTDOWN"]

    # Step shutdown timer by 1.0 day
    ctrl.current_campaign_duration.value = 1.0
    next_mode = ctrl.update_campaign(ore2_stock_level=25000.0)

    # Above critical -> next is MODE_A
    assert next_mode == MILL_MODES["MODE_A"]


def test_operating_mode_controller_contingency_and_surging():
    ctrl = OperatingModeController(
        duration_of_production_campaigns=34.0,
        duration_of_shutdowns=1.0,
        duration_of_contingency_segments=1.0,
        target_total_stock=60000.0,
    )

    # Standard Mode A
    mode = ctrl.resolve_operating_mode(
        MILL_MODES["MODE_A"],
        ore1_level=30000.0,
        ore2_level=20000.0,
    )
    assert mode == MILL_MODES["MODE_A"]
    assert ctrl.mode_timers["MODE_A"].rate == 1.0

    # Starvation of Ore 2 triggers contingency
    mode_contingency = ctrl.resolve_operating_mode(
        MILL_MODES["MODE_A"],
        ore1_level=30000.0,
        ore2_level=0.0,
    )
    assert mode_contingency == MILL_MODES["MODE_A_CONTINGENCY"]
    assert ctrl.mode_timers["MODE_A_CONTINGENCY"].rate == 1.0

    # Advance contingency timer so contingency completes (1.0 day duration)
    ctrl.current_contingency_duration.value = 1.0
    mode_resumed = ctrl.resolve_operating_mode(
        MILL_MODES["MODE_A"],
        ore1_level=30000.0,
        ore2_level=20000.0,
    )
    assert mode_resumed == MILL_MODES["MODE_A"]

    # Surging: total stock > target_total_stock (40k + 25k = 65k > 60k)
    mode_surging = ctrl.resolve_operating_mode(
        MILL_MODES["MODE_A"],
        ore1_level=40000.0,
        ore2_level=25000.0,
    )
    assert mode_surging == MILL_MODES["MODE_A_MINE_SURGING"]
    assert ctrl.mode_timers["MODE_A_MINE_SURGING"].rate == 1.0

import math

from drs_mining.components.modes import MODES
from drs_mining.components.models import ConcentratorModel
from drs_mining.components.config import ConcentratorConfig


def test_contingency_timer_resets_when_entering_mode_a_contingency():
    sim = ConcentratorModel(ConcentratorConfig())
    sim.ore2_stock.current_mass.value = 0.0
    sim.controller.active_operating_mode.value = MODES["MODE_A"]
    sim.controller.current_contingency_duration.value = 0.75

    sim.controller.forward()

    assert sim.controller.active_operating_mode.value is MODES["MODE_A_CONTINGENCY"]
    assert sim.controller.current_contingency_duration.value == 0.0


def test_contingency_timer_resets_when_entering_mode_b_contingency():
    sim = ConcentratorModel(ConcentratorConfig())
    sim.ore1_stock.current_mass.value = 0.0
    sim.controller.active_operating_mode.value = MODES["MODE_B"]
    sim.controller.current_contingency_duration.value = 0.75

    sim.controller.forward()

    assert sim.controller.active_operating_mode.value is MODES["MODE_B_CONTINGENCY"]
    assert sim.controller.current_contingency_duration.value == 0.0


def test_contingency_timer_resets_when_leaving_contingency():
    sim = ConcentratorModel(ConcentratorConfig())
    sim.controller.active_operating_mode.value = MODES["MODE_A_CONTINGENCY"]
    sim.controller.current_contingency_duration.value = (
        sim.controller.config.duration_of_contingency_segments
    )
    sim.controller.current_campaign_duration.value = 0.0

    sim.controller.forward()

    assert sim.controller.active_operating_mode.value is MODES["MODE_A"]
    assert sim.controller.current_contingency_duration.value == 0.0


def test_mine_surging_arms_component_ore_lower_bounds():
    sim = ConcentratorModel(ConcentratorConfig())

    sim.controller.total_system_ore_mass.value = sim.config.target_ore_stock_level * 1.1

    sim.controller.active_operating_mode.value = MODES["MODE_A_MINE_SURGING"]
    sim.controller.active_operating_mode.value.get_target_rates(sim)

    assert (
        sim.controller.total_system_ore_mass.lower_threshold
        == sim.config.target_ore_stock_level
    )

    sim.controller.active_operating_mode.value = MODES["MODE_B_MINE_SURGING"]
    sim.controller.active_operating_mode.value.get_target_rates(sim)

    assert (
        sim.controller.total_system_ore_mass.lower_threshold
        == sim.config.target_ore_stock_level
    )

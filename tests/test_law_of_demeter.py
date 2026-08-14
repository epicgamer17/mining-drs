import pytest
from drs_mining.components.mine_face import ConcentratorMineFace
from drs_mining.components.controllers import ConcentratorController
from drs_mining.components.models import ConcentratorModel


def test_mine_face_net_extracted_mass():
    mine = ConcentratorMineFace(
        total_ore_to_extract=6600000.0,
        ore_to_be_extracted_during_warming_period=600000.0,
    )
    mine.cumulative_extracted_mass.value = 1000000.0
    assert mine.net_extracted_mass == 400000.0


def test_controller_durations():
    sim = ConcentratorModel()
    controller = sim.controller

    controller.cumulative_time_mode_a.value = 10.0
    controller.cumulative_time_mode_b.value = 5.0
    controller.cumulative_time_shutdown.value = 2.0

    assert controller.total_duration == 17.0
    assert controller.active_duration(controller.total_duration) == 15.0
    assert controller.active_duration() == 15.0
    assert controller.active_duration(10.0) == 8.0


def test_model_delegation_properties():
    sim = ConcentratorModel()
    sim.ore1_stock.current_mass.value = 1000.0
    sim.ore2_stock.current_mass.value = 2000.0
    sim.fleet.stockpile2_routing_fraction.value = 0.35
    sim.controller.target_mine_mass_rate.value = 5000.0

    assert sim.ore1_mass == 1000.0
    assert sim.ore2_mass == 2000.0
    assert sim.total_stockpile_mass == 3000.0
    assert sim.stockpile2_routing_fraction == 0.35
    assert sim.target_mine_mass_rate == 5000.0


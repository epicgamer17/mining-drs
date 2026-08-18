from drs_mining.components.mine_face import MineFace
from drs_mining.components.factories import build_mining_simulation


def test_mine_face_net_extracted_mass():
    mine = MineFace(
        total_ore_to_extract=6600000.0,
        ore_to_be_extracted_during_warming_period=600000.0,
    )
    mine.cumulative_extracted_mass.value = 1000000.0
    assert mine.net_extracted_mass == 400000.0


def test_controller_durations():
    _, _, _, controller, _, _ = build_mining_simulation()

    controller.cumulative_time_mode_a.value = 10.0
    controller.cumulative_time_mode_b.value = 5.0
    controller.cumulative_time_shutdown.value = 2.0

    assert controller.total_duration == 17.0
    assert controller.active_duration(controller.total_duration) == 15.0
    assert controller.active_duration() == 15.0
    assert controller.active_duration(10.0) == 8.0


def test_flat_build_wiring():
    faces, fleet, plant, controller, ore1_stock, ore2_stock = (
        build_mining_simulation()
    )

    ore1_stock.current_mass.value = 1000.0
    ore2_stock.current_mass.value = 2000.0
    fleet.stockpile2_routing_fraction.value = 0.35
    controller.target_mine_mass_rate.value = 5000.0

    assert ore1_stock.current_mass.value == 1000.0
    assert ore2_stock.current_mass.value == 2000.0
    assert ore1_stock.current_mass.value + ore2_stock.current_mass.value == 3000.0
    assert fleet.stockpile2_routing_fraction.value == 0.35
    assert controller.target_mine_mass_rate.value == 5000.0
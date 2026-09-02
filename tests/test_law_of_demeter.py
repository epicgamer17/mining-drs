from drs_mining.components.mine_face import MineFace
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.factories import build_tactical_simulation


def test_mine_face_net_extracted_mass():
    gen = StochasticFaciesGenerator(mean_fraction=0.3, std_dev=0.05, prob_new_facies=0.3, variation_same_facies=0.01)
    mine = MineFace(
        name="mine_face",
        face_id=1,
        generator=gen,
        min_ore_mass=30000.0,
        max_ore_mass=50000.0,
        total_ore_to_extract=6600000.0,
        ore_to_be_extracted_during_warming_period=600000.0,
        mean_ore_fraction=0.3,
        std_dev_ore_fraction=0.05,
        prob_new_facies=0.3,
        variation_same_facies=0.01,
        initial_parcel_mass=30000.0,
    )
    mine.cumulative_extracted_mass.value = 1000000.0
    assert mine.net_extracted_mass == 400000.0


def test_controller_durations():
    _, plant, _, _, _ = build_tactical_simulation()

    plant.cumulative_time_mode_a.value = 10.0
    plant.cumulative_time_mode_b.value = 5.0
    plant.cumulative_time_shutdown.value = 2.0

    assert plant.total_duration == 17.0
    assert plant.active_duration(plant.total_duration) == 15.0
    assert plant.active_duration() == 15.0
    assert plant.active_duration(10.0) == 8.0


def test_flat_build_wiring():
    faces, plant, mode_controller, ore1_stock, ore2_stock = (
        build_tactical_simulation()
    )

    ore1_stock._level.value = 1000.0
    ore2_stock._level.value = 2000.0
    plant.target_mine_mass_rate.value = 5000.0

    assert ore1_stock.level == 1000.0
    assert ore2_stock.level == 2000.0
    assert ore1_stock.level + ore2_stock.level == 3000.0
    assert plant.target_mine_mass_rate.value == 5000.0
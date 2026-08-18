import pytest

from drs import Processor, Storage

from drs_mining.components import (
    MineFace,
    MetallurgicalPlant,
    ContinuousFleetLogistics,
    Stockpile,
)


def test_storage_level_reads_internal_level():
    storage = Storage(name="tank", capacity=1000.0, initial_level=250.0)
    assert storage.level == storage._level.value
    assert storage.level == 250.0


def test_storage_rate_setter_writes_internal_level():
    storage = Storage(name="tank", capacity=1000.0, initial_level=0.0)
    storage.rate = 42.0
    assert storage._level.rate == 42.0
    assert storage.rate == 42.0


def test_storage_bounds_initialized_to_capacity():
    storage = Storage(name="tank", capacity=1000.0, initial_level=100.0)
    assert storage._level.lower_threshold == 0.0
    assert storage._level.upper_threshold == 1000.0

    unbounded = Storage(name="tank", initial_level=100.0)
    assert unbounded._level.lower_threshold == 0.0
    assert unbounded._level.upper_threshold == float("inf")


def test_processor_exposes_rate_surface():
    proc = Processor(name="mill", max_rate=500.0)
    assert proc.max_rate == 500.0

    proc.target_rate = 300.0
    assert proc.actual_rate == 300.0
    assert proc.output_rate == proc.actual_rate


def test_processor_actual_rate_caps_and_applies_efficiency():
    proc = Processor(name="mill", max_rate=500.0)
    proc.target_rate = 800.0
    assert proc.actual_rate == 500.0

    proc.efficiency = 0.8
    proc.target_rate = 500.0
    assert proc.actual_rate == 400.0
    assert proc.output_rate == 400.0


def test_mining_stockpile_is_a_storage():
    stock = Stockpile(
        name="Ore1Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=0.0,
        capacity=1_000_000.0,
    )
    assert isinstance(stock, Storage)
    assert stock.level == stock._level.value
    assert stock._level.lower_threshold == 0.0
    assert stock._level.upper_threshold == 1_000_000.0

    stock.rate = 10.0
    assert stock._level.rate == 10.0
    assert stock.current_mass is stock._level


def test_mining_plant_is_a_processor():
    fleet = ContinuousFleetLogistics()
    ore1 = Stockpile(name="Ore1Stock", expected_attributes=["x"], initial_mass=100.0)
    ore2 = Stockpile(name="Ore2Stock", expected_attributes=["x"], initial_mass=100.0)
    plant = MetallurgicalPlant(None, fleet, ore1, ore2, max_rate=600.0)

    assert isinstance(plant, Processor)
    assert plant.max_rate == 600.0

    plant.target_rate = 450.0
    assert plant.actual_rate == 450.0

    plant.target_rate = 1e6
    assert plant.actual_rate == 600.0
    assert plant.output_rate == plant.actual_rate


def test_mining_face_is_a_processor_driven_by_target_rate():
    face = MineFace(
        mean_ore_fraction=0.3,
        std_dev_ore_fraction=0.05,
        min_ore_mass=30000.0,
        max_ore_mass=50000.0,
    )
    assert isinstance(face, Processor)

    face.target_rate = 100.0
    assert face.actual_rate == 100.0
    face.target_rate = 1e6
    assert face.actual_rate == 1e6
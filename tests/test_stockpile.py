import pytest
from drs import Storage, Flow, Entity


def test_storage_initialization():
    storage = Storage(
        name="CuStock",
        capacity=10000.0,
        initial_level=5000.0,
        initial_attributes={"Cu": 0.80, "Au": 1.5},
    )
    assert storage.level == 5000.0
    assert storage.capacity == 10000.0
    assert storage.grade("Cu") == pytest.approx(0.80)
    assert storage.grade("Au") == pytest.approx(1.5)
    assert storage.grade("Unknown") == 0.0


def test_storage_continuous_feed_and_draw():
    storage = Storage(
        name="BlendStock",
        capacity=20000.0,
        initial_level=10000.0,
        initial_attributes={"grade": 0.50},
    )

    inflow = Flow(rate=2000.0, attributes={"grade": 1.0})
    outflow = storage.feed_and_draw(inflow, draw_rate=1000.0)

    assert outflow.rate == 1000.0
    assert outflow.attributes["grade"] == pytest.approx(0.50)
    assert storage.rate == pytest.approx(1000.0)

    storage.step(1.0)
    assert storage.level == pytest.approx(11000.0)
    assert storage.grade("grade") == pytest.approx(6500.0 / 11000.0)


def test_storage_discrete_dump_and_scoop():
    storage = Storage(
        name="ROM_Pad",
        capacity=50000.0,
        initial_level=1000.0,
        initial_attributes={"Cu": 1.0},
    )

    # Dump discrete truckload: 100t @ 2.0% Cu
    truck = Entity(mass=100.0, attributes={"Cu": 2.0})
    storage.dump(truck)

    assert storage.level == 1100.0
    # Contained Cu: 1000*1.0 + 100*2.0 = 1200 / 1100 = 1.090909%
    assert storage.grade("Cu") == pytest.approx(1200.0 / 1100.0)

    # Scoop 200t discrete batch
    batch = storage.scoop(200.0)
    assert batch.mass == 200.0
    assert batch.attributes["Cu"] == pytest.approx(1200.0 / 1100.0)
    assert storage.level == 900.0
    # Remaining grade remains identical after scoop
    assert storage.grade("Cu") == pytest.approx(1200.0 / 1100.0)


def test_storage_empty_draw_limit():
    storage = Storage(name="EmptyStock", initial_level=0.0)
    assert storage.is_empty

    inflow = Flow(rate=500.0, attributes={"Cu": 0.5})
    outflow = storage.feed_and_draw(inflow, draw_rate=1000.0)

    assert outflow.rate == pytest.approx(500.0)
    assert storage.rate == pytest.approx(0.0)

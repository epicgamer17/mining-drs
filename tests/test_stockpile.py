import pytest
from drs_mining.components import Stockpile, Flow


def test_stockpile_initialization():
    sp = Stockpile(
        name="CuStock",
        capacity=10000.0,
        initial_mass=5000.0,
        initial_attributes={"Cu": 0.80, "Au": 1.5},
    )
    assert sp.level == 5000.0
    assert sp.capacity == 10000.0
    assert sp.grade("Cu") == pytest.approx(0.80)
    assert sp.grade("Au") == pytest.approx(1.5)
    assert sp.grade("Unknown") == 0.0


def test_stockpile_feed_and_draw_conservation():
    sp = Stockpile(
        name="BlendStock",
        capacity=20000.0,
        initial_mass=10000.0,
        initial_attributes={"grade": 0.50},
    )

    # Feed at 2000 t/d with grade 1.0, draw at 1000 t/d
    inflow = Flow(rate=2000.0, attributes={"grade": 1.0})
    outflow = sp.feed_and_draw(inflow, draw_rate=1000.0)

    assert outflow.rate == 1000.0
    # Outflow grade should match instantaneous stockpile grade (0.50)
    assert outflow.attributes["grade"] == pytest.approx(0.50)

    # Net rate of mass change should be 2000 - 1000 = 1000
    assert sp.rate == pytest.approx(1000.0)

    # Step forward by 1 day
    sp.step(1.0)
    assert sp.level == pytest.approx(11000.0)

    # Contained grade mass after 1 day:
    # Initial: 10000 * 0.5 = 5000
    # In: 2000 * 1.0 = 2000
    # Out: 1000 * 0.5 = 500
    # Expected total grade mass = 5000 + 2000 - 500 = 6500
    # New concentration = 6500 / 11000 = 0.590909...
    assert sp.grade("grade") == pytest.approx(6500.0 / 11000.0)


def test_stockpile_empty_draw_limit():
    sp = Stockpile(name="EmptyStock", initial_mass=0.0)
    assert sp.is_empty

    # If inflow is 500 but draw requested is 1000, actual draw is capped at 500
    inflow = Flow(rate=500.0, attributes={"Cu": 0.5})
    outflow = sp.feed_and_draw(inflow, draw_rate=1000.0)

    assert outflow.rate == pytest.approx(500.0)
    assert sp.rate == pytest.approx(0.0)

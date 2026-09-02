import pytest
from drs_mining.components import StochasticReserve, StochasticFaciesGenerator


def test_stochastic_reserve_extraction():
    gen = StochasticFaciesGenerator(
        mean_fraction=0.30,
        std_dev=0.0,
        attribute_name="ore2_fraction",
    )
    reserve = StochasticReserve(
        name="test_reserve",
        total_tonnes=100_000.0,
        generator=gen,
        min_parcel_mass=20_000.0,
        max_parcel_mass=20_000.0,
        initial_parcel_mass=20_000.0,
    )

    assert reserve.remaining_reserve == 100_000.0
    assert not reserve.is_exhausted
    assert reserve.current_attributes["ore2_fraction"] == pytest.approx(0.30)

    # Extract at 5000 t/day
    flow = reserve.extract(rate=5000.0)
    assert flow.rate == 5000.0
    assert flow.attributes["ore2_fraction"] == pytest.approx(0.30)

    # Step by 2 days (10,000 tonnes extracted)
    reserve.step(2.0)
    assert reserve.cumulative_extracted_mass.value == pytest.approx(10_000.0)
    assert reserve.remaining_reserve == pytest.approx(90_000.0)
    assert not reserve.is_exhausted


def test_stochastic_reserve_exhaustion():
    gen = StochasticFaciesGenerator(mean_fraction=0.50, std_dev=0.0)
    reserve = StochasticReserve(
        name="small_reserve",
        total_tonnes=1000.0,
        generator=gen,
        min_parcel_mass=500.0,
        max_parcel_mass=500.0,
        initial_parcel_mass=500.0,
    )

    reserve.extract(1000.0)
    reserve.step(1.0)

    assert reserve.is_exhausted
    assert reserve.is_terminating_condition_met()

    # Further extraction should yield 0 flow
    flow = reserve.extract(500.0)
    assert flow.rate == 0.0

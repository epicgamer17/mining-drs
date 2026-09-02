import pytest
from drs_mining.components import MaterialSource


def test_material_source_extraction():
    source = MaterialSource(
        name="test_source",
        total_tonnes=100_000.0,
        mean_attributes={"ore2_fraction": 0.30},
        attribute_std_dev=0.0,
        min_parcel_mass=20_000.0,
        max_parcel_mass=20_000.0,
        initial_parcel_mass=20_000.0,
    )

    assert source.remaining_reserve == 100_000.0
    assert not source.is_exhausted
    assert source.current_attributes["ore2_fraction"] == pytest.approx(0.30)

    # Extract at 5000 t/day
    flow = source.extract(rate=5000.0)
    assert flow.rate == 5000.0
    assert flow.attributes["ore2_fraction"] == pytest.approx(0.30)

    # Step by 2 days (10,000 tonnes extracted)
    source.step(2.0)
    assert source.cumulative_extracted_mass.value == pytest.approx(10_000.0)
    assert source.remaining_reserve == pytest.approx(90_000.0)
    assert not source.is_exhausted


def test_material_source_exhaustion():
    source = MaterialSource(
        name="small_source",
        total_tonnes=1000.0,
        mean_attributes={"ore2_fraction": 0.50},
        attribute_std_dev=0.0,
        min_parcel_mass=500.0,
        max_parcel_mass=500.0,
        initial_parcel_mass=500.0,
    )

    source.extract(1000.0)
    source.step(1.0)

    assert source.is_exhausted
    assert source.is_terminating_condition_met()

    # Further extraction yields 0 flow
    flow = source.extract(500.0)
    assert flow.rate == 0.0

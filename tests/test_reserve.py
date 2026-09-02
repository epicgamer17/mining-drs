import pytest
from drs import Entity
from drs_mining.components import MaterialSource, autocorrelated_generator


def test_material_source_with_custom_stream():
    # Stream of discrete blocks / parcels
    blocks = [
        Entity(mass=5000.0, attributes={"Cu": 1.2, "Au": 0.5}),
        Entity(mass=5000.0, attributes={"Cu": 0.8, "Au": 0.3}),
    ]

    source = MaterialSource(
        name="stope_source",
        total_tonnes=10000.0,
        stream=blocks,
    )

    assert source.remaining_reserve == 10000.0
    assert not source.is_exhausted
    assert source.current_attributes["Cu"] == pytest.approx(1.2)

    # Extract first block at 5000 t/day over 1 day
    flow1 = source.extract(5000.0)
    assert flow1.rate == 5000.0
    assert flow1.attributes["Cu"] == pytest.approx(1.2)

    source.step(1.0)
    assert source.remaining_reserve == pytest.approx(5000.0)

    # Automatically transitions to second block
    flow2 = source.extract(5000.0)
    assert flow2.attributes["Cu"] == pytest.approx(0.8)

    source.step(1.0)
    assert source.is_exhausted
    assert source.is_terminating_condition_met()

    # Further extraction yields 0
    flow3 = source.extract(1000.0)
    assert flow3.rate == 0.0


def test_material_source_with_autocorrelated_generator():
    stream = autocorrelated_generator(
        mean_fraction=0.30,
        std_dev=0.0,
        min_mass=20000.0,
        max_mass=20000.0,
        initial_mass=20000.0,
        attribute_name="ore2_fraction",
    )

    source = MaterialSource(
        name="test_source",
        total_tonnes=100_000.0,
        stream=stream,
    )

    assert source.remaining_reserve == 100_000.0
    assert not source.is_exhausted
    assert source.current_attributes["ore2_fraction"] == pytest.approx(0.30)

    flow = source.extract(5000.0)
    assert flow.rate == 5000.0
    assert flow.attributes["ore2_fraction"] == pytest.approx(0.30)

    source.step(2.0)
    assert source.cumulative_extracted_mass.value == pytest.approx(10_000.0)
    assert source.remaining_reserve == pytest.approx(90_000.0)

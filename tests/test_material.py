import pytest
from drs_mining.components import Flow, Entity, blend_flows, split_flow


def test_flow_creation_and_attributes():
    flow = Flow(rate=100.0, attributes={"Cu": 0.65, "Au": 1.2})
    assert flow.rate == 100.0
    assert flow.attributes["Cu"] == 0.65
    assert flow.attributes["Au"] == 1.2

    with pytest.raises(AttributeError):
        flow.rate = 200.0  # Frozen dataclass


def test_entity_creation():
    entity = Entity(mass=500.0, attributes={"ore2_fraction": 0.35})
    assert entity.mass == 500.0
    assert entity.attributes["ore2_fraction"] == 0.35


def test_blend_flows():
    f1 = Flow(rate=100.0, attributes={"Cu": 1.0, "Au": 2.0})
    f2 = Flow(rate=100.0, attributes={"Cu": 2.0, "Au": 4.0})
    blended = blend_flows([f1, f2])

    assert blended.rate == 200.0
    # Mass-weighted average: (100*1 + 100*2)/200 = 1.5
    assert blended.attributes["Cu"] == pytest.approx(1.5)
    # Mass-weighted average: (100*2 + 100*4)/200 = 3.0
    assert blended.attributes["Au"] == pytest.approx(3.0)


def test_blend_flows_zero_rate():
    f1 = Flow(rate=0.0, attributes={"Cu": 1.0})
    f2 = Flow(rate=0.0, attributes={"Cu": 2.0})
    blended = blend_flows([f1, f2])
    assert blended.rate == 0.0


def test_split_flow():
    source = Flow(rate=1000.0, attributes={"Cu": 0.8, "moisture": 0.05})
    splits = split_flow(source, {"stream_a": 0.7, "stream_b": 0.3})

    assert len(splits) == 2
    assert splits["stream_a"].rate == pytest.approx(700.0)
    assert splits["stream_b"].rate == pytest.approx(300.0)
    assert splits["stream_a"].attributes["Cu"] == 0.8
    assert splits["stream_b"].attributes["Cu"] == 0.8

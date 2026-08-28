"""Unit tests for RoadSegment topology component."""

import pytest
from drs_mining.components.topology import RoadSegment
from drs_mining.components.fleet import Truck


def test_road_segment_occupy_and_step():
    seg = RoadSegment("ramp_01", length_m=300.0, segment_type="ramp")
    assert seg.is_available() is True
    assert seg.occupying_truck is None

    speeds = {"ramp": {"empty": 15.0, "loaded": 10.0}}
    truck = Truck(
        truck_id="T01",
        truck_type="AD30",
        ore_payload_cap=30.0,
        waste_payload_cap=30.0,
        speeds=speeds,
    )

    # Empty truck: speed = 15 km/h = 4.1667 m/s -> time = 300 / 4.1667 = 72 s
    dur = seg.occupy_segment(truck)
    assert pytest.approx(dur) == 72.0
    assert seg.is_available() is False
    assert seg.occupying_truck == truck

    # Step 50 seconds
    seg.update_continuous_step(50.0)
    assert seg.is_available() is False

    # Step remaining 25 seconds
    seg.update_continuous_step(25.0)
    assert seg.is_available() is True
    assert seg.occupying_truck is None

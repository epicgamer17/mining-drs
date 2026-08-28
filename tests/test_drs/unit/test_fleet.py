"""Unit tests for Truck and LHD fleet component classes."""

import pytest
from drs_mining.components.fleet import Truck, LHD, TruckState, ContinuousFleetLogistics


def test_truck_dataclass_and_speeds():
    speeds = {
        "ramp": {"empty": 12.0, "loaded": 8.0},
        "level": {"empty": 10.0, "loaded": 6.0},
    }
    truck = Truck(
        truck_id="T01",
        truck_type="AD30",
        ore_payload_cap=30.0,
        waste_payload_cap=30.0,
        speeds=speeds,
    )
    assert truck.truck_id == "T01"
    assert truck.state == TruckState.PARKED

    # Speed empty
    mps_empty = truck.get_speed_mps("ramp")
    assert pytest.approx(mps_empty) == 12.0 / 3.6

    # Speed loaded
    truck.current_payload = 25.0
    mps_loaded = truck.get_speed_mps("ramp")
    assert pytest.approx(mps_loaded) == 8.0 / 3.6


def test_lhd_dataclass():
    lhd = LHD(
        lhd_id="L01",
        level_index=3,
        bucket_ore_cap=10.0,
        bucket_waste_cap=10.0,
        load_spot_min=0.5,
        load_min=2.0,
        dump_min=0.5,
        tram_dist_m=50.0,
        speed_loaded_kph=10.0,
        speed_empty_kph=12.0,
    )
    assert lhd.lhd_id == "L01"
    assert lhd.level_index == 3

import pytest
from drs_mining.components import HaulRoute


def test_haul_route_cycle_time_and_congestion():
    route = HaulRoute(
        distance_km=2.0,
        base_cycle_time_min=20.0,
        congestion_factor=0.05,
        truck_payload_tonnes=100.0,
    )
    # 1 truck: no congestion penalty
    assert route.cycle_time(1) == 20.0

    # 5 trucks: 1 + 4*0.05 = 1.20x base cycle time = 24.0 min
    assert route.cycle_time(5) == 24.0

    # 1 truck haulage in 24 hours: (24*60 / 20) * 100 = 7200 t
    assert route.max_daily_haulage(1) == 7200.0

    # 5 trucks haulage in 24 hours: 5 * (24*60 / 24) * 100 = 30000 t
    assert route.max_daily_haulage(5) == 30000.0

    # 0 trucks
    assert route.max_daily_haulage(0) == 0.0

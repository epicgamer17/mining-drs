import pytest
from drs_mining.components import truck_haul_capacity, truck_cycle_time_breakdown


def test_truck_haul_capacity_calculations():
    # 1. Direct variable rate: fixed = 4.0 min, variable = 16.0 min/km.
    # At distance = 1.0 km, cycle time = 4.0 + 16.0 = 20.0 min.
    # 1 truck: 100t, 20 min cycle time, 100% availability -> (24*60 / 20) * 100 = 7200 t/day
    cap_1 = truck_haul_capacity(
        distance_km=1.0,
        num_trucks=1,
        truck_payload_tonnes=100.0,
        fixed_cycle_time_min=4.0,
        variable_time_per_km_min=16.0,
        mechanical_availability=1.0,
        operator_efficiency=1.0,
        congestion_factor=0.05,
    )
    assert cap_1 == pytest.approx(7200.0)

    # 5 trucks: 1 + 4*0.05 = 1.2x cycle time = 24 min -> 5 * (1440/24) * 100 = 30000 t/day
    cap_5 = truck_haul_capacity(
        distance_km=1.0,
        num_trucks=5,
        truck_payload_tonnes=100.0,
        fixed_cycle_time_min=4.0,
        variable_time_per_km_min=16.0,
        mechanical_availability=1.0,
        operator_efficiency=1.0,
        congestion_factor=0.05,
    )
    assert cap_5 == pytest.approx(30000.0)

    # Realistic availability (85%) and efficiency (90%): 30000 * 0.85 * 0.90 = 22950 t/day
    cap_realistic = truck_haul_capacity(
        distance_km=1.0,
        num_trucks=5,
        truck_payload_tonnes=100.0,
        fixed_cycle_time_min=4.0,
        variable_time_per_km_min=16.0,
        mechanical_availability=0.85,
        operator_efficiency=0.90,
        congestion_factor=0.05,
    )
    assert cap_realistic == pytest.approx(22950.0)

    # 0 trucks returns 0.0
    assert (
        truck_haul_capacity(
            distance_km=1.0,
            num_trucks=0,
            truck_payload_tonnes=100.0,
            mechanical_availability=0.85,
            operator_efficiency=0.90,
            fixed_cycle_time_min=4.0,
            variable_time_per_km_min=16.0,
        )
        == 0.0
    )

    # 0 distance returns 0.0
    assert (
        truck_haul_capacity(
            distance_km=0.0,
            num_trucks=5,
            truck_payload_tonnes=100.0,
            mechanical_availability=0.85,
            operator_efficiency=0.90,
            fixed_cycle_time_min=4.0,
            variable_time_per_km_min=16.0,
        )
        == 0.0
    )


def test_truck_haul_capacity_speeds_and_breakdown():
    """Verifies SME Handbook physics: fixed spot/load/dump time plus speed-based travel."""
    # Haul distance = 2.0 km
    # Haul speed = 25.0 km/h -> travel time = 2.0 * (60 / 25) = 4.8 min
    # Return speed = 40.0 km/h -> travel time = 2.0 * (60 / 40) = 3.0 min
    # Total variable travel = 7.8 min
    # Fixed time = 4.2 min
    # Total cycle time = 4.2 + 7.8 = 12.0 min
    # 1 truck of 150t, 100% avail/eff -> 1440 / 12 = 120 trips -> 120 * 150 = 18,000 t/day
    cap = truck_haul_capacity(
        distance_km=2.0,
        num_trucks=1,
        truck_payload_tonnes=150.0,
        fixed_cycle_time_min=4.2,
        haul_speed_kmh=25.0,
        return_speed_kmh=40.0,
        mechanical_availability=1.0,
        operator_efficiency=1.0,
    )
    assert cap == pytest.approx(18000.0)

    # Detailed breakdown
    b = truck_cycle_time_breakdown(
        distance_km=2.0,
        num_trucks=1,
        truck_payload_tonnes=150.0,
        fixed_cycle_time_min=4.2,
        haul_speed_kmh=25.0,
        return_speed_kmh=40.0,
        mechanical_availability=1.0,
        operator_efficiency=1.0,
    )
    assert b["fixed_time_min"] == pytest.approx(4.2)
    assert b["haul_travel_min"] == pytest.approx(4.8)
    assert b["return_travel_min"] == pytest.approx(3.0)
    assert b["travel_time_min"] == pytest.approx(7.8)
    assert b["total_cycle_time_min"] == pytest.approx(12.0)
    assert b["trips_per_truck_day"] == pytest.approx(120.0)
    assert b["fleet_daily_tonnes"] == pytest.approx(18000.0)

    # Physics check: Fixed time does NOT scale with distance!
    # At 0.2 km: fixed time remains 4.2 min!
    b_short = truck_cycle_time_breakdown(
        distance_km=0.2,
        num_trucks=1,
        truck_payload_tonnes=150.0,
        fixed_cycle_time_min=4.2,
        haul_speed_kmh=25.0,
        return_speed_kmh=40.0,
        mechanical_availability=1.0,
        operator_efficiency=1.0,
    )
    assert b_short["fixed_time_min"] == pytest.approx(4.2)
    assert b_short["travel_time_min"] == pytest.approx(0.78)  # 1/10th of 7.8


def test_truck_haulage_validation():
    """Verifies input validation and error raising."""
    # Negative fixed time
    with pytest.raises(ValueError, match="fixed_cycle_time_min must be non-negative"):
        truck_haul_capacity(
            distance_km=1.0,
            num_trucks=1,
            truck_payload_tonnes=100.0,
            fixed_cycle_time_min=-1.0,
            variable_time_per_km_min=10.0,
            mechanical_availability=0.9,
            operator_efficiency=0.9,
        )

    # Non-positive speeds
    with pytest.raises(ValueError, match="Haul and return speeds must be strictly positive"):
        truck_haul_capacity(
            distance_km=1.0,
            num_trucks=1,
            truck_payload_tonnes=100.0,
            fixed_cycle_time_min=4.0,
            haul_speed_kmh=0.0,
            return_speed_kmh=40.0,
            mechanical_availability=0.9,
            operator_efficiency=0.9,
        )

    # Missing variable travel parameters raises ValueError
    with pytest.raises(ValueError, match="Must provide either variable_time_per_km_min"):
        truck_haul_capacity(
            distance_km=1.0,
            num_trucks=1,
            truck_payload_tonnes=100.0,
            fixed_cycle_time_min=4.0,
            mechanical_availability=0.9,
            operator_efficiency=0.9,
        )

    # Missing required fixed_cycle_time_min raises TypeError
    with pytest.raises(TypeError):
        truck_haul_capacity(  # type: ignore[call-arg]
            distance_km=1.0,
            num_trucks=1,
            truck_payload_tonnes=100.0,
            mechanical_availability=0.9,
            operator_efficiency=0.9,
            variable_time_per_km_min=10.0,
        )

    # Missing required availability/efficiency raises TypeError
    with pytest.raises(TypeError):
        truck_haul_capacity(  # type: ignore[call-arg]
            distance_km=1.0,
            num_trucks=1,
            truck_payload_tonnes=100.0,
            fixed_cycle_time_min=4.0,
            variable_time_per_km_min=10.0,
        )

import pytest
from drs_mining.components import truck_haul_capacity


def test_truck_haul_capacity_calculations():
    # 1 truck: 100t, 20 min cycle time, 100% availability -> (24*60 / 20) * 100 = 7200 t/day
    cap_1 = truck_haul_capacity(
        distance_km=1.0,
        num_trucks=1,
        truck_payload_tonnes=100.0,
        base_cycle_time_min=20.0,
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
        base_cycle_time_min=20.0,
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
        base_cycle_time_min=20.0,
        mechanical_availability=0.85,
        operator_efficiency=0.90,
        congestion_factor=0.05,
    )
    assert cap_realistic == pytest.approx(22950.0)

    # 0 trucks
    assert (
        truck_haul_capacity(
            distance_km=1.0,
            num_trucks=0,
            truck_payload_tonnes=100.0,
            base_cycle_time_min=20.0,
            mechanical_availability=0.85,
            operator_efficiency=0.90,
        )
        == 0.0
    )

    # Verify missing required heuristics raises TypeError
    with pytest.raises(TypeError):
        truck_haul_capacity(1.0, 1, 100.0, 20.0)  # Missing availability and efficiency

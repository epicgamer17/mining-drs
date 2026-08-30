import pytest
import math
import drs

from drs_mining.components.fleet import Truck, TruckPhase, LHD
from drs_mining.components.topology import RoadSegment
from drs_mining.components.bays import LoadingBay, DumpingBay
from drs_mining.components.dispatch import ShelswellDispatchController


def test_truck_discrete_state():
    speeds = {"surface": {"empty": 17.4, "loaded": 13.4}}
    truck = Truck(
        truck_id="T01",
        truck_type="AD30",
        ore_payload_cap=26.1,
        waste_payload_cap=24.6,
        speeds=speeds,
    )
    assert truck.truck_id == "T01"
    assert truck.phase == TruckPhase.PARKED
    assert truck.ore_payload_cap == 26.1
    assert truck.waste_payload_cap == 24.6

    # Test corridor speed profile calculations
    v_surface_empty = truck.get_speed_mps("surface")
    assert math.isclose(v_surface_empty, 17.4 / 3.6, rel_tol=1e-5)

    truck.current_payload = 26.1
    v_surface_loaded = truck.get_speed_mps("surface")
    assert math.isclose(v_surface_loaded, 13.4 / 3.6, rel_tol=1e-5)


def test_road_availability_timer():
    road = RoadSegment("test_segment", length_m=100.0, segment_type="decline")

    assert road.is_available()
    truck = Truck(
        truck_id="T01",
        truck_type="AD30",
        ore_payload_cap=26.1,
        waste_payload_cap=24.6,
        speeds={"decline": {"empty": 15.1, "loaded": 11.2}},
    )

    travel_time_s = road.occupy_segment(truck)
    assert travel_time_s > 0.0
    assert not road.is_available()

    # Update step decay
    road.update_continuous_step(travel_time_s + 1.0)
    assert road.is_available()


def test_loading_and_dumping_bays():
    lhd = LHD(
        lhd_id="LHD_L1",
        level_index=1,
        bucket_ore_cap=14.0,
        bucket_waste_cap=12.5,
        load_spot_min=0.46,
        load_min=0.88,
        dump_min=0.73,
        tram_dist_m=35.0,
        speed_loaded_kph=5.89,
        speed_empty_kph=6.78,
    )
    bay = LoadingBay(
        bay_id="L1_ORE",
        bay_type="ORE",
        level_index=1,
        initial_muck=1000.0,
        truck_spot_min=0.82,
        acquisition_delay_min=1.5,
        bucket_passes=2.0,
        lhd=lhd,
    )
    truck = Truck(
        truck_id="T01",
        truck_type="AD30",
        ore_payload_cap=26.1,
        waste_payload_cap=24.6,
        speeds={"surface": {"empty": 17.4, "loaded": 13.4}},
    )

    started = bay.start_loading(truck)
    assert started
    assert truck.phase == TruckPhase.LOADING
    assert bay.load_rate.value > 0.0

    # Step simulation forward
    bay.update_continuous_step(bay.total_load_duration_sec + 1.0)
    assert truck.phase == TruckPhase.LOADED
    assert truck.current_payload > 0.0
    assert bay.active_truck is None

    # Test dumping bay
    dump_bay = DumpingBay(
        bay_id="ROM_PAD",
        bay_type="ORE",
        location_name="SURFACE_ROM",
        dump_spot_min=0.57,
        bed_raise_dump_min=0.88,
    )

    dump_started = dump_bay.start_dumping(truck)
    assert dump_started
    assert truck.phase == TruckPhase.DUMPING

    dump_bay.update_continuous_step(100.0)
    assert dump_bay.dumped_total.value > 0.0
    assert truck.phase == TruckPhase.EMPTY


def test_dispatch_controller():
    truck = Truck(
        truck_id="T01",
        truck_type="AD30",
        ore_payload_cap=26.1,
        waste_payload_cap=24.6,
        speeds={"surface": {"empty": 17.4, "loaded": 13.4}},
    )
    lhd1 = LHD("LHD_L1", 1, 14.0, 12.5, 0.46, 0.88, 0.73, 35.0, 5.89, 6.78)
    lhd2 = LHD("LHD_L2", 2, 14.0, 12.5, 0.46, 0.88, 0.73, 35.0, 5.89, 6.78)
    bay1 = LoadingBay("L1_ORE", "ORE", 1, 500.0, 0.82, 1.5, 2.0, lhd1)
    bay2 = LoadingBay("L2_ORE", "ORE", 2, 2000.0, 0.82, 1.5, 2.0, lhd2)

    controller = ShelswellDispatchController(
        trucks=[truck],
        loading_bays=[bay1, bay2],
        roads={},
        waste_trip_interval=0,
        refuel_threshold_pct=15.0,
        fuel_depot_location="SURFACE_FUEL_DEPOT",
        parking_location="SURFACE_PARKING",
        dispatch_strategy="highest_muck",
    )

    # Normal dispatch: should select bay2 with highest unclaimed muck
    controller.assign_next_destination(truck)
    assert truck.target_bay_id == "L2_ORE"
    assert truck.phase == TruckPhase.EMPTY

    # Low fuel intercept test
    truck.fuel_level_pct = 10.0
    controller.assign_next_destination(truck)
    assert truck.phase == TruckPhase.REFUELING
    assert truck.current_location == "SURFACE_FUEL_DEPOT"

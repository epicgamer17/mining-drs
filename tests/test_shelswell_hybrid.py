import pytest
import math
import drs

from drs_mining.components.fleet import Truck, TruckState, LHD
from drs_mining.components.topology import DRSRoadSegment
from drs_mining.components.bays import DRSLoadingBay, DRSDumpingBay
from drs_mining.controllers.dispatch import ShelswellDispatchController
from drs_mining.simulation import ShelswellHybridSimulation, HybridDRSModule


def test_truck_discrete_state():
    truck = Truck(truck_id="T01", truck_type="AD30")
    assert truck.truck_id == "T01"
    assert truck.state == TruckState.PARKED
    assert truck.ore_payload_cap == 26.1
    assert truck.waste_payload_cap == 24.6

    # Test corridor speed profile calculations
    v_surface_empty = truck.get_speed_mps("surface")
    assert math.isclose(v_surface_empty, 17.4 / 3.6, rel_tol=1e-5)

    truck.current_payload = 26.1
    v_surface_loaded = truck.get_speed_mps("surface")
    assert math.isclose(v_surface_loaded, 13.4 / 3.6, rel_tol=1e-5)


def test_road_availability_timer():
    module = HybridDRSModule()
    engine = drs.DRSEngine(module)
    road = DRSRoadSegment(engine, "test_segment", length_m=100.0, segment_type="decline")

    assert road.is_available()
    truck = Truck(truck_id="T01")

    travel_time_s = road.occupy_segment(truck)
    assert travel_time_s > 0.0
    assert not road.is_available()

    # Update step decay
    road.update_continuous_step(travel_time_s + 1.0)
    assert road.is_available()


def test_loading_and_dumping_bays():
    module = HybridDRSModule()
    engine = drs.DRSEngine(module)
    bay = DRSLoadingBay(engine, "L1_ORE", "ORE", 1, initial_muck=1000.0)
    truck = Truck(truck_id="T01")

    started = bay.start_loading(truck)
    assert started
    assert truck.state == TruckState.LOADING
    assert bay.load_rate.value > 0.0

    # Step simulation forward
    bay.update_continuous_step(bay.total_load_duration_sec + 1.0)
    assert truck.state == TruckState.TRAVEL_LOADED
    assert truck.current_payload > 0.0
    assert bay.active_truck is None

    # Test dumping bay
    dump_bay = DRSDumpingBay(engine, "ROM_PAD", "ORE", "SURFACE_ROM")
    dump_started = dump_bay.start_dumping(truck)
    assert dump_started
    assert truck.state == TruckState.DUMPING

    dump_bay.update_continuous_step(100.0)
    assert dump_bay.dumped_total.value > 0.0
    assert truck.state == TruckState.PARKED


def test_dispatch_controller():
    truck = Truck(truck_id="T01")
    bay1 = DRSLoadingBay(None, "L1_ORE", "ORE", 1, initial_muck=500.0)
    bay2 = DRSLoadingBay(None, "L2_ORE", "ORE", 2, initial_muck=2000.0)

    controller = ShelswellDispatchController([truck], [bay1, bay2])

    # Normal dispatch: should select bay2 with highest unclaimed muck
    controller.assign_next_destination(truck)
    assert truck.target_bay_id == "L2_ORE"
    assert truck.state == TruckState.TRAVEL_EMPTY

    # Low fuel intercept test
    truck.fuel_level_pct = 10.0
    controller.assign_next_destination(truck)
    assert truck.state == TruckState.REFUELING
    assert truck.current_location == "SURFACE_FUEL_DEPOT"


def test_hybrid_simulation_step():
    sim = ShelswellHybridSimulation(num_trucks=4, num_operators=4, mechanical_availability=1.0)
    assert len(sim.trucks) == 4

    # Run short 1-day simulation
    prod = sim.run_simulation(total_days=1.0)
    assert prod >= 0.0

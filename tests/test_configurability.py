import math
import pytest
import drs

from drs_mining import (
    Truck,
    LHD,
    create_truck_fleet,
    create_lhd_fleet,
    Stockpile,
    create_stockpiles,
    DRSLoadingBay,
    DRSDumpingBay,
    ShelswellDispatchController,
    build_multi_face_simulation,
    build_concentrator_simulation,
    ContinuousMineFace,
    ConcentratorPlant,
)
from drs_mining.components.fleet import ContinuousFleetLogistics, TruckState
from drs_mining.components.controllers import MultiFaceConcentratorController
from drs_mining.components.generators import StochasticFaciesGenerator


def test_create_truck_and_lhd_fleets():
    trucks = create_truck_fleet(5, prefix="HAUL", ore_payload_cap=35.0, waste_payload_cap=30.0)
    assert len(trucks) == 5
    assert trucks[0].truck_id == "HAUL01"
    assert trucks[4].truck_id == "HAUL05"
    assert trucks[0].ore_payload_cap == 35.0

    lhds = create_lhd_fleet(levels=[1, 2, 3], count_per_level=2, bucket_ore_cap=16.0)
    assert len(lhds) == 6
    assert lhds[0].lhd_id == "LHD_L1_1"
    assert lhds[1].lhd_id == "LHD_L1_2"
    assert lhds[0].bucket_ore_cap == 16.0


def test_create_stockpiles_factory():
    configs = [
        {"name": "Ore1", "initial_mass": 1000.0, "attr_inflow": 1.0},
        {"name": "Ore2", "initial_mass": 2000.0, "attr_inflow": 0.0},
        {"name": "Waste", "initial_mass": 500.0, "attr_inflow": 0.0},
    ]
    stocks = create_stockpiles(configs)
    assert len(stocks) == 3
    assert stocks[0].name == "Ore1"
    assert stocks[0].level == 1000.0
    assert stocks[2].name == "Waste"


def test_configurable_loading_and_dumping_bays():
    engine = drs.DRSEngine()
    lhd = LHD("LHD1", 1, tram_dist_m=20.0)
    bay = DRSLoadingBay(
        engine,
        "L1_ORE",
        "ORE",
        1,
        initial_muck=5000.0,
        lhd=lhd,
        truck_spot_min=0.5,
        acquisition_delay_min=1.0,
        bucket_passes=3.0,
    )
    truck = Truck("T01", ore_payload_cap=30.0)
    assert bay.truck_spot_min == 0.5
    assert bay.bucket_passes == 3.0

    duration = bay.calculate_load_duration_sec(truck)
    assert duration > 0.0

    bay.start_loading(truck)
    assert truck.state == TruckState.LOADING
    assert bay.total_load_duration_sec == duration

    dump_bay = DRSDumpingBay(
        engine,
        "ROM",
        "ORE",
        "SURFACE_ROM",
        dump_spot_min=0.4,
        bed_raise_dump_min=0.6,
    )
    assert dump_bay.dump_spot_min == 0.4
    assert math.isclose(dump_bay.calculate_dump_duration_sec(truck), 60.0)


def test_configurable_dispatch_controller():
    trucks = create_truck_fleet(3)
    bay1 = DRSLoadingBay(None, "L1_ORE", "ORE", 1, initial_muck=100.0)
    bay2 = DRSLoadingBay(None, "L2_ORE", "ORE", 2, initial_muck=500.0)

    controller = ShelswellDispatchController(
        trucks=trucks,
        loading_bays=[bay1, bay2],
        waste_trip_interval=5,
        refuel_threshold_pct=20.0,
        fuel_depot_location="CUSTOM_DEPOT",
        parking_location="CUSTOM_PARKING",
    )

    # Test custom refuel threshold
    trucks[0].fuel_level_pct = 19.0
    controller.assign_next_destination(trucks[0])
    assert trucks[0].state == TruckState.REFUELING
    assert trucks[0].current_location == "CUSTOM_DEPOT"

    # Test round robin strategy
    rr_ctrl = ShelswellDispatchController(
        trucks=trucks,
        loading_bays=[bay1, bay2],
        dispatch_strategy="round_robin",
    )
    trucks[1].fuel_level_pct = 100.0
    rr_ctrl.assign_next_destination(trucks[1])
    assert trucks[1].state == TruckState.TRAVEL_EMPTY


def test_multi_stockpile_fleet_routing():
    fleet = ContinuousFleetLogistics(num_stockpiles=3)
    gen1 = StochasticFaciesGenerator(mean_fraction=0.2, std_dev=0.05)
    face = ContinuousMineFace(face_id=1, generator=gen1)
    face.target_rate = 1000.0
    face.step(1.0)


    # Custom 3-way split function: 50% stock 1, 30% stock 2, 20% stock 3
    inflows = fleet.route_multi(
        sources=[face],
        split_fn=lambda r, g: [0.5 * r, 0.3 * r, 0.2 * r],
    )
    assert len(inflows) == 3
    assert math.isclose(inflows[0], 500.0)
    assert math.isclose(inflows[1], 300.0)
    assert math.isclose(inflows[2], 200.0)


def test_multi_face_simulation_arbitrary_n():
    for num_faces in [1, 2, 3, 4, 5]:
        faces, fleet, plant, controller, ore1_stock, ore2_stock = build_multi_face_simulation(
            num_faces=num_faces,
            total_truck_count=15.0,
            total_lhd_count=5.0,
            face_haul_distance=1.8,  # scalar broadcast
            face_accessibility_fraction=0.95,  # scalar broadcast
            max_trucks_per_face=5.0,
            max_lhds_per_face=2.0,
        )
        assert len(faces) == num_faces
        assert controller.total_truck_count == 15.0
        assert controller.total_lhd_count == 5.0

        # Check allocations exist for all standard modes
        for mode in ["MODE_A", "MODE_B", "MODE_A_CONTINGENCY", "MODE_B_CONTINGENCY", "MODE_A_MINE_SURGING", "MODE_B_MINE_SURGING"]:
            alloc = controller._get_allocations_for_mode(mode)
            assert len(alloc) == num_faces
            assert math.isclose(sum(alloc), 1.0, rel_tol=1e-5)


def test_multi_face_simulation_execution_3_faces():
    faces, fleet, plant, controller, ore1_stock, ore2_stock = build_multi_face_simulation(
        num_faces=3,
        face_mean_fractions=[0.10, 0.30, 0.50],
        total_truck_count=12.0,
        total_lhd_count=4.0,
    )

    engine = drs.DRSEngine()
    engine.register(*faces, fleet, plant, controller, ore1_stock, ore2_stock)

    @engine.on_step
    def manage_step(t):
        mode = controller.update_mode(ore1_stock, ore2_stock)
        mine_target, s1_target, s2_target = controller.get_target_rates(mode, fleet)
        controller.schedule_fleet_shifts(mode)
        controller.drive_faces(mine_target)
        ore1_in, ore2_in = fleet.route(sources=faces)
        out1 = ore1_stock.feed_and_draw(ore1_in, s1_target)
        out2 = ore2_stock.feed_and_draw(ore2_in, s2_target)
        plant.process(out1 + out2)

    engine.run(until=2.0)
    assert engine.current_time >= 2.0
    assert sum(f.cumulative_extracted_mass.value for f in faces) > 0.0


def test_plant_with_multiple_stockpiles():
    stocks = create_stockpiles([
        {"name": "S1", "initial_mass": 500.0},
        {"name": "S2", "initial_mass": 500.0},
        {"name": "S3", "initial_mass": 500.0},
    ])
    plant = ConcentratorPlant(stockpiles=stocks, max_rate=1200.0)
    assert len(plant.stockpiles) == 3
    assert plant.ore1_stock is stocks[0]
    assert plant.ore2_stock is stocks[1]

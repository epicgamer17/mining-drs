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
    LoadingBay,
    DumpingBay,
    ShelswellDispatchController,
    build_mining_simulation,
    MineFace,
    MetallurgicalPlant,
    BlendingController,
    ContinuousFleetLogistics,
    StochasticFaciesGenerator,
    TruckState,
)


def test_create_truck_and_lhd_fleets():
    trucks = create_truck_fleet(
        count=5,
        truck_type="CAT AD30",
        ore_payload_cap=35.0,
        waste_payload_cap=30.0,
        speeds={"surface": {"empty": 17.4, "loaded": 13.4}},
        prefix="HAUL",
    )
    assert len(trucks) == 5
    assert trucks[0].truck_id == "HAUL01"
    assert trucks[4].truck_id == "HAUL05"
    assert trucks[0].ore_payload_cap == 35.0

    lhds = create_lhd_fleet(
        levels=[1, 2, 3],
        bucket_ore_cap=16.0,
        bucket_waste_cap=14.0,
        load_spot_min=0.4,
        load_min=0.8,
        dump_min=0.7,
        tram_dist_m=30.0,
        speed_loaded_kph=6.0,
        speed_empty_kph=7.0,
        count_per_level=2,
    )
    assert len(lhds) == 6
    assert lhds[0].lhd_id == "LHD_L1_1"
    assert lhds[1].lhd_id == "LHD_L1_2"
    assert lhds[0].bucket_ore_cap == 16.0


def test_create_stockpiles_factory():
    configs = [
        {
            "name": "ROM_A",
            "expected_attributes": ["cu_grade"],
            "initial_mass": 1000.0,
            "initial_attributes": {"cu_grade": 25.0},
        },
        {
            "name": "ROM_B",
            "expected_attributes": ["cu_grade"],
            "initial_mass": 2000.0,
            "initial_attributes": {"cu_grade": 40.0},
        },
    ]
    stocks = create_stockpiles(configs)
    assert len(stocks) == 2
    assert stocks[0].name == "ROM_A"
    assert stocks[0].current_mass.value == 1000.0
    assert stocks[1].name == "ROM_B"
    assert stocks[1].current_mass.value == 2000.0


def test_configurable_loading_and_dumping_bays():
    lhd = LHD(
        lhd_id="LHD1",
        level_index=1,
        bucket_ore_cap=14.0,
        bucket_waste_cap=12.5,
        load_spot_min=0.4,
        load_min=0.8,
        dump_min=0.7,
        tram_dist_m=30.0,
        speed_loaded_kph=6.0,
        speed_empty_kph=7.0,
    )
    bay = LoadingBay(
        bay_id="L1_ORE",
        bay_type="ORE",
        level_index=1,
        initial_muck=5000.0,
        truck_spot_min=0.5,
        acquisition_delay_min=1.0,
        bucket_passes=3.0,
        lhd=lhd,
    )
    truck = Truck(
        truck_id="T01",
        truck_type="AD30",
        ore_payload_cap=30.0,
        waste_payload_cap=25.0,
        speeds={"surface": {"empty": 17.4, "loaded": 13.4}},
    )
    assert bay.truck_spot_min == 0.5
    assert bay.bucket_passes == 3.0

    duration = bay.calculate_load_duration_sec(truck)
    assert duration > 0.0

    bay.start_loading(truck)
    assert truck.state == TruckState.LOADING
    assert bay.total_load_duration_sec == duration

    dump_bay = DumpingBay(
        bay_id="ROM",
        bay_type="ORE",
        location_name="SURFACE_ROM",
        dump_spot_min=0.4,
        bed_raise_dump_min=0.6,
    )
    assert dump_bay.dump_spot_min == 0.4
    assert math.isclose(dump_bay.calculate_dump_duration_sec(truck), 60.0)


def test_configurable_dispatch_controller():
    truck_speeds = {"surface": {"empty": 17.4, "loaded": 13.4}}
    trucks = create_truck_fleet(
        count=3,
        truck_type="AD30",
        ore_payload_cap=26.1,
        waste_payload_cap=24.6,
        speeds=truck_speeds,
    )
    lhd1 = LHD("LHD1", 1, 14.0, 12.5, 0.4, 0.8, 0.7, 30.0, 6.0, 7.0)
    lhd2 = LHD("LHD2", 2, 14.0, 12.5, 0.4, 0.8, 0.7, 30.0, 6.0, 7.0)
    bay1 = LoadingBay(
        bay_id="L1_ORE",
        bay_type="ORE",
        level_index=1,
        initial_muck=100.0,
        truck_spot_min=0.8,
        acquisition_delay_min=1.5,
        bucket_passes=2.0,
        lhd=lhd1,
    )
    bay2 = LoadingBay(
        bay_id="L2_ORE",
        bay_type="ORE",
        level_index=2,
        initial_muck=500.0,
        truck_spot_min=0.8,
        acquisition_delay_min=1.5,
        bucket_passes=2.0,
        lhd=lhd2,
    )

    controller = ShelswellDispatchController(
        trucks=trucks,
        loading_bays=[bay1, bay2],
        roads={},
        waste_trip_interval=5,
        refuel_threshold_pct=20.0,
        fuel_depot_location="CUSTOM_DEPOT",
        parking_location="CUSTOM_PARKING",
        dispatch_strategy="highest_muck",
    )

    trucks[0].fuel_level_pct = 19.0
    controller.assign_next_destination(trucks[0])
    assert trucks[0].state == TruckState.REFUELING
    assert trucks[0].current_location == "CUSTOM_DEPOT"

    rr_ctrl = ShelswellDispatchController(
        trucks=trucks,
        loading_bays=[bay1, bay2],
        roads={},
        waste_trip_interval=0,
        refuel_threshold_pct=15.0,
        fuel_depot_location="SURFACE_FUEL_DEPOT",
        parking_location="SURFACE_PARKING",
        dispatch_strategy="round_robin",
    )
    trucks[1].fuel_level_pct = 100.0
    rr_ctrl.assign_next_destination(trucks[1])
    assert trucks[1].state == TruckState.TRAVEL_EMPTY


def test_multi_stockpile_fleet_routing():
    fleet = ContinuousFleetLogistics(num_stockpiles=3)
    gen1 = StochasticFaciesGenerator(mean_fraction=0.2, std_dev=0.05, prob_new_facies=0.3, variation_same_facies=0.01)
    face = MineFace(
        name="mine_face_1",
        face_id=1,
        generator=gen1,
        min_ore_mass=30000.0,
        max_ore_mass=50000.0,
        total_ore_to_extract=6600000.0,
        ore_to_be_extracted_during_warming_period=600000.0,
        mean_ore_fraction=0.2,
        std_dev_ore_fraction=0.05,
        prob_new_facies=0.3,
        variation_same_facies=0.01,
        initial_parcel_mass=30000.0,
    )
    face.target_rate = 1000.0
    face.step(1.0)

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
        faces, fleet, plant, controller, ore1_stock, ore2_stock = build_mining_simulation(
            num_faces=num_faces,
            total_truck_count=15.0,
            total_lhd_count=5.0,
            face_haul_distance=1.8,
            face_accessibility_fraction=0.95,
            max_trucks_per_face=5.0,
            max_lhds_per_face=2.0,
        )
        assert len(faces) == num_faces
        assert controller.total_truck_count == 15.0
        assert controller.total_lhd_count == 5.0

        for mode in ["MODE_A", "MODE_B", "MODE_A_CONTINGENCY", "MODE_B_CONTINGENCY", "MODE_A_MINE_SURGING", "MODE_B_MINE_SURGING"]:
            alloc = controller._get_allocations_for_mode(mode)
            assert len(alloc) == num_faces
            assert math.isclose(sum(alloc), 1.0, rel_tol=1e-5)


def test_multi_face_simulation_execution_3_faces():
    faces, fleet, plant, controller, ore1_stock, ore2_stock = build_mining_simulation(
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
    plant = MetallurgicalPlant(stockpiles=stocks, max_rate=1200.0)
    assert len(plant.stockpiles) == 3
    assert plant.ore1_stock is stocks[0]
    assert plant.ore2_stock is stocks[1]

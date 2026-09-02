"""Unit tests for decoupled components: HaulRoute, StochasticReserve, MineFace, Stockpile, Plant, and Controller."""

import pytest
import drs
from drs_mining.components import (
    HaulRoute,
    GeologySource,
    StochasticReserve,
    Parcel,
    MineFace,
    Stockpile,
    MetallurgicalPlant,
    MillingSetpoints,
    OperatingModeController,
    StochasticFaciesGenerator,
)
from drs_mining.config import MILL_MODES


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
    # Max daily haulage with 5 trucks (24 hrs * 60 min / 24 min/cycle * 100 t = 6000 t)
    assert route.max_daily_haulage(5) == 6000.0


def test_stochastic_reserve_parcels_and_depletion():
    gen = StochasticFaciesGenerator(
        mean_fraction=0.40,
        std_dev=0.0,
        prob_new_facies=0.0,
        variation_same_facies=0.0,
    )
    reserve = StochasticReserve(
        name="test_reserve",
        total_tonnes=100_000.0,
        generator=gen,
        min_parcel_mass=20_000.0,
        max_parcel_mass=20_000.0,
        initial_parcel_mass=20_000.0,
        seed=1,
    )
    assert not reserve.is_exhausted
    assert reserve.active_parcel is not None
    assert reserve.active_parcel.mass == 20_000.0
    assert reserve.active_parcel.ore2_fraction == 0.40
    assert reserve.remaining_reserve == 100_000.0

    # Extract 20,000 tonnes to deplete current parcel
    reserve.parcel_extracted_mass.value = 20_000.0
    reserve.cumulative_extracted_mass.value = 20_000.0
    assert reserve.is_parcel_exhausted

    # Advance state loads next parcel
    reserve.advance_parcel_state()
    assert not reserve.is_parcel_exhausted
    assert reserve.parcel_extracted_mass.value == 0.0

    # Deplete full reserve
    reserve.cumulative_extracted_mass.value = 100_000.0
    assert reserve.is_exhausted


def test_mine_face_composition():
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.0)
    reserve = StochasticReserve(
        name="face_res",
        total_tonnes=50_000.0,
        generator=gen,
        min_parcel_mass=10_000.0,
        max_parcel_mass=10_000.0,
        initial_parcel_mass=10_000.0,
    )
    haul = HaulRoute(distance_km=1.5)
    face = MineFace(name="face", geology=reserve, haulage=haul, max_rate=500.0)

    assert not face.is_exhausted
    assert not face.is_terminating_condition_met()
    assert face.is_ore_available
    assert face.haulage.distance_km == 1.5

    # Target rate drive
    face.target_rate = 250.0
    assert face.actual_rate == 250.0

    # Step advances levels
    face.step(1.0)
    assert face.geology.cumulative_extracted_mass.rate == 250.0
    assert face.geology.parcel_extracted_mass.rate == 250.0


def test_stockpile_attributes_and_grades():
    stock = Stockpile(
        name="stock",
        expected_attributes=["contained_ore"],
        initial_mass=1000.0,
        initial_attributes={"contained_ore": 300.0},  # 30% concentration
    )
    assert stock.level == 1000.0
    assert stock.attributes["contained_ore"].value == 300.0
    assert pytest.approx(stock.current_concentration("contained_ore")) == 0.30

    # Feed 100 t/s with 0.50 grade, draw 50 t/s
    stock.set_inout(inflow_rate=100.0, outflow_rate=50.0, attr_inflow=0.50)
    assert stock.rate == 50.0
    # Inflow adds 100 * 0.5 = 50, outflow removes 50 * 0.3 = 15 -> rate = 35
    assert stock.attributes["contained_ore"].rate == 35.0


def test_operating_mode_controller():
    ctrl = OperatingModeController(
        duration_of_production_campaigns=30.0,
        duration_of_shutdowns=2.0,
        critical_ore2_level=15_000.0,
    )
    assert ctrl.active_campaign_mode.value == MILL_MODES["MODE_A"]
    assert ctrl.current_target_duration == 30.0

    # Step before threshold -> stays in MODE_A
    ctrl.current_campaign_duration.value = 10.0
    mode = ctrl.update(ore2_stock_level=20_000.0)
    assert mode == MILL_MODES["MODE_A"]

    # Reach 30.0 days -> transitions to SHUTDOWN
    ctrl.current_campaign_duration.value = 30.0
    mode = ctrl.update(ore2_stock_level=20_000.0)
    assert mode == MILL_MODES["SHUTDOWN"]
    assert ctrl.current_target_duration == 2.0
    assert ctrl.current_campaign_duration.value == 0.0

    # Shutdown complete -> switches back to MODE_A (ore2 > critical)
    ctrl.current_campaign_duration.value = 2.0
    mode = ctrl.update(ore2_stock_level=20_000.0)
    assert mode == MILL_MODES["MODE_A"]

    # If ore2 <= critical -> switches to MODE_B
    ctrl.current_campaign_duration.value = 30.0
    ctrl.update(ore2_stock_level=10_000.0)  # to shutdown
    ctrl.current_campaign_duration.value = 2.0
    mode = ctrl.update(ore2_stock_level=10_000.0)  # from shutdown
    assert mode == MILL_MODES["MODE_B"]


def test_metallurgical_plant_setpoints_and_rates():
    s1 = Stockpile("s1", initial_mass=50_000.0)
    s2 = Stockpile("s2", initial_mass=30_000.0)
    setpoints = MillingSetpoints(
        mode_a_ore1=3600.0,
        mode_a_ore2=2400.0,
        mode_b_ore1=4600.0,
        mode_b_ore2=800.0,
    )
    plant = MetallurgicalPlant(
        stockpiles=[s1, s2],
        setpoints=setpoints,
        target_ore_stock_level=60_000.0,
    )

    # In MODE_A with healthy stocks
    r1, r2, target = plant.get_target_rates(
        MILL_MODES["MODE_A"],
        ore1_level=50_000.0,
        ore2_level=30_000.0,
        stockpile2_routing_fraction=0.30,
    )
    assert r1 == 3600.0
    assert r2 == 2400.0
    assert target == 6000.0
    assert plant.cumulative_time_mode_a.rate == 1.0
    assert plant.cumulative_time_mode_b.rate == 0.0

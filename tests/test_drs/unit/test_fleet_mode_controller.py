"""Unit tests for FleetModeController and FleetOperatingMode."""

import pytest
import drs
from drs_mining.components.fleet_controller import (
    FleetOperatingMode,
    FleetModeController,
)
from drs_mining.components.fleet import Truck, MissionType, TruckPhase
from drs_mining.components.mine_face import MineFace, FaceState
from drs_mining.components.generators import StochasticFaciesGenerator


def test_fleet_operating_mode_enum():
    """Verify FleetOperatingMode has exactly PRODUCTION and DEVELOPMENT."""
    assert FleetOperatingMode.PRODUCTION.value == "PRODUCTION"
    assert FleetOperatingMode.DEVELOPMENT.value == "DEVELOPMENT"
    assert len(FleetOperatingMode) == 2


def test_fleet_mode_controller_policy1_always_production():
    """Verify Policy 1 strictly maintains PRODUCTION mode regardless of schedule."""
    fmc = FleetModeController()
    mode = fmc.evaluate_mode(
        policy=1,
        current_day=50.0,
        dev_progress_m=100.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
    )
    assert mode == FleetOperatingMode.PRODUCTION
    assert fmc.mode == FleetOperatingMode.PRODUCTION


def test_fleet_mode_controller_policy2_trajectory_switching():
    """Verify Policy 2 switches to DEVELOPMENT when lagging behind and back to PRODUCTION when on schedule or unlocked."""
    fmc = FleetModeController()

    # Day 50: Expected dev = 4000 * (50/365) = 547.9m. Actual = 200m (lagging!) -> DEVELOPMENT mode
    mode = fmc.evaluate_mode(
        policy=2,
        current_day=50.0,
        dev_progress_m=200.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
    )
    assert mode == FleetOperatingMode.DEVELOPMENT

    # Day 50: Actual = 600m (ahead of schedule!) -> PRODUCTION mode
    mode = fmc.evaluate_mode(
        policy=2,
        current_day=50.0,
        dev_progress_m=600.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
    )
    assert mode == FleetOperatingMode.PRODUCTION

    # Area 2 Unlocked (area2_locked = False) -> PRODUCTION mode
    mode = fmc.evaluate_mode(
        policy=2,
        current_day=50.0,
        dev_progress_m=4000.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=False,
    )
    assert mode == FleetOperatingMode.PRODUCTION


def test_fleet_mode_mission_selection_production_priority():
    """Verify ore production is primary whenever stock < 60kt."""
    fmc = FleetModeController(initial_mode=FleetOperatingMode.PRODUCTION)
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.05)
    face1 = MineFace("face1", face_id=1, area_id=1, level_index=3, generator=gen, min_ore_mass=30000, max_ore_mass=50000, total_ore_to_extract=100000)
    tr = Truck("T01", drs.Timer("tmr", 0.0))

    mission = fmc.select_mission(
        truck=tr,
        current_total_stock=55000.0,  # Below 60kt
        is_plant_shutdown=False,
        can_mine_ore=True,
        active_prod_trucks=2,
        active_capital_dev_trucks=0,
        faces=[face1],
        area2_locked=True,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
    )
    assert mission is not None
    mission_type, face_id, level, is_waste = mission
    assert mission_type == MissionType.ORE_HAUL
    assert face_id == 1
    assert level == 3
    assert not is_waste


def test_fleet_mode_mission_selection_extra_trucks_production_mode():
    """In PRODUCTION mode, extra trucks go to stope turnaround first, then capital decline."""
    fmc = FleetModeController(initial_mode=FleetOperatingMode.PRODUCTION)
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.05)
    face1 = MineFace("face1", face_id=1, area_id=1, level_index=3, generator=gen, min_ore_mass=30000, max_ore_mass=50000, total_ore_to_extract=100000)
    face1.state = FaceState.DEVELOPMENT_TURNAROUND
    tr = Truck("T01", drs.Timer("tmr", 0.0))

    # Stock is full -> extra truck
    mission = fmc.select_mission(
        truck=tr,
        current_total_stock=60000.0,
        is_plant_shutdown=False,
        can_mine_ore=True,
        active_prod_trucks=4,
        active_capital_dev_trucks=0,
        faces=[face1],
        area2_locked=True,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
    )
    assert mission is not None
    mission_type, face_id, level, is_waste = mission
    assert mission_type == MissionType.STOPE_TURNAROUND_DEV
    assert is_waste

    # If no stope in turnaround, extra truck routes to capital decline
    face1.state = FaceState.ORE_READY
    mission2 = fmc.select_mission(
        truck=tr,
        current_total_stock=60000.0,
        is_plant_shutdown=False,
        can_mine_ore=True,
        active_prod_trucks=4,
        active_capital_dev_trucks=0,
        faces=[face1],
        area2_locked=True,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
    )
    assert mission2 is not None
    mission_type2, face_id2, level2, is_waste2 = mission2
    assert mission_type2 == MissionType.CAPITAL_DECLINE_DEV
    assert is_waste2


def test_fleet_mode_mission_selection_extra_trucks_development_mode():
    """In DEVELOPMENT mode, extra trucks prioritize capital decline first, then stope turnaround."""
    fmc = FleetModeController(initial_mode=FleetOperatingMode.DEVELOPMENT)
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.05)
    face1 = MineFace("face1", face_id=1, area_id=1, level_index=3, generator=gen, min_ore_mass=30000, max_ore_mass=50000, total_ore_to_extract=100000)
    face1.state = FaceState.DEVELOPMENT_TURNAROUND
    tr = Truck("T01", drs.Timer("tmr", 0.0))

    # Stock full, Area 2 locked -> in DEVELOPMENT mode, routes to CAPITAL_DECLINE_DEV
    mission = fmc.select_mission(
        truck=tr,
        current_total_stock=60000.0,
        is_plant_shutdown=False,
        can_mine_ore=True,
        active_prod_trucks=4,
        active_capital_dev_trucks=0,
        faces=[face1],
        area2_locked=True,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
    )
    assert mission is not None
    mission_type, face_id, level, is_waste = mission
    assert mission_type == MissionType.CAPITAL_DECLINE_DEV
    assert is_waste

    # If Area 2 is unlocked, extra truck routes to stope turnaround
    mission2 = fmc.select_mission(
        truck=tr,
        current_total_stock=60000.0,
        is_plant_shutdown=False,
        can_mine_ore=True,
        active_prod_trucks=4,
        active_capital_dev_trucks=0,
        faces=[face1],
        area2_locked=False,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
    )
    assert mission2 is not None
    mission_type2, face_id2, level2, is_waste2 = mission2
    assert mission_type2 == MissionType.STOPE_TURNAROUND_DEV
    assert is_waste2

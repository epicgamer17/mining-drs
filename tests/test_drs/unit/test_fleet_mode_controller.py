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
    """Verify Policy 1 strictly maintains PRODUCTION mode across all plant modes."""
    fmc = FleetModeController()
    
    # In MODE_A
    mode_a = fmc.evaluate_mode(
        policy=1,
        current_day=50.0,
        dev_progress_m=100.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
        plant_operating_mode="MODE_A",
    )
    assert mode_a == FleetOperatingMode.PRODUCTION

    # In MODE_B
    mode_b = fmc.evaluate_mode(
        policy=1,
        current_day=60.0,
        dev_progress_m=100.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
        plant_operating_mode="MODE_B",
    )
    assert mode_b == FleetOperatingMode.PRODUCTION
    assert fmc.mode == FleetOperatingMode.PRODUCTION


def test_fleet_mode_controller_policy2_campaign_coupling():
    """Verify Policy 2 switches to DEVELOPMENT in MODE_B and PRODUCTION in MODE_A."""
    fmc = FleetModeController()

    # In MODE_A (High Ore 2 draw campaign) -> PRODUCTION mode
    mode = fmc.evaluate_mode(
        policy=2,
        current_day=10.0,
        dev_progress_m=200.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
        plant_operating_mode="MODE_A",
    )
    assert mode == FleetOperatingMode.PRODUCTION

    # In MODE_B (Low Ore 2 draw campaign) -> DEVELOPMENT mode
    mode = fmc.evaluate_mode(
        policy=2,
        current_day=20.0,
        dev_progress_m=200.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
        plant_operating_mode="MODE_B",
    )
    assert mode == FleetOperatingMode.DEVELOPMENT

    # Switch back to MODE_A -> PRODUCTION mode
    mode = fmc.evaluate_mode(
        policy=2,
        current_day=30.0,
        dev_progress_m=300.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
        plant_operating_mode="MODE_A",
    )
    assert mode == FleetOperatingMode.PRODUCTION


def test_fleet_mode_controller_policy2_emergency_overrides():
    """Verify emergency submodes (Contingency, Surging) override to PRODUCTION mode."""
    fmc = FleetModeController()

    # Normal MODE_B is DEVELOPMENT
    mode = fmc.evaluate_mode(
        policy=2,
        current_day=10.0,
        dev_progress_m=200.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
        plant_operating_mode="MODE_B",
    )
    assert mode == FleetOperatingMode.DEVELOPMENT

    # Emergency MODE_B_CONTINGENCY -> PRODUCTION override to rescue starved Ore 1 stockpile
    mode_cont = fmc.evaluate_mode(
        policy=2,
        current_day=15.0,
        dev_progress_m=250.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
        plant_operating_mode="MODE_B_CONTINGENCY",
    )
    assert mode_cont == FleetOperatingMode.PRODUCTION

    # Emergency MODE_B_MINE_SURGING -> PRODUCTION override to surge ore
    mode_surge = fmc.evaluate_mode(
        policy=2,
        current_day=20.0,
        dev_progress_m=300.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=True,
        plant_operating_mode="MODE_B_MINE_SURGING",
    )
    assert mode_surge == FleetOperatingMode.PRODUCTION


def test_fleet_mode_controller_policy2_post_unlock():
    """Verify once Area 2 is unlocked, Policy 2 operates in PRODUCTION mode even in MODE_B."""
    fmc = FleetModeController()

    # Area 2 Unlocked (area2_locked = False) -> PRODUCTION mode
    mode = fmc.evaluate_mode(
        policy=2,
        current_day=250.0,
        dev_progress_m=4000.0,
        required_dev_m=4000.0,
        deadline_day=365.0,
        area2_locked=False,
        plant_operating_mode="MODE_B",
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


def test_fleet_mode_mission_selection_shutdown_maintenance():
    """Verify trucks stand down (return None) during site-wide SHUTDOWN maintenance."""
    fmc = FleetModeController(initial_mode=FleetOperatingMode.PRODUCTION)
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.05)
    face1 = MineFace("face1", face_id=1, area_id=1, level_index=3, generator=gen, min_ore_mass=30000, max_ore_mass=50000, total_ore_to_extract=100000)
    tr = Truck("T01", drs.Timer("tmr", 0.0))

    # During SHUTDOWN, no missions are assigned
    mission = fmc.select_mission(
        truck=tr,
        current_total_stock=10000.0,
        is_plant_shutdown=True,
        can_mine_ore=True,
        active_prod_trucks=0,
        active_capital_dev_trucks=0,
        faces=[face1],
        area2_locked=True,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
    )
    assert mission is None


def test_fleet_mode_mission_selection_production_mode_strict_area1_containment():
    """In PRODUCTION mode, extra trucks do NOT leak to Area 2 capital decline while Area 1 is active."""
    fmc = FleetModeController(initial_mode=FleetOperatingMode.PRODUCTION)
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.05)
    face1 = MineFace("face1", face_id=1, area_id=1, level_index=3, generator=gen, min_ore_mass=30000, max_ore_mass=50000, total_ore_to_extract=100000)
    face1.state = FaceState.ORE_READY
    tr = Truck("T01", drs.Timer("tmr", 0.0))

    # Stock is full (60kt), Area 1 active (not exhausted) -> returns None (standby on Level 3, 0 capital dev)
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
        area1_exhausted=False,
    )
    assert mission is None


def test_fleet_mode_mission_selection_production_mode_post_depletion():
    """In PRODUCTION mode, when Area 1 IS exhausted, trucks finally develop Area 2."""
    fmc = FleetModeController(initial_mode=FleetOperatingMode.PRODUCTION)
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.05)
    face1 = MineFace("face1", face_id=1, area_id=1, level_index=3, generator=gen, min_ore_mass=30000, max_ore_mass=50000, total_ore_to_extract=100000)
    face1.state = FaceState.EXHAUSTED
    tr = Truck("T01", drs.Timer("tmr", 0.0))

    # Area 1 exhausted -> dispatches to capital decline
    mission = fmc.select_mission(
        truck=tr,
        current_total_stock=10000.0,
        is_plant_shutdown=False,
        can_mine_ore=False,
        active_prod_trucks=0,
        active_capital_dev_trucks=0,
        faces=[face1],
        area2_locked=True,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
        area1_exhausted=True,
    )
    assert mission is not None
    mission_type, face_id, level, is_waste = mission
    assert mission_type == MissionType.CAPITAL_DECLINE_DEV
    assert is_waste


def test_fleet_mode_mission_selection_development_mode_reserved_push():
    """In DEVELOPMENT mode, capital decline gets dedicated reserved trucks first."""
    fmc = FleetModeController(initial_mode=FleetOperatingMode.DEVELOPMENT, reserved_dev_trucks=2)
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.05)
    face1 = MineFace("face1", face_id=1, area_id=1, level_index=3, generator=gen, min_ore_mass=30000, max_ore_mass=50000, total_ore_to_extract=100000)
    face1.state = FaceState.ORE_READY
    tr = Truck("T01", drs.Timer("tmr", 0.0))

    # 0 active capital dev trucks (< 2 reserved) -> routes to CAPITAL_DECLINE_DEV even though stock < 60kt
    mission = fmc.select_mission(
        truck=tr,
        current_total_stock=20000.0,
        is_plant_shutdown=False,
        can_mine_ore=True,
        active_prod_trucks=2,
        active_capital_dev_trucks=0,
        faces=[face1],
        area2_locked=True,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
        reserved_dev_trucks=2,
    )
    assert mission is not None
    mission_type, face_id, level, is_waste = mission
    assert mission_type == MissionType.CAPITAL_DECLINE_DEV
    assert is_waste

    # 2 active capital dev trucks (reserved quota met) -> routes next truck to ORE_HAUL
    mission2 = fmc.select_mission(
        truck=tr,
        current_total_stock=20000.0,
        is_plant_shutdown=False,
        can_mine_ore=True,
        active_prod_trucks=2,
        active_capital_dev_trucks=2,
        faces=[face1],
        area2_locked=True,
        preferred_face_id=1,
        face_levels={1: 3, 2: 6},
        reserved_dev_trucks=2,
    )
    assert mission2 is not None
    mission_type2, face_id2, level2, is_waste2 = mission2
    assert mission_type2 == MissionType.ORE_HAUL
    assert not is_waste2


def test_fleet_mode_mission_selection_60kt_cap_and_surging():
    """Verify stockpile >= 60kt blocks ore haulage unless is_surging is True."""
    fmc = FleetModeController(initial_mode=FleetOperatingMode.PRODUCTION)
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.05)
    face1 = MineFace("face1", face_id=1, area_id=1, level_index=3, generator=gen, min_ore_mass=30000, max_ore_mass=50000, total_ore_to_extract=100000)
    face1.state = FaceState.ORE_READY
    tr = Truck("T01", drs.Timer("tmr", 0.0))

    # Stock >= 60kt, not surging -> blocked (returns None)
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
        is_surging=False,
    )
    assert mission is None

    # Stock >= 60kt, but IS surging -> ore haulage allowed!
    mission_surge = fmc.select_mission(
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
        is_surging=True,
    )
    assert mission_surge is not None
    assert mission_surge[0] == MissionType.ORE_HAUL

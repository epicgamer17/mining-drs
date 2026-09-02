"""Unit tests for centralized drs_mining.config configuration classes."""

import pytest
from drs_mining.config import (
    CalendarConfig,
    TopologyConfig,
    PlantConfig,
    GeologyConfig,
    StrategicPlanningConfig,
    MillModeConfig,
    FleetModeConfig,
    MILL_MODES,
    FLEET_MODES,
    SimulationConfig,
    create_default_simulation_config,
    DEFAULT_CONFIG,
)


def test_calendar_config_defaults():
    cal = CalendarConfig()
    assert cal.days_in_year == 365.0
    assert cal.shift_seconds == 12.0 * 3600.0


def test_topology_config_defaults():
    top = TopologyConfig()
    assert top.decline_m == 2100.0
    assert top.level_spacing_m == 300.0
    assert top.area1_level == 3
    assert top.area2_level == 6


def test_plant_config_defaults():
    plant = PlantConfig()
    assert plant.target_ore_stock_level == 60000.0
    assert plant.critical_ore2_level == 20400.0
    assert plant.total_ore_to_extract == 6600000.0
    assert plant.mode_a_ore1_milling_rate == 3600.0
    assert plant.mode_a_ore2_milling_rate == 2400.0


def test_geology_config_defaults():
    geo = GeologyConfig()
    assert geo.area1_mean_fraction == 0.30
    assert geo.area2_mean_fraction == 0.35
    assert geo.min_parcel_mass == 30000.0
    assert geo.max_parcel_mass == 50000.0


def test_planning_config_defaults():
    plan = StrategicPlanningConfig()
    assert plan.area2_required_development == 4000.0
    assert plan.area2_ready_by_day == 365.0
    assert plan.tactical_review_period_days == 30.0


def test_mill_modes_created():
    assert "MODE_A" in MILL_MODES
    assert MILL_MODES["MODE_A"].category == "mill"


def test_fleet_modes_created():
    assert "PRODUCTION" in FLEET_MODES
    assert FLEET_MODES["PRODUCTION"].category == "fleet"
    assert "DEVELOPMENT" in FLEET_MODES


def test_simulation_config_defaults():
    cfg = DEFAULT_CONFIG
    assert cfg.plant.target_ore_stock_level == 60000.0
    assert cfg.planning.area2_required_development == 4000.0
    assert cfg.seed == 42


def test_simulation_config_factory_overrides():
    custom_cfg = create_default_simulation_config(
        plant=PlantConfig(target_ore_stock_level=80000.0),
        planning=StrategicPlanningConfig(area2_required_development=5000.0),
        seed=999,
    )
    assert custom_cfg.plant.target_ore_stock_level == 80000.0
    assert custom_cfg.planning.area2_required_development == 5000.0
    assert custom_cfg.seed == 999

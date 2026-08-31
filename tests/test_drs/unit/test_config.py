"""Unit tests for centralized drs_mining.config configuration classes and propagation."""

import pytest
from drs_mining.config import (
    CalendarConfig,
    TopologyConfig,
    HaulageFleetConfig,
    PlantConfig,
    GeologyConfig,
    StrategicPlanningConfig,
    EconomicParameters,
    MillModeConfig,
    FleetModeConfig,
    MILL_MODES,
    FLEET_MODES,
    MILL_MODE_CONFIGS,
    FLEET_MODE_CONFIGS,
    POLICY_2_FLEET_MODE_MAP,
    SimulationConfig,
    create_default_simulation_config,
    DEFAULT_CONFIG,
)
from drs_mining.components import MiningSimulationBase, AreaReadinessTarget, StrategicYearTarget


def test_calendar_config_properties():
    cal = CalendarConfig(
        days_in_year=365.0,
        shift_seconds=43200.0,
        haulage_seat_fraction=0.80,
    )
    assert cal.days_in_year == 365.0
    assert cal.seat_per_shift_sec == 0.80 * 43200.0


def test_topology_config_defaults():
    top = TopologyConfig()
    assert top.decline_m == 2100.0
    assert top.level_spacing_m == 300.0
    assert top.area1_level == 3
    assert top.area2_level == 6
    assert "surface" in top.speeds
    assert top.speeds["surface"]["loaded"] == 13.4


def test_haulage_fleet_config_defaults():
    fleet = HaulageFleetConfig()
    assert fleet.num_trucks == 18
    assert fleet.num_operators == 18
    assert fleet.truck_payload == 26.1
    assert fleet.availability == 0.85


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
    assert geo.stope_a1_mean_fractions == (0.28, 0.30, 0.32)
    assert geo.stope_a2_mean_fractions == (0.33, 0.35, 0.37)


def test_planning_config_defaults():
    plan = StrategicPlanningConfig()
    assert plan.area2_required_development == 4000.0
    assert plan.area2_ready_by_day == 365.0
    assert plan.annual_min_development_m == 10000.0
    assert plan.annual_min_ore1_production_t == 1300000.0
    assert plan.annual_min_ore2_production_t == 850000.0


def test_economics_config_defaults():
    econ = EconomicParameters()
    assert econ.copper_price_per_lb == 4.00
    assert econ.gold_price_per_oz == 1900.0
    assert econ.annual_discount_rate == 0.05
    assert econ.ore1_net_value_per_processed_tonne == 577.48
    assert econ.ore2_net_value_per_processed_tonne == 709.83


def test_simulation_config_factory_overrides():
    custom_cfg = create_default_simulation_config(
        fleet=HaulageFleetConfig(num_trucks=24, truck_payload=35.0),
        planning=StrategicPlanningConfig(area2_required_development=5000.0),
        seed=999,
    )
    assert custom_cfg.fleet.num_trucks == 24
    assert custom_cfg.fleet.truck_payload == 35.0
    assert custom_cfg.planning.area2_required_development == 5000.0
    assert custom_cfg.seed == 999


def test_simulation_base_config_propagation():
    """Verify that custom SimulationConfig values propagate into MiningSimulationBase subsystems."""
    custom_cfg = SimulationConfig(
        fleet=HaulageFleetConfig(num_trucks=12, num_operators=12, truck_payload=30.0, availability=0.90),
        plant=PlantConfig(target_ore_stock_level=80000.0, critical_ore2_level=25000.0),
        planning=StrategicPlanningConfig(area2_required_development=7500.0, area2_ready_by_day=180.0),
        economics=EconomicParameters(annual_discount_rate=0.08),
        seed=101,
    )

    sim = MiningSimulationBase(config=custom_cfg)
    assert len(sim.trucks) == 12
    assert len(sim.operators) == 12
    assert sim.availability == 0.90
    assert sim.target_ore_stock_level == 80000.0
    assert sim.critical_ore2_level == 25000.0
    assert sim.area2_readiness_target.required_development == 7500.0
    assert sim.area2_readiness_target.ready_by_day == 180.0
    assert sim.annual_discount_rate == 0.08
    assert sim.seed == 101

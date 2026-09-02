"""Unified Configuration Subsystem for DRS Mining Simulations.

Contains all physical dimensions, metallurgical parameters,
geological facies distributions, strategic targets,
and operating mode definitions in a single centralized module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from drs_mining.components.modes import OperatingMode


# ===========================================================================
# 1. Operating Modes Configuration (Mill Campaigns & Fleet Allocation)
# ===========================================================================

@dataclass(frozen=True)
class MillModeConfig:
    """Processing plant campaign draw rates and mode properties."""

    name: str
    id: int
    draw_rates: Mapping[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class FleetModeConfig:
    """Underground haulage fleet allocation and development reservation properties."""

    name: str
    id: int
    dev_reservation_fraction: float = 0.0  # Fleet fraction reserved for development [0.0 - 1.0]
    area2_dev_share: float = 0.50  # Split of surplus trucks dispatched to Area 2 capital decline [0.0 - 1.0]
    description: str = ""


# Default Mill Campaign Modes
MILL_MODE_CONFIGS: Dict[str, MillModeConfig] = {
    "MODE_A": MillModeConfig(
        name="MODE_A",
        id=0,
        draw_rates={"Ore1Stock": 3600.0, "Ore2Stock": 2400.0},
        description="High Ore 2 draw campaign (6,000 t/d total feed: 60% Ore 1 / 40% Ore 2)",
    ),
    "MODE_A_CONTINGENCY": MillModeConfig(
        name="MODE_A_CONTINGENCY",
        id=1,
        draw_rates={"Ore1Stock": 3900.0, "Ore2Stock": 0.0},
        description="Ore 1 emergency contingency campaign when Ore 2 stockpile is starved",
    ),
    "MODE_A_MINE_SURGING": MillModeConfig(
        name="MODE_A_MINE_SURGING",
        id=2,
        draw_rates={"Ore1Stock": 3600.0, "Ore2Stock": 2400.0},
        description="Mine surging reduced haulage target to manage high stockpile surges",
    ),
    "MODE_B": MillModeConfig(
        name="MODE_B",
        id=3,
        draw_rates={"Ore1Stock": 4600.0, "Ore2Stock": 800.0},
        description="Low Ore 2 draw campaign (5,400 t/d total feed: 85% Ore 1 / 15% Ore 2)",
    ),
    "MODE_B_CONTINGENCY": MillModeConfig(
        name="MODE_B_CONTINGENCY",
        id=4,
        draw_rates={"Ore1Stock": 0.0, "Ore2Stock": 2500.0},
        description="Ore 2 emergency contingency campaign when Ore 1 stockpile is starved",
    ),
    "MODE_B_MINE_SURGING": MillModeConfig(
        name="MODE_B_MINE_SURGING",
        id=5,
        draw_rates={"Ore1Stock": 4600.0, "Ore2Stock": 800.0},
        description="Mine surging reduced haulage target during high stockpile buildup",
    ),
    "SHUTDOWN": MillModeConfig(
        name="SHUTDOWN",
        id=6,
        draw_rates={"Ore1Stock": 0.0, "Ore2Stock": 0.0},
        description="Planned metallurgical plant maintenance shutdown",
    ),
}

# Default Mine Fleet Allocation Modes
FLEET_MODE_CONFIGS: Dict[str, FleetModeConfig] = {
    "PRODUCTION": FleetModeConfig(
        name="PRODUCTION",
        id=0,
        dev_reservation_fraction=0.0,
        area2_dev_share=0.35,
        description="Maximize active ore haulage throughput; surplus capacity to stope development",
    ),
    "DEVELOPMENT": FleetModeConfig(
        name="DEVELOPMENT",
        id=1,
        dev_reservation_fraction=0.20,
        area2_dev_share=0.85,
        description="Prioritize capital decline development for surplus trucks to unlock Area 2",
    ),
}

# Standard instantiated OperatingMode dictionaries
MILL_MODES: Dict[str, OperatingMode] = {
    name: OperatingMode(name, id=cfg.id, category="mill", description=cfg.description)
    for name, cfg in MILL_MODE_CONFIGS.items()
}

FLEET_MODES: Dict[str, OperatingMode] = {
    name: OperatingMode(
        name,
        id=cfg.id,
        category="fleet",
        dev_reservation_fraction=cfg.dev_reservation_fraction,
        area2_dev_share=cfg.area2_dev_share,
        description=cfg.description,
    )
    for name, cfg in FLEET_MODE_CONFIGS.items()
}


# ===========================================================================
# 2. Shift Calendar & Work Schedule Configuration
# ===========================================================================

@dataclass(frozen=True)
class CalendarConfig:
    """Shift calendar, work duration, and operational schedule."""

    days_in_year: float = 365.0  # Standard operating days per calendar year [days]
    non_production_days: int = 0  # Scheduled full-site stoppage days per year [days]
    shift_seconds: float = 12.0 * 3600.0  # Shift length in seconds (43,200 s = 12.0 hours)
    shift_work_hours: float = 12.0  # Active shift duration [hours]


# ===========================================================================
# 3. Mine Topology & Road Network Configuration
# ===========================================================================

@dataclass(frozen=True)
class TopologyConfig:
    """Basic mine geometry for analytical reference."""

    decline_m: float = 2100.0
    level_spacing_m: float = 300.0
    area1_level: int = 3
    area2_level: int = 6
    level_drift_m: float = 60.0
    surface_m: float = 300.0
    capital_decline_cross_section_m2: float = 25.0
    stope_cross_section_m2: float = 16.0
    rock_density_t_per_m3: float = 2.7


# ===========================================================================
# 4. Metallurgical Plant & Processing Configuration
# ===========================================================================

@dataclass(frozen=True)
class PlantConfig:
    """Metallurgical plant capacity, campaign durations, milling rates, and stockpile buffers."""

    target_ore_stock_level: float = 60000.0
    critical_ore2_level: float = 20400.0
    total_ore_to_extract: float = 6600000.0
    ore_to_be_extracted_during_warming_period: float = 600000.0
    duration_of_production_campaigns: float = 34.0
    duration_of_shutdowns: float = 1.0
    duration_of_contingency_segments: float = 1.0

    mode_a_ore1_milling_rate: float = 3600.0
    mode_a_ore2_milling_rate: float = 2400.0
    mode_a_contingency_ore1_milling_rate: float = 3900.0
    mode_b_ore1_milling_rate: float = 4600.0
    mode_b_ore2_milling_rate: float = 800.0
    mode_b_contingency_ore2_milling_rate: float = 2500.0

    stockpile_capacity: float = 120000.0
    initial_stock_fraction: float = 0.50


# ===========================================================================
# 5. Geology, Facies, and Stope Lifecycle Configuration
# ===========================================================================

@dataclass(frozen=True)
class GeologyConfig:
    """Stochastic geological facies parameters, parcel mass, reserves, and turnaround advance."""

    area1_mean_fraction: float = 0.30
    area1_std_dev: float = 0.05
    area2_mean_fraction: float = 0.35
    area2_std_dev: float = 0.05

    stope_a1_mean_fractions: Tuple[float, ...] = (0.28, 0.30, 0.32)
    stope_a2_mean_fractions: Tuple[float, ...] = (0.33, 0.35, 0.37)
    stope_std_dev: float = 0.03

    prob_new_facies: float = 0.30
    variation_same_facies: float = 0.01

    min_parcel_mass: float = 30000.0
    max_parcel_mass: float = 50000.0
    initial_parcel_mass: float = 40000.0

    stope_min_parcel_mass: float = 25000.0
    stope_max_parcel_mass: float = 40000.0
    area1_stope_reserve: float = 600000.0
    area2_stope_reserve: float = 1600000.0

    waste_to_ore_ratio: float = 0.10
    stope_a1_waste_to_ore_ratio: float = 0.15
    stope_a2_waste_to_ore_ratio: float = 0.20

    turnaround_dev_per_parcel_m: float = 75.0
    stope_turnaround_dev_per_parcel_m: float = 5.0

    # Satellite ore body parameters
    sporadic_probability: float = 0.7  # Probability a satellite is available per review period
    min_waste_fraction: float = 0.0
    max_waste_fraction: float = 0.3


# ===========================================================================
# 6. Strategic & Tactical Planning Configuration
# ===========================================================================

@dataclass(frozen=True)
class StrategicPlanningConfig:
    """Tactical review thresholds and minimal strategic development targets."""

    area2_required_development: float = 4000.0
    area2_ready_by_day: float = 365.0
    strategic_period_days: float = 365.0
    tactical_review_period_days: float = 30.0
    tactical_progress_tolerance: float = 0.90
    development_priority_truck_reservation_fraction: float = 0.33

    # Minimal development target (only to prevent catastrophic stockout)
    annual_min_development_m: float = 10000.0
    annual_min_ore1_production_t: float = 1300000.0
    annual_min_ore2_production_t: float = 850000.0


# ===========================================================================
# 7. Master Simulation Configuration Dataclass & Helper Singleton
# ===========================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """Master simulation configuration combining all subsystem parameters."""

    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    geology: GeologyConfig = field(default_factory=GeologyConfig)
    planning: StrategicPlanningConfig = field(default_factory=StrategicPlanningConfig)
    total_days: float = 365.0
    dt_max: float = 900.0
    telemetry_dt: float = 1800.0
    seed: int = 42


def create_default_simulation_config(**overrides) -> SimulationConfig:
    """Factory helper to instantiate a SimulationConfig with custom parameter overrides."""
    return SimulationConfig(**overrides)


# Singleton default configuration instance for convenient importing
DEFAULT_CONFIG: SimulationConfig = SimulationConfig()

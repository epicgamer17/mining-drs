"""Comprehensive physical, fleet, plant, and simulation configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from drs_mining.config.economics import EconomicParameters
from drs_mining.components.topology import DEFAULT_SPEEDS


@dataclass(frozen=True)
class CalendarConfig:
    """Shift calendar and work scheduling parameters."""

    days_in_year: float = 365.0
    non_production_days: int = 0
    shift_seconds: float = 12.0 * 3600.0  # 12 hours
    shift_work_hours: float = 12.0
    haulage_seat_fraction: float = 0.85
    holidays: Tuple[int, ...] = (1, 100, 200, 300)

    @property
    def seat_per_shift_sec(self) -> float:
        return self.haulage_seat_fraction * self.shift_seconds


@dataclass(frozen=True)
class TopologyConfig:
    """Underground mine geometry, level depths, drifts, and speed profiles."""

    decline_m: float = 2100.0
    level_spacing_m: float = 300.0
    area1_level: int = 3
    area2_level: int = 6
    level_drift_m: float = 60.0
    surface_m: float = 300.0
    speeds: Dict[str, Dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_SPEEDS))


@dataclass(frozen=True)
class HaulageFleetConfig:
    """Haul truck fleet specifications, spotting/dumping durations, and traffic congestion."""

    num_trucks: int = 8
    num_operators: int = 8
    num_lhds: int = 2
    truck_payload: float = 26.1
    availability: float = 0.85
    load_spot_min: float = 0.50
    lhd_acquisition_max_min: float = 0.80
    load_dur_min: float = 3.50
    dump_spot_min: float = 0.57
    dump_dur_min: float = 0.88
    surface_tip_sites: int = 2
    fuel_burn_pct_per_sec: float = 100.0 / (7.5 * 3600.0)
    refuel_dur_min: float = 25.0
    num_fuel_pumps: int = 2
    base_pass_bay_delay_sec: float = 13.0
    per_truck_pass_bay_delay_sec: float = 1.0
    dev_m_per_extra_truck_day: float = 5.0


@dataclass(frozen=True)
class PlantConfig:
    """Metallurgical plant capacity, campaign durations, and stockpile buffer targets."""

    target_ore_stock_level: float = 60000.0
    critical_ore2_level: float = 20400.0
    total_ore_to_extract: float = 6600000.0
    ore_to_be_extracted_during_warming_period: float = 600000.0
    duration_of_production_campaigns: float = 34.0
    duration_of_shutdowns: float = 1.0
    duration_of_contingency_segments: float = 1.0
    mode_a_ore1_milling_rate: float = 540.0 * 24.0
    mode_a_ore2_milling_rate: float = 60.0 * 24.0
    mode_a_contingency_ore1_milling_rate: float = 500.0 * 24.0
    mode_b_ore1_milling_rate: float = 300.0 * 24.0
    mode_b_ore2_milling_rate: float = 300.0 * 24.0
    mode_b_contingency_ore2_milling_rate: float = 650.0 * 24.0


@dataclass(frozen=True)
class GeologyConfig:
    """Stochastic facies generator parameters and parcel size distribution."""

    area1_mean_fraction: float = 0.30
    area1_std_dev: float = 0.05
    area2_mean_fraction: float = 0.35
    area2_std_dev: float = 0.05
    prob_new_facies: float = 0.30
    variation_same_facies: float = 0.01
    min_parcel_mass: float = 30000.0
    max_parcel_mass: float = 50000.0
    initial_parcel_mass: float = 40000.0


@dataclass(frozen=True)
class StrategicPlanningConfig:
    """Strategic annual targets, capital readiness requirements, and review tolerances."""

    area2_required_development: float = 4000.0
    area2_ready_by_day: float = 365.0
    strategic_period_days: float = 365.0
    tactical_review_period_days: float = 30.0
    tactical_progress_tolerance: float = 0.90
    development_priority_truck_reservation_fraction: float = 0.33


@dataclass(frozen=True)
class SimulationConfig:
    """Master configuration dataclass combining all simulation subsystems."""

    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    fleet: HaulageFleetConfig = field(default_factory=HaulageFleetConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    geology: GeologyConfig = field(default_factory=GeologyConfig)
    planning: StrategicPlanningConfig = field(default_factory=StrategicPlanningConfig)
    economics: EconomicParameters = field(default_factory=EconomicParameters)
    dt_max: float = 900.0
    seed: int = 42


def create_default_simulation_config(**overrides) -> SimulationConfig:
    """Factory helper to build a SimulationConfig with custom overrides."""
    return SimulationConfig(**overrides)

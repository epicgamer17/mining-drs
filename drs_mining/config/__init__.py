"""Unified Configuration Subsystem for DRS Mining Simulations.

Contains all physical dimensions, fleet specifications, metallurgical parameters,
geological facies distributions, strategic targets, DCF economic parameters,
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
    ore1_draw_rate: Optional[float] = None  # Planned Ore 1 draw rate [tonnes/day]
    ore2_draw_rate: Optional[float] = None  # Planned Ore 2 draw rate [tonnes/day]
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
        ore1_draw_rate=3600.0,
        ore2_draw_rate=2400.0,
        description="High Ore 2 draw campaign (6,000 t/d total feed: 60% Ore 1 / 40% Ore 2)",
    ),
    "MODE_A_CONTINGENCY": MillModeConfig(
        name="MODE_A_CONTINGENCY",
        id=1,
        ore1_draw_rate=3900.0,
        ore2_draw_rate=0.0,
        description="Ore 1 emergency contingency campaign when Ore 2 stockpile is starved",
    ),
    "MODE_A_MINE_SURGING": MillModeConfig(
        name="MODE_A_MINE_SURGING",
        id=2,
        description="Mine surging reduced haulage target to manage high stockpile surges",
    ),
    "MODE_B": MillModeConfig(
        name="MODE_B",
        id=3,
        ore1_draw_rate=4600.0,
        ore2_draw_rate=800.0,
        description="Low Ore 2 draw campaign (5,400 t/d total feed: 85% Ore 1 / 15% Ore 2)",
    ),
    "MODE_B_CONTINGENCY": MillModeConfig(
        name="MODE_B_CONTINGENCY",
        id=4,
        ore1_draw_rate=0.0,
        ore2_draw_rate=2500.0,
        description="Ore 2 emergency contingency campaign when Ore 1 stockpile is starved",
    ),
    "MODE_B_MINE_SURGING": MillModeConfig(
        name="MODE_B_MINE_SURGING",
        id=5,
        description="Mine surging reduced haulage target during high stockpile buildup",
    ),
    "SHUTDOWN": MillModeConfig(
        name="SHUTDOWN",
        id=6,
        ore1_draw_rate=0.0,
        ore2_draw_rate=0.0,
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

# Standard Mill-to-Fleet Mode Mapping Table for Policy 2 Value-Oriented Control
POLICY_2_FLEET_MODE_MAP: Dict[str, str] = {
    "MODE_A": "PRODUCTION",
    "MODE_A_CONTINGENCY": "PRODUCTION",
    "MODE_A_MINE_SURGING": "PRODUCTION",
    "MODE_B": "DEVELOPMENT",
    "MODE_B_CONTINGENCY": "PRODUCTION",
    "MODE_B_MINE_SURGING": "PRODUCTION",
    "SHUTDOWN": "PRODUCTION",
}


# ===========================================================================
# 2. Shift Calendar & Work Schedule Configuration
# ===========================================================================

@dataclass(frozen=True)
class CalendarConfig:
    """Shift calendar, work duration, operator seat ratio, and statutory holidays."""

    days_in_year: float = 365.0  # Standard operating days per calendar year [days]
    non_production_days: int = 0  # Scheduled full-site stoppage days per year [days]
    shift_seconds: float = 12.0 * 3600.0  # Shift length in seconds (43,200 s = 12.0 hours)
    shift_work_hours: float = 12.0  # Active shift duration [hours]
    haulage_seat_fraction: float = 0.85  # Effective operator seat time fraction (crib/breaks/pre-shift) [0.0 - 1.0]
    holidays: Tuple[int, ...] = (1, 100, 200, 300)  # Statutory holiday day indices

    @property
    def seat_per_shift_sec(self) -> float:
        """Effective operator productive seat time per shift in seconds (36,720 s = 10.2 h)."""
        return self.haulage_seat_fraction * self.shift_seconds


# ===========================================================================
# 3. Mine Topology & Road Network Configuration
# ===========================================================================

DEFAULT_SPEEDS: Dict[str, Dict[str, float]] = {
    "surface": {"empty": 17.4, "loaded": 13.4},  # Surface corridor truck speeds [km/h]
    "decline": {"empty": 15.1, "loaded": 11.2},  # Main decline corridor truck speeds [km/h]
    "ramp": {"empty": 12.9, "loaded": 9.2},      # Internal spiral ramp truck speeds [km/h]
    "level": {"empty": 7.6, "loaded": 6.6},      # Level access drift truck speeds [km/h]
}


@dataclass(frozen=True)
class TopologyConfig:
    """Underground mine geometry, level depths, drifts, passing bays, and speed profiles."""

    decline_m: float = 2100.0  # Length of surface decline from portal to upper mining horizon [metres]
    level_spacing_m: float = 300.0  # Vertical distance between successive underground levels [metres]
    area1_level: int = 3  # Level index for upper Area 1 production stopes (Level 3 = 900m depth)
    area2_level: int = 6  # Level index for deep Area 2 capital expansion (Level 6 = 1800m depth)
    level_drift_m: float = 60.0  # Horizontal cross-cut drift distance from decline to muck bay [metres]
    surface_m: float = 300.0  # Surface road distance from portal to surface dump pocket [metres]
    capital_decline_cross_section_m2: float = 25.0  # Heading cross-sectional area of capital decline [m²]
    stope_cross_section_m2: float = 16.0  # Heading cross-sectional area of stope ore drive [m²]
    rock_density_t_per_m3: float = 2.7  # In-situ solid rock mass density [tonnes/m³]
    speeds: Dict[str, Dict[str, float]] = field(default_factory=lambda: dict(DEFAULT_SPEEDS))  # Speed table [km/h]
    base_pass_bay_delay_sec: float = 13.0  # Base wait time at single-lane passing bays [seconds]
    per_truck_pass_bay_delay_sec: float = 1.0  # Additional passing bay delay per interacting truck [seconds/truck]
    traffic_variation_tol: float = 0.20  # Triangular stochastic noise width on traffic delay (±20%) [0.0 - 1.0]


# ===========================================================================
# 4. Haulage Fleet & Loading Equipment Configuration
# ===========================================================================

@dataclass(frozen=True)
class HaulageFleetConfig:
    """Haul truck fleet specifications, loader allocation, spotting/dumping, and refueling."""

    num_trucks: int = 18  # Total underground articulated dump trucks in fleet (e.g. Caterpillar AD30)
    num_operators: int = 18  # Total certified operators per shift
    num_lhds_per_face: int = 2  # Number of LHD loaders allocated per active muck bay
    num_lhds_per_decline: int = 1  # Number of LHD loaders dedicated to capital decline advance
    truck_payload: float = 26.1  # Rated nominal truck payload capacity [tonnes]
    availability: float = 0.85  # Mechanical availability fraction of haul trucks [0.0 - 1.0]
    load_spot_min: float = 0.50  # Truck reversing and spotting duration at muck bay [minutes]
    lhd_acquisition_max_min: float = 0.80  # Maximum waiting duration for LHD acquisition [minutes]
    load_dur_min: float = 3.50  # LHD loading cycle duration [minutes]
    dump_spot_min: float = 0.57  # Truck reversing and spotting duration at surface dump station [minutes]
    dump_dur_min: float = 0.88  # Truck box hoist, discharge, and lower duration [minutes]
    surface_tip_sites: int = 2  # Number of concurrent dumping bays at surface crusher
    fuel_burn_pct_per_sec: float = 100.0 / (7.5 * 3600.0)  # Fuel burn rate (% tank/sec, 7.5 operating hours/tank)
    refuel_dur_min: float = 25.0  # Full tank refueling dwell time [minutes]
    num_fuel_pumps: int = 2  # Concurrent underground refueling bays
    dev_m_per_extra_truck_day: float = 5.0  # Development advance rate per dedicated truck-day [metres/truck-day]
    truck_cycle_time_sec: float = 2100.0  # Nominal round-trip haulage cycle time for analytical dispatch [seconds = 35 min]


# ===========================================================================
# 5. Metallurgical Plant & Processing Configuration
# ===========================================================================

@dataclass(frozen=True)
class PlantConfig:
    """Metallurgical plant capacity, campaign durations, milling rates, and stockpile buffers."""

    target_ore_stock_level: float = 60000.0  # Target surface ROM stockpile buffer (Ore 1 + Ore 2) [tonnes]
    critical_ore2_level: float = 20400.0  # Critical high-grade Ore 2 buffer triggering contingency milling [tonnes]
    total_ore_to_extract: float = 6600000.0  # Total life-of-mine production ore reserves to extract [tonnes = 6.6 Mt]
    ore_to_be_extracted_during_warming_period: float = 600000.0  # Commissioning warmup ore extraction [tonnes = 0.6 Mt]
    duration_of_production_campaigns: float = 34.0  # Duration of standard processing campaign [days]
    duration_of_shutdowns: float = 1.0  # Duration of planned mill maintenance shutdown [days]
    duration_of_contingency_segments: float = 1.0  # Duration of emergency contingency milling batch [days]

    # Milling Draw Rates [tonnes/day]
    mode_a_ore1_milling_rate: float = 3600.0  # Mode A Ore 1 draw rate [t/d]
    mode_a_ore2_milling_rate: float = 2400.0  # Mode A Ore 2 draw rate [t/d] (total 6,000 t/d)
    mode_a_contingency_ore1_milling_rate: float = 3900.0  # Mode A contingency Ore 1 rate (Ore 2 starved) [t/d]
    mode_a_contingency_ore2_milling_rate: float = 0.0  # Mode A contingency Ore 2 rate [t/d]
    mode_b_ore1_milling_rate: float = 4600.0  # Mode B Ore 1 draw rate [t/d]
    mode_b_ore2_milling_rate: float = 800.0   # Mode B Ore 2 draw rate [t/d] (total 5,400 t/d)
    mode_b_contingency_ore1_milling_rate: float = 0.0  # Mode B contingency Ore 1 rate [t/d]
    mode_b_contingency_ore2_milling_rate: float = 2500.0  # Mode B contingency Ore 2 rate (Ore 1 starved) [t/d]

    # Stockpile Storage Limits
    stockpile_capacity: float = 120000.0  # Maximum physical capacity per individual stockpile [tonnes]
    initial_stock_fraction: float = 0.50  # Initial fill percentage of target stock level at t=0 [0.0 - 1.0]


# ===========================================================================
# 6. Geology, Facies, and Stope Lifecycle Configuration
# ===========================================================================

@dataclass(frozen=True)
class GeologyConfig:
    """Stochastic geological facies parameters, parcel mass, reserves, and turnaround advance."""

    # Two-Area Primary Face Geology (Ore 2 grade fraction: 0.0 to 1.0)
    area1_mean_fraction: float = 0.30  # Area 1 (Level 3) mean high-grade Ore 2 mass fraction [0.0 - 1.0]
    area1_std_dev: float = 0.05  # Area 1 Ore 2 fraction standard deviation
    area2_mean_fraction: float = 0.35  # Area 2 (Level 6) mean high-grade Ore 2 mass fraction [0.0 - 1.0]
    area2_std_dev: float = 0.05  # Area 2 Ore 2 fraction standard deviation

    # Multi-Stope Facies Means (for two_area_stope_lifecycle study)
    stope_a1_mean_fractions: Tuple[float, ...] = (0.28, 0.30, 0.32)  # Stopes 1A, 1B, 1C mean Ore 2 fractions
    stope_a2_mean_fractions: Tuple[float, ...] = (0.33, 0.35, 0.37)  # Stopes 2A, 2B, 2C mean Ore 2 fractions
    stope_std_dev: float = 0.03  # Stope facies standard deviation

    # Stochastic Markov Facies Generator Parameters
    prob_new_facies: float = 0.30  # Transition probability to new geological facies round [0.0 - 1.0]
    variation_same_facies: float = 0.01  # Stochastic variance within same facies round

    # Parcel & Reserve Masses
    min_parcel_mass: float = 30000.0  # Minimum mass of a blasted ore round / parcel [tonnes]
    max_parcel_mass: float = 50000.0  # Maximum mass of a blasted ore round / parcel [tonnes]
    initial_parcel_mass: float = 40000.0  # Mass of initial blasted round ready at t=0 [tonnes]

    stope_min_parcel_mass: float = 25000.0  # Stope lifecycle study parcel minimum mass [tonnes]
    stope_max_parcel_mass: float = 40000.0  # Stope lifecycle study parcel maximum mass [tonnes]
    area1_stope_reserve: float = 600000.0  # Total reserve per stope in Area 1 (3 stopes = 1.8 Mt) [tonnes]
    area2_stope_reserve: float = 1600000.0  # Total reserve per stope in Area 2 (3 stopes = 4.8 Mt) [tonnes]

    waste_to_ore_ratio: float = 0.10  # Waste rock mass to ore mass ratio in two-area heading [fraction]
    stope_a1_waste_to_ore_ratio: float = 0.15  # Waste-to-ore ratio for Area 1 stopes [fraction]
    stope_a2_waste_to_ore_ratio: float = 0.20  # Waste-to-ore ratio for Area 2 stopes [fraction]

    turnaround_dev_per_parcel_m: float = 75.0  # Heading advance required between blasted rounds [metres]
    stope_turnaround_dev_per_parcel_m: float = 5.0  # Stope turnaround advance per parcel in multi-stope study [metres]


# ===========================================================================
# 7. Strategic & Tactical Planning Configuration
# ===========================================================================

@dataclass(frozen=True)
class StrategicPlanningConfig:
    """Strategic annual targets, capital readiness milestones, and monthly review thresholds."""

    area2_required_development: float = 4000.0  # Required capital decline metres to unlock Area 2 [metres]
    area2_ready_by_day: float = 365.0  # Target schedule deadline for Area 2 unlock [days]
    strategic_period_days: float = 365.0  # Strategic planning annual evaluation epoch [days]
    tactical_review_period_days: float = 30.0  # Monthly tactical review and mode evaluation interval [days]
    tactical_progress_tolerance: float = 0.90  # Trajectory adherence tolerance threshold [0.0 - 1.0]
    development_priority_truck_reservation_fraction: float = 0.33  # Fleet fraction reserved for dev in DEV mode [0.0 - 1.0]

    # Annual Strategic Plan Targets
    annual_min_development_m: float = 10000.0  # Annual target for total mine development advance [metres/year]
    annual_min_ore1_production_t: float = 1300000.0  # Annual target for low-grade Ore 1 production [tonnes/year]
    annual_min_ore2_production_t: float = 850000.0  # Annual target for high-grade Ore 2 production [tonnes/year]


# ===========================================================================
# 8. Economics & Discounted Cash Flow Valuation Configuration
# ===========================================================================

@dataclass(frozen=True)
class EconomicParameters:
    """Commodity prices, metallurgical recoveries, unit costs, and DCF parameters."""

    # Commodity Market Prices
    copper_price_per_lb: float = 4.00  # Market copper price [USD/lb]
    gold_price_per_oz: float = 1900.0  # Market gold price [USD/troy oz]

    # In-Situ Feed Grades
    ore1_cu_grade: float = 0.007  # Ore 1 copper grade (0.70% Cu) [mass fraction]
    ore1_au_grade_gpt: float = 0.40  # Ore 1 gold grade [grams/tonne]
    ore2_cu_grade: float = 0.015  # Ore 2 copper grade (1.50% Cu) [mass fraction]
    ore2_au_grade_gpt: float = 1.20  # Ore 2 gold grade [grams/tonne]

    # Metallurgical Concentrator Recoveries
    copper_recovery_ore1: float = 0.88  # Plant copper recovery for Ore 1 [0.0 - 1.0]
    gold_recovery_ore1: float = 0.70  # Plant gold recovery for Ore 1 [0.0 - 1.0]
    copper_recovery_ore2: float = 0.92  # Plant copper recovery for Ore 2 [0.0 - 1.0]
    gold_recovery_ore2: float = 0.80  # Plant gold recovery for Ore 2 [0.0 - 1.0]

    # Unit Operating & Capital Costs
    milling_cost_per_tonne: float = 14.0  # Processing plant operating cost [USD/tonne]
    haulage_cost_per_tonne: float = 4.50  # Underground truck haulage operating cost [USD/tonne]
    development_cost_per_metre: float = 4500.0  # Capital heading advance cost [USD/linear metre]
    annual_discount_rate: float = 0.08  # Annual financial discount rate for NPV calculation [0.08 = 8% per Slide 5]
    fixed_cost_per_day: float = 74460.0  # Fixed daily site overhead, baseload power & indirect costs [USD/day]
    stockout_penalty_per_day: float = 25000.0  # Mill idling, thermal/reagent standby, and contract penalty [USD/day]

    # Direct Net-Value Mode Parameters (Two-Area DCF & Counterfactual Incremental NPV Study)
    ore1_net_value_per_processed_tonne: Optional[float] = 577.48  # Direct net value of Ore 1 [USD/tonne]
    ore2_net_value_per_processed_tonne: Optional[float] = 709.83  # Direct net value of Ore 2 [USD/tonne]
    production_cost_per_tonne: Optional[float] = 135.0  # Combined mining & processing operating cost [USD/tonne]
    development_cost_per_unit: Optional[float] = 15000.0  # Capital development advance cost [USD/metre]


# ===========================================================================
# 9. Master Simulation Configuration Dataclass & Helper Singleton
# ===========================================================================

@dataclass(frozen=True)
class SimulationConfig:
    """Master simulation configuration combining all subsystem parameters."""

    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    fleet: HaulageFleetConfig = field(default_factory=HaulageFleetConfig)
    plant: PlantConfig = field(default_factory=PlantConfig)
    geology: GeologyConfig = field(default_factory=GeologyConfig)
    planning: StrategicPlanningConfig = field(default_factory=StrategicPlanningConfig)
    economics: EconomicParameters = field(default_factory=EconomicParameters)
    total_days: float = 365.0  # Default simulation duration [days]
    dt_max: float = 900.0  # Maximum discrete event integration time step [seconds = 15 min]
    telemetry_dt: float = 1800.0  # Telemetry time step [seconds = 30 min]
    seed: int = 42  # Random seed for stochastic reproducibility


def create_default_simulation_config(**overrides) -> SimulationConfig:
    """Factory helper to instantiate a SimulationConfig with custom parameter overrides."""
    return SimulationConfig(**overrides)


# Singleton default configuration instance for convenient importing
DEFAULT_CONFIG: SimulationConfig = SimulationConfig()

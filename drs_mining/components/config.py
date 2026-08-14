import math
from dataclasses import dataclass, field

from .planning import AreaReadinessTarget, StrategicYearTarget


@dataclass
class BaseDualStockpileConfig:
    replication_length: float = math.inf
    """
    Shared configuration for dual-stockpile surging mine operations.
    Contains mass-balance and scheduling parameters common to all models.
    """

    # Mass Balance Parameters
    target_ore_stock_level: float = 60000.0
    total_ore_to_extract: float = 6600000.0
    ore_to_be_extracted_during_warming_period: float = 600000.0
    critical_ore2_level: float = 20400.0

    # Timing / Scheduling
    duration_of_production_campaigns: float = 34.0
    duration_of_shutdowns: float = 1.0
    duration_of_contingency_segments: float = 1.0
    fleet_shift_duration: float = 0.5  # 12 hours; simulation time is in days.

    # Strategic / Tactical planning horizons.
    # Strategic planning sets annual commitments; the existing controller
    # reviews progress monthly. Zero targets preserve the current baseline.
    strategic_period_days: float = 365.0
    tactical_review_period_days: float = 30.0
    tactical_progress_tolerance: float = 0.90
    strategic_targets: tuple = field(
        default_factory=lambda: (StrategicYearTarget(),)
    )

    # Future-area readiness is a long-horizon strategic commitment.  It is
    # tracked cumulatively from the start of strategic planning and does not
    # reset at the annual boundary.
    area2_readiness_target: AreaReadinessTarget = field(
        default_factory=AreaReadinessTarget
    )

    # Fractions of the EXISTING total development rate allocated specifically to
    # the future Area 2 project. These are scenario/control assumptions.
    area2_development_fraction_production: float = 0.35
    area2_development_fraction_balanced: float = 0.60
    area2_development_fraction_development: float = 0.85

    # Face index is zero-based; 1 means the second production face.
    area2_physical_unlock_enabled: bool = True
    area2_face_index: int = 1
    area2_redeploy_locked_face_trucks_to_development: bool = True

    # Explicit development unit calibration. The legacy
    # `development_rate_per_extra_truck=50` is intentionally NOT used as metres.
    # 5 m per extra-truck-day is a research calibration parameter.
    development_metres_per_extra_truck_per_day: float = 5.0
    annual_development_benchmark_metres: float = 10000.0

    # Counterfactual switch used only for the true incremental-NPV comparison.
    # When True, Area 2 is permanently unavailable and no Area-2 project
    # development is charged.
    area2_counterfactual_disable: bool = False

    # Keep operating targets inside the blend that currently available faces can
    # sustain, especially while the Area 2 face is physically locked.
    mode_blend_feasibility_enabled: bool = True

    # When annual development falls behind plan and the tactical review selects
    # DEVELOPMENT, reserve part of the production truck fleet for development.
    development_priority_truck_reservation_fraction: float = 0.20

    # Strategic economic assumptions.  All unit values/costs default to zero so
    # strategic accounting does not change physical simulation behaviour.  Value
    # is recognized when ore is processed by the plant, not when it is mined.
    annual_discount_rate: float = 0.08
    ore1_net_value_per_processed_tonne: float = 0.0
    ore2_net_value_per_processed_tonne: float = 0.0
    production_cost_per_tonne: float = 0.0
    development_cost_per_unit: float = 0.0
    fixed_cost_per_day: float = 0.0

    # Optional strategic proxy for future value unlocked when Area 2 becomes
    # ready.  Keep at zero when Area 2 production is explicitly modelled later
    # to avoid double counting future value.
    area2_future_access_value_at_readiness: float = 0.0

    # Helper Constants
    stockout_epsilon: float = 1e-9


@dataclass
class ConcentratorConfig(BaseDualStockpileConfig):
    """
    Configuration for Navarra (2019): Base-metal flotation concentrator.
    Attribute: Ore Grade (%)
    """

    mean_ore_fraction: float = 0.30
    std_dev_ore_fraction: float = 0.05

    # Generator Parameters
    min_ore_mass: float = 30000.0
    max_ore_mass: float = 50000.0
    prob_new_facies: float = 0.3  # NOTE Arena example incorrectly set this to 30.
    variation_same_facies: float = 0.01

    # Milling rates specific to the Concentrator paper
    mode_a_ore1_milling_rate: float = 3600.0
    mode_a_ore2_milling_rate: float = 2400.0
    mode_a_contingency_ore1_milling_rate: float = 3900.0
    mode_b_ore1_milling_rate: float = 4600.0
    mode_b_ore2_milling_rate: float = 800.0
    mode_b_contingency_ore2_milling_rate: float = 2500.0

    # --- New Parameters for Cycle Time & Match Factor ---
    truck_velocity: float = 15.0  # e.g., km/h
    loader_cycle_time_hours: float = 0.0833  # 5 mins to load 1 bucket (15t)
    truck_dump_time_hours: float = 0.033  # e.g., 2 mins to dump

    # --- Traffic Delay Parameters ---
    # Traffic delay increases cycle time based on the number of trucks at a face
    traffic_delay_per_truck_hours: float = 0.015  # Added delay per truck

    # --- Face Physical Parameters (Default) ---
    total_lhd_count: float = 3.0
    total_truck_count: float = 10.0
    max_lhds_per_face: float = 2.0  # Physical constraint on LHDs per face
    max_trucks_per_face: float = 6.0  # Physical constraint on trucks per face
    face_haul_distance: tuple = (1.5, 2.2)
    face_accessibility_fraction: tuple = (
        0.93,
        0.91,
    )  # this is suppose to mimic blasting and times when face is not accesible by cars (so we decrease the throughput by the time its not available instead of making like a new "mode" or phase). TODO: is this going to be made obsolete by modelling development?

    # --- Mine Development Parameters ---
    development_rate_per_extra_truck: float = (
        50.0  # meters (or tonnes) developed per unused truck per day
    )

    # Underground Logistics
    fleet_mechanical_availability: float = 0.85
    loader_payload_tonnes: float = 15.0
    truck_payload_tonnes: float = 30.0


@dataclass
class ShelswellConfig(BaseDualStockpileConfig):
    # Geometry and Layout (meters)
    decline_length: float = 2100.0
    decline_grade: float = 0.05
    ramp_length: float = 1800.0
    ramp_grades: tuple = (0.08, 0.13)
    level_spacing: float = 300.0
    ore_loadout_distance: float = 40.0
    waste_loadout_distance: float = 55.0
    air_door_distance: float = 20.0
    rom_pad_distance: float = 300.0
    waste_stockpile_distance: float = 440.0
    maintenance_shop_distance: float = 260.0
    fuel_depot_distance: float = 270.0
    
    # Speeds (kph)
    speed_surface_loaded: float = 13.4
    speed_surface_empty: float = 17.4
    speed_decline_loaded: float = 11.2
    speed_decline_empty: float = 15.1
    speed_ramp_loaded: float = 9.2
    speed_ramp_empty: float = 12.9
    speed_level_loaded: float = 6.6
    speed_level_empty: float = 7.6
    speed_lhd_loaded: float = 5.89
    speed_lhd_empty: float = 6.78
    
    # Shift "Seat Time" Workable Availability fractions
    seat_time_fraction_truck: float = 0.5417
    seat_time_fraction_lhd: float = 0.5833
    seat_time_fraction_maintenance: float = 0.7917
    
    # Cycle times (minutes)
    lhd_load_spot_minutes: float = 0.46
    lhd_load_minutes: float = 0.88
    lhd_dump_minutes: float = 0.73
    lhd_tram_distance: float = 35.0
    lhd_acquisition_delay_minutes: float = 1.5
    
    truck_load_spot_minutes: float = 0.82
    truck_dump_spot_minutes: float = 0.57
    truck_dump_minutes: float = 0.88
    
    # Capacity and payloads (tonnes)
    truck_payload_ore: float = 26.1
    truck_payload_waste: float = 24.6
    lhd_payload_ore: float = 14.0
    lhd_payload_waste: float = 12.5
    
    # Fleet sizes and availability (defaults)
    total_truck_count: float = 10.0
    total_operators: float = 10.0
    overall_mechanical_availability: float = 1.0
    traffic_delay_per_truck_hours: float = 0.005


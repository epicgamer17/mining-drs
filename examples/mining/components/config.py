import math
from dataclasses import dataclass


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
    max_trucks_per_face: float = 5.0  # Physical constraint on trucks per face
    face_haul_distance: tuple = (1.0, 1.2)
    face_accessibility_fraction: tuple = (0.93, 0.91)

    # --- Mine Development Parameters ---
    development_rate_per_extra_truck: float = (
        50.0  # meters (or tonnes) developed per unused truck per day
    )

    # Underground Logistics
    fleet_mechanical_availability: float = 0.85
    loader_payload_tonnes: float = 15.0
    truck_payload_tonnes: float = 30.0

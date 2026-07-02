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

    # Physical operating bounds for mine-side surging targets.
    max_mine_extraction_rate: float = 6000.0
    max_surging_extraction_rate: float = 8000.0
    min_effective_surging_fraction: float = 0.05

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

    # Face-level fleet capacity model. Disabled by default so Policy 1 stays
    # the current fixed mode-dependent allocation baseline.
    enable_face_capacity_limit: bool = False

    # --- New Parameters for Cycle Time & Match Factor ---
    truck_velocity: float = 20.0  # e.g., km/h
    loader_cycle_time_hours: float = 0.05  # e.g., 3 mins to load a truck
    truck_dump_time_hours: float = 0.033  # e.g., 2 mins to dump

    # --- Traffic Delay Parameters ---
    # Traffic delay increases cycle time based on the number of trucks at a face
    traffic_delay_base: float = 0.01
    traffic_delay_multiplier: float = 0.005  # Added delay per truck

    # --- Mine Development Parameters ---
    development_rate_per_extra_truck: float = (
        50.0  # meters (or tonnes) developed per unused truck per day
    )

    # --- Generalize for N Faces (Change tuples to support > 2 faces) ---
    num_faces: int = 3  # Increase as needed
    face_lhd_count: tuple = (
        0.33,
        0.33,
        0.33,
    )  # TODO: count of 0.33 doesnt really make sense.
    face_truck_count: tuple = (0.33, 0.33, 0.33)
    # Underground Logistics
    fleet_mechanical_availability: float = 0.85
    face_accessibility_fraction: tuple = (0.95, 0.95, 0.95)
    face_haul_distance: tuple = (4.5, 3.0, 6.0)
    face_shift_allocation_fraction: tuple = (1.0, 1.0, 1.0)
    loader_payload_tonnes: float = 13000.0
    truck_payload_tonnes: float = 13000.0

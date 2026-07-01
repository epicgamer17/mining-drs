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
    face_lhd_allocation: tuple = (0.5, 0.5)
    face_truck_allocation: tuple = (0.5, 0.5)
    face_availability: tuple = (0.95, 0.95)
    face_haul_distance: tuple = (1.0, 1.0)
    face_delay_factor: tuple = (0.0, 0.0)
    face_gas_delay_factor: tuple = (0.0, 0.0)
    face_truck_congestion_threshold: tuple = (0.5, 0.5)
    face_shift_capacity_factor: tuple = (1.0, 1.0)
    lhd_capacity_per_allocation: float = 13000.0
    truck_capacity_per_allocation: float = 13000.0
    truck_congestion_delay_sensitivity: float = 0.0
    haul_distance_reference: float = 1.0
    haul_distance_sensitivity: float = 0.10

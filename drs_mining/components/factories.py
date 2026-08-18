"""Unified factory functions and simulation builders for mining operations."""

import json
import math
from typing import Sequence, Mapping, Union, List, Optional
from .stockpiles import Stockpile
from .mine_face import MineFace
from .fleet import ContinuousFleetLogistics, Truck, LHD
from .plant import MetallurgicalPlant
from .controllers import OperatingModeController, FleetController
from .generators import StochasticFaciesGenerator


def create_stockpiles(configs: Sequence[Mapping]) -> List[Stockpile]:
    """Factory helper to construct a list of Stockpiles from config dicts."""
    stockpiles = []
    for cfg in configs:
        stockpiles.append(
            Stockpile(
                name=cfg["name"],
                expected_attributes=cfg.get("expected_attributes", ()),
                initial_mass=cfg.get("initial_mass", 0.0),
                initial_attributes=cfg.get("initial_attributes", {}),
                capacity=cfg.get("capacity", math.inf),
                attr_inflow=cfg.get("attr_inflow", 1.0),
            )
        )
    return stockpiles


def load_topology_dict(topology: Union[dict, list, str, bytes]) -> Union[dict, list]:
    """Loads native Python dictionary/list topology representation.

    Args:
        topology: Path to JSON file, raw JSON string/bytes, or an already-loaded dict/list.

    Returns:
        The loaded topology dict/list structure.
    """
    if isinstance(topology, (dict, list)):
        return topology
    elif isinstance(topology, bytes):
        return json.loads(topology.decode("utf-8"))
    elif isinstance(topology, str):
        trimmed = topology.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (
            trimmed.startswith("[") and trimmed.endswith("]")
        ):
            return json.loads(trimmed)
        else:
            with open(topology, "r", encoding="utf-8") as f:
                return json.load(f)
    else:
        raise TypeError(
            f"Unsupported topology type: {type(topology)}. Expected dict, list, str (path or JSON), or bytes."
        )


def create_truck_fleet(
    count: int,
    truck_type: str,
    ore_payload_cap: float,
    waste_payload_cap: float,
    speeds: Mapping,
    prefix: str = "T",
    fuel_burn_rate_pct_per_sec: float = 0.005,
    **kwargs,
) -> List[Truck]:
    """Creates a fleet of Truck dataclasses with given configuration."""
    fleet = []
    for i in range(1, count + 1):
        truck_id = f"{prefix}{i:02d}" if prefix else str(i)
        t = Truck(
            truck_id=truck_id,
            truck_type=truck_type,
            ore_payload_cap=ore_payload_cap,
            waste_payload_cap=waste_payload_cap,
            speeds=dict(speeds),
            fuel_burn_rate_pct_per_sec=fuel_burn_rate_pct_per_sec,
            **kwargs,
        )
        fleet.append(t)
    return fleet


def create_lhd_fleet(
    levels: Sequence[int],
    bucket_ore_cap: float,
    bucket_waste_cap: float,
    load_spot_min: float,
    load_min: float,
    dump_min: float,
    tram_dist_m: float,
    speed_loaded_kph: float,
    speed_empty_kph: float,
    count_per_level: int = 1,
    prefix: str = "LHD",
    **kwargs,
) -> List[LHD]:
    """Creates a fleet of LHD loaders assigned to specific mine levels."""
    level_list = list(levels) if isinstance(levels, (list, tuple, range)) else list(range(1, int(levels) + 1))
    lhds = []
    for lvl in level_list:
        for idx in range(1, count_per_level + 1):
            suffix = f"_{idx}" if count_per_level > 1 else ""
            lhd_id = f"{prefix}_L{lvl}{suffix}"
            lhds.append(
                LHD(
                    lhd_id=lhd_id,
                    level_index=lvl,
                    bucket_ore_cap=bucket_ore_cap,
                    bucket_waste_cap=bucket_waste_cap,
                    load_spot_min=load_spot_min,
                    load_min=load_min,
                    dump_min=dump_min,
                    tram_dist_m=tram_dist_m,
                    speed_loaded_kph=speed_loaded_kph,
                    speed_empty_kph=speed_empty_kph,
                    **kwargs,
                )
            )
    return lhds


def build_mining_simulation(
    *,
    num_faces: int = 1,
    faces: Sequence[MineFace] = (),
    face_generators: Sequence = (),
    face_mean_fractions: Sequence[float] = (),
    mode_controller_cls=OperatingModeController,
    fleet_controller_cls=FleetController,
    mean_ore_fraction: float = 0.30,
    std_dev_ore_fraction: float = 0.05,
    target_ore_stock_level: float = 60000.0,
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    critical_ore2_level: float = 20400.0,
    duration_of_production_campaigns: float = 34.0,
    duration_of_shutdowns: float = 1.0,
    duration_of_contingency_segments: float = 1.0,
    min_ore_mass: float = 30000.0,
    max_ore_mass: float = 50000.0,
    prob_new_facies: float = 0.3,
    variation_same_facies: float = 0.01,
    mode_a_ore1_milling_rate: float = 3600.0,
    mode_a_ore2_milling_rate: float = 2400.0,
    mode_a_contingency_ore1_milling_rate: float = 3900.0,
    mode_b_ore1_milling_rate: float = 4600.0,
    mode_b_ore2_milling_rate: float = 800.0,
    mode_b_contingency_ore2_milling_rate: float = 2500.0,
    fleet_shift_duration: float = 0.5,
    total_lhd_count: float = 3.0,
    total_truck_count: float = 10.0,
    max_lhds_per_face: Sequence[float] = (2.0,),
    max_trucks_per_face: Sequence[float] = (6.0,),
    face_haul_distance: Sequence[float] = (1.5, 2.2),
    face_accessibility_fraction: Sequence[float] = (0.93, 0.91),
    truck_velocity: float = 15.0,
    loader_cycle_time_hours: float = 0.0833,
    truck_dump_time_hours: float = 0.033,
    traffic_delay_per_truck_hours: float = 0.015,
    fleet_mechanical_availability: float = 0.85,
    loader_payload_tonnes: float = 15.0,
    truck_payload_tonnes: float = 30.0,
    development_rate_per_extra_truck: float = 50.0,
    mode_allocations: Mapping = (),
):
    """Builds a mining simulation for arbitrary number of mine faces (N >= 1).

    Returns:
        tuple: ``(faces, fleet, plant, mode_controller, fleet_controller, ore1_stock, ore2_stock)``
    """
    if not faces:
        if face_generators:
            created_faces = []
            for i, gen in enumerate(face_generators, 1):
                created_faces.append(
                    MineFace(
                        name=f"mine_face_{i}",
                        face_id=i,
                        generator=gen,
                        min_ore_mass=min_ore_mass,
                        max_ore_mass=max_ore_mass,
                        total_ore_to_extract=total_ore_to_extract,
                        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
                        mean_ore_fraction=gen.mean_fraction,
                        std_dev_ore_fraction=gen.std_dev,
                        prob_new_facies=prob_new_facies,
                        variation_same_facies=variation_same_facies,
                        initial_parcel_mass=min_ore_mass,
                    )
                )
            faces = created_faces
        else:
            if face_mean_fractions:
                means = list(face_mean_fractions)
                num_faces = len(means)
            elif num_faces == 1:
                means = [mean_ore_fraction]
            elif num_faces == 2:
                means = [0.15, 0.45]
            else:
                step = (0.45 - 0.15) / (num_faces - 1)
                means = [0.15 + i * step for i in range(num_faces)]

            created_faces = []
            for i, mean_frac in enumerate(means, 1):
                std_dev = 0.075 if mean_frac <= 0.25 else 0.025
                gen = StochasticFaciesGenerator(
                    mean_fraction=mean_frac,
                    std_dev=std_dev,
                    prob_new_facies=prob_new_facies,
                    variation_same_facies=variation_same_facies,
                )
                created_faces.append(
                    MineFace(
                        name=f"mine_face_{i}" if num_faces > 1 else "mine_face",
                        face_id=i if num_faces > 1 else 1,
                        generator=gen,
                        min_ore_mass=min_ore_mass,
                        max_ore_mass=max_ore_mass,
                        total_ore_to_extract=total_ore_to_extract,
                        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
                        mean_ore_fraction=mean_frac,
                        std_dev_ore_fraction=std_dev,
                        prob_new_facies=prob_new_facies,
                        variation_same_facies=variation_same_facies,
                        initial_parcel_mass=min_ore_mass,
                    )
                )
            faces = created_faces
    else:
        faces = list(faces)

    fleet = ContinuousFleetLogistics()

    initial_fraction = mean_ore_fraction
    initial_mass1 = (1 - initial_fraction) * target_ore_stock_level
    ore1_stock = Stockpile(
        name="Ore1Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=initial_mass1,
        initial_attributes={
            "contained_ore_fraction_mass": initial_mass1 * initial_fraction
        },
        attr_inflow=1.0,
    )
    initial_mass2 = initial_fraction * target_ore_stock_level
    ore2_stock = Stockpile(
        name="Ore2Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=initial_mass2,
        initial_attributes={
            "contained_ore_fraction_mass": initial_mass2 * initial_fraction
        },
        attr_inflow=0.0,
    )

    plant = MetallurgicalPlant(
        stockpiles=[ore1_stock, ore2_stock],
        target_ore_stock_level=target_ore_stock_level,
        duration_of_contingency_segments=duration_of_contingency_segments,
        mode_a_ore1_milling_rate=mode_a_ore1_milling_rate,
        mode_a_ore2_milling_rate=mode_a_ore2_milling_rate,
        mode_a_contingency_ore1_milling_rate=mode_a_contingency_ore1_milling_rate,
        mode_b_ore1_milling_rate=mode_b_ore1_milling_rate,
        mode_b_ore2_milling_rate=mode_b_ore2_milling_rate,
        mode_b_contingency_ore2_milling_rate=mode_b_contingency_ore2_milling_rate,
    )

    mode_controller = mode_controller_cls(
        duration_of_production_campaigns=duration_of_production_campaigns,
        duration_of_shutdowns=duration_of_shutdowns,
        critical_ore2_level=critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
    )

    mode_rates = {
        "MODE_A": (mode_a_ore1_milling_rate, mode_a_ore2_milling_rate),
        "MODE_A_CONTINGENCY": (mode_a_contingency_ore1_milling_rate, 0.0),
        "MODE_B": (mode_b_ore1_milling_rate, mode_b_ore2_milling_rate),
        "MODE_B_CONTINGENCY": (0.0, mode_b_contingency_ore2_milling_rate),
    }

    fleet_controller = fleet_controller_cls(
        faces=faces,
        fleet_shift_duration=fleet_shift_duration,
        total_lhd_count=total_lhd_count,
        total_truck_count=total_truck_count,
        max_lhds_per_face=max_lhds_per_face,
        max_trucks_per_face=max_trucks_per_face,
        face_haul_distance=face_haul_distance,
        face_accessibility_fraction=face_accessibility_fraction,
        truck_velocity=truck_velocity,
        loader_cycle_time_hours=loader_cycle_time_hours,
        truck_dump_time_hours=truck_dump_time_hours,
        traffic_delay_per_truck_hours=traffic_delay_per_truck_hours,
        fleet_mechanical_availability=fleet_mechanical_availability,
        loader_payload_tonnes=loader_payload_tonnes,
        truck_payload_tonnes=truck_payload_tonnes,
        development_rate_per_extra_truck=development_rate_per_extra_truck,
        mode_allocations=mode_allocations,
        mode_rates=mode_rates,
    )

    return faces, fleet, plant, mode_controller, fleet_controller, ore1_stock, ore2_stock


create_blending_system = build_mining_simulation

"""Factory functions for building tactical DRS mining simulations."""

from typing import Sequence, Mapping, List
import math
from .stockpiles import Stockpile
from .mine_face import MineFace
from .plant import MetallurgicalPlant
from .controllers import OperatingModeController
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


def build_tactical_simulation(
    *,
    faces: Sequence[MineFace] = (),
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
):
    """Builds a tactical mining simulation with continuous DRS stepping.

    Returns ``(faces, plant, mode_controller, ore1_stock, ore2_stock)``.
    """
    if not faces:
        gen = StochasticFaciesGenerator(
            mean_fraction=mean_ore_fraction,
            std_dev=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
        )
        face = MineFace(
            name="mine_face",
            face_id=1,
            generator=gen,
            min_ore_mass=min_ore_mass,
            max_ore_mass=max_ore_mass,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            mean_ore_fraction=mean_ore_fraction,
            std_dev_ore_fraction=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
            initial_parcel_mass=40000.0,
        )
        faces = [face]

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

    mode_controller = OperatingModeController(
        duration_of_production_campaigns=duration_of_production_campaigns,
        duration_of_shutdowns=duration_of_shutdowns,
        critical_ore2_level=critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
    )

    return faces, plant, mode_controller, ore1_stock, ore2_stock

import sys
import os

# Ensure the root directory is on the path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import random
import numpy as np
import matplotlib.pyplot as plt

from drs import DRSEngine, Telemetry
from drs_mining.components import (
    MineFace,
    StochasticReserve,
    HaulRoute,
    MetallurgicalPlant,
    MillingSetpoints,
    OperatingModeController,
    Stockpile,
    StochasticFaciesGenerator,
    TacticalMiningSimulation,
)
from drs_mining.config import MILL_MODES, SimulationConfig
from drs_mining.components.plot import (
    plot_single_face_dashboard,
    prepare_history,
    print_deficit_by_mode,
    print_transition_log,
)


def build_blending_network(
    mean_ore_fraction: float = 0.30,
    std_dev_ore_fraction: float = 0.05,
    prob_new_facies: float = 0.3,
    variation_same_facies: float = 0.01,
    min_ore_mass: float = 30000.0,
    max_ore_mass: float = 50000.0,
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    target_ore_stock_level: float = 60000.0,
    critical_ore2_level: float = 20400.0,
    duration_of_production_campaigns: float = 34.0,
    duration_of_shutdowns: float = 1.0,
    duration_of_contingency_segments: float = 1.0,
    **kwargs,
) -> tuple:
    gen = StochasticFaciesGenerator(
        mean_fraction=mean_ore_fraction,
        std_dev=std_dev_ore_fraction,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
    )
    geology = StochasticReserve(
        name="mine_face_reserve",
        total_tonnes=total_ore_to_extract,
        generator=gen,
        min_parcel_mass=min_ore_mass,
        max_parcel_mass=max_ore_mass,
        initial_parcel_mass=40000.0,
        warming_period=ore_to_be_extracted_during_warming_period,
    )
    haulage = HaulRoute(distance_km=1.0)
    mine = MineFace(
        name="mine_face",
        geology=geology,
        haulage=haulage,
    )


    initial_mass1 = (1 - mean_ore_fraction) * target_ore_stock_level
    ore1_stock = Stockpile(
        name="Ore1Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=initial_mass1,
        initial_attributes={
            "contained_ore_fraction_mass": initial_mass1 * mean_ore_fraction
        },
        attr_inflow=1.0,
    )
    initial_mass2 = mean_ore_fraction * target_ore_stock_level
    ore2_stock = Stockpile(
        name="Ore2Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=initial_mass2,
        initial_attributes={
            "contained_ore_fraction_mass": initial_mass2 * mean_ore_fraction
        },
        attr_inflow=0.0,
    )

    plant = MetallurgicalPlant(
        stockpiles=[ore1_stock, ore2_stock],
        setpoints=MillingSetpoints(
            mode_a_ore1=3600.0,
            mode_a_ore2=2400.0,
            mode_a_contingency_ore1=3900.0,
            mode_b_ore1=4600.0,
            mode_b_ore2=800.0,
            mode_b_contingency_ore2=2500.0,
        ),
        target_ore_stock_level=target_ore_stock_level,
        duration_of_contingency_segments=duration_of_contingency_segments,
    )

    mode_controller = OperatingModeController(
        duration_of_production_campaigns=duration_of_production_campaigns,
        duration_of_shutdowns=duration_of_shutdowns,
        critical_ore2_level=critical_ore2_level,
    )

    return mine, plant, mode_controller, ore1_stock, ore2_stock


def _register_and_policy(engine, network):
    mine, plant, mode_ctrl, ore1_stock, ore2_stock = network
    engine.register(mine, plant, mode_ctrl, ore1_stock, ore2_stock)

    @engine.on_step
    def manage_blending(t: float):
        mode = mode_ctrl.update(ore2_stock.level)

        ore2_frac = (
            mine.geology.active_parcel.ore2_fraction
            if mine.geology.active_parcel
            else 0.0
        )
        ore1_rate, ore2_rate, mine_target = plant.get_target_rates(
            mode,
            ore1_level=ore1_stock.level,
            ore2_level=ore2_stock.level,
            stockpile2_routing_fraction=ore2_frac,
        )

        mine.target_rate = mine_target

        ore1_frac = 1.0 - ore2_frac
        actual = mine.actual_rate
        ore1_in = actual * ore1_frac
        ore2_in = actual * ore2_frac

        out1 = ore1_stock.feed_and_draw(ore1_in, ore1_rate)
        out2 = ore2_stock.feed_and_draw(ore2_in, ore2_rate)
        plant.process(out1 + out2)


def print_statistics(plant, mine):
    print("\n--- Output Statistics ---")
    total_time = plant.total_duration

    if total_time > 0:
        for attr, label in [
            ("cumulative_time_mode_a", "PortionOfTimeInModeA"),
            ("cumulative_time_mode_a_contingency", "PortionOfTimeInModeAContingency"),
            ("cumulative_time_mode_a_surging", "PortionOfTimeInModeAMineSurging"),
            ("cumulative_time_mode_b", "PortionOfTimeInModeB"),
            ("cumulative_time_mode_b_contingency", "PortionOfTimeInModeBContingency"),
            ("cumulative_time_mode_b_surging", "PortionOfTimeInModeBMineSurging"),
            ("cumulative_time_shutdown", "PortionOfTimeInShutdown"),
        ]:
            print(f"{label}: {getattr(plant, attr).value / total_time:.4f}")
    else:
        print("Total time is 0. Cannot calculate mode portions.")

    active_time = plant.active_duration(total_time)
    if active_time > 0:
        total_ore_processed = plant.cumulative_milled_mass.value
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(11)
    random.seed(11)
    network = build_blending_network(
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
        prob_new_facies=0.3,
    )
    mine, plant, mode_ctrl, ore1_stock, ore2_stock = network

    mode_ctrl.active_campaign_mode.value = MILL_MODES["MODE_A"]

    engine = DRSEngine()
    _register_and_policy(engine, network)

    mine.geology.total_tonnes = 600000.0
    warmup_result = engine.run(until=99999.0)

    plant.reset_mode_timers()

    mine.geology.total_tonnes = 6600000.0
    telemetry = Telemetry(model=engine)
    telemetry.register_metric(
        "active_operating_mode",
        lambda t, m, s, _: plant.active_operating_mode.value,
    )
    telemetry.register_metric(
        "MassOfCurrentParcel",
        lambda t, m, s, _: (
            mine.geology.active_parcel.mass if mine.geology.active_parcel else 0.0
        ),
    )
    telemetry.register_metric(
        "CurrentParcelOreFraction",
        lambda t, m, s, _: (
            mine.geology.active_parcel.ore2_fraction
            if mine.geology.active_parcel
            else 0.0
        ),
    )
    telemetry.register_metric(
        "Campaign_Shutdown",
        lambda t, m, s, _: mode_ctrl.current_campaign_duration.value,
    )
    telemetry.register_metric(
        "Contingency",
        lambda t, m, s, _: plant.current_contingency_duration.value,
    )
    engine.attach_telemetry(telemetry)
    result = engine.run(until=99999.0)

    print(result.summary())

    print_statistics(plant, mine)

    df = prepare_history(result.history)
    print_transition_log(
        df,
        critical_ore2_level=20400.0,
        target_ore_stock_level=args.total_stockpile_level,
    )

    print_deficit_by_mode(
        df,
        extraction_cols=["cumulative_extracted_mass"],
        ideal_rate=6000.0,
    )

    plot_single_face_dashboard(df)

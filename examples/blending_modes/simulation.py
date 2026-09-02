import sys
import os

# Ensure the root directory is on the path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import random
import numpy as np
import matplotlib.pyplot as plt

import drs
from drs import DRSEngine, Telemetry, Processor
from drs_mining.components import (
    Flow,
    blend_flows,
    StochasticReserve,
    OperatingModeController,
    Stockpile,
    StochasticFaciesGenerator,
)
from drs_mining.config import MILL_MODES
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
        attribute_name="ore2_fraction",
    )
    reserve = StochasticReserve(
        name="mine_face_reserve",
        total_tonnes=total_ore_to_extract,
        generator=gen,
        min_parcel_mass=min_ore_mass,
        max_parcel_mass=max_ore_mass,
        initial_parcel_mass=40000.0,
        warming_period=ore_to_be_extracted_during_warming_period,
    )

    initial_mass1 = (1 - mean_ore_fraction) * target_ore_stock_level
    ore1_stock = Stockpile(
        name="Ore1Stock",
        initial_mass=initial_mass1,
        initial_attributes={"ore2_fraction": 0.0},
    )
    initial_mass2 = mean_ore_fraction * target_ore_stock_level
    ore2_stock = Stockpile(
        name="Ore2Stock",
        initial_mass=initial_mass2,
        initial_attributes={"ore2_fraction": 1.0},
    )

    mill = Processor(name="mill", max_rate=6000.0)

    mode_controller = OperatingModeController(
        duration_of_production_campaigns=duration_of_production_campaigns,
        duration_of_shutdowns=duration_of_shutdowns,
        duration_of_contingency_segments=duration_of_contingency_segments,
        critical_ore2_level=critical_ore2_level,
        target_total_stock=target_ore_stock_level,
    )

    return reserve, mill, mode_controller, ore1_stock, ore2_stock


def _register_and_policy(engine: DRSEngine, network: tuple) -> drs.Level:
    reserve, mill, mode_ctrl, ore1_stock, ore2_stock = network
    cumulative_milled_mass = drs.Level("cumulative_milled_mass", initial_value=0.0, owner=mill)
    mill.cumulative_milled_mass = cumulative_milled_mass
    engine.register(reserve, mill, mode_ctrl, ore1_stock, ore2_stock)

    @engine.on_step
    def manage_blending(t: float):
        campaign_mode = mode_ctrl.update_campaign(ore2_stock.level)
        active_mode = mode_ctrl.resolve_operating_mode(
            campaign_mode,
            ore1_level=ore1_stock.level,
            ore2_level=ore2_stock.level,
        )

        draw_rates = mode_ctrl.get_draw_rates(active_mode)
        ore1_target = draw_rates.get("Ore1Stock", 0.0)
        ore2_target = draw_rates.get("Ore2Stock", 0.0)

        ore2_frac = reserve.current_attributes.get("ore2_fraction", 0.0)
        mode_name = active_mode.name
        if "_MINE_SURGING" in mode_name:
            if mode_name == "MODE_A_MINE_SURGING":
                effective_fraction = max(1.0 - ore2_frac, 0.01)
                mine_target = ore1_target / effective_fraction
            else:
                effective_fraction = max(ore2_frac, 0.01)
                mine_target = ore2_target / effective_fraction
        else:
            mine_target = ore1_target + ore2_target

        extraction_flow = reserve.extract(mine_target)

        in_flow1 = Flow(
            rate=extraction_flow.rate * (1.0 - ore2_frac),
            attributes=extraction_flow.attributes,
        )
        in_flow2 = Flow(
            rate=extraction_flow.rate * ore2_frac,
            attributes=extraction_flow.attributes,
        )

        out1 = ore1_stock.feed_and_draw(in_flow1, ore1_target)
        out2 = ore2_stock.feed_and_draw(in_flow2, ore2_target)

        blended_feed = blend_flows([out1, out2])
        mill.rate = blended_feed.rate
        cumulative_milled_mass.rate = mill.actual_rate

    return cumulative_milled_mass


def print_statistics(mode_ctrl: OperatingModeController, cumulative_milled_mass: drs.Level, total_time: float):
    print("\n--- Output Statistics ---")
    if total_time > 0:
        for mode_name, label in [
            ("MODE_A", "PortionOfTimeInModeA"),
            ("MODE_A_CONTINGENCY", "PortionOfTimeInModeAContingency"),
            ("MODE_A_MINE_SURGING", "PortionOfTimeInModeAMineSurging"),
            ("MODE_B", "PortionOfTimeInModeB"),
            ("MODE_B_CONTINGENCY", "PortionOfTimeInModeBContingency"),
            ("MODE_B_MINE_SURGING", "PortionOfTimeInModeBMineSurging"),
            ("SHUTDOWN", "PortionOfTimeInShutdown"),
        ]:
            timer_val = mode_ctrl.mode_timers[mode_name].value
            print(f"{label}: {timer_val / total_time:.4f}")
    else:
        print("Total time is 0. Cannot calculate mode portions.")

    shutdown_time = mode_ctrl.mode_timers["SHUTDOWN"].value
    active_time = max(0.0, total_time - shutdown_time)
    if active_time > 0:
        total_ore_processed = cumulative_milled_mass.value
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
    reserve, mill, mode_ctrl, ore1_stock, ore2_stock = network

    mode_ctrl.active_campaign_mode.value = MILL_MODES["MODE_A"]

    engine = DRSEngine()
    cumulative_milled_mass = _register_and_policy(engine, network)

    reserve.total_tonnes = 600000.0
    warmup_result = engine.run(until=99999.0)

    mode_ctrl.reset_mode_timers()

    reserve.total_tonnes = 6600000.0
    telemetry = Telemetry(model=engine)
    telemetry.register_metric(
        "active_operating_mode",
        lambda t, m, s, _: mode_ctrl.active_operating_mode.value,
    )
    telemetry.register_metric(
        "MassOfCurrentParcel",
        lambda t, m, s, _: (
            reserve.active_entity.mass if reserve.active_entity else 0.0
        ),
    )
    telemetry.register_metric(
        "CurrentParcelOreFraction",
        lambda t, m, s, _: reserve.current_attributes.get("ore2_fraction", 0.0),
    )
    telemetry.register_metric(
        "Campaign_Shutdown",
        lambda t, m, s, _: mode_ctrl.current_campaign_duration.value,
    )
    telemetry.register_metric(
        "Contingency",
        lambda t, m, s, _: mode_ctrl.current_contingency_duration.value,
    )
    engine.attach_telemetry(telemetry)
    result = engine.run(until=99999.0)

    print(result.summary())

    total_time = engine.current_time
    print_statistics(mode_ctrl, cumulative_milled_mass, total_time)

    df = prepare_history(result.history)
    print_transition_log(
        df,
        critical_ore2_level=20400.0,
        target_ore_stock_level=args.total_stockpile_level,
    )

    print_deficit_by_mode(
        df,
        extraction_cols=["mine_face_reserve_cumulative_extracted_mass"],
        ideal_rate=6000.0,
    )

    plot_single_face_dashboard(df)

import argparse
from typing import Optional

import pandas as pd

import drs
from drs import (
    DRSEngine,
    Telemetry,
    Storage,
    Processor,
    Flow,
    blend_flows,
)
from drs_mining.components import (
    OperatingModeController,
    MaterialSource,
    autocorrelated_generator,
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
    stream = autocorrelated_generator(
        mean_fraction=mean_ore_fraction,
        std_dev=std_dev_ore_fraction,
        prob_new_facies=prob_new_facies,
        variation_step=variation_same_facies,
        min_mass=min_ore_mass,
        max_mass=max_ore_mass,
        initial_mass=40000.0,
        attribute_name="ore2_fraction",
    )
    source = MaterialSource(
        name="mine_face_reserve",
        total_tonnes=total_ore_to_extract,
        stream=stream,
        warming_period=ore_to_be_extracted_during_warming_period,
    )

    initial_mass1 = (1.0 - mean_ore_fraction) * target_ore_stock_level
    ore1_stock = Storage(
        name="Ore1Stock",
        initial_level=initial_mass1,
        initial_attributes={"ore2_fraction": 0.0},
    )
    initial_mass2 = mean_ore_fraction * target_ore_stock_level
    ore2_stock = Storage(
        name="Ore2Stock",
        initial_level=initial_mass2,
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

    return source, mill, mode_controller, ore1_stock, ore2_stock


def _register_and_policy(engine: DRSEngine, network: tuple) -> drs.Level:
    source, mill, mode_ctrl, ore1_stock, ore2_stock = network
    cumulative_milled_mass = drs.Level("cumulative_milled_mass", initial_value=0.0, owner=mill)
    mill.cumulative_milled_mass = cumulative_milled_mass
    engine.register(source, mill, mode_ctrl, ore1_stock, ore2_stock)

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

        ore2_frac = source.current_attributes.get("ore2_fraction", 0.0)
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

        extraction_flow = source.extract(mine_target)

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
        mill.process(blended_feed)
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
            ("SHUTDOWN", "PortionOfTimeInShutdown"),
        ]:
            timer = mode_ctrl.mode_timers.get(mode_name)
            fraction = (timer.value / total_time) if timer else 0.0
            print(f"{label}: {fraction:.4f}")
    print(f"CumulativeTonsMilled: {cumulative_milled_mass.value:.2f}")


def run_blending_simulation(
    replication_length: float = 999999.0,
    plot: bool = False,
    seed: int = 11,
    **kwargs,
) -> tuple[drs.SimResult, Optional[pd.DataFrame]]:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    network = build_blending_network(**kwargs)
    source, mill, mode_ctrl, ore1_stock, ore2_stock = network

    mode_ctrl.active_campaign_mode.value = MILL_MODES["MODE_A"]

    engine = DRSEngine()
    cumulative_milled_mass = _register_and_policy(engine, network)

    # Warmup phase (pre-burns 600,000 tonnes)
    source.total_tonnes = 600000.0
    warmup_result = engine.run(until=99999.0)

    mode_ctrl.reset_mode_timers()

    # Production phase (6,600,000 tonnes)
    source.total_tonnes = 6600000.0

    telemetry = None
    if plot:
        telemetry = Telemetry(model=engine)
        telemetry.register_metric("active_operating_mode", lambda t, m, s, _: mode_ctrl.active_operating_mode.value.name)
        telemetry.register_metric("active_operating_mode_name", lambda t, m, s, _: mode_ctrl.active_operating_mode.value.name)
        telemetry.register_metric("total_stock_level", lambda t, m, s, _: ore1_stock.level + ore2_stock.level)
        telemetry.register_metric(
            "MassOfCurrentParcel",
            lambda t, m, s, _: source.active_entity.mass if source.active_entity else 0.0,
        )
        telemetry.register_metric(
            "CurrentParcelRoutingFraction",
            lambda t, m, s, _: source.current_attributes.get("ore2_fraction", 0.0),
        )
        telemetry.register_metric(
            "CurrentParcelOreFraction",
            lambda t, m, s, _: source.current_attributes.get("ore2_fraction", 0.0),
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

    result = engine.run(until=replication_length)

    print_statistics(mode_ctrl, cumulative_milled_mass, result.duration)

    history = None
    if plot and telemetry is not None:
        history = telemetry.to_dataframe()
        history = prepare_history(history)
        print_transition_log(history)
        print_deficit_by_mode(history, extraction_cols=["mine_face_reserve_cumulative_extracted_mass"])
        plot_single_face_dashboard(history)

    return result, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tactical Blending Simulation")
    parser.add_argument("--days", type=float, default=999999.0, help="Simulation duration (days)")
    parser.add_argument("--plot", action="store_true", default=True, help="Display telemetry dashboard")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plotting")
    args = parser.parse_args()

    run_blending_simulation(replication_length=args.days, plot=args.plot)

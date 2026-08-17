import sys
import os

# Ensure the root directory is on the path so we can import 'examples.mining'
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import types

import drs
from drs import DRSEngine, Telemetry
from drs_mining.components import (
    ConcentratorMineFace,
    ConcentratorPlant,
    ConcentratorController,
    ContinuousFleetLogistics,
    Stockpile,
)
from drs_mining.components.modes import MODES


# ============================================================
# Inline control policy
# ------------------------------------------------------------
# A verbatim copy of the single-face path of ``drs_mining.control``
# adapted to the plain local entities built below. Operating-mode
# bookkeeping lives on the controller (``step_mode``), extraction
# mechanics on the mine (``set_extraction_rate``), and stockpile
# balancing on the stockpiles (``set_inout``); this policy owns the
# target-rate calculations, routing, and calling order.
# ============================================================

_RATE_MAP = {
    "MODE_A": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
    "MODE_A_CONTINGENCY": ("mode_a_contingency_ore1_milling_rate", None),
    "MODE_A_MINE_SURGING": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
    "MODE_B": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
    "MODE_B_CONTINGENCY": (None, "mode_b_contingency_ore2_milling_rate"),
    "MODE_B_MINE_SURGING": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
    "SHUTDOWN": (None, None),
}

DEFAULT_RATES = {
    "mode_a_ore1_milling_rate": 3600.0,
    "mode_a_ore2_milling_rate": 2400.0,
    "mode_a_contingency_ore1_milling_rate": 3900.0,
    "mode_b_ore1_milling_rate": 4600.0,
    "mode_b_ore2_milling_rate": 800.0,
    "mode_b_contingency_ore2_milling_rate": 2500.0,
}


def blending_step_policy(
    time,
    mine,
    fleet,
    controller,
    ore1_stock,
    ore2_stock,
    plant,
    global_time,
):
    """Top-level blending policy invoked by the engine once per step."""
    global_time.rate = 1.0
    ctrl = controller

    ctrl.step_mode(ore1_stock, ore2_stock)
    _target_rates(controller, ore1_stock, ore2_stock, fleet, mine)

    if mine is not None:
        mine.set_extraction_rate(ctrl.target_mine_mass_rate.value)

    ore1_rate, ore2_rate = _route([mine], fleet)
    _balance_stockpiles(
        controller, plant, ore1_stock, ore2_stock, ore1_rate, ore2_rate
    )

    ctrl.total_system_ore_mass.rate = (
        ore1_stock.current_mass.rate + ore2_stock.current_mass.rate
    )


def _route(mining_sources, fleet):
    ore1 = ore2 = total = 0.0
    for src in mining_sources:
        rate = src.actual_rate
        ore2_frac = src._get_current_attr_value()
        ore2 += rate * ore2_frac
        ore1 += rate * (1.0 - ore2_frac)
        total += rate
    if total > 1e-6 and fleet is not None:
        fleet.stockpile2_routing_fraction.value = ore2 / total
    return ore1, ore2


def _balance_stockpiles(
    controller, plant, ore1_stock, ore2_stock, ore1_rate, ore2_rate
):
    ctrl = controller
    out1 = ore1_stock.set_inout(
        ore1_rate, ctrl.target_stock1_outflow_rate.value, attr_inflow=1.0
    )
    out2 = ore2_stock.set_inout(
        ore2_rate, ctrl.target_stock2_outflow_rate.value, attr_inflow=0.0
    )
    if plant is not None:
        plant.target_rate = out1 + out2
        plant.cumulative_milled_mass.rate = plant.actual_rate


def _target_rates(controller, ore1_stock, ore2_stock, fleet, mine):
    ctrl = controller
    name = ctrl.active_operating_mode.value.name

    ore1, ore2 = _read_rates(name, ctrl)

    if "_MINE_SURGING" in name:
        target_stock = getattr(ctrl, "target_ore_stock_level", 60000.0)
        ctrl.total_system_ore_mass.lower_threshold = target_stock
        p = fleet.stockpile2_routing_fraction.value if fleet is not None else 0.0
        if p <= 1e-4 and mine is not None and hasattr(mine, "_get_current_attr_value"):
            p = mine._get_current_attr_value()
        if name == "MODE_A_MINE_SURGING":
            effective_fraction = max(1.0 - p, 0.01)
            extraction = ore1 / effective_fraction
        else:
            effective_fraction = max(p, 0.01)
            extraction = ore2 / effective_fraction
    else:
        extraction = ore1 + ore2

    ctrl.target_mine_mass_rate.value = extraction
    ctrl.target_stock1_outflow_rate.value = ore1
    ctrl.target_stock2_outflow_rate.value = ore2


def _read_rates(name, obj):
    ore1_attr, ore2_attr = _RATE_MAP.get(name, (None, None))
    ore1 = (
        getattr(obj, ore1_attr, DEFAULT_RATES.get(ore1_attr, 0.0))
        if ore1_attr
        else 0.0
    )
    ore2 = (
        getattr(obj, ore2_attr, DEFAULT_RATES.get(ore2_attr, 0.0))
        if ore2_attr
        else 0.0
    )
    return ore1, ore2


# ============================================================
# Entity construction
# ============================================================


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
    ore1_capacity: float = float("inf"),
    ore2_capacity: float = float("inf"),
    plant_max_rate: float = float("inf"),
    **kwargs,
) -> tuple:
    """Build the blending network flat: every entity is constructed here.

    Returns ``(mine, fleet, plant, controller, ore1_stock, ore2_stock, global_time)``.
    There is no scenario container; the engine registers the controller's
    state components plus the residual leaves below.
    """
    mine = ConcentratorMineFace(
        mean_ore_fraction=mean_ore_fraction,
        std_dev_ore_fraction=std_dev_ore_fraction,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
        min_ore_mass=min_ore_mass,
        max_ore_mass=max_ore_mass,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
    )
    fleet = ContinuousFleetLogistics()

    initial_mass1 = (1 - mean_ore_fraction) * target_ore_stock_level
    ore1_stock = Stockpile(
        name="Ore1Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=initial_mass1,
        initial_attributes={
            "contained_ore_fraction_mass": initial_mass1 * mean_ore_fraction
        },
        capacity=ore1_capacity,
    )
    initial_mass2 = mean_ore_fraction * target_ore_stock_level
    ore2_stock = Stockpile(
        name="Ore2Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=initial_mass2,
        initial_attributes={
            "contained_ore_fraction_mass": initial_mass2 * mean_ore_fraction
        },
        capacity=ore2_capacity,
    )

    plant = ConcentratorPlant(
        mine, fleet, ore1_stock, ore2_stock, max_rate=plant_max_rate
    )
    controller = ConcentratorController(
        mine=mine,
        fleet=fleet,
        plant=plant,
        target_ore_stock_level=target_ore_stock_level,
        critical_ore2_level=critical_ore2_level,
        duration_of_production_campaigns=duration_of_production_campaigns,
        duration_of_shutdowns=duration_of_shutdowns,
        duration_of_contingency_segments=duration_of_contingency_segments,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        total_ore_to_extract=total_ore_to_extract,
    )
    global_time = drs.Timer("GlobalTime", initial_value=0.0)

    return mine, fleet, plant, controller, ore1_stock, ore2_stock, global_time


def _register(engine, mine, fleet, plant, controller, ore1_stock, ore2_stock, global_time):
    """Register the stateful leaves and wire the inline policy onto the engine."""
    engine.register(
        *controller.state_components,
        plant,
        ore1_stock,
        ore2_stock,
        global_time,
    )

    @engine.on_step
    def _policy(time):
        blending_step_policy(
            time, mine, fleet, controller, ore1_stock, ore2_stock, plant, global_time
        )


def print_statistics(controller, plant, mine):
    """Print operating-mode time-shares and throughput for a blending network."""
    print("\n--- Output Statistics ---")
    total_time = controller.total_duration

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
            print(
                f"{label}: {getattr(controller, attr).value / total_time:.4f}"
            )
    else:
        print("Total time is 0. Cannot calculate mode portions.")

    active_time = controller.active_duration(total_time)
    if active_time > 0:
        if hasattr(plant, "cumulative_milled_mass"):
            total_ore_processed = plant.cumulative_milled_mass.value
        else:
            total_ore_processed = mine.net_extracted_mass
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


def evaluate_throughput(config_kwargs: dict, N: int) -> tuple[float, float]:
    """Runs the simulation N times, extracting throughputs.
    Returns (mean_throughput, std_dev_throughput).
    """
    throughputs = []

    for idx in range(N):
        np.random.seed(idx)
        random.seed(idx)

        mine, fleet, plant, controller, ore1_stock, ore2_stock, global_time = (
            build_blending_network(**config_kwargs)
        )

        engine = DRSEngine()
        _register(engine, mine, fleet, plant, controller, ore1_stock, ore2_stock, global_time)
        engine.run(until=config_kwargs.get("replication_length", float("inf")))

        active_time = controller.active_duration(engine.current_time)
        if active_time > 0:
            throughput = mine.net_extracted_mass / active_time
            throughputs.append(throughput)

    if not throughputs:
        return 0.0, 0.0
    return float(np.mean(throughputs)), float(np.std(throughputs))


def plot_monte_carlo_throughput(N: int = 1, total_stockpile_level: float = 60000.0):
    sigmas = [5.0]
    results = []

    print(f"\n--- Running Monte Carlo Evaluation for Standard (N={N}) ---")
    for sigma in sigmas:
        config_kwargs = dict(
            replication_length=99999.0,
            std_dev_ore_fraction=sigma / 100.0,
            target_ore_stock_level=total_stockpile_level,
            prob_new_facies=0.3,
        )
        mean, std = evaluate_throughput(config_kwargs, N)
        results.append((sigma, mean, std))
        print(f"Sigma: {sigma}%, Mean Throughput: {mean:.2f}, Std Dev: {std:.2f}")

    # Plot results to match Figure 5 from the paper
    means = [r[1] for r in results]
    stds = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sigmas,
        means,
        yerr=stds,
        fmt="-o",
        capsize=5,
        capthick=2,
        ecolor="black",
        markerfacecolor="blue",
        markeredgecolor="blue",
        color="gray",
    )

    plt.title(
        f"Expected Simulated Throughput by Geological Uncertainty (Standard, N={N})",
        fontsize=14,
    )
    plt.xlabel("Sigma geo (%)", fontsize=12)
    plt.ylabel("Mean Campaign Throughput (t/d)", fontsize=12)
    plt.ylim(5500, 6000)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(
        "plots/Monte_Carlo_Throughput_Fig5_Standard.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved 'plots/Monte_Carlo_Throughput_Fig5_Standard.png'.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(11)
    random.seed(11)
    mine, fleet, plant, controller, ore1_stock, ore2_stock, global_time = (
        build_blending_network(
            replication_length=99999.0,
            target_ore_stock_level=args.total_stockpile_level,
            std_dev_ore_fraction=args.std_dev_ore_fraction,
            prob_new_facies=0.3,
        )
    )

    controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine()
    _register(engine, mine, fleet, plant, controller, ore1_stock, ore2_stock, global_time)

    telemetry = Telemetry(model=engine)
    telemetry.register_metric(
        "MassOfCurrentParcel",
        lambda t, m, s, _: mine.active_parcel_initial_mass.value,
    )
    telemetry.register_metric(
        "CurrentParcelRoutingFraction",
        lambda t, m, s, _: fleet.stockpile2_routing_fraction.value,
    )
    telemetry.register_metric(
        "Campaign_Shutdown",
        lambda t, m, s, _: controller.current_campaign_duration.value,
    )
    telemetry.register_metric(
        "Contingency",
        lambda t, m, s, _: controller.current_contingency_duration.value,
    )
    engine.attach_telemetry(telemetry)
    result = engine.run(until=99999.0)

    print(result.summary())

    print_statistics(controller, plant, mine)

    df = result.history

    # --- Mode Transition Log ---
    state_change_events = [
        e
        for e in result.events
        if e.event_type == "STATE_CHANGE"
        and e.details.get("variable") == "active_operating_mode"
    ]
    if state_change_events:
        print("\n--- Mode Transition Log ---")
        for e in state_change_events:
            old = (
                e.details["old_value"].name
                if hasattr(e.details["old_value"], "name")
                else str(e.details["old_value"])
            )
            new = (
                e.details["new_value"].name
                if hasattr(e.details["new_value"], "name")
                else str(e.details["new_value"])
            )
            print(f"Time: {e.time:.2f} | Transition: {old} -> {new}")
        print("---------------------------\n")

    # --- Cumulative Deficit by Mode Log ---
    import pandas as pd

    dt = df["time"].diff().fillna(0)
    actual_extraction_step = df["cumulative_extracted_mass"].diff().fillna(0)
    ideal_extraction_step = dt * 6000.0
    step_deficit = (ideal_extraction_step - actual_extraction_step).clip(lower=0)

    # We still need the active_operating_mode_name column for downstream plotting and analysis
    df["active_operating_mode_name"] = df["active_operating_mode"].apply(
        lambda x: x.name if x else "None"
    )

    deficit_df = pd.DataFrame(
        {"mode": df["active_operating_mode_name"], "deficit": step_deficit}
    )

    total_deficit_by_mode = (
        deficit_df.groupby("mode")["deficit"].sum().sort_values(ascending=False)
    )

    print("\n--- Cumulative Lost Production (Deficit) by Mode ---")
    total_lost = total_deficit_by_mode.sum()
    for mode, lost in total_deficit_by_mode.items():
        mode_name = str(mode).split(".")[-1]
        pct = (lost / total_lost * 100) if total_lost > 0 else 0
        print(f"{mode_name}: {lost:.1f} tons ({pct:.1f}%)")
    print(f"TOTAL: {total_lost:.1f} tons")
    print("----------------------------------------------------\n")

    # Create Modes Series
    df["Mode A"] = df["active_operating_mode_name"].apply(
        lambda m: (
            3
            if m
            in (
                "MODE_A",
                "MODE_A_CONTINGENCY",
                "MODE_A_MINE_SURGING",
            )
            else 0
        )
    )
    df["Mode B"] = df["active_operating_mode_name"].apply(
        lambda m: (
            2
            if m
            in (
                "MODE_B",
                "MODE_B_CONTINGENCY",
                "MODE_B_MINE_SURGING",
            )
            else 0
        )
    )
    df["Shutdown"] = df["active_operating_mode_name"].apply(
        lambda m: 1 if m == "SHUTDOWN" else 0
    )

    # Create Ore Level Series (scaled by 1000)
    df["Total Ore Stockpile Level"] = df["total_system_ore_mass"] / 1000.0
    df["Ore 1 Stockpile Level"] = df["Ore1Stock_mass"] / 1000.0
    df["Ore 2 Stockpile Level"] = df["Ore2Stock_mass"] / 1000.0

    from drs.plot import (
        plot_time_series,
        plot_dual_axis_step,
        plot_safety_margin,
        Dashboard,
    )
    from drs_mining.components.plot import (
        plot_ore_with_modes,
        plot_mode_distribution,
        plot_mode_dwell_times,
        plot_normalized_deviation_violin,
        plot_attributed_deficit,
        plot_deficit_disparity,
        plot_deficit_breakdown_bar,
        plot_structural_vs_operational_deficit,
        plot_normalized_cumulative_deficit,
        plot_structural_vs_operational_by_mode,
    )

    palette = {
        "MODE_A": "#1f77b4",
        "MODE_A_CONTINGENCY": "#2ca02c",
        "MODE_A_MINE_SURGING": "#9467bd",
        "MODE_B": "#d62728",
        "MODE_B_CONTINGENCY": "#ff7f0e",
        "MODE_B_MINE_SURGING": "#8c564b",
        "SHUTDOWN": "#FFD700",
    }

    structural_modes = ["SHUTDOWN", "MODE_A"]

    dash = Dashboard(
        nrows=14, ncols=1, figsize=(18, 69), sharex=False, title="Comprehensive Mine Diagnostics"
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 8, 11, 12])

    plot_time_series(
        df,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Modes (Step)",
        is_step=True,
        ax=dash[0],
    )
    plot_ore_with_modes(
        df,
        time_col="time",
        ore_cols=[
            "total_system_ore_mass",
            "Ore1Stock_mass",
            "Ore2Stock_mass",
        ],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Ore Stockpiles & Campaigns",
        palette=palette,
        hlines=[
            {
                "y": 60000,
                "color": "black",
                "linestyle": "--",
                "linewidth": 1.5,
                "alpha": 0.7,
                "label": "Target Total (60k)",
            },
            {
                "y": 20400,
                "color": "red",
                "linestyle": ":",
                "linewidth": 2,
                "alpha": 0.8,
                "label": "Critical Ore 2 (20.4k)",
            },
        ],
        ax=dash[1],
    )
    plot_dual_axis_step(
        df,
        y1_col="MassOfCurrentParcel",
        y2_col="CurrentParcelRoutingFraction",
        y1_label="Parcel Mass (tons)",
        y2_label="Grade (% Ore 2)",
        title="Current Parcel Properties",
        ax=dash[2],
    )
    plot_safety_margin(
        df,
        level_col="Ore1Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 1 Distance to Floor",
        danger_threshold=1000.0,
        ax=dash[3],
    )
    plot_safety_margin(
        df,
        level_col="Ore2Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 2 Distance to Floor",
        danger_threshold=1000.0,
        ax=dash[4],
    )
    plot_mode_distribution(
        df,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% of Time Spent)",
        palette=palette,
        ax=dash[5],
    )
    plot_mode_dwell_times(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        title="Mode Stability (Dwell Times)",
        ax=dash[6],
    )
    plot_normalized_deviation_violin(
        df,
        title="Stockpile Deviation Variance (Violin)",
        target_total=60000.0,
        target_ore1=42000.0,
        target_ore2=18000.0,
        ax=dash[7],
    )
    plot_attributed_deficit(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        extraction_col="cumulative_extracted_mass",
        ideal_rate_per_day=6000.0,
        title="Cumulative Production Deficit by Mode",
        palette=palette,
        ax=dash[8],
    )
    plot_deficit_disparity(
        df,
        mode_col="active_operating_mode_name",
        title="Mode Efficiency (Time Spent vs. Deficit Caused)",
        ideal_rate=6000.0,
        ax=dash[9],
    )
    plot_deficit_breakdown_bar(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[10],
    )
    plot_structural_vs_operational_deficit(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate=6000.0,
        structural_modes=structural_modes,
        ax=dash[11],
    )
    plot_normalized_cumulative_deficit(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[12],
    )
    plot_structural_vs_operational_by_mode(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate=6000.0,
        structural_modes=structural_modes,
        ax=dash[13],
    )

    dash.save("plots/Comprehensive_Diagnostics_Plot.png")

    # Recreate Figure 5 from paper
    plot_monte_carlo_throughput(
        N=args.N, total_stockpile_level=args.total_stockpile_level
    )
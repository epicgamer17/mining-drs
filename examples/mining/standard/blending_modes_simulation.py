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

from examples.mining.components import ConcentratorConfig, ConcentratorModel
from drs import DRSEngine


def evaluate_throughput(config: ConcentratorConfig, N: int) -> tuple[float, float]:
    """
    Runs the simulation N times, extracting throughputs.
    Returns (mean_throughput, std_dev_throughput).
    """
    throughputs = []

    for idx in range(N):
        # By setting the replication length very high we let it hit the 6.6M extraction condition
        # and then the model automatically terminates
        sim = ConcentratorModel(config)

        engine = DRSEngine(sim)

        # Set a random seed so different replications are different
        np.random.seed(idx)
        random.seed(idx)

        engine.run(max_time=config.replication_length)

        # Calculate Throughput manually as the paper defined it
        active_time = (
            engine.current_time - sim.controller.cumulative_time_shutdown.value
        )
        if active_time > 0:
            throughput = (
                sim.mine.cumulative_extracted_mass.value
                - sim.config.ore_to_be_extracted_during_warming_period
            ) / active_time
            throughputs.append(throughput)

    if not throughputs:
        return 0.0, 0.0
    return float(np.mean(throughputs)), float(np.std(throughputs))


def plot_monte_carlo_throughput(N: int = 1, total_stockpile_level: float = 60000.0):
    sigmas = [5.0]
    results = []

    print(f"\n--- Running Monte Carlo Evaluation for Standard (N={N}) ---")
    for sigma in sigmas:
        config = ConcentratorConfig(
            replication_length=99999.0,
            std_dev_ore_fraction=sigma / 100.0,
            target_ore_stock_level=total_stockpile_level,
            prob_new_facies=0.3,
        )
        mean, std = evaluate_throughput(config, N)
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
        "Monte_Carlo_Throughput_Fig5_Standard.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("Saved 'Monte_Carlo_Throughput_Fig5_Standard.png'.\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=1)
    args = parser.parse_args()

    # You can also run it a single time and print out the statistics to evaluate how it spends time
    np.random.seed(42)
    random.seed(11)  # 11
    config = ConcentratorConfig(
        replication_length=99999.0,
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
        prob_new_facies=0.3,
    )
    sim = ConcentratorModel(config, enable_telemetry=True)

    # Generates an interactive dashboard spanning all operating modes
    from examples.mining.components.modes import MODES

    # Run your massive Monte Carlo simulation at lightning speed
    sim.controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine(sim, progress_bar=True, log_level="INFO")
    if sim.enable_telemetry and hasattr(sim, "telemetry"):
        engine.attach_telemetry(sim.telemetry)
    result = engine.run(max_time=config.replication_length)
    
    print(result.summary())
    
    sim.print_statistics()

    from drs.vis.module_graph import save_module_graph_report

    save_module_graph_report(sim, path_prefix="Concentrator_Module_Graph")

    df = result.history

    # --- Mode Transition Log ---
    state_change_events = [
        e for e in result.events 
        if e.event_type == "STATE_CHANGE" and e.details.get("variable") == "active_operating_mode"
    ]
    if state_change_events:
        print("\n--- Mode Transition Log ---")
        for e in state_change_events:
            old = e.details['old_value'].name if hasattr(e.details['old_value'], 'name') else str(e.details['old_value'])
            new = e.details['new_value'].name if hasattr(e.details['new_value'], 'name') else str(e.details['new_value'])
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
        build_dashboard,
    )
    from examples.mining.components.plot import (
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

    configs = [
        {
            "func": plot_time_series,
            "kwargs": {
                "y_columns": ["Mode A", "Mode B", "Shutdown"],
                "title": "Modes (Step)",
                "is_step": True,
            },
        },
        {
            "func": plot_ore_with_modes,
            "kwargs": {
                "time_col": "time",
                "ore_cols": [
                    "total_system_ore_mass",
                    "Ore1Stock_mass",
                    "Ore2Stock_mass",
                ],
                "mode_col": "active_operating_mode_name",
                "campaign_split_mode": "SHUTDOWN",
                "title": "Ore Stockpiles & Campaigns",
                "palette": palette,
                "hlines": [
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
            },
        },
        {
            "func": plot_dual_axis_step,
            "kwargs": {
                "y1_col": "MassOfCurrentParcel",
                "y2_col": "CurrentParcelRoutingFraction",
                "y1_label": "Parcel Mass (tons)",
                "y2_label": "Grade (% Ore 2)",
                "title": "Current Parcel Properties",
            },
        },
        {
            "func": plot_safety_margin,
            "kwargs": {
                "level_col": "Ore1Stock_mass",
                "constraint_value": 0.0,
                "constraint_type": "lower",
                "title": "Safety Margin: Ore 1 Distance to Floor",
                "danger_threshold": 1000.0,
            },
        },
        {
            "func": plot_safety_margin,
            "kwargs": {
                "level_col": "Ore2Stock_mass",
                "constraint_value": 0.0,
                "constraint_type": "lower",
                "title": "Safety Margin: Ore 2 Distance to Floor",
                "danger_threshold": 1000.0,
            },
        },
        {
            "func": plot_mode_distribution,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "time_col": "time",
                "title": "Mode Distribution (% of Time Spent)",
                "palette": palette,
            },
        },
        {
            "func": plot_mode_dwell_times,
            "kwargs": {
                "time_col": "time",
                "mode_col": "active_operating_mode_name",
                "title": "Mode Stability (Dwell Times)",
            },
        },
        {
            "func": plot_normalized_deviation_violin,
            "kwargs": {
                "title": "Stockpile Deviation Variance (Violin)",
                "target_total": 60000.0,
                "target_ore1": 42000.0,
                "target_ore2": 18000.0,
            },
        },
        {
            "func": plot_attributed_deficit,
            "kwargs": {
                "time_col": "time",
                "mode_col": "active_operating_mode_name",
                "extraction_col": "cumulative_extracted_mass",
                "ideal_rate_per_day": 6000.0,
                "title": "Cumulative Production Deficit by Mode",
                "palette": palette,
            },
        },
        {
            "func": plot_deficit_disparity,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "title": "Mode Efficiency (Time Spent vs. Deficit Caused)",
                "ideal_rate": 6000.0,
            },
        },
        {
            "func": plot_deficit_breakdown_bar,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate_per_day": 6000.0,
                "palette": palette,
            },
        },
        {
            "func": plot_structural_vs_operational_deficit,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate": 6000.0,
                "structural_modes": structural_modes,
            },
        },
        {
            "func": plot_normalized_cumulative_deficit,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate_per_day": 6000.0,
                "palette": palette,
            },
        },
        {
            "func": plot_structural_vs_operational_by_mode,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate": 6000.0,
                "structural_modes": structural_modes,
            },
        },
    ]

    fig_comp = build_dashboard(
        df, configs, title="Comprehensive Mine Diagnostics", figsize=(18, 69)
    )
    fig_comp.savefig("Comprehensive_Diagnostics_Plot.png")

    # Recreate Figure 5 from paper
    plot_monte_carlo_throughput(
        N=args.N, total_stockpile_level=args.total_stockpile_level
    )

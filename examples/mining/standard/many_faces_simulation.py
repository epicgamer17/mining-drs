"""
Two faces	Simplest multi-face extension. With 2 faces, allocation can be expressed as a single ratio. N-faces would require a linear program per mode.
Continuous fleet	Avoids event-based truck scheduling complexity. Continuous flows match the mill's steady-state assumption and the stockpile's continuous-time ODE.
Face allocation = fixed means per mode	Using face generator means (not current parcels) gives stable, campaign-long ratios. Dynamic per-timestep solves would jitter with parcel changes.
Surging = extreme allocation	Surging must produce an OFF-target blend to drain the stockpile. Using the base-mode allocation creates a degenerate equilibrium (extraction = milling).
50/50 face composition	Face means chosen so a 50/50 split matches the single-face's effective 70% ore1.
"""

import sys
import os

# Ensure the root directory is on the path so we can import 'examples.mining'
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import random
import types
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from examples.mining.components import (
    ConcentratorConfig,
    ConcentratorModel,
    ActiveFleetConcentratorModel,
)
from examples.mining.components.controllers import MultiFaceConcentratorController
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
        sim = ActiveFleetConcentratorModel(config)

        engine = DRSEngine(sim)

        # Set a random seed so different replications are different
        np.random.seed(idx)
        random.seed(idx)

        engine.run(max_time=config.replication_length)

        # Calculate Throughput manually as the paper defined it
        active_time = engine.current_time - sim.controller.cumulative_time_shutdown.value
        if active_time > 0:
            throughput = (
                (
                    sim.face1.cumulative_extracted_mass.value
                    + sim.face2.cumulative_extracted_mass.value
                )
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


def run_and_analyze(config, equal_allocation=False, name="Managed"):
    """Run the multi-face simulation and produce full diagnostics dashboard."""
    np.random.seed(42)
    random.seed(11)
    sim = ActiveFleetConcentratorModel(config, enable_telemetry=True)

    if equal_allocation:
        orig_forward = MultiFaceConcentratorController.forward
        def _equal_forward(self):
            orig_forward(self)
            n = len(self.faces)
            rate = self.target_mine_mass_rate.value / n
            for i in range(n):
                self.face_target_rates[i].value = rate
        sim.controller.forward = types.MethodType(_equal_forward, sim.controller)

    from examples.mining.components.modes import MODES

    sim.controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine(sim)
    engine.run(max_time=config.replication_length)
    sim.print_statistics()

    from drs.vis.module_graph import save_module_graph_report

    prefix = name.replace(" ", "_")
    save_module_graph_report(sim, path_prefix=f"Concentrator_Module_Graph_{prefix}")

    df = sim.telemetry.to_dataframe()

    # --- Mode Transition Log ---
    print(f"\n--- Mode Transition Log ({name}) ---")
    df["active_operating_mode_name"] = df["active_operating_mode"].apply(
        lambda x: x.name if x else "None"
    )
    print(df["active_operating_mode_name"].unique()[:5])
    df["prev_mode_name"] = df["active_operating_mode_name"].shift(1)
    transitions = df[
        (df["active_operating_mode_name"] != df["prev_mode_name"])
        & df["prev_mode_name"].notna()
    ]

    for idx, row in transitions.iterrows():
        print(
            f"Time: {row['time']:.2f} | Transition: {row['prev_mode_name']} -> {row['active_operating_mode_name']}"
        )
        print(
            f"  ↳ Ore1 Stock: {row['Ore1Stock_mass']:.1f} | Ore2 Stock: {row['Ore2Stock_mass']:.1f} (Critical: {config.critical_ore2_level}) | Total Stock: {row['total_system_ore_mass']:.1f} (Target: {config.target_ore_stock_level})"
        )
        print(
            f"  ↳ Campaign/Shutdown Timer: {row['current_campaign_duration']:.2f} | Contingency Timer: {row['current_contingency_duration']:.2f}"
        )
    print("---------------------------\n")

    # --- Cumulative Deficit by Mode Log ---
    dt = df["time"].diff().fillna(0)
    actual_extraction_step = (
        (df["face1_extracted_mass"] + df["face2_extracted_mass"]).diff().fillna(0)
    )
    ideal_extraction_step = dt * 6000.0
    step_deficit = (ideal_extraction_step - actual_extraction_step).clip(lower=0)

    deficit_df = pd.DataFrame(
        {"mode": df["active_operating_mode_name"], "deficit": step_deficit}
    )

    total_deficit_by_mode = (
        deficit_df.groupby("mode")["deficit"].sum().sort_values(ascending=False)
    )

    print(f"\n--- Cumulative Lost Production (Deficit) by Mode ({name}) ---")
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
        plot_mode_distribution,
        plot_mode_dwell_times,
        build_dashboard,
    )
    from examples.mining.components.plot import (
        plot_ore_with_modes,
        plot_state_space,
        plot_cumulative_throughput,
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
                "y1_col": "face1_parcel_mass",
                "y2_col": "face1_parcel_ratio",
                "y1_label": "Face 1 Parcel Mass (tons)",
                "y2_label": "Face 1 Ore 1 Fraction",
                "title": "Face 1 Current Parcel Properties",
                "y1_color": "saddlebrown",
                "y2_color": "darkorange",
            },
        },
        {
            "func": plot_dual_axis_step,
            "kwargs": {
                "y1_col": "face2_parcel_mass",
                "y2_col": "face2_parcel_ratio",
                "y1_label": "Face 2 Parcel Mass (tons)",
                "y2_label": "Face 2 Ore 1 Fraction",
                "title": "Face 2 Current Parcel Properties",
                "y1_color": "saddlebrown",
                "y2_color": "darkorange",
            },
        },
        {
            "func": plot_dual_axis_step,
            "kwargs": {
                "y1_col": "mixed_extraction_rate",
                "y2_col": "mixed_ore1_fraction",
                "y1_label": "Combined Extraction Rate (t/d)",
                "y2_label": "Mixed Ore 1 Fraction",
                "title": "Combined Mine Output Properties",
                "y1_color": "saddlebrown",
                "y2_color": "darkorange",
            },
        },
        {
            "func": plot_time_series,
            "kwargs": {
                "y_columns": ["face1_alloc", "face2_alloc", "ore2_ratio"],
                "title": "Active Fleet Allocation & Stockpile Ratio",
                "is_step": True,
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
        df, configs, title=f"Comprehensive Mine Diagnostics ({name})", figsize=(18, 69)
    )
    fig_comp.savefig(f"Comprehensive_Diagnostics_Plot_{prefix}.png")
    plt.close(fig_comp)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=1)
    args = parser.parse_args()

    config = ConcentratorConfig(
        replication_length=99999.0,
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
        prob_new_facies=0.3,
    )

    df_managed = run_and_analyze(config, equal_allocation=False, name="Managed")
    df_equal = run_and_analyze(config, equal_allocation=True, name="Equal Allocation")

    # Print summary comparison
    print("\n" + "=" * 72)
    print("COMPARISON SUMMARY: Managed vs Equal Allocation")
    print("=" * 72)

    for label, df in [("Managed", df_managed), ("Equal Allocation", df_equal)]:
        dt = df["time"].diff().fillna(0)
        final_ore1 = df["Ore1Stock_mass"].iloc[-1]
        final_ore2 = df["Ore2Stock_mass"].iloc[-1]
        total_extracted = (
            df["face1_extracted_mass"].iloc[-1] + df["face2_extracted_mass"].iloc[-1]
        )
        print(f"\n--- {label} ---")
        print(f"  Total extracted: {total_extracted:,.0f} tons")
        print(f"  Final Ore1 stock: {final_ore1:,.0f} t | Final Ore2 stock: {final_ore2:,.0f} t")
        print(f"  Mode breakdown:")
        for mode_name in df["active_operating_mode_name"].unique():
            mode_dt = dt[df["active_operating_mode_name"] == mode_name].sum()
            pct = (mode_dt / dt.sum() * 100) if dt.sum() > 0 else 0
            print(f"    {str(mode_name):35s}: {mode_dt:8.2f} days ({pct:5.1f}%)")
    print("=" * 72 + "\n")

    # Recreate Figure 5 from paper
    plot_monte_carlo_throughput(
        N=args.N, total_stockpile_level=args.total_stockpile_level
    )

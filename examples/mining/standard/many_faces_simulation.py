"""
Two faces	Simplest multi-face extension. With 2 faces, allocation can be expressed as a single ratio. N-faces would require a linear program per mode.
Continuous fleet	Avoids event-based truck scheduling complexity. Continuous flows match the mill's steady-state assumption and the stockpile's continuous-time ODE.
Face allocation = fixed means per mode	Using face generator means (not current parcels) gives stable, campaign-long ratios. Dynamic per-timestep solves would jitter with parcel changes.
Surging = extreme allocation	Surging must produce an OFF-target blend to drain the stockpile. Using the base-mode allocation creates a degenerate equilibrium (extraction = milling).
50/50 face composition	Face means chosen so a 50/50 split matches the single-face's effective 70% ore1.
"""

import sys
import os
from dataclasses import replace

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

from examples.mining.components import (
    ConcentratorConfig,
    ConcentratorModel,
    ActiveFleetConcentratorModel,
)
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
        # Throughput = Total Processed / Active Production Time (Time - Shutdown Time)
        total_time = (
            sim.controller.cumulative_time_mode_a.value
            + sim.controller.cumulative_time_mode_a_contingency.value
            + sim.controller.cumulative_time_mode_a_surging.value
            + sim.controller.cumulative_time_mode_b.value
            + sim.controller.cumulative_time_mode_b_contingency.value
            + sim.controller.cumulative_time_mode_b_surging.value
            + sim.controller.cumulative_time_shutdown.value
        )

        active_time = total_time - sim.controller.cumulative_time_shutdown.value
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


def _run_capacity_case(
    label: str,
    config: ConcentratorConfig,
    max_time: float,
    np_seed: int = 42,
    random_seed: int = 11,
):
    from examples.mining.components.modes import MODES

    np.random.seed(np_seed)
    random.seed(random_seed)

    sim = ActiveFleetConcentratorModel(config, enable_telemetry=True)
    sim.controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine(sim)
    engine.run(max_time=max_time)

    df = sim.telemetry.to_dataframe()
    df["scenario"] = label
    df["active_operating_mode_name"] = df["active_operating_mode"].apply(
        lambda x: x.name if x else "None"
    )
    df["total_required_rate"] = (
        df["face1_required_rate"] + df["face2_required_rate"]
    )
    df["total_max_extraction_rate"] = (
        df["face1_max_extraction_rate"] + df["face2_max_extraction_rate"]
    )
    df["total_actual_rate"] = df["face1_actual_rate"] + df["face2_actual_rate"]
    df["capacity_gap_rate"] = (
        df["total_required_rate"] - df["total_actual_rate"]
    ).clip(lower=0.0)
    df["capacity_utilization"] = np.where(
        df["total_required_rate"] > 1e-12,
        df["total_actual_rate"] / df["total_required_rate"],
        0.0,
    )

    dt = df["time"].diff().shift(-1).fillna(0.0)
    capacity_lost_mass = float((df["capacity_gap_rate"] * dt).sum())
    active_utilization = df.loc[
        df["total_required_rate"] > 1e-12, "capacity_utilization"
    ]
    summary = {
        "scenario": label,
        "enable_face_capacity_limit": config.enable_face_capacity_limit,
        "final_time": float(df["time"].iloc[-1]),
        "fleet_shift_count": float(df["fleet_shift_count"].max()),
        "mean_total_required_rate": float(df["total_required_rate"].mean()),
        "mean_total_max_extraction_rate": float(
            df["total_max_extraction_rate"].mean()
        ),
        "mean_total_actual_rate": float(df["total_actual_rate"].mean()),
        "mean_face1_effective_delay_factor": float(
            df["face1_effective_delay_factor"].mean()
        ),
        "mean_face2_effective_delay_factor": float(
            df["face2_effective_delay_factor"].mean()
        ),
        "mean_capacity_utilization": float(active_utilization.mean()),
        "max_capacity_gap_rate": float(df["capacity_gap_rate"].max()),
        "capacity_lost_mass": capacity_lost_mass,
        "final_ore1_stock": float(df["Ore1Stock_mass"].iloc[-1]),
        "final_ore2_stock": float(df["Ore2Stock_mass"].iloc[-1]),
        "min_ore2_stock": float(df["Ore2Stock_mass"].min()),
    }
    return df, summary


def run_capacity_comparison(
    base_config: ConcentratorConfig,
    max_time: float = 60.0,
):
    import pandas as pd
    from examples.mining.components.plot import plot_ore_with_modes

    baseline_config = replace(base_config, enable_face_capacity_limit=False)
    constrained_config = replace(
        base_config,
        enable_face_capacity_limit=True,
        face_lhd_allocation=(0.50, 0.50),
        face_truck_allocation=(0.50, 0.50),
        face_availability=(0.93, 0.91),
        face_haul_distance=(1.0, 1.2),
        face_delay_factor=(0.025, 0.04),
        face_gas_delay_factor=(0.005, 0.01),
        face_truck_congestion_threshold=(0.45, 0.45),
        truck_congestion_delay_sensitivity=0.10,
        face_shift_capacity_factor=(1.0, 1.0),
    )

    cases = [
        ("Policy 1 baseline", baseline_config),
        ("Policy 1 + fleet capacity limit", constrained_config),
    ]

    frames = []
    summaries = []
    for label, config in cases:
        df, summary = _run_capacity_case(label, config, max_time=max_time)
        frames.append(df)
        summaries.append(summary)
        print(
            f"{label}: mean actual rate={summary['mean_total_actual_rate']:.1f} t/d, "
            f"capacity lost={summary['capacity_lost_mass']:.1f} t, "
            f"min Ore2={summary['min_ore2_stock']:.1f} t"
        )

    combined = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    selected_columns = [
        "scenario",
        "time",
        "active_operating_mode_name",
        "fleet_shift_count",
        "fleet_shift_timer",
        "face1_required_rate",
        "face1_max_extraction_rate",
        "face1_actual_rate",
        "face1_effective_delay_factor",
        "face2_required_rate",
        "face2_max_extraction_rate",
        "face2_actual_rate",
        "face2_effective_delay_factor",
        "total_required_rate",
        "total_max_extraction_rate",
        "total_actual_rate",
        "capacity_gap_rate",
        "capacity_utilization",
        "Ore1Stock_mass",
        "Ore2Stock_mass",
        "total_system_ore_mass",
        "mixed_ore1_fraction",
    ]
    combined[selected_columns].to_csv(
        "capacity_policy_comparison.csv", index=False, encoding="utf-8"
    )
    summary_df.to_csv(
        "capacity_policy_comparison_summary.csv", index=False, encoding="utf-8"
    )

    mode_order = [
        "SHUTDOWN",
        "MODE_A",
        "MODE_A_CONTINGENCY",
        "MODE_A_MINE_SURGING",
        "MODE_B",
        "MODE_B_CONTINGENCY",
        "MODE_B_MINE_SURGING",
    ]
    mode_to_y = {mode: idx for idx, mode in enumerate(mode_order)}

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    for label, group in combined.groupby("scenario"):
        axes[0].plot(
            group["time"],
            group["total_required_rate"],
            linestyle="--",
            label=f"{label} required",
        )
        axes[0].plot(
            group["time"], group["total_actual_rate"], label=f"{label} actual"
        )
        axes[1].plot(group["time"], group["capacity_gap_rate"], label=label)
        axes[2].plot(group["time"], group["capacity_utilization"], label=label)
        axes[3].plot(group["time"], group["Ore2Stock_mass"], label=label)
        mode_y = group["active_operating_mode_name"].map(mode_to_y)
        axes[4].step(group["time"], mode_y, where="post", label=label)
        axes[5].plot(
            group["time"], group["face1_actual_rate"], label=f"{label} face1"
        )
        axes[5].plot(
            group["time"],
            group["face2_actual_rate"],
            linestyle="--",
            label=f"{label} face2",
        )

    axes[0].set_ylabel("Rate (t/d)")
    axes[0].set_title("Required vs Actual Extraction Rate")
    axes[1].set_ylabel("Capacity Gap (t/d)")
    axes[1].set_title("Lost Rate Due to Fleet Capacity Limit")
    axes[2].set_ylabel("Actual / Required")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Capacity Utilization")
    axes[3].set_ylabel("Ore 2 Stockpile (t)")
    axes[3].set_title("Ore 2 Stockpile Response")
    axes[4].set_ylabel("Mode")
    axes[4].set_yticks(list(mode_to_y.values()))
    axes[4].set_yticklabels(mode_order)
    axes[4].set_title("Operating Mode Timeline")
    axes[5].set_ylabel("Face Actual Rate (t/d)")
    axes[5].set_xlabel("Time (days)")
    axes[5].set_title("Face-Level Actual Extraction Rates")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    fig.tight_layout()
    fig.savefig("Capacity_Policy_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    palette = {
        "MODE_A": "#1f77b4",
        "MODE_A_CONTINGENCY": "#2ca02c",
        "MODE_A_MINE_SURGING": "#9467bd",
        "MODE_B": "#d62728",
        "MODE_B_CONTINGENCY": "#ff7f0e",
        "MODE_B_MINE_SURGING": "#8c564b",
        "SHUTDOWN": "#FFD700",
    }
    mode_groups = {
        "Mode A": ("MODE_A", "MODE_A_CONTINGENCY", "MODE_A_MINE_SURGING"),
        "Mode B": ("MODE_B", "MODE_B_CONTINGENCY", "MODE_B_MINE_SURGING"),
        "Shutdown": ("SHUTDOWN",),
    }
    for label, group in combined.groupby("scenario"):
        diagnostic_df = group.copy()
        for mode_label, mode_names in mode_groups.items():
            diagnostic_df[mode_label] = diagnostic_df[
                "active_operating_mode_name"
            ].apply(lambda mode: len(mode_groups) - list(mode_groups).index(mode_label) if mode in mode_names else 0)

        fig_diag, axes_diag = plt.subplots(
            4,
            1,
            figsize=(14, 15),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 2.2, 1, 1]},
        )
        axes_diag[0].step(
            diagnostic_df["time"],
            diagnostic_df["Mode A"],
            where="post",
            label="Mode A",
        )
        axes_diag[0].step(
            diagnostic_df["time"],
            diagnostic_df["Mode B"],
            where="post",
            label="Mode B",
        )
        axes_diag[0].step(
            diagnostic_df["time"],
            diagnostic_df["Shutdown"],
            where="post",
            label="Shutdown",
        )
        axes_diag[0].set_title("Modes (Step)")
        axes_diag[0].set_yticks([0, 1, 2, 3])
        axes_diag[0].legend(loc="upper right")

        plot_ore_with_modes(
            diagnostic_df,
            time_col="time",
            ore_cols=[
                "total_system_ore_mass",
                "Ore1Stock_mass",
                "Ore2Stock_mass",
            ],
            mode_col="active_operating_mode_name",
            campaign_split_mode="SHUTDOWN",
            title="Ore Stockpiles & Mode Changes",
            palette=palette,
            hlines=[
                {
                    "y": base_config.target_ore_stock_level,
                    "color": "black",
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": "Target Total",
                },
                {
                    "y": base_config.critical_ore2_level,
                    "color": "red",
                    "linestyle": ":",
                    "linewidth": 2,
                    "alpha": 0.8,
                    "label": "Critical Ore 2",
                },
            ],
            ax=axes_diag[1],
        )

        axes_diag[2].step(
            diagnostic_df["time"],
            diagnostic_df["total_required_rate"],
            where="post",
            linestyle="--",
            label="Required rate",
        )
        axes_diag[2].step(
            diagnostic_df["time"],
            diagnostic_df["total_actual_rate"],
            where="post",
            label="Actual rate",
        )
        axes_diag[2].set_ylabel("Rate (t/d)")
        axes_diag[2].set_title("Required vs Actual Extraction Rate")
        axes_diag[2].legend(loc="upper right")

        axes_diag[3].step(
            diagnostic_df["time"],
            diagnostic_df["capacity_utilization"],
            where="post",
            label="Capacity utilization",
        )
        axes_diag[3].set_ylim(-0.05, 1.05)
        axes_diag[3].set_ylabel("Actual / Required")
        axes_diag[3].set_xlabel("Time (days)")
        axes_diag[3].set_title("Fleet Capacity Utilization")
        axes_diag[3].legend(loc="upper right")

        for ax in axes_diag:
            ax.grid(True, linestyle="--", alpha=0.35)
        fig_diag.suptitle(label, fontsize=15)
        fig_diag.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
        safe_label = label.lower().replace(" ", "_").replace("+", "plus")
        safe_label = safe_label.replace("/", "_")
        fig_diag.savefig(
            f"Capacity_Policy_Diagnostics_{safe_label}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_diag)

    print("Saved capacity_policy_comparison.csv")
    print("Saved capacity_policy_comparison_summary.csv")
    print("Saved Capacity_Policy_Comparison.png")
    print("Saved Capacity_Policy_Diagnostics_*.png")
    return combined, summary_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--compare_capacity_cases", action="store_true")
    parser.add_argument("--comparison_max_time", type=float, default=60.0)
    args = parser.parse_args()

    # You can also run it a single time and print out the statistics to evaluate how it spends time
    np.random.seed(42)
    random.seed(11)
    config = ConcentratorConfig(
        replication_length=99999.0,
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
        prob_new_facies=0.3,
    )

    if args.compare_capacity_cases:
        run_capacity_comparison(config, max_time=args.comparison_max_time)
        raise SystemExit(0)

    sim = ActiveFleetConcentratorModel(config, enable_telemetry=True)

    # Generates an interactive dashboard spanning all operating modes
    from examples.mining.components.modes import MODES

    # Run your massive Monte Carlo simulation at lightning speed
    sim.controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine(sim)
    engine.run(max_time=config.replication_length)
    sim.print_statistics()

    from drs.vis.module_graph import save_module_graph_report

    save_module_graph_report(sim, path_prefix="Concentrator_Module_Graph")

    df = sim.telemetry.to_dataframe()

    # --- Mode Transition Log ---
    print("\n--- Mode Transition Log ---")
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
    import pandas as pd

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
                "y_columns": [
                    "mixed_required_extraction_rate",
                    "mixed_max_extraction_rate",
                    "mixed_extraction_rate",
                ],
                "title": "Fleet-Constrained Extraction Rates",
                "is_step": True,
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
        df, configs, title="Comprehensive Mine Diagnostics", figsize=(18, 69)
    )
    fig_comp.savefig("Comprehensive_Diagnostics_Plot.png")

    # Recreate Figure 5 from paper
    plot_monte_carlo_throughput(
        N=args.N, total_stockpile_level=args.total_stockpile_level
    )

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
from drs_mining.control import step_policy


class Scenario(drs.Module):
    """Flat scenario container for the blending example.

    Holds the physical entities constructed below and exposes:
      * the policy surface consumed by ``step_policy``,
      * ``state_components`` (the registered leaves the engine advances),
      * termination + observation via ``drs.Telemetry(scenario)``.

    The container itself is a no-op stepper so only the registered leaves move.
    """

    def __init__(
        self,
        mine,
        fleet,
        plant,
        controller,
        ore1_stock,
        ore2_stock,
        total_ore_to_extract: float = 6600000.0,
    ):
        super().__init__()
        self.mine = mine
        self.fleet = fleet
        self.plant = plant
        self.controller = controller
        self.ore1_stock = ore1_stock
        self.ore2_stock = ore2_stock
        self.total_ore_to_extract = total_ore_to_extract
        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)

    @property
    def faces(self):
        return getattr(self.controller, "faces", None)

    @property
    def ore1_mass(self) -> float:
        return self.ore1_stock.current_mass.value if self.ore1_stock else 0.0

    @property
    def ore2_mass(self) -> float:
        return self.ore2_stock.current_mass.value if self.ore2_stock else 0.0

    @property
    def total_stockpile_mass(self) -> float:
        return self.ore1_mass + self.ore2_mass

    @property
    def stockpile2_routing_fraction(self) -> float:
        return self.fleet.stockpile2_routing_fraction.value if self.fleet else 0.0

    @property
    def state_components(self) -> list:
        comps = []
        if self.controller is not None:
            comps.extend(self.controller.state_components)
        if self.plant is not None:
            comps.append(self.plant)
        if self.ore1_stock is not None:
            comps.append(self.ore1_stock)
        if self.ore2_stock is not None:
            comps.append(self.ore2_stock)
        comps.append(self.global_time)
        return comps

    def is_terminating_condition_met(self) -> bool:
        sources = self.faces or ([self.mine] if self.mine is not None else [])
        total = sum(s.cumulative_extracted_mass.value for s in sources)
        return total >= self.total_ore_to_extract

    def time_to_event(self) -> float:
        return float("inf")

    def step(self, dt: float) -> None:
        pass


def build_blending_scenario(
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
) -> Scenario:
    """Construct the blending scenario flat: every physical entity is built here.

    The mine face parcel limits, the two initial stockpiles (Ore1/Ore2), the
    concentrator plant cap, the continuous fleet, and the blending controller
    are all instantiated at the top level with explicit keyword arguments.
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
    )

    scenario = Scenario(
        mine=mine,
        fleet=fleet,
        plant=plant,
        controller=controller,
        ore1_stock=ore1_stock,
        ore2_stock=ore2_stock,
        total_ore_to_extract=total_ore_to_extract,
    )
    scenario.replication_length = kwargs.get("replication_length", float("inf"))
    scenario.enable_telemetry = kwargs.get("enable_telemetry", False)
    return scenario


def print_statistics(sim):
    """Print operating-mode time-shares and throughput for a blending scenario."""
    print("\n--- Output Statistics ---")
    total_time = sim.controller.total_duration

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
                f"{label}: {getattr(sim.controller, attr).value / total_time:.4f}"
            )
    else:
        print("Total time is 0. Cannot calculate mode portions.")

    active_time = sim.controller.active_duration(total_time)
    if active_time > 0:
        if hasattr(sim.plant, "cumulative_milled_mass"):
            total_ore_processed = sim.plant.cumulative_milled_mass.value
        else:
            total_ore_processed = sim.mine.net_extracted_mass
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


def evaluate_throughput(config_kwargs: dict, N: int) -> tuple[float, float]:
    """
    Runs the simulation N times, extracting throughputs.
    Returns (mean_throughput, std_dev_throughput).
    """
    throughputs = []

    for idx in range(N):
        np.random.seed(idx)
        random.seed(idx)

        sim = build_blending_scenario(**config_kwargs)

        engine = DRSEngine()
        engine.register(sim, *sim.state_components)

        @engine.on_step
        def _policy(time):
            step_policy(sim, time)

        engine.run(until=config_kwargs.get("replication_length", float("inf")))

        active_time = sim.controller.active_duration(engine.current_time)
        if active_time > 0:
            throughput = sim.mine.net_extracted_mass / active_time
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
    sim = build_blending_scenario(
        replication_length=99999.0,
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
        prob_new_facies=0.3,
    )

    from drs_mining.components.modes import MODES

    sim.controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine()
    engine.register(sim, *sim.state_components)

    @engine.on_step
    def _policy(time):
        step_policy(sim, time)

    telemetry = Telemetry(sim)
    telemetry.register_metric(
        "MassOfCurrentParcel",
        lambda t, m, s, _: m.mine.active_parcel_initial_mass.value,
    )
    telemetry.register_metric(
        "CurrentParcelRoutingFraction",
        lambda t, m, s, _: m.fleet.stockpile2_routing_fraction.value,
    )
    telemetry.register_metric(
        "Campaign_Shutdown",
        lambda t, m, s, _: m.controller.current_campaign_duration.value,
    )
    telemetry.register_metric(
        "Contingency",
        lambda t, m, s, _: m.controller.current_contingency_duration.value,
    )
    engine.attach_telemetry(telemetry)
    result = engine.run(until=99999.0)

    print(result.summary())

    print_statistics(sim)

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

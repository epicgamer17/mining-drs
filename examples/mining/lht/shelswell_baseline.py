"""
Shelswell (2017) Baseline Truck Haulage Simulation (Hybrid DRS Architecture)

===============================================================================
1. SUMMARY OF SHELSWELL (2017) PAPER SPECIFICATIONS
===============================================================================
- Mine Layout: 2100m access decline (5% grade) + 1800m spiral ramp connecting 7 mine levels (300m spacing).
- Loadouts: 40m access drift off ramp for Ore, 55m access drift for Waste. Air doors 20m off ramp.
- Surface Destinations: ROM crusher pad at 300m, Waste Dump at 440m, Maintenance Shop at 260m, Fuel Depot at 270m.
- Production Schedule: 365 calendar days with 5.5:1 ratio of Ore to Waste.
- Shift Schedule: 2 shifts per day, 10.5 hours working time each (12h total, 1.5h shift gap).
- Equipment Specs: CAT AD30 haul trucks (26.1t Ore, 24.6t Waste) and underground LHD loaders (14.0t Ore, 12.5t Waste).

===============================================================================
2. THREE-LAYER HYBRID DISCRETE RATE SYSTEM (DRS) ARCHITECTURE
===============================================================================
This implementation uses a Pythonic Three-Layer Hybrid DRS Pattern:
- Layer 1: Discrete Domain Model (Trucks, LHDs, Fuel/Maintenance state, Unclaimed Tonnes Dispatch Logic).
- Layer 2: DRS Interface & Rate Gateways (Loading/Dumping Impulse Rate Enforcers, Continuous Road Availability Timers).
- Layer 3: DRS Core Engine (`drs` library continuous numerical integration loop).
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from drs_mining.simulation import ShelswellHybridSimulation


def run_simulation(trucks: int, operators: int, availability: float) -> float:
    """Executes a single Hybrid DRS simulation run for a given fleet configuration.

    Args:
        trucks: Number of haul trucks in fleet (3 to 10).
        operators: Number of truck operators available per shift (1 to 10).
        availability: Mechanical availability fraction (0.5 to 1.0).

    Returns:
        float: Average daily haulage productivity (tonnes/day).
    """
    random.seed(42)
    np.random.seed(42)

    sim = ShelswellHybridSimulation(
        num_trucks=trucks,
        num_operators=operators,
        mechanical_availability=availability,
    )
    return sim.run_simulation(total_days=365.0, dt=300.0, show_progress=False)


def _run_task(args):
    """Top-level helper function for multiprocessing worker execution."""
    trucks, operators, avail = args
    return (trucks, operators, avail, run_simulation(trucks, operators, avail))


def generate_figure_2():
    """Replicates Figure 2 from Shelswell (2017): Productivity vs Fleet Size without operator constraints."""
    print("Generating Figure 2 (Productivity vs Fleet Size without operator constraints)...")
    availabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    truck_sizes = list(range(3, 11))

    base_prod = run_simulation(10, 10, 1.0)
    print(f"Base Hybrid DRS productivity (10 trucks, 10 ops, 100% avail): {base_prod:.2f} t/d")

    tasks = [(t, t, a) for a in availabilities for t in truck_sizes]

    results_map = {}
    with ProcessPoolExecutor() as executor:
        for t, o, a, prod in tqdm(executor.map(_run_task, tasks), total=len(tasks), desc="Figure 2 Parallel Sweep"):
            results_map[(t, o, a)] = prod

    plt.figure(figsize=(10, 6))
    colors = {
        0.5: "green",
        0.6: "brown",
        0.7: "orange",
        0.8: "blue",
        0.9: "hotpink",
        1.0: "black",
    }

    for avail in availabilities:
        prod_list = [results_map[(t, t, avail)] / base_prod for t in truck_sizes]
        plt.plot(
            truck_sizes,
            prod_list,
            marker="o",
            label=f"{int(avail*100)}% availability",
            color=colors[avail],
        )

    plt.title(
        "Haulage productivity analysis without haulage operator constraints (Hybrid DRS)",
        fontsize=14,
    )
    plt.xlabel("Number of trucks in fleet", fontsize=12)
    plt.ylabel("Normalised productivity", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.ylim(0, 1.1)

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/shelswell_fig2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/shelswell_fig2.png")


def generate_figures_3_to_8():
    """Replicates Figures 3 through 8 from Shelswell (2017): Productivity vs Operator Count."""
    availabilities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    operator_counts = list(range(1, 11))
    truck_counts = list(range(3, 11))

    tasks = [(t, min(o, t), a) for a in availabilities for t in truck_counts for o in operator_counts]

    results_map = {}
    with ProcessPoolExecutor() as executor:
        for t, o, a, prod in tqdm(executor.map(_run_task, tasks), total=len(tasks), desc="Figures 3-8 Parallel Sweep"):
            results_map[(t, o, a)] = prod

    colors = {
        3: "green",
        4: "brown",
        5: "orange",
        6: "blue",
        7: "hotpink",
        8: "purple",
        9: "yellow",
        10: "black",
    }
    markers = {
        3: "o",
        4: "s",
        5: "^",
        6: "v",
        7: "D",
        8: "P",
        9: "X",
        10: "*",
    }

    for avail in availabilities:
        base_prod = results_map[(10, 10, avail)]

        plt.figure(figsize=(10, 6))
        # Plot in reverse order (10 trucks down to 3 trucks) so smaller fleet curves remain visible on top when overlapping
        for trucks in reversed(truck_counts):
            prod_list = [results_map[(trucks, min(ops, trucks), avail)] / base_prod for ops in operator_counts]
            plt.plot(
                operator_counts,
                prod_list,
                marker=markers[trucks],
                markersize=6,
                label=f"{trucks} trucks",
                color=colors[trucks],
                zorder=10 - trucks,  # higher zorder for smaller fleets to prevent hiding under 10 trucks
            )

        plt.title(
            f"Haulage fleet size productivity analysis with operator constraints ({int(avail*100)}% Availability)",
            fontsize=13,
        )
        plt.xlabel("Number of haulage operators", fontsize=12)
        plt.ylabel("Normalised productivity", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="lower right")
        plt.ylim(0, 1.1)

        fig_num = {1.0: 3, 0.9: 4, 0.8: 5, 0.7: 6, 0.6: 7, 0.5: 8}[avail]
        plt.savefig(f"plots/shelswell_fig{fig_num}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print("Saved plots/shelswell_fig3.png through plots/shelswell_fig8.png")


if __name__ == "__main__":
    generate_figure_2()
    generate_figures_3_to_8()
    print("Replication of all figures complete with High-Performance Hybrid DRS Architecture!")

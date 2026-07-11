import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from drs_mining.components.config import ConcentratorConfig
from examples.mining.standard.many_faces_simulation import _run_capacity_case


def plot_theoretical_fleet_diminishing_returns():
    """
    Shows the mathematical diminishing returns of adding trucks to a face,
    incorporating traffic delay, cycle time increases, and match factor saturation.
    """
    c = ConcentratorConfig()

    # We will hold loaders constant, and vary trucks from 1 to 40
    lhd_alloc = c.total_lhd_count
    truck_allocs = np.linspace(1, 40, 100)

    # Face 1 properties
    distance = c.face_haul_distance[0]
    travel_time = (2 * distance) / c.truck_velocity

    cycle_times = []
    match_factors = []
    throughputs = []

    for truck_alloc in truck_allocs:
        # 1. Traffic Delay
        traffic_delay = c.traffic_delay_per_truck_hours * truck_alloc

        # Calculate how long it takes to completely fill one truck
        truck_loading_time_hours = c.loader_cycle_time_hours * (
            c.truck_payload_tonnes / c.loader_payload_tonnes
        )

        # 2. Cycle Time
        truck_cycle_time = (
            travel_time
            + truck_loading_time_hours
            + c.truck_dump_time_hours
            + traffic_delay
        )
        cycle_times.append(truck_cycle_time)

        # 3. Match Factor
        mf = (truck_alloc * truck_loading_time_hours) / (lhd_alloc * truck_cycle_time)
        match_factors.append(mf)

        # 4. Throughput
        if mf < 1.0:
            rate = (truck_alloc / truck_cycle_time) * c.truck_payload_tonnes * 24.0
        else:
            rate = (
                (lhd_alloc / c.loader_cycle_time_hours) * c.loader_payload_tonnes * 24.0
            )

        throughputs.append(rate)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Throughput Plot
    axes[0].plot(truck_allocs, throughputs, color="green", linewidth=2.5)
    axes[0].axvline(x=10, color="r", linestyle="--", label="Current Fleet (10 Trucks)")
    axes[0].set_ylabel("Extraction Rate (t/d)", fontsize=12)
    axes[0].set_xlabel("Number of Trucks Assigned to Face", fontsize=12)
    axes[0].set_title(
        f"Diminishing Returns: Throughput vs Fleet Size ({int(lhd_alloc)} LHDs Constant)",
        fontsize=14,
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Match Factor Plot
    axes[1].plot(truck_allocs, match_factors, color="blue", linewidth=2.5)
    axes[1].axhline(y=1.0, color="k", linestyle="--", label="Perfect Match (MF=1.0)")
    axes[1].axvline(x=10, color="r", linestyle="--")
    axes[1].set_ylabel("Match Factor", fontsize=12)
    axes[1].set_xlabel("Number of Trucks Assigned to Face", fontsize=12)
    axes[1].set_title(
        "Fleet Match Factor (MF > 1 means Loaders are Bottleneck)", fontsize=14
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Cycle Time Plot
    axes[2].plot(truck_allocs, cycle_times, color="purple", linewidth=2.5)
    axes[2].axvline(x=10, color="r", linestyle="--")
    axes[2].set_ylabel("Cycle Time (Hours)", fontsize=12)
    axes[2].set_xlabel("Number of Trucks Assigned to Face", fontsize=12)
    axes[2].set_title(
        "Traffic Effect: Cycle Time Increases linearly with Trucks", fontsize=14
    )
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig("docs/assets/comparisons/Diminishing_Returns_Theoretical.png", dpi=300)
    print("Saved docs/assets/comparisons/Diminishing_Returns_Theoretical.png")
    plt.close(fig)


def generate_infinite_vs_constrained_plots():
    base_config = ConcentratorConfig()

    cases = [
        (
            "Severely Constrained (10 Trucks, 40kt/d Target)",
            replace(
                base_config,
                total_truck_count=10.0,
                mode_a_ore1_milling_rate=24000.0,
                mode_a_ore2_milling_rate=16000.0,
                mode_b_ore1_milling_rate=32000.0,
                mode_b_ore2_milling_rate=8000.0,
            ),
        ),
        (
            "Heavily Constrained (20 Trucks, 40kt/d Target)",
            replace(
                base_config,
                total_truck_count=20.0,
                mode_a_ore1_milling_rate=24000.0,
                mode_a_ore2_milling_rate=16000.0,
                mode_b_ore1_milling_rate=32000.0,
                mode_b_ore2_milling_rate=8000.0,
            ),
        ),
        (
            "Maximum Capacity (40 Trucks, 40kt/d Target)",
            replace(
                base_config,
                total_truck_count=40.0,
                mode_a_ore1_milling_rate=24000.0,
                mode_a_ore2_milling_rate=16000.0,
                mode_b_ore1_milling_rate=32000.0,
                mode_b_ore2_milling_rate=8000.0,
            ),
        ),
        (
            "Infinite Fleet (200 Trucks, 200 LHDs, 40kt/d Target)",
            replace(
                base_config,
                total_truck_count=200.0,
                total_lhd_count=200.0,
                traffic_delay_per_truck_hours=0.0,
                mode_a_ore1_milling_rate=24000.0,
                mode_a_ore2_milling_rate=16000.0,
                mode_b_ore1_milling_rate=32000.0,
                mode_b_ore2_milling_rate=8000.0,
            ),
        ),
    ]

    frames = []

    for label, config in cases:
        print(f"Running simulation: {label}...")
        # Run using dynamic fleet allocation
        df, summary = _run_capacity_case(
            label, config, max_time=60.0, equal_allocation=False
        )
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Create custom plot for the comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]

    for (label, group), color in zip(combined.groupby("scenario", sort=False), colors):
        # 1. Total Achieved Extraction Rate
        axes[0].plot(
            group["time"],
            group["total_achieved_extraction_rate"],
            label=label,
            color=color,
            alpha=0.8,
        )

        # 2. Capacity Gap Rate
        axes[1].plot(
            group["time"],
            group["capacity_gap_rate"],
            label=label,
            color=color,
            alpha=0.8,
        )

        # 3. Ore 2 Stockpile
        axes[2].plot(
            group["time"], group["Ore2Stock_mass"], label=label, color=color, alpha=0.8
        )

    # Plot target required rate on axis 0
    axes[0].plot(
        frames[-1]["time"],
        frames[-1]["total_target_extraction_rate"],
        "k--",
        label="Target Required Rate",
        alpha=0.6,
    )

    axes[0].set_title(
        "Simulation: Achieved Extraction Rate under varying Fleet Constraints",
        fontsize=14,
    )
    axes[0].set_ylabel("Extraction Rate (t/d)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title(
        "Simulation: Lost Rate (Deficit) due to Fleet Limits", fontsize=14
    )
    axes[1].set_ylabel("Capacity Gap (t/d)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Simulation: Ore 2 Stockpile Depletion", fontsize=14)
    axes[2].set_ylabel("Ore 2 Stock (t)")
    axes[2].set_xlabel("Time (days)")
    axes[2].axhline(
        y=base_config.critical_ore2_level,
        color="red",
        linestyle=":",
        label="Critical Limit",
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(
        "docs/assets/comparisons/Infinite_vs_Constrained_Simulation.png", dpi=300
    )
    print("Saved docs/assets/comparisons/Infinite_vs_Constrained_Simulation.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_theoretical_fleet_diminishing_returns()
    generate_infinite_vs_constrained_plots()

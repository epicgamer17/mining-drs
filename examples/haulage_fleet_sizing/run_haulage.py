"""Example: Open-Pit Haulage Cycle Time Breakdown & Fleet Sizing.

Demonstrates open-pit truck and shovel fleet sizing adhering to SME Mining
Engineering Handbook (§9.2) (TODO: Manually Verify) and Caterpillar Performance
Handbook principles:
1. Configures mine-specific parameters without reliance on prohibited defaults:
   - Fixed cycle time (spotting, loading passes, turning, dumping).
   - Loaded haul speed vs. empty return speed.
   - Mechanical availability and job operating efficiency.
2. Evaluates cycle time breakdown and congestion queueing penalty.
3. Sizes the haul truck fleet to satisfy a target daily concentrator feed rate.
4. Performs parametric sensitivity analyses:
   - Pit deepening: Fleet requirements vs. one-way haul distance (1.0 to 6.0 km).
   - Payload selection: Fleet sizing comparing 100t, 150t, 220t, and 300t classes.
5. Generates a 4-panel engineering diagnostic dashboard.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drs_mining.components.logistics import (
    truck_haul_capacity,
    truck_cycle_time_breakdown,
)


def run_haulage_analysis(
    target_tonnes_per_day: float = 45000.0,
    distance_km: float = 3.5,
    truck_payload_tonnes: float = 220.0,
    fixed_cycle_time_min: float = 4.2,
    haul_speed_kmh: float = 22.0,
    return_speed_kmh: float = 38.0,
    mechanical_availability: float = 0.85,
    operator_efficiency: float = 0.88,
    congestion_factor: float = 0.04,
    operating_hours_per_day: float = 24.0,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Computes base circuit performance, distance sensitivity, and payload sensitivity."""
    # 1. Base circuit sizing: determine minimum fleet size to hit target
    required_trucks = 1
    while True:
        cap = truck_haul_capacity(
            distance_km=distance_km,
            num_trucks=required_trucks,
            truck_payload_tonnes=truck_payload_tonnes,
            mechanical_availability=mechanical_availability,
            operator_efficiency=operator_efficiency,
            fixed_cycle_time_min=fixed_cycle_time_min,
            haul_speed_kmh=haul_speed_kmh,
            return_speed_kmh=return_speed_kmh,
            congestion_factor=congestion_factor,
            operating_hours_per_day=operating_hours_per_day,
        )
        if cap >= target_tonnes_per_day or required_trucks > 60:
            break
        required_trucks += 1

    base_breakdown = truck_cycle_time_breakdown(
        distance_km=distance_km,
        num_trucks=required_trucks,
        truck_payload_tonnes=truck_payload_tonnes,
        mechanical_availability=mechanical_availability,
        operator_efficiency=operator_efficiency,
        fixed_cycle_time_min=fixed_cycle_time_min,
        haul_speed_kmh=haul_speed_kmh,
        return_speed_kmh=return_speed_kmh,
        congestion_factor=congestion_factor,
        operating_hours_per_day=operating_hours_per_day,
    )
    base_breakdown["allocated_trucks"] = required_trucks
    base_breakdown["target_tonnes_per_day"] = target_tonnes_per_day

    # 2. Distance Sensitivity (Pit Deepening from 1.0 km to 6.0 km)
    distances = np.linspace(1.0, 6.0, 11)
    dist_records = []
    for d in distances:
        n = 1
        while True:
            cap = truck_haul_capacity(
                distance_km=d,
                num_trucks=n,
                truck_payload_tonnes=truck_payload_tonnes,
                mechanical_availability=mechanical_availability,
                operator_efficiency=operator_efficiency,
                fixed_cycle_time_min=fixed_cycle_time_min,
                haul_speed_kmh=haul_speed_kmh,
                return_speed_kmh=return_speed_kmh,
                congestion_factor=congestion_factor,
                operating_hours_per_day=operating_hours_per_day,
            )
            if cap >= target_tonnes_per_day or n >= 80:
                break
            n += 1
        bd = truck_cycle_time_breakdown(
            distance_km=d,
            num_trucks=n,
            truck_payload_tonnes=truck_payload_tonnes,
            mechanical_availability=mechanical_availability,
            operator_efficiency=operator_efficiency,
            fixed_cycle_time_min=fixed_cycle_time_min,
            haul_speed_kmh=haul_speed_kmh,
            return_speed_kmh=return_speed_kmh,
            congestion_factor=congestion_factor,
            operating_hours_per_day=operating_hours_per_day,
        )
        dist_records.append({
            "distance_km": d,
            "required_trucks": n,
            "cycle_time_min": bd["total_cycle_time_min"],
            "trips_per_truck_day": bd["trips_per_truck_day"],
            "fleet_capacity_tpd": bd["fleet_daily_tonnes"],
        })
    df_dist = pd.DataFrame(dist_records)

    # 3. Payload Class Sensitivity (100t, 140t, 180t, 220t, 300t ultra-class)
    payload_classes = [100.0, 140.0, 180.0, 220.0, 300.0]
    # Fixed times scale slightly with payload class due to shovel pass matching
    fixed_times = {100.0: 3.2, 140.0: 3.6, 180.0: 3.9, 220.0: 4.2, 300.0: 5.0}
    payload_records = []
    for p in payload_classes:
        ft = fixed_times[p]
        n = 1
        while True:
            cap = truck_haul_capacity(
                distance_km=distance_km,
                num_trucks=n,
                truck_payload_tonnes=p,
                mechanical_availability=mechanical_availability,
                operator_efficiency=operator_efficiency,
                fixed_cycle_time_min=ft,
                haul_speed_kmh=haul_speed_kmh,
                return_speed_kmh=return_speed_kmh,
                congestion_factor=congestion_factor,
                operating_hours_per_day=operating_hours_per_day,
            )
            if cap >= target_tonnes_per_day or n >= 80:
                break
            n += 1
        bd = truck_cycle_time_breakdown(
            distance_km=distance_km,
            num_trucks=n,
            truck_payload_tonnes=p,
            mechanical_availability=mechanical_availability,
            operator_efficiency=operator_efficiency,
            fixed_cycle_time_min=ft,
            haul_speed_kmh=haul_speed_kmh,
            return_speed_kmh=return_speed_kmh,
            congestion_factor=congestion_factor,
            operating_hours_per_day=operating_hours_per_day,
        )
        payload_records.append({
            "payload_tonnes": p,
            "fixed_time_min": ft,
            "required_trucks": n,
            "cycle_time_min": bd["total_cycle_time_min"],
            "fleet_capacity_tpd": bd["fleet_daily_tonnes"],
        })
    df_payload = pd.DataFrame(payload_records)

    return base_breakdown, df_dist, df_payload


def plot_haulage_dashboard(
    base_breakdown: dict[str, float],
    df_dist: pd.DataFrame,
    df_payload: pd.DataFrame,
    distance_km: float,
    truck_payload_tonnes: float,
    fixed_cycle_time_min: float,
    haul_speed_kmh: float,
    return_speed_kmh: float,
    mechanical_availability: float,
    operator_efficiency: float,
    congestion_factor: float,
    title: str = "Open-Pit Mine Haulage Performance & Fleet Sizing Analysis",
    figsize: tuple[float, float] = (14.0, 10.0),
) -> tuple[plt.Figure, np.ndarray]:
    """Renders a comprehensive 4-panel diagnostic dashboard for haulage logistics."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: Cycle Time Breakdown vs Fleet Size (Congestion growth)
    ax1 = axes[0, 0]
    fleet_sizes = list(range(1, 25))
    fixed_times = []
    haul_times = []
    return_times = []
    queue_times = []
    for fs in fleet_sizes:
        bd = truck_cycle_time_breakdown(
            distance_km=distance_km,
            num_trucks=fs,
            truck_payload_tonnes=truck_payload_tonnes,
            mechanical_availability=mechanical_availability,
            operator_efficiency=operator_efficiency,
            fixed_cycle_time_min=fixed_cycle_time_min,
            haul_speed_kmh=haul_speed_kmh,
            return_speed_kmh=return_speed_kmh,
            congestion_factor=congestion_factor,
        )
        ft = bd["fixed_time_min"]
        ht = bd["haul_travel_min"]
        rt = bd["return_travel_min"]
        tot = bd["total_cycle_time_min"]
        qt = max(0.0, tot - (ft + ht + rt))
        fixed_times.append(ft)
        haul_times.append(ht)
        return_times.append(rt)
        queue_times.append(qt)

    x_arr = np.array(fleet_sizes)
    ax1.bar(x_arr, fixed_times, label="Fixed (Spot/Load/Dump)", color="#3498db", alpha=0.9)
    ax1.bar(x_arr, haul_times, bottom=np.array(fixed_times), label="Loaded Haul Uphill", color="#e67e22", alpha=0.9)
    ax1.bar(x_arr, return_times, bottom=np.array(fixed_times) + np.array(haul_times), label="Empty Return Downhill", color="#2ecc71", alpha=0.9)
    ax1.bar(x_arr, queue_times, bottom=np.array(fixed_times) + np.array(haul_times) + np.array(return_times), label="Queue / Congestion Delay", color="#e74c3c", alpha=0.9)
    ax1.axvline(base_breakdown["allocated_trucks"], color="black", linestyle="--", linewidth=1.5, label=f"Selected Fleet (N={int(base_breakdown['allocated_trucks'])})")
    ax1.set_xlabel("Fleet Size (Number of Active Trucks)", fontweight="bold")
    ax1.set_ylabel("Round-Trip Cycle Time (minutes)", fontweight="bold")
    ax1.set_title("Panel A: Cycle Time Expansion vs. Fleet Congestion", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: Fleet Daily Capacity & Law of Diminishing Returns
    ax2 = axes[0, 1]
    fleet_capacities = []
    for fs in fleet_sizes:
        cap = truck_haul_capacity(
            distance_km=distance_km,
            num_trucks=fs,
            truck_payload_tonnes=truck_payload_tonnes,
            mechanical_availability=mechanical_availability,
            operator_efficiency=operator_efficiency,
            fixed_cycle_time_min=fixed_cycle_time_min,
            haul_speed_kmh=haul_speed_kmh,
            return_speed_kmh=return_speed_kmh,
            congestion_factor=congestion_factor,
        )
        fleet_capacities.append(cap / 1e3)  # kt/day

    ax2.plot(fleet_sizes, fleet_capacities, marker="o", color="#2980b9", linewidth=2.0, label="Actual Capacity (with Congestion)")
    # Theoretical linear capacity without congestion
    linear_caps = [fleet_capacities[0] * fs for fs in fleet_sizes]
    ax2.plot(fleet_sizes, linear_caps, linestyle=":", color="#7f8c8d", label="Theoretical Uncongested Capacity")
    ax2.axhline(base_breakdown["target_tonnes_per_day"] / 1e3, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Target ({base_breakdown['target_tonnes_per_day']/1e3:.1f} kt/day)")
    ax2.scatter([base_breakdown["allocated_trucks"]], [base_breakdown["fleet_daily_tonnes"] / 1e3], color="#e74c3c", s=100, zorder=5)
    ax2.set_xlabel("Fleet Size (Number of Active Trucks)", fontweight="bold")
    ax2.set_ylabel("Fleet Haulage Capacity (kt / day)", fontweight="bold")
    ax2.set_title("Panel B: Daily Haulage Capacity vs. Target Feed Rate", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Panel 3: Required Fleet Size vs Haul Distance (Pit Deepening)
    ax3 = axes[1, 0]
    ax3.plot(df_dist["distance_km"], df_dist["required_trucks"], marker="s", color="#8e44ad", linewidth=2.0, label="Trucks Required")
    ax3.set_xlabel("One-Way Haul Distance (km)", fontweight="bold")
    ax3.set_ylabel("Required Trucks for Target Feed", fontweight="bold", color="#8e44ad")
    ax3.tick_params(axis="y", labelcolor="#8e44ad")
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(df_dist["distance_km"], df_dist["cycle_time_min"], marker="^", color="#d35400", linestyle="--", linewidth=1.5, label="Cycle Time")
    ax3_twin.set_ylabel("Cycle Time (min)", fontweight="bold", color="#d35400")
    ax3_twin.tick_params(axis="y", labelcolor="#d35400")
    ax3.set_title("Panel C: Pit Deepening Sensitivity (Distance vs. Fleet Size)", fontweight="bold")

    # Panel 4: Fleet Sizing vs Equipment Payload Class
    ax4 = axes[1, 1]
    bars = ax4.bar(
        [str(int(p)) + "t" for p in df_payload["payload_tonnes"]],
        df_payload["required_trucks"],
        color=["#16a085", "#1abc9c", "#27ae60", "#2ecc71", "#3498db"],
        width=0.6,
        edgecolor="black",
    )
    for bar, n in zip(bars, df_payload["required_trucks"]):
        ax4.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3, f"{int(n)} trucks", ha="center", va="bottom", fontweight="bold")
    ax4.set_xlabel("Truck Nominal Payload Class (tonnes)", fontweight="bold")
    ax4.set_ylabel("Required Fleet Size", fontweight="bold")
    ax4.set_ylim(0, df_payload["required_trucks"].max() * 1.25)
    ax4.set_title("Panel D: Equipment Sizing Trade-Off (Payload Class Matching)", fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6, axis="y")

    plt.tight_layout()
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description="Open-Pit Haulage Fleet Sizing Demonstrator")
    parser.add_argument(
        "--target-tpd",
        type=float,
        default=45000.0,
        help="Target daily ore + waste haulage throughput (tonnes/day)",
    )
    parser.add_argument(
        "--distance-km",
        type=float,
        default=3.5,
        help="One-way haul distance from loading shovel to primary crusher/dump (km)",
    )
    parser.add_argument(
        "--payload-tonnes",
        type=float,
        default=220.0,
        help="Nominal haul truck payload capacity (tonnes)",
    )
    parser.add_argument(
        "--fixed-time-min",
        type=float,
        default=4.2,
        help="Fixed cycle time in minutes (spot, load, turn, dump, delays)",
    )
    parser.add_argument(
        "--haul-speed",
        type=float,
        default=22.0,
        help="Average loaded uphill haul speed (km/h)",
    )
    parser.add_argument(
        "--return-speed",
        type=float,
        default=38.0,
        help="Average empty downhill return speed (km/h)",
    )
    parser.add_argument(
        "--availability",
        type=float,
        default=0.85,
        help="Fleet mechanical availability fraction (e.g. 0.85 for 85%)",
    )
    parser.add_argument(
        "--operator-efficiency",
        type=float,
        default=0.88,
        help="Job operational efficiency fraction (e.g. 0.88 for 88%)",
    )
    parser.add_argument(
        "--congestion-factor",
        type=float,
        default=0.04,
        help="Congestion queueing expansion per additional truck",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable dashboard plot generation and saving",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("      OPEN-PIT HAULAGE CYCLE TIME BREAKDOWN & FLEET SIZING ANALYSIS")
    print("=" * 80)
    print("Standards Compliance: SME Mining Engineering Handbook (§9.2) (TODO: Manually Verify)")
    print("                      Caterpillar Performance Handbook (TODO: Manually Verify)")
    print("-" * 80)

    print("\n[Step 1] Circuit Operational Parameters (All Mine-Specific Defaults Required):")
    print(f"  • Production Target            : {args.target_tpd:,.0f} tonnes/day")
    print(f"  • One-Way Haul Distance        : {args.distance_km:.2f} km")
    print(f"  • Truck Payload Rating         : {args.payload_tonnes:.0f} tonnes (CAT 793 Class)")
    print(f"  • Fixed Cycle Time             : {args.fixed_time_min:.2f} minutes")
    print(f"  • Loaded Haul Speed (Uphill)   : {args.haul_speed:.1f} km/h")
    print(f"  • Empty Return Speed (Downhill): {args.return_speed:.1f} km/h")
    print(f"  • Mechanical Availability      : {args.availability * 100:.1f}%")
    print(f"  • Job Operator Efficiency      : {args.operator_efficiency * 100:.1f}%")
    print(f"  • Traffic Congestion Penalty   : {args.congestion_factor * 100:.1f}% per truck")

    # Run core sizing and sensitivity routines
    base_bd, df_dist, df_payload = run_haulage_analysis(
        target_tonnes_per_day=args.target_tpd,
        distance_km=args.distance_km,
        truck_payload_tonnes=args.payload_tonnes,
        fixed_cycle_time_min=args.fixed_time_min,
        haul_speed_kmh=args.haul_speed,
        return_speed_kmh=args.return_speed,
        mechanical_availability=args.availability,
        operator_efficiency=args.operator_efficiency,
        congestion_factor=args.congestion_factor,
    )

    print("\n[Step 2] Base Circuit Sizing Results:")
    print(f"  • Required Active Fleet Size   : {int(base_bd['allocated_trucks'])} trucks")
    print(f"  • Fixed Spot/Load/Dump Time    : {base_bd['fixed_time_min']:.2f} min ({base_bd['fixed_time_min']/base_bd['total_cycle_time_min']*100:.1f}% of cycle)")
    print(f"  • Loaded Haul Travel Time      : {base_bd['haul_travel_min']:.2f} min ({base_bd['haul_travel_min']/base_bd['total_cycle_time_min']*100:.1f}% of cycle)")
    print(f"  • Empty Return Travel Time     : {base_bd['return_travel_min']:.2f} min ({base_bd['return_travel_min']/base_bd['total_cycle_time_min']*100:.1f}% of cycle)")
    print(f"  • Base Uncongested Cycle Time  : {base_bd['base_cycle_time_min']:.2f} min")
    print(f"  • Congestion Delay Multiplier  : {base_bd['congestion_multiplier']:.3f}x")
    print(f"  • Total Effective Cycle Time   : {base_bd['total_cycle_time_min']:.2f} min")
    print(f"  • Round Trips per Truck-Day    : {base_bd['trips_per_truck_day']:.1f} trips")
    print(f"  • Productivity per Truck-Day   : {base_bd['daily_tonnes_per_truck']:,.0f} tonnes/day")
    print(f"  • Fleet Total Daily Capacity   : {base_bd['fleet_daily_tonnes']:,.0f} tonnes/day ({(base_bd['fleet_daily_tonnes']/args.target_tpd - 1.0)*100:+.1f}% surplus)")

    print("\n[Step 3] Pit Deepening Sensitivity (Haul Distance 1.0 to 6.0 km):")
    cols_dist = ["distance_km", "required_trucks", "cycle_time_min", "trips_per_truck_day", "fleet_capacity_tpd"]
    print(df_dist[cols_dist].round(1).to_string(index=False))

    print("\n[Step 4] Equipment Payload Class Sizing Trade-Off:")
    cols_pay = ["payload_tonnes", "fixed_time_min", "required_trucks", "cycle_time_min", "fleet_capacity_tpd"]
    print(df_payload[cols_pay].round(1).to_string(index=False))

    if not args.no_plot:
        output_dir = Path("plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, _ = plot_haulage_dashboard(
            base_breakdown=base_bd,
            df_dist=df_dist,
            df_payload=df_payload,
            distance_km=args.distance_km,
            truck_payload_tonnes=args.payload_tonnes,
            fixed_cycle_time_min=args.fixed_time_min,
            haul_speed_kmh=args.haul_speed,
            return_speed_kmh=args.return_speed,
            mechanical_availability=args.availability,
            operator_efficiency=args.operator_efficiency,
            congestion_factor=args.congestion_factor,
        )
        dashboard_path = output_dir / "haulage_fleet_analysis.png"
        fig.savefig(dashboard_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[Step 5] Haulage dashboard saved to: {dashboard_path}")
    else:
        print("\n[Plotting skipped via --no-plot]")

    print("\n" + "=" * 80)
    print("                    HAULAGE ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

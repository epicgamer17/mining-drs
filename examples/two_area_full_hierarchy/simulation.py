"""Two-Area Full Hierarchy Simulation: Strategic Planning, Tactical Control, and Analytical Operational Blending.

Implements the unified three-level hierarchical decision framework:
  - Level 1 (Strategic): Discounted Cash Flow (DCF) NPV valuation & multi-year trajectory targets.
  - Level 2 (Tactical): Monthly progress reviews, trajectory ratios, and adaptive mining priority selection.
  - Level 3 (Operational): Closed-form analytical face-allocation equations solving exact mass rates and dispatch weights.

Comparative Benchmark:
  - Policy 1 (Myopic Baseline): Local tonnage maximization without capital development reservation.
  - Policy 2 (Hierarchical Value-Oriented Control with Analytical Blending): Proactive fleet reservation and analytical blending.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple, Any

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from drs_mining.config import FLEET_MODES
from drs_mining.components import (
    MiningSimulationBase,
    AreaReadinessTarget,
    StrategicYearTarget,
)
from drs_mining.components.allocation import solve_face_allocation_rates
from drs_mining.components.plot import (
    MODE_PALETTE,
    prepare_history,
    plot_full_hierarchy_dashboard,
    plot_ore_with_modes,
    plot_truck_idle_and_utilization,
    plot_mode_distribution,
    plot_two_area_dashboard,
    print_transition_log,
)


class TwoAreaFullHierarchyEngine(MiningSimulationBase):

    """Full 3-Level Hierarchical Simulation Engine with dual development tracking & analytical blending."""

    def __init__(
        self,
        policy: int = 2,
        enable_analytical_blending: bool = True,
        **kwargs,
    ):
        if policy == 1:
            kwargs["development_priority_truck_reservation_fraction"] = 0.0
            kwargs["area2_redeploy_locked_face_trucks_to_development"] = False
            kwargs["policy_name"] = "POLICY_1_MYOPIC"
        else:
            kwargs["policy_name"] = "POLICY_2_VALUE_ORIENTED"

        super().__init__(
            policy=policy,
            enable_analytical_blending=enable_analytical_blending,
            **kwargs,
        )



def print_full_hierarchy_summary(df_p1: pd.DataFrame, df_p2: pd.DataFrame) -> None:
    """Print comparative metrics between Policy 1 and Policy 2."""
    if df_p1.empty or df_p2.empty:
        print("Simulation histories are empty.")
        return

    p1_last = df_p1.iloc[-1]
    p2_last = df_p2.iloc[-1]

    days = p2_last.get("day", 0.0)
    face1_p1 = p1_last.get("face1_mined", p1_last.get("area1_mined", 0.0))
    face1_p2 = p2_last.get("face1_mined", p2_last.get("area1_mined", 0.0))
    face2_p1 = p1_last.get("face2_mined", p1_last.get("area2_mined", 0.0))
    face2_p2 = p2_last.get("face2_mined", p2_last.get("area2_mined", 0.0))
    tot_ore_p1 = p1_last.get("total_mined", face1_p1 + face2_p1)
    tot_ore_p2 = p2_last.get("total_mined", face1_p2 + face2_p2)

    dev_p1 = p1_last.get("cumulative_mine_development", p1_last.get("cumulative_development", 0.0))
    dev_p2 = p2_last.get("cumulative_mine_development", p2_last.get("cumulative_development", 0.0))
    cap_dev_p1 = p1_last.get("area2_cumulative_development", 0.0)
    cap_dev_p2 = p2_last.get("area2_cumulative_development", 0.0)

    # Unlock days
    u1 = df_p1[df_p1["area2_ready"] == True]
    u1_day = f"{float(u1['time'].iloc[0]):.1f} d" if not u1.empty else "N/A"
    u2 = df_p2[df_p2["area2_ready"] == True]
    u2_day = f"{float(u2['time'].iloc[0]):.1f} d" if not u2.empty else "N/A"

    # Depletion days
    d1_p1 = df_p1[df_p1.get("area1_exhausted", False) == True]
    d1_p1_day = f"{float(d1_p1['time'].iloc[0]):.1f} d" if not d1_p1.empty else "N/A"
    if d1_p1_day == "N/A" and "area1_depleted_day" in df_p1.columns:
        valid_p1 = df_p1[df_p1["area1_depleted_day"] >= 0.0]
        if not valid_p1.empty:
            d1_p1_day = f"{float(valid_p1['area1_depleted_day'].iloc[0]):.1f} d"

    d1_p2 = df_p2[df_p2.get("area1_exhausted", False) == True]
    d1_p2_day = f"{float(d1_p2['time'].iloc[0]):.1f} d" if not d1_p2.empty else "N/A"
    if d1_p2_day == "N/A" and "area1_depleted_day" in df_p2.columns:
        valid_p2 = df_p2[df_p2["area1_depleted_day"] >= 0.0]
        if not valid_p2.empty:
            d1_p2_day = f"{float(valid_p2['area1_depleted_day'].iloc[0]):.1f} d"

    d2_p1 = df_p1[df_p1.get("area2_exhausted", False) == True]
    d2_p1_day = f"{float(d2_p1['time'].iloc[0]):.1f} d" if not d2_p1.empty else "N/A"
    if d2_p1_day == "N/A" and "area2_depleted_day" in df_p1.columns:
        valid2_p1 = df_p1[df_p1["area2_depleted_day"] >= 0.0]
        if not valid2_p1.empty:
            d2_p1_day = f"{float(valid2_p1['area2_depleted_day'].iloc[0]):.1f} d"

    d2_p2 = df_p2[df_p2.get("area2_exhausted", False) == True]
    d2_p2_day = f"{float(d2_p2['time'].iloc[0]):.1f} d" if not d2_p2.empty else "N/A"
    if d2_p2_day == "N/A" and "area2_depleted_day" in df_p2.columns:
        valid2_p2 = df_p2[df_p2["area2_depleted_day"] >= 0.0]
        if not valid2_p2.empty:
            d2_p2_day = f"{float(valid2_p2['area2_depleted_day'].iloc[0]):.1f} d"

    npv_p1 = p1_last.get("cumulative_npv", 0.0)
    npv_p2 = p2_last.get("cumulative_npv", 0.0)
    npv_gain = npv_p2 - npv_p1
    npv_gain_pct = (npv_gain / abs(npv_p1) * 100.0) if abs(npv_p1) > 1e-6 else 0.0

    print("\n" + "=" * 78)
    print(f"      THREE-LEVEL HIERARCHY COMPARISON SUMMARY ({days:.0f} DAYS)")
    print("=" * 78)
    print(f"{'Metric':<40} {'Policy 1 (Myopic)':<18} {'Policy 2 (Hierarchy)':<18}")
    print("-" * 78)
    print(f"{'Total Ore Mined (t)':<40} {tot_ore_p1:>16,.1f}  {tot_ore_p2:>16,.1f}")
    print(f"{'  ↳ Face 1 Ore (Level 3) (t)':<40} {face1_p1:>16,.1f}  {face1_p2:>16,.1f}")
    print(f"{'  ↳ Face 2 Ore (Level 6) (t)':<40} {face2_p1:>16,.1f}  {face2_p2:>16,.1f}")
    print(f"{'Area 2 Unlock Day':<40} {u1_day:>16}  {u2_day:>16}")
    print(f"{'Area 1 Depletion Day':<40} {d1_p1_day:>16}  {d1_p2_day:>16}")
    print(f"{'Area 2 Depletion Day':<40} {d2_p1_day:>16}  {d2_p2_day:>16}")
    print(f"{'Area 2 Capital Dev Advance (m)':<40} {cap_dev_p1:>16,.1f}  {cap_dev_p2:>16,.1f}")
    print(f"{'Total Mine Development Advance (m)':<40} {dev_p1:>16,.1f}  {dev_p2:>16,.1f}")
    print(f"{'Cumulative Net NPV ($M)':<40} {npv_p1/1e6:>16.2f}M {npv_p2/1e6:>16.2f}M")
    print(f"{'Incremental NPV Benefit ($M)':<40} {'-':>16}  {npv_gain/1e6:>+15.2f}M ({npv_gain_pct:+.1f}%)")
    print("=" * 78 + "\n")




def run_full_hierarchy_study(
    total_days: Optional[float] = None,
    total_ore_to_extract: float = 6600000.0,
    warmup_ore: float = 0.0,
    area2_required_dev: float = 4000.0,
    area2_ready_by_day: float = 365.0,
    num_trucks: int = 18,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes Policy 1 and Policy 2 full hierarchy study until Area 1 and Area 2 are both exhausted (or total_days)."""
    strat_target = StrategicYearTarget(
        min_development=10000.0,
        min_ore1_production=1300000.0,
        min_ore2_production=850000.0,
    )
    area2_target = AreaReadinessTarget(
        required_development=area2_required_dev,
        ready_by_day=area2_ready_by_day,
    )

    duration_p1 = (total_days * 86400.0) if total_days is not None else float("inf")
    duration_p2 = (total_days * 86400.0) if total_days is not None else float("inf")

    print("Running Policy 1 (Myopic Baseline)...")
    sim_p1 = TwoAreaFullHierarchyEngine(
        policy=1,
        enable_analytical_blending=False,
        num_trucks=num_trucks,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warmup_ore,
        strategic_targets=(strat_target,),
        area2_readiness_target=area2_target,
        seed=seed,
    )
    sim_p1.step(duration_p1)
    df_p1 = pd.DataFrame(sim_p1.telemetry_history)

    print("Running Policy 2 (Three-Level Hierarchical Control)...")
    sim_p2 = TwoAreaFullHierarchyEngine(
        policy=2,
        enable_analytical_blending=True,
        num_trucks=num_trucks,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warmup_ore,
        strategic_targets=(strat_target,),
        area2_readiness_target=area2_target,
        seed=seed,
    )
    sim_p2.step(duration_p2)
    df_p2 = pd.DataFrame(sim_p2.telemetry_history)

    print_full_hierarchy_summary(df_p1, df_p2)

    if plot and len(df_p2) > 0:
        print(f"Generating dashboard plot...")
        plot_full_hierarchy_dashboard(df_p1, df_p2)

    return df_p1, df_p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full Three-Level Hierarchy Simulation Study"
    )
    parser.add_argument("--total_days", type=float, default=None, help="Total days to simulate (default: run until Area 1 and Area 2 are both depleted)")
    parser.add_argument("--total_ore", type=float, default=6600000.0, help="Total ore to extract across Area 1 and Area 2 (default: 6,600,000 t)")
    parser.add_argument("--warmup_ore", type=float, default=0.0)
    parser.add_argument("--area2_required_dev", type=float, default=4000.0)
    parser.add_argument("--area2_ready_by_day", type=float, default=365.0)
    parser.add_argument("--trucks", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    run_full_hierarchy_study(
        total_days=args.total_days,
        total_ore_to_extract=args.total_ore,
        warmup_ore=args.warmup_ore,
        area2_required_dev=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        num_trucks=args.trucks,
        seed=args.seed,
        plot=not args.no_plot,
    )

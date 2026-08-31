"""Policy 1 (Local-Objective Myopic Baseline) vs Policy 2 (Value-Oriented Control).

Evaluates:
- Policy 1 (Myopic Baseline): Maximizes instantaneous production, ignoring development targets.
- Policy 2 (Hierarchical Value-Oriented Control): Proactively monitors trajectory ratios
  and reserves haulage capacity to unlock Area 2 on time, maximizing NPV.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from drs_mining.config import (
    SimulationConfig,
    DEFAULT_CONFIG,
    FLEET_MODES,
)
from drs_mining.components import (
    MiningSimulationBase,
    AreaReadinessTarget,
    StrategicYearTarget,
)
from drs_mining.components.plot import (
    plot_two_area_dashboard,
    prepare_history,
    print_transition_log,
)


class TwoAreaPolicySimulationEngine(MiningSimulationBase):
    """Simulation engine supporting Policy 1 (Myopic) and Policy 2 (Value-Oriented)."""

    def __init__(
        self,
        policy: int = 2,
        **kwargs,
    ):
        if policy == 1:
            kwargs["development_priority_truck_reservation_fraction"] = 0.0
            kwargs["area2_redeploy_locked_face_trucks_to_development"] = False
            kwargs["policy_name"] = "POLICY_1_MYOPIC"
        else:
            kwargs["policy_name"] = "POLICY_2_VALUE_ORIENTED"
        super().__init__(policy=policy, **kwargs)


def plot_policy_comparison_dashboard(
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    output_path: str = "plots/policy_comparison_dashboard.png",
    **kwargs,
):
    """Plots comparative metrics between Policy 1 and Policy 2."""
    return plot_two_area_dashboard(
        df_p2,
        output_path=output_path,
        title="Policy 1 (Myopic) vs Policy 2 (Hierarchical Value-Oriented)",
        **kwargs,
    )


def print_policy_comparison_summary(df_p1: pd.DataFrame, df_p2: pd.DataFrame) -> None:
    """Print comparative metrics between Policy 1 and Policy 2."""
    if df_p1.empty or df_p2.empty:
        print("Simulation histories are empty.")
        return

    p1_last = df_p1.iloc[-1]
    p2_last = df_p2.iloc[-1]

    days = p2_last.get("day", 0.0)
    ore1_p1 = p1_last.get("ore1_mined", p1_last.get("ore1_dumped_total", 0.0))
    ore1_p2 = p2_last.get("ore1_mined", p2_last.get("ore1_dumped_total", 0.0))
    ore2_p1 = p1_last.get("ore2_mined", p1_last.get("ore2_dumped_total", 0.0))
    ore2_p2 = p2_last.get("ore2_mined", p2_last.get("ore2_dumped_total", 0.0))
    tot_ore_p1 = p1_last.get("total_mined", ore1_p1 + ore2_p1)
    tot_ore_p2 = p2_last.get("total_mined", ore1_p2 + ore2_p2)

    dev_p1 = p1_last.get("cumulative_development", 0.0)
    dev_p2 = p2_last.get("cumulative_development", 0.0)

    npv_p1 = p1_last.get("cumulative_npv", 0.0)
    npv_p2 = p2_last.get("cumulative_npv", 0.0)
    npv_gain = npv_p2 - npv_p1
    npv_gain_pct = (npv_gain / abs(npv_p1) * 100.0) if abs(npv_p1) > 1e-6 else 0.0

    print("\n" + "=" * 78)
    print(f"      POLICY 1 vs POLICY 2 COMPARISON SUMMARY ({days:.0f} DAYS)")
    print("=" * 78)
    print(f"{'Metric':<40} {'Policy 1 (Myopic)':<18} {'Policy 2 (Value-Opt)':<18}")
    print("-" * 78)
    print(f"{'Total Ore Mined (t)':<40} {tot_ore_p1:>16,.1f}  {tot_ore_p2:>16,.1f}")
    print(f"{'  ↳ Area 1 Ore (t)':<40} {ore1_p1:>16,.1f}  {ore1_p2:>16,.1f}")
    print(f"{'  ↳ Area 2 Ore (t)':<40} {ore2_p1:>16,.1f}  {ore2_p2:>16,.1f}")
    print(f"{'Capital Dev Advance (m)':<40} {dev_p1:>16,.1f}  {dev_p2:>16,.1f}")
    print(f"{'Cumulative Net NPV ($M)':<40} {npv_p1/1e6:>16.2f}M {npv_p2/1e6:>16.2f}M")
    print(f"{'Incremental NPV Benefit ($M)':<40} {'-':>16}  {npv_gain/1e6:>+15.2f}M ({npv_gain_pct:+.1f}%)")
    print("=" * 78 + "\n")


def run_policy_comparison_study(
    config: Optional[SimulationConfig] = None,
    total_days: Optional[float] = None,
    warmup_ore: Optional[float] = None,
    area2_required_dev: Optional[float] = None,
    area2_ready_by_day: Optional[float] = None,
    num_trucks: Optional[int] = None,
    seed: Optional[int] = None,
    plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes Policy 1 and Policy 2 side-by-side study."""
    cfg = config or DEFAULT_CONFIG

    strat_target = StrategicYearTarget(
        min_development=cfg.planning.annual_min_development_m,
        min_ore1_production=cfg.planning.annual_min_ore1_production_t,
        min_ore2_production=cfg.planning.annual_min_ore2_production_t,
    )
    req_dev = area2_required_dev if area2_required_dev is not None else cfg.planning.area2_required_development
    rdy_day = area2_ready_by_day if area2_ready_by_day is not None else cfg.planning.area2_ready_by_day
    area2_target = AreaReadinessTarget(
        required_development=req_dev,
        ready_by_day=rdy_day,
    )

    days = total_days if total_days is not None else cfg.total_days
    warm = warmup_ore if warmup_ore is not None else cfg.plant.ore_to_be_extracted_during_warming_period

    print("Running Policy 1 (Myopic Baseline)...")
    sim_p1 = TwoAreaPolicySimulationEngine(
        config=cfg,
        policy=1,
        num_trucks=num_trucks,
        ore_to_be_extracted_during_warming_period=warm,
        strategic_targets=(strat_target,),
        area2_readiness_target=area2_target,
        seed=seed,
    )
    sim_p1.step(days * 86400.0)
    df_p1 = pd.DataFrame(sim_p1.telemetry_history)

    print("Running Policy 2 (Hierarchical Value-Oriented Control)...")
    sim_p2 = TwoAreaPolicySimulationEngine(
        config=cfg,
        policy=2,
        num_trucks=num_trucks,
        ore_to_be_extracted_during_warming_period=warm,
        strategic_targets=(strat_target,),
        area2_readiness_target=area2_target,
        seed=seed,
    )
    sim_p2.step(days * 86400.0)
    df_p2 = pd.DataFrame(sim_p2.telemetry_history)

    print_policy_comparison_summary(df_p1, df_p2)

    if plot and len(df_p2) > 0:
        print("Generating dashboard plot...")
        plot_policy_comparison_dashboard(df_p1, df_p2)

    return df_p1, df_p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Policy 1 vs Policy 2 Comparison Simulation"
    )
    parser.add_argument(
        "--total_days",
        type=float,
        default=DEFAULT_CONFIG.total_days,
        help=f"Simulation duration in days (default: {DEFAULT_CONFIG.total_days:,.1f} d)",
    )
    parser.add_argument(
        "--warmup_ore",
        type=float,
        default=DEFAULT_CONFIG.plant.ore_to_be_extracted_during_warming_period,
        help=f"Warmup period ore tonnage (default: {DEFAULT_CONFIG.plant.ore_to_be_extracted_during_warming_period:,.1f} t)",
    )
    parser.add_argument(
        "--area2_required_dev",
        type=float,
        default=DEFAULT_CONFIG.planning.area2_required_development,
        help=f"Required development metres to unlock Area 2 (default: {DEFAULT_CONFIG.planning.area2_required_development:,.1f} m)",
    )
    parser.add_argument(
        "--area2_ready_by_day",
        type=float,
        default=DEFAULT_CONFIG.planning.area2_ready_by_day,
        help=f"Target schedule deadline for Area 2 (default: {DEFAULT_CONFIG.planning.area2_ready_by_day:,.1f} d)",
    )
    parser.add_argument(
        "--trucks",
        type=int,
        default=DEFAULT_CONFIG.fleet.num_trucks,
        help=f"Number of haulage trucks (default: {DEFAULT_CONFIG.fleet.num_trucks})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.seed,
        help=f"Random seed for reproducibility (default: {DEFAULT_CONFIG.seed})",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable dashboard plot generation",
    )
    args = parser.parse_args()

    run_policy_comparison_study(
        total_days=args.total_days,
        warmup_ore=args.warmup_ore,
        area2_required_dev=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        num_trucks=args.trucks,
        seed=args.seed,
        plot=not args.no_plot,
    )

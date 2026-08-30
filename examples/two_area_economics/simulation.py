"""Two-Area Strategic Planning, Area 2 Readiness, and Counterfactual Incremental NPV.

Combines:
1. Two Distinct Mining Areas / Faces with Stochastic Geological Facies (Area 1 Level 3, Area 2 Level 6).
2. Area 2 Readiness & Physical Unlock Mechanics.
3. Strategic & Tactical Planning Framework (Annual StrategicYearTarget, Monthly Tactical Review).
4. Strategic Economics & Counterfactual Incremental NPV Evaluation (WITH Area 2 vs WITHOUT Area 2).
5. Shelswell (2017) DES Haulage Engine & DRS Blending Modes.
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

import drs
import pandas as pd

from drs_mining.components import (
    MiningSimulationBase,
    AreaReadinessTarget,
    StrategicYearTarget,
)
from drs_mining.components.plot import (
    plot_two_area_dashboard,
    print_strategic_economic_summary,
    prepare_history,
    print_transition_log,
)


class TwoAreaEconomicSimulation(MiningSimulationBase):

    """Two-Area Strategic DES Simulation with Area 2 Readiness & Discounted Cash Flow Economics."""
    pass


def plot_two_area_economic_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/two_area_economic_dashboard.png",
    **kwargs,
):
    """Builds and saves the comprehensive economic & operational diagnostics dashboard."""
    return plot_two_area_dashboard(
        df,
        output_path=output_path,
        title="Two-Area Strategic Planning, Area 2 Readiness & Discounted Cash Flow Economics",
        **kwargs,
    )


def run_two_area_economic_simulation(
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    availability: float = 0.85,
    target_ore_stock_level: float = 60000.0,
    strategic_target: Optional[StrategicYearTarget] = None,
    area2_target: Optional[AreaReadinessTarget] = None,
    area2_required_development: float = 4000.0,
    area2_ready_by_day: float = 365.0,
    annual_discount_rate: float = 0.05,
    ore1_net_value: float = 577.48,
    ore2_net_value: float = 709.83,
    production_cost: float = 135.0,
    development_cost: float = 15000.0,
    fixed_cost: float = 74460.0,
    seed: int = 42,
    run_counterfactual: bool = True,
    plot: bool = True,
) -> Tuple[TwoAreaEconomicSimulation, pd.DataFrame]:
    """Runs the two-area strategic DCF economic simulation and counterfactual."""
    if strategic_target is None:
        strategic_target = StrategicYearTarget(
            min_development=10000.0,
            min_ore1_production=1300000.0,
            min_ore2_production=850000.0,
        )
    if area2_target is None:
        area2_target = AreaReadinessTarget(
            required_development=area2_required_development,
            ready_by_day=area2_ready_by_day,
        )
    warmup = 0.0 if total_days is not None else ore_to_be_extracted_during_warming_period

    # 1. Base Case: WITH Area 2
    print("\n" + "=" * 70)
    print(" RUNNING BASE CASE: WITH AREA 2 CAPITAL EXPANSION")
    print("=" * 70)
    sim_with = TwoAreaEconomicSimulation(
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warmup,
        strategic_targets=(strategic_target,),
        area2_readiness_target=area2_target,
        area2_physical_unlock_enabled=True,
        annual_discount_rate=annual_discount_rate,
        ore1_net_value_per_processed_tonne=ore1_net_value,
        ore2_net_value_per_processed_tonne=ore2_net_value,
        production_cost_per_tonne=production_cost,
        development_cost_per_unit=development_cost,
        fixed_cost_per_day=fixed_cost,
        seed=seed,
    )

    days_to_run = total_days if total_days is not None else 365.0
    sim_with.step(days_to_run * 86400.0)
    df_with = pd.DataFrame(sim_with.telemetry_history)

    # 2. Counterfactual: WITHOUT Area 2 (if requested)
    df_without = None
    if run_counterfactual:
        sim_without = TwoAreaEconomicSimulation(
            num_trucks=num_trucks,
            num_operators=num_operators,
            availability=availability,
            target_ore_stock_level=target_ore_stock_level,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=warmup,
            strategic_targets=(strategic_target,),
            area2_readiness_target=area2_target,
            area2_counterfactual_disable=True,
            annual_discount_rate=annual_discount_rate,
            ore1_net_value_per_processed_tonne=ore1_net_value,
            ore2_net_value_per_processed_tonne=ore2_net_value,
            production_cost_per_tonne=production_cost,
            development_cost_per_unit=development_cost,
            fixed_cost_per_day=fixed_cost,
            seed=seed,
        )
        sim_without.step(days_to_run * 86400.0)
        df_without = pd.DataFrame(sim_without.telemetry_history)

    print_strategic_economic_summary(df_with, df_without)

    df_prepared = prepare_history(df_with)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim_with.critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        label="Two-Area Economic Blending",
    )

    if plot and len(df_prepared) > 0:
        plot_two_area_economic_dashboard(df_prepared)

    return sim_with, df_prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Area Strategic Planning & Counterfactual Incremental NPV Simulation"
    )
    parser.add_argument(
        "--total_ore_to_extract",
        type=float,
        default=6600000.0,
        help="Total production ore tonnage to extract (default: 6,600,000.0 t)",
    )
    parser.add_argument(
        "--warmup_ore",
        type=float,
        default=600000.0,
        help="Warmup period ore tonnage to extract (default: 600,000.0 t)",
    )
    parser.add_argument(
        "--total_days",
        type=float,
        default=None,
        help="Total simulation duration in days (optional)",
    )
    parser.add_argument(
        "--trucks",
        type=int,
        default=18,
        help="Number of AD30 haulage trucks (default: 18)",
    )
    parser.add_argument(
        "--operators",
        type=int,
        default=18,
        help="Number of operators per shift (default: 18)",
    )
    parser.add_argument(
        "--availability",
        type=float,
        default=0.85,
        help="Mechanical availability fraction (default: 0.85)",
    )
    parser.add_argument(
        "--stockpile_target",
        type=float,
        default=60000.0,
        help="Target total ore stockpile buffer (default: 60000.0 t)",
    )
    parser.add_argument(
        "--area2_required_dev",
        type=float,
        default=4000.0,
        help="Required development metres to unlock Area 2 (default: 4,000.0 m)",
    )
    parser.add_argument(
        "--area2_ready_by_day",
        type=float,
        default=365.0,
        help="Target schedule deadline for Area 2 (default: 365.0 d)",
    )
    parser.add_argument(
        "--discount_rate",
        type=float,
        default=0.05,
        help="Annual discount rate (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip_counterfactual",
        action="store_true",
        help="Skip the WITHOUT Area 2 counterfactual run",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable dashboard plot generation",
    )
    args = parser.parse_args()

    run_two_area_economic_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        availability=args.availability,
        target_ore_stock_level=args.stockpile_target,
        area2_required_development=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        annual_discount_rate=args.discount_rate,
        seed=args.seed,
        run_counterfactual=not args.skip_counterfactual,
        plot=not args.no_plot,
    )

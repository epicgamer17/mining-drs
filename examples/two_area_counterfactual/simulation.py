"""Two-Area Strategic Planning & Counterfactual Incremental NPV Simulation.

Runs a side-by-side comparative simulation of:
1. Base Case (WITH Area 2 Capital Expansion)
2. Counterfactual Baseline (WITHOUT Area 2 Capital Expansion)

Calculates the True Incremental Net Present Value (NPV):
   Incremental NPV = NPV(WITH Area 2) - NPV(WITHOUT Area 2)
using identical random seeds for an exact, paired counterfactual comparison.
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


class TwoAreaCounterfactualSimulation(MiningSimulationBase):

    """Two-Area Strategic DES Simulation for Counterfactual Incremental NPV Analysis."""
    pass


def run_two_area_counterfactual_simulation(
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
    seed: int = 42,
    plot: bool = True,
) -> Tuple[TwoAreaCounterfactualSimulation, TwoAreaCounterfactualSimulation, pd.DataFrame, pd.DataFrame, float]:
    """Runs paired counterfactual simulations WITH and WITHOUT Area 2."""
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

    days_to_run = total_days if total_days is not None else 365.0

    # 1. Base Case: WITH Area 2
    sim_with = TwoAreaCounterfactualSimulation(
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        strategic_targets=(strategic_target,),
        area2_readiness_target=area2_target,
        area2_physical_unlock_enabled=True,
        annual_discount_rate=annual_discount_rate,
        seed=seed,
    )
    sim_with.step(days_to_run * 86400.0)
    df_with = pd.DataFrame(sim_with.telemetry_history)

    # 2. Counterfactual Case: WITHOUT Area 2
    sim_without = TwoAreaCounterfactualSimulation(
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        strategic_targets=(strategic_target,),
        area2_readiness_target=area2_target,
        area2_counterfactual_disable=True,
        annual_discount_rate=annual_discount_rate,
        seed=seed,
    )
    sim_without.step(days_to_run * 86400.0)
    df_without = pd.DataFrame(sim_without.telemetry_history)

    npv_with = sim_with.economics.cumulative_npv
    npv_without = sim_without.economics.cumulative_npv
    incremental_npv = npv_with - npv_without

    print_strategic_economic_summary(df_with, df_without)

    df_prepared = prepare_history(df_with)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim_with.critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        label="Counterfactual Blending",
    )

    if plot and len(df_prepared) > 0:
        plot_two_area_dashboard(
            df_prepared,
            output_path="plots/two_area_counterfactual_dashboard.png",
            title="Two-Area Counterfactual Incremental NPV Analysis",
        )

    return sim_with, sim_without, df_with, df_without, incremental_npv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Area Strategic Counterfactual Simulation"
    )
    parser.add_argument(
        "--total_ore_to_extract",
        type=float,
        default=6600000.0,
    )
    parser.add_argument(
        "--warmup_ore",
        type=float,
        default=600000.0,
    )
    parser.add_argument(
        "--total_days",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--trucks",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--area2_required_dev",
        type=float,
        default=4000.0,
    )
    parser.add_argument(
        "--area2_ready_by_day",
        type=float,
        default=365.0,
    )
    parser.add_argument(
        "--discount_rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
    )
    args = parser.parse_args()

    run_two_area_counterfactual_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        area2_required_development=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        annual_discount_rate=args.discount_rate,
        seed=args.seed,
        plot=not args.no_plot,
    )

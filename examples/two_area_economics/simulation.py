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

import pandas as pd

from drs_mining.config import (
    SimulationConfig,
    DEFAULT_CONFIG,
)
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
    config: Optional[SimulationConfig] = None,
    total_ore_to_extract: Optional[float] = None,
    ore_to_be_extracted_during_warming_period: Optional[float] = None,
    total_days: Optional[float] = None,
    num_trucks: Optional[int] = None,
    num_operators: Optional[int] = None,
    availability: Optional[float] = None,
    target_ore_stock_level: Optional[float] = None,
    strategic_target: Optional[StrategicYearTarget] = None,
    area2_target: Optional[AreaReadinessTarget] = None,
    area2_required_development: Optional[float] = None,
    area2_ready_by_day: Optional[float] = None,
    annual_discount_rate: Optional[float] = None,
    ore1_net_value: Optional[float] = None,
    ore2_net_value: Optional[float] = None,
    production_cost: Optional[float] = None,
    development_cost: Optional[float] = None,
    fixed_cost: Optional[float] = None,
    seed: Optional[int] = None,
    run_counterfactual: bool = True,
    plot: bool = True,
) -> Tuple[TwoAreaEconomicSimulation, pd.DataFrame]:
    """Runs the two-area strategic DCF economic simulation and counterfactual."""
    cfg = config or DEFAULT_CONFIG

    if strategic_target is None:
        strategic_target = StrategicYearTarget(
            min_development=cfg.planning.annual_min_development_m,
            min_ore1_production=cfg.planning.annual_min_ore1_production_t,
            min_ore2_production=cfg.planning.annual_min_ore2_production_t,
        )
    if area2_target is None:
        req_dev = area2_required_development if area2_required_development is not None else cfg.planning.area2_required_development
        rdy_day = area2_ready_by_day if area2_ready_by_day is not None else cfg.planning.area2_ready_by_day
        area2_target = AreaReadinessTarget(
            required_development=req_dev,
            ready_by_day=rdy_day,
        )

    warmup = (
        0.0 if total_days is not None
        else (ore_to_be_extracted_during_warming_period if ore_to_be_extracted_during_warming_period is not None
              else cfg.plant.ore_to_be_extracted_during_warming_period)
    )

    # 1. Base Case: WITH Area 2
    print("\n" + "=" * 70)
    print(" RUNNING BASE CASE: WITH AREA 2 CAPITAL EXPANSION")
    print("=" * 70)
    sim_with = TwoAreaEconomicSimulation(
        config=cfg,
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

    days_to_run = total_days if total_days is not None else cfg.total_days
    sim_with.step(days_to_run * 86400.0)
    df_with = pd.DataFrame(sim_with.telemetry_history)

    # 2. Counterfactual: WITHOUT Area 2 (if requested)
    df_without = None
    if run_counterfactual:
        sim_without = TwoAreaEconomicSimulation(
            config=cfg,
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

    stock_target = target_ore_stock_level if target_ore_stock_level is not None else cfg.plant.target_ore_stock_level
    df_prepared = prepare_history(df_with)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim_with.critical_ore2_level,
        target_ore_stock_level=stock_target,
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
        default=DEFAULT_CONFIG.plant.total_ore_to_extract,
        help=f"Total production ore tonnage to extract (default: {DEFAULT_CONFIG.plant.total_ore_to_extract:,.1f} t)",
    )
    parser.add_argument(
        "--warmup_ore",
        type=float,
        default=DEFAULT_CONFIG.plant.ore_to_be_extracted_during_warming_period,
        help=f"Warmup period ore tonnage to extract (default: {DEFAULT_CONFIG.plant.ore_to_be_extracted_during_warming_period:,.1f} t)",
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
        default=DEFAULT_CONFIG.fleet.num_trucks,
        help=f"Number of haulage trucks (default: {DEFAULT_CONFIG.fleet.num_trucks})",
    )
    parser.add_argument(
        "--operators",
        type=int,
        default=DEFAULT_CONFIG.fleet.num_operators,
        help=f"Number of operators per shift (default: {DEFAULT_CONFIG.fleet.num_operators})",
    )
    parser.add_argument(
        "--availability",
        type=float,
        default=DEFAULT_CONFIG.fleet.availability,
        help=f"Mechanical availability fraction (default: {DEFAULT_CONFIG.fleet.availability})",
    )
    parser.add_argument(
        "--stockpile_target",
        type=float,
        default=DEFAULT_CONFIG.plant.target_ore_stock_level,
        help=f"Target total ore stockpile buffer (default: {DEFAULT_CONFIG.plant.target_ore_stock_level:,.1f} t)",
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
        "--discount_rate",
        type=float,
        default=DEFAULT_CONFIG.economics.annual_discount_rate,
        help=f"Annual discount rate (default: {DEFAULT_CONFIG.economics.annual_discount_rate})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.seed,
        help=f"Random seed for reproducibility (default: {DEFAULT_CONFIG.seed})",
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

"""Two-Area Strategic & Tactical Planning DES Blending Modes Simulation.

Features:
1. Two Distinct Mining Areas / Faces with Stochastic Geological Facies.
2. Shelswell (2017) DES Haulage Engine.
3. Strategic & Tactical Planning Framework (Annual StrategicYearTarget, Monthly Tactical Review).
4. Dynamic Fleet Operating Modes (PRODUCTION, DEVELOPMENT).
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
    prepare_history,
    print_transition_log,
)


class TwoAreaStrategicSimulation(MiningSimulationBase):
    """Two-Area Strategic & Tactical Planning DES Blending Simulation."""
    pass


def plot_two_area_strategic_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/two_area_strategic_dashboard.png",
    **kwargs,
):
    """Builds and saves the comprehensive operational & strategic planning dashboard."""
    return plot_two_area_dashboard(
        df,
        output_path=output_path,
        title="Two-Area Strategic & Tactical Planning Simulation",
        **kwargs,
    )


def run_two_area_strategic_simulation(
    config: Optional[SimulationConfig] = None,
    total_ore_to_extract: Optional[float] = None,
    ore_to_be_extracted_during_warming_period: Optional[float] = None,
    total_days: Optional[float] = None,
    num_trucks: Optional[int] = None,
    num_operators: Optional[int] = None,
    availability: Optional[float] = None,
    target_ore_stock_level: Optional[float] = None,
    strategic_target: Optional[StrategicYearTarget] = None,
    seed: Optional[int] = None,
    plot: bool = True,
) -> Tuple[TwoAreaStrategicSimulation, pd.DataFrame]:
    """Runs the two-area strategic planning simulation."""
    cfg = config or DEFAULT_CONFIG

    if strategic_target is None:
        strategic_target = StrategicYearTarget(
            min_development=cfg.planning.annual_min_development_m,
            min_ore1_production=cfg.planning.annual_min_ore1_production_t,
            min_ore2_production=cfg.planning.annual_min_ore2_production_t,
        )

    sim = TwoAreaStrategicSimulation(
        config=cfg,
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        strategic_targets=(strategic_target,),
        area2_physical_unlock_enabled=False,
        seed=seed,
    )

    days_to_run = total_days if total_days is not None else cfg.total_days
    sim.step(days_to_run * 86400.0)
    df = pd.DataFrame(sim.telemetry_history)

    df_prepared = prepare_history(df)
    stock_target = target_ore_stock_level if target_ore_stock_level is not None else cfg.plant.target_ore_stock_level
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim.critical_ore2_level,
        target_ore_stock_level=stock_target,
        label="Two-Area Strategic Planning",
    )

    if plot and len(df_prepared) > 0:
        plot_two_area_strategic_dashboard(df_prepared)

    return sim, df_prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Area Strategic & Tactical Planning DES Simulation"
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

    run_two_area_strategic_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        availability=args.availability,
        target_ore_stock_level=args.stockpile_target,
        seed=args.seed,
        plot=not args.no_plot,
    )

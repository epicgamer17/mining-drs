"""Two-Area Strategic & Tactical Planning DES Blending Modes Simulation.

Features:
1. Two Distinct Mining Areas / Faces with Stochastic Geological Facies.
2. Shelswell (2017) DES Haulage Engine.
3. Strategic & Tactical Planning Framework (Annual StrategicYearTarget, Monthly Tactical Review).
4. Dynamic Fleet Operating Modes (BALANCED, PRODUCTION, DEVELOPMENT).
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
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    availability: float = 0.85,
    target_ore_stock_level: float = 60000.0,
    strategic_target: Optional[StrategicYearTarget] = None,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[TwoAreaStrategicSimulation, pd.DataFrame]:
    """Runs the two-area strategic planning simulation."""
    if strategic_target is None:
        strategic_target = StrategicYearTarget(
            min_development=10000.0,
            min_ore1_production=1300000.0,
            min_ore2_production=850000.0,
        )

    sim = TwoAreaStrategicSimulation(
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

    days_to_run = total_days if total_days is not None else 365.0
    sim.step(days_to_run * 86400.0)
    df = pd.DataFrame(sim.telemetry_history)

    df_prepared = prepare_history(df)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim.critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
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

"""Two-Area Simulation with Operational Analytical Face-Allocation Blending (Appendix A & B).

Implements closed-form analytical face mass allocation equations (Slide 29) to determine
optimal truck dispatch weights between High-Ore1 Face 1 and Balanced Face 2:
    r1 = (R1 - R_tot * f2) / (f1 - f2)
    r2 = R_tot - r1
    w1 = r1 / R_tot,  w2 = r2 / R_tot

These exact weights drive the DES truck dispatch policy dynamically as active facies parcels
change and operating modes switch on surface.
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
    TwoAreaSimulationBase,
    AreaReadinessTarget,
    StrategicYearTarget,
)
from drs_mining.components.allocation import solve_face_allocation_rates
from drs_mining.components.plot import (
    plot_two_area_dashboard,
    print_strategic_economic_summary,
    prepare_history,
    print_transition_log,
)


class TwoAreaAnalyticalBlendingSimulation(TwoAreaSimulationBase):
    """Two-Area Discrete-Event Simulation with Closed-Form Analytical Blending Dispatch."""

    def compute_analytical_weights(self) -> Tuple[float, float]:
        """Solves closed-form optimal face weights for current plant target draw rates."""
        if self.is_area2_locked(self.gt.value / 86400.0):
            return 1.0, 0.0

        r_o1 = getattr(self.plant, "active_ore1_draw_rate", 3600.0)
        r_o2 = getattr(self.plant, "active_ore2_draw_rate", 2400.0)
        f1_ore1 = 1.0 - self.face1.current_ore_grade
        f2_ore1 = 1.0 - self.face2.current_ore_grade

        res = solve_face_allocation_rates(
            target_ore1_rate=r_o1,
            target_ore2_rate=r_o2,
            face1_ore1_fraction=f1_ore1,
            face2_ore1_fraction=f2_ore1,
        )
        return res.face1_weight, res.face2_weight

    def select_face_for_truck(self, tr: Optional[Any] = None) -> int:
        """Dispatches truck according to optimal analytical face allocation weights."""
        if self.is_area2_locked(self.gt.value / 86400.0):
            return 1

        w1, _ = self.compute_analytical_weights()
        return 1 if self.rng.random() < w1 else 2

    def _record_telemetry(self, t: float) -> None:
        super()._record_telemetry(t)
        w1, w2 = self.compute_analytical_weights()
        self.telemetry_history[-1]["analytical_face1_weight"] = w1
        self.telemetry_history[-1]["analytical_face2_weight"] = w2


def plot_two_area_analytical_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/two_area_analytical_dashboard.png",
    **kwargs,
):
    """Builds and saves the comprehensive analytical blending diagnostics dashboard."""
    return plot_two_area_dashboard(
        df,
        output_path=output_path,
        title="Two-Area Operational Analytical Blending & Haulage Optimization",
        **kwargs,
    )


def run_two_area_analytical_simulation(
    total_ore_to_extract: float = 6600000.0,
    warmup_ore: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    availability: float = 0.85,
    target_ore_stock_level: float = 60000.0,
    strategic_target: Optional[StrategicYearTarget] = None,
    area2_target: Optional[AreaReadinessTarget] = None,
    area2_required_dev: float = 4000.0,
    area2_ready_by_day: float = 365.0,
    annual_discount_rate: float = 0.05,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[TwoAreaAnalyticalBlendingSimulation, pd.DataFrame]:
    """Runs the two-area analytical blending simulation."""
    if strategic_target is None:
        strategic_target = StrategicYearTarget(
            min_development=10000.0,
            min_ore1_production=1300000.0,
            min_ore2_production=850000.0,
        )
    if area2_target is None:
        area2_target = AreaReadinessTarget(
            required_development=area2_required_dev,
            ready_by_day=area2_ready_by_day,
        )

    sim = TwoAreaAnalyticalBlendingSimulation(
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warmup_ore,
        strategic_targets=(strategic_target,),
        area2_readiness_target=area2_target,
        area2_physical_unlock_enabled=True,
        annual_discount_rate=annual_discount_rate,
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
        label="Analytical Blending",
    )

    if plot and len(df_prepared) > 0:
        plot_two_area_analytical_dashboard(df_prepared)

    return sim, df_prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-Area Analytical Blending Simulation"
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
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
    )
    args = parser.parse_args()

    run_two_area_analytical_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        warmup_ore=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        area2_required_dev=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        seed=args.seed,
        plot=not args.no_plot,
    )

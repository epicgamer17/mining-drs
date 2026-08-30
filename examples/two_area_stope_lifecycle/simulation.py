"""Two-Area Multi-Stope Lifecycle Simulation: Turnaround Development, Waste Rock & Hierarchical Dispatch.

Implements the multi-stope operational underground environment:
  - Area 1 (Level 3): 3 active stopes (1A, 1B, 1C) with finite reserves (1.8M tonnes total).
  - Area 2 (Level 6): 3 deep stopes (2A, 2B, 2C) unlocked via 4,000 m capital decline development.
  - Stope Lifecycle:
      1. ORE_READY: Blasted ore mucked out by LHDs into 26.1t AD30 haul trucks.
      2. DEVELOPMENT_TURNAROUND: Ore round depleted. Requires waste rock extraction / development advance (30m).
      3. EXHAUSTED: Total stope reserve depleted; stope is permanently closed.
  - Two-Tier Hierarchical Closed-Loop Dispatch:
      - Tier 1: Maintain 6,000 t/d total plant feed.
      - Tier 2: Match analytical dispatch weights (w1, w2) for high-grade Mode A.
      - Tier 3: If preferred stope is in turnaround, dynamically redirect truck to next available stope.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple, Any

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import drs
import pandas as pd

from drs_mining.config import FLEET_MODES
from drs_mining.components import (
    TwoAreaSimulationBase,
    AreaReadinessTarget,
    StrategicYearTarget,
    StopeFace,
    StopeState,
    TwoTierHierarchicalDispatchController,
    StochasticFaciesGenerator,
    TruckPhase,
    ORE_PAYLOAD,
)
from drs_mining.components.plot import (
    plot_two_area_dashboard,
    prepare_history,
    print_transition_log,
)


class TwoAreaStopeLifecycleEngine(TwoAreaSimulationBase):
    """Underground DES simulation module with multi-stope lifecycles, turnaround dev & two-tier dispatch."""

    def __init__(
        self,
        policy_name: str = "POLICY_2_VALUE_ORIENTED",
        use_two_tier_dispatch: bool = True,
        num_lhds_per_stope: int = 1,
        **kwargs,
    ):
        self.policy_name = policy_name
        self.use_two_tier_dispatch = use_two_tier_dispatch
        self.num_lhds_per_stope = num_lhds_per_stope

        if policy_name == "POLICY_1_MYOPIC":
            kwargs["development_priority_truck_reservation_fraction"] = 0.0
            kwargs["area2_redeploy_locked_face_trucks_to_development"] = False

        super().__init__(**kwargs)

        # Multi-Stope Topology: 3 Stopes in Area 1 (Level 3), 3 Stopes in Area 2 (Level 6)
        self.stopes: List[StopeFace] = []
        seed = self.seed

        # Area 1 Stopes (Level 3)
        a1_means = [0.28, 0.30, 0.32]
        for i in range(1, 4):
            mean_f = a1_means[i - 1]
            gen = StochasticFaciesGenerator(
                mean_fraction=mean_f,
                std_dev=0.03,
                prob_new_facies=0.3,
                variation_same_facies=0.01,
            )
            stope = StopeFace(
                name=f"stope_1{chr(64+i)}",
                face_id=i,
                area_id=1,
                level_index=3,
                generator=gen,
                mean_ore_fraction=mean_f,
                std_dev_ore_fraction=0.03,
                total_stope_reserve=600000.0,
                min_parcel_ore_mass=25000.0,
                max_parcel_ore_mass=40000.0,
                waste_to_ore_ratio=0.15,
                turnaround_dev_per_parcel_m=5.0,
                seed=seed + i,
            )
            self.stopes.append(stope)

        # Area 2 Stopes (Level 6)
        a2_means = [0.33, 0.35, 0.37]
        for i in range(4, 7):
            mean_f = a2_means[i - 4]
            gen = StochasticFaciesGenerator(
                mean_fraction=mean_f,
                std_dev=0.03,
                prob_new_facies=0.3,
                variation_same_facies=0.01,
            )
            stope = StopeFace(
                name=f"stope_2{chr(64+i-3)}",
                face_id=i,
                area_id=2,
                level_index=6,
                generator=gen,
                mean_ore_fraction=mean_f,
                std_dev_ore_fraction=0.03,
                total_stope_reserve=1600000.0,
                min_parcel_ore_mass=25000.0,
                max_parcel_ore_mass=40000.0,
                waste_to_ore_ratio=0.20,
                turnaround_dev_per_parcel_m=5.0,
                seed=seed + i,
            )
            self.stopes.append(stope)

        # Two-Tier Hierarchical Dispatcher
        self.dispatcher = TwoTierHierarchicalDispatchController(
            stopes=self.stopes,
            target_daily_ore_tonnes=6000.0,
            target_stockpile_buffer_tonnes=self.target_ore_stock_level,
            seed=seed,
        )

        self.stope_turnaround_development = drs.Level("stope_turnaround_development", 0.0)
        self.fallback_dispatch_count = drs.Level("fallback_dispatch_count", 0.0)
        self.history_records: List[Dict[str, Any]] = []

    def select_face_for_truck(self, tr: Optional[Any] = None) -> int:
        """Selects target stope using two-tier hierarchical dispatch controller."""
        if self.is_area2_locked(self.gt.value / 86400.0):
            # Only Area 1 stopes accessible
            a1_stopes = [s for s in self.stopes if s.area_id == 1 and s.state == StopeState.ORE_READY]
            if a1_stopes:
                return a1_stopes[self.rng.randint(0, len(a1_stopes) - 1)].face_id
            return 1

        active_mode = self.plant.active_operating_mode.value.name
        res = self.dispatcher.dispatch(
            active_operating_mode_name=active_mode,
            truck_payload=ORE_PAYLOAD,
            truck_cycle_time_sec=2100.0,
            allow_area2=not self.is_area2_locked(self.gt.value / 86400.0),
        )
        if res.is_fallback:
            self.fallback_dispatch_count.value += 1.0

        return res.selected_stope_id or 1

    def _calendar_update(self, t: float) -> None:
        day = int(t // 86400.0)
        if day != self._cur_day:
            self._cur_day = day
            self._holiday_today = (day % 365) in self.holidays

            # Stope turnaround advance step
            turnaround_dev = 3.5
            self.stope_turnaround_development.value += turnaround_dev

            # Area 2 capital development step
            if self.policy_name == "POLICY_1_MYOPIC":
                cap_dev_step = 1.0 if self.is_area2_locked(day) else 0.0
            else:
                cap_dev_step = self._compute_daily_development_meters()

            a2_locked = self.is_area2_locked(day)
            area2_dev = cap_dev_step if a2_locked else 0.0

            self.face2.advance_development(area2_dev, current_day=float(day))

            total_mine_dev = float(self.face2.cumulative_development.value + self.stope_turnaround_development.value)

            self.plant.step_daily_economics(
                current_day=float(day),
                ore1_mined_t=self.ore1_dumped_total.value,
                ore2_mined_t=self.ore2_dumped_total.value,
                development_units=total_mine_dev,
            )

            force_mode = FLEET_MODES["PRODUCTION"] if self.policy_name == "POLICY_1_MYOPIC" else None
            self.tactical_controller.step_daily_tactical_review(
                current_day=float(day),
                cum_development=total_mine_dev,
                cum_ore1=float(self.ore1_dumped_total.value),
                cum_ore2=float(self.ore2_dumped_total.value),
                area2_readiness_tracker=self.face2,
                total_trucks=self.num_trucks,
                force_mode=force_mode,
            )

        shift = int(t // 43200.0)
        if shift != self._shift_marker:
            self._shift_marker = shift
            for tr in self.trucks:
                tr.seat_used = 0.0
                self._schedule_down_window(tr)
            for op in self.operators:
                op.used_seat = 0.0

    def _record_telemetry(self, t: float) -> None:
        super()._record_telemetry(t)
        cap_dev = float(self.readiness_tracker.cumulative_development.value)
        turn_dev = float(self.stope_turnaround_development.value)
        tot_dev = cap_dev + turn_dev

        record = self.telemetry_history[-1]
        record["area2_cumulative_development"] = cap_dev
        record["stope_turnaround_dev_m"] = turn_dev
        record["cumulative_mine_development"] = tot_dev
        record["cumulative_development"] = tot_dev
        record["fallback_dispatch_count"] = float(self.fallback_dispatch_count.value)
        self.history_records = self.telemetry_history


def plot_stope_lifecycle_dashboard(
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    output_path: str = "plots/two_area_stope_lifecycle_dashboard.png",
    **kwargs,
):
    """Builds and saves the comprehensive multi-stope lifecycle diagnostics dashboard."""
    return plot_two_area_dashboard(
        df_p2,
        output_path=output_path,
        title="Multi-Stope Underground Lifecycle & Turnaround Development Benchmark",
        **kwargs,
    )


def run_stope_lifecycle_study(
    total_ore_to_extract: float = 6600000.0,
    warmup_ore: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    availability: float = 0.85,
    stockpile_target: float = 60000.0,
    area2_required_dev: float = 4000.0,
    area2_ready_by_day: float = 365.0,
    discount_rate: float = 0.05,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes Policy 1 vs Policy 2 comparative benchmark across multi-stope underground lifecycle."""
    strat_target = StrategicYearTarget(
        min_development=10000.0,
        min_ore1_production=1300000.0,
        min_ore2_production=850000.0,
    )
    area2_target = AreaReadinessTarget(
        required_development=area2_required_dev,
        ready_by_day=area2_ready_by_day,
    )

    days_to_run = total_days if total_days is not None else 365.0

    # Policy 2
    sim_p2 = TwoAreaStopeLifecycleEngine(
        policy_name="POLICY_2_VALUE_ORIENTED",
        use_two_tier_dispatch=True,
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=stockpile_target,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warmup_ore,
        strategic_targets=(strat_target,),
        area2_readiness_target=area2_target,
        annual_discount_rate=discount_rate,
        seed=seed,
    )
    sim_p2.step(days_to_run * 86400.0)
    df_p2 = pd.DataFrame(sim_p2.telemetry_history)

    # Policy 1
    sim_p1 = TwoAreaStopeLifecycleEngine(
        policy_name="POLICY_1_MYOPIC",
        use_two_tier_dispatch=False,
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=stockpile_target,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warmup_ore,
        strategic_targets=(strat_target,),
        area2_readiness_target=area2_target,
        annual_discount_rate=discount_rate,
        seed=seed,
    )
    sim_p1.step(days_to_run * 86400.0)
    df_p1 = pd.DataFrame(sim_p1.telemetry_history)

    if plot and len(df_p2) > 0 and len(df_p1) > 0:
        plot_stope_lifecycle_dashboard(df_p1, df_p2)

    return df_p1, df_p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Stope Underground Lifecycle Simulation Benchmark")
    parser.add_argument("--total_ore_to_extract", type=float, default=6600000.0)
    parser.add_argument("--warmup_ore", type=float, default=600000.0)
    parser.add_argument("--total_days", type=float, default=None)
    parser.add_argument("--trucks", type=int, default=18)
    parser.add_argument("--operators", type=int, default=18)
    parser.add_argument("--availability", type=float, default=0.85)
    parser.add_argument("--stockpile_target", type=float, default=60000.0)
    parser.add_argument("--area2_required_dev", type=float, default=4000.0)
    parser.add_argument("--area2_ready_by_day", type=float, default=365.0)
    parser.add_argument("--discount_rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()

    run_stope_lifecycle_study(
        total_ore_to_extract=args.total_ore_to_extract,
        warmup_ore=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        availability=args.availability,
        stockpile_target=args.stockpile_target,
        area2_required_dev=args.area2_required_dev,
        area2_ready_by_day=args.area2_ready_by_day,
        discount_rate=args.discount_rate,
        seed=args.seed,
        plot=not args.no_plot,
    )

"""Two-Area Multi-Stope Lifecycle Simulation: Turnaround Development, Waste Rock & Hierarchical Dispatch.

Implements the multi-stope operational underground environment:
  - Area 1 (Level 3): 3 active stopes (1A, 1B, 1C) with finite reserves (1.8M tonnes total).
  - Area 2 (Level 6): 3 deep stopes (2A, 2B, 2C) unlocked via capital decline development.
  - Stope Lifecycle:
      1. ORE_READY: Blasted ore mucked out by LHDs into haul trucks.
      2. DEVELOPMENT_TURNAROUND: Ore round depleted. Requires waste rock extraction / development advance.
      3. EXHAUSTED: Total stope reserve depleted; stope is permanently closed.
  - Two-Tier Hierarchical Closed-Loop Dispatch:
      - Tier 1: Maintain total plant feed target.
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

from drs_mining.config import (
    SimulationConfig,
    DEFAULT_CONFIG,
    FLEET_MODES,
)
from drs_mining.components import (
    MiningSimulationBase,
    AreaReadinessTarget,
    StrategicYearTarget,
    MineFace,
    StopeState,
    TwoTierHierarchicalDispatchController,
    StochasticFaciesGenerator,
    TruckPhase,
)
from drs_mining.components.plot import (
    plot_two_area_dashboard,
    prepare_history,
    print_transition_log,
)


class TwoAreaStopeLifecycleEngine(MiningSimulationBase):
    """Underground DES simulation module with multi-stope lifecycles, turnaround dev & two-tier dispatch."""

    def __init__(
        self,
        policy_name: str = "POLICY_2_VALUE_ORIENTED",
        use_two_tier_dispatch: bool = True,
        num_lhds_per_stope: int = 1,
        config: Optional[SimulationConfig] = None,
        **kwargs,
    ):
        self.policy_name = policy_name
        self.use_two_tier_dispatch = use_two_tier_dispatch
        self.num_lhds_per_stope = num_lhds_per_stope

        if policy_name == "POLICY_1_MYOPIC":
            kwargs["development_priority_truck_reservation_fraction"] = 0.0
            kwargs["area2_redeploy_locked_face_trucks_to_development"] = False

        super().__init__(config=config, **kwargs)
        cfg = self.config

        # Multi-Stope Topology: 3 Stopes in Area 1 (Level 3), 3 Stopes in Area 2 (Level 6)
        self.stopes: List[MineFace] = []
        seed = self.seed

        # Area 1 Stopes (Level 3)
        a1_means = cfg.geology.stope_a1_mean_fractions
        for i in range(1, 4):
            mean_f = a1_means[i - 1]
            gen = StochasticFaciesGenerator(
                mean_fraction=mean_f,
                std_dev=cfg.geology.stope_std_dev,
                prob_new_facies=cfg.geology.prob_new_facies,
                variation_same_facies=cfg.geology.variation_same_facies,
            )
            stope = MineFace(
                name=f"stope_1{chr(64+i)}",
                face_id=i,
                area_id=1,
                level_index=cfg.topology.area1_level,
                generator=gen,
                mean_ore_fraction=mean_f,
                std_dev_ore_fraction=cfg.geology.stope_std_dev,
                total_stope_reserve=cfg.geology.area1_stope_reserve,
                min_parcel_ore_mass=cfg.geology.stope_min_parcel_mass,
                max_parcel_ore_mass=cfg.geology.stope_max_parcel_mass,
                waste_to_ore_ratio=cfg.geology.stope_a1_waste_to_ore_ratio,
                turnaround_dev_per_parcel_m=cfg.geology.stope_turnaround_dev_per_parcel_m,
                heading_cross_section_m2=cfg.topology.stope_cross_section_m2,
                rock_density_t_per_m3=cfg.topology.rock_density_t_per_m3,
                seed=seed + i,
            )
            self.stopes.append(stope)

        # Area 2 Stopes (Level 6)
        a2_means = cfg.geology.stope_a2_mean_fractions
        for i in range(4, 7):
            mean_f = a2_means[i - 4]
            gen = StochasticFaciesGenerator(
                mean_fraction=mean_f,
                std_dev=cfg.geology.stope_std_dev,
                prob_new_facies=cfg.geology.prob_new_facies,
                variation_same_facies=cfg.geology.variation_same_facies,
            )
            stope = MineFace(
                name=f"stope_2{chr(64+i-3)}",
                face_id=i,
                area_id=2,
                level_index=cfg.topology.area2_level,
                generator=gen,
                mean_ore_fraction=mean_f,
                std_dev_ore_fraction=cfg.geology.stope_std_dev,
                total_stope_reserve=cfg.geology.area2_stope_reserve,
                min_parcel_ore_mass=cfg.geology.stope_min_parcel_mass,
                max_parcel_ore_mass=cfg.geology.stope_max_parcel_mass,
                waste_to_ore_ratio=cfg.geology.stope_a2_waste_to_ore_ratio,
                turnaround_dev_per_parcel_m=cfg.geology.stope_turnaround_dev_per_parcel_m,
                heading_cross_section_m2=cfg.topology.stope_cross_section_m2,
                rock_density_t_per_m3=cfg.topology.rock_density_t_per_m3,
                seed=seed + i,
            )
            self.stopes.append(stope)

        # Two-Tier Hierarchical Dispatcher
        target_daily = cfg.plant.mode_a_ore1_milling_rate + cfg.plant.mode_a_ore2_milling_rate
        self.dispatcher = TwoTierHierarchicalDispatchController(
            stopes=self.stopes,
            target_daily_ore_tonnes=target_daily,
            target_stockpile_buffer_tonnes=self.target_ore_stock_level,
            seed=seed,
        )

        self.faces = self.stopes
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
            truck_payload=self.config.fleet.truck_payload,
            truck_cycle_time_sec=self.config.fleet.truck_cycle_time_sec,
            allow_area2=not self.is_area2_locked(self.gt.value / 86400.0),
        )
        if res.is_fallback:
            self.fallback_dispatch_count.value += 1.0

        return res.selected_stope_id or 1

    def _record_telemetry(self, t: float) -> None:
        super()._record_telemetry(t)
        cap_dev = float(self.area2_cumulative_development.value)
        turn_dev = float(self.sustaining_cumulative_development.value)
        self.stope_turnaround_development.value = turn_dev
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
    config: Optional[SimulationConfig] = None,
    total_ore_to_extract: Optional[float] = None,
    warmup_ore: Optional[float] = None,
    total_days: Optional[float] = None,
    num_trucks: Optional[int] = None,
    num_operators: Optional[int] = None,
    availability: Optional[float] = None,
    stockpile_target: Optional[float] = None,
    area2_required_dev: Optional[float] = None,
    area2_ready_by_day: Optional[float] = None,
    discount_rate: Optional[float] = None,
    seed: Optional[int] = None,
    plot: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Executes Policy 1 vs Policy 2 comparative benchmark across multi-stope underground lifecycle."""
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

    days_to_run = total_days if total_days is not None else cfg.total_days
    warm = warmup_ore if warmup_ore is not None else cfg.plant.ore_to_be_extracted_during_warming_period

    # Policy 2
    sim_p2 = TwoAreaStopeLifecycleEngine(
        config=cfg,
        policy_name="POLICY_2_VALUE_ORIENTED",
        use_two_tier_dispatch=True,
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=stockpile_target,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warm,
        strategic_targets=(strat_target,),
        area2_readiness_target=area2_target,
        annual_discount_rate=discount_rate,
        seed=seed,
    )
    sim_p2.step(days_to_run * 86400.0)
    df_p2 = pd.DataFrame(sim_p2.telemetry_history)

    # Policy 1
    sim_p1 = TwoAreaStopeLifecycleEngine(
        config=cfg,
        policy_name="POLICY_1_MYOPIC",
        use_two_tier_dispatch=False,
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=availability,
        target_ore_stock_level=stockpile_target,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=warm,
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
    parser.add_argument(
        "--total_ore_to_extract",
        type=float,
        default=DEFAULT_CONFIG.plant.total_ore_to_extract,
        help=f"Total ore extraction target (default: {DEFAULT_CONFIG.plant.total_ore_to_extract:,.1f} t)",
    )
    parser.add_argument(
        "--warmup_ore",
        type=float,
        default=DEFAULT_CONFIG.plant.ore_to_be_extracted_during_warming_period,
        help=f"Warmup ore extraction (default: {DEFAULT_CONFIG.plant.ore_to_be_extracted_during_warming_period:,.1f} t)",
    )
    parser.add_argument("--total_days", type=float, default=None, help="Total simulation days (optional)")
    parser.add_argument("--trucks", type=int, default=DEFAULT_CONFIG.fleet.num_trucks)
    parser.add_argument("--operators", type=int, default=DEFAULT_CONFIG.fleet.num_operators)
    parser.add_argument("--availability", type=float, default=DEFAULT_CONFIG.fleet.availability)
    parser.add_argument("--stockpile_target", type=float, default=DEFAULT_CONFIG.plant.target_ore_stock_level)
    parser.add_argument("--area2_required_dev", type=float, default=DEFAULT_CONFIG.planning.area2_required_development)
    parser.add_argument("--area2_ready_by_day", type=float, default=DEFAULT_CONFIG.planning.area2_ready_by_day)
    parser.add_argument("--discount_rate", type=float, default=DEFAULT_CONFIG.economics.annual_discount_rate)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
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

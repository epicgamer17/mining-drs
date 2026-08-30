"""Shelswell (2017) DES Haulage + Single-Face DRS Blending Modes Simulation.

Subclasses MiningSimulationBase to orchestrate single-face discrete-event haulage,
stochastic parcel geology, dual stockpiles, and supervisory blending campaigns.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

# Ensure repository root is in sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import drs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drs.plot import (
    Dashboard,
    plot_time_series,
    plot_safety_margin,
    plot_dual_axis_step,
)
from drs_mining.config import (
    MILL_MODES,
    CalendarConfig,
    TopologyConfig,
    HaulageFleetConfig,
    PlantConfig,
    GeologyConfig,
    SimulationConfig,
)
from drs_mining.components import (
    MiningSimulationBase,
    OperatingMode,
    MetallurgicalPlant,
    PlantDrawRates,
    Stockpile,
    OperatingModeController,
    StochasticFaciesGenerator,
    MineFace,
    TruckPhase,
    Operator,
    Truck,
    SurfaceDumpStation,
    OPERATING_PHASES,
    SEAT_PHASES,
    DUE_PHASES,
)
from drs_mining.components.plot import (
    MODE_PALETTE,
    prepare_history,
    plot_ore_with_modes,
    plot_mode_distribution,
    plot_mode_dwell_times,
    plot_attributed_deficit,
    plot_deficit_disparity,
    plot_deficit_breakdown_bar,
    plot_truck_idle_and_utilization,
    print_transition_log,
    print_deficit_by_mode,
)

_CFG = SimulationConfig()
DT_MAX = _CFG.dt_max


class ShelswellSingleFaceBlending(MiningSimulationBase):
    """Hybrid simulation combining Shelswell DES Truck-Loader Haulage with
    Single-Face Stochastic Parcel Geology and DRS Blending Modes.
    """

    def __init__(
        self,
        num_trucks: int = 18,
        num_operators: int = 18,
        num_lhds: int = 2,
        availability: float = 0.85,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        seed: int = 42,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        mode_a_ore1_milling_rate: float = 3600.0,
        mode_a_ore2_milling_rate: float = 2400.0,
        mode_a_contingency_ore1_milling_rate: float = 3900.0,
        mode_b_ore1_milling_rate: float = 4600.0,
        mode_b_ore2_milling_rate: float = 800.0,
        mode_b_contingency_ore2_milling_rate: float = 2500.0,
    ):
        facies_gen = StochasticFaciesGenerator(
            mean_fraction=mean_ore_fraction,
            std_dev=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
        )
        face = MineFace(
            name="mine_face",
            face_id=1,
            generator=facies_gen,
            min_ore_mass=min_ore_mass,
            max_ore_mass=max_ore_mass,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            mean_ore_fraction=mean_ore_fraction,
            std_dev_ore_fraction=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
            initial_parcel_mass=40000.0,
        )
        super().__init__(
            faces=[face],
            num_trucks=num_trucks,
            num_operators=num_operators,
            num_lhds_per_face=num_lhds,
            availability=availability,
            target_ore_stock_level=target_ore_stock_level,
            critical_ore2_level=critical_ore2_level,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            mode_a_ore1_milling_rate=mode_a_ore1_milling_rate,
            mode_a_ore2_milling_rate=mode_a_ore2_milling_rate,
            mode_a_contingency_ore1_milling_rate=mode_a_contingency_ore1_milling_rate,
            mode_b_ore1_milling_rate=mode_b_ore1_milling_rate,
            mode_b_ore2_milling_rate=mode_b_ore2_milling_rate,
            mode_b_contingency_ore2_milling_rate=mode_b_contingency_ore2_milling_rate,
            seed=seed,
        )
        self.mine_face = self.face1
        self.history_records = self.telemetry_history
        self.ore1_hauled = self.ore1_dumped_total
        self.ore2_hauled = self.ore2_dumped_total
        self.trips = 0
        self._cycle_sum = 0.0
        self.traffic_delay_sum = 0.0
        self.horizon_sec = float("inf")


def print_statistics(plant: MetallurgicalPlant, mine: MineFace):
    """Print operating-mode time-shares and throughput matching blending_modes format."""
    print("\n--- Output Statistics ---")
    total_time = plant.total_duration

    if total_time > 0:
        for attr, label in [
            ("cumulative_time_mode_a", "PortionOfTimeInModeA"),
            ("cumulative_time_mode_a_contingency", "PortionOfTimeInModeAContingency"),
            ("cumulative_time_mode_a_surging", "PortionOfTimeInModeAMineSurging"),
            ("cumulative_time_mode_b", "PortionOfTimeInModeB"),
            ("cumulative_time_mode_b_contingency", "PortionOfTimeInModeBContingency"),
            ("cumulative_time_mode_b_surging", "PortionOfTimeInModeBMineSurging"),
            ("cumulative_time_shutdown", "PortionOfTimeInShutdown"),
        ]:
            print(
                f"{label}: {getattr(plant, attr).value / total_time:.4f}"
            )
    else:
        print("Total time is 0. Cannot calculate mode portions.")

    active_time = plant.active_duration(total_time)
    if active_time > 0:
        if hasattr(plant, "cumulative_milled_mass"):
            total_ore_processed = plant.cumulative_milled_mass.value
        else:
            total_ore_processed = mine.cumulative_extracted_mass.value
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


def print_simulation_statistics(sim: ShelswellSingleFaceBlending, df: pd.DataFrame):
    """Prints operational summary statistics."""
    total_days = sim.gt.value / 86400.0
    total_ore_hauled = sim.mine_face.cumulative_extracted_mass.value
    total_milled = sim.plant.cumulative_milled_mass.value
    active_days = sim.plant.active_duration(sim.plant.total_duration)

    print("\n" + "=" * 70)
    print(" SHELSWELL SINGLE-FACE DES + DRS BLENDING MODES SIMULATION RESULTS")
    print("=" * 70)
    print(f"Simulation Horizon:        {total_days:.1f} days")
    print(f"Total Ore Hauled:          {total_ore_hauled:,.1f} t ({total_ore_hauled / max(1e-3, total_days):.1f} t/d)")
    print(f"  ↳ Ore 1 Equivalent:      {sim.ore1_hauled.value:,.1f} t ({sim.ore1_hauled.value / max(1e-3, total_days):.1f} t/d)")
    print(f"  ↳ Ore 2 Equivalent:      {sim.ore2_hauled.value:,.1f} t ({sim.ore2_hauled.value / max(1e-3, total_days):.1f} t/d)")
    print(f"Total Ore Milled:          {total_milled:,.1f} t ({total_milled / max(1e-3, active_days):.1f} t/active-day)")
    print(f"Final Ore 1 Stockpile:     {sim.ore1_stock.level:,.1f} t")
    print(f"Final Ore 2 Stockpile:     {sim.ore2_stock.level:,.1f} t")
    print(f"Final Total Stockpile:     {sim.ore1_stock.level + sim.ore2_stock.level:,.1f} t")

    print("\n--- Operating Mode Time-Shares ---")
    tot_dur = sim.plant.total_duration
    if tot_dur > 0:
        for attr, label in [
            ("cumulative_time_mode_a", "Mode A (Normal)"),
            ("cumulative_time_mode_a_contingency", "Mode A (Contingency)"),
            ("cumulative_time_mode_a_surging", "Mode A (Surging)"),
            ("cumulative_time_mode_b", "Mode B (Normal)"),
            ("cumulative_time_mode_b_contingency", "Mode B (Contingency)"),
            ("cumulative_time_mode_b_surging", "Mode B (Surging)"),
            ("cumulative_time_shutdown", "Shutdown"),
        ]:
            val = getattr(sim.plant, attr).value
            print(f"  {label:<25}: {val:.2f} days ({100.0 * val / tot_dur:5.1f} %)")
    print("=" * 70 + "\n")


def plot_single_face_shelswell_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/shelswell_single_face_dashboard.png",
    palette: dict = None,
    figsize: Tuple[int, int] = (16, 48),
):
    """Builds and saves the diagnostics dashboard."""
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if "active_operating_mode_name" not in df.columns or "Mode A" not in df.columns:
        df = prepare_history(df)

    dash = Dashboard(
        nrows=12,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title="Shelswell Single-Face DES + DRS Blending Modes Diagnostics",
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 9, 11])

    # 0. Operating Modes Step Timeline
    plot_time_series(
        df,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes (Step Timeline)",
        is_step=True,
        ax=dash[0],
    )

    # 1. Stockpiles & Operating Modes
    plot_ore_with_modes(
        df,
        time_col="time",
        ore_cols=["total_system_ore_mass", "Ore1Stock_mass", "Ore2Stock_mass"],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Ore Stockpiles & Operating Campaigns",
        palette=palette,
        hlines=[
            {
                "y": 60000.0,
                "color": "black",
                "linestyle": "--",
                "linewidth": 1.5,
                "alpha": 0.7,
                "label": "Target Total (60k)",
            },
            {
                "y": 20400.0,
                "color": "red",
                "linestyle": ":",
                "linewidth": 1.5,
                "alpha": 0.8,
                "label": "Critical Ore 2 (20.4k)",
            },
        ],
        ax=dash[1],
    )

    # 2. Safety Margin
    plot_safety_margin(
        df,
        stock_column="Ore2Stock_mass",
        critical_threshold=20400.0,
        title="Ore 2 Critical Safety Margin Buffer",
        ax=dash[2],
    )

    # 3. Ore Extraction & Processing
    plot_time_series(
        df,
        y_columns=["cumulative_milled_mass", "cumulative_extracted_mass"],
        title="Cumulative Ore Milled vs Face Extracted",
        ax=dash[3],
    )

    # 4. Mode Distribution Pie Chart
    plot_mode_distribution(
        df,
        mode_col="active_operating_mode_name",
        palette=palette,
        title="Operating Mode Time Distribution",
        ax=dash[4],
    )

    # 5. Attributed Deficit
    plot_attributed_deficit(
        df,
        ideal_rate=6000.0,
        title="Attributed Production Deficit by Cause",
        palette=palette,
        ax=dash[5],
    )

    # 6. Mode Dwell Times
    plot_mode_dwell_times(
        df,
        mode_col="active_operating_mode_name",
        palette=palette,
        title="Campaign Dwell Time Boxplots",
        ax=dash[6],
    )

    # 7. Deficit Disparity
    plot_deficit_disparity(
        df,
        ideal_rate=6000.0,
        title="Production Deficit Disparity Breakdown",
        palette=palette,
        ax=dash[7],
    )

    # 8. Deficit Breakdown Bar
    plot_deficit_breakdown_bar(
        df,
        ideal_rate=6000.0,
        title="Total Attributed Deficit by Operating Regime",
        palette=palette,
        ax=dash[8],
    )

    # 9. Truck Activity
    plot_truck_idle_and_utilization(
        df,
        idle_col="trucks_idle",
        operating_col="trucks_operating",
        total_trucks=18,
        title="Truck Fleet Utilization & Idle Dynamics",
        ax=dash[9],
    )

    dash.save(output_path)
    print(f"Saved dashboard visualization to '{output_path}'.")
    return dash


def run_shelswell_single_face_simulation(
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    total_days: Optional[float] = None,
    num_trucks: int = 18,
    num_operators: int = 18,
    num_lhds: int = 2,
    availability: float = 0.85,
    target_ore_stock_level: float = 60000.0,
    seed: int = 42,
    plot: bool = True,
) -> Tuple[ShelswellSingleFaceBlending, pd.DataFrame]:
    """Builds and runs the single face hybrid simulation."""
    sim = ShelswellSingleFaceBlending(
        num_trucks=num_trucks,
        num_operators=num_operators,
        num_lhds=num_lhds,
        availability=availability,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        seed=seed,
    )

    days_to_run = total_days if total_days is not None else 365.0
    sim.step(days_to_run * 86400.0)

    df = pd.DataFrame(sim.history_records)
    print_simulation_statistics(sim, df)
    print_statistics(sim.plant, sim.mine_face)

    df_prepared = prepare_history(df)
    print_transition_log(
        df_prepared,
        critical_ore2_level=sim.critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        label="Shelswell Single-Face Blending",
    )
    print_deficit_by_mode(
        df_prepared,
        extraction_cols=["ore1_mined", "ore2_mined"],
        ideal_rate=6000.0,
    )

    if plot and len(df_prepared) > 0:
        plot_single_face_shelswell_dashboard(df_prepared)
    return sim, df_prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Shelswell Single-Face DES + DRS Blending Modes Simulation"
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
        help="Total simulation duration in days",
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
        "--lhds",
        type=int,
        default=2,
        help="Number of LHD loaders at the single face (default: 2)",
    )
    parser.add_argument(
        "--availability",
        type=float,
        default=0.85,
        help="Overall mechanical availability fraction (default: 0.85)",
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

    run_shelswell_single_face_simulation(
        total_ore_to_extract=args.total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=args.warmup_ore,
        total_days=args.total_days,
        num_trucks=args.trucks,
        num_operators=args.operators,
        num_lhds=args.lhds,
        availability=args.availability,
        target_ore_stock_level=args.stockpile_target,
        seed=args.seed,
        plot=not args.no_plot,
    )

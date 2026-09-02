"""Tactical Blending Simulation: Navarra Concentrator & Satellite Development.

Implements the tactical operational framework from NAVARRA_MEETING.md:
- Main Ore Body: High reserve, primary source, close to plant.
- Satellite Ore Body: Requires development to reach, sporadic daily availability,
  higher Ore 2 concentration for stockpile balancing, longer haul cycle times.
- Concentrator (Mill Processor): Mode A (primary target), Mode B, and Contingency.
- Tactical Policy: Keep concentrator in Mode A while using the satellite as little as possible.
- Direct extraction from MaterialSource to Stockpiles to Mill Processor.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drs
from drs import DRSEngine, Telemetry, Processor, Storage, Flow, blend_flows

from drs_mining.components import (
    OperatingMode,
    OperatingModeController,
    MaterialSource,
    autocorrelated_generator,
    truck_haul_capacity,
)
from drs_mining.config import MILL_MODES


def build_navarra_network(
    seed: int = 42,
    main_reserve_tonnes: float = 6_000_000.0,
    sat_reserve_tonnes: float = 2_000_000.0,
    sat_required_dev_m: float = 300.0,
    target_ore_stock_level: float = 60_000.0,
    critical_ore2_level: float = 20_400.0,
    contingency_segment_duration: float = 1.0,
    campaign_duration: float = 34.0,
    shutdown_duration: float = 1.0,
):
    """Constructs the tactical simulation network for the Navarra operation."""
    rng = random.Random(seed)

    # 1. Main Ore Body (70% Ore 1, 30% Ore 2 on average, near haulage)
    stream_main = autocorrelated_generator(
        mean_fraction=0.30,
        std_dev=0.04,
        prob_new_facies=0.25,
        variation_step=0.01,
        min_mass=30_000.0,
        max_mass=50_000.0,
        initial_mass=40_000.0,
        seed=seed,
    )
    res_main = MaterialSource(
        name="main_reserve",
        total_tonnes=main_reserve_tonnes,
        stream=stream_main,
    )

    # 2. Satellite Ore Body (35% Ore 1, 65% Ore 2 on average, far haulage, sporadic)
    stream_sat = autocorrelated_generator(
        mean_fraction=0.65,
        std_dev=0.06,
        prob_new_facies=0.30,
        variation_step=0.02,
        min_mass=25_000.0,
        max_mass=45_000.0,
        initial_mass=35_000.0,
        seed=seed + 100,
    )
    res_sat = MaterialSource(
        name="satellite_reserve",
        total_tonnes=sat_reserve_tonnes,
        stream=stream_sat,
    )

    # 3. Stockpiles
    init_ore1 = target_ore_stock_level * 0.65
    init_ore2 = target_ore_stock_level * 0.35
    ore1_stock = Storage(
        name="Ore1Stock",
        initial_level=init_ore1,
        initial_attributes={"ore2_fraction": 0.0},
    )
    ore2_stock = Storage(
        name="Ore2Stock",
        initial_level=init_ore2,
        initial_attributes={"ore2_fraction": 1.0},
    )

    # 4. Concentrator Mill Processor
    mill = Processor(name="mill", max_rate=6000.0)

    # 5. Operating Mode Controller
    mode_ctrl = OperatingModeController(
        duration_of_production_campaigns=campaign_duration,
        duration_of_shutdowns=shutdown_duration,
        duration_of_contingency_segments=contingency_segment_duration,
        critical_ore2_level=critical_ore2_level,
        target_total_stock=target_ore_stock_level,
    )

    return (
        res_main,
        res_sat,
        mill,
        mode_ctrl,
        ore1_stock,
        ore2_stock,
        sat_required_dev_m,
    )


class NavarraTacticalSimulation:
    """Manages the DRS simulation run, tactical blending policy, and telemetry."""

    def __init__(
        self,
        seed: int = 42,
        satellite_availability_prob: float = 0.80,
        development_rate_m_per_day: float = 8.0,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.satellite_availability_prob = satellite_availability_prob
        self.development_rate_m_per_day = development_rate_m_per_day

        (
            self.res_main,
            self.res_sat,
            self.mill,
            self.mode_ctrl,
            self.ore1_stock,
            self.ore2_stock,
            self.sat_required_dev_m,
        ) = build_navarra_network(seed=seed)

        self.satellite_unlocked = False
        self.satellite_sporadic_available = False
        self.satellite_cumulative_dev = 0.0

        self.cumulative_milled_mass = drs.Level(
            "cumulative_concentrator_throughput", initial_value=0.0, owner=self.mill
        )
        self.mill.cumulative_concentrator_throughput = self.cumulative_milled_mass

        self.engine = DRSEngine()
        self._setup_policy()

    def _setup_policy(self) -> None:
        self.engine.register(
            self.res_main,
            self.res_sat,
            self.mill,
            self.mode_ctrl,
            self.ore1_stock,
            self.ore2_stock,
        )

        last_eval_day = [-1]

        @self.engine.on_step
        def _policy(t: float):
            day = int(t)

            # Daily tactical review & sporadic evaluation
            if day > last_eval_day[0]:
                last_eval_day[0] = day
                self.satellite_sporadic_available = (
                    self.rng.random() < self.satellite_availability_prob
                )

            # 1. Update campaign mode
            active_campaign = self.mode_ctrl.update_campaign(
                ore2_stock_level=self.ore2_stock.level
            )

            # 2. Advance development when in Mode B or shutdown
            if not self.satellite_unlocked:
                if "MODE_B" in active_campaign.name or active_campaign.name == "SHUTDOWN":
                    self.satellite_cumulative_dev += self.development_rate_m_per_day
                    if self.satellite_cumulative_dev >= self.sat_required_dev_m:
                        self.satellite_unlocked = True

            # 3. Resolve active operating mode (handling contingency / surging)
            active_mode = self.mode_ctrl.resolve_operating_mode(
                active_campaign,
                ore1_level=self.ore1_stock.level,
                ore2_level=self.ore2_stock.level,
            )

            # 4. Get draw rates
            draw_rates = self.mode_ctrl.get_draw_rates(active_mode)
            ore1_rate = draw_rates.get("Ore1Stock", 0.0)
            ore2_rate = draw_rates.get("Ore2Stock", 0.0)

            p1 = self.res_main.current_attributes.get("ore2_fraction", 0.30)
            mode_name = active_mode.name
            if "_MINE_SURGING" in mode_name:
                if mode_name == "MODE_A_MINE_SURGING":
                    effective_fraction = max(1.0 - p1, 0.01)
                    mine_target = ore1_rate / effective_fraction
                else:
                    effective_fraction = max(p1, 0.01)
                    mine_target = ore2_rate / effective_fraction
            else:
                mine_target = ore1_rate + ore2_rate

            # 5. Tactical Source Allocation (Navarra requirement: use satellite as little as possible)
            sat_weight = 0.0
            if (
                self.satellite_unlocked
                and self.satellite_sporadic_available
                and not self.res_sat.is_exhausted
            ):
                ore2_buffer = self.ore2_stock.level - self.mode_ctrl.critical_ore2_level
                if ore2_buffer < 10_000.0:
                    sat_weight = min(0.40, max(0.10, (10_000.0 - ore2_buffer) / 25_000.0))

            main_weight = 1.0 - sat_weight

            # 6. Extract material directly from sources
            main_target = mine_target * main_weight
            sat_target = mine_target * sat_weight

            flow_main = self.res_main.extract(main_target)
            flow_sat = self.res_sat.extract(sat_target)

            p_sat = flow_sat.attributes.get("ore2_fraction", 0.65)

            # Route to stockpiles
            inflow1 = Flow(
                rate=flow_main.rate * (1.0 - p1) + flow_sat.rate * (1.0 - p_sat),
                attributes={"ore2_fraction": 0.0},
            )
            inflow2 = Flow(
                rate=flow_main.rate * p1 + flow_sat.rate * p_sat,
                attributes={"ore2_fraction": 1.0},
            )

            out1 = self.ore1_stock.feed_and_draw(inflow1, ore1_rate)
            out2 = self.ore2_stock.feed_and_draw(inflow2, ore2_rate)

            blended_feed = blend_flows([out1, out2])
            self.mill.rate = blended_feed.rate
            self.cumulative_milled_mass.rate = self.mill.actual_rate

    def run(self, days: float = 365.0) -> pd.DataFrame:
        """Run the tactical simulation for specified duration and record telemetry."""
        telemetry = Telemetry(model=self.engine)
        telemetry.register_metric(
            "active_operating_mode",
            lambda t, m, s, _: self.mode_ctrl.active_operating_mode.value.name,
        )
        telemetry.register_metric(
            "ore1_stock",
            lambda t, m, s, _: self.ore1_stock.level,
        )
        telemetry.register_metric(
            "ore2_stock",
            lambda t, m, s, _: self.ore2_stock.level,
        )
        telemetry.register_metric(
            "total_stock",
            lambda t, m, s, _: self.ore1_stock.level + self.ore2_stock.level,
        )
        telemetry.register_metric(
            "main_face_extracted",
            lambda t, m, s, _: self.res_main.cumulative_extracted_mass.value,
        )
        telemetry.register_metric(
            "satellite_face_extracted",
            lambda t, m, s, _: self.res_sat.cumulative_extracted_mass.value,
        )
        telemetry.register_metric(
            "satellite_development",
            lambda t, m, s, _: self.satellite_cumulative_dev,
        )
        telemetry.register_metric(
            "satellite_unlocked",
            lambda t, m, s, _: 1.0 if self.satellite_unlocked else 0.0,
        )
        telemetry.register_metric(
            "cumulative_concentrator_throughput",
            lambda t, m, s, _: self.cumulative_milled_mass.value,
        )

        self.engine.attach_telemetry(telemetry)
        res = self.engine.run(until=days)
        df = telemetry.to_dataframe()
        if not df.empty and "time" in df.columns:
            df["day"] = df["time"]
        return df


def print_navarra_summary(
    df: pd.DataFrame, sim: NavarraTacticalSimulation
) -> None:
    """Prints comprehensive tactical operational metrics."""
    if df.empty:
        print("Telemetry history is empty.")
        return

    last = df.iloc[-1]
    days = last.get("day", 0.0)
    milled = last.get("cumulative_concentrator_throughput", 0.0)
    main_m = last.get("main_face_extracted", 0.0)
    sat_m = last.get("satellite_face_extracted", 0.0)
    dev_m = last.get("satellite_development", 0.0)

    print("=" * 60)
    print("      NAVARRA TACTICAL BLENDING SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Simulation Duration:       {days:.1f} days")
    print(f"Concentrator Throughput:   {milled:,.1f} tonnes")
    print(f"Daily Mill Average:        {milled / max(1.0, days):,.1f} t/day")
    print("-" * 60)
    print(f"Main Face Extracted:       {main_m:,.1f} tonnes ({main_m / max(1.0, main_m + sat_m) * 100:.1f}%)")
    print(f"Satellite Extracted:       {sat_m:,.1f} tonnes ({sat_m / max(1.0, main_m + sat_m) * 100:.1f}%)")
    print(f"Satellite Development:     {dev_m:.1f} / {sim.sat_required_dev_m:.1f} meters ({'UNLOCKED' if sim.satellite_unlocked else 'LOCKED'})")
    print("-" * 60)

    # Mode distribution
    if "active_operating_mode" in df.columns:
        mode_counts = df["active_operating_mode"].value_counts(normalize=True) * 100
        print("Concentrator Operating Mode Breakdown:")
        for mode, pct in mode_counts.items():
            print(f"  {mode:<25}: {pct:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Navarra Tactical Blending Simulation")
    parser.add_argument("--days", type=float, default=365.0, help="Simulation duration (days)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    sim = NavarraTacticalSimulation(seed=args.seed)
    df = sim.run(days=args.days)
    print_navarra_summary(df, sim)


if __name__ == "__main__":
    main()

"""Tactical Blending Simulation: Navarra Concentrator & Satellite Development.

Implements the tactical operational framework from NAVARRA_MEETING.md:
- Main Ore Body: High reserve, primary source, close to plant.
- Satellite Ore Body: Requires development to reach, sporadic daily availability,
  higher Ore 2 concentration for stockpile balancing, longer haul cycle times.
- Concentrator (Metallurgical Plant): Mode A (primary target), Mode B, and Contingency.
- Tactical Policy: Keep concentrator in Mode A while using the satellite as little as possible.
- Rough Truck Allocation: Cycle times determined by haul route distance and face congestion.
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
from drs import DRSEngine, Telemetry

from drs_mining.components import (
    HaulRoute,
    MetallurgicalPlant,
    MillingSetpoints,
    MineFace,
    OperatingMode,
    OperatingModeController,
    Parcel,
    Stockpile,
    StochasticFaciesGenerator,
    StochasticReserve,
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
    gen_main = StochasticFaciesGenerator(
        mean_fraction=0.30,
        std_dev=0.04,
        prob_new_facies=0.25,
        variation_same_facies=0.01,
    )
    res_main = StochasticReserve(
        name="main_reserve",
        total_tonnes=main_reserve_tonnes,
        generator=gen_main,
        min_parcel_mass=30_000.0,
        max_parcel_mass=50_000.0,
        initial_parcel_mass=40_000.0,
        seed=seed,
    )
    haul_main = HaulRoute(
        distance_km=1.2,
        base_cycle_time_min=16.0,
        congestion_factor=0.02,
        truck_payload_tonnes=100.0,
    )
    main_face = MineFace(
        name="main_face",
        geology=res_main,
        haulage=haul_main,
        max_rate=8000.0 / 86400.0,
    )

    # 2. Satellite Ore Body (35% Ore 1, 65% Ore 2 on average, far haulage, sporadic)
    gen_sat = StochasticFaciesGenerator(
        mean_fraction=0.65,
        std_dev=0.06,
        prob_new_facies=0.30,
        variation_same_facies=0.02,
    )
    res_sat = StochasticReserve(
        name="satellite_reserve",
        total_tonnes=sat_reserve_tonnes,
        generator=gen_sat,
        min_parcel_mass=25_000.0,
        max_parcel_mass=45_000.0,
        initial_parcel_mass=35_000.0,
        seed=seed + 100,
    )
    haul_sat = HaulRoute(
        distance_km=4.8,
        base_cycle_time_min=38.0,
        congestion_factor=0.05,
        truck_payload_tonnes=100.0,
    )
    satellite_face = MineFace(
        name="satellite_face",
        geology=res_sat,
        haulage=haul_sat,
        max_rate=5000.0 / 86400.0,
    )

    # 3. Stockpiles
    init_ore1 = target_ore_stock_level * 0.65
    init_ore2 = target_ore_stock_level * 0.35
    ore1_stock = Stockpile(
        name="Ore1Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=init_ore1,
        initial_attributes={"contained_ore_fraction_mass": init_ore1 * 0.30},
        attr_inflow=0.0,
    )
    ore2_stock = Stockpile(
        name="Ore2Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=init_ore2,
        initial_attributes={"contained_ore_fraction_mass": init_ore2 * 0.30},
        attr_inflow=0.0,
    )

    # 4. Concentrator (Metallurgical Plant)
    plant = MetallurgicalPlant(
        stockpiles=[ore1_stock, ore2_stock],
        setpoints=MillingSetpoints(
            mode_a_ore1=3600.0,
            mode_a_ore2=2400.0,
            mode_a_contingency_ore1=3900.0,
            mode_b_ore1=4600.0,
            mode_b_ore2=800.0,
            mode_b_contingency_ore2=2500.0,
        ),
        target_ore_stock_level=target_ore_stock_level,
        duration_of_contingency_segments=contingency_segment_duration,
    )

    # 5. Operating Mode Controller
    mode_ctrl = OperatingModeController(
        duration_of_production_campaigns=campaign_duration,
        duration_of_shutdowns=shutdown_duration,
        critical_ore2_level=critical_ore2_level,
    )

    return (
        main_face,
        satellite_face,
        plant,
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
            self.main_face,
            self.satellite_face,
            self.plant,
            self.mode_ctrl,
            self.ore1_stock,
            self.ore2_stock,
            self.sat_required_dev_m,
        ) = build_navarra_network(seed=seed)

        self.satellite_unlocked = False
        self.satellite_sporadic_available = False
        self.satellite_cumulative_dev = 0.0

        self.engine = DRSEngine()
        self._setup_policy()

    def _setup_policy(self) -> None:
        self.engine.register_processors(
            [self.main_face, self.satellite_face, self.plant]
        )
        self.engine.register_storages([self.ore1_stock, self.ore2_stock])
        self.engine.register_module(self.mode_ctrl)

        last_eval_day = [-1]

        def _policy(t: float):
            day = int(t // 86400.0)

            # Daily tactical review & sporadic evaluation
            if day > last_eval_day[0]:
                last_eval_day[0] = day
                self.satellite_sporadic_available = (
                    self.rng.random() < self.satellite_availability_prob
                )

            # 1. Update campaign mode
            active_campaign = self.mode_ctrl.update(
                ore2_stock_level=self.ore2_stock.level,
                total_stock_level=self.ore1_stock.level + self.ore2_stock.level,
            )

            # 2. Advance development when in Mode B or shutdown
            dt_step = 86400.0  # reference rate base
            if not self.satellite_unlocked:
                if "MODE_B" in active_campaign.name or active_campaign.name == "SHUTDOWN":
                    self.satellite_cumulative_dev += (
                        self.development_rate_m_per_day / 86400.0 * dt_step
                    )
                    if self.satellite_cumulative_dev >= self.sat_required_dev_m:
                        self.satellite_unlocked = True

            # 3. Determine available faces and blend fraction
            f_blend = 0.0
            p1 = (
                self.main_face.geology.active_parcel.ore2_fraction
                if self.main_face.geology.active_parcel
                else 0.30
            )
            f_blend = p1

            # 4. Get plant target rates
            ore1_rate, ore2_rate, mine_target = self.plant.get_target_rates(
                active_campaign,
                ore1_level=self.ore1_stock.level,
                ore2_level=self.ore2_stock.level,
                stockpile2_routing_fraction=f_blend,
            )

            # 5. Tactical Face Allocation (Navarra requirement: use satellite as little as possible)
            # Check if Ore 2 needs balancing and satellite is unlocked & available
            sat_weight = 0.0
            if (
                self.satellite_unlocked
                and self.satellite_sporadic_available
                and not self.satellite_face.is_exhausted
            ):
                ore2_buffer = self.ore2_stock.level - self.mode_ctrl.critical_ore2_level
                if ore2_buffer < 10_000.0:
                    # Allocate just enough to restore Ore 2 buffer
                    sat_weight = min(0.40, max(0.10, (10_000.0 - ore2_buffer) / 25_000.0))

            main_weight = 1.0 - sat_weight

            # 6. Apply extraction rates
            target_per_sec = mine_target / 86400.0
            self.main_face.target_rate = target_per_sec * main_weight
            self.satellite_face.target_rate = target_per_sec * sat_weight

            # 7. Collect mined parcels and route to stockpiles
            actual_main = self.main_face.actual_rate
            actual_sat = self.satellite_face.actual_rate

            p_sat = (
                self.satellite_face.geology.active_parcel.ore2_fraction
                if self.satellite_face.geology.active_parcel
                else 0.65
            )

            ore1_inflow = actual_main * (1.0 - p1) + actual_sat * (1.0 - p_sat)
            ore2_inflow = actual_main * p1 + actual_sat * p_sat

            out1 = self.ore1_stock.feed_and_draw(ore1_inflow, ore1_rate / 86400.0)
            out2 = self.ore2_stock.feed_and_draw(ore2_inflow, ore2_rate / 86400.0)

            self.plant.process(out1 + out2)

        self.engine.register_policy(_policy)

    def run(self, days: float = 365.0) -> pd.DataFrame:
        """Run the tactical simulation for specified duration and record telemetry."""
        until_sec = days * 86400.0

        telemetry = Telemetry(model=self.engine)
        telemetry.register_metric(
            "active_operating_mode",
            lambda t, m, s, _: self.plant.active_operating_mode.value.name,
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
            lambda t, m, s, _: self.main_face.geology.cumulative_extracted_mass.value,
        )
        telemetry.register_metric(
            "satellite_face_extracted",
            lambda t, m, s, _: self.satellite_face.geology.cumulative_extracted_mass.value,
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
            lambda t, m, s, _: self.plant.cumulative_milled_mass.value,
        )

        res = self.engine.run(until=until_sec, telemetry=telemetry)
        df = telemetry.to_dataframe()
        if not df.empty and "time" in df.columns:
            df["day"] = df["time"] / 86400.0
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
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    sim = NavarraTacticalSimulation(seed=args.seed)
    df = sim.run(days=args.days)
    print_navarra_summary(df, sim)


if __name__ == "__main__":
    main()

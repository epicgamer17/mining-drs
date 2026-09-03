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
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import drs
from drs import DRSEngine, Telemetry, Processor, Storage, Flow, blend_flows

from drs_mining.components import (
    OperatingMode,
    MaterialSource,
    autocorrelated_generator,
    truck_haul_capacity,
)

# ==============================================================================
# Navarra Simulation Hyperparameters & Constants
# ==============================================================================
MAIN_RESERVE_TONNES = 6_000_000.0
SAT_RESERVE_TONNES = 2_000_000.0
SAT_REQUIRED_DEV_METERS = 300.0

TARGET_ORE_STOCK_TONNES = 60_000.0
CRITICAL_ORE2_STOCK_TONNES = 20_400.0

CAMPAIGN_DURATION_DAYS = 34.0
SHUTDOWN_DURATION_DAYS = 1.0
CONTINGENCY_SEGMENT_DURATION_DAYS = 1.0

MILL_MAX_RATE_TPD = 6_000.0
SATELLITE_AVAILABILITY_PROB = 0.80
DEVELOPMENT_RATE_M_PER_DAY = 8.0


def create_navarra_modes() -> dict[str, OperatingMode]:
    """Instantiate discrete campaign and operating modes for the Navarra concentrator."""
    return {
        "MODE_A": OperatingMode("MODE_A", id=0, draw_rates={"Ore1Stock": 3600.0, "Ore2Stock": 2400.0}),
        "MODE_A_CONTINGENCY": OperatingMode("MODE_A_CONTINGENCY", id=1, draw_rates={"Ore1Stock": 3900.0, "Ore2Stock": 0.0}),
        "MODE_A_MINE_SURGING": OperatingMode("MODE_A_MINE_SURGING", id=2, draw_rates={"Ore1Stock": 3600.0, "Ore2Stock": 2400.0}),
        "MODE_B": OperatingMode("MODE_B", id=3, draw_rates={"Ore1Stock": 4600.0, "Ore2Stock": 800.0}),
        "MODE_B_CONTINGENCY": OperatingMode("MODE_B_CONTINGENCY", id=4, draw_rates={"Ore1Stock": 0.0, "Ore2Stock": 2500.0}),
        "MODE_B_MINE_SURGING": OperatingMode("MODE_B_MINE_SURGING", id=5, draw_rates={"Ore1Stock": 4600.0, "Ore2Stock": 800.0}),
        "SHUTDOWN": OperatingMode("SHUTDOWN", id=6, draw_rates={"Ore1Stock": 0.0, "Ore2Stock": 0.0}),
    }


class CampaignTimers(drs.Module):
    """Holds campaign and contingency countdown timers for DRS integration."""

    def __init__(
        self,
        campaign_duration: float = CAMPAIGN_DURATION_DAYS,
        contingency_duration: float = CONTINGENCY_SEGMENT_DURATION_DAYS,
    ):
        super().__init__()
        self.campaign_timer = drs.Timer("current_campaign_duration", initial_value=0.0)
        self.campaign_timer.rate = 1.0
        self.campaign_timer.upper_threshold = campaign_duration

        self.contingency_timer = drs.Timer("current_contingency_duration", initial_value=0.0)
        self.contingency_timer.rate = 0.0
        self.contingency_timer.upper_threshold = contingency_duration

    def levels(self) -> Sequence[drs.Level]:
        return (self.campaign_timer, self.contingency_timer)

    def variables(self) -> Iterator[drs.Variable]:
        yield self.campaign_timer
        yield self.contingency_timer


def update_campaign_mode(
    campaign_timer: drs.Timer,
    active_campaign: drs.Variable,
    ore2_stock_level: float,
    modes: Mapping[str, OperatingMode],
    campaign_duration: float = CAMPAIGN_DURATION_DAYS,
    shutdown_duration: float = SHUTDOWN_DURATION_DAYS,
    critical_ore2_level: float = CRITICAL_ORE2_STOCK_TONNES,
) -> OperatingMode:
    """Check campaign timer and transition between production and shutdown."""
    target_duration = shutdown_duration if active_campaign.value.name == "SHUTDOWN" else campaign_duration
    campaign_timer.upper_threshold = target_duration
    if campaign_timer.value >= (target_duration - 1e-6):
        campaign_timer.reset()
        if active_campaign.value.name == "SHUTDOWN":
            next_name = "MODE_A" if ore2_stock_level > critical_ore2_level else "MODE_B"
            active_campaign.value = modes[next_name]
        else:
            active_campaign.value = modes["SHUTDOWN"]

    target_duration = shutdown_duration if active_campaign.value.name == "SHUTDOWN" else campaign_duration
    campaign_timer.upper_threshold = target_duration
    return active_campaign.value


def resolve_operating_mode(
    campaign_mode: OperatingMode,
    current_mode: OperatingMode,
    ore1_level: float,
    ore2_level: float,
    contingency_timer: drs.Timer,
    modes: Mapping[str, OperatingMode],
    target_total_stock: float = TARGET_ORE_STOCK_TONNES,
    contingency_duration: float = CONTINGENCY_SEGMENT_DURATION_DAYS,
) -> OperatingMode:
    """Resolve active operating mode, handling contingencies and stockpile surging."""
    c_name = campaign_mode.name
    if c_name == "SHUTDOWN":
        return modes["SHUTDOWN"]

    total_stock = ore1_level + ore2_level
    current_name = current_mode.name

    if not current_name.startswith(c_name):
        return modes[f"{c_name}_MINE_SURGING"] if total_stock > target_total_stock + 1e-6 else modes[c_name]

    if "_CONTINGENCY" in current_name:
        if contingency_timer.value >= (contingency_duration - 1e-6):
            contingency_timer.reset()
            contingency_timer.rate = 0.0
            return modes[c_name]
        return modes[current_name]

    # Check starvation triggers for contingency
    if c_name == "MODE_A" and ore2_level <= 1e-6:
        contingency_timer.reset()
        contingency_timer.upper_threshold = contingency_duration
        contingency_timer.rate = 1.0
        return modes["MODE_A_CONTINGENCY"]
    elif c_name == "MODE_B" and ore1_level <= 1e-6:
        contingency_timer.reset()
        contingency_timer.upper_threshold = contingency_duration
        contingency_timer.rate = 1.0
        return modes["MODE_B_CONTINGENCY"]

    # Surging checks
    if total_stock > target_total_stock + 1e-6:
        return modes[f"{c_name}_MINE_SURGING"]
    return modes[c_name]


def build_navarra_network(
    seed: int = 42,
    main_reserve_tonnes: float = MAIN_RESERVE_TONNES,
    sat_reserve_tonnes: float = SAT_RESERVE_TONNES,
    sat_required_dev_m: float = SAT_REQUIRED_DEV_METERS,
    target_ore_stock_level: float = TARGET_ORE_STOCK_TONNES,
    critical_ore2_level: float = CRITICAL_ORE2_STOCK_TONNES,
    contingency_segment_duration: float = CONTINGENCY_SEGMENT_DURATION_DAYS,
    campaign_duration: float = CAMPAIGN_DURATION_DAYS,
    shutdown_duration: float = SHUTDOWN_DURATION_DAYS,
    modes: Optional[Mapping[str, OperatingMode]] = None,
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
    mill = Processor(name="mill", max_rate=MILL_MAX_RATE_TPD)

    navarra_modes = dict(modes or create_navarra_modes())
    timers = CampaignTimers(
        campaign_duration=campaign_duration,
        contingency_duration=contingency_segment_duration,
    )
    active_campaign = drs.Variable("active_campaign_mode", navarra_modes["MODE_A"])
    active_mode = drs.Variable("active_operating_mode", navarra_modes["MODE_A"])
    navarra_modes["MODE_A"].activate()

    control_state = {
        "timers": timers,
        "campaign_timer": timers.campaign_timer,
        "contingency_timer": timers.contingency_timer,
        "active_campaign": active_campaign,
        "active_mode": active_mode,
        "modes": navarra_modes,
        "campaign_duration": campaign_duration,
        "shutdown_duration": shutdown_duration,
        "contingency_duration": contingency_segment_duration,
        "critical_ore2_level": critical_ore2_level,
        "target_total_stock": target_ore_stock_level,
    }

    return (
        res_main,
        res_sat,
        mill,
        control_state,
        ore1_stock,
        ore2_stock,
        sat_required_dev_m,
    )


class NavarraTacticalSimulation:
    """Manages the DRS simulation run, tactical blending policy, and telemetry."""

    def __init__(
        self,
        seed: int = 42,
        satellite_availability_prob: float = SATELLITE_AVAILABILITY_PROB,
        development_rate_m_per_day: float = DEVELOPMENT_RATE_M_PER_DAY,
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.satellite_availability_prob = satellite_availability_prob
        self.development_rate_m_per_day = development_rate_m_per_day

        (
            self.res_main,
            self.res_sat,
            self.mill,
            self.control_state,
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
            self.ore1_stock,
            self.ore2_stock,
            self.control_state["timers"],
            *self.control_state["modes"].values(),
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
            active_campaign = update_campaign_mode(
                self.control_state["campaign_timer"],
                self.control_state["active_campaign"],
                ore2_stock_level=self.ore2_stock.level,
                modes=self.control_state["modes"],
                campaign_duration=self.control_state["campaign_duration"],
                shutdown_duration=self.control_state["shutdown_duration"],
                critical_ore2_level=self.control_state["critical_ore2_level"],
            )

            # 2. Advance development when in Mode B or shutdown
            if not self.satellite_unlocked:
                if "MODE_B" in active_campaign.name or active_campaign.name == "SHUTDOWN":
                    self.satellite_cumulative_dev += self.development_rate_m_per_day
                    if self.satellite_cumulative_dev >= self.sat_required_dev_m:
                        self.satellite_unlocked = True

            # 3. Resolve active operating mode (handling contingency / surging)
            active_mode = resolve_operating_mode(
                active_campaign,
                self.control_state["active_mode"].value,
                ore1_level=self.ore1_stock.level,
                ore2_level=self.ore2_stock.level,
                contingency_timer=self.control_state["contingency_timer"],
                modes=self.control_state["modes"],
                target_total_stock=self.control_state["target_total_stock"],
                contingency_duration=self.control_state["contingency_duration"],
            )

            if self.control_state["active_mode"].value != active_mode:
                self.control_state["active_mode"].value.deactivate()
                active_mode.activate()
                self.control_state["active_mode"].value = active_mode

            # 4. Get draw rates
            draw_rates = active_mode.draw_rates
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
                ore2_buffer = self.ore2_stock.level - self.control_state["critical_ore2_level"]
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
            lambda t, m, s, _: self.control_state["active_mode"].value.name,
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
        self.engine.run(until=days)
        return telemetry.to_dataframe()


def main():
    parser = argparse.ArgumentParser(description="Navarra Tactical Blending Simulation")
    parser.add_argument("--days", type=float, default=365.0, help="Simulation duration (days)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="navarra_tactical_results.csv")
    args = parser.parse_args()

    print(f"Starting Navarra Tactical Blending Simulation ({args.days} days, seed={args.seed})...")
    sim = NavarraTacticalSimulation(seed=args.seed)
    df = sim.run(days=args.days)

    total_throughput = sim.cumulative_milled_mass.value
    sat_unlocked = sim.satellite_unlocked
    main_extracted = sim.res_main.cumulative_extracted_mass.value
    sat_extracted = sim.res_sat.cumulative_extracted_mass.value
    total_extracted = main_extracted + sat_extracted
    sat_share = (sat_extracted / total_extracted * 100.0) if total_extracted > 0 else 0.0

    print(f"\n--- Simulation Results ({args.days:.0f} Days) ---")
    print(f"Cumulative Concentrator Throughput : {total_throughput:,.0f} tonnes")
    print(f"Main Face Extraction              : {main_extracted:,.0f} tonnes")
    print(f"Satellite Face Extraction         : {sat_extracted:,.0f} tonnes ({sat_share:.1f}%)")
    print(f"Satellite Developed Meters         : {sim.satellite_cumulative_dev:.1f} / {sim.sat_required_dev_m:.1f} m")
    print(f"Satellite Unlocked Status          : {'YES' if sat_unlocked else 'NO'}")

    modes = sim.control_state["modes"]
    print("\nOperating Mode Breakdown:")
    for name, mode in modes.items():
        pct = (mode.cumulative_time / args.days * 100.0) if args.days > 0 else 0.0
        print(f"  {name:20s}: {mode.cumulative_time:6.1f} days ({pct:5.1f}%)")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nTelemetry saved to {args.output}")


if __name__ == "__main__":
    main()

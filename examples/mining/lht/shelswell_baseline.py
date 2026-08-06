"""
Shelswell (2017) Baseline Truck Haulage Simulation & Paper Notes

===============================================================================
1. SUMMARY OF SHELSWELL (2017) PAPER SPECIFICATIONS
===============================================================================

Trucks vary in size, capacity and dump method to accommodate mine designs. Additional options include engine type, or modifications like sideboards.

Key strength to truck haulage is the ability to adapt accordingly in response to the current operations, future mining processes, and expansions or changes in mine plans as the life of the mine progresses.

Especially important in the absence of rail lines, conveyors and shafts.

Key limitations: wear and tear on equipment. Downtime is impacted by: fleet age, muck fragmentation, truck payloads, truck loading practices, roadway conditions, ventilation and cooling, haulage distance, productivity targets, maintenance regimes, and mine design.

It is important to maximize truck availability with respect to the underground conditions.

Operators are also important. Too few and available trucks can't be used, too many is unnecessary cost.

Need operator and equipment availability with randomness.

Upstream boundary used was the production of ore from stopes and the generation of waste from lateral development.
Assumed that mining activities were able to efficiently generate sufficient tonnes to meet the production and development targets for the simulation.

Downstream boundary was the dumping of material at the run of mine pad and waste stockpile site on surface. ROM and stockpile were considered unconstrained.

Exact mine design:
- 2100m long 5% grade decline access to a single spiral ramp system.
- Ramp was 1800m long and a grade between 8 and 13% with seven primary mine levels. Sublevels not included.
- Distance between each mine level on the ramp was 300m.
- Each level had 2 loadouts: one for ore and one for waste along the access drift of the ramp.
- Ore loadouts were 40m off the ramp while waste loadouts were 55m from the ramp.
- Single truck air doors were incorporated 20m down the level access drift coming off the ramp to control ventilation.
- ROM was located 300m from the portal while the waste stockpile dump was located 440m from the portal.
- Maintenance shop and fuel depot were located on the surface at 260m and 270m respectively.

Production & Schedule:
- The production and development schedule represented mine targets for 1 calendar year (365 days) with a 5.5:1 ratio of ore to waste.
- 11 non-production days were included for holidays, maintenance events, and random shutdowns.
- Ore and waste tonnes were scheduled daily based on a triangular distribution. Tonnes scheduled additively to muck bays.

Shift Schedules & Availabilities:
- 2 shifts per day, 10.5 hours each (12h elapsed).
- Workable availabilities: Haulage 54.17%, Underground LHD 58.33%, Surface Maintenance 79.17%.

Equipment Specs:
- Payload - ore: Truck 26.1 t, LHD 14.0 t
- Payload - waste: Truck 24.6 t, LHD 12.5 t
- Load spot duration: Truck 0.82 min, LHD 0.46 min
- Load duration (ore/waste): Truck 6.69 min, LHD 0.88 min
- Dump spot duration: Truck 0.57 min, LHD 0.55 min
- Dump duration: Truck 0.88 min, LHD 0.73 min
- Speed - surface: loaded 13.4 kph, empty 17.4 kph
- Speed - decline: loaded 11.2 kph, empty 15.1 kph
- Speed - ramp: loaded 9.2 kph, empty 12.9 kph
- Speed - level: loaded 6.6 kph, empty 7.6 kph (LHD: loaded 5.89 kph, empty 6.78 kph)
- Acquisition delay max: 3.0 min (avg 1.5 min)
- Remuck stockpile tram distance: 35 m
- Scheduled PM-associated availability: 99.8 - 100%
- Random failure-associated availability: 30.2 - 95%

Dispatch & Operation Rules:
- Payload type determined by 5.5:1 ore vs waste ratio.
- Dispatched to loadout with highest "unclaimed" tonnes remaining.
- 1 LHD active per level.
- Bounded effective fleet: eff_trucks = min(N_trucks * Availability, N_operators).
- Availability formulas:
  PM Availability = (FreqPM / (FreqPM + DurPM)) * (FreqFUEL / (FreqFUEL + DurFUEL))
  Random Failure Availability = (AVGMTBF / (AVGMTBF + AVGMTTR))
  Overall Mechanical Availability = PM Availability * Random Failure Availability

===============================================================================
2. DRS IMPLEMENTATION DIFFERENCES FROM PAPER
===============================================================================

1. Simulation Methodology (Discrete Rate vs. Discrete Event):
   - Paper: A true Discrete Event Simulation (DES) where individual trucks and loader units are simulated as independent entities executing discrete tasks.
   - Implementation: A Discrete Rate Simulation (DRS) where flow capacities are represented as continuous rates (tonnes/day). Capacity limits are solved analytically at each time step rather than simulated step-by-step.

2. Truck & Loader Cycles:
   - Paper: Truck loading times are simulated using a series of stochastic LHD loading bucket cycles, spot times, and uniform acquisition delays.
   - Implementation: The loading cycle is represented analytically based on average load spot, average acquisition delay, and LHD bucket cycle times.

3. Operator Pooling and Fleet Constraints:
   - Paper: Explicitly models operators and trucks as individual resources that must be acquired. Available trucks remain idle if operators are unavailable.
   - Implementation: Represented analytically by capping the effective fleet size: eff_trucks = min(N_trucks * Availability, N_operators).

4. Mechanical Availability:
   - Paper: Simulates scheduled planned maintenance (PM) at utilization intervals and random breakdowns using MTBF/MTTR probability distributions.
   - Implementation: Approximated analytically by scaling down the effective fleet size by the overall mechanical availability bracket.

5. Traffic & Congestion Delays:
   - Paper: Restricts decline/ramp roadway segments to single-direction travel with passing pull-outs. Loaded trucks are prioritized, and congestion creates queuing bottlenecks.
   - Implementation: Approximated using a linear traffic delay penalty applied to truck cycle times, scaled by the number of trucks allocated per level.
"""

from dataclasses import dataclass
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import drs


@dataclass
class ShelswellConfig:
    """Configuration parameters for the Shelswell (2017) mine design and haulage simulation."""

    # Mine Design Parameters
    decline_length: float = 2100.0
    decline_grade: float = 0.05
    ramp_length: float = 1800.0
    ramp_grades: tuple = (0.08, 0.13)
    level_spacing: float = 300.0
    ore_loadout_distance: float = 40.0
    waste_loadout_distance: float = 55.0
    air_door_distance: float = 20.0
    rom_pad_distance: float = 300.0
    waste_stockpile_distance: float = 440.0
    maintenance_shop_distance: float = 260.0
    fuel_depot_distance: float = 270.0

    # Speeds (kph)
    speed_surface_loaded: float = 13.4
    speed_surface_empty: float = 17.4
    speed_decline_loaded: float = 11.2
    speed_decline_empty: float = 15.1
    speed_ramp_loaded: float = 9.2
    speed_ramp_empty: float = 12.9
    speed_level_loaded: float = 6.6
    speed_level_empty: float = 7.6
    speed_lhd_loaded: float = 5.89
    speed_lhd_empty: float = 6.78

    # Shift "Seat Time" Workable Availability fractions
    seat_time_fraction_truck: float = 0.5417
    seat_time_fraction_lhd: float = 0.5833
    seat_time_fraction_maintenance: float = 0.7917

    # Cycle times (minutes)
    lhd_load_spot_minutes: float = 0.46
    lhd_load_minutes: float = 0.88
    lhd_dump_minutes: float = 0.73
    lhd_tram_distance: float = 35.0
    lhd_acquisition_delay_minutes: float = 1.5

    truck_load_spot_minutes: float = 0.82
    truck_dump_spot_minutes: float = 0.57
    truck_dump_minutes: float = 0.88

    # Capacity and payloads (tonnes)
    truck_payload_ore: float = 26.1
    truck_payload_waste: float = 24.6
    lhd_payload_ore: float = 14.0
    lhd_payload_waste: float = 12.5

    # Fleet sizes and availability (defaults)
    total_truck_count: float = 10.0
    total_operators: float = 10.0
    overall_mechanical_availability: float = 1.0
    traffic_delay_per_truck_hours: float = 0.005


def get_travel_times_hours(
    level: int, muck_type: str, config: ShelswellConfig
) -> tuple[float, float]:
    """Calculates empty and loaded travel times in hours for a given level.

    Levels are 1-indexed (1 to 7).
    muck_type is either 'ore' or 'waste'.
    """
    v_surf_empty = config.speed_surface_empty * 1000.0
    v_surf_loaded = config.speed_surface_loaded * 1000.0
    v_dec_empty = config.speed_decline_empty * 1000.0
    v_dec_loaded = config.speed_decline_loaded * 1000.0
    v_ramp_empty = config.speed_ramp_empty * 1000.0
    v_ramp_loaded = config.speed_ramp_loaded * 1000.0
    v_lvl_empty = config.speed_level_empty * 1000.0
    v_lvl_loaded = config.speed_level_loaded * 1000.0

    d_dec = config.decline_length
    d_ramp = (level - 1) * config.level_spacing

    if muck_type == "ore":
        d_lvl = config.ore_loadout_distance
        d_surf = config.rom_pad_distance
    else:
        d_lvl = config.waste_loadout_distance
        d_surf = config.waste_stockpile_distance

    # Empty path travel time (ROM/Stockpile -> Portal -> Decline -> Ramp -> Level Loadout)
    t_empty = (
        (d_surf / v_surf_empty)
        + (d_dec / v_dec_empty)
        + (d_ramp / v_ramp_empty)
        + (d_lvl / v_lvl_empty)
    )

    # Loaded path travel time (Level Loadout -> Ramp -> Decline -> Portal -> ROM/Stockpile)
    t_loaded = (
        (d_lvl / v_lvl_loaded)
        + (d_ramp / v_ramp_loaded)
        + (d_dec / v_dec_loaded)
        + (d_surf / v_surf_loaded)
    )

    return t_empty, t_loaded


def get_truck_loading_time_hours(muck_type: str, config: ShelswellConfig) -> float:
    """Calculates LHD cycle time and total time to load a truck in hours."""
    v_lhd_empty = config.speed_lhd_empty * 1000.0
    v_lhd_loaded = config.speed_lhd_loaded * 1000.0

    t_lhd_tram_empty_min = config.lhd_tram_distance / v_lhd_empty * 60.0
    t_lhd_tram_loaded_min = config.lhd_tram_distance / v_lhd_loaded * 60.0

    t_lhd_bucket_cycle_min = (
        config.lhd_load_spot_minutes
        + config.lhd_load_minutes
        + t_lhd_tram_loaded_min
        + config.lhd_dump_minutes
        + t_lhd_tram_empty_min
    )

    # 2 buckets per truck
    t_load_truck_min = (
        config.lhd_acquisition_delay_minutes
        + config.truck_load_spot_minutes
        + 2 * t_lhd_bucket_cycle_min
    )
    return t_load_truck_min / 60.0


class ShelswellHaulageModel(drs.Module):
    """Discrete Rate Simulation model approximating the Shelswell (2017) truck haulage fleet simulation."""

    def __init__(self, config: ShelswellConfig):
        super().__init__()
        self.config = config

        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)
        self.schedule_timer = drs.Timer("ScheduleTimer", initial_value=0.0)
        self.schedule_timer.upper_threshold = 1.0

        # Muck piles representing scheduled ore/waste not yet hauled
        self.ore_muck = drs.Level("OreMuck", initial_value=100000.0)
        self.waste_muck = drs.Level("WasteMuck", initial_value=20000.0)

        # Total hauled tonnes
        self.ore_hauled = drs.Level("OreHauled", initial_value=0.0)
        self.waste_hauled = drs.Level("WasteHauled", initial_value=0.0)

        # Instantaneous haul rates
        self.ore_haul_rate = drs.Variable("ore_haul_rate", 0.0)
        self.waste_haul_rate = drs.Variable("waste_haul_rate", 0.0)

        # Parametric fleet inputs
        self.truck_count = drs.Variable("truck_count", config.total_truck_count)
        self.operator_count = drs.Variable("operator_count", config.total_operators)
        self.availability = drs.Variable(
            "availability", config.overall_mechanical_availability
        )

    def forward(self):
        self.global_time.rate = 1.0
        self.schedule_timer.rate = 1.0

        # Daily scheduling update
        if self.schedule_timer.value >= 1.0 - 1e-6:
            self.schedule_timer.reset()
            ore_scheduled = random.triangular(3900.0, 8100.0, 6000.0)
            waste_scheduled = random.triangular(436.4, 1745.4, 1090.9)

            self.ore_muck.value += ore_scheduled
            self.waste_muck.value += waste_scheduled

        num_trucks = self.truck_count.value
        num_operators = self.operator_count.value
        avail = self.availability.value

        # Bounded effective fleet size based on mechanical availability and operators
        eff_trucks = min(num_trucks * avail, num_operators)

        # Number of active levels based on paper's configuration
        active_levels_count = max(1, min(7, int(num_trucks) // 2 + 1))

        start_lvl = max(1, 4 - (active_levels_count - 1) // 2)
        active_levels = list(range(start_lvl, start_lvl + active_levels_count))

        trucks_per_level = eff_trucks / len(active_levels)

        total_ore_cap = 0.0
        total_waste_cap = 0.0

        for lvl in active_levels:
            frac_ore = 5.5 / 6.5
            frac_waste = 1.0 / 6.5

            # --- Ore Haulage Capacity ---
            t_empty_ore, t_loaded_ore = get_travel_times_hours(lvl, "ore", self.config)
            t_load_ore = get_truck_loading_time_hours("ore", self.config)
            t_dump_ore = (
                self.config.truck_dump_spot_minutes + self.config.truck_dump_minutes
            ) / 60.0
            t_traffic = self.config.traffic_delay_per_truck_hours * trucks_per_level

            t_cycle_ore = (
                t_empty_ore + t_load_ore + t_loaded_ore + t_dump_ore + t_traffic
            )

            mf_ore = (trucks_per_level * t_load_ore) / t_cycle_ore
            if mf_ore < 1.0:
                ore_cap = (
                    (trucks_per_level * frac_ore)
                    * (24.0 * self.config.seat_time_fraction_truck / t_cycle_ore)
                    * self.config.truck_payload_ore
                )
            else:
                ore_cap = (
                    (1.0 * frac_ore)
                    * (24.0 * self.config.seat_time_fraction_lhd / t_load_ore)
                    * self.config.truck_payload_ore
                )

            # --- Waste Haulage Capacity ---
            t_empty_waste, t_loaded_waste = get_travel_times_hours(
                lvl, "waste", self.config
            )
            t_load_waste = get_truck_loading_time_hours("waste", self.config)
            t_dump_waste = (
                self.config.truck_dump_spot_minutes + self.config.truck_dump_minutes
            ) / 60.0

            t_cycle_waste = (
                t_empty_waste + t_load_waste + t_loaded_waste + t_dump_waste + t_traffic
            )

            mf_waste = (trucks_per_level * t_load_waste) / t_cycle_waste
            if mf_waste < 1.0:
                waste_cap = (
                    (trucks_per_level * frac_waste)
                    * (24.0 * self.config.seat_time_fraction_truck / t_cycle_waste)
                    * self.config.truck_payload_waste
                )
            else:
                waste_cap = (
                    (1.0 * frac_waste)
                    * (24.0 * self.config.seat_time_fraction_lhd / t_load_waste)
                    * self.config.truck_payload_waste
                )

            total_ore_cap += ore_cap
            total_waste_cap += waste_cap

        self.ore_haul_rate.value = total_ore_cap
        self.waste_haul_rate.value = total_waste_cap

        self.ore_muck.rate = -self.ore_haul_rate.value
        self.waste_muck.rate = -self.waste_haul_rate.value
        self.ore_hauled.rate = self.ore_haul_rate.value
        self.waste_hauled.rate = self.waste_haul_rate.value

        self.ore_muck.lower_threshold = 0.0
        self.waste_muck.lower_threshold = 0.0

    def is_terminating_condition_met(self) -> bool:
        return self.global_time.value >= 365.0


def run_simulation(trucks: int, operators: int, availability: float) -> float:
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    config = ShelswellConfig()

    # Run parameters
    model = ShelswellHaulageModel(config)
    model.truck_count.value = float(trucks)
    model.operator_count.value = float(operators)
    model.availability.value = float(availability)

    engine = drs.DRSEngine(model)
    # Run for 354 production days
    engine.run(max_time=354.0)

    total_hauled = model.ore_hauled.value + model.waste_hauled.value
    avg_daily_productivity = total_hauled / 354.0
    return avg_daily_productivity


def generate_figure_2():
    print(
        "Generating Figure 2 (Productivity vs Fleet Size without operator constraints)..."
    )
    availabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    truck_sizes = list(range(3, 11))

    # Base normalization value: 10 trucks, 10 operators at 1.0 availability
    base_prod = run_simulation(10, 10, 1.0)
    print(f"Base productivity (10 trucks, 10 ops, 100% avail): {base_prod:.2f} t/d")

    plt.figure(figsize=(10, 6))

    colors = {
        0.5: "green",
        0.6: "brown",
        0.7: "orange",
        0.8: "blue",
        0.9: "hotpink",
        1.0: "black",
    }

    for avail in availabilities:
        prod_list = []
        for trucks in truck_sizes:
            prod = run_simulation(trucks, trucks, avail)
            prod_list.append(prod / base_prod)

        plt.plot(
            truck_sizes,
            prod_list,
            marker="o",
            label=f"{int(avail*100)}% availability",
            color=colors[avail],
        )

    plt.title(
        "Haulage productivity analysis without haulage operator constraints",
        fontsize=14,
    )
    plt.xlabel("Number of trucks in fleet", fontsize=12)
    plt.ylabel("Normalised productivity", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.ylim(0, 1.1)

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/shelswell_fig2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/shelswell_fig2.png")


def generate_figures_3_to_8():
    availabilities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    operator_counts = list(range(1, 11))
    truck_counts = list(range(3, 11))

    colors = {
        3: "green",
        4: "brown",
        5: "orange",
        6: "blue",
        7: "hotpink",
        8: "purple",
        9: "yellow",
        10: "black",
    }

    for avail in availabilities:
        print(
            f"Generating productivity curves for {int(avail*100)}% mechanical availability..."
        )
        # Base normalization: 10 trucks, 10 operators at the current availability
        base_prod = run_simulation(10, 10, avail)

        plt.figure(figsize=(10, 6))
        for trucks in truck_counts:
            prod_list = []
            for ops in operator_counts:
                # Operator count is capped at number of trucks for this sweep
                prod = run_simulation(trucks, min(ops, trucks), avail)
                prod_list.append(prod / base_prod)

            plt.plot(
                operator_counts,
                prod_list,
                marker="o",
                label=f"{trucks} trucks",
                color=colors[trucks],
            )

        plt.title(
            f"Haulage fleet size productivity analysis with operator constraints ({int(avail*100)}% Availability)",
            fontsize=13,
        )
        plt.xlabel("Number of haulage operators", fontsize=12)
        plt.ylabel("Normalised productivity", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="lower right")
        plt.ylim(0, 1.1)

        fig_num = {1.0: 3, 0.9: 4, 0.8: 5, 0.7: 6, 0.6: 7, 0.5: 8}[avail]
        plt.savefig(f"plots/shelswell_fig{fig_num}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plots/shelswell_fig{fig_num}.png")


if __name__ == "__main__":
    generate_figure_2()
    generate_figures_3_to_8()
    print("Replication of all figures complete!")

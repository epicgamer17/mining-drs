"""
Shelswell (2017) Baseline Truck Haulage Simulation (Hybrid DRS Architecture)

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
This implementation uses a Pythonic Three-Layer Hybrid DRS Pattern:
- Layer 1: Discrete Domain Model (Trucks, LHDs, Fuel/Maintenance state, Unclaimed Tonnes Dispatch Logic).
- Layer 2: DRS Interface & Rate Gateways (Loading/Dumping Impulse Rate Enforcers, Continuous Road Availability Timers).
- Layer 3: DRS Core Engine (`drs` library continuous numerical integration loop).
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from drs_mining.simulation import ShelswellHybridSimulation
from drs_mining.components.fleet import Truck, TruckState, LHD
from drs_mining.components.topology import DRSRoadSegment
from drs_mining.components.bays import DRSLoadingBay, DRSDumpingBay
from drs_mining.controllers.dispatch import ShelswellDispatchController


def run_simulation(trucks: int, operators: int, availability: float) -> float:
    """Executes a single Hybrid DRS simulation run for a given fleet configuration.

    Args:
        trucks: Number of haul trucks in fleet (3 to 10).
        operators: Number of truck operators available per shift (1 to 10).
        availability: Mechanical availability fraction (0.5 to 1.0).

    Returns:
        float: Average daily haulage productivity (tonnes/day).
    """
    random.seed(42)
    np.random.seed(42)

    sim = ShelswellHybridSimulation(
        num_trucks=trucks,
        num_operators=operators,
        mechanical_availability=availability,
    )
    return sim.run_simulation(total_days=365.0)


def generate_figure_2():
    """Replicates Figure 2 from Shelswell (2017): Productivity vs Fleet Size without operator constraints."""
    print(
        "Generating Figure 2 (Productivity vs Fleet Size without operator constraints)..."
    )
    availabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    truck_sizes = list(range(3, 11))

    # Baseline reference productivity (10 trucks, 10 ops, 100% availability)
    base_prod = run_simulation(10, 10, 1.0)
    print(
        f"Base Hybrid DRS productivity (10 trucks, 10 ops, 100% avail): {base_prod:.2f} t/d"
    )

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
        "Haulage productivity analysis without haulage operator constraints (Hybrid DRS)",
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
    """Replicates Figures 3 through 8 from Shelswell (2017): Productivity vs Operator Count."""
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
        base_prod = run_simulation(10, 10, avail)

        plt.figure(figsize=(10, 6))
        for trucks in truck_counts:
            prod_list = []
            for ops in operator_counts:
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
    print("Replication of all figures complete with Hybrid DRS Architecture!")

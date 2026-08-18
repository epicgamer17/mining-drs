"""
Shelswell (2017) Baseline Truck Haulage Simulation (Hybrid DRS Architecture)

===============================================================================
1. SUMMARY OF SHELSWELL (2017) PAPER SPECIFICATIONS
===============================================================================
- Mine Layout: 2100m access decline (5% grade) + 1800m spiral ramp connecting 7 mine levels (300m spacing).
- Loadouts: 40m access drift off ramp for Ore, 55m access drift for Waste. Air doors 20m off ramp.
- Surface Destinations: ROM crusher pad at 300m, Waste Dump at 440m, Maintenance Shop at 260m, Fuel Depot at 270m.
- Production Schedule: 365 calendar days with 5.5:1 ratio of Ore to Waste.
- Shift Schedule: 2 shifts per day, 10.5 hours working time each (12h total, 1.5h shift gap).
- Equipment Specs: CAT AD30 haul trucks (26.1t Ore, 24.6t Waste) and underground LHD loaders (14.0t Ore, 12.5t Waste).

===============================================================================
2. THREE-LAYER HYBRID DISCRETE RATE SYSTEM (DRS) ARCHITECTURE
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
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import drs
from drs_mining.components.fleet import Truck, TruckState, LHD
from drs_mining.components.topology import RoadSegment
from drs_mining.components.factories import load_topology_dict
from drs_mining.components.bays import LoadingBay, DumpingBay
from drs_mining.components.dispatch import ShelswellDispatchController


def build_shelswell_simulation(
    num_trucks: int = 10,
    num_operators: int = 10,
    mechanical_availability: float = 1.0,
    topology_dict=None,
) -> dict:
    """Constructs the haulage simulation state from library components.

    Returns:
        dict: The simulation state (roads, bays, trucks, dispatch, arrays,
        timers) consumed by :func:`step_simulation` / :func:`run_haulage_simulation`.
    """
    if topology_dict is not None:
        top_data = load_topology_dict(topology_dict)
        if isinstance(top_data, dict) and "attributes" in top_data:
            attrs = top_data["attributes"]
            num_trucks = attrs.get("num_trucks", num_trucks)
            num_operators = attrs.get("num_operators", num_operators)
            mechanical_availability = attrs.get(
                "mechanical_availability", mechanical_availability
            )

    engine = drs.DRSEngine()

    # Mine topology & availability timers
    decline = RoadSegment("decline_2100m", 2100.0, "decline")
    ramp_levels = [
        RoadSegment(f"ramp_L{i}", 300.0, "ramp")
        for i in range(1, 8)
    ]
    surface_rom_road = RoadSegment("surf_rom", 300.0, "surface")
    surface_waste_road = RoadSegment("surf_waste", 440.0, "surface")

    # Loading & dumping bays (unconstrained upstream muck supply per paper spec)
    loading_bays = []
    for i in range(1, 8):
        lhd = LHD(
            lhd_id=f"LHD_L{i}",
            level_index=i,
            bucket_ore_cap=14.0,
            bucket_waste_cap=12.5,
            load_spot_min=0.46,
            load_min=0.88,
            dump_min=0.73,
            tram_dist_m=35.0,
            speed_loaded_kph=5.89,
            speed_empty_kph=6.78,
        )
        loading_bays.append(
            LoadingBay(
                bay_id=f"L{i}_ORE",
                bay_type="ORE",
                level_index=i,
                initial_muck=10_000_000.0,
                truck_spot_min=0.82,
                acquisition_delay_min=1.5,
                bucket_passes=2.0,
                lhd=lhd,
            )
        )
        loading_bays.append(
            LoadingBay(
                bay_id=f"L{i}_WASTE",
                bay_type="WASTE",
                level_index=i,
                initial_muck=2_000_000.0,
                truck_spot_min=0.82,
                acquisition_delay_min=1.5,
                bucket_passes=2.0,
                lhd=lhd,
            )
        )

    rom_dump_bay = DumpingBay(
        bay_id="ROM_PAD",
        bay_type="ORE",
        location_name="SURFACE_ROM",
        dump_spot_min=0.57,
        bed_raise_dump_min=0.88,
    )
    waste_dump_bay = DumpingBay(
        bay_id="WASTE_DUMP",
        bay_type="WASTE",
        location_name="SURFACE_WASTE_DUMP",
        dump_spot_min=0.57,
        bed_raise_dump_min=0.88,
    )
    dump_bays = [rom_dump_bay, waste_dump_bay]

    # Fleet & vectorized state arrays
    eff_trucks_count = int(min(num_trucks * mechanical_availability, num_operators))
    eff_trucks_count = max(1, eff_trucks_count)

    truck_speeds = {
        "surface": {"empty": 17.4, "loaded": 13.4},
        "decline": {"empty": 15.1, "loaded": 11.2},
        "ramp":    {"empty": 12.9, "loaded": 9.2},
        "level":   {"empty": 7.6,  "loaded": 6.6},
    }

    trucks = [
        Truck(
            truck_id=f"T{i:02d}",
            truck_type="AD30",
            ore_payload_cap=26.1,
            waste_payload_cap=24.6,
            fuel_burn_rate_pct_per_sec=0.005,
            speeds=truck_speeds,
        )
        for i in range(1, eff_trucks_count + 1)
    ]
    dispatch = ShelswellDispatchController(
        trucks=trucks,
        loading_bays=loading_bays,
        roads={},
        waste_trip_interval=13,
    )


    timers = np.zeros(len(trucks), dtype=np.float64)
    fuel_pct = np.full(len(trucks), 100.0, dtype=np.float64)

    global_time = drs.Timer("GlobalTime", initial_value=0.0)
    ore_hauled = drs.Level("OreHauled", initial_value=0.0)
    waste_hauled = drs.Level("WasteHauled", initial_value=0.0)

    return {
        "engine": engine,
        "decline": decline,
        "ramp_levels": ramp_levels,
        "surface_rom_road": surface_rom_road,
        "surface_waste_road": surface_waste_road,
        "loading_bays": loading_bays,
        "rom_dump_bay": rom_dump_bay,
        "waste_dump_bay": waste_dump_bay,
        "dump_bays": dump_bays,
        "trucks": trucks,
        "dispatch": dispatch,
        "timers": timers,
        "fuel_pct": fuel_pct,
        "global_time": global_time,
        "ore_hauled": ore_hauled,
        "waste_hauled": waste_hauled,
    }


def get_travel_time_sec(truck: Truck, is_loaded: bool) -> float:
    """Calculates exact baseline travel duration in seconds based on mine layout."""
    level = truck.target_level or 4
    muck_type = truck.payload_type.lower()

    v_surf = truck.get_speed_mps("surface")
    v_dec = truck.get_speed_mps("decline")
    v_ramp = truck.get_speed_mps("ramp")
    v_lvl = truck.get_speed_mps("level")

    d_dec = 2100.0
    d_ramp = (level - 1) * 300.0
    d_lvl = 40.0 if muck_type == "ore" else 55.0
    d_surf = 300.0 if muck_type == "ore" else 440.0

    if not is_loaded:
        t_total_s = (d_surf / v_surf) + (d_dec / v_dec) + (d_ramp / v_ramp) + (d_lvl / v_lvl)
    else:
        t_total_s = (d_lvl / v_lvl) + (d_ramp / v_ramp) + (d_dec / v_dec) + (d_surf / v_surf)

    return t_total_s


def step_simulation(state: dict, dt_step: float):
    """Single simulation step execution."""
    timers = state["timers"]
    trucks = state["trucks"]
    fuel_pct = state["fuel_pct"]
    dispatch = state["dispatch"]
    loading_bays = state["loading_bays"]
    rom_dump_bay = state["rom_dump_bay"]
    waste_dump_bay = state["waste_dump_bay"]
    dump_bays = state["dump_bays"]

    active_mask = timers > 0
    timers[active_mask] = np.maximum(0.0, timers[active_mask] - dt_step)

    for i, truck in enumerate(trucks):
        if truck.state in (TruckState.TRAVEL_EMPTY, TruckState.TRAVEL_LOADED):
            fuel_pct[i] -= truck.fuel_burn_rate_pct_per_sec * dt_step
            truck.fuel_level_pct = fuel_pct[i]

    for i, truck in enumerate(trucks):
        if truck.state == TruckState.PARKED:
            dispatch.assign_next_destination(truck)
            if truck.state == TruckState.TRAVEL_EMPTY:
                t_travel = get_travel_time_sec(truck, is_loaded=False)
                timers[i] = t_travel

        elif truck.state == TruckState.TRAVEL_EMPTY:
            if timers[i] <= 0.0:
                truck.state = TruckState.WAITING_LOAD

        if truck.state == TruckState.WAITING_LOAD:
            target_bay = next(
                (b for b in loading_bays if b.bay_id == truck.target_bay_id), None
            )
            if target_bay and target_bay.start_loading(truck):
                timers[i] = target_bay.total_load_duration_sec

        elif truck.state == TruckState.TRAVEL_LOADED:
            if timers[i] <= 0.0:
                truck.state = TruckState.WAITING_DUMP

        if truck.state == TruckState.WAITING_DUMP:
            target_dump = rom_dump_bay if truck.payload_type == "ORE" else waste_dump_bay
            if target_dump.start_dumping(truck):
                timers[i] = target_dump.dump_time_remaining

        elif truck.state == TruckState.REFUELING:
            fuel_pct[i] = 100.0
            truck.fuel_level_pct = 100.0
            truck.state = TruckState.PARKED

    state["decline"].update_continuous_step(dt_step)
    for ramp in state["ramp_levels"]:
        ramp.update_continuous_step(dt_step)

    for bay in loading_bays:
        bay.update_continuous_step(dt_step)
        if bay.active_truck is not None and bay.active_truck.state == TruckState.TRAVEL_LOADED:
            idx = trucks.index(bay.active_truck)
            if timers[idx] <= 0.0:
                t_travel = get_travel_time_sec(bay.active_truck, is_loaded=True)
                timers[idx] = t_travel

    for dump_bay in dump_bays:
        dump_bay.update_continuous_step(dt_step)

    state["ore_hauled"].value = rom_dump_bay.dumped_total.value
    state["waste_hauled"].value = waste_dump_bay.dumped_total.value
    state["global_time"].rate = 1.0
    state["global_time"]._update(dt_step / 86400.0)


def run_haulage_simulation(
    state: dict, total_days: float = 365.0, dt: float = 300.0, show_progress: bool = False
) -> float:
    """Runs event-driven DRS integration over specified days (365 calendar days baseline)."""
    total_seconds = float(total_days * 24.0 * 3600.0)
    current_sec = 0.0
    step_dt = float(dt)

    pbar = tqdm(
        total=int(total_seconds),
        desc=f"Simulating {total_days:.0f} days",
        disable=not show_progress,
    )

    while current_sec < total_seconds:
        time_in_day = current_sec % 86400.0
        is_shift_gap = (10.5 * 3600.0 <= time_in_day < 12.0 * 3600.0) or (
            22.5 * 3600.0 <= time_in_day < 24.0 * 3600.0
        )

        if is_shift_gap:
            state["global_time"]._update(step_dt / 86400.0)
            current_sec += step_dt
            pbar.update(int(step_dt))
            continue

        step_simulation(state, step_dt)

        current_sec += step_dt
        pbar.update(int(step_dt))

    pbar.close()
    total_hauled = (
        state["rom_dump_bay"].dumped_total.value
        + state["waste_dump_bay"].dumped_total.value
    )
    return total_hauled / total_days


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

    state = build_shelswell_simulation(
        num_trucks=trucks,
        num_operators=operators,
        mechanical_availability=availability,
    )
    return run_haulage_simulation(state, total_days=365.0, dt=300.0, show_progress=False)


def _run_task(args):
    """Top-level helper function for multiprocessing worker execution."""
    trucks, operators, avail = args
    return (trucks, operators, avail, run_simulation(trucks, operators, avail))


def generate_figure_2():
    """Replicates Figure 2 from Shelswell (2017): Productivity vs Fleet Size without operator constraints."""
    print("Generating Figure 2 (Productivity vs Fleet Size without operator constraints)...")
    availabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    truck_sizes = list(range(3, 11))

    base_prod = run_simulation(10, 10, 1.0)
    print(f"Base Hybrid DRS productivity (10 trucks, 10 ops, 100% avail): {base_prod:.2f} t/d")

    tasks = [(t, t, a) for a in availabilities for t in truck_sizes]

    results_map = {}
    with ProcessPoolExecutor() as executor:
        for t, o, a, prod in tqdm(executor.map(_run_task, tasks), total=len(tasks), desc="Figure 2 Parallel Sweep"):
            results_map[(t, o, a)] = prod

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
        prod_list = [results_map[(t, t, avail)] / base_prod for t in truck_sizes]
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

    tasks = [(t, min(o, t), a) for a in availabilities for t in truck_counts for o in operator_counts]

    results_map = {}
    with ProcessPoolExecutor() as executor:
        for t, o, a, prod in tqdm(executor.map(_run_task, tasks), total=len(tasks), desc="Figures 3-8 Parallel Sweep"):
            results_map[(t, o, a)] = prod

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
    markers = {
        3: "o",
        4: "s",
        5: "^",
        6: "v",
        7: "D",
        8: "P",
        9: "X",
        10: "*",
    }

    for avail in availabilities:
        base_prod = results_map[(10, 10, avail)]

        plt.figure(figsize=(10, 6))
        # Plot in reverse order (10 trucks down to 3 trucks) so smaller fleet curves remain visible on top when overlapping
        for trucks in reversed(truck_counts):
            prod_list = [results_map[(trucks, min(ops, trucks), avail)] / base_prod for ops in operator_counts]
            plt.plot(
                operator_counts,
                prod_list,
                marker=markers[trucks],
                markersize=6,
                label=f"{trucks} trucks",
                color=colors[trucks],
                zorder=10 - trucks,  # higher zorder for smaller fleets to prevent hiding under 10 trucks
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

    print("Saved plots/shelswell_fig3.png through plots/shelswell_fig8.png")


if __name__ == "__main__":
    generate_figure_2()
    generate_figures_3_to_8()
    print("Replication of all figures complete with High-Performance Hybrid DRS Architecture!")

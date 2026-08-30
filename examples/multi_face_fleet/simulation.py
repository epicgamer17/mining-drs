"""
Two faces	Simplest multi-face extension. With 2 faces, allocation can be expressed as a single ratio. N-faces would require a linear program per mode.
Continuous fleet	Avoids event-based truck scheduling complexity. Continuous flows match the mill's steady-state assumption and the stockpile's continuous-time ODE.
Face allocation = fixed means per mode	Using face generator means (not current parcels) gives stable, campaign-long ratios. Dynamic per-timestep solves would jitter with parcel changes.
Surging = extreme allocation	Surging must produce an OFF-target blend to drain the stockpile. Using the base-mode allocation creates a degenerate equilibrium (extraction = milling).
50/50 face composition	Face means chosen so a 50/50 split matches the single-face's effective 70% ore1.
"""

import sys
import os

# Ensure the root directory is on the path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from drs import DRSEngine, Telemetry
from drs_mining.components import (
    MetallurgicalPlant,
    ContinuousFleetLogistics,
    MineFace,
    OperatingModeController,
    FleetController,
    StochasticFaciesGenerator,
    Stockpile,
)
from drs_mining.config import MILL_MODES
from drs_mining.components.plot import (
    plot_multi_face_dashboard,
    prepare_history,
    print_deficit_by_mode,
    print_transition_log,
)


def build_multi_face_network(
    mean_ore_fraction: float = 0.30,
    std_dev_ore_fraction: float = 0.05,
    target_ore_stock_level: float = 60000.0,
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    critical_ore2_level: float = 20400.0,
    duration_of_production_campaigns: float = 34.0,
    duration_of_shutdowns: float = 1.0,
    duration_of_contingency_segments: float = 1.0,
    prob_new_facies: float = 0.3,
    variation_same_facies: float = 0.01,
    replication_length: float = float("inf"),
    ore1_capacity: float = float("inf"),
    ore2_capacity: float = float("inf"),
    plant_max_rate: float = float("inf"),
    mode_a_ore1_milling_rate: float = 3600.0,
    mode_a_ore2_milling_rate: float = 2400.0,
    mode_a_contingency_ore1_milling_rate: float = 3900.0,
    mode_b_ore1_milling_rate: float = 4600.0,
    mode_b_ore2_milling_rate: float = 800.0,
    mode_b_contingency_ore2_milling_rate: float = 2500.0,
    fleet_shift_duration: float = 0.5,
    total_lhd_count: float = 3.0,
    total_truck_count: float = 10.0,
    max_lhds_per_face: float = 2.0,
    max_trucks_per_face: float = 6.0,
    face_haul_distance=(1.5, 2.2),
    face_accessibility_fraction=(0.93, 0.91),
    truck_velocity: float = 15.0,
    loader_cycle_time_hours: float = 0.0833,
    truck_dump_time_hours: float = 0.033,
    traffic_delay_per_truck_hours: float = 0.015,
    fleet_mechanical_availability: float = 0.85,
    loader_payload_tonnes: float = 15.0,
    truck_payload_tonnes: float = 30.0,
    development_rate_per_extra_truck: float = 50.0,
    enable_telemetry: bool = False,
) -> tuple:
    """Construct the two-face fleet network flat: every entity is built here.

    Returns ``(faces, fleet, plant, mode_controller, fleet_controller, ore1_stock, ore2_stock)``.
    """
    gen1 = StochasticFaciesGenerator(
        mean_fraction=0.30,
        std_dev=0.075,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
    )
    face1 = MineFace(
        name="mine_face_1",
        face_id=1,
        generator=gen1,
        min_ore_mass=30000.0,
        max_ore_mass=50000.0,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        mean_ore_fraction=0.30,
        std_dev_ore_fraction=0.075,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
        initial_parcel_mass=40000.0,
    )

    gen2 = StochasticFaciesGenerator(
        mean_fraction=0.35,
        std_dev=0.025,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
    )
    face2 = MineFace(
        name="mine_face_2",
        face_id=2,
        generator=gen2,
        min_ore_mass=30000.0,
        max_ore_mass=50000.0,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        mean_ore_fraction=0.35,
        std_dev_ore_fraction=0.025,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
        initial_parcel_mass=40000.0,
    )

    fleet = ContinuousFleetLogistics()

    initial_mass1 = (1 - mean_ore_fraction) * target_ore_stock_level
    ore1_stock = Stockpile(
        name="Ore1Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=initial_mass1,
        initial_attributes={
            "contained_ore_fraction_mass": initial_mass1 * mean_ore_fraction
        },
        capacity=ore1_capacity,
        attr_inflow=1.0,
    )
    initial_mass2 = mean_ore_fraction * target_ore_stock_level
    ore2_stock = Stockpile(
        name="Ore2Stock",
        expected_attributes=["contained_ore_fraction_mass"],
        initial_mass=initial_mass2,
        initial_attributes={
            "contained_ore_fraction_mass": initial_mass2 * mean_ore_fraction
        },
        capacity=ore2_capacity,
        attr_inflow=0.0,
    )

    plant = MetallurgicalPlant(
        stockpiles=[ore1_stock, ore2_stock],
        max_rate=plant_max_rate,
        target_ore_stock_level=target_ore_stock_level,
        duration_of_contingency_segments=duration_of_contingency_segments,
        mode_a_ore1_milling_rate=mode_a_ore1_milling_rate,
        mode_a_ore2_milling_rate=mode_a_ore2_milling_rate,
        mode_a_contingency_ore1_milling_rate=mode_a_contingency_ore1_milling_rate,
        mode_b_ore1_milling_rate=mode_b_ore1_milling_rate,
        mode_b_ore2_milling_rate=mode_b_ore2_milling_rate,
        mode_b_contingency_ore2_milling_rate=mode_b_contingency_ore2_milling_rate,
    )

    mode_controller = OperatingModeController(
        duration_of_production_campaigns=duration_of_production_campaigns,
        duration_of_shutdowns=duration_of_shutdowns,
        critical_ore2_level=critical_ore2_level,
        target_ore_stock_level=target_ore_stock_level,
        total_ore_to_extract=total_ore_to_extract,
    )

    mode_rates = {
        "MODE_A": (mode_a_ore1_milling_rate, mode_a_ore2_milling_rate),
        "MODE_A_CONTINGENCY": (mode_a_contingency_ore1_milling_rate, 0.0),
        "MODE_B": (mode_b_ore1_milling_rate, mode_b_ore2_milling_rate),
        "MODE_B_CONTINGENCY": (0.0, mode_b_contingency_ore2_milling_rate),
    }

    fleet_controller = FleetController(
        faces=[face1, face2],
        fleet_shift_duration=fleet_shift_duration,
        total_lhd_count=total_lhd_count,
        total_truck_count=total_truck_count,
        max_lhds_per_face=max_lhds_per_face,
        max_trucks_per_face=max_trucks_per_face,
        face_haul_distance=face_haul_distance,
        face_accessibility_fraction=face_accessibility_fraction,
        truck_velocity=truck_velocity,
        loader_cycle_time_hours=loader_cycle_time_hours,
        truck_dump_time_hours=truck_dump_time_hours,
        traffic_delay_per_truck_hours=traffic_delay_per_truck_hours,
        fleet_mechanical_availability=fleet_mechanical_availability,
        loader_payload_tonnes=loader_payload_tonnes,
        truck_payload_tonnes=truck_payload_tonnes,
        development_rate_per_extra_truck=development_rate_per_extra_truck,
        mode_allocations={},
        mode_rates=mode_rates,
    )

    return [face1, face2], fleet, plant, mode_controller, fleet_controller, ore1_stock, ore2_stock


def _register_and_policy(engine, network):
    """Register the stateful leaves and wire the inline policy onto the engine."""
    faces, fleet, plant, mode_ctrl, fleet_ctrl, ore1_stock, ore2_stock = network
    engine.register(*faces, fleet, plant, mode_ctrl, fleet_ctrl, ore1_stock, ore2_stock)

    @engine.on_step
    def manage_blending(t: float):
        # 1. Update operating mode based on campaign timers & stockpile levels
        mode = mode_ctrl.update(ore2_stock.level)

        # 2. Compute the aggregate target flow rates and plant draw rates
        plant_draw, mine_target = plant.get_target_rates(
            mode,
            ore1_level=ore1_stock.level,
            ore2_level=ore2_stock.level,
            stockpile2_routing_fraction=fleet.stockpile2_routing_fraction.value,
        )

        # 3. Schedule fleet shifts and compute achievable face rates
        face_rates = fleet_ctrl.allocate(mine_target, plant.active_operating_mode.value)
        for face, rate in zip(faces, face_rates):
            face.target_rate = rate

        # 4. Route mined parcels into the stockpiles
        ore1_in, ore2_in = fleet.route(sources=faces)

        # 5. Feed stockpiles and draw into plant
        out1 = ore1_stock.feed_and_draw(ore1_in, plant_draw.ore1)
        out2 = ore2_stock.feed_and_draw(ore2_in, plant_draw.ore2)
        plant.process(out1 + out2)


def build_telemetry(engine, network):
    """Explicit telemetry channels for the capacity-policy analysis."""
    telemetry = Telemetry(model=engine)
    faces, fleet, plant, mode_ctrl, fleet_ctrl, ore1_stock, ore2_stock = network
    face1, face2 = faces

    def _face1_alloc(t, m, s, _):
        return fleet_ctrl.face_target_rates[0].value / max(
            1e-12, plant.target_mine_mass_rate.value
        )

    def _face2_alloc(t, m, s, _):
        return fleet_ctrl.face_target_rates[1].value / max(
            1e-12, plant.target_mine_mass_rate.value
        )

    def _ore2_ratio(t, m, s, _):
        return ore2_stock.level / max(
            1e-6,
            ore1_stock.level + ore2_stock.level,
        )

    def _mixed_achieved(t, m, s, _):
        return sum(r.value for r in fleet_ctrl.face_achieved_extraction_rates)

    def _mixed_target(t, m, s, _):
        return sum(r.value for r in fleet_ctrl.face_target_extraction_rates)

    def _mixed_real(t, m, s, _):
        return sum(r.value for r in fleet_ctrl.face_real_extraction_rates)

    metrics = {
        "face1_alloc": _face1_alloc,
        "face2_alloc": _face2_alloc,
        "face1_target_extraction_rate": lambda t, m, s, _: fleet_ctrl.face_target_extraction_rates[0].value,
        "face1_real_extraction_rate": lambda t, m, s, _: fleet_ctrl.face_real_extraction_rates[0].value,
        "face1_achieved_extraction_rate": lambda t, m, s, _: fleet_ctrl.face_achieved_extraction_rates[0].value,
        "face1_operational_downtime_fraction": lambda t, m, s, _: fleet_ctrl.face_operational_downtime_fractions[0].value,
        "face2_target_extraction_rate": lambda t, m, s, _: fleet_ctrl.face_target_extraction_rates[1].value,
        "face2_real_extraction_rate": lambda t, m, s, _: fleet_ctrl.face_real_extraction_rates[1].value,
        "face2_achieved_extraction_rate": lambda t, m, s, _: fleet_ctrl.face_achieved_extraction_rates[1].value,
        "face2_operational_downtime_fraction": lambda t, m, s, _: fleet_ctrl.face_operational_downtime_fractions[1].value,
        "fleet_shift_count": lambda t, m, s, _: fleet_ctrl.fleet_shift_count.value,
        "fleet_shift_timer": lambda t, m, s, _: fleet_ctrl.fleet_shift_timer.value,
        "face1_real_capacity": lambda t, m, s, _: fleet_ctrl.face_real_extraction_rates[0].value,
        "face1_target_rate": lambda t, m, s, _: fleet_ctrl.face_target_rates[0].value,
        "face1_match_factor": lambda t, m, s, _: fleet_ctrl.face_match_factors[0].value,
        "face1_truck_cycle_time_hours": lambda t, m, s, _: fleet_ctrl.face_truck_cycle_times[0].value,
        "face2_real_capacity": lambda t, m, s, _: fleet_ctrl.face_real_extraction_rates[1].value,
        "face2_target_rate": lambda t, m, s, _: fleet_ctrl.face_target_rates[1].value,
        "face2_match_factor": lambda t, m, s, _: fleet_ctrl.face_match_factors[1].value,
        "face2_truck_cycle_time_hours": lambda t, m, s, _: fleet_ctrl.face_truck_cycle_times[1].value,
        "total_unused_trucks": lambda t, m, s, _: fleet_ctrl.total_extra_trucks.value,
        "ore2_ratio": _ore2_ratio,
        "face1_extracted_mass": lambda t, m, s, _: face1.cumulative_extracted_mass.value,
        "face2_extracted_mass": lambda t, m, s, _: face2.cumulative_extracted_mass.value,
        "face1_parcel_mass": lambda t, m, s, _: face1.active_parcel_initial_mass.value,
        "face1_parcel_ratio": lambda t, m, s, _: face1.active_parcel_ore_fraction.value,
        "face2_parcel_mass": lambda t, m, s, _: face2.active_parcel_initial_mass.value,
        "face2_parcel_ratio": lambda t, m, s, _: face2.active_parcel_ore_fraction.value,
        "mixed_achieved_extraction_rate": _mixed_achieved,
        "mixed_target_extraction_rate": _mixed_target,
        "mixed_real_extraction_rate": _mixed_real,
        "mixed_ore1_fraction": lambda t, m, s, _: 1.0 - fleet.stockpile2_routing_fraction.value,
        "Campaign_Shutdown": lambda t, m, s, _: mode_ctrl.current_campaign_duration.value,
        "Contingency": lambda t, m, s, _: plant.current_contingency_duration.value,
    }
    for name, fn in metrics.items():
        telemetry.register_metric(name, fn)
    return telemetry


def print_statistics(plant, faces):
    """Print operating-mode time-shares and throughput for a multi-face network."""
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
            total_ore_processed = sum(f.net_extracted_mass for f in faces)
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


def evaluate_throughput(config_kwargs: dict = None, N: int = 1) -> tuple[float, float]:
    """
    Runs the simulation N times, extracting throughputs.
    Returns (mean_throughput, std_dev_throughput).
    """
    kwargs = config_kwargs or {}
    throughputs = []

    for idx in range(N):
        network = build_multi_face_network(**kwargs)
        faces, fleet, plant, mode_ctrl, fleet_ctrl, ore1_stock, ore2_stock = network

        engine = DRSEngine()
        _register_and_policy(engine, network)

        np.random.seed(idx)
        random.seed(idx)

        engine.run(until=kwargs.get("replication_length", float("inf")))

        active_time = plant.active_duration(engine.current_time)
        if active_time > 0:
            throughput = (
                faces[0].net_extracted_mass + faces[1].net_extracted_mass
            ) / active_time
            throughputs.append(throughput)

    if not throughputs:
        return 0.0, 0.0
    return float(np.mean(throughputs)), float(np.std(throughputs))


def plot_monte_carlo_throughput(N: int = 1, total_stockpile_level: float = 60000.0):
    sigmas = [5.0]
    results = []

    print(f"\n--- Running Monte Carlo Evaluation for Multi-Face Fleet (N={N}) ---")
    for sigma in sigmas:
        config_kwargs = dict(
            replication_length=99999.0,
            std_dev_ore_fraction=sigma / 100.0,
            target_ore_stock_level=total_stockpile_level,
            prob_new_facies=0.3,
        )
        mean, std = evaluate_throughput(config_kwargs, N)
        results.append((sigma, mean, std))
        print(f"Sigma: {sigma}%, Mean Throughput: {mean:.2f}, Std Dev: {std:.2f}")

    # Plot results to match Figure 5 from the paper
    means = [r[1] for r in results]
    stds = [r[2] for r in results]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sigmas,
        means,
        yerr=stds,
        fmt="-o",
        capsize=5,
        capthick=2,
        ecolor="black",
        markerfacecolor="blue",
        markeredgecolor="blue",
        color="gray",
    )

    plt.title(
        f"Expected Simulated Throughput by Geological Uncertainty (Multi-Face, N={N})",
        fontsize=14,
    )
    plt.xlabel("Sigma geo (%)", fontsize=12)
    plt.ylabel("Mean Campaign Throughput (t/d)", fontsize=12)
    plt.ylim(5500, 6000)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(
        "plots/Monte_Carlo_Throughput_Fig5_MultiFace.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved 'plots/Monte_Carlo_Throughput_Fig5_MultiFace.png'.\n")


def _apply_equal_allocation(fleet_controller):
    n = len(fleet_controller.faces)
    fracs = [1.0 / n] * n
    # Force the physical fleet dispatch to always be evenly split
    for k in fleet_controller._mode_allocations.keys():
        fleet_controller._mode_allocations[k] = fracs

    orig_real = fleet_controller._face_real_extraction_rate

    def _unconstrained_real(face_index, target_extraction_rate):
        orig_real(face_index, target_extraction_rate)
        return target_extraction_rate

    fleet_controller._face_real_extraction_rate = _unconstrained_real


def _run_capacity_case(
    label: str,
    config_kwargs: dict = None,
    max_time: float = 60.0,
    equal_allocation: bool = False,
    np_seed: int = 42,
    random_seed: int = 11,
):
    kwargs = config_kwargs or {}
    network = build_multi_face_network(**kwargs)
    faces, fleet, plant, mode_ctrl, fleet_ctrl, ore1_stock, ore2_stock = network
    if equal_allocation:
        _apply_equal_allocation(fleet_ctrl)

    mode_ctrl.active_campaign_mode.value = MILL_MODES["MODE_A"]

    engine = DRSEngine()
    _register_and_policy(engine, network)

    telemetry = build_telemetry(engine, network)
    engine.attach_telemetry(telemetry)
    engine.run(until=max_time)

    df = telemetry.to_dataframe()
    df["scenario"] = label
    df["active_operating_mode_name"] = df["active_operating_mode"].apply(
        lambda x: x.name if x else "None"
    )
    df["total_system_ore_mass"] = df["Ore1Stock_mass"] + df["Ore2Stock_mass"]
    df["total_target_extraction_rate"] = (

        df["face1_target_extraction_rate"] + df["face2_target_extraction_rate"]
    )
    df["total_real_extraction_rate"] = (
        df["face1_real_extraction_rate"] + df["face2_real_extraction_rate"]
    )
    df["total_achieved_extraction_rate"] = (
        df["face1_achieved_extraction_rate"] + df["face2_achieved_extraction_rate"]
    )
    df["capacity_gap_rate"] = (
        df["total_target_extraction_rate"] - df["total_achieved_extraction_rate"]
    ).clip(lower=0.0)
    df["capacity_utilization"] = np.where(
        df["total_target_extraction_rate"] > 1e-12,
        df["total_achieved_extraction_rate"] / df["total_target_extraction_rate"],
        0.0,
    )

    dt = df["time"].diff().shift(-1).fillna(0.0)
    capacity_lost_mass = float((df["capacity_gap_rate"] * dt).sum())
    active_utilization = df.loc[
        df["total_target_extraction_rate"] > 1e-12, "capacity_utilization"
    ]
    summary = {
        "scenario": label,
        "final_time": float(df["time"].iloc[-1]),
        "fleet_shift_count": float(df["fleet_shift_count"].max()),
        "mean_total_target_extraction_rate": float(
            df["total_target_extraction_rate"].mean()
        ),
        "mean_total_real_extraction_rate": float(
            df["total_real_extraction_rate"].mean()
        ),
        "mean_total_achieved_extraction_rate": float(
            df["total_achieved_extraction_rate"].mean()
        ),
        "mean_face1_operational_downtime_fraction": float(
            df["face1_operational_downtime_fraction"].mean()
        ),
        "mean_face2_operational_downtime_fraction": float(
            df["face2_operational_downtime_fraction"].mean()
        ),
        "mean_capacity_utilization": float(active_utilization.mean()),
        "max_capacity_gap_rate": float(df["capacity_gap_rate"].max()),
        "capacity_lost_mass": capacity_lost_mass,
        "final_ore1_stock": float(df["Ore1Stock_mass"].iloc[-1]),
        "final_ore2_stock": float(df["Ore2Stock_mass"].iloc[-1]),
        "min_ore2_stock": float(df["Ore2Stock_mass"].min()),
    }
    return df, summary


def run_capacity_comparison(
    base_kwargs: dict = None,
    max_time: float = 60.0,
):
    import pandas as pd
    from drs_mining.components.plot import plot_ore_with_modes

    # Both configurations are physically identical, we only change the control strategy
    cases = [
        ("Dynamic Fleet Allocation", False),
        ("Equal Fleet Allocation", True),
    ]

    frames = []
    summaries = []
    for label, is_equal in cases:
        df, summary = _run_capacity_case(
            label, base_kwargs, max_time=max_time, equal_allocation=is_equal
        )
        frames.append(df)
        summaries.append(summary)
        print(
            f"{label}: mean actual rate={summary['mean_total_achieved_extraction_rate']:.1f} t/d, "
            f"capacity lost={summary['capacity_lost_mass']:.1f} t, "
            f"min Ore2={summary['min_ore2_stock']:.1f} t"
        )

    combined = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    selected_columns = [
        "scenario",
        "time",
        "active_operating_mode_name",
        "fleet_shift_count",
        "fleet_shift_timer",
        "face1_target_extraction_rate",
        "face1_real_extraction_rate",
        "face1_achieved_extraction_rate",
        "face1_operational_downtime_fraction",
        "face2_target_extraction_rate",
        "face2_real_extraction_rate",
        "face2_achieved_extraction_rate",
        "face2_operational_downtime_fraction",
        "total_target_extraction_rate",
        "total_real_extraction_rate",
        "total_achieved_extraction_rate",
        "capacity_gap_rate",
        "capacity_utilization",
        "Ore1Stock_mass",
        "Ore2Stock_mass",
        "total_system_ore_mass",
        "mixed_ore1_fraction",
    ]
    combined[selected_columns].to_csv(
        "capacity_policy_comparison.csv", index=False, encoding="utf-8"
    )
    summary_df.to_csv(
        "capacity_policy_comparison_summary.csv", index=False, encoding="utf-8"
    )

    mode_order = [
        "MODE_A",
        "MODE_B",
        "MODE_A_CONTINGENCY",
        "MODE_B_CONTINGENCY",
        "MODE_A_MINE_SURGING",
        "MODE_B_MINE_SURGING",
        "SHUTDOWN",
    ]

    # Side-by-side plot comparing stockpiles, face rates, and downtime
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)

    for col_idx, (label, _) in enumerate(cases):
        sub = combined[combined["scenario"] == label]
        ax_top = axes[0, col_idx]
        ax_mid = axes[1, col_idx]
        ax_bot = axes[2, col_idx]

        plot_ore_with_modes(
            sub,
            time_col="time",
            ore_cols=["Ore1Stock_mass", "Ore2Stock_mass"],
            mode_col="active_operating_mode_name",
            title=f"{label} - Stockpiles",
            ax=ax_top,
        )


        ax_mid.plot(
            sub["time"],
            sub["face1_achieved_extraction_rate"],
            label="Face 1 Achieved Rate",
            color="navy",
        )
        ax_mid.plot(
            sub["time"],
            sub["face2_achieved_extraction_rate"],
            label="Face 2 Achieved Rate",
            color="darkgreen",
        )
        ax_mid.plot(
            sub["time"],
            sub["total_target_extraction_rate"],
            label="Aggregate Target",
            color="crimson",
            linestyle="--",
        )
        ax_mid.set_ylabel("Extraction Rate (t/d)")
        ax_mid.set_title(f"{label} - Face Throughputs")
        ax_mid.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_mid.legend(loc="upper right")

        ax_bot.plot(
            sub["time"],
            sub["face1_operational_downtime_fraction"],
            label="Face 1 Capacity Gap",
            color="navy",
            linestyle=":",
        )
        ax_bot.plot(
            sub["time"],
            sub["face2_operational_downtime_fraction"],
            label="Face 2 Capacity Gap",
            color="darkgreen",
            linestyle=":",
        )
        ax_bot.plot(
            sub["time"],
            sub["capacity_gap_rate"],
            label="Aggregate Capacity Gap (t/d)",
            color="red",
        )
        ax_bot.set_ylabel("Bottleneck / Gap")
        ax_bot.set_xlabel("Simulation Time (Days)")
        ax_bot.set_title(f"{label} - Bottleneck Losses")
        ax_bot.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_bot.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/Capacity_Policy_Comparison.png", dpi=300)
    plt.close()

    print("Saved capacity_policy_comparison.csv")
    print("Saved capacity_policy_comparison_summary.csv")
    print("Saved Capacity_Policy_Comparison.png")
    return combined, summary_df


def run_and_analyze(config, equal_allocation=False, name="Dynamic Fleet Allocation"):
    """Run the multi-face simulation and produce full diagnostics dashboard."""
    np.random.seed(42)
    random.seed(11)
    network = build_multi_face_network(**config)
    faces, fleet, plant, mode_ctrl, fleet_ctrl, ore1_stock, ore2_stock = network
    if equal_allocation:
        _apply_equal_allocation(fleet_ctrl)

    mode_ctrl.active_campaign_mode.value = MILL_MODES["MODE_A"]

    engine = DRSEngine()
    _register_and_policy(engine, network)

    telemetry = build_telemetry(engine, network)
    engine.attach_telemetry(telemetry)
    result = engine.run(until=config.get("replication_length", 99999.0))
    print_statistics(plant, faces)

    df = prepare_history(result.history)
    print_transition_log(
        df,
        critical_ore2_level=config.get("critical_ore2_level", 20400.0),
        target_ore_stock_level=config.get("target_ore_stock_level", 60000.0),
        label=name,
    )

    print_deficit_by_mode(
        df,
        extraction_cols=["face1_extracted_mass", "face2_extracted_mass"],
        ideal_rate=6000.0,
        heading=f"Cumulative Lost Production (Deficit) by Mode ({name})",
    )

    plot_multi_face_dashboard(df, name=name)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--compare_capacity_cases", action="store_true")
    parser.add_argument("--comparison_max_time", type=float, default=60.0)
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(11)
    config_kwargs = dict(
        replication_length=99999.0,
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
        prob_new_facies=0.3,
    )

    if args.compare_capacity_cases:
        run_capacity_comparison(config_kwargs, max_time=args.comparison_max_time)
        raise SystemExit(0)

    df_managed = run_and_analyze(
        config_kwargs, equal_allocation=False, name="Dynamic Fleet Allocation"
    )
    df_equal = run_and_analyze(
        config_kwargs, equal_allocation=True, name="Equal Fleet Allocation"
    )

    # Print summary comparison
    print("\n" + "=" * 72)
    print("COMPARISON SUMMARY: Dynamic Fleet Allocation vs Equal Fleet Allocation")
    print("=" * 72)

    for label, df in [
        ("Dynamic Fleet Allocation", df_managed),
        ("Equal Fleet Allocation", df_equal),
    ]:
        dt = df["time"].diff().fillna(0)
        final_ore1 = df["Ore1Stock_mass"].iloc[-1]
        final_ore2 = df["Ore2Stock_mass"].iloc[-1]
        total_extracted = (
            df["face1_extracted_mass"].iloc[-1] + df["face2_extracted_mass"].iloc[-1]
        )
        print(f"\n--- {label} ---")
        print(f"  Total extracted: {total_extracted:,.0f} tons")
        print(
            f"  Final Ore1 stock: {final_ore1:,.0f} t | Final Ore2 stock: {final_ore2:,.0f} t"
        )
        print(f"  Mode breakdown:")
        for mode_name in df["active_operating_mode_name"].unique():
            mode_dt = dt[df["active_operating_mode_name"] == mode_name].sum()
            pct = (mode_dt / dt.sum() * 100) if dt.sum() > 0 else 0
            print(f"    {str(mode_name):35s}: {mode_dt:8.2f} days ({pct:5.1f}%)")
    print("=" * 72 + "\n")

    # Recreate Figure 5 from paper
    plot_monte_carlo_throughput(
        N=args.N, total_stockpile_level=args.total_stockpile_level
    )

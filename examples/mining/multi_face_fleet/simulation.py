"""
Two faces	Simplest multi-face extension. With 2 faces, allocation can be expressed as a single ratio. N-faces would require a linear program per mode.
Continuous fleet	Avoids event-based truck scheduling complexity. Continuous flows match the mill's steady-state assumption and the stockpile's continuous-time ODE.
Face allocation = fixed means per mode	Using face generator means (not current parcels) gives stable, campaign-long ratios. Dynamic per-timestep solves would jitter with parcel changes.
Surging = extreme allocation	Surging must produce an OFF-target blend to drain the stockpile. Using the base-mode allocation creates a degenerate equilibrium (extraction = milling).
50/50 face composition	Face means chosen so a 50/50 split matches the single-face's effective 70% ore1.
"""

import sys
import os

# Ensure the root directory is on the path so we can import 'examples.mining'
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import drs
from drs import DRSEngine, Telemetry
from drs_mining.components import (
    ConcentratorPlant,
    ContinuousFleetLogistics,
    ContinuousMineFace,
    MultiFaceConcentratorController,
    StochasticFaciesGenerator,
    Stockpile,
)
from drs_mining.components.modes import MODES


# ============================================================
# Inline control policy
# ------------------------------------------------------------
# A verbatim copy of the multi-face path of ``drs_mining.control``
# adapted to the plain local entities built below. Operating-mode
# bookkeeping lives on the controller (``step_mode``), extraction
# mechanics on the mine faces (``set_extraction_rate``), and
# stockpile balancing on the stockpiles (``set_inout``); this
# policy owns the fleet shift allocation, target rates, routing,
# and calling order.
# ============================================================

_RATE_MAP = {
    "MODE_A": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
    "MODE_A_CONTINGENCY": ("mode_a_contingency_ore1_milling_rate", None),
    "MODE_A_MINE_SURGING": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
    "MODE_B": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
    "MODE_B_CONTINGENCY": (None, "mode_b_contingency_ore2_milling_rate"),
    "MODE_B_MINE_SURGING": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
    "SHUTDOWN": (None, None),
}

DEFAULT_RATES = {
    "mode_a_ore1_milling_rate": 3600.0,
    "mode_a_ore2_milling_rate": 2400.0,
    "mode_a_contingency_ore1_milling_rate": 3900.0,
    "mode_b_ore1_milling_rate": 4600.0,
    "mode_b_ore2_milling_rate": 800.0,
    "mode_b_contingency_ore2_milling_rate": 2500.0,
}


def multi_face_step_policy(
    time,
    fleet,
    plant,
    controller,
    ore1_stock,
    ore2_stock,
    global_time,
):
    """Top-level multi-face fleet policy invoked by the engine once per step."""
    global_time.rate = 1.0
    ctrl = controller
    ctrl.total_extra_trucks.value = 0.0

    mode_name = ctrl.step_mode(ore1_stock, ore2_stock)

    ctrl.fleet_shift_timer.rate = 1.0
    ctrl.fleet_shift_timer.upper_threshold = ctrl.fleet_shift_duration

    shift_due = ctrl.fleet_shift_timer.value >= ctrl.fleet_shift_duration - 1e-6
    mode_changed = mode_name != ctrl.current_shift_mode_name
    if ctrl.current_shift_allocations is None or shift_due or mode_changed:
        ctrl.fleet_shift_timer.reset()
        ctrl._reallocate_fleet_for_shift()

    _target_rates(controller, ore1_stock, ore2_stock, fleet, None)

    fracs = ctrl.current_shift_allocations
    for i, face in enumerate(ctrl.faces):
        if fracs is not None and i < len(fracs):
            target = ctrl.target_mine_mass_rate.value * fracs[i]
            real = ctrl._face_real_extraction_rate(i, target)
            ctrl.face_target_extraction_rates[i].value = target
            ctrl.face_real_extraction_rates[i].value = real
            ctrl.face_achieved_extraction_rates[i].value = min(target, real)
            face.set_extraction_rate(ctrl.face_achieved_extraction_rates[i].value)
        else:
            ctrl.face_target_extraction_rates[i].value = 0.0
            ctrl.face_real_extraction_rates[i].value = 0.0
            ctrl.face_achieved_extraction_rates[i].value = 0.0
            ctrl.face_operational_downtime_fractions[i].value = 0.0
            face.set_extraction_rate(0.0)

    ctrl.cumulative_mine_development.rate = (
        ctrl.total_extra_trucks.value * ctrl.development_rate_per_extra_truck
    )

    ore1_rate, ore2_rate = _route(ctrl.faces, fleet)
    _balance_stockpiles(
        controller, plant, ore1_stock, ore2_stock, ore1_rate, ore2_rate
    )

    ctrl.total_system_ore_mass.rate = (
        ore1_stock.current_mass.rate + ore2_stock.current_mass.rate
    )


def _route(mining_sources, fleet):
    ore1 = ore2 = total = 0.0
    for src in mining_sources:
        rate = src.actual_rate
        ore2_frac = src._get_current_attr_value()
        ore2 += rate * ore2_frac
        ore1 += rate * (1.0 - ore2_frac)
        total += rate
    if total > 1e-6 and fleet is not None:
        fleet.stockpile2_routing_fraction.value = ore2 / total
    return ore1, ore2


def _balance_stockpiles(
    controller, plant, ore1_stock, ore2_stock, ore1_rate, ore2_rate
):
    ctrl = controller
    out1 = ore1_stock.set_inout(
        ore1_rate, ctrl.target_stock1_outflow_rate.value, attr_inflow=1.0
    )
    out2 = ore2_stock.set_inout(
        ore2_rate, ctrl.target_stock2_outflow_rate.value, attr_inflow=0.0
    )
    if plant is not None:
        plant.target_rate = out1 + out2
        plant.cumulative_milled_mass.rate = plant.actual_rate


def _target_rates(controller, ore1_stock, ore2_stock, fleet, mine):
    ctrl = controller
    name = ctrl.active_operating_mode.value.name

    ore1, ore2 = _read_rates(name, ctrl)

    if "_MINE_SURGING" in name:
        target_stock = getattr(ctrl, "target_ore_stock_level", 60000.0)
        ctrl.total_system_ore_mass.lower_threshold = target_stock
        p = fleet.stockpile2_routing_fraction.value if fleet is not None else 0.0
        if p <= 1e-4 and mine is not None and hasattr(mine, "_get_current_attr_value"):
            p = mine._get_current_attr_value()
        if name == "MODE_A_MINE_SURGING":
            effective_fraction = max(1.0 - p, 0.01)
            extraction = ore1 / effective_fraction
        else:
            effective_fraction = max(p, 0.01)
            extraction = ore2 / effective_fraction
    else:
        extraction = ore1 + ore2

    ctrl.target_mine_mass_rate.value = extraction
    ctrl.target_stock1_outflow_rate.value = ore1
    ctrl.target_stock2_outflow_rate.value = ore2


def _read_rates(name, obj):
    ore1_attr, ore2_attr = _RATE_MAP.get(name, (None, None))
    ore1 = (
        getattr(obj, ore1_attr, DEFAULT_RATES.get(ore1_attr, 0.0))
        if ore1_attr
        else 0.0
    )
    ore2 = (
        getattr(obj, ore2_attr, DEFAULT_RATES.get(ore2_attr, 0.0))
        if ore2_attr
        else 0.0
    )
    return ore1, ore2


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

    Returns ``(face1, face2, fleet, plant, controller, ore1_stock, ore2_stock,
    global_time)``. There is no scenario container; the engine registers the
    controller's state components plus the residual leaves below.

    Two geologically distinct mine faces (face1 low-grade ~15%, face2
    high-grade ~45%), the shared continuous truck/LHD fleet, the two initial
    stockpiles (Ore1/Ore2), the concentrator plant cap, and the multi-face
    controller with its explicit fleet-fit/development parameters.
    """
    gen1 = StochasticFaciesGenerator(
        mean_fraction=0.15,
        std_dev=0.075,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
    )
    gen2 = StochasticFaciesGenerator(
        mean_fraction=0.45,
        std_dev=0.025,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
    )
    face1 = ContinuousMineFace(
        face_id=1,
        generator=gen1,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
    )
    face2 = ContinuousMineFace(
        face_id=2,
        generator=gen2,
        total_ore_to_extract=total_ore_to_extract,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
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
    )

    plant = ConcentratorPlant(
        None, fleet, ore1_stock, ore2_stock, max_rate=plant_max_rate
    )
    controller = MultiFaceConcentratorController(
        faces=[face1, face2],
        fleet=fleet,
        plant=plant,
        target_ore_stock_level=target_ore_stock_level,
        critical_ore2_level=critical_ore2_level,
        duration_of_production_campaigns=duration_of_production_campaigns,
        duration_of_shutdowns=duration_of_shutdowns,
        duration_of_contingency_segments=duration_of_contingency_segments,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        total_ore_to_extract=total_ore_to_extract,
        mode_a_ore1_milling_rate=mode_a_ore1_milling_rate,
        mode_a_ore2_milling_rate=mode_a_ore2_milling_rate,
        mode_a_contingency_ore1_milling_rate=mode_a_contingency_ore1_milling_rate,
        mode_b_ore1_milling_rate=mode_b_ore1_milling_rate,
        mode_b_ore2_milling_rate=mode_b_ore2_milling_rate,
        mode_b_contingency_ore2_milling_rate=mode_b_contingency_ore2_milling_rate,
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
    )
    global_time = drs.Timer("GlobalTime", initial_value=0.0)

    return (
        face1,
        face2,
        fleet,
        plant,
        controller,
        ore1_stock,
        ore2_stock,
        global_time,
    )


def _register_and_policy(engine, fleet, plant, controller, ore1_stock, ore2_stock, global_time):
    """Register the stateful leaves and wire the inline policy onto the engine."""
    engine.register(
        *controller.state_components,
        plant,
        ore1_stock,
        ore2_stock,
        global_time,
    )

    @engine.on_step
    def _policy(time):
        multi_face_step_policy(
            time, fleet, plant, controller, ore1_stock, ore2_stock, global_time
        )


def build_telemetry(engine, controller, face1, face2, fleet, ore1_stock, ore2_stock):
    """Explicit telemetry channels for the capacity-policy analysis.

    Attaching ``Telemetry(model=engine)`` records every state variable of the
    registered leaves (``Ore1Stock_mass``, ``active_operating_mode``,
    ``total_system_ore_mass``, ...). These are the derived per-face rates and
    productivities the analysis joins into its dataframe.
    """
    telemetry = Telemetry(model=engine)

    def _face1_alloc(t, m, s, _):
        return controller.face_target_rates[0].value / max(
            1e-12, controller.target_mine_mass_rate.value
        )

    def _face2_alloc(t, m, s, _):
        return controller.face_target_rates[1].value / max(
            1e-12, controller.target_mine_mass_rate.value
        )

    def _ore2_ratio(t, m, s, _):
        return ore2_stock.current_mass.value / max(
            1e-6,
            ore1_stock.current_mass.value + ore2_stock.current_mass.value,
        )

    def _mixed_achieved(t, m, s, _):
        return sum(r.value for r in controller.face_achieved_extraction_rates)

    def _mixed_target(t, m, s, _):
        return sum(r.value for r in controller.face_target_extraction_rates)

    def _mixed_real(t, m, s, _):
        return sum(r.value for r in controller.face_real_extraction_rates)

    metrics = {
        "face1_alloc": _face1_alloc,
        "face2_alloc": _face2_alloc,
        "face1_target_extraction_rate": lambda t, m, s, _: controller.face_target_extraction_rates[0].value,
        "face1_real_extraction_rate": lambda t, m, s, _: controller.face_real_extraction_rates[0].value,
        "face1_achieved_extraction_rate": lambda t, m, s, _: controller.face_achieved_extraction_rates[0].value,
        "face1_operational_downtime_fraction": lambda t, m, s, _: controller.face_operational_downtime_fractions[0].value,
        "face2_target_extraction_rate": lambda t, m, s, _: controller.face_target_extraction_rates[1].value,
        "face2_real_extraction_rate": lambda t, m, s, _: controller.face_real_extraction_rates[1].value,
        "face2_achieved_extraction_rate": lambda t, m, s, _: controller.face_achieved_extraction_rates[1].value,
        "face2_operational_downtime_fraction": lambda t, m, s, _: controller.face_operational_downtime_fractions[1].value,
        "fleet_shift_count": lambda t, m, s, _: controller.fleet_shift_count.value,
        "fleet_shift_timer": lambda t, m, s, _: controller.fleet_shift_timer.value,
        "face1_real_capacity": lambda t, m, s, _: controller.face_real_extraction_rates[0].value,
        "face1_target_rate": lambda t, m, s, _: controller.face_target_rates[0].value,
        "face1_match_factor": lambda t, m, s, _: controller.face_match_factors[0].value,
        "face1_truck_cycle_time_hours": lambda t, m, s, _: controller.face_truck_cycle_times[0].value,
        "face2_real_capacity": lambda t, m, s, _: controller.face_real_extraction_rates[1].value,
        "face2_target_rate": lambda t, m, s, _: controller.face_target_rates[1].value,
        "face2_match_factor": lambda t, m, s, _: controller.face_match_factors[1].value,
        "face2_truck_cycle_time_hours": lambda t, m, s, _: controller.face_truck_cycle_times[1].value,
        "total_unused_trucks": lambda t, m, s, _: controller.total_extra_trucks.value,
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
        "Campaign_Shutdown": lambda t, m, s, _: controller.current_campaign_duration.value,
        "Contingency": lambda t, m, s, _: controller.current_contingency_duration.value,
    }
    for name, fn in metrics.items():
        telemetry.register_metric(name, fn)
    return telemetry


def print_statistics(controller, plant, faces):
    """Print operating-mode time-shares and throughput for a multi-face network."""
    print("\n--- Output Statistics ---")
    total_time = controller.total_duration

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
                f"{label}: {getattr(controller, attr).value / total_time:.4f}"
            )
    else:
        print("Total time is 0. Cannot calculate mode portions.")

    active_time = controller.active_duration(total_time)
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
        face1, face2, fleet, plant, controller, ore1_stock, ore2_stock, global_time = (
            build_multi_face_network(**kwargs)
        )

        engine = DRSEngine()
        _register_and_policy(
            engine, fleet, plant, controller, ore1_stock, ore2_stock, global_time
        )

        np.random.seed(idx)
        random.seed(idx)

        engine.run(until=kwargs.get("replication_length", float("inf")))

        active_time = controller.active_duration(engine.current_time)
        if active_time > 0:
            throughput = (
                face1.net_extracted_mass + face2.net_extracted_mass
            ) / active_time
            throughputs.append(throughput)

    if not throughputs:
        return 0.0, 0.0
    return float(np.mean(throughputs)), float(np.std(throughputs))


def plot_monte_carlo_throughput(N: int = 1, total_stockpile_level: float = 60000.0):
    sigmas = [5.0]
    results = []

    print(f"\n--- Running Monte Carlo Evaluation for Standard (N={N}) ---")
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
        f"Expected Simulated Throughput by Geological Uncertainty (Standard, N={N})",
        fontsize=14,
    )
    plt.xlabel("Sigma geo (%)", fontsize=12)
    plt.ylabel("Mean Campaign Throughput (t/d)", fontsize=12)
    plt.ylim(5500, 6000)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(
        "plots/Monte_Carlo_Throughput_Fig5_Standard.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("Saved 'plots/Monte_Carlo_Throughput_Fig5_Standard.png'.\n")


def _apply_equal_allocation(controller):
    n = len(controller.faces)
    fracs = [1.0 / n] * n
    # Force the physical fleet dispatch to always be evenly split
    for k in controller._mode_allocations.keys():
        controller._mode_allocations[k] = fracs

    # Reproduce the "equal allocation" control strategy: the fleet is treated as
    # unconstrained, so each face is driven at its equal share of the requested
    # rate regardless of physical fleet capacity. The fleet-limited real rate is
    # still recorded for telemetry.
    ctrl = controller
    orig_real = ctrl._face_real_extraction_rate

    def _unconstrained_real(face_index, target_extraction_rate):
        orig_real(face_index, target_extraction_rate)
        return target_extraction_rate

    ctrl._face_real_extraction_rate = _unconstrained_real


def _run_capacity_case(
    label: str,
    config_kwargs: dict = None,
    max_time: float = 60.0,
    equal_allocation: bool = False,
    np_seed: int = 42,
    random_seed: int = 11,
):
    from drs_mining.components.modes import MODES

    np.random.seed(np_seed)
    random.seed(random_seed)

    kwargs = config_kwargs or {}
    face1, face2, fleet, plant, controller, ore1_stock, ore2_stock, global_time = (
        build_multi_face_network(**kwargs)
    )
    if equal_allocation:
        _apply_equal_allocation(controller)

    controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine()
    _register_and_policy(
        engine, fleet, plant, controller, ore1_stock, ore2_stock, global_time
    )

    telemetry = build_telemetry(engine, controller, face1, face2, fleet, ore1_stock, ore2_stock)
    engine.attach_telemetry(telemetry)
    engine.run(until=max_time)

    df = telemetry.to_dataframe()
    df["scenario"] = label
    df["active_operating_mode_name"] = df["active_operating_mode"].apply(
        lambda x: x.name if x else "None"
    )
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
        "SHUTDOWN",
        "MODE_A",
        "MODE_A_CONTINGENCY",
        "MODE_A_MINE_SURGING",
        "MODE_B",
        "MODE_B_CONTINGENCY",
        "MODE_B_MINE_SURGING",
    ]
    mode_to_y = {mode: idx for idx, mode in enumerate(mode_order)}

    fig, axes = plt.subplots(6, 1, figsize=(14, 20), sharex=True)
    for label, group in combined.groupby("scenario"):
        axes[0].plot(
            group["time"],
            group["total_target_extraction_rate"],
            linestyle="--",
            label=f"{label} required",
        )
        axes[0].plot(
            group["time"],
            group["total_achieved_extraction_rate"],
            label=f"{label} actual",
        )
        axes[1].plot(group["time"], group["capacity_gap_rate"], label=label)
        axes[2].plot(group["time"], group["capacity_utilization"], label=label)
        axes[3].plot(group["time"], group["Ore2Stock_mass"], label=label)
        mode_y = group["active_operating_mode_name"].map(mode_to_y)
        axes[4].step(group["time"], mode_y, where="post", label=label)
        axes[5].plot(
            group["time"],
            group["face1_achieved_extraction_rate"],
            label=f"{label} face1",
        )
        axes[5].plot(
            group["time"],
            group["face2_achieved_extraction_rate"],
            linestyle="--",
            label=f"{label} face2",
        )

    axes[0].set_ylabel("Rate (t/d)")
    axes[0].set_title("Required vs Actual Extraction Rate")
    axes[1].set_ylabel("Capacity Gap (t/d)")
    axes[1].set_title("Lost Rate Due to Fleet Capacity Limit")
    axes[2].set_ylabel("Actual / Required")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Capacity Utilization")
    axes[3].set_ylabel("Ore 2 Stockpile (t)")
    axes[3].set_title("Ore 2 Stockpile Response")
    axes[4].set_ylabel("Mode")
    axes[4].set_yticks(list(mode_to_y.values()))
    axes[4].set_yticklabels(mode_order)
    axes[4].set_title("Operating Mode Timeline")
    axes[5].set_ylabel("Face Actual Rate (t/d)")
    axes[5].set_xlabel("Time (days)")
    axes[5].set_title("Face-Level Actual Extraction Rates")
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        "plots/Capacity_Policy_Comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    palette = {
        "MODE_A": "#1f77b4",
        "MODE_A_CONTINGENCY": "#2ca02c",
        "MODE_A_MINE_SURGING": "#9467bd",
        "MODE_B": "#d62728",
        "MODE_B_CONTINGENCY": "#ff7f0e",
        "MODE_B_MINE_SURGING": "#8c564b",
        "SHUTDOWN": "#FFD700",
    }
    mode_groups = {
        "Mode A": ("MODE_A", "MODE_A_CONTINGENCY", "MODE_A_MINE_SURGING"),
        "Mode B": ("MODE_B", "MODE_B_CONTINGENCY", "MODE_B_MINE_SURGING"),
        "Shutdown": ("SHUTDOWN",),
    }
    for label, group in combined.groupby("scenario"):
        diagnostic_df = group.copy()
        for mode_label, mode_names in mode_groups.items():
            diagnostic_df[mode_label] = diagnostic_df[
                "active_operating_mode_name"
            ].apply(
                lambda mode: (
                    len(mode_groups) - list(mode_groups).index(mode_label)
                    if mode in mode_names
                    else 0
                )
            )

        fig_diag, axes_diag = plt.subplots(
            4,
            1,
            figsize=(14, 15),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 2.2, 1, 1]},
        )
        axes_diag[0].step(
            diagnostic_df["time"],
            diagnostic_df["Mode A"],
            where="post",
            label="Mode A",
        )
        axes_diag[0].step(
            diagnostic_df["time"],
            diagnostic_df["Mode B"],
            where="post",
            label="Mode B",
        )
        axes_diag[0].step(
            diagnostic_df["time"],
            diagnostic_df["Shutdown"],
            where="post",
            label="Shutdown",
        )
        axes_diag[0].set_title("Modes (Step)")
        axes_diag[0].set_yticks([0, 1, 2, 3])
        axes_diag[0].legend(loc="upper right")

        plot_ore_with_modes(
            diagnostic_df,
            time_col="time",
            ore_cols=[
                "total_system_ore_mass",
                "Ore1Stock_mass",
                "Ore2Stock_mass",
            ],
            mode_col="active_operating_mode_name",
            campaign_split_mode="SHUTDOWN",
            title="Ore Stockpiles & Mode Changes",
            palette=palette,
            hlines=[
                {
                    "y": base_kwargs.get("target_ore_stock_level", 60000.0),
                    "color": "black",
                    "linestyle": "--",
                    "linewidth": 1.5,
                    "alpha": 0.7,
                    "label": "Target Total",
                },
                {
                    "y": base_kwargs.get("critical_ore2_level", 20400.0),
                    "color": "red",
                    "linestyle": ":",
                    "linewidth": 2,
                    "alpha": 0.8,
                    "label": "Critical Ore 2",
                },
            ],
            ax=axes_diag[1],
        )

        axes_diag[2].step(
            diagnostic_df["time"],
            diagnostic_df["total_target_extraction_rate"],
            where="post",
            linestyle="--",
            label="Required rate",
        )
        axes_diag[2].step(
            diagnostic_df["time"],
            diagnostic_df["total_achieved_extraction_rate"],
            where="post",
            label="Actual rate",
        )
        axes_diag[2].set_ylabel("Rate (t/d)")
        axes_diag[2].set_title("Required vs Actual Extraction Rate")
        axes_diag[2].legend(loc="upper right")

        axes_diag[3].step(
            diagnostic_df["time"],
            diagnostic_df["capacity_utilization"],
            where="post",
            label="Capacity utilization",
        )
        axes_diag[3].set_ylim(-0.05, 1.05)
        axes_diag[3].set_ylabel("Actual / Required")
        axes_diag[3].set_xlabel("Time (days)")
        axes_diag[3].set_title("Fleet Capacity Utilization")
        axes_diag[3].legend(loc="upper right")

        for ax in axes_diag:
            ax.grid(True, linestyle="--", alpha=0.35)
        fig_diag.suptitle(label, fontsize=15)
        fig_diag.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.0)
        safe_label = label.lower().replace(" ", "_").replace("+", "plus")
        safe_label = safe_label.replace("/", "_")
        fig_diag.savefig(
            f"plots/Capacity_Policy_Diagnostics_{safe_label}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig_diag)

    print("Saved capacity_policy_comparison.csv")
    print("Saved capacity_policy_comparison_summary.csv")
    print("Saved Capacity_Policy_Comparison.png")
    print("Saved Capacity_Policy_Diagnostics_*.png")
    return combined, summary_df


def run_and_analyze(config, equal_allocation=False, name="Dynamic Fleet Allocation"):
    """Run the multi-face simulation and produce full diagnostics dashboard."""
    np.random.seed(42)
    random.seed(11)
    face1, face2, fleet, plant, controller, ore1_stock, ore2_stock, global_time = (
        build_multi_face_network(**config)
    )

    if equal_allocation:
        _apply_equal_allocation(controller)

    controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine()
    _register_and_policy(
        engine, fleet, plant, controller, ore1_stock, ore2_stock, global_time
    )

    telemetry = build_telemetry(engine, controller, face1, face2, fleet, ore1_stock, ore2_stock)
    engine.attach_telemetry(telemetry)
    result = engine.run(until=config.get("replication_length", 99999.0))
    print_statistics(controller, plant, [face1, face2])

    df = result.history

    # --- Mode Transition Log ---
    print(f"\n--- Mode Transition Log ({name}) ---")
    df["active_operating_mode_name"] = df["active_operating_mode"].apply(
        lambda x: x.name if x else "None"
    )
    print(df["active_operating_mode_name"].unique()[:5])
    df["prev_mode_name"] = df["active_operating_mode_name"].shift(1)
    transitions = df[
        (df["active_operating_mode_name"] != df["prev_mode_name"])
        & df["prev_mode_name"].notna()
    ]

    for idx, row in transitions.iterrows():
        print(
            f"Time: {row['time']:.2f} | Transition: {row['prev_mode_name']} -> {row['active_operating_mode_name']}"
        )
        crit_lvl = config.get("critical_ore2_level", 20400.0)
        target_lvl = config.get("target_ore_stock_level", 60000.0)
        print(
            f"  ↳ Ore1 Stock: {row['Ore1Stock_mass']:.1f} | Ore2 Stock: {row['Ore2Stock_mass']:.1f} (Critical: {crit_lvl}) | Total Stock: {row['total_system_ore_mass']:.1f} (Target: {target_lvl})"
        )
        print(
            f"  ↳ Campaign/Shutdown Timer: {row['current_campaign_duration']:.2f} | Contingency Timer: {row['current_contingency_duration']:.2f}"
        )
    print("---------------------------\n")

    # --- Cumulative Deficit by Mode Log ---
    dt = df["time"].diff().fillna(0)
    actual_extraction_step = (
        (df["face1_extracted_mass"] + df["face2_extracted_mass"]).diff().fillna(0)
    )
    ideal_extraction_step = dt * 6000.0
    step_deficit = (ideal_extraction_step - actual_extraction_step).clip(lower=0)

    deficit_df = pd.DataFrame(
        {"mode": df["active_operating_mode_name"], "deficit": step_deficit}
    )

    total_deficit_by_mode = (
        deficit_df.groupby("mode")["deficit"].sum().sort_values(ascending=False)
    )

    print(f"\n--- Cumulative Lost Production (Deficit) by Mode ({name}) ---")
    total_lost = total_deficit_by_mode.sum()
    for mode, lost in total_deficit_by_mode.items():
        mode_name = str(mode).split(".")[-1]
        pct = (lost / total_lost * 100) if total_lost > 0 else 0
        print(f"{mode_name}: {lost:.1f} tons ({pct:.1f}%)")
    print(f"TOTAL: {total_lost:.1f} tons")
    print("----------------------------------------------------\n")

    # Create Modes Series
    df["Mode A"] = df["active_operating_mode_name"].apply(
        lambda m: (
            3
            if m
            in (
                "MODE_A",
                "MODE_A_CONTINGENCY",
                "MODE_A_MINE_SURGING",
            )
            else 0
        )
    )
    df["Mode B"] = df["active_operating_mode_name"].apply(
        lambda m: (
            2
            if m
            in (
                "MODE_B",
                "MODE_B_CONTINGENCY",
                "MODE_B_MINE_SURGING",
            )
            else 0
        )
    )
    df["Shutdown"] = df["active_operating_mode_name"].apply(
        lambda m: 1 if m == "SHUTDOWN" else 0
    )

    # Create Ore Level Series (scaled by 1000)
    df["Total Ore Stockpile Level"] = df["total_system_ore_mass"] / 1000.0
    df["Ore 1 Stockpile Level"] = df["Ore1Stock_mass"] / 1000.0
    df["Ore 2 Stockpile Level"] = df["Ore2Stock_mass"] / 1000.0

    from drs.plot import (
        plot_time_series,
        plot_dual_axis_step,
        plot_safety_margin,
        Dashboard,
    )
    from drs_mining.components.plot import (
        plot_ore_with_modes,
        plot_mode_distribution,
        plot_mode_dwell_times,
        plot_normalized_deviation_violin,
        plot_attributed_deficit,
        plot_deficit_disparity,
        plot_deficit_breakdown_bar,
        plot_structural_vs_operational_deficit,
        plot_normalized_cumulative_deficit,
        plot_structural_vs_operational_by_mode,
    )

    palette = {
        "MODE_A": "#1f77b4",
        "MODE_A_CONTINGENCY": "#2ca02c",
        "MODE_A_MINE_SURGING": "#9467bd",
        "MODE_B": "#d62728",
        "MODE_B_CONTINGENCY": "#ff7f0e",
        "MODE_B_MINE_SURGING": "#8c564b",
        "SHUTDOWN": "#FFD700",
    }

    structural_modes = ["SHUTDOWN", "MODE_A"]

    dash = Dashboard(
        nrows=23, ncols=1, figsize=(18, 69), sharex=False, title=f"Comprehensive Mine Diagnostics ({name})"
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 20, 21])

    plot_time_series(
        df,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Modes (Step)",
        is_step=True,
        ax=dash[0],
    )
    plot_ore_with_modes(
        df,
        time_col="time",
        ore_cols=[
            "total_system_ore_mass",
            "Ore1Stock_mass",
            "Ore2Stock_mass",
        ],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Ore Stockpiles & Campaigns",
        palette=palette,
        hlines=[
            {
                "y": 60000,
                "color": "black",
                "linestyle": "--",
                "linewidth": 1.5,
                "alpha": 0.7,
                "label": "Target Total (60k)",
            },
            {
                "y": 20400,
                "color": "red",
                "linestyle": ":",
                "linewidth": 2,
                "alpha": 0.8,
                "label": "Critical Ore 2 (20.4k)",
            },
        ],
        ax=dash[1],
    )
    plot_dual_axis_step(
        df,
        y1_col="face1_parcel_mass",
        y2_col="face1_parcel_ratio",
        y1_label="Face 1 Parcel Mass (tons)",
        y2_label="Face 1 Ore 1 Fraction",
        title="Face 1 Current Parcel Properties",
        y1_color="saddlebrown",
        y2_color="darkorange",
        ax=dash[2],
    )
    plot_dual_axis_step(
        df,
        y1_col="face2_parcel_mass",
        y2_col="face2_parcel_ratio",
        y1_label="Face 2 Parcel Mass (tons)",
        y2_label="Face 2 Ore 1 Fraction",
        title="Face 2 Current Parcel Properties",
        y1_color="saddlebrown",
        y2_color="darkorange",
        ax=dash[3],
    )
    plot_dual_axis_step(
        df,
        y1_col="mixed_achieved_extraction_rate",
        y2_col="mixed_ore1_fraction",
        y1_label="Combined Extraction Rate (t/d)",
        y2_label="Mixed Ore 1 Fraction",
        title="Combined Mine Output Properties",
        y1_color="saddlebrown",
        y2_color="darkorange",
        ax=dash[4],
    )
    plot_time_series(
        df,
        y_columns=[
            "mixed_target_extraction_rate",
            "mixed_real_extraction_rate",
            "mixed_achieved_extraction_rate",
        ],
        title="Fleet-Constrained Extraction Rates",
        is_step=True,
        ax=dash[5],
    )
    plot_time_series(
        df,
        y_columns=["face1_alloc", "face2_alloc", "ore2_ratio"],
        title="Active Fleet Allocation & Stockpile Ratio",
        is_step=True,
        ax=dash[6],
    )
    plot_time_series(
        df,
        y_columns=["face1_real_capacity", "face1_target_rate"],
        title="Face 1 Real Capacity vs Target Rate (Headroom)",
        is_step=True,
        ax=dash[7],
    )
    plot_time_series(
        df,
        y_columns=["face2_real_capacity", "face2_target_rate"],
        title="Face 2 Real Capacity vs Target Rate (Headroom)",
        is_step=True,
        ax=dash[8],
    )
    plot_time_series(
        df,
        y_columns=["face1_match_factor", "face2_match_factor"],
        title="Match Factor per Face (1.0 = balanced)",
        is_step=True,
        ax=dash[9],
    )
    plot_time_series(
        df,
        y_columns=["total_unused_trucks"],
        title="Total Unused Trucks (Spare Fleet Capacity)",
        is_step=True,
        ax=dash[10],
    )
    plot_time_series(
        df,
        y_columns=[
            "face1_truck_cycle_time_hours",
            "face2_truck_cycle_time_hours",
        ],
        title="Truck Cycle Times (Hours) & Traffic Delays",
        is_step=True,
        ax=dash[11],
    )
    plot_safety_margin(
        df,
        level_col="Ore1Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 1 Distance to Floor",
        danger_threshold=1000.0,
        ax=dash[12],
    )
    plot_safety_margin(
        df,
        level_col="Ore2Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 2 Distance to Floor",
        danger_threshold=1000.0,
        ax=dash[13],
    )
    plot_mode_distribution(
        df,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% of Time Spent)",
        palette=palette,
        ax=dash[14],
    )
    plot_mode_dwell_times(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        title="Mode Stability (Dwell Times)",
        ax=dash[15],
    )
    plot_normalized_deviation_violin(
        df,
        title="Stockpile Deviation Variance (Violin)",
        target_total=60000.0,
        target_ore1=42000.0,
        target_ore2=18000.0,
        ax=dash[16],
    )
    plot_attributed_deficit(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        extraction_col="cumulative_extracted_mass",
        ideal_rate_per_day=6000.0,
        title="Cumulative Production Deficit by Mode",
        palette=palette,
        ax=dash[17],
    )
    plot_deficit_disparity(
        df,
        mode_col="active_operating_mode_name",
        title="Mode Efficiency (Time Spent vs. Deficit Caused)",
        ideal_rate=6000.0,
        ax=dash[18],
    )
    plot_deficit_breakdown_bar(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[19],
    )
    plot_structural_vs_operational_deficit(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate=6000.0,
        structural_modes=structural_modes,
        ax=dash[20],
    )
    plot_normalized_cumulative_deficit(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[21],
    )
    plot_structural_vs_operational_by_mode(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate=6000.0,
        structural_modes=structural_modes,
        ax=dash[22],
    )

    prefix = name.lower().replace(" ", "_")
    dash.save(f"plots/Comprehensive_Diagnostics_Plot_{prefix}.png")
    plt.close(dash.fig)

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

    # You can also run it a single time and print out the statistics to evaluate how it spends time
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

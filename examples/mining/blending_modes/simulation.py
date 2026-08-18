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
import matplotlib.pyplot as plt

from drs import DRSEngine, Telemetry
from drs_mining.components import (
    BlendingNetwork,
    ConcentratorMineFace,
    ConcentratorPlant,
    ConcentratorController,
    ContinuousFleetLogistics,
    Stockpile,
)
from drs_mining.components.modes import MODES
from drs_mining.components.plot import (
    plot_single_face_dashboard,
    prepare_history,
    print_deficit_by_mode,
    print_state_change_transitions,
)


# ============================================================
# Inline control policy
# ------------------------------------------------------------
# The ``on_step`` handler is a thin orchestrator of the component
# APIs: the controller updates the operating mode and computes the
# target flow rates (``update_mode`` / ``get_target_rates``), the
# mine is driven by setting ``target_rate`` (parcel mechanics run
# inside the mine's ``step``), the fleet routes the mined material
# (``fleet.route``), the stockpiles feed-and-draw (``feed_and_draw``),
# and the plant processes the combined draw (``process``).
# ============================================================


# ============================================================
# Entity construction
# ============================================================


def build_blending_network(
    mean_ore_fraction: float = 0.30,
    std_dev_ore_fraction: float = 0.05,
    prob_new_facies: float = 0.3,
    variation_same_facies: float = 0.01,
    min_ore_mass: float = 30000.0,
    max_ore_mass: float = 50000.0,
    total_ore_to_extract: float = 6600000.0,
    ore_to_be_extracted_during_warming_period: float = 600000.0,
    target_ore_stock_level: float = 60000.0,
    critical_ore2_level: float = 20400.0,
    duration_of_production_campaigns: float = 34.0,
    duration_of_shutdowns: float = 1.0,
    duration_of_contingency_segments: float = 1.0,
    ore1_capacity: float = float("inf"),
    ore2_capacity: float = float("inf"),
    plant_max_rate: float = float("inf"),
    **kwargs,
) -> "BlendingNetwork":
    """Build the blending network flat: every entity is constructed here.

    Returns a ``BlendingNetwork`` holding ``(mine, fleet, plant, controller,
    ore1_stock, ore2_stock)``. There is no scenario container; call
    ``network.register(engine)`` to register every stateful leaf.
    """
    mine = ConcentratorMineFace(
        mean_ore_fraction=mean_ore_fraction,
        std_dev_ore_fraction=std_dev_ore_fraction,
        prob_new_facies=prob_new_facies,
        variation_same_facies=variation_same_facies,
        min_ore_mass=min_ore_mass,
        max_ore_mass=max_ore_mass,
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

    plant = ConcentratorPlant(
        mine, fleet, ore1_stock, ore2_stock, max_rate=plant_max_rate
    )
    controller = ConcentratorController(
        mine=mine,
        fleet=fleet,
        plant=plant,
        target_ore_stock_level=target_ore_stock_level,
        critical_ore2_level=critical_ore2_level,
        duration_of_production_campaigns=duration_of_production_campaigns,
        duration_of_shutdowns=duration_of_shutdowns,
        duration_of_contingency_segments=duration_of_contingency_segments,
        ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        total_ore_to_extract=total_ore_to_extract,
    )

    return BlendingNetwork(
        mine=mine,
        fleet=fleet,
        plant=plant,
        controller=controller,
        ore1_stock=ore1_stock,
        ore2_stock=ore2_stock,
    )


def _register_and_policy(engine, network):
    """Register the stateful leaves and wire the inline policy onto the engine."""
    network.register(engine)

    mine = network.mine
    fleet = network.fleet
    ctrl = network.controller
    plant = network.plant
    ore1_stock = network.ore1_stock
    ore2_stock = network.ore2_stock

    @engine.on_step
    def manage_blending(t: float):
        # 1. Update operating mode based on campaign timers & stockpile levels
        mode = ctrl.update_mode(ore1_stock, ore2_stock)

        # 2. Compute target flow rates for this mode
        mine_target, stock1_target, stock2_target = ctrl.get_target_rates(
            mode, fleet
        )

        # 3. Mine & route parcels
        mine.target_rate = mine_target
        ore1_in, ore2_in = fleet.route(mine.actual_rate, mine.current_ore_grade)

        # 4. Feed stockpiles and draw into plant
        out1 = ore1_stock.feed_and_draw(ore1_in, stock1_target)
        out2 = ore2_stock.feed_and_draw(ore2_in, stock2_target)
        plant.process(out1 + out2)


def print_statistics(controller, plant, mine):
    """Print operating-mode time-shares and throughput for a blending network."""
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
            total_ore_processed = mine.net_extracted_mass
        throughput = total_ore_processed / active_time
        print(f"Throughput: {throughput:.4f} tons/day")
    else:
        print("Active time is 0. Cannot calculate throughput.")


def evaluate_throughput(config_kwargs: dict, N: int) -> tuple[float, float]:
    """Runs the simulation N times, extracting throughputs.
    Returns (mean_throughput, std_dev_throughput).
    """
    throughputs = []

    for idx in range(N):
        np.random.seed(idx)
        random.seed(idx)

        network = build_blending_network(**config_kwargs)

        engine = DRSEngine()
        _register_and_policy(engine, network)
        engine.run(until=config_kwargs.get("replication_length", float("inf")))

        active_time = network.controller.active_duration(engine.current_time)
        if active_time > 0:
            throughput = network.mine.net_extracted_mass / active_time
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=0.05)
    parser.add_argument("--N", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(11)
    random.seed(11)
    network = build_blending_network(
        replication_length=99999.0,
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
        prob_new_facies=0.3,
    )
    mine = network.mine
    fleet = network.fleet
    controller = network.controller

    controller.active_operating_mode.value = MODES["MODE_A"]

    engine = DRSEngine()
    _register_and_policy(engine, network)

    telemetry = Telemetry(model=engine)
    telemetry.register_metric(
        "MassOfCurrentParcel",
        lambda t, m, s, _: mine.active_parcel_initial_mass.value,
    )
    telemetry.register_metric(
        "CurrentParcelRoutingFraction",
        lambda t, m, s, _: fleet.stockpile2_routing_fraction.value,
    )
    telemetry.register_metric(
        "Campaign_Shutdown",
        lambda t, m, s, _: controller.current_campaign_duration.value,
    )
    telemetry.register_metric(
        "Contingency",
        lambda t, m, s, _: controller.current_contingency_duration.value,
    )
    engine.attach_telemetry(telemetry)
    result = engine.run(until=99999.0)

    print(result.summary())

    print_statistics(controller, network.plant, mine)

    df = prepare_history(result.history)
    print_state_change_transitions(result.events)

    print_deficit_by_mode(
        df,
        extraction_cols=["cumulative_extracted_mass"],
        ideal_rate=6000.0,
    )

    plot_single_face_dashboard(df)

    # Recreate Figure 5 from paper
    plot_monte_carlo_throughput(
        N=args.N, total_stockpile_level=args.total_stockpile_level
    )
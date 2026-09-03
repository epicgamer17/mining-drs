import argparse
from typing import Any, Iterator, Mapping, Optional, Sequence

import pandas as pd

import drs
from drs import (
    DRSEngine,
    Telemetry,
    Storage,
    Processor,
    Flow,
    blend_flows,
)
from drs_mining.components import (
    OperatingMode,
    MaterialSource,
    autocorrelated_generator,
)
from drs_mining.components.plot import (
    plot_single_face_dashboard,
    prepare_history,
    print_deficit_by_mode,
    print_transition_log,
)

# ==============================================================================
# Tactical Blending Hyperparameters & Simulation Constants
# ==============================================================================
TOTAL_ORE_TO_EXTRACT_TONNES = 6_600_000.0
WARMING_PERIOD_ORE_TONNES = 600_000.0

CAMPAIGN_DURATION_DAYS = 34.0
SHUTDOWN_DURATION_DAYS = 1.0
CONTINGENCY_SEGMENT_DURATION_DAYS = 1.0

TARGET_ORE_STOCK_TONNES = 60_000.0
CRITICAL_ORE2_STOCK_TONNES = 20_400.0

MEAN_ORE_FRACTION = 0.30
STD_DEV_ORE_FRACTION = 0.05
PROB_NEW_FACIES = 0.30
VARIATION_SAME_FACIES = 0.01
MIN_PARCEL_MASS_TONNES = 30_000.0
MAX_PARCEL_MASS_TONNES = 50_000.0
INITIAL_PARCEL_MASS_TONNES = 40_000.0

MILL_MAX_RATE_TPD = 6_000.0


def create_blending_modes() -> dict[str, OperatingMode]:
    """Instantiate discrete campaign and operating modes for the blending network."""
    return {
        "MODE_A": OperatingMode(
            "MODE_A",
            id=0,
            category="mill",
            draw_rates={"Ore1Stock": 3600.0, "Ore2Stock": 2400.0},
        ),
        "MODE_A_CONTINGENCY": OperatingMode(
            "MODE_A_CONTINGENCY",
            id=1,
            category="mill",
            draw_rates={"Ore1Stock": 3900.0, "Ore2Stock": 0.0},
        ),
        "MODE_A_MINE_SURGING": OperatingMode(
            "MODE_A_MINE_SURGING",
            id=2,
            category="mill",
            draw_rates={"Ore1Stock": 3600.0, "Ore2Stock": 2400.0},
        ),
        "MODE_B": OperatingMode(
            "MODE_B",
            id=3,
            category="mill",
            draw_rates={"Ore1Stock": 4600.0, "Ore2Stock": 800.0},
        ),
        "MODE_B_CONTINGENCY": OperatingMode(
            "MODE_B_CONTINGENCY",
            id=4,
            category="mill",
            draw_rates={"Ore1Stock": 0.0, "Ore2Stock": 2500.0},
        ),
        "MODE_B_MINE_SURGING": OperatingMode(
            "MODE_B_MINE_SURGING",
            id=5,
            category="mill",
            draw_rates={"Ore1Stock": 4600.0, "Ore2Stock": 800.0},
        ),
        "SHUTDOWN": OperatingMode(
            "SHUTDOWN",
            id=6,
            category="mill",
            draw_rates={"Ore1Stock": 0.0, "Ore2Stock": 0.0},
        ),
    }




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


def build_blending_network(
    mean_ore_fraction: float = MEAN_ORE_FRACTION,
    std_dev_ore_fraction: float = STD_DEV_ORE_FRACTION,
    prob_new_facies: float = PROB_NEW_FACIES,
    variation_same_facies: float = VARIATION_SAME_FACIES,
    min_ore_mass: float = MIN_PARCEL_MASS_TONNES,
    max_ore_mass: float = MAX_PARCEL_MASS_TONNES,
    total_ore_to_extract: float = TOTAL_ORE_TO_EXTRACT_TONNES,
    ore_to_be_extracted_during_warming_period: float = WARMING_PERIOD_ORE_TONNES,
    target_ore_stock_level: float = TARGET_ORE_STOCK_TONNES,
    critical_ore2_level: float = CRITICAL_ORE2_STOCK_TONNES,
    duration_of_production_campaigns: float = CAMPAIGN_DURATION_DAYS,
    duration_of_shutdowns: float = SHUTDOWN_DURATION_DAYS,
    duration_of_contingency_segments: float = CONTINGENCY_SEGMENT_DURATION_DAYS,
    modes: Optional[Mapping[str, OperatingMode]] = None,
    **kwargs,
) -> tuple:
    stream = autocorrelated_generator(
        mean_fraction=mean_ore_fraction,
        std_dev=std_dev_ore_fraction,
        prob_new_facies=prob_new_facies,
        variation_step=variation_same_facies,
        min_mass=min_ore_mass,
        max_mass=max_ore_mass,
        initial_mass=INITIAL_PARCEL_MASS_TONNES,
        attribute_name="ore2_fraction",
    )
    source = MaterialSource(
        name="mine_face_reserve",
        total_tonnes=total_ore_to_extract,
        stream=stream,
        warming_period=ore_to_be_extracted_during_warming_period,
    )

    initial_mass1 = (1.0 - mean_ore_fraction) * target_ore_stock_level
    ore1_stock = Storage(
        name="Ore1Stock",
        initial_level=initial_mass1,
        initial_attributes={"ore2_fraction": 0.0},
    )
    initial_mass2 = mean_ore_fraction * target_ore_stock_level
    ore2_stock = Storage(
        name="Ore2Stock",
        initial_level=initial_mass2,
        initial_attributes={"ore2_fraction": 1.0},
    )

    mill = Processor(name="mill", max_rate=MILL_MAX_RATE_TPD)
    mill.campaign_timer = drs.Timer("current_campaign_duration", initial_value=0.0)
    mill.campaign_timer.rate = 1.0
    mill.campaign_timer.upper_threshold = duration_of_production_campaigns

    mill.contingency_timer = drs.Timer("current_contingency_duration", initial_value=0.0)
    mill.contingency_timer.rate = 0.0
    mill.contingency_timer.upper_threshold = duration_of_contingency_segments

    blending_modes = dict(modes or create_blending_modes())

    active_campaign = drs.Variable("active_campaign_mode", blending_modes["MODE_A"])
    active_mode = drs.Variable("active_operating_mode", blending_modes["MODE_A"])
    blending_modes["MODE_A"].activate()

    control_state = {
        "campaign_timer": mill.campaign_timer,
        "contingency_timer": mill.contingency_timer,
        "active_campaign": active_campaign,
        "active_mode": active_mode,
        "modes": blending_modes,
        "campaign_duration": duration_of_production_campaigns,
        "shutdown_duration": duration_of_shutdowns,
        "contingency_duration": duration_of_contingency_segments,
        "critical_ore2_level": critical_ore2_level,
        "target_total_stock": target_ore_stock_level,
    }

    return source, mill, control_state, ore1_stock, ore2_stock


def _register_and_policy(engine: DRSEngine, network: tuple) -> drs.Level:
    source, mill, control_state, ore1_stock, ore2_stock = network
    cumulative_milled_mass = drs.Level("cumulative_milled_mass", initial_value=0.0, owner=mill)
    mill.cumulative_milled_mass = cumulative_milled_mass

    modes = control_state["modes"]
    campaign_timer = control_state["campaign_timer"]
    contingency_timer = control_state["contingency_timer"]
    active_campaign = control_state["active_campaign"]
    active_mode = control_state["active_mode"]

    engine.register(
        source,
        mill,
        ore1_stock,
        ore2_stock,
        *modes.values(),
    )

    @engine.on_step
    def manage_blending(t: float):
        # 1. Update campaign mode
        campaign_mode = update_campaign_mode(
            campaign_timer,
            active_campaign,
            ore2_stock.level,
            modes,
            campaign_duration=control_state["campaign_duration"],
            shutdown_duration=control_state["shutdown_duration"],
            critical_ore2_level=control_state["critical_ore2_level"],
        )

        # 2. Resolve operating mode
        next_mode = resolve_operating_mode(
            campaign_mode,
            active_mode.value,
            ore1_stock.level,
            ore2_stock.level,
            contingency_timer,
            modes,
            target_total_stock=control_state["target_total_stock"],
            contingency_duration=control_state["contingency_duration"],
        )

        # 3. Transition active mode if changed
        if active_mode.value != next_mode:
            active_mode.value.deactivate()
            next_mode.activate()
            active_mode.value = next_mode

        # 4. Setpoints & milling
        draw_rates = next_mode.draw_rates
        ore1_target = draw_rates.get("Ore1Stock", 0.0)
        ore2_target = draw_rates.get("Ore2Stock", 0.0)

        ore2_frac = source.current_attributes.get("ore2_fraction", 0.0)
        mode_name = next_mode.name
        if "_MINE_SURGING" in mode_name:
            if mode_name == "MODE_A_MINE_SURGING":
                effective_fraction = max(1.0 - ore2_frac, 0.01)
                mine_target = ore1_target / effective_fraction
            else:
                effective_fraction = max(ore2_frac, 0.01)
                mine_target = ore2_target / effective_fraction
        else:
            mine_target = ore1_target + ore2_target

        extraction_flow = source.extract(mine_target)

        in_flow1 = Flow(
            rate=extraction_flow.rate * (1.0 - ore2_frac),
            attributes=extraction_flow.attributes,
        )
        in_flow2 = Flow(
            rate=extraction_flow.rate * ore2_frac,
            attributes=extraction_flow.attributes,
        )

        out1 = ore1_stock.feed_and_draw(in_flow1, ore1_target)
        out2 = ore2_stock.feed_and_draw(in_flow2, ore2_target)

        blended_feed = blend_flows([out1, out2])
        mill.process(blended_feed)
        cumulative_milled_mass.rate = mill.actual_rate

    return cumulative_milled_mass


def print_statistics(modes: Mapping[str, OperatingMode], cumulative_milled_mass: drs.Level, total_time: float):
    print("\n--- Output Statistics ---")
    if total_time > 0:
        for mode_name, label in [
            ("MODE_A", "PortionOfTimeInModeA"),
            ("MODE_A_CONTINGENCY", "PortionOfTimeInModeAContingency"),
            ("MODE_A_MINE_SURGING", "PortionOfTimeInModeAMineSurging"),
            ("MODE_B", "PortionOfTimeInModeB"),
            ("MODE_B_CONTINGENCY", "PortionOfTimeInModeBContingency"),
            ("SHUTDOWN", "PortionOfTimeInShutdown"),
        ]:
            mode = modes.get(mode_name)
            fraction = (mode.cumulative_time / total_time) if mode else 0.0
            print(f"{label}: {fraction:.4f}")
    print(f"CumulativeTonsMilled: {cumulative_milled_mass.value:.2f}")


def run_blending_simulation(
    replication_length: float = 999999.0,
    plot: bool = False,
    seed: int = 11,
    **kwargs,
) -> tuple[drs.SimResult, Optional[pd.DataFrame]]:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    network = build_blending_network(**kwargs)
    source, mill, control_state, ore1_stock, ore2_stock = network
    modes = control_state["modes"]
    active_mode = control_state["active_mode"]
    campaign_timer = control_state["campaign_timer"]
    contingency_timer = control_state["contingency_timer"]

    engine = DRSEngine()
    cumulative_milled_mass = _register_and_policy(engine, network)

    # Warmup phase (pre-burns warming period tonnage)
    source.total_tonnes = WARMING_PERIOD_ORE_TONNES
    engine.run(until=99999.0)

    # Reset cumulative mode timers after warmup
    for m in modes.values():
        m.reset_timer()

    # Production phase
    source.total_tonnes = TOTAL_ORE_TO_EXTRACT_TONNES

    telemetry = None
    if plot:
        telemetry = Telemetry(model=engine)
        telemetry.register_metric("active_operating_mode", lambda t, m, s, _: active_mode.value.name)
        telemetry.register_metric("active_operating_mode_name", lambda t, m, s, _: active_mode.value.name)
        telemetry.register_metric("total_stock_level", lambda t, m, s, _: ore1_stock.level + ore2_stock.level)
        telemetry.register_metric(
            "MassOfCurrentParcel",
            lambda t, m, s, _: source.active_entity.mass if source.active_entity else 0.0,
        )
        telemetry.register_metric(
            "CurrentParcelRoutingFraction",
            lambda t, m, s, _: source.current_attributes.get("ore2_fraction", 0.0),
        )
        telemetry.register_metric(
            "CurrentParcelOreFraction",
            lambda t, m, s, _: source.current_attributes.get("ore2_fraction", 0.0),
        )
        telemetry.register_metric(
            "Campaign_Shutdown",
            lambda t, m, s, _: campaign_timer.value,
        )
        telemetry.register_metric(
            "Contingency",
            lambda t, m, s, _: contingency_timer.value,
        )
        engine.attach_telemetry(telemetry)

    result = engine.run(until=replication_length)

    print_statistics(modes, cumulative_milled_mass, result.duration)

    history = None
    if plot and telemetry is not None:
        history = telemetry.to_dataframe()
        history = prepare_history(history)
        print_transition_log(history)
        print_deficit_by_mode(history, extraction_cols=["mine_face_reserve_cumulative_extracted_mass"])
        plot_single_face_dashboard(history)

    return result, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tactical Blending Simulation")
    parser.add_argument("--days", type=float, default=999999.0, help="Simulation duration (days)")
    parser.add_argument("--plot", action="store_true", default=True, help="Display telemetry dashboard")
    parser.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plotting")
    args = parser.parse_args()

    run_blending_simulation(replication_length=args.days, plot=args.plot)

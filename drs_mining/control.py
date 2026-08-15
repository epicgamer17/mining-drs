"""Top-level control policy for the mining blending simulation.

The policy is a pure function of the simulation state, decoupled from the DRS
component physics. The component modules expose only physical mechanics
(parcel boundaries, stockpile accumulation, plant throughput); the policy
owns the operating-mode logic, target rates, routing, and stockpile balancing.
"""

from .components.modes import RequireDecision, MODES


TIMER_MAP = {
    "MODE_A": "cumulative_time_mode_a",
    "MODE_A_CONTINGENCY": "cumulative_time_mode_a_contingency",
    "MODE_A_MINE_SURGING": "cumulative_time_mode_a_surging",
    "MODE_B": "cumulative_time_mode_b",
    "MODE_B_CONTINGENCY": "cumulative_time_mode_b_contingency",
    "MODE_B_MINE_SURGING": "cumulative_time_mode_b_surging",
    "SHUTDOWN": "cumulative_time_shutdown",
}

CONTINGENCY_MODES = {"MODE_A_CONTINGENCY", "MODE_B_CONTINGENCY"}

ACTION_MODES = [
    MODES["MODE_A"],
    MODES["MODE_B"],
    MODES["MODE_A_MINE_SURGING"],
    MODES["MODE_B_MINE_SURGING"],
]

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


def step_policy(sim, time):
    """Top-level control policy invoked by the engine once per step."""
    sim.global_time.rate = 1.0
    ctrl = sim.controller

    if _is_multi_face(sim):
        _multi_face_step(sim)
    else:
        _single_face_step(sim)

    ore1_rate, ore2_rate = _route(sim)
    _balance_stockpiles(sim, ore1_rate, ore2_rate)

    if ctrl is not None:
        ctrl.total_system_ore_mass.rate = (
            sim.ore1_stock.current_mass.rate + sim.ore2_stock.current_mass.rate
        )


def _is_multi_face(sim) -> bool:
    return bool(getattr(sim.controller, "faces", None))


def _single_face_step(sim):
    ctrl = sim.controller

    if sim.mine is not None and abs(sim.mine.net_extracted_mass) < 1e-6:
        _reset_mode_timers(ctrl)

    ctrl.total_system_ore_mass.value = sim.total_stockpile_mass
    _apply_next_mode(sim)
    _update_mode_timers(sim)
    _target_rates(sim)

    if sim.mine is not None:
        _drive_face(sim.mine, ctrl.target_mine_mass_rate.value)


def _multi_face_step(sim):
    ctrl = sim.controller
    ctrl.total_extra_trucks.value = 0.0

    total_net_extracted = sum(f.net_extracted_mass for f in ctrl.faces)
    if abs(total_net_extracted) < 1e-6:
        _reset_mode_timers(ctrl)

    ctrl.total_system_ore_mass.value = sim.total_stockpile_mass
    _apply_next_mode(sim)
    mode_name = ctrl.active_operating_mode.value.name
    _update_mode_timers(sim)

    ctrl.fleet_shift_timer.rate = 1.0
    ctrl.fleet_shift_timer.upper_threshold = ctrl.fleet_shift_duration

    shift_due = ctrl.fleet_shift_timer.value >= ctrl.fleet_shift_duration - 1e-6
    mode_changed = mode_name != ctrl.current_shift_mode_name
    if ctrl.current_shift_allocations is None or shift_due or mode_changed:
        ctrl.fleet_shift_timer.reset()
        ctrl._reallocate_fleet_for_shift()

    _target_rates(sim)

    fracs = ctrl.current_shift_allocations
    for i, face in enumerate(ctrl.faces):
        if fracs is not None and i < len(fracs):
            target = ctrl.target_mine_mass_rate.value * fracs[i]
            real = ctrl._face_real_extraction_rate(i, target)
            ctrl.face_target_extraction_rates[i].value = target
            ctrl.face_real_extraction_rates[i].value = real
            ctrl.face_achieved_extraction_rates[i].value = min(target, real)
            _drive_face(face, ctrl.face_achieved_extraction_rates[i].value)
        else:
            ctrl.face_target_extraction_rates[i].value = 0.0
            ctrl.face_real_extraction_rates[i].value = 0.0
            ctrl.face_achieved_extraction_rates[i].value = 0.0
            ctrl.face_operational_downtime_fractions[i].value = 0.0
            _drive_face(face, 0.0)

    ctrl.cumulative_mine_development.rate = (
        ctrl.total_extra_trucks.value * ctrl.development_rate_per_extra_truck
    )


def _mine_sources(sim):
    ctrl = sim.controller
    if getattr(ctrl, "faces", None):
        return ctrl.faces
    if getattr(sim, "mine", None) is not None:
        return [sim.mine]
    return []


def _drive_face(face, target_rate):
    target = (
        target_rate.value if hasattr(target_rate, "value") else float(target_rate)
    )
    face.target_rate = target
    face.advance_parcel_state()
    rate = face.actual_rate
    face.cumulative_extracted_mass.rate = rate
    face.parcel_extracted_mass.rate = rate


def _route(sim):
    fleet = sim.fleet
    ore1 = ore2 = total = 0.0
    for src in _mine_sources(sim):
        rate = src.actual_rate
        ore2_frac = src._get_current_attr_value()
        ore2 += rate * ore2_frac
        ore1 += rate * (1.0 - ore2_frac)
        total += rate
    if total > 1e-6 and fleet is not None:
        fleet.stockpile2_routing_fraction.value = ore2 / total
    return ore1, ore2


def _balance_stockpiles(sim, ore1_rate, ore2_rate):
    ctrl = sim.controller
    out1 = _drive_stock(
        sim.ore1_stock, ore1_rate, 1.0, ctrl.target_stock1_outflow_rate.value
    )
    out2 = _drive_stock(
        sim.ore2_stock, ore2_rate, 0.0, ctrl.target_stock2_outflow_rate.value
    )
    if sim.plant is not None:
        sim.plant.target_rate = out1 + out2
        sim.plant.cumulative_milled_mass.rate = sim.plant.actual_rate


def _drive_stock(stock, inflow_rate, inflow_attr, requested_outflow):
    current_inflow = inflow_rate
    actual_outflow = requested_outflow
    if stock.level <= 1e-6:
        actual_outflow = min(actual_outflow, current_inflow)

    net = current_inflow - actual_outflow
    stock.current_mass.rate = net
    for attr in stock.expected_attributes:
        level = getattr(stock, attr)
        level.rate = (
            current_inflow * inflow_attr - actual_outflow * stock.current_concentration(attr)
        )

    if net < 0:
        stock.current_mass.lower_threshold = 0.0
        for attr in stock.expected_attributes:
            getattr(stock, attr).lower_threshold = 0.0

    stock.actual_outflow_rate.value = actual_outflow
    return actual_outflow


def _reset_mode_timers(ctrl):
    for timer_name in TIMER_MAP.values():
        getattr(ctrl, timer_name).reset()


def _apply_next_mode(sim):
    next_mode = _next_mode(sim)
    if next_mode is not None:
        sim.controller.active_operating_mode.value = next_mode


def _next_mode(sim):
    ctrl = sim.controller
    name = ctrl.active_operating_mode.value.name
    eps = getattr(ctrl, "stockout_epsilon", 1e-9)
    target_stock = getattr(ctrl, "target_ore_stock_level", 60000.0)

    if _campaign_complete(sim):
        if name == "SHUTDOWN":
            rl_action = getattr(ctrl, "pending_rl_action", None)
            if rl_action is not None:
                ctrl.current_campaign_duration.reset()
                ctrl.pending_rl_action = None
                return ACTION_MODES[rl_action]
            if hasattr(ctrl, "pending_rl_action"):
                raise RequireDecision()
        ctrl.current_campaign_duration.reset()
        return _choose_next_campaign_mode(sim) if name == "SHUTDOWN" else MODES["SHUTDOWN"]

    if name == "SHUTDOWN":
        return None

    ore1 = sim.ore1_mass
    ore2 = sim.ore2_mass

    if "_CONTINGENCY" in name:
        if _contingency_complete(sim):
            ctrl.current_contingency_duration.reset()
            return MODES[name.replace("_CONTINGENCY", "")]
        base = name.replace("_CONTINGENCY", "")
        if base == "MODE_A" and ore1 <= eps:
            return MODES[base + "_MINE_SURGING"]
        if base == "MODE_B" and ore2 <= eps:
            return MODES[base + "_MINE_SURGING"]
        return None

    if "_MINE_SURGING" in name:
        if ctrl.total_system_ore_mass.value <= target_stock + 1e-6:
            return MODES[name.replace("_MINE_SURGING", "")]
        return None

    if name == "MODE_A":
        if ore1 <= eps:
            return MODES[name + "_MINE_SURGING"]
        if ore2 <= eps:
            ctrl.current_contingency_duration.reset()
            return MODES[name + "_CONTINGENCY"]
        return None

    if name == "MODE_B":
        if ore1 <= eps:
            ctrl.current_contingency_duration.reset()
            return MODES[name + "_CONTINGENCY"]
        if ore2 <= eps:
            return MODES[name + "_MINE_SURGING"]
        return None

    return None


def _campaign_complete(sim) -> bool:
    ctrl = sim.controller
    threshold = (
        ctrl.duration_of_shutdowns
        if ctrl.active_operating_mode.value.name == "SHUTDOWN"
        else ctrl.duration_of_production_campaigns
    )
    ctrl.current_campaign_duration.upper_threshold = threshold
    return ctrl.current_campaign_duration.value >= (threshold - 1e-6)


def _contingency_complete(sim) -> bool:
    ctrl = sim.controller
    threshold = ctrl.duration_of_contingency_segments
    ctrl.current_contingency_duration.upper_threshold = threshold
    return ctrl.current_contingency_duration.value >= (threshold - 1e-6)


def _choose_next_campaign_mode(sim):
    ctrl = sim.controller
    ore2 = sim.ore2_mass
    total_stock = ctrl.total_system_ore_mass.value
    EPS = 1e-6
    if ore2 > ctrl.critical_ore2_level:
        return (
            MODES["MODE_A"]
            if total_stock <= ctrl.target_ore_stock_level + EPS
            else MODES["MODE_A_MINE_SURGING"]
        )
    return (
        MODES["MODE_B"]
        if total_stock <= ctrl.target_ore_stock_level + EPS
        else MODES["MODE_B_MINE_SURGING"]
    )


def _update_mode_timers(sim):
    ctrl = sim.controller
    m = ctrl.active_operating_mode.value.name
    for timer_name in TIMER_MAP.values():
        getattr(ctrl, timer_name).rate = 0.0
    timer_attr = TIMER_MAP.get(m)
    if timer_attr:
        getattr(ctrl, timer_attr).rate = 1.0
    ctrl.current_campaign_duration.rate = 1.0
    ctrl.current_campaign_duration.upper_threshold = (
        ctrl.duration_of_shutdowns
        if m == "SHUTDOWN"
        else ctrl.duration_of_production_campaigns
    )
    if m in CONTINGENCY_MODES:
        ctrl.current_contingency_duration.rate = 1.0
        ctrl.current_contingency_duration.upper_threshold = (
            ctrl.duration_of_contingency_segments
        )
    else:
        ctrl.current_contingency_duration.rate = 0.0


def _target_rates(sim):
    ctrl = sim.controller
    name = ctrl.active_operating_mode.value.name

    ore1, ore2 = _read_rates(name, ctrl)

    if "_MINE_SURGING" in name:
        target_stock = getattr(ctrl, "target_ore_stock_level", 60000.0)
        ctrl.total_system_ore_mass.lower_threshold = target_stock
        p = sim.stockpile2_routing_fraction
        if (
            p <= 1e-4
            and getattr(sim, "mine", None) is not None
            and hasattr(sim.mine, "_get_current_attr_value")
        ):
            p = sim.mine._get_current_attr_value()
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
"""Top-level control policy for the mining blending simulation.

The policy is a pure function of the simulation state, decoupled from the DRS
component physics. Operating-mode bookkeeping lives on the controller
(``controller.step_mode``), extraction-rate mechanics on the mine faces
(``mine.set_extraction_rate``), and stockpile flow balancing on the stockpiles
(``stockpile.set_inout``). This policy owns the target-rate calculations,
routing, and calling those component methods in the right order.
"""

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
    ctrl.step_mode(sim.ore1_stock, sim.ore2_stock)
    _target_rates(sim)

    if sim.mine is not None:
        sim.mine.set_extraction_rate(ctrl.target_mine_mass_rate.value)


def _multi_face_step(sim):
    ctrl = sim.controller
    ctrl.total_extra_trucks.value = 0.0

    mode_name = ctrl.step_mode(sim.ore1_stock, sim.ore2_stock)

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


def _mine_sources(sim):
    ctrl = sim.controller
    if getattr(ctrl, "faces", None):
        return ctrl.faces
    if getattr(sim, "mine", None) is not None:
        return [sim.mine]
    return []


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
    out1 = sim.ore1_stock.set_inout(
        ore1_rate, ctrl.target_stock1_outflow_rate.value, attr_inflow=1.0
    )
    out2 = sim.ore2_stock.set_inout(
        ore2_rate, ctrl.target_stock2_outflow_rate.value, attr_inflow=0.0
    )
    if sim.plant is not None:
        sim.plant.target_rate = out1 + out2
        sim.plant.cumulative_milled_mass.rate = sim.plant.actual_rate


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
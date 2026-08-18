"""Top-level control policy for the mining blending simulation.

The policy is a thin orchestrator of the component APIs: operating-mode
bookkeeping lives on the controller (``update_mode`` / ``get_target_rates``),
parcel extraction mechanics on the mine faces (the policy only sets
``target_rate``; parcels are advanced in ``step``), routing on the fleet
(``fleet.route``), flow balancing on the stockpiles (``feed_and_draw``), and
draw-down on the plant (``process``).
"""


def step_policy(sim, time):
    """Top-level control policy invoked by the engine once per step."""
    sim.global_time.rate = 1.0
    ctrl = sim.controller

    mode = ctrl.update_mode(sim.ore1_stock, sim.ore2_stock)
    mine_target, stock1_target, stock2_target = ctrl.get_target_rates(
        mode, sim.fleet
    )

    if _is_multi_face(sim):
        ctrl.schedule_fleet_shifts(mode)
        ctrl.drive_faces(mine_target)
        sources = ctrl.faces
    else:
        if sim.mine is not None:
            sim.mine.target_rate = mine_target
        sources = [sim.mine]

    ore1_in, ore2_in = sim.fleet.route(sources=sources)
    out1 = sim.ore1_stock.feed_and_draw(ore1_in, stock1_target)
    out2 = sim.ore2_stock.feed_and_draw(ore2_in, stock2_target)

    if sim.plant is not None:
        sim.plant.process(out1 + out2)


def _is_multi_face(sim) -> bool:
    return bool(getattr(sim.controller, "faces", None))
from drs_mining.components import (
    ActiveFleetConcentratorModel,
    AreaReadinessTarget,
    ConcentratorConfig,
)
from drs_mining.components.modes import MODES
from drs_mining.components.planning import MiningPriority


def test_area2_deadline_miss_persists_after_late_completion():
    config = ConcentratorConfig(
        ore_to_be_extracted_during_warming_period=0.0,
        area2_readiness_target=AreaReadinessTarget(
            required_development=4000.0,
            ready_by_day=10.0,
        ),
    )
    sim = ActiveFleetConcentratorModel(config)
    controller = sim.controller

    controller.strategic_planning_started.value = True
    controller.strategic_year_timer.value = 20.0
    controller.area2_cumulative_development.value = 4000.0
    controller._update_area2_readiness()

    assert controller.area2_ready.value is True
    assert controller.area2_ready_day.value == 20.0
    assert controller.area2_deadline_missed.value is True
    assert controller.area2_currently_late.value is False
    assert controller.area2_completed_late.value is True

    controller.strategic_year_timer.value = 30.0
    controller._update_area2_readiness()

    assert controller.area2_deadline_missed.value is True
    assert controller.area2_completed_late.value is True


def test_b_surging_recomputes_extraction_against_available_face_blend():
    config = ConcentratorConfig(
        ore_to_be_extracted_during_warming_period=0.0,
        area2_readiness_target=AreaReadinessTarget(
            required_development=4000.0,
            ready_by_day=365.0,
        ),
    )
    sim = ActiveFleetConcentratorModel(config)
    controller = sim.controller

    controller.strategic_planning_started.value = True
    controller.area2_ready.value = False
    controller.active_operating_mode.value = MODES["MODE_B_MINE_SURGING"]
    controller._update_development_truck_reservation()
    controller._reallocate_fleet_for_shift()

    original = controller.active_operating_mode.value.get_target_rates(sim)
    adjusted = controller._targets_constrained_for_available_blend(original)

    assert controller.current_shift_allocations == [1.0, 0.0]
    assert round(controller.achievable_ore2_fraction.value, 2) == 0.15
    assert adjusted.extraction_rate == (
        original.ore2_milling_rate / controller.achievable_ore2_fraction.value
    )
    assert controller.constrained_mode_active.value is True
    assert controller.mode_blend_feasible.value is False


def test_development_priority_reserves_production_trucks_for_development():
    config = ConcentratorConfig(
        ore_to_be_extracted_during_warming_period=0.0,
        area2_readiness_target=AreaReadinessTarget(),
    )
    sim = ActiveFleetConcentratorModel(config)
    controller = sim.controller

    controller.strategic_planning_started.value = True
    controller.mining_priority.value = MiningPriority.DEVELOPMENT
    controller._update_development_truck_reservation()
    controller._reallocate_fleet_for_shift()

    assert controller.development_priority_reserved_trucks.value == 2.0
    assert sum(v.value for v in controller.face_truck_allocations) == 8.0


def test_area2_counterfactual_disables_development_truck_accounting():
    config = ConcentratorConfig(
        ore_to_be_extracted_during_warming_period=0.0,
        area2_counterfactual_disable=True,
        area2_readiness_target=AreaReadinessTarget(
            required_development=4000.0,
            ready_by_day=365.0,
        ),
    )
    sim = ActiveFleetConcentratorModel(config)
    controller = sim.controller

    controller.strategic_planning_started.value = True
    controller.mining_priority.value = MiningPriority.DEVELOPMENT
    controller.active_operating_mode.value = MODES["MODE_B_MINE_SURGING"]
    controller._update_development_truck_reservation()
    controller._reallocate_fleet_for_shift()

    assert controller.development_priority_reserved_trucks.value == 0.0

    for i, rate in enumerate(controller.face_target_extraction_rates):
        rate.value = 1.0
        controller._face_real_extraction_rate(i, target_extraction_rate=1.0)

    assert controller.total_extra_trucks.value == 0.0

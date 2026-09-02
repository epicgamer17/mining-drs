"""Integration tests for Area 2 readiness tracking and physical unlock."""

import pytest
from drs_mining.components.planning import AreaReadinessTarget


def test_area2_readiness_and_physical_unlock():
    from examples.two_area_readiness.simulation import TwoAreaReadinessSimulation

    sim = TwoAreaReadinessSimulation(
        num_trucks=18,
        area2_readiness_target=AreaReadinessTarget(
            required_development=4000.0,
            ready_by_day=365.0,
        ),
        seed=42,
    )
    sim.strategic_planning_started = True

    # Initially locked
    assert sim.is_area2_locked() is True
    assert sim.area2_ready is False
    assert sim.area2_readiness_fraction.value == 0.0

    # Advance development past required threshold
    sim.area2_cumulative_development.value = 4000.0
    sim.strategic_year_timer.value = 120.0
    sim._update_area2_readiness()

    # Now unlocked on time
    assert sim.area2_ready is True
    assert sim.is_area2_locked() is False
    assert sim.area2_readiness_fraction.value == 1.0
    assert sim.area2_ready_day.value == 120.0
    assert sim.area2_deadline_missed is False
    assert sim.area2_completed_late is False


def test_area2_deadline_miss_persists_after_late_completion():
    from examples.two_area_readiness.simulation import TwoAreaReadinessSimulation

    sim = TwoAreaReadinessSimulation(
        num_trucks=18,
        area2_readiness_target=AreaReadinessTarget(
            required_development=4000.0,
            ready_by_day=100.0,
        ),
        seed=42,
    )
    sim.strategic_planning_started = True

    # Day 120: Deadline passed (100d) and development is incomplete
    sim.strategic_year_timer.value = 120.0
    sim.area2_cumulative_development.value = 3000.0
    sim._update_area2_readiness()

    assert sim.area2_ready is False
    assert sim.area2_deadline_missed is True
    assert sim.area2_currently_late is True

    # Day 150: Completed late
    sim.strategic_year_timer.value = 150.0
    sim.area2_cumulative_development.value = 4000.0
    sim._update_area2_readiness()

    assert sim.area2_ready is True
    assert sim.area2_ready_day.value == 150.0
    assert sim.area2_deadline_missed is True
    assert sim.area2_currently_late is False
    assert sim.area2_completed_late is True

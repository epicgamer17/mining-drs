"""Unit tests for tactical planning components."""

import pytest
from drs_mining.components.planning import (
    AreaReadinessTarget,
    select_fleet_mode,
)
from drs_mining.config import FLEET_MODES, MILL_MODES
from drs_mining.components.modes import OperatingMode


def test_select_fleet_mode():
    # All on track -> PRODUCTION
    mode = select_fleet_mode(development_ratio=1.0, ore1_ratio=1.0, ore2_ratio=1.0, tolerance=0.90)
    assert mode == FLEET_MODES["PRODUCTION"]
    assert mode.name == "PRODUCTION"
    assert mode.category == "fleet"

    # Development lagging -> DEVELOPMENT
    mode_dev = select_fleet_mode(development_ratio=0.75, ore1_ratio=1.0, ore2_ratio=1.0, tolerance=0.90)
    assert mode_dev == FLEET_MODES["DEVELOPMENT"]
    assert mode_dev.category == "fleet"

    # Ore 2 lagging -> PRODUCTION
    mode_prod = select_fleet_mode(development_ratio=1.0, ore1_ratio=1.0, ore2_ratio=0.70, tolerance=0.90)
    assert mode_prod == FLEET_MODES["PRODUCTION"]
    assert mode_prod.name == "PRODUCTION"


def test_operating_mode_separation_and_custom_creation():
    # Mill vs Fleet separation
    assert MILL_MODES["MODE_A"].category == "mill"
    assert FLEET_MODES["PRODUCTION"].category == "fleet"
    assert FLEET_MODES["PRODUCTION"] != MILL_MODES["MODE_A"]

    # Dynamic Custom OperatingMode creation
    custom_mode = OperatingMode("HIGH_EFFICIENCY_DEV", category="fleet", boost=1.5)
    assert custom_mode.name == "HIGH_EFFICIENCY_DEV"
    assert custom_mode.category == "fleet"
    assert custom_mode.metadata["boost"] == 1.5


def test_tactical_review_controller_lifecycle():
    from drs_mining.components.planning import TacticalReviewController

    controller = TacticalReviewController(
        tactical_review_period_days=30.0,
        tactical_progress_tolerance=0.90,
    )

    assert not controller.planning_started
    controller.start_planning()
    assert controller.planning_started

    # First update triggers immediate review (review_count == 0)
    mode = controller.update_mode(area2_readiness_trajectory_ratio=1.0)
    assert mode == FLEET_MODES["PRODUCTION"]
    assert controller.tactical_review_count.value == 1.0

    # Not yet time for next review
    controller.step_timers(15.0)
    mode2 = controller.update_mode(area2_readiness_trajectory_ratio=1.0)
    assert controller.tactical_review_count.value == 1.0

    # Now past review period
    controller.step_timers(20.0)
    mode3 = controller.update_mode(area2_readiness_trajectory_ratio=0.5)
    assert controller.tactical_review_count.value == 2.0
    # Area2 ratio 0.5 -> DEVELOPMENT mode
    assert mode3 == FLEET_MODES["DEVELOPMENT"]

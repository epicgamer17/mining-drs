import pytest
from drs_mining.components.planning import (
    AreaReadinessTarget,
    select_fleet_mode,
    TacticalReviewController,
)
from drs_mining.config import FLEET_MODES


def test_area_readiness_target():
    target = AreaReadinessTarget(required_development=500.0, ready_by_day=100.0)
    assert target.required_development == 500.0
    assert target.ready_by_day == 100.0


def test_select_fleet_mode():
    # When everything is on schedule (ratios = 1.0), default to PRODUCTION
    mode = select_fleet_mode(
        development_ratio=1.0,
        ore1_ratio=1.0,
        ore2_ratio=1.0,
        tolerance=0.90,
    )
    assert mode == FLEET_MODES["PRODUCTION"]

    # When development is delayed (ratio = 0.50 vs tolerance 0.90)
    mode_dev = select_fleet_mode(
        development_ratio=0.50,
        ore1_ratio=1.0,
        ore2_ratio=1.0,
        tolerance=0.90,
    )
    assert mode_dev == FLEET_MODES["DEVELOPMENT"]


def test_tactical_review_controller():
    ctrl = TacticalReviewController(
        tactical_review_period_days=30.0,
        tactical_progress_tolerance=0.90,
    )
    assert ctrl.fleet_mode == FLEET_MODES["PRODUCTION"]

    ctrl.start_planning()
    # Trigger review with development lag
    mode = ctrl.update_mode(
        area2_readiness_trajectory_ratio=0.50,
    )
    assert mode == FLEET_MODES["DEVELOPMENT"]
    assert ctrl.fleet_mode == FLEET_MODES["DEVELOPMENT"]

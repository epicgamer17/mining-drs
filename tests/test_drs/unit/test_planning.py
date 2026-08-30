"""Unit tests for strategic and tactical planning components."""

import pytest
from drs_mining.components.planning import (
    AreaReadinessTarget,
    StrategicYearTarget,
    strategic_target_for_year,
    trajectory_progress_ratio,
    select_fleet_mode,
)
from drs_mining.config import FLEET_MODES, MILL_MODES
from drs_mining.components.modes import OperatingMode


def test_strategic_target_for_year():
    targets = [
        StrategicYearTarget(min_development=1000.0, min_ore1_production=50000.0),
        StrategicYearTarget(min_development=2000.0, min_ore1_production=60000.0),
    ]
    # Year 0
    t0 = strategic_target_for_year(targets, 0)
    assert t0.min_development == 1000.0
    # Year 1
    t1 = strategic_target_for_year(targets, 1)
    assert t1.min_development == 2000.0
    # Beyond list length -> reuses last target
    t2 = strategic_target_for_year(targets, 5)
    assert t2.min_development == 2000.0
    # Empty list -> returns default
    t_empty = strategic_target_for_year([], 0)
    assert t_empty.min_development == 0.0


def test_trajectory_progress_ratio():
    # Inactive target (<= 0)
    assert trajectory_progress_ratio(actual=500.0, annual_target=0.0, elapsed_fraction=0.5) == 1.0

    # On track (expected 500, actual 500)
    assert pytest.approx(trajectory_progress_ratio(actual=500.0, annual_target=1000.0, elapsed_fraction=0.5)) == 1.0

    # Ahead of track (expected 500, actual 600)
    assert pytest.approx(trajectory_progress_ratio(actual=600.0, annual_target=1000.0, elapsed_fraction=0.5)) == 1.2

    # Behind track (expected 500, actual 400)
    assert pytest.approx(trajectory_progress_ratio(actual=400.0, annual_target=1000.0, elapsed_fraction=0.5)) == 0.8


def test_select_fleet_mode():
    # All on track -> BALANCED
    mode = select_fleet_mode(development_ratio=1.0, ore1_ratio=1.0, ore2_ratio=1.0, tolerance=0.90)
    assert mode == FLEET_MODES["BALANCED"]
    assert mode.name == "BALANCED"
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



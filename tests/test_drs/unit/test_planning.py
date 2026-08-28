"""Unit tests for strategic and tactical planning components."""

import pytest
from drs_mining.components.planning import (
    MiningPriority,
    AreaReadinessTarget,
    StrategicYearTarget,
    strategic_target_for_year,
    trajectory_progress_ratio,
    select_mining_priority,
)


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


def test_select_mining_priority():
    # All on track -> BALANCED
    prio = select_mining_priority(development_ratio=1.0, ore1_ratio=1.0, ore2_ratio=1.0, tolerance=0.90)
    assert prio == MiningPriority.BALANCED

    # Development lagging -> DEVELOPMENT
    prio_dev = select_mining_priority(development_ratio=0.75, ore1_ratio=1.0, ore2_ratio=1.0, tolerance=0.90)
    assert prio_dev == MiningPriority.DEVELOPMENT

    # Ore 2 lagging -> PRODUCTION
    prio_prod = select_mining_priority(development_ratio=1.0, ore1_ratio=1.0, ore2_ratio=0.70, tolerance=0.90)
    assert prio_prod == MiningPriority.PRODUCTION

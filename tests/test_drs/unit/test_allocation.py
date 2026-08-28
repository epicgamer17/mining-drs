"""Unit tests for Analytical Face Allocation Equations (Slide 29)."""

import pytest
from drs_mining.components.allocation import (
    FaceAllocationResult,
    solve_face_allocation_rates,
)


def test_slide_29_mode_a_allocation():
    """Slide 29: Mode A (Ore1: 3600, Ore2: 2400) -> Face1 weight = 0.917, Face2 weight = 0.083."""
    res = solve_face_allocation_rates(
        target_ore1_rate=3600.0,
        target_ore2_rate=2400.0,
        face1_ore1_fraction=0.85,
        face2_ore1_fraction=0.55,
    )
    assert res.is_feasible is True
    assert round(res.face1_rate, 1) == 1000.0
    assert round(res.face2_rate, 1) == 5000.0
    assert round(res.face1_weight, 3) == round(1000.0 / 6000.0, 3)
    assert round(res.face2_weight, 3) == round(5000.0 / 6000.0, 3)


def test_mode_b_allocation():
    """Mode B (Ore1: 4600, Ore2: 800, Total: 5400) -> 85.18% Ore 1 target."""
    res = solve_face_allocation_rates(
        target_ore1_rate=4600.0,
        target_ore2_rate=800.0,
        face1_ore1_fraction=0.85,
        face2_ore1_fraction=0.55,
    )
    assert res.face1_rate == 5400.0
    assert res.face2_rate == 0.0
    assert res.face1_weight == 1.0
    assert res.face2_weight == 0.0
    assert res.is_feasible is False  # Target requires 85.18% Ore 1, max face1 is 85%


def test_contingency_allocation():
    """Mode A Contingency (Ore1: 3900, Ore2: 0) -> Pure Face 1."""
    res = solve_face_allocation_rates(
        target_ore1_rate=3900.0,
        target_ore2_rate=0.0,
        face1_ore1_fraction=0.85,
        face2_ore1_fraction=0.55,
    )
    assert res.face1_rate == 3900.0
    assert res.face2_rate == 0.0
    assert res.face1_weight == 1.0
    assert res.face2_weight == 0.0


def test_zero_target_allocation():
    """Test shutdown/zero rate allocation."""
    res = solve_face_allocation_rates(
        target_ore1_rate=0.0,
        target_ore2_rate=0.0,
    )
    assert res.face1_rate == 0.0
    assert res.face2_rate == 0.0
    assert res.face1_weight == 0.50
    assert res.face2_weight == 0.50
    assert res.is_feasible is True


def test_identical_face_grades():
    """Test singular case when both faces have identical grade."""
    res = solve_face_allocation_rates(
        target_ore1_rate=3000.0,
        target_ore2_rate=3000.0,
        face1_ore1_fraction=0.50,
        face2_ore1_fraction=0.50,
    )
    assert res.face1_rate == 3000.0
    assert res.face2_rate == 3000.0
    assert res.is_feasible is True

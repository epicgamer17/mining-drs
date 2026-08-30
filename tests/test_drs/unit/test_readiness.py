"""Unit tests for MineFace development advance and physical readiness mechanics."""

import pytest
from drs_mining.components.mine_face import MineFace


def test_mine_face_readiness_advancement_and_unlock():
    unlocked = False

    def on_unlock():
        nonlocal unlocked
        unlocked = True

    face = MineFace(
        name="test_face_2",
        face_id=2,
        required_development=1000.0,
        ready_by_day=100.0,
        on_unlock_callback=on_unlock,
    )

    assert not face.is_ready
    assert face.ready_day.value == -1.0
    assert face.is_locked()

    # Advance 500m on day 50 -> 50% ready, on track
    just_unlocked = face.advance_development(delta_meters=500.0, current_day=50.0)
    assert not just_unlocked
    assert not face.is_ready
    assert not unlocked
    assert face.readiness_fraction.value == 0.50
    assert pytest.approx(face.readiness_trajectory_ratio.value) == 1.0

    # Advance another 500m on day 80 -> Unlocks on time!
    just_unlocked = face.advance_development(delta_meters=500.0, current_day=80.0)
    assert just_unlocked
    assert face.is_ready
    assert not face.is_locked()
    assert unlocked
    assert face.ready_day.value == 80.0
    assert not face.deadline_missed
    assert not face.completed_late


def test_mine_face_readiness_late_deadline():
    face = MineFace(
        name="test_face_2",
        face_id=2,
        required_development=1000.0,
        ready_by_day=100.0,
    )

    # Day 120, only 800m -> Late!
    face.advance_development(delta_meters=800.0, current_day=120.0)
    assert not face.is_ready
    assert face.deadline_missed
    assert face.currently_late

    # Day 130, complete remaining 200m -> Unlocks, but completed late!
    just_unlocked = face.advance_development(delta_meters=200.0, current_day=130.0)
    assert just_unlocked
    assert face.is_ready
    assert face.completed_late
    assert face.ready_day.value == 130.0

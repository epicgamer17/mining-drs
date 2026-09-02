"""Unit tests for MineFace development advance and readiness mechanics."""

import pytest
from drs_mining.components.mine_face import MineFace


def test_mine_face_readiness_advancement_and_unlock():
    unlocked = False

    def on_unlock():
        nonlocal unlocked
        unlocked = True

    face = MineFace(
        name="test_face",
        face_id=2,
        required_development=1000.0,
        on_unlock_callback=on_unlock,
    )

    assert not face.is_ready
    assert face.ready_day.value == -1.0
    assert face.is_locked()

    # Advance 500m -> 50% ready
    just_unlocked = face.advance_development(delta_meters=500.0)
    assert not just_unlocked
    assert not face.is_ready
    assert not unlocked
    assert face.readiness_fraction.value == 0.50

    # Advance another 500m -> Unlocks!
    just_unlocked = face.advance_development(delta_meters=500.0)
    assert just_unlocked
    assert face.is_ready
    assert not face.is_locked()
    assert unlocked


def test_face_no_development_required():
    face = MineFace(name="test_face", face_id=1, required_development=0.0)
    assert face.is_ready
    assert not face.is_locked()


def test_face_counterfactual_disable():
    face = MineFace(
        name="test_face",
        face_id=1,
        required_development=0.0,
        counterfactual_disable=True,
    )
    assert not face.is_ready
    assert face.is_locked()

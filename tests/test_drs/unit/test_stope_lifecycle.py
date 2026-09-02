"""Unit tests for MineFace parcel lifecycle."""

import pytest
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.mine_face import MineFace, FaceState


def test_mine_face_initialization():
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.02)
    face = MineFace(
        name="test_face",
        face_id=1,
        area_id=1,
        level_index=3,
        generator=gen,
        mean_ore_fraction=0.30,
        std_dev_ore_fraction=0.02,
        total_ore_to_extract=100000.0,
        min_ore_mass=30000.0,
        max_ore_mass=30000.0,
    )

    assert face.state == FaceState.ORE_READY
    assert face.is_ore_available is True
    assert face.is_exhausted is False
    assert face.remaining_reserve == 100000.0


def test_face_depletion_transitions():
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.02)
    face = MineFace(
        name="face_1",
        face_id=1,
        area_id=1,
        level_index=3,
        generator=gen,
        mean_ore_fraction=0.30,
        std_dev_ore_fraction=0.02,
        total_ore_to_extract=60000.0,
        min_ore_mass=30000.0,
        max_ore_mass=30000.0,
    )

    # First parcel: 30000t
    assert face.is_ore_available is True
    assert face.remaining_reserve == 60000.0

    # Simulate extraction by advancing parcel state when parcel is depleted
    face.parcel_extracted_mass.value = 30000.0
    face.cumulative_extracted_mass.value = 30000.0
    face.advance_parcel_state()

    # Should have loaded next parcel
    assert face.state == FaceState.ORE_READY
    assert face.is_ore_available is True
    assert face.remaining_reserve == 30000.0

    # Second parcel: exhaust remaining
    face.parcel_extracted_mass.value = 30000.0
    face.cumulative_extracted_mass.value = 60000.0
    face.advance_parcel_state()

    assert face.is_exhausted is True

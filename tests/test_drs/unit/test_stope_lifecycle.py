"""Unit tests for MineFace multi-phase stope lifecycle and TwoTierHierarchicalDispatchController."""

import pytest
from drs_mining.components.generators import StochasticFaciesGenerator
from drs_mining.components.mine_face import MineFace, FaceState
from drs_mining.components.dispatch import TwoTierHierarchicalDispatchController


def test_stope_face_initialization():
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.02)
    stope = MineFace(
        name="test_stope",
        face_id=1,
        area_id=1,
        level_index=3,
        generator=gen,
        mean_ore_fraction=0.30,
        std_dev_ore_fraction=0.02,
        total_stope_reserve=100000.0,
        min_parcel_ore_mass=30000.0,
        max_parcel_ore_mass=30000.0,
        turnaround_dev_per_parcel_m=30.0,
    )

    assert stope.state == FaceState.ORE_READY
    assert stope.is_ore_available is True
    assert stope.is_in_turnaround is False
    assert stope.is_exhausted is False
    assert stope.remaining_reserve == 100000.0
    assert stope.remaining_parcel_ore == 30000.0


def test_stope_face_extraction_and_turnaround_cycle():
    gen = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.02)
    stope = MineFace(
        name="stope_1",
        face_id=1,
        area_id=1,
        level_index=3,
        generator=gen,
        mean_ore_fraction=0.30,
        std_dev_ore_fraction=0.02,
        total_stope_reserve=60000.0,
        min_parcel_ore_mass=30000.0,
        max_parcel_ore_mass=30000.0,
        turnaround_dev_per_parcel_m=20.0,
    )

    # 1. Extract partial ore (10,000 t)
    ext, o1, o2 = stope.extract_ore(10000.0)
    assert ext == 10000.0
    assert stope.state == FaceState.ORE_READY
    assert stope.remaining_parcel_ore == 20000.0

    # 2. Extract remaining 20,000 t in parcel
    ext2, _, _ = stope.extract_ore(20000.0)
    assert ext2 == 20000.0
    # Parcel ore exhausted -> stope enters DEVELOPMENT_TURNAROUND
    assert stope.state == FaceState.DEVELOPMENT_TURNAROUND
    assert stope.is_ore_available is False
    assert stope.is_in_turnaround is True
    assert stope.remaining_turnaround_dev == 20.0

    # 3. Advance turnaround development
    dev_done, is_complete = stope.advance_turnaround_development(10.0)
    assert dev_done == 10.0
    assert is_complete is False
    assert stope.state == FaceState.DEVELOPMENT_TURNAROUND

    dev_done2, is_complete2 = stope.advance_turnaround_development(10.0)
    assert dev_done2 == 10.0
    assert is_complete2 is True
    # Turnaround complete -> transitions back to ORE_READY (Parcel 2)
    assert stope.state == FaceState.ORE_READY
    assert stope.is_ore_available is True
    assert stope.remaining_reserve == 30000.0

    # 4. Extract last parcel -> stope should become EXHAUSTED
    stope.extract_ore(30000.0)
    assert stope.is_exhausted is True
    assert stope.state == FaceState.EXHAUSTED


def test_two_tier_hierarchical_dispatch():
    gen1 = StochasticFaciesGenerator(mean_fraction=0.30, std_dev=0.02)
    gen2 = StochasticFaciesGenerator(mean_fraction=0.35, std_dev=0.02)

    s1 = MineFace(
        name="stope_1a", face_id=1, area_id=1, level_index=3, generator=gen1, mean_ore_fraction=0.30, std_dev_ore_fraction=0.02
    )
    s2 = MineFace(
        name="stope_2a", face_id=2, area_id=2, level_index=6, generator=gen2, mean_ore_fraction=0.35, std_dev_ore_fraction=0.02
    )


    disp = TwoTierHierarchicalDispatchController(stopes=[s1, s2], target_daily_ore_tonnes=6000.0)

    # When Area 2 is locked, dispatch must select Area 1
    selected, is_fallback = disp.select_stope_for_truck(
        current_total_stock=50000.0,
        daily_hauled_so_far=1000.0,
        day_progress_fraction=0.50,
        analytical_w2=0.83,
        area2_locked=True,
    )
    assert selected == s1
    assert is_fallback is False

    # When Area 2 is unlocked and w2=1.0, dispatch selects Area 2
    selected2, is_fallback2 = disp.select_stope_for_truck(
        current_total_stock=50000.0,
        daily_hauled_so_far=1000.0,
        day_progress_fraction=0.50,
        analytical_w2=1.0,
        area2_locked=False,
    )
    assert selected2 == s2
    assert is_fallback2 is False

    # If Area 2 enters turnaround, dispatch falls back to Area 1 (Tier 3 fallback)
    s2.state = FaceState.DEVELOPMENT_TURNAROUND
    selected3, is_fallback3 = disp.select_stope_for_truck(
        current_total_stock=50000.0,
        daily_hauled_so_far=1000.0,
        day_progress_fraction=0.50,
        analytical_w2=1.0,
        area2_locked=False,
    )
    assert selected3 == s1
    assert is_fallback3 is True

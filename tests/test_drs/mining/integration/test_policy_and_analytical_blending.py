"""Integration tests for Policy Comparison and Analytical Face Allocation."""

import pytest
from drs_mining.components.planning import StrategicYearTarget, AreaReadinessTarget
from examples.two_area_analytical_blending.simulation import (
    TwoAreaAnalyticalBlendingSimulation,
    run_two_area_analytical_simulation,
)
from examples.two_area_full_hierarchy.simulation import (
    TwoAreaFullHierarchyEngine,
    run_full_hierarchy_study,
)
from examples.two_area_policy_comparison.simulation import (
    TwoAreaPolicySimulationEngine,
    run_policy_comparison_study,
)


def test_analytical_blending_simulation_short():
    """Verify that analytical blending simulation executes and computes weights properly."""
    sim, df = run_two_area_analytical_simulation(
        total_days=60.0,
        warmup_ore=0.0,
        area2_required_dev=500.0,
        plot=False,
    )
    assert not df.empty
    assert "analytical_face1_weight" in df.columns
    assert "analytical_face2_weight" in df.columns
    assert "operating_npv_proxy" in df.columns
    # Check that weights stay in [0, 1]
    assert (df["analytical_face1_weight"] >= 0.0).all()
    assert (df["analytical_face1_weight"] <= 1.0).all()


def test_policy_comparison_study_short():
    """Verify that Policy 2 achieves higher/timely development than Policy 1."""
    df_p1, df_p2 = run_policy_comparison_study(
        total_days=90.0,
        warmup_ore=0.0,
        area2_required_dev=800.0,
        plot=False,
    )
    assert not df_p1.empty
    assert not df_p2.empty

    final_p1_dev = df_p1["area2_cumulative_development"].iloc[-1]
    final_p2_dev = df_p2["area2_cumulative_development"].iloc[-1]

    # Policy 2 proactively prioritizes development when needed
    assert final_p2_dev > final_p1_dev


def test_full_hierarchy_study_short():
    """Verify full three-level hierarchy executes with dual development and analytical blending."""
    df_p1, df_p2 = run_full_hierarchy_study(
        total_days=90.0,
        warmup_ore=0.0,
        area2_required_dev=800.0,
        plot=False,
    )
    assert not df_p1.empty
    assert not df_p2.empty
    assert "sustaining_cumulative_development" in df_p1.columns
    assert "area2_cumulative_development" in df_p1.columns
    assert "analytical_face1_weight" in df_p2.columns

    final_p1_cap = df_p1["area2_cumulative_development"].iloc[-1]
    final_p2_cap = df_p2["area2_cumulative_development"].iloc[-1]
    assert final_p2_cap > final_p1_cap


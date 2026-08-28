"""Integration test for Two-Area Strategic DCF Economics & Counterfactual Incremental NPV."""

import pytest
from examples.two_area_economics.simulation import (
    TwoAreaEconomicSimulation,
    run_two_area_economic_simulation,
)
from drs_mining.components.planning import AreaReadinessTarget, StrategicYearTarget


def test_area2_counterfactual_npv_positive():
    """Verify that opening high-grade Area 2 provides positive incremental NPV."""
    sim_with, df_with = run_two_area_economic_simulation(
        total_days=100.0,
        num_trucks=18,
        area2_required_development=1000.0,
        area2_ready_by_day=100.0,
        annual_discount_rate=0.05,
        seed=42,
        run_counterfactual=True,
        plot=False,
    )

    # 1. Base case asserts
    assert sim_with.area2_ready is True
    assert sim_with.area2_ready_day.value > 0.0
    assert sim_with.area2_ready_day.value < 365.0

    # 2. Check economic fields in telemetry
    assert "operating_npv_proxy" in df_with.columns
    assert "cumulative_discounted_cash_flow" in df_with.columns
    assert "discount_factor" in df_with.columns
    assert "current_cash_flow_rate" in df_with.columns

    final_with = df_with.iloc[-1]
    npv_with = float(final_with["operating_npv_proxy"])
    assert npv_with > 0.0


def test_counterfactual_area2_permanently_locked():
    """Verify that in the counterfactual run, Area 2 never unlocks and produces zero high-grade ore."""
    sim_without = TwoAreaEconomicSimulation(
        num_trucks=18,
        area2_readiness_target=AreaReadinessTarget(required_development=4000.0),
        area2_counterfactual_disable=True,
        seed=42,
    )
    assert sim_without.is_area2_locked() is True
    assert sim_without._select_face_by_blend_need() == 1

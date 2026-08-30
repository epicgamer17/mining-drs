"""Integration tests for two_area_stope_lifecycle simulation."""

import pytest
from examples.two_area_stope_lifecycle.simulation import run_stope_lifecycle_study


def test_stope_lifecycle_short_study():
    """Verifies that multi-stope simulation runs with turnaround development and two-tier dispatch."""
    df_p1, df_p2 = run_stope_lifecycle_study(
        total_days=90.0,
        warmup_ore=0.0,
        area2_required_dev=500.0,
        plot=False,
    )

    assert len(df_p2) > 0
    assert len(df_p1) > 0

    # Policy 2 should unlock Area 2 early with fast capital development
    final_p2 = df_p2.iloc[-1]
    final_p1 = df_p1.iloc[-1]

    assert float(final_p2["area2_ready_day"]) > 0.0
    assert float(final_p2["area2_cumulative_development"]) >= float(final_p1["area2_cumulative_development"])
    assert float(final_p2["stope_turnaround_dev_m"]) > 0.0
    assert abs(
        float(final_p2["cumulative_mine_development"])
        - (float(final_p2["area2_cumulative_development"]) + float(final_p2["stope_turnaround_dev_m"]))
    ) < 1e-3

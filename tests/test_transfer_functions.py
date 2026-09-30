"""Unit tests for non-linear transfer functions and volumetric simulation uncertainty."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from drs_mining.components.transfer_functions import (
    simulation_grade_tonnage_uncertainty,
)
from drs_mining.components.estimation import (
    plot_grade_tonnage_curve,
    grade_tonnage_table,
)


def test_simulation_grade_tonnage_uncertainty_validation():
    # Realizations dimension check
    with pytest.raises(ValueError, match="must be a 2D array"):
        simulation_grade_tonnage_uncertainty(
            realizations=np.zeros((10,)),
            cutoffs=[0.5],
            tonnes=np.ones(10),
        )

    # Empty realizations
    with pytest.raises(ValueError, match="cannot be empty"):
        simulation_grade_tonnage_uncertainty(
            realizations=np.zeros((0, 5)),
            cutoffs=[0.5],
            tonnes=np.ones(0),
        )

    # Empty cutoffs
    with pytest.raises(ValueError, match="cutoffs list cannot be empty"):
        simulation_grade_tonnage_uncertainty(
            realizations=np.ones((10, 5)),
            cutoffs=[],
            tonnes=np.ones(10),
        )

    # Negative cutoffs
    with pytest.raises(ValueError, match="cutoffs cannot be negative"):
        simulation_grade_tonnage_uncertainty(
            realizations=np.ones((10, 5)),
            cutoffs=[-0.5, 1.0],
            tonnes=np.ones(10),
        )

    # Negative tonnes
    with pytest.raises(ValueError, match="tonnes cannot contain negative values"):
        simulation_grade_tonnage_uncertainty(
            realizations=np.ones((10, 5)),
            cutoffs=[0.5],
            tonnes=np.array([-1.0] + [1.0] * 9),
        )

    # Invalid percentiles
    with pytest.raises(ValueError, match="strictly between 0 and 100"):
        simulation_grade_tonnage_uncertainty(
            realizations=np.ones((10, 5)),
            cutoffs=[0.5],
            tonnes=np.ones(10),
            percentiles=(-5.0, 50.0, 95.0),
        )


def test_simulation_grade_tonnage_uncertainty_properties():
    rng = np.random.default_rng(42)
    n_points = 60
    n_real = 25
    realizations = rng.lognormal(mean=0.2, sigma=0.5, size=(n_points, n_real))
    tonnes = np.full(n_points, 1000.0)
    cutoffs = [0.0, 0.5, 1.0, 1.5, 2.0]

    df_gt = simulation_grade_tonnage_uncertainty(
        realizations=realizations,
        cutoffs=cutoffs,
        tonnes=tonnes,
        percentiles=(10.0, 50.0, 90.0),
        grade_unit="% Cu",
    )

    assert list(df_gt.index) == cutoffs
    # Monotonically decreasing ore tonnage with cutoff
    assert np.all(np.diff(df_gt["expected_ore_tonnes"]) <= 0)
    # Monotonically increasing grade with cutoff
    assert np.all(np.diff(df_gt["expected_grade"]) >= -1e-9)

    # Check individual realization tables
    assert len(df_gt.attrs["realization_tables"]) == n_real
    for table in df_gt.attrs["realization_tables"]:
        assert isinstance(table, pd.DataFrame)
        assert "ore_tonnes" in table.columns
        assert "ore_grade" in table.columns
        assert "metal_tonnes" in table.columns
        # Each realization must have monotonically decreasing tonnage with cutoff
        assert np.all(np.diff(table["ore_tonnes"]) <= 0)

    # Verify conservation of total mass at zero cutoff
    expected_total_tonnes = np.sum(tonnes)
    assert np.isclose(df_gt.loc[0.0, "expected_ore_tonnes"], expected_total_tonnes)


def test_plot_grade_tonnage_curve_with_realizations():
    cutoffs = [0.2, 0.5, 0.8, 1.0, 1.5]
    gt_main = pd.DataFrame(
        {
            "ore_tonnes": [1000.0, 800.0, 600.0, 400.0, 100.0],
            "ore_grade": [0.8, 1.1, 1.4, 1.8, 2.3],
            "metal_tonnes": [8.0, 8.8, 8.4, 7.2, 2.3],
        },
        index=cutoffs,
    )

    # 5 realization tables with slightly varying curves
    rng = np.random.default_rng(123)
    r_tables = []
    for _ in range(5):
        noise = rng.uniform(0.9, 1.1, size=len(cutoffs))
        df_r = pd.DataFrame(
            {
                "ore_tonnes": gt_main["ore_tonnes"] * noise,
                "ore_grade": gt_main["ore_grade"] * noise,
                "metal_tonnes": gt_main["metal_tonnes"] * noise,
            },
            index=cutoffs,
        )
        r_tables.append(df_r)

    fig, ax = plot_grade_tonnage_curve(
        gt_data=gt_main,
        grade_unit="% Cu",
        tonnage_unit="kt",
        title="Grade-Tonnage Sensitivity with Realization Envelopes",
        realization_tables=r_tables,
    )
    assert fig is not None
    assert ax is not None
    plt.close(fig)

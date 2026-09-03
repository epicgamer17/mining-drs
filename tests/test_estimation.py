import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drs_mining.components.estimation import (
    polygonal_estimation,
    polygonal_reserve_summary,
    format_reserve_summary,
    grade_tonnage_table,
    plot_polygonal_map,
    inverse_distance_weighting,
    nearest_neighbor_grid_estimation,
    is_within_convex_hull,
    simple_kriging_grid_estimation,
    ordinary_kriging_grid_estimation,
    plot_grade_tonnage_curve,
    _theoretical_covariance,
)


@pytest.fixture
def sample_polygons_df():
    """Sample polygonal reserve table with known values for testing."""
    return pd.DataFrame({
        "hole_id": ["DH01", "DH02", "DH03"],
        "x": [100.0, 200.0, 300.0],
        "y": [100.0, 200.0, 100.0],
        "grade": [1.0, 2.0, 0.5],
        "thickness": [10.0, 10.0, 10.0],
        "area_m2": [1000.0, 2000.0, 1000.0],
        "volume_m3": [10000.0, 20000.0, 10000.0],
        "tonnes": [27000.0, 54000.0, 27000.0],  # 2.7 t/m3 density
        "contained_metal": [27000.0, 108000.0, 13500.0],  # tonnes * grade
        "vertices": [
            [(50, 50), (150, 50), (150, 150), (50, 150)],
            [(150, 150), (250, 150), (250, 250), (150, 250)],
            [(250, 50), (350, 50), (350, 150), (250, 150)],
        ],
    })


def test_polygonal_estimation_execution():
    df_holes = pd.DataFrame({
        "hole_id": ["DH01", "DH02", "DH03"],
        "x": [100.0, 200.0, 300.0],
        "y": [100.0, 200.0, 100.0],
        "grade": [1.2, 0.8, 1.5],
        "thickness": [15.0, 15.0, 15.0],
    })
    boundary = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    polys = polygonal_estimation(
        df_holes,
        boundary=boundary,
        bulk_density=2.7,
        max_radius=100.0,
    )
    assert len(polys) == 3
    assert set(polys["hole_id"]) == {"DH01", "DH02", "DH03"}
    assert (polys["area_m2"] > 0).all()
    assert (polys["tonnes"] > 0).all()
    assert (polys["contained_metal"] > 0).all()
    assert all(len(v) >= 3 for v in polys["vertices"])


def test_polygonal_estimation_clip_to_convex_hull():
    df_holes = pd.DataFrame({
        "hole_id": ["DH01", "DH02", "DH03"],
        "x": [100.0, 300.0, 200.0],
        "y": [100.0, 100.0, 200.0],
        "grade": [1.0, 1.0, 1.0],
        "thickness": [10.0, 10.0, 10.0],
    })
    boundary = [(0.0, 0.0), (500.0, 0.0), (500.0, 500.0), (0.0, 500.0)]

    # 1. Unclipped: polygons expand across entire 500x500 boundary (250,000 m2)
    unclipped = polygonal_estimation(
        df_holes, boundary=boundary, clip_to_convex_hull=False
    )
    assert unclipped["area_m2"].sum() == pytest.approx(250000.0)

    # 2. Clipped to convex hull: total area matches exact triangle area (10,000 m2)
    # Triangle: base=200, height=100 -> area = 0.5 * 200 * 100 = 10,000
    clipped = polygonal_estimation(
        df_holes, boundary=boundary, clip_to_convex_hull=True
    )
    assert clipped["area_m2"].sum() == pytest.approx(10000.0)
    assert (clipped["area_m2"] > 0).all()



def test_polygonal_reserve_summary(sample_polygons_df):
    summary = polygonal_reserve_summary(sample_polygons_df)
    
    assert summary["total_tonnes"] == pytest.approx(108000.0)
    assert summary["total_area_m2"] == pytest.approx(4000.0)
    assert summary["total_volume_m3"] == pytest.approx(40000.0)
    assert summary["drillhole_count"] == 3
    assert summary["mean_polygon_area_m2"] == pytest.approx(4000.0 / 3)

    # Weighted mean grade: (27k*1.0 + 54k*2.0 + 27k*0.5) / 108k = 148,500 / 108,000 = 1.375
    assert summary["contained_metal"] == pytest.approx(148500.0)
    assert summary["mean_grade"] == pytest.approx(1.375)

    # Test text formatting
    formatted = format_reserve_summary(summary)
    assert "108,000.0 tonnes" in formatted
    assert "1.3750 %" in formatted


def test_polygonal_reserve_summary_empty():
    empty_df = pd.DataFrame(columns=["tonnes", "grade", "area_m2", "volume_m3"])
    summary = polygonal_reserve_summary(empty_df)
    assert summary["total_tonnes"] == 0.0
    assert summary["mean_grade"] == 0.0
    assert summary["drillhole_count"] == 0


def test_grade_tonnage_table(sample_polygons_df):
    gt = grade_tonnage_table(sample_polygons_df, cutoffs=[0.0, 0.75, 1.5, 3.0])

    # Cutoff 0.0: all 3 polygons (108k tonnes, 1.375 grade)
    row_0 = gt.loc[0.0]
    assert row_0["ore_tonnes"] == pytest.approx(108000.0)
    assert row_0["ore_grade"] == pytest.approx(1.375)
    assert row_0["waste_tonnes"] == 0.0
    assert row_0["strip_ratio"] == 0.0
    assert row_0["ore_recovery_pct"] == pytest.approx(100.0)
    assert row_0["metal_recovery_pct"] == pytest.approx(100.0)

    # Cutoff 0.75: DH01 (1.0) and DH02 (2.0) -> 81k tonnes
    row_075 = gt.loc[0.75]
    assert row_075["ore_tonnes"] == pytest.approx(81000.0)
    assert row_075["waste_tonnes"] == pytest.approx(27000.0)
    assert row_075["strip_ratio"] == pytest.approx(27000.0 / 81000.0)
    # Ore grade = (27k*1 + 54k*2) / 81k = 135k / 81k = 1.6667
    assert row_075["ore_grade"] == pytest.approx(1.66666667)
    assert row_075["metal_recovery_pct"] == pytest.approx(135000.0 / 148500.0 * 100.0)

    # Cutoff 1.5: only DH02 (54k tonnes, 2.0 grade)
    row_15 = gt.loc[1.5]
    assert row_15["ore_tonnes"] == pytest.approx(54000.0)
    assert row_15["ore_grade"] == pytest.approx(2.0)

    # Cutoff 3.0: exceeds all assays -> 0 ore
    row_30 = gt.loc[3.0]
    assert row_30["ore_tonnes"] == 0.0
    assert row_30["ore_grade"] == 0.0
    assert row_30["waste_tonnes"] == pytest.approx(108000.0)
    assert np.isinf(row_30["strip_ratio"])


def test_plot_polygonal_map(sample_polygons_df):
    boundary = [(0.0, 0.0), (400.0, 0.0), (400.0, 300.0), (0.0, 300.0)]
    fig, ax = plt.subplots()
    ret_ax = plot_polygonal_map(sample_polygons_df, boundary=boundary, ax=ax)
    assert ret_ax is ax
    plt.close(fig)


def test_inverse_distance_weighting_symmetric():
    samples_xy = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    sample_grades = np.array([1.0, 2.0, 3.0, 4.0])
    grid_points = np.array([[5.0, 5.0], [0.0, 0.0], [100.0, 100.0]])

    grades, dists = inverse_distance_weighting(
        samples_xy, sample_grades, grid_points, power=2.0, k_neighbors=4, max_radius=20.0
    )

    # Center point is symmetrically equidistant -> exact arithmetic mean = 2.5
    assert grades[0] == pytest.approx(2.5)
    # Exact collocation -> exact sample grade = 1.0
    assert grades[1] == pytest.approx(1.0)
    # Beyond max_radius=20.0 -> NaN
    assert np.isnan(grades[2])


def test_nearest_neighbor_grid_estimation():
    samples_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    sample_grades = np.array([1.5, 3.5])
    grid_points = np.array([[1.0, 0.0], [9.0, 0.0], [50.0, 50.0]])

    grades, dists = nearest_neighbor_grid_estimation(
        samples_xy, sample_grades, grid_points, max_radius=20.0
    )

    # Point (1, 0) is closest to (0, 0)
    assert grades[0] == pytest.approx(1.5)
    # Point (9, 0) is closest to (10, 0)
    assert grades[1] == pytest.approx(3.5)
    # Point (50, 50) is beyond max_radius=20
    assert np.isnan(grades[2])
    # dists should have 1D shape (M,) for k=1
    assert dists.shape == (3,)


def test_is_within_convex_hull():
    samples_xy = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    grid_points = np.array([
        [5.0, 2.0],    # Centroid of triangle -> True
        [15.0, 15.0],  # Far outside -> False
        [-1.0, 0.0],   # Just outside -> False
    ])
    mask = is_within_convex_hull(samples_xy, grid_points)
    assert mask[0] == True
    assert mask[1] == False
    assert mask[2] == False


def test_idw_mask_extrapolation():
    samples_xy = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    sample_grades = np.array([1.0, 2.0, 3.0])
    # Point at (5, -2) is outside the convex hull, but close (dist=2 < max_radius=10)
    grid_points = np.array([[5.0, 2.0], [5.0, -2.0]])

    # Without masking: both points get estimated
    grades_nomask, _ = inverse_distance_weighting(
        samples_xy, sample_grades, grid_points, max_radius=10.0, mask_extrapolation=False
    )
    assert not np.isnan(grades_nomask[0])
    assert not np.isnan(grades_nomask[1])

    # With masking: outside point gets masked to NaN
    grades_masked, _ = inverse_distance_weighting(
        samples_xy, sample_grades, grid_points, max_radius=10.0, mask_extrapolation=True
    )
    assert not np.isnan(grades_masked[0])
    assert np.isnan(grades_masked[1])


def test_theoretical_covariance():
    h = np.array([0.0, 50.0, 100.0, 150.0])
    # Spherical model: range=100, nugget=0.2, sill=0.8 -> total sill = 1.0
    cov = _theoretical_covariance(h, model="spherical", nugget=0.2, sill=0.8, range_param=100.0)

    # At h=0: Covariance = total sill = 1.0
    assert cov[0] == pytest.approx(1.0)
    # At h=50 (half range): gamma = 0.2 + 0.8*(1.5*0.5 - 0.5*0.125) = 0.2 + 0.8*(0.75 - 0.0625) = 0.2 + 0.55 = 0.75
    # Cov = 1.0 - 0.75 = 0.25
    assert cov[1] == pytest.approx(0.25)
    # At h >= 100 (range): Covariance = 0.0
    assert cov[2] == pytest.approx(0.0)
    assert cov[3] == pytest.approx(0.0)

    # Exponential and Gaussian models run without error
    cov_exp = _theoretical_covariance(h, model="exponential", nugget=0.1, sill=0.9, range_param=100.0)
    assert cov_exp[0] == pytest.approx(1.0)
    assert cov_exp[1] < 1.0

    cov_gau = _theoretical_covariance(h, model="gaussian", nugget=0.1, sill=0.9, range_param=100.0)
    assert cov_gau[0] == pytest.approx(1.0)
    assert cov_gau[1] < 1.0

    with pytest.raises(ValueError, match="Unknown variogram model"):
        _theoretical_covariance(h, model="invalid_model")


def test_simple_kriging_collocation_and_reversion():
    samples_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    sample_grades = np.array([2.0, 4.0])
    mean = 3.0

    grid_points = np.array([
        [0.0, 0.0],     # Exact match with sample 0
        [10.0, 0.0],    # Exact match with sample 1
        [5.0, 0.0],     # Symmetrical midpoint
        [1000.0, 0.0],  # Far beyond range
    ])

    estimates, variances = simple_kriging_grid_estimation(
        samples_xy,
        sample_grades,
        grid_points,
        mean=mean,
        variogram_model="spherical",
        nugget=0.0,
        sill=1.0,
        range_param=20.0,
        k_neighbors=2,
    )

    # 1. Collocated points return exact grade and 0 variance
    assert estimates[0] == pytest.approx(2.0, abs=1e-5)
    assert variances[0] == pytest.approx(0.0, abs=1e-5)
    assert estimates[1] == pytest.approx(4.0, abs=1e-5)
    assert variances[1] == pytest.approx(0.0, abs=1e-5)

    # 2. Midpoint gives unbiased symmetrical estimate (3.0) and reduced variance (< 1.0)
    assert estimates[2] == pytest.approx(3.0, abs=1e-5)
    assert 0.0 < variances[2] < 1.0

    # 3. Far point beyond range reverts to prior mean and total sill (1.0)
    assert estimates[3] == pytest.approx(mean, abs=1e-5)
    assert variances[3] == pytest.approx(1.0, abs=1e-5)


def test_simple_kriging_mask_extrapolation():
    samples_xy = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    sample_grades = np.array([1.0, 2.0, 3.0])
    grid_points = np.array([
        [5.0, 2.0],     # Inside triangle hull
        [5.0, -5.0],    # Outside hull
    ])

    # Unmasked: both evaluated
    est_unmasked, var_unmasked = simple_kriging_grid_estimation(
        samples_xy, sample_grades, grid_points, mean=2.0, mask_extrapolation=False
    )
    assert not np.isnan(est_unmasked[0])
    assert not np.isnan(est_unmasked[1])

    # Masked: outside point masked to NaN
    est_masked, var_masked = simple_kriging_grid_estimation(
        samples_xy, sample_grades, grid_points, mean=2.0, mask_extrapolation=True
    )
    assert not np.isnan(est_masked[0])
    assert not np.isnan(var_masked[0])
    assert np.isnan(est_masked[1])
    assert np.isnan(var_masked[1])


def test_ordinary_kriging_collocation_and_unbiasedness():
    samples_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    sample_grades = np.array([2.0, 4.0])

    grid_points = np.array([
        [0.0, 0.0],    # Exact match with sample 0
        [10.0, 0.0],   # Exact match with sample 1
        [5.0, 0.0],    # Symmetrical midpoint
        [50.0, 0.0],   # Far point within search
    ])

    estimates, variances = ordinary_kriging_grid_estimation(
        samples_xy,
        sample_grades,
        grid_points,
        variogram_model="spherical",
        nugget=0.0,
        sill=1.0,
        range_param=20.0,
        k_neighbors=2,
    )

    # 1. Collocated points return exact grade and 0 variance
    assert estimates[0] == pytest.approx(2.0, abs=1e-5)
    assert variances[0] == pytest.approx(0.0, abs=1e-5)
    assert estimates[1] == pytest.approx(4.0, abs=1e-5)
    assert variances[1] == pytest.approx(0.0, abs=1e-5)

    # 2. Midpoint gives unbiased 50/50 weighting: estimate = 3.0
    assert estimates[2] == pytest.approx(3.0, abs=1e-5)
    assert 0.0 < variances[2] < 1.0


def test_ordinary_kriging_max_radius_and_mask_extrapolation():
    samples_xy = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    sample_grades = np.array([1.0, 2.0, 3.0])
    grid_points = np.array([
        [5.0, 2.0],     # Inside triangle hull
        [5.0, -5.0],    # Outside hull
        [100.0, 100.0], # Far beyond max_radius
    ])

    # Unmasked with max_radius: far point gets NaN, others estimated
    est, var = ordinary_kriging_grid_estimation(
        samples_xy, sample_grades, grid_points, max_radius=30.0, mask_extrapolation=False
    )
    assert not np.isnan(est[0])
    assert not np.isnan(est[1])
    assert np.isnan(est[2])  # Beyond max_radius -> NaN in OK

    # Masked: outside point masked to NaN
    est_masked, var_masked = ordinary_kriging_grid_estimation(
        samples_xy, sample_grades, grid_points, max_radius=30.0, mask_extrapolation=True
    )
    assert not np.isnan(est_masked[0])
    assert np.isnan(est_masked[1])  # Outside hull -> NaN
    assert np.isnan(est_masked[2])


def test_plot_grade_tonnage_curve_single_table():
    df = pd.DataFrame({
        "grade": [0.5, 0.8, 1.2, 1.5, 2.0],
        "tonnes": [10000.0, 10000.0, 10000.0, 10000.0, 10000.0],
    })
    gt = grade_tonnage_table(df, cutoffs=[0.0, 0.5, 1.0, 1.5])

    fig, ax = plot_grade_tonnage_curve(
        gt,
        grade_unit="% Cu",
        tonnage_unit="kt",
        title="Test Single Curve",
        show_metal=True,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_plot_grade_tonnage_curve_multi_model():
    df1 = pd.DataFrame({
        "grade": [0.5, 0.8, 1.2, 1.5],
        "tonnes": [10000.0, 10000.0, 10000.0, 10000.0],
    })
    df2 = pd.DataFrame({
        "grade": [0.6, 0.9, 1.1, 1.4],
        "tonnes": [10000.0, 10000.0, 10000.0, 10000.0],
    })
    gt1 = grade_tonnage_table(df1, cutoffs=[0.0, 0.5, 1.0])
    gt2 = grade_tonnage_table(df2, cutoffs=[0.0, 0.5, 1.0])

    model_dict = {"Model A": gt1, "Model B": gt2}

    fig, ax = plot_grade_tonnage_curve(
        model_dict,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        title="Comparative Audit",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

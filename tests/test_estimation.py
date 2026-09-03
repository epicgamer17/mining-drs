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
    cell_declustering,
    kriging_quality_metrics,
    classify_mineral_resources,
    format_resource_statement,
    plot_swath_analysis,
    plot_cell_declustering_curve,
    calculate_cut_off_grade,
    convert_resource_to_reserve,
    format_reserve_statement,
    plot_resource_to_reserve_waterfall,
    plot_reserve_classification_map,
    plot_in_situ_vs_diluted_curves,
    composite_drillhole_intervals,
    apply_grade_capping,
    exploratory_data_analysis,
    plot_eda_distributions,
    contact_profile_analysis,
    plot_contact_profile,
    reconcile_production_to_reserve,
    plot_production_reconciliation,
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
        samples_xy,
        sample_grades,
        grid_points,
        mean=2.0,
        sill=1.0,
        range_param=100.0,
        mask_extrapolation=False,
    )
    assert not np.isnan(est_unmasked[0])
    assert not np.isnan(est_unmasked[1])

    # Masked: outside point masked to NaN
    est_masked, var_masked = simple_kriging_grid_estimation(
        samples_xy,
        sample_grades,
        grid_points,
        mean=2.0,
        sill=1.0,
        range_param=100.0,
        mask_extrapolation=True,
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
        samples_xy,
        sample_grades,
        grid_points,
        sill=1.0,
        range_param=100.0,
        max_radius=30.0,
        mask_extrapolation=False,
    )
    assert not np.isnan(est[0])
    assert not np.isnan(est[1])
    assert np.isnan(est[2])  # Beyond max_radius -> NaN in OK

    # Masked: outside point masked to NaN
    est_masked, var_masked = ordinary_kriging_grid_estimation(
        samples_xy,
        sample_grades,
        grid_points,
        sill=1.0,
        range_param=100.0,
        max_radius=30.0,
        mask_extrapolation=True,
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

    results = plot_grade_tonnage_curve(
        model_dict,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        title="Comparative Audit",
    )
    assert isinstance(results, dict)
    assert "Model A" in results and "Model B" in results
    for fig, ax in results.values():
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)


def test_sme_jorc_stubs():
    pts = np.array([[15.0, 35.0]])
    samples = np.array([[10.0, 30.0], [20.0, 40.0]])

    with pytest.raises(NotImplementedError):
        kriging_quality_metrics(np.array([0.1]), block_dispersion_variance=0.25)

    with pytest.raises(NotImplementedError):
        classify_mineral_resources(pts, samples)


def test_plot_swath_analysis_dual_axis():
    # Synthetic block model with X coordinates from 50 to 350
    np.random.seed(42)
    x_coords = np.linspace(50, 350, 60)
    y_coords = np.linspace(100, 400, 60)
    block_model = pd.DataFrame({
        "x": x_coords,
        "y": y_coords,
        "grade": 0.8 + 0.3 * np.sin(x_coords / 50.0),
        "nn_grade": 0.8 + 0.35 * np.sin(x_coords / 50.0) + np.random.normal(0, 0.05, len(x_coords)),
        "tonnes": 50000.0,
    })

    drillholes = pd.DataFrame({
        "x": [80.0, 150.0, 220.0, 290.0],
        "y": [120.0, 200.0, 280.0, 350.0],
        "grade": [0.95, 1.15, 0.75, 0.55],
    })

    # Test along Easting (X)
    fig_x, ax_x = plot_swath_analysis(
        block_model,
        drillholes=drillholes,
        axis="x",
        bin_width=50.0,
        grade_col="grade",
        validation_grade_col="nn_grade",
        tonnes_col="tonnes",
        model_name="Ordinary Kriging",
        validation_model_name="Nearest Neighbor",
        grade_unit="% Cu",
        tonnage_unit="Mt",
    )
    assert isinstance(fig_x, plt.Figure)
    assert isinstance(ax_x, plt.Axes)
    plt.close(fig_x)

    # Test along Northing (Y)
    fig_y, ax_y = plot_swath_analysis(
        block_model,
        axis="northing",
        bin_width=60.0,
        grade_col="grade",
        tonnes_col="tonnes",
    )
    assert isinstance(fig_y, plt.Figure)
    assert isinstance(ax_y, plt.Axes)
    plt.close(fig_y)


def test_format_resource_statement_sig_figs_and_footnotes():
    # Synthetic block model with Measured, Indicated, and Inferred blocks
    block_df = pd.DataFrame({
        "category": [
            "Measured", "Measured", "Indicated", "Indicated", "Inferred", "Inferred"
        ],
        "grade": [1.234, 1.456, 0.876, 0.954, 0.654, 0.432],
        "tonnes": [1_000_000.0, 2_450_000.0, 3_120_000.0, 1_850_000.0, 4_560_000.0, 2_100_000.0],
    })

    statement = format_resource_statement(
        block_df,
        category_col="category",
        grade_col="grade",
        tonnes_col="tonnes",
        cutoff_grade=0.50,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        metal_unit="kt",
        commodity_price="$3.85/lb Cu",
        metallurgical_recovery=88.5,
        rpeee_constraint="Constrained within Lerchs-Grossmann optimized pit shell",
    )

    assert isinstance(statement, pd.DataFrame)
    assert len(statement) == 4  # Measured, Indicated, Measured + Indicated, Inferred
    assert list(statement["Classification"]) == [
        "Measured", "Indicated", "Measured + Indicated", "Inferred"
    ]

    # Check footnotes metadata
    assert "footnotes" in statement.attrs
    footnotes = statement.attrs["footnotes"]
    assert len(footnotes) == 5
    # Verify mandatory footnote 2 exists
    assert any("Totals may not sum due to rounding" in fn for fn in footnotes)
    # Verify RPEEE footnote exists
    assert any("Lerchs-Grossmann" in fn for fn in footnotes)
    assert any("$3.85/lb Cu" in fn for fn in footnotes)
    assert any("88.5%" in fn for fn in footnotes)


def test_cell_declustering_removes_clustering_bias():
    # 4 clustered drillholes in high-grade sweet spot (grade=2.0)
    # 1 isolated drillhole in background (grade=0.5)
    df = pd.DataFrame({
        "x": [10.0, 11.0, 10.0, 12.0, 100.0],
        "y": [10.0, 10.0, 11.0, 11.0, 100.0],
        "grade": [2.0, 2.0, 2.0, 2.0, 0.5],
    })
    naive_mean = df["grade"].mean()  # (4*2.0 + 0.5) / 5 = 1.70

    cell_sizes = [5.0, 15.0, 25.0, 50.0, 80.0, 150.0]
    weights, sensitivity_df, opt_cs = cell_declustering(df, cell_sizes=cell_sizes, min_mean=True)

    # Weights must sum to 1.0
    assert np.isclose(weights.sum(), 1.0)
    assert len(weights) == 5

    # Clustered samples must have smaller individual weight than the isolated sample
    assert weights[4] > weights[0]
    assert weights[4] > weights[1]

    # Declustered mean must be lower than naive mean (clustering bias removed)
    min_declust_mean = sensitivity_df["declustered_mean"].min()
    assert min_declust_mean < naive_mean
    # At cell size ~50m, 4 clustered points are in 1 cell (share 50% weight), isolated point is in 1 cell (50% weight)
    # So expected declustered mean is roughly 0.5 * 2.0 + 0.5 * 0.5 = 1.25
    assert min_declust_mean < 1.45


def test_plot_cell_declustering_curve():
    sensitivity_df = pd.DataFrame({
        "cell_size": [10.0, 25.0, 50.0, 100.0],
        "declustered_mean": [1.60, 1.35, 1.25, 1.50],
        "declustered_variance": [0.35, 0.40, 0.45, 0.38],
    })

    fig, ax = plot_cell_declustering_curve(
        sensitivity_df,
        naive_mean=1.70,
        optimal_cell_size=50.0,
        grade_unit="% Cu",
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_kriging_twin_holes_duplicate_handling():
    # Twin drillholes at identical coordinates (10, 10) with grades 1.0 and 2.0 (average 1.5)
    # Plus a third distinct drillhole at (30, 10) with grade 3.0
    samples_xy = np.array([[10.0, 10.0], [10.0, 10.0], [30.0, 10.0]])
    sample_grades = np.array([1.0, 2.0, 3.0])
    target = np.array([[20.0, 10.0]])

    # Simple Kriging must not raise LinAlgError
    sk_est, sk_var = simple_kriging_grid_estimation(
        samples_xy, sample_grades, target, mean=2.0, sill=0.5, nugget=0.1, range_param=50.0
    )
    assert not np.isnan(sk_est[0])

    # Ordinary Kriging must not raise LinAlgError and should correctly average twin grades
    ok_est, ok_var = ordinary_kriging_grid_estimation(
        samples_xy, sample_grades, target, sill=0.5, nugget=0.1, range_param=50.0
    )
    assert not np.isnan(ok_est[0])
    # Midpoint between twin hole (avg=1.5) and third hole (3.0): symmetric midpoint should be ~2.25
    assert np.isclose(ok_est[0], 2.25, atol=0.05)


def test_kriging_missing_required_sill_and_range_raises_type_error():
    samples_xy = np.array([[0.0, 0.0], [10.0, 10.0]])
    grades = np.array([1.0, 2.0])
    target = np.array([[5.0, 5.0]])

    # Simple kriging requires sill and range_param
    with pytest.raises(TypeError):
        simple_kriging_grid_estimation(samples_xy, grades, target, mean=1.5)  # Missing sill & range_param

    # Ordinary kriging requires sill and range_param
    with pytest.raises(TypeError):
        ordinary_kriging_grid_estimation(samples_xy, grades, target)  # Missing sill & range_param


def test_calculate_cut_off_grade_breakeven_and_marginal():
    # Base case: Cu mine with $3.80/lb Cu price, $12/t processing, $2/t G&A, $2.50/t mining
    # Met recovery = 88%, 1% Cu = 22.0462 lbs/t
    # Net price = $3.80/lb
    # Revenue per 1% Cu = 3.80 * 0.88 * 22.0462 = $73.7225 / (% Cu)
    # Breakeven cost = 12 + 2 + 2.50 = $16.50/t -> cutoff = 16.50 / 73.7225 = 0.2238% Cu
    # Marginal cost = 12 + 2 = $14.00/t -> cutoff = 14.00 / 73.7225 = 0.1899% Cu

    co_be = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        metal_conversion_factor=22.0462,
    )
    assert np.isclose(co_be, 0.2238, atol=0.001)

    co_marg = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=None,  # Marginal / Internal (sunk mining cost)
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        metal_conversion_factor=22.0462,
    )
    assert np.isclose(co_marg, 0.1899, atol=0.001)
    assert co_marg < co_be

    # Royalties and selling deductions reduce net price and increase cut-off grade
    co_royalty = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        selling_cost=0.30,  # $0.30/lb deduction
        royalty_pct=2.0,    # 2% NSR royalty
        metallurgical_recovery=88.0,
        metal_conversion_factor=22.0462,
    )
    assert co_royalty > co_be

    # Validation errors
    with pytest.raises(ValueError, match="negative"):
        calculate_cut_off_grade(processing_cost=-5.0, ga_cost=2.0, commodity_price=3.80, metallurgical_recovery=88.0)
    with pytest.raises(ValueError, match="positive"):
        calculate_cut_off_grade(processing_cost=12.0, ga_cost=2.0, commodity_price=-3.80, metallurgical_recovery=88.0)


def test_convert_resource_to_reserve_strict_inferred_exclusion_and_modifying_factors():
    # 4 blocks: Measured, Indicated, Inferred, and low-grade Measured
    resource_df = pd.DataFrame({
        "category": ["Measured", "Indicated", "Inferred", "Measured"],
        "grade": [1.20, 0.90, 1.50, 0.30],  # 0.30 is below cutoff
        "tonnes": [1000.0, 2000.0, 5000.0, 1000.0],
    })

    cutoff = 0.50
    dilution_pct = 10.0  # 10% dilution with 0.0% grade
    recovery_pct = 95.0  # 95% mining recovery (5% ore loss)

    reserve_df = convert_resource_to_reserve(
        resource_df,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        cutoff_grade=cutoff,
        dilution_grade=0.0,
        allow_inferred=False,
    )

    # 1. Verification of Inferred exclusion: Inferred block (5000t, grade 1.50%) MUST NOT BE in reserves!
    assert "Inferred" not in reserve_df["reserve_category"].values
    assert reserve_df.attrs["excluded_inferred_tonnes"] == 5000.0

    # 2. Verification of Cutoff filtering: block with grade 0.30% (< 0.50%) must be excluded
    assert len(reserve_df) == 2  # Only Measured (1.20%) and Indicated (0.90%)

    # 3. Verification of category mapping:
    # Measured -> Proven Reserve, Indicated -> Probable Reserve
    assert list(reserve_df["reserve_category"]) == ["Proven Reserve", "Probable Reserve"]

    # 4. Verification of Modifying Factors on Block 0 (Measured: 1000t in-situ, 1.20% grade):
    # Dilution: T_dil = 1000 * 1.10 = 1100t, g_dil = (1000 * 1.20 + 100 * 0.0) / 1100 = 1.0909%
    # Recovery: T_rom = 1100 * 0.95 = 1045t, g_rom = 1.0909%
    row0 = reserve_df.iloc[0]
    assert np.isclose(row0["rom_tonnes"], 1045.0)
    assert np.isclose(row0["rom_grade"], 1.0909, atol=1e-3)
    assert np.isclose(row0["contained_metal"], 1045.0 * (1.090909 / 100.0), atol=1e-3)

    # Missing required modifying factor raises TypeError
    with pytest.raises(TypeError):
        convert_resource_to_reserve(resource_df, mining_dilution_pct=5.0, cutoff_grade=0.5)  # Missing mining_recovery_pct


def test_format_reserve_statement_sig_figs_and_footnotes():
    reserve_df = pd.DataFrame({
        "reserve_category": ["Proven Reserve", "Probable Reserve", "Proven Reserve"],
        "rom_tonnes": [1_234_567.0, 3_456_789.0, 2_345_678.0],
        "rom_grade": [1.25, 0.85, 1.15],
    })

    stmt = format_reserve_statement(
        reserve_df,
        cutoff_grade=0.50,
        mining_dilution_pct=6.0,
        mining_recovery_pct=94.0,
        commodity_price="$3.80/lb Cu",
        metallurgical_recovery=88.5,
        tonnage_unit="Mt",
        grade_unit="% Cu",
        metal_unit="kt",
    )

    # 3 rows: Proven Reserve, Probable Reserve, Total Proven + Probable
    assert len(stmt) == 3
    assert list(stmt["Classification"]) == ["Proven Reserve", "Probable Reserve", "Total Proven + Probable"]

    # Verify regulatory footnotes
    fns = stmt.attrs["footnotes"]
    assert len(fns) == 5
    assert any("CIM Definition Standards" in f for f in fns)
    assert any("Mining Dilution = 6.0%" in f for f in fns)
    assert any("Mining Recovery = 94.0%" in f for f in fns)
    assert any("Metallurgical Recovery = 88.5%" in f for f in fns)
    assert any("Pre-Feasibility" in f for f in fns)


def test_plot_resource_to_reserve_waterfall():
    res_df = pd.DataFrame({
        "category": ["Measured", "Indicated", "Inferred", "Measured"],
        "grade": [1.2, 0.9, 1.5, 0.3],
        "tonnes": [1e6, 2e6, 3e6, 0.5e6],
    })
    res_reserve = convert_resource_to_reserve(
        res_df,
        mining_dilution_pct=5.0,
        mining_recovery_pct=95.0,
        cutoff_grade=0.50,
        dilution_grade=0.0,
    )

    fig, (ax1, ax2) = plot_resource_to_reserve_waterfall(
        res_df,
        res_reserve,
        cutoff_grade=0.50,
        mining_dilution_pct=5.0,
        mining_recovery_pct=95.0,
        tonnage_unit="Mt",
        metal_unit="kt",
        grade_unit="% Cu",
    )
    assert fig is not None
    assert ax1.get_ylabel().startswith("Ore Tonnage")
    assert ax2.get_ylabel().startswith("Contained Metal")
    plt.close(fig)


def test_plot_reserve_classification_map():
    block_model = pd.DataFrame({
        "x": [10.0, 20.0, 30.0, 40.0],
        "y": [10.0, 20.0, 30.0, 40.0],
        "status": ["Proven Reserve", "Probable Reserve", "Inferred Resource (Excluded)", "Sub-Economic / Waste"],
    })
    boundary = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)]
    drillholes = pd.DataFrame({"x": [15.0, 35.0], "y": [15.0, 35.0]})

    fig, ax = plot_reserve_classification_map(
        block_model,
        boundary=boundary,
        drillholes=drillholes,
        title="Test Classification Map",
    )
    assert fig is not None
    assert ax.get_title() == "Test Classification Map"
    plt.close(fig)


def test_plot_in_situ_vs_diluted_curves():
    cutoffs = [0.0, 0.4, 0.8, 1.2]
    insitu_gt = pd.DataFrame({
        "ore_tonnes": [10e6, 8e6, 5e6, 2e6],
        "ore_grade": [1.0, 1.2, 1.5, 1.8],
    }, index=cutoffs)

    diluted_gt = pd.DataFrame({
        "ore_tonnes": [10.5e6, 8.4e6, 5.2e6, 2.1e6],
        "ore_grade": [0.95, 1.14, 1.42, 1.71],
    }, index=cutoffs)

    fig, ax = plot_in_situ_vs_diluted_curves(
        insitu_gt,
        diluted_gt,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        title="Test Grade-Tonnage Shift",
    )
    assert fig is not None
    assert ax.get_title() == "Test Grade-Tonnage Shift"
    plt.close(fig)


def test_composite_drillhole_intervals_length_weighting_and_domains():
    # Hole 1: 3 raw intervals
    # 0.0 - 1.0m: grade 2.0%
    # 1.0 - 2.5m: grade 4.0%
    # 2.5 - 3.0m: grade 1.0%
    assays_df = pd.DataFrame({
        "hole_id": ["DH01", "DH01", "DH01"],
        "from_m": [0.0, 1.0, 2.5],
        "to_m": [1.0, 2.5, 3.0],
        "grade": [2.0, 4.0, 1.0],
        "x": [100.0, 100.0, 100.0],
        "y": [200.0, 200.0, 200.0],
    })

    # Composite to 2.0m with remnant_strategy="discard" and min_length_ratio=0.5
    # Comp 1: [0.0, 2.0] -> 1.0m @ 2.0 + 1.0m @ 4.0 -> grade = (2 + 4)/2 = 3.0%
    # Comp 2: [2.0, 3.0] -> 0.5m @ 4.0 + 0.5m @ 1.0 -> grade = (2.0 + 0.5)/1 = 2.5% (length=1.0m = 50% of 2m)
    comp_df = composite_drillhole_intervals(
        assays_df,
        composite_length=2.0,
        min_length_ratio=0.50,
        remnant_strategy="discard",
    )
    assert len(comp_df) == 2
    assert np.isclose(comp_df.iloc[0]["grade"], 3.0)
    assert np.isclose(comp_df.iloc[0]["length"], 2.0)
    assert np.isclose(comp_df.iloc[1]["grade"], 2.5)
    assert np.isclose(comp_df.iloc[1]["length"], 1.0)

    # Discard remnant when min_length_ratio is strict (e.g. 0.75 > 1.0/2.0)
    comp_discard = composite_drillhole_intervals(
        assays_df,
        composite_length=2.0,
        min_length_ratio=0.75,
        remnant_strategy="discard",
    )
    assert len(comp_discard) == 1
    assert comp_discard.attrs["discarded_remnants_count"] == 1

    # Distribute strategy: 3.0m total length / 2.0m target -> 2 equal composites of 1.5m
    comp_dist = composite_drillhole_intervals(
        assays_df,
        composite_length=2.0,
        remnant_strategy="distribute",
    )
    assert len(comp_dist) == 2
    assert np.isclose(comp_dist.iloc[0]["length"], 1.5)
    assert np.isclose(comp_dist.iloc[1]["length"], 1.5)

    # Domain constraint test: Never cross domain boundaries!
    domain_df = pd.DataFrame({
        "hole_id": ["DH01", "DH01", "DH01"],
        "from_m": [0.0, 1.5, 3.5],
        "to_m": [1.5, 3.5, 5.0],
        "grade": [0.5, 3.0, 4.0],
        "domain": ["Waste", "Ore", "Ore"],
    })
    # Target 2.0m composites:
    # Waste domain (0 to 1.5m): 1.5m run. With min_length_ratio=0.5 (1.0m), kept as [0, 1.5] (or discarded if > 1.5).
    # Ore domain (1.5 to 5.0m): 3.5m run -> [1.5, 3.5] (2.0m) and [3.5, 5.0] (1.5m).
    # A composite must NEVER span across the Waste/Ore contact at 1.5m!
    comp_dom = composite_drillhole_intervals(
        domain_df,
        composite_length=2.0,
        domain_col="domain",
        min_length_ratio=0.50,
    )
    assert len(comp_dom) == 3
    # First composite is purely Waste
    assert comp_dom.iloc[0]["domain"] == "Waste"
    assert comp_dom.iloc[0]["to_m"] == 1.5
    # Second and third composites are purely Ore
    assert comp_dom.iloc[1]["domain"] == "Ore"
    assert comp_dom.iloc[1]["from_m"] == 1.5
    assert comp_dom.iloc[1]["to_m"] == 3.5


def test_apply_grade_capping_and_metal_reduction():
    # Sample composites with a severe high-grade outlier (50.0 g/t)
    df = pd.DataFrame({
        "grade": [1.0, 1.5, 2.0, 1.2, 0.8, 1.4, 50.0],
        "length": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    })

    # Capping at explicit threshold 5.0 g/t
    capped_df = apply_grade_capping(df, cap_grade=5.0, grade_col="grade", length_col="length")
    assert "capped_grade" in capped_df.columns
    assert capped_df["capped_grade"].max() == 5.0
    assert capped_df.loc[6, "capped_grade"] == 5.0

    summary = capped_df.attrs["capping_summary"]
    assert summary["samples_capped"] == 1
    assert summary["cap_grade"] == 5.0
    assert summary["metal_reduction_pct"] > 0.0
    assert summary["capped_cv"] < summary["uncapped_cv"]

    # Capping by percentile (e.g. P90)
    capped_pct = apply_grade_capping(df, percentile=90.0, grade_col="grade")
    assert capped_pct["capped_grade"].max() < 50.0
    assert capped_pct.attrs["capping_summary"]["samples_capped"] >= 1

    # Missing cap_grade and percentile raises ValueError
    with pytest.raises(ValueError, match="Must specify"):
        apply_grade_capping(df, grade_col="grade")


def test_exploratory_data_analysis_metrics_and_clustering_bias():
    grades = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 25.0]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
    df = pd.DataFrame({"grade": grades, "declust_weight": weights})

    # Run EDA without weights
    eda_naive = exploratory_data_analysis(df, grade_col="grade")
    assert "Naive" in eda_naive.columns
    assert "Declustered" not in eda_naive.columns
    assert eda_naive.loc["Sample Count", "Naive"] == 11
    assert np.isclose(eda_naive.loc["Minimum", "Naive"], 0.2)
    assert np.isclose(eda_naive.loc["Maximum", "Naive"], 25.0)
    assert eda_naive.loc["Coeff. of Variation (CV)", "Naive"] > 1.5
    assert eda_naive.attrs["cv_status"].startswith("Highly skewed")

    # Run EDA with declustering weights
    eda_dec = exploratory_data_analysis(df, grade_col="grade", weights_col="declust_weight")
    assert "Declustered" in eda_dec.columns
    # With low weight (0.1) on the high-grade outlier (10.0), declustered mean should be lower than naive
    assert eda_dec.loc["Mean", "Declustered"] < eda_dec.loc["Mean", "Naive"]
    assert "clustering_bias_pct" in eda_dec.attrs
    assert eda_dec.attrs["clustering_bias_pct"] > 0.0


def test_plot_eda_distributions():
    np.random.seed(42)
    grades = np.random.lognormal(mean=0.0, sigma=0.6, size=150)
    capped_grades = np.minimum(grades, 2.5)
    df = pd.DataFrame({"grade": grades, "capped_grade": capped_grades})

    fig, axes = plot_eda_distributions(
        df,
        grade_col="grade",
        capped_grade_col="capped_grade",
        cap_grade=2.5,
        bins=20,
        grade_unit="% Cu",
        title="Deposit EDA Summary & Capping Diagnostic",
    )
    assert fig is not None
    assert len(axes) == 3
    assert "Histogram" in axes[0].get_title()
    assert "Log-Transformed" in axes[1].get_title()
    assert "Log-Probability" in axes[2].get_title()
    plt.close(fig)


def test_contact_profile_analysis_hard_vs_soft_and_plot():
    # Synthetic dataset with a sharp Hard Boundary at distance = 0
    # Domain A (Host rock): distances -20 to 0, grades around 0.25%
    # Domain B (Ore zone): distances 0 to +20, grades around 1.80%
    dists_a = np.linspace(-18.0, -1.0, 30)
    grades_a = 0.25 + 0.05 * np.random.randn(30)
    dom_a = ["HostRock"] * 30

    dists_b = np.linspace(1.0, 18.0, 30)
    grades_b = 1.80 + 0.10 * np.random.randn(30)
    dom_b = ["OreZone"] * 30

    hard_df = pd.DataFrame({
        "distance": np.concatenate([dists_a, dists_b]),
        "grade": np.concatenate([grades_a, grades_b]),
        "lithology": dom_a + dom_b,
    })

    profile_hard = contact_profile_analysis(
        hard_df,
        domain_col="lithology",
        grade_col="grade",
        distance_col="distance",
        bin_width=2.0,
        max_distance=20.0,
        domain_a="HostRock",
        domain_b="OreZone",
    )

    assert profile_hard.attrs["boundary_type"] == "Hard"
    assert profile_hard.attrs["step_change"] > 1.0
    assert "strictly segregated" in profile_hard.attrs["recommendation"].lower()

    # Test plot rendering
    fig, (ax1, ax2) = plot_contact_profile(
        profile_hard,
        domain_a_name="Country Rock",
        domain_b_name="High-Grade Ore",
        grade_unit="% Cu",
    )
    assert fig is not None
    assert ax1.get_ylabel().startswith("Average Grade")
    assert ax2.get_ylabel().startswith("Composites")
    plt.close(fig)

    # Synthetic dataset with a Soft Boundary (continuous gradient, no step change)
    all_dists = np.linspace(-15.0, 15.0, 60)
    smooth_grades = 1.00 + 0.01 * all_dists  # ~1.00 on both sides of 0
    soft_df = pd.DataFrame({
        "distance": all_dists,
        "grade": smooth_grades,
        "zone": ["ZoneA" if d < 0 else "ZoneB" for d in all_dists],
    })

    profile_soft = contact_profile_analysis(
        soft_df,
        domain_col="zone",
        grade_col="grade",
        distance_col="distance",
        bin_width=2.0,
        max_distance=16.0,
        domain_a="ZoneA",
        domain_b="ZoneB",
    )
    assert profile_soft.attrs["boundary_type"] == "Soft"
    assert "freely shared" in profile_soft.attrs["recommendation"].lower()


def test_reconcile_production_to_reserve_f_factors_and_ratios():
    # Multi-period reconciliation (3 months)
    reserve_df = pd.DataFrame({
        "period": ["Jan", "Feb", "Mar"],
        "tonnes": [100_000.0, 110_000.0, 105_000.0],
        "grade": [1.00, 1.10, 0.95],
    })
    gc_df = pd.DataFrame({
        "period": ["Jan", "Feb", "Mar"],
        "tonnes": [105_000.0, 108_000.0, 104_000.0],
        "grade": [0.98, 1.12, 0.96],
    })
    plant_df = pd.DataFrame({
        "period": ["Jan", "Feb", "Mar"],
        "tonnes": [103_000.0, 107_000.0, 103_500.0],
        "grade": [0.97, 1.11, 0.95],
    })

    rec_df = reconcile_production_to_reserve(
        reserve_df,
        plant_df,
        grade_control_data=gc_df,
        period_col="period",
        grade_unit="% Cu",
    )

    # Output rows: 3 periods + 1 "Total" row = 4 rows
    assert len(rec_df) == 4
    assert rec_df.iloc[-1]["period"] == "Total"

    # Verify F1 * F2 == F3 identity within numerical precision
    for i in range(len(rec_df)):
        f1 = rec_df.iloc[i]["f1_metal_factor"]
        f2 = rec_df.iloc[i]["f2_metal_factor"]
        f3 = rec_df.iloc[i]["f3_metal_factor"]
        assert np.isclose(f1 * f2, f3, atol=1e-5)

    # Check component ratios: R_T * R_G == F_factor
    for i in range(len(rec_df)):
        rt = rec_df.iloc[i]["f3_tonnes_ratio"]
        rg = rec_df.iloc[i]["f3_grade_ratio"]
        f3 = rec_df.iloc[i]["f3_metal_factor"]
        assert np.isclose(rt * rg, f3, atol=1e-5)

    # Verify health status attribute
    assert "health_status" in rec_df.attrs
    assert "EXCELLENT" in rec_df.attrs["health_status"] or "GOOD" in rec_df.attrs["health_status"]

    # Test single-period dictionary input
    res_dict = {"tonnes": 50_000.0, "grade": 1.20}
    plant_dict = {"tonnes": 51_000.0, "grade": 1.15}
    rec_single = reconcile_production_to_reserve(res_dict, plant_dict, grade_unit="% Cu")
    assert len(rec_single) == 1
    assert "f3_metal_factor" in rec_single.columns
    assert np.isclose(rec_single.iloc[0]["f3_tonnes_ratio"], 51000.0 / 50000.0)


def test_plot_production_reconciliation():
    rec_df = pd.DataFrame({
        "period": ["Q1", "Q2", "Q3", "Total"],
        "reserve_tonnes": [1.0, 1.1, 1.2, 3.3],
        "reserve_grade": [1.0, 1.0, 1.0, 1.0],
        "reserve_metal": [0.01, 0.011, 0.012, 0.033],
        "gc_tonnes": [1.02, 1.08, 1.18, 3.28],
        "gc_grade": [0.99, 1.01, 0.99, 1.0],
        "gc_metal": [0.0101, 0.0109, 0.0117, 0.0327],
        "plant_tonnes": [1.01, 1.09, 1.19, 3.29],
        "plant_grade": [0.98, 1.00, 0.98, 0.99],
        "plant_metal": [0.0099, 0.0109, 0.0117, 0.0325],
        "f1_metal_factor": [1.01, 0.99, 0.975, 0.99],
        "f2_metal_factor": [0.98, 1.00, 1.00, 0.994],
        "f3_metal_factor": [0.99, 0.99, 0.975, 0.985],
    })
    rec_df.attrs["health_status"] = "EXCELLENT: Production is within +/-5% of reserve model."

    fig, axes = plot_production_reconciliation(
        rec_df,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        metal_unit="kt",
        title="Mine-to-Mill Production Reconciliation",
    )
    assert fig is not None
    assert len(axes) == 4
    assert "Tonnage" in axes[0].get_title()
    assert "Grade" in axes[1].get_title()
    assert "Metal" in axes[2].get_title()
    assert "Parker" in axes[3].get_title()
    plt.close(fig)










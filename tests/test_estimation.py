import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drs_mining.components.estimation import (
    polygonal_estimation,
    polygonal_resource_summary,
    format_polygonal_summary,
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
    classify_resources_by_drill_spacing,
    classify_resources_by_sor,
    classify_resources_by_kriging_variance,
    smooth_resource_categories,
    classify_mineral_resources,
    format_resource_statement,
    plot_swath_analysis,
    plot_cell_declustering_curve,
    calculate_cut_off_grade,
    cut_off_grade_breakdown,
    convert_resource_to_reserve,
    format_reserve_statement,
    plot_resource_to_reserve_waterfall,
    plot_reserve_classification_map,
    plot_resource_classification_map,
    plot_in_situ_vs_diluted_curves,
    composite_drillhole_intervals,
    apply_grade_capping,
    exploratory_data_analysis,
    plot_eda_distributions,
    contact_profile_analysis,
    plot_contact_profile,
    reconcile_production_to_reserve,
    plot_production_reconciliation,
    sequential_gaussian_simulation,
    compute_etype_mtype_maps,
    plot_simulation_realizations_dashboard,
    create_block_model,
    ordinary_kriging_block_estimation,
    simple_kriging_block_estimation,
    plot_block_model_orthogonal_slices,
    plot_block_model_bench_gallery,
    plot_block_model_3d_isometric,
    plot_block_model_3d_interactive,
    plot_block_model_grade_uncertainty,
    _theoretical_covariance,
    compute_anisotropy_rotation_matrix,
    transform_anisotropic_coordinates,
    compute_anisotropic_distance,
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
        "contained_metal": [270.0, 1080.0, 135.0],  # tonnes * grade * 0.01
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

    # Test custom metal_factor (e.g. g/t Au -> tonnes metal: 1e-6)
    polys_gold = polygonal_estimation(
        df_holes,
        boundary=boundary,
        bulk_density=2.7,
        max_radius=100.0,
        metal_factor=1e-6,
    )
    assert np.allclose(polys_gold["contained_metal"], polys["tonnes"] * polys["grade"] * 1e-6)


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
        df_holes, bulk_density=2.7, boundary=boundary, clip_to_convex_hull=False
    )
    assert unclipped["area_m2"].sum() == pytest.approx(250000.0)

    # 2. Clipped to convex hull: total area matches exact triangle area (10,000 m2)
    clipped = polygonal_estimation(
        df_holes, bulk_density=2.7, boundary=boundary, clip_to_convex_hull=True
    )
    assert clipped["area_m2"].sum() == pytest.approx(10000.0)
    assert (clipped["area_m2"] > 0).all()

    # 3. Validations: required bulk_density and positive check
    with pytest.raises(TypeError):
        polygonal_estimation(df_holes, boundary=boundary)  # missing bulk_density
    with pytest.raises(ValueError, match="bulk_density must be positive"):
        polygonal_estimation(df_holes, bulk_density=0.0, boundary=boundary)
    with pytest.raises(ValueError, match="bulk_density must be positive"):
        polygonal_estimation(df_holes, bulk_density=-1.0, boundary=boundary)


def test_polygonal_resource_summary(sample_polygons_df):
    summary = polygonal_resource_summary(sample_polygons_df)
    
    assert summary["total_tonnes"] == pytest.approx(108000.0)
    assert summary["total_area_m2"] == pytest.approx(4000.0)
    assert summary["total_volume_m3"] == pytest.approx(40000.0)
    assert summary["drillhole_count"] == 3
    assert summary["mean_polygon_area_m2"] == pytest.approx(4000.0 / 3)

    # Weighted mean grade: (27k*1.0 + 54k*2.0 + 27k*0.5) / 108k = 148,500 / 108,000 = 1.375
    # Contained metal with metal_factor=0.01: 1485.0 tonnes
    assert summary["contained_metal"] == pytest.approx(1485.0)
    assert summary["mean_grade"] == pytest.approx(1.375)

    # Test text formatting
    formatted = format_polygonal_summary(summary, metal_unit="t Cu")
    assert "POLYGONAL IN-SITU RESOURCE SUMMARY" in formatted
    assert "108,000.0 tonnes" in formatted
    assert "1.3750 %" in formatted
    assert "1,485.0 t Cu" in formatted


def test_polygonal_resource_summary_empty():
    empty_df = pd.DataFrame(columns=["tonnes", "grade", "area_m2", "volume_m3"])
    summary = polygonal_resource_summary(empty_df)
    assert summary["total_tonnes"] == 0.0
    assert summary["mean_grade"] == 0.0
    assert summary["contained_metal"] == 0.0
    assert summary["drillhole_count"] == 0


def test_grade_tonnage_table(sample_polygons_df):
    gt = grade_tonnage_table(sample_polygons_df, cutoffs=[0.0, 0.75, 1.5, 3.0])

    # Cutoff 0.0: all 3 polygons (108k tonnes, 1.375 grade)
    row_0 = gt.loc[0.0]
    assert row_0["ore_tonnes"] == pytest.approx(108000.0)
    assert row_0["ore_grade"] == pytest.approx(1.375)
    assert row_0["contained_metal"] == pytest.approx(1485.0)
    assert row_0["internal_waste_tonnes"] == 0.0
    assert row_0["internal_waste_ratio"] == 0.0
    assert row_0["ore_recovery_pct"] == pytest.approx(100.0)
    assert row_0["metal_recovery_pct"] == pytest.approx(100.0)

    # Cutoff 0.75: DH01 (1.0) and DH02 (2.0) -> 81k tonnes
    row_075 = gt.loc[0.75]
    assert row_075["ore_tonnes"] == pytest.approx(81000.0)
    assert row_075["internal_waste_tonnes"] == pytest.approx(27000.0)
    assert row_075["internal_waste_ratio"] == pytest.approx(27000.0 / 81000.0)
    # Ore grade = (27k*1 + 54k*2) / 81k = 135k / 81k = 1.6667
    assert row_075["ore_grade"] == pytest.approx(1.66666667)
    assert row_075["contained_metal"] == pytest.approx(1350.0)
    assert row_075["metal_recovery_pct"] == pytest.approx(1350.0 / 1485.0 * 100.0)

    # Cutoff 1.5: only DH02 (54k tonnes, 2.0 grade)
    row_15 = gt.loc[1.5]
    assert row_15["ore_tonnes"] == pytest.approx(54000.0)
    assert row_15["contained_metal"] == pytest.approx(1080.0)
    assert row_15["ore_grade"] == pytest.approx(2.0)

    # Cutoff 3.0: exceeds all assays -> 0 ore
    row_30 = gt.loc[3.0]
    assert row_30["ore_tonnes"] == 0.0
    assert row_30["ore_grade"] == 0.0
    assert row_30["internal_waste_tonnes"] == pytest.approx(108000.0)
    assert np.isinf(row_30["internal_waste_ratio"])

    # Test open-pit stripping ratio when external wall-rock waste is provided
    # 162k external waste + 27k internal subgrade = 189k total waste / 81k ore = 2.333
    gt_pit = grade_tonnage_table(
        sample_polygons_df, cutoffs=[0.75], external_waste_tonnes=162000.0
    )
    assert "strip_ratio" in gt_pit.columns
    assert gt_pit.loc[0.75, "strip_ratio"] == pytest.approx((162000.0 + 27000.0) / 81000.0)


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


def test_classify_resources_by_drill_spacing():
    # 3 samples in 2D
    samples = np.array([[10.0, 10.0], [10.0, 30.0], [30.0, 20.0]])
    # Points:
    # pt0: close to all 3 samples (at (15, 20), dists to all <= 20)
    # pt1: close to 1 sample within 25m, but 3 samples within 50m (at (10, 50))
    # pt2: far from all samples (at (100, 100))
    pts = np.array([
        [15.0, 20.0],
        [10.0, 50.0],
        [100.0, 100.0],
    ])

    # 1. Standard classification with min_holes_measured=3, min_holes_indicated=2
    cats = classify_resources_by_drill_spacing(
        grid_points=pts,
        samples_xy=samples,
        max_radius_measured=25.0,
        max_radius_indicated=50.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        max_radius_inferred=80.0,
    )
    assert cats[0] == "Measured"     # 3 samples within 25m
    assert cats[1] == "Indicated"    # 2 samples within 50m (samples 0 and 1)
    assert cats[2] == "Unclassified" # >80m from all samples

    # 2. When max_radius_inferred is None, pt2 defaults to Inferred
    cats_inf = classify_resources_by_drill_spacing(
        grid_points=pts,
        samples_xy=samples,
        max_radius_measured=25.0,
        max_radius_indicated=50.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        max_radius_inferred=None,
    )
    assert cats_inf[2] == "Inferred"

    # 3. is_interpolated mask constraint
    cats_masked = classify_resources_by_drill_spacing(
        grid_points=pts,
        samples_xy=samples,
        max_radius_measured=25.0,
        max_radius_indicated=50.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        is_interpolated=np.array([False, True, False]),
    )
    assert cats_masked[0] == "Inferred"  # Even though close to 3 holes, not interpolated -> capped at Inferred
    assert cats_masked[1] == "Indicated" # Interpolated and satisfies Indicated

    # 4. 3D grid points support
    pts_3d = np.array([[10.0, 10.0, 5.0], [50.0, 50.0, 5.0]])
    samples_3d = np.array([[10.0, 10.0, 5.0], [10.0, 10.0, 10.0], [10.0, 10.0, 0.0]])
    cats_3d = classify_resources_by_drill_spacing(
        grid_points=pts_3d,
        samples_xy=samples_3d,
        max_radius_measured=10.0,
        max_radius_indicated=20.0,
        min_holes_measured=3,
        min_holes_indicated=2,
    )
    assert cats_3d[0] == "Measured"
    assert cats_3d[1] == "Inferred"

    # 5. Empty inputs and edge cases
    empty_cats = classify_resources_by_drill_spacing(
        np.empty((0, 2)), samples, max_radius_measured=25.0, max_radius_indicated=50.0
    )
    assert len(empty_cats) == 0

    no_samples_cats = classify_resources_by_drill_spacing(
        pts, np.empty((0, 2)), max_radius_measured=25.0, max_radius_indicated=50.0
    )
    assert np.all(no_samples_cats == "Unclassified")

    # 6. Validations
    with pytest.raises(TypeError):
        classify_resources_by_drill_spacing(pts, samples)  # missing required radii
    with pytest.raises(ValueError, match="must have shape"):
        classify_resources_by_drill_spacing(
            pts, np.array([[10.0, 10.0, 10.0]]), max_radius_measured=25.0, max_radius_indicated=50.0
        )
    with pytest.raises(ValueError, match="Search radii"):
        classify_resources_by_drill_spacing(
            pts, samples, max_radius_measured=-5.0, max_radius_indicated=50.0
        )
    with pytest.raises(ValueError, match=r"max_radius_measured .* must be <="):
        classify_resources_by_drill_spacing(
            pts, samples, max_radius_measured=50.0, max_radius_indicated=20.0
        )
    with pytest.raises(ValueError, match=r"min_holes_measured .* must be >="):
        classify_resources_by_drill_spacing(
            pts, samples, max_radius_measured=25.0, max_radius_indicated=50.0, min_holes_measured=1, min_holes_indicated=3
        )


def test_classify_resources_by_sor():
    sor = np.array([0.95, 0.65, 0.35, 1.20, np.nan])
    ke = np.array([0.70, 0.40, -0.10, 0.85, 0.0])

    cats = classify_resources_by_sor(
        slopes_of_regression=sor,
        threshold_measured=0.80,
        threshold_indicated=0.50,
        kriging_efficiencies=ke,
        max_slope_measured=1.05,
        min_kriging_efficiency=0.0,
    )

    assert cats[0] == "Measured"      # SoR 0.95, KE 0.70
    assert cats[1] == "Indicated"     # SoR 0.65, KE 0.40
    assert cats[2] == "Inferred"      # SoR 0.35
    assert cats[3] == "Inferred"      # SoR 1.20 > max_slope_measured -> not Measured
    assert cats[4] == "Unclassified"  # NaN SoR

    # KE disqualification: high SoR but negative KE
    sor_high = np.array([0.90])
    ke_neg = np.array([-0.20])
    cats_ke_disqual = classify_resources_by_sor(
        slopes_of_regression=sor_high,
        threshold_measured=0.80,
        threshold_indicated=0.50,
        kriging_efficiencies=ke_neg,
        min_kriging_efficiency=0.0,
    )
    assert cats_ke_disqual[0] == "Inferred"

    # Validations
    with pytest.raises(TypeError):
        classify_resources_by_sor(sor)  # missing required thresholds
    assert len(classify_resources_by_sor(np.array([]), threshold_measured=0.80, threshold_indicated=0.50)) == 0
    with pytest.raises(ValueError, match=r"threshold_measured .* must be >"):
        classify_resources_by_sor(sor, threshold_measured=0.40, threshold_indicated=0.60)
    with pytest.raises(ValueError, match=r"max_slope_measured .* must be >="):
        classify_resources_by_sor(sor, threshold_measured=0.80, threshold_indicated=0.50, max_slope_measured=0.75)
    with pytest.raises(ValueError, match="Shape mismatch"):
        classify_resources_by_sor(
            sor, threshold_measured=0.80, threshold_indicated=0.50, kriging_efficiencies=np.array([0.5, 0.5])
        )


def test_classify_resources_by_kriging_variance():
    variances = np.array([0.05, 0.15, 0.45, 0.85, -0.01, np.nan])

    cats = classify_resources_by_kriging_variance(
        kriging_variances=variances,
        variance_threshold_measured=0.10,
        variance_threshold_indicated=0.30,
        variance_threshold_inferred=0.70,
    )

    assert cats[0] == "Measured"     # 0.05 <= 0.10
    assert cats[1] == "Indicated"    # 0.10 < 0.15 <= 0.30
    assert cats[2] == "Inferred"     # 0.30 < 0.45 <= 0.70
    assert cats[3] == "Unclassified" # 0.85 > 0.70
    assert cats[4] == "Unclassified" # Negative variance
    assert cats[5] == "Unclassified" # NaN variance

    # Without inferred threshold, 0.85 defaults to Inferred
    cats_no_inf = classify_resources_by_kriging_variance(
        kriging_variances=variances,
        variance_threshold_measured=0.10,
        variance_threshold_indicated=0.30,
        variance_threshold_inferred=None,
    )
    assert cats_no_inf[3] == "Inferred"

    # Validations
    assert len(classify_resources_by_kriging_variance(np.array([]), 0.1, 0.2)) == 0
    with pytest.raises(ValueError, match="Variance thresholds must be strictly positive"):
        classify_resources_by_kriging_variance(variances, -0.1, 0.2)
    with pytest.raises(ValueError, match=r"variance_threshold_measured .* must be <"):
        classify_resources_by_kriging_variance(variances, 0.3, 0.2)
    with pytest.raises(ValueError, match=r"variance_threshold_inferred .* must be >"):
        classify_resources_by_kriging_variance(variances, 0.1, 0.3, variance_threshold_inferred=0.25)


def test_classify_mineral_resources_unified():
    # Grid points and samples
    samples = np.array([[10.0, 10.0], [10.0, 30.0], [30.0, 20.0]])
    pts = np.array([
        [15.0, 20.0],  # Close to 3 samples -> Spacing: Measured (3)
        [10.0, 50.0],  # Close to 1 sample within 25m, 3 samples within 50m -> Spacing: Indicated (2)
        [80.0, 80.0],  # Extrapolated (>50m from all samples) -> Spacing: Inferred (1)
    ])

    # 1. Pure drill spacing mode
    cats_spacing = classify_mineral_resources(
        grid_points=pts,
        samples_xy=samples,
        max_radius_measured=25.0,
        max_radius_indicated=50.0,
        min_holes_measured=3,
        min_holes_indicated=2,
    )
    assert cats_spacing[0] == "Measured"
    assert cats_spacing[1] == "Indicated"
    assert cats_spacing[2] == "Inferred"

    # 2. Multi-criteria mode: Spacing + Kriging Variance (conservative downgrade rule)
    # Block 0: Spacing is Measured, but high variance (0.25) -> downgraded to Indicated
    # Block 1: Spacing is Indicated, low variance (0.05) -> still Indicated (cannot exceed spacing)
    # Block 2: Spacing is Inferred, low variance (0.05) -> still Inferred
    ok_vars = np.array([0.25, 0.05, 0.05])
    cats_multi = classify_mineral_resources(
        grid_points=pts,
        samples_xy=samples,
        kriging_variances=ok_vars,
        max_radius_measured=25.0,
        max_radius_indicated=50.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        variance_threshold_measured=0.10,
        variance_threshold_indicated=0.30,
    )
    assert cats_multi[0] == "Indicated"  # Downgraded from Measured due to high variance
    assert cats_multi[1] == "Indicated"
    assert cats_multi[2] == "Inferred"

    # 3. Triple-criteria mode: Spacing + Variance + Slope of Regression
    # Block 0 with perfect variance and perfect SoR -> Measured
    # Block 1 with low SoR (0.30) -> downgraded to Inferred
    ok_vars_ideal = np.array([0.05, 0.05, 0.05])
    sor = np.array([0.95, 0.30, 0.95])
    ke = np.array([0.80, 0.20, 0.80])
    cats_triple = classify_mineral_resources(
        grid_points=pts,
        samples_xy=samples,
        max_radius_measured=25.0,
        max_radius_indicated=50.0,
        kriging_variances=ok_vars_ideal,
        variance_threshold_measured=0.10,
        variance_threshold_indicated=0.30,
        slopes_of_regression=sor,
        sor_threshold_measured=0.80,
        sor_threshold_indicated=0.50,
        kriging_efficiencies=ke,
    )
    assert cats_triple[0] == "Measured"
    assert cats_triple[1] == "Inferred"  # Downgraded from Indicated to Inferred because SoR < 0.50

    # 4. Extrapolation mask
    cats_interp = classify_mineral_resources(
        grid_points=pts,
        samples_xy=samples,
        max_radius_measured=25.0,
        max_radius_indicated=50.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        is_interpolated=np.array([False, True, True]),
    )
    assert cats_interp[0] == "Inferred"  # Outside interpolation mask -> capped at Inferred

    # 5. Error handling and validations
    with pytest.raises(ValueError, match="At least one classification criterion"):
        classify_mineral_resources(grid_points=pts)
    with pytest.raises(ValueError, match="max_radius_measured and max_radius_indicated must be explicitly specified"):
        classify_mineral_resources(grid_points=pts, samples_xy=samples)
    with pytest.raises(ValueError, match="sor_threshold_measured and sor_threshold_indicated must be explicitly specified"):
        classify_mineral_resources(grid_points=pts, slopes_of_regression=sor)
    with pytest.raises(ValueError, match="variance_threshold_measured and variance_threshold_indicated must be specified"):
        classify_mineral_resources(grid_points=pts, kriging_variances=ok_vars)
    with pytest.raises(ValueError, match="kriging_variances must be provided when variance thresholds"):
        classify_mineral_resources(grid_points=pts, variance_threshold_measured=0.10)
    with pytest.raises(ValueError, match="Shape mismatch"):
        classify_mineral_resources(
            grid_points=pts,
            kriging_variances=np.array([0.1, 0.2]),
            variance_threshold_measured=0.1,
            variance_threshold_indicated=0.2,
        )

    # Empty inputs
    assert len(classify_mineral_resources(np.empty((0, 2)), samples_xy=samples, max_radius_measured=25.0, max_radius_indicated=50.0)) == 0


def test_smooth_resource_categories_spotted_dog():
    # 3x3 regular grid on 10m spacing
    # Center block (10, 10) is an isolated "Measured" block in a sea of "Indicated" blocks (Classic Spotted Dog)
    pts = []
    cats = []
    for x in (0.0, 10.0, 20.0):
        for y in (0.0, 10.0, 20.0):
            pts.append([x, y])
            if x == 10.0 and y == 10.0:
                cats.append("Measured")  # Isolated spotted dog
            else:
                cats.append("Indicated")

    pts = np.array(pts, dtype=float)
    cats = np.array(cats, dtype=object)

    # 1. Majority smoothing eliminates the isolated Measured block
    smoothed = smooth_resource_categories(pts, cats, smoothing_radius=15.0)
    assert cats[4] == "Measured"
    assert smoothed[4] == "Indicated"
    assert all(c == "Indicated" for c in smoothed)

    # 2. Conservative downgrade principle:
    # A single Inferred block surrounded by 8 Measured blocks is NOT upgraded to Measured
    cats_hole = cats.copy()
    cats_hole[0:4] = "Measured"
    cats_hole[5:9] = "Measured"
    cats_hole[4] = "Inferred"  # Center is Inferred
    smoothed_hole = smooth_resource_categories(pts, cats_hole, smoothing_radius=15.0, downgrade_isolated=True)
    assert smoothed_hole[4] == "Inferred"  # Preserves conservative classification

    # If downgrade_isolated=False, full mode filter can fill the hole
    smoothed_filled = smooth_resource_categories(pts, cats_hole, smoothing_radius=15.0, downgrade_isolated=False)
    assert smoothed_filled[4] == "Measured"

    # 3. Minimum cluster size filtering:
    # Cluster of 2 Measured blocks surrounded by Indicated; with min_cluster_size=4, both are downgraded
    cats_pair = np.array(["Indicated"] * len(pts), dtype=object)
    cats_pair[0] = "Measured"
    cats_pair[1] = "Measured"  # Pair of only 2 blocks
    smoothed_clust = smooth_resource_categories(pts, cats_pair, min_cluster_size=4)
    assert smoothed_clust[0] == "Indicated"
    assert smoothed_clust[1] == "Indicated"

    # 4. 3D block model coordinates support:
    pts3d = np.array([
        [0, 0, 0], [10, 0, 0], [20, 0, 0],
        [0, 10, 0], [10, 10, 0], [20, 10, 0],
        [0, 20, 0], [10, 20, 0], [20, 20, 0],
    ], dtype=float)
    smoothed_3d = smooth_resource_categories(pts3d, cats, smoothing_radius=15.0)
    assert smoothed_3d[4] == "Indicated"

    # 5. Input validation and edge cases:
    assert len(smooth_resource_categories(np.empty((0, 2)), [])) == 0
    with pytest.raises(ValueError, match="Shape mismatch"):
        smooth_resource_categories(pts, ["Measured"])
    with pytest.raises(ValueError, match="smoothing_radius must be strictly positive"):
        smooth_resource_categories(pts, cats, smoothing_radius=-5.0)
    with pytest.raises(ValueError, match="k_neighbors must be strictly positive"):
        smooth_resource_categories(pts, cats, k_neighbors=0)
    with pytest.raises(ValueError, match="min_cluster_size must be >= 1"):
        smooth_resource_categories(pts, cats, min_cluster_size=0)


def test_classify_resources_with_spotted_dog_smoothing():
    # 3x3 block model
    pts = np.array([
        [0.0, 0.0], [10.0, 0.0], [20.0, 0.0],
        [0.0, 10.0], [10.0, 10.0], [20.0, 10.0],
        [0.0, 20.0], [10.0, 20.0], [20.0, 20.0],
    ])
    # Center block has low kriging variance (0.05 <= 0.10 -> Measured), outer blocks have 0.20 (Indicated)
    vars_grid = np.array([0.20, 0.20, 0.20, 0.20, 0.05, 0.20, 0.20, 0.20, 0.20])

    # 1. Standalone classify_resources_by_kriging_variance issues warning without smoothing
    with pytest.warns(UserWarning, match="CIM MRMR Best Practice Guidelines"):
        cats_raw = classify_resources_by_kriging_variance(
            kriging_variances=vars_grid,
            variance_threshold_measured=0.10,
            variance_threshold_indicated=0.30,
        )
    assert cats_raw[4] == "Measured"  # Center is raw spotted dog

    # 2. When grid_points and smoothing_radius are provided to classify_resources_by_kriging_variance:
    # Spotted dog is smoothed out!
    cats_smooth = classify_resources_by_kriging_variance(
        kriging_variances=vars_grid,
        variance_threshold_measured=0.10,
        variance_threshold_indicated=0.30,
        grid_points=pts,
        smoothing_radius=15.0,
        warn_standalone=False,
    )
    assert cats_smooth[4] == "Indicated"

    # 3. classify_mineral_resources with smoothing_radius:
    samples = np.array([[10.0, 10.0]])
    cats_multi_smooth = classify_mineral_resources(
        grid_points=pts,
        samples_xy=samples,
        kriging_variances=vars_grid,
        variance_threshold_measured=0.10,
        variance_threshold_indicated=0.30,
        max_radius_measured=15.0,
        max_radius_indicated=35.0,
        min_holes_measured=1,
        min_holes_indicated=1,
        smoothing_radius=15.0,
    )
    assert cats_multi_smooth[4] == "Indicated"

    # 4. classify_mineral_resources issues warning if only kriging_variances is passed
    with pytest.warns(UserWarning, match="CIM MRMR Best Practice Guidelines"):
        classify_mineral_resources(
            grid_points=pts,
            kriging_variances=vars_grid,
            variance_threshold_measured=0.10,
            variance_threshold_indicated=0.30,
        )


def test_kriging_quality_metrics():
    # 1. Standard calculation without lagrange multipliers
    bv = 0.5
    variances = np.array([0.0, 0.1, 0.25, 0.5, 0.6])
    ke, sor = kriging_quality_metrics(variances, block_dispersion_variance=bv)

    assert sor is None
    expected_ke = (bv - variances) / bv
    np.testing.assert_allclose(ke, expected_ke)
    assert ke[0] == 1.0  # Zero variance -> 100% efficiency
    assert ke[3] == 0.0  # Estimation variance = block variance -> 0% efficiency
    assert ke[4] < 0.0  # Poorly estimated -> negative efficiency

    # 2. Calculation with lagrange multipliers (SoR)
    mus = np.array([0.0, 0.02, -0.05, 0.1, 0.1])
    ke, sor = kriging_quality_metrics(
        variances, block_dispersion_variance=bv, lagrange_multipliers=mus
    )
    assert sor is not None
    assert len(sor) == len(variances)
    # Block 0: bv=0.5, var=0.0, mu=0.0 -> SoR = 0.5 / 0.5 = 1.0
    assert np.isclose(sor[0], 1.0)
    # Block 1: bv=0.5, var=0.1, mu=0.02 -> (0.4 - 0.02) / (0.4 - 0.04) = 0.38 / 0.36
    assert np.isclose(sor[1], 0.38 / 0.36)
    # Block 2: bv=0.5, var=0.25, mu=-0.05 -> (0.25 - (-0.05)) / (0.25 - 2*(-0.05)) = 0.30 / 0.35
    assert np.isclose(sor[2], 0.30 / 0.35)

    # 3. Edge case: empty input arrays
    ke_empty, sor_empty = kriging_quality_metrics(
        [], block_dispersion_variance=0.5
    )
    assert len(ke_empty) == 0
    assert sor_empty is None

    ke_empty2, sor_empty2 = kriging_quality_metrics(
        [], block_dispersion_variance=0.5, lagrange_multipliers=[]
    )
    assert len(ke_empty2) == 0
    assert len(sor_empty2) == 0

    # 4. Edge case: non-positive or invalid dispersion variance
    with pytest.raises(ValueError, match="strictly positive"):
        kriging_quality_metrics(variances, block_dispersion_variance=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        kriging_quality_metrics(variances, block_dispersion_variance=-0.2)

    with pytest.raises(ValueError, match="valid numeric scalar"):
        kriging_quality_metrics(variances, block_dispersion_variance=np.nan)

    # 5. Edge case: shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        kriging_quality_metrics(
            variances,
            block_dispersion_variance=bv,
            lagrange_multipliers=np.array([0.01, 0.02]),
        )

    # 6. Edge case: zero denominator in SoR guarded with NaN
    # Denominator: bv - var - 2*mu = 0.5 - 0.7 - 2*(-0.1) = 0.0
    _, sor_div_zero = kriging_quality_metrics(
        np.array([0.7]),
        block_dispersion_variance=0.5,
        lagrange_multipliers=np.array([-0.1]),
    )
    assert np.isnan(sor_div_zero[0])


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
    assert len(footnotes) == 7
    # Verify mandatory footnote 2 exists
    assert any("Totals may not sum due to rounding" in fn for fn in footnotes)
    # Verify RPEEE footnote exists
    assert any("Lerchs-Grossmann" in fn for fn in footnotes)
    assert any("$3.85/lb Cu" in fn for fn in footnotes)
    assert any("88.5%" in fn for fn in footnotes)
    # Verify mandatory inclusive/exclusive declaration footnote exists
    assert statement.attrs["reporting_basis"] == "exclusive"
    assert any("exclusive of Mineral Reserves" in fn for fn in footnotes)
    # Verify Point of Reference declaration exists
    assert statement.attrs["point_of_reference"] == "In situ"
    assert any("Point of Reference: In situ" in fn for fn in footnotes)

    # Test custom point of reference
    stmt_custom_por = format_resource_statement(
        block_df,
        cutoff_grade=0.50,
        point_of_reference="Run-of-Mine / Plant feed",
    )
    assert stmt_custom_por.attrs["point_of_reference"] == "Run-of-Mine / Plant feed"
    assert any("Point of Reference: Run-of-Mine / Plant feed" in fn for fn in stmt_custom_por.attrs["footnotes"])

    # Test invalid point of reference raises ValueError
    with pytest.raises(ValueError, match="point_of_reference must be a non-empty string"):
        format_resource_statement(block_df, cutoff_grade=0.50, point_of_reference="")

    # Test inclusive reporting basis
    stmt_incl = format_resource_statement(
        block_df,
        cutoff_grade=0.50,
        reporting_basis="inclusive",
    )
    assert stmt_incl.attrs["reporting_basis"] == "inclusive"
    assert any("inclusive of Mineral Reserves" in fn for fn in stmt_incl.attrs["footnotes"])

    # Test invalid reporting basis raises ValueError
    with pytest.raises(ValueError, match="Invalid reporting_basis"):
        format_resource_statement(block_df, cutoff_grade=0.50, reporting_basis="invalid_basis")


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

    # Test automatic cell size range calculation (cell_sizes=None)
    auto_weights, auto_df, auto_opt_cs = cell_declustering(df, min_mean=True)
    assert len(auto_weights) == 5
    assert len(auto_df) == 45
    assert auto_df["cell_size"].iloc[0] < 10.0
    assert auto_df["cell_size"].iloc[-1] > 200.0
    # U-shape verification: both ends should approach naive mean
    assert np.isclose(auto_df["declustered_mean"].iloc[0], naive_mean, atol=1e-3)
    assert np.isclose(auto_df["declustered_mean"].iloc[-1], naive_mean, atol=0.05)



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
    # Net price = $3.80/lb (payable_metal_factor = 1.0)
    # Revenue per 1% Cu = 3.80 * 0.88 * 22.0462 = $73.7225 / (% Cu)
    # Breakeven cost = 12 + 2 + 2.50 = $16.50/t -> cutoff = 16.50 / 73.7225 = 0.2238% Cu
    # Marginal cost = 12 + 2 = $14.00/t -> cutoff = 14.00 / 73.7225 = 0.1899% Cu

    # Missing payable_metal_factor raises TypeError (required parameter, no default)
    with pytest.raises(TypeError):
        calculate_cut_off_grade(  # type: ignore[call-arg]
            processing_cost=12.0,
            ga_cost=2.0,
            mining_cost=2.50,
            commodity_price=3.80,
            metallurgical_recovery=88.0,
            metal_conversion_factor=22.0462,
        )

    co_be = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=1.0,
        metal_conversion_factor=22.0462,
    )
    assert np.isclose(co_be, 0.2238, atol=0.001)

    # Percentage input for payable metal factor (100.0 == 1.0)
    co_be_pct = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=100.0,
        metal_conversion_factor=22.0462,
    )
    assert np.isclose(co_be, co_be_pct)

    co_marg = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=None,  # Marginal / Internal (sunk mining cost)
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=1.0,
        metal_conversion_factor=22.0462,
    )
    assert np.isclose(co_marg, 0.1899, atol=0.001)
    assert co_marg < co_be

    # Payable metal factor (e.g. 95% off-take payability) increases required cut-off grade
    co_pay95 = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=0.95,
        metal_conversion_factor=22.0462,
    )
    assert np.isclose(co_pay95, 0.2238 / 0.95, atol=0.001)
    assert co_pay95 > co_be

    # Sustaining capital (Reserve LOM economics) increases required cut-off grade
    co_sust = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=1.0,
        sustaining_capital=1.50,  # $1.50/t ore sustaining capital
        metal_conversion_factor=22.0462,
    )
    # Total cost = 16.50 + 1.50 = $18.00/t -> 18.00 / 73.7225 = 0.24416%
    assert np.isclose(co_sust, 18.00 / 73.7225, atol=0.001)
    assert co_sust > co_be

    # Mining dilution adjustment: in-situ equivalent cut-off
    # Mill feed cutoff = 0.2238%, 10% dilution with 0.0% grade -> in-situ cutoff = 0.2238 / (1 - 0.10) = 0.2487%
    co_dil = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=1.0,
        mining_dilution_pct=10.0,
        dilution_grade=0.0,
        metal_conversion_factor=22.0462,
    )
    assert np.isclose(co_dil, 0.2238 / 0.90, atol=0.001)
    assert co_dil > co_be

    # Dilution with non-zero grade (e.g. wall rock contains 0.10% Cu)
    co_dil_mineralized = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=1.0,
        mining_dilution_pct=10.0,
        dilution_grade=0.10,
        metal_conversion_factor=22.0462,
    )
    # (0.2238 - 0.10 * 0.10) / 0.90 = (0.2238 - 0.01) / 0.90 = 0.2138 / 0.90 = 0.2376%
    assert co_dil_mineralized < co_dil
    assert co_dil_mineralized > co_be

    # Royalties and selling deductions reduce net price and increase cut-off grade
    co_royalty = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        selling_cost=0.30,  # $0.30/lb deduction
        royalty_pct=2.0,    # 2% NSR royalty
        metallurgical_recovery=88.0,
        payable_metal_factor=1.0,
        metal_conversion_factor=22.0462,
    )
    assert co_royalty > co_be

    # Validation errors
    with pytest.raises(ValueError, match="negative"):
        calculate_cut_off_grade(
            processing_cost=-5.0, ga_cost=2.0, commodity_price=3.80, metallurgical_recovery=88.0, payable_metal_factor=0.95
        )
    with pytest.raises(ValueError, match="positive"):
        calculate_cut_off_grade(
            processing_cost=12.0, ga_cost=2.0, commodity_price=-3.80, metallurgical_recovery=88.0, payable_metal_factor=0.95
        )
    with pytest.raises(ValueError, match="payable_metal_factor"):
        calculate_cut_off_grade(
            processing_cost=12.0, ga_cost=2.0, commodity_price=3.80, metallurgical_recovery=88.0, payable_metal_factor=0.0
        )
    with pytest.raises(ValueError, match="payable_metal_factor"):
        calculate_cut_off_grade(
            processing_cost=12.0, ga_cost=2.0, commodity_price=3.80, metallurgical_recovery=88.0, payable_metal_factor=150.0
        )
    with pytest.raises(ValueError, match="(?i)sustaining capital"):
        calculate_cut_off_grade(
            processing_cost=12.0, ga_cost=2.0, commodity_price=3.80, metallurgical_recovery=88.0, payable_metal_factor=0.95,
            sustaining_capital=-1.0
        )
    with pytest.raises(ValueError, match="mining_dilution_pct"):
        calculate_cut_off_grade(
            processing_cost=12.0, ga_cost=2.0, commodity_price=3.80, metallurgical_recovery=88.0, payable_metal_factor=0.95,
            mining_dilution_pct=-5.0
        )
    with pytest.raises(ValueError, match="mining_dilution_pct"):
        calculate_cut_off_grade(
            processing_cost=12.0, ga_cost=2.0, commodity_price=3.80, metallurgical_recovery=88.0, payable_metal_factor=0.95,
            mining_dilution_pct=100.0
        )
    with pytest.raises(ValueError, match="dilution_grade"):
        calculate_cut_off_grade(
            processing_cost=12.0, ga_cost=2.0, commodity_price=3.80, metallurgical_recovery=88.0, payable_metal_factor=0.95,
            dilution_grade=-0.1
        )


def test_cut_off_grade_breakdown():
    # Test breakdown dictionary helper
    breakdown = cut_off_grade_breakdown(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=0.95,
        sustaining_capital=1.50,
        selling_cost=0.30,
        royalty_pct=2.0,
        metal_conversion_factor=22.0462,
        mining_dilution_pct=10.0,
        dilution_grade=0.05,
    )

    assert "mill_feed_cutoff_grade" in breakdown
    assert "in_situ_cutoff_grade" in breakdown
    assert "net_realized_price" in breakdown
    assert "revenue_per_grade_unit" in breakdown
    assert "operating_cost_per_tonne" in breakdown
    assert "total_cost_per_tonne" in breakdown
    assert "sustaining_capital_per_tonne" in breakdown
    assert "cutoff_type" in breakdown
    assert breakdown["cutoff_type"] == "Breakeven Reserve (LOM)"

    # Operating cost = 12 + 2 + 2.50 = 16.50
    assert np.isclose(breakdown["operating_cost_per_tonne"], 16.50)
    # Total cost = 16.50 + 1.50 = 18.00
    assert np.isclose(breakdown["total_cost_per_tonne"], 18.00)

    # In-situ cutoff should equal direct call to calculate_cut_off_grade with same params
    direct_cog = calculate_cut_off_grade(
        processing_cost=12.0,
        ga_cost=2.0,
        mining_cost=2.50,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        payable_metal_factor=0.95,
        sustaining_capital=1.50,
        selling_cost=0.30,
        royalty_pct=2.0,
        metal_conversion_factor=22.0462,
        mining_dilution_pct=10.0,
        dilution_grade=0.05,
    )
    assert np.isclose(breakdown["in_situ_cutoff_grade"], direct_cog)


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

    # allow_inferred is completely removed from the API (per CONTRIBUTING.md) and raises TypeError
    with pytest.raises(TypeError, match="unexpected keyword argument 'allow_inferred'"):
        convert_resource_to_reserve(
            resource_df,
            mining_dilution_pct=dilution_pct,
            mining_recovery_pct=recovery_pct,
            cutoff_grade=cutoff,
            allow_inferred=True,
        )

    # Verification that unclassified/waste blocks are also excluded
    res_with_unclass = pd.DataFrame({
        "category": ["Measured", "Indicated", "Inferred", "Unclassified"],
        "grade": [1.20, 0.90, 1.50, 1.80],  # 1.80 is high grade but unclassified
        "tonnes": [1000.0, 2000.0, 5000.0, 3000.0],
    })
    res_out = convert_resource_to_reserve(
        res_with_unclass,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        cutoff_grade=cutoff,
    )
    assert len(res_out) == 2
    assert res_out.attrs["excluded_inferred_tonnes"] == 5000.0
    assert res_out.attrs["excluded_unclassified_tonnes"] == 3000.0


def test_convert_resource_to_reserve_dilution_cutoff_spoilage():
    # Demonstrates Item 2.1: In-situ blocks that meet cut-off in-situ (e.g. 0.52% >= 0.50%)
    # but fall below cut-off after dilution (0.52 / 1.10 = 0.4727% < 0.50%)
    resource_df = pd.DataFrame({
        "category": ["Measured", "Indicated", "Measured"],
        "grade": [1.20, 0.52, 0.45],  # 0.52 is in-situ above 0.50 cutoff, 0.45 is in-situ below
        "tonnes": [1000.0, 1000.0, 1000.0],
    })

    cutoff = 0.50
    dilution_pct = 10.0  # 10% zero-grade dilution
    recovery_pct = 95.0

    # 1. Default (ROM cut-off per CIM MRMR §7.2.1 and SME Handbook Ch. 6.1):
    # Block 1 (grade 0.52%):
    # Diluted head grade = (1000 * 0.52 + 100 * 0.0) / 1100 = 0.4727% < 0.50%
    # This block is sub-economic at the plant gate and must be excluded from reserves!
    res_rom = convert_resource_to_reserve(
        resource_df,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        cutoff_grade=cutoff,
        dilution_grade=0.0,
        cutoff_type="rom",
    )

    # Only Block 0 (1.20% -> 1.0909%) survives
    assert len(res_rom) == 1
    assert res_rom["reserve_category"].iloc[0] == "Proven Reserve"
    assert np.isclose(res_rom["rom_tonnes"].iloc[0], 1045.0)

    # Audit metadata verifies tracking of dilution spoilage
    # Block 1: ROM tonnes = 1000 * 1.10 * 0.95 = 1045t, contained metal = 1045 * 0.472727 * 0.01
    assert np.isclose(res_rom.attrs["excluded_subeconomic_diluted_tonnes"], 1045.0)
    assert np.isclose(res_rom.attrs["excluded_subeconomic_insitu_tonnes"], 1000.0)
    expected_spoilage_metal = 1045.0 * (0.52 / 1.10) * 0.01
    assert np.isclose(res_rom.attrs["excluded_subeconomic_diluted_metal"], expected_spoilage_metal, atol=1e-3)
    assert res_rom.attrs["cutoff_type"] == "rom"

    # 2. In-situ cut-off mode (e.g., when cutoff was pre-adjusted by practitioner):
    res_insitu = convert_resource_to_reserve(
        resource_df,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        cutoff_grade=cutoff,
        dilution_grade=0.0,
        cutoff_type="in_situ",
    )
    # Both Block 0 and Block 1 are included
    assert len(res_insitu) == 2
    assert res_insitu.attrs["excluded_subeconomic_diluted_tonnes"] == 0.0
    assert res_insitu.attrs["cutoff_type"] == "in_situ"

    # 3. Input validation:
    with pytest.raises(ValueError, match="cutoff_type must be 'rom' or 'in_situ'"):
        convert_resource_to_reserve(
            resource_df,
            mining_dilution_pct=dilution_pct,
            mining_recovery_pct=recovery_pct,
            cutoff_grade=cutoff,
            cutoff_type="invalid_type",  # type: ignore
        )

    with pytest.raises(ValueError, match="cutoff_grade cannot be negative"):
        convert_resource_to_reserve(
            resource_df,
            mining_dilution_pct=dilution_pct,
            mining_recovery_pct=recovery_pct,
            cutoff_grade=-0.1,
        )

    # 4. Scenario where ALL candidate blocks become sub-economic after dilution:
    all_marginal_df = pd.DataFrame({
        "category": ["Measured"],
        "grade": [0.52],
        "tonnes": [1000.0],
    })
    res_empty = convert_resource_to_reserve(
        all_marginal_df,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        cutoff_grade=cutoff,
    )
    assert len(res_empty) == 0
    assert np.isclose(res_empty.attrs["excluded_subeconomic_diluted_tonnes"], 1045.0)

    # 5. Waterfall reconciliation with dilution spoilage step:
    fig, (ax1, ax2) = plot_resource_to_reserve_waterfall(
        resource_df,
        res_rom,
        cutoff_grade=cutoff,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        dilution_grade=0.0,
    )
    assert fig is not None
    # Verify the x tick labels include Dilution Spoilage
    xticklabels = [label.get_text() for label in ax1.get_xticklabels()]
    assert any("Dilution Spoilage" in lbl for lbl in xticklabels)
    plt.close(fig)


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
    assert len(fns) == 6
    assert any("CIM Definition Standards" in f for f in fns)
    assert any("Mining Dilution = 6.0%" in f for f in fns)
    assert any("Mining Recovery = 94.0%" in f for f in fns)
    assert any("Metallurgical Recovery = 88.5%" in f for f in fns)
    assert any("Pre-Feasibility" in f for f in fns)
    assert stmt.attrs["point_of_reference"] == "Run-of-Mine (ROM) delivered to processing facility"
    assert any("Point of Reference: Run-of-Mine (ROM) delivered to processing facility" in f for f in fns)

    # Test custom point of reference
    stmt_custom_por = format_reserve_statement(
        reserve_df,
        cutoff_grade=0.50,
        mining_dilution_pct=6.0,
        mining_recovery_pct=94.0,
        commodity_price="$3.80/lb Cu",
        metallurgical_recovery=88.5,
        point_of_reference="Plant feed delivered to concentrator",
    )
    assert stmt_custom_por.attrs["point_of_reference"] == "Plant feed delivered to concentrator"
    assert any("Point of Reference: Plant feed delivered to concentrator" in f for f in stmt_custom_por.attrs["footnotes"])

    # Test invalid point of reference raises ValueError
    with pytest.raises(ValueError, match="point_of_reference must be a non-empty string"):
        format_reserve_statement(
            reserve_df,
            cutoff_grade=0.50,
            mining_dilution_pct=6.0,
            mining_recovery_pct=94.0,
            commodity_price="$3.80/lb Cu",
            metallurgical_recovery=88.5,
            point_of_reference="   ",
        )


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


def test_plot_resource_classification_map():
    block_model = pd.DataFrame({
        "x": [10.0, 20.0, 30.0, 40.0],
        "y": [10.0, 20.0, 30.0, 40.0],
        "category": ["Measured", "Indicated", "Inferred", "Unclassified"],
    })
    boundary = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)]
    drillholes = pd.DataFrame({"x": [15.0, 35.0], "y": [15.0, 35.0]})

    fig, ax = plot_resource_classification_map(
        block_model,
        boundary=boundary,
        drillholes=drillholes,
        title="Test Resource Classification Map",
    )
    assert fig is not None
    assert ax.get_title() == "Test Resource Classification Map"
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
    assert rec_single.attrs["stockpile_mode"] == "Direct feed (no intermediate stockpiling)"


def test_reconcile_production_stockpile_adjustments():
    # Multi-period reconciliation demonstrating how stockpiling distorts unadjusted F2,
    # and how Harry Parker (2012) mass-balance adjustment restores true operational stability.
    # Month 1: Dig 100kt @ 1.0% Cu (1000t Cu). Mill only takes 70kt @ 1.0% Cu (700t Cu). 30kt stockpiled.
    # Month 2: Dig 50kt @ 1.0% Cu (500t Cu). Mill draws 30kt from stockpile + 50kt from pit = 80kt (800t Cu).
    reserve_df = pd.DataFrame({
        "period": ["M1", "M2"],
        "tonnes": [100_000.0, 50_000.0],
        "grade": [1.00, 1.00],
    })
    gc_df = pd.DataFrame({
        "period": ["M1", "M2"],
        "tonnes": [100_000.0, 50_000.0],
        "grade": [1.00, 1.00],
    })
    plant_df = pd.DataFrame({
        "period": ["M1", "M2"],
        "tonnes": [70_000.0, 80_000.0],
        "grade": [1.00, 1.00],
    })

    # Test Format 1: Opening & closing stockpile balances
    stockpile_balances = pd.DataFrame({
        "period": ["M1", "M2"],
        "opening_tonnes": [0.0, 30_000.0],
        "opening_grade": [0.0, 1.00],
        "closing_tonnes": [30_000.0, 0.0],
        "closing_grade": [1.00, 0.0],
    })

    rec_df = reconcile_production_to_reserve(
        reserve_data=reserve_df,
        plant_data=plant_df,
        grade_control_data=gc_df,
        stockpile_data=stockpile_balances,
        period_col="period",
        grade_unit="% Cu",
    )

    # Output rows: 2 periods + 1 "Total" row = 3 rows
    assert len(rec_df) == 3
    assert "stockpile_delta_tonnes" in rec_df.columns
    assert "plant_adj_tonnes" in rec_df.columns
    assert "f2_adj_metal_factor" in rec_df.columns
    assert "f3_adj_metal_factor" in rec_df.columns

    # Unadjusted F2 swings wildly: 0.70 in M1 and 1.60 in M2!
    assert np.isclose(rec_df.iloc[0]["f2_metal_factor"], 0.70)
    assert np.isclose(rec_df.iloc[1]["f2_metal_factor"], 1.60)

    # Stockpile-adjusted F2 is perfectly 1.00 in both months!
    assert np.isclose(rec_df.iloc[0]["f2_adj_metal_factor"], 1.00)
    assert np.isclose(rec_df.iloc[1]["f2_adj_metal_factor"], 1.00)
    assert np.isclose(rec_df.iloc[2]["f2_adj_metal_factor"], 1.00)

    # Mathematical identities hold: F1 * F2_adj == F3_adj
    for i in range(len(rec_df)):
        f1 = rec_df.iloc[i]["f1_metal_factor"]
        f2_adj = rec_df.iloc[i]["f2_adj_metal_factor"]
        f3_adj = rec_df.iloc[i]["f3_adj_metal_factor"]
        assert np.isclose(f1 * f2_adj, f3_adj, atol=1e-5)

        rt_adj = rec_df.iloc[i]["f2_adj_tonnes_ratio"]
        rg_adj = rec_df.iloc[i]["f2_adj_grade_ratio"]
        assert np.isclose(rt_adj * rg_adj, f2_adj, atol=1e-5)

    assert rec_df.attrs["stockpile_mode"] == "Inventory adjusted"
    assert np.isclose(rec_df.attrs["f3_adjusted_factor"], 1.00)
    assert "EXCELLENT" in rec_df.attrs["health_status"]

    # Test Format 2: Movement receipts and reclaims
    stockpile_movements = pd.DataFrame({
        "period": ["M1", "M2"],
        "added_tonnes": [30_000.0, 0.0],
        "added_grade": [1.00, 0.0],
        "reclaimed_tonnes": [0.0, 30_000.0],
        "reclaimed_grade": [0.0, 1.00],
    })
    rec_mvt = reconcile_production_to_reserve(
        reserve_df, plant_df, gc_df, stockpile_data=stockpile_movements, period_col="period"
    )
    assert np.isclose(rec_mvt.iloc[0]["f2_adj_metal_factor"], 1.00)
    assert np.isclose(rec_mvt.iloc[1]["f2_adj_metal_factor"], 1.00)

    # Test Format 3: Direct deltas
    stockpile_deltas = pd.DataFrame({
        "period": ["M1", "M2"],
        "delta_tonnes": [30_000.0, -30_000.0],
        "delta_grade": [1.00, 1.00],
    })
    rec_deltas = reconcile_production_to_reserve(
        reserve_df, plant_df, gc_df, stockpile_data=stockpile_deltas, period_col="period"
    )
    assert np.isclose(rec_deltas.iloc[0]["f2_adj_metal_factor"], 1.00)
    assert np.isclose(rec_deltas.iloc[1]["f2_adj_metal_factor"], 1.00)

    # Single-period dictionary with stockpile
    res_dict = {"tonnes": 100_000.0, "grade": 1.00}
    plant_dict = {"tonnes": 70_000.0, "grade": 1.00}
    gc_dict = {"tonnes": 100_000.0, "grade": 1.00}
    sp_dict = {"opening_tonnes": 0.0, "opening_grade": 0.0, "closing_tonnes": 30_000.0, "closing_grade": 1.00}
    rec_single_sp = reconcile_production_to_reserve(
        res_dict, plant_dict, grade_control_data=gc_dict, stockpile_data=sp_dict
    )
    assert np.isclose(rec_single_sp.iloc[0]["f2_metal_factor"], 0.70)
    assert np.isclose(rec_single_sp.iloc[0]["f2_adj_metal_factor"], 1.00)

    # Invalid stockpile data format raises ValueError
    with pytest.raises(ValueError, match="stockpile_data must contain"):
        reconcile_production_to_reserve(
            reserve_df, plant_df, gc_df, stockpile_data=pd.DataFrame({"invalid_col": [1, 2]}), period_col="period"
        )


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
        "plant_adj_tonnes": [1.02, 1.08, 1.18, 3.28],
        "plant_adj_grade": [0.99, 1.01, 0.99, 1.00],
        "plant_adj_metal": [0.0101, 0.0109, 0.0117, 0.0327],
        "f1_metal_factor": [1.01, 0.99, 0.975, 0.99],
        "f2_metal_factor": [0.98, 1.00, 1.00, 0.994],
        "f2_adj_metal_factor": [1.00, 1.00, 1.00, 1.00],
        "f3_metal_factor": [0.99, 0.99, 0.975, 0.985],
        "f3_adj_metal_factor": [1.01, 0.99, 0.975, 0.99],
    })
    rec_df.attrs["health_status"] = "EXCELLENT: Production is within +/-5% of reserve model (stockpile-adjusted)."

    fig, axes = plot_production_reconciliation(
        rec_df,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        metal_unit="kt",
        title="Mine-to-Mill Production Reconciliation with Stockpile Adjustments",
    )
    assert fig is not None
    assert len(axes) == 4
    assert "Tonnage" in axes[0].get_title()
    assert "Grade" in axes[1].get_title()
    assert "Metal" in axes[2].get_title()
    assert "Parker" in axes[3].get_title()
    plt.close(fig)

    # Test single-period plot with stockpile adjustments
    rec_single = pd.DataFrame([{
        "period": "Total",
        "reserve_tonnes": 1.0, "reserve_grade": 1.0, "reserve_metal": 0.01,
        "gc_tonnes": 1.0, "gc_grade": 1.0, "gc_metal": 0.01,
        "plant_tonnes": 0.7, "plant_grade": 1.0, "plant_metal": 0.007,
        "plant_adj_tonnes": 1.0, "plant_adj_grade": 1.0, "plant_adj_metal": 0.01,
        "f1_metal_factor": 1.0,
        "f2_metal_factor": 0.7,
        "f2_adj_metal_factor": 1.0,
        "f3_metal_factor": 0.7,
        "f3_adj_metal_factor": 1.0,
    }])
    fig_s, axes_s = plot_production_reconciliation(rec_single)
    assert fig_s is not None
    plt.close(fig_s)


def test_compute_etype_mtype_maps_and_smoothing_ratio():
    np.random.seed(42)
    n_pts = 100
    n_real = 30
    # Simulate realizations with spatial fluctuations
    realizations = np.random.normal(loc=1.20, scale=0.40, size=(n_pts, n_real))

    df_maps = compute_etype_mtype_maps(realizations, cutoff_grade=1.00)

    assert len(df_maps) == n_pts
    assert "e_type" in df_maps.columns
    assert "m_type" in df_maps.columns
    assert "conditional_std" in df_maps.columns
    assert "prob_exceedance" in df_maps.columns
    assert "p10" in df_maps.columns
    assert "p90" in df_maps.columns

    # Verify E-type is arithmetic mean and M-type is median
    assert np.allclose(df_maps["e_type"].to_numpy(), np.mean(realizations, axis=1))
    assert np.allclose(df_maps["m_type"].to_numpy(), np.median(realizations, axis=1))

    # Probability of exceedance above cutoff 1.00
    expected_prob = np.mean(realizations >= 1.00, axis=1)
    assert np.allclose(df_maps["prob_exceedance"].to_numpy(), expected_prob)

    # Verify smoothing effect: Var(E-type) must be significantly lower than average realization variance
    smoothing_ratio = df_maps.attrs["smoothing_ratio"]
    assert smoothing_ratio < 0.50, f"Expected smoothing ratio < 0.50, got {smoothing_ratio}"
    assert "limitations_note" in df_maps.attrs


def test_simulation_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        sequential_gaussian_simulation(
            samples_xy=np.zeros((5, 2)),
            sample_grades=np.ones(5),
            grid_points=np.zeros((10, 2)),
            sill=1.0,
            range_param=100.0,
        )

    with pytest.raises(NotImplementedError):
        plot_simulation_realizations_dashboard(
            realizations=np.zeros((10, 5)),
            grid_xy=np.zeros((10, 2)),
        )


def test_create_block_model():
    # Test create_block_model
    bm = create_block_model(
        origin=(100.0, 200.0, 0.0),
        block_size=(10.0, 10.0, 5.0),
        n_blocks=(5, 4, 3),  # 5 * 4 * 3 = 60 blocks
        default_density=2.70,
        default_domain="Porphyry",
    )
    assert len(bm) == 60
    assert set(bm.columns) >= {"x", "y", "z", "dx", "dy", "dz", "volume_m3", "density", "tonnes", "domain"}
    assert np.isclose(bm["volume_m3"].iloc[0], 10.0 * 10.0 * 5.0)
    assert np.isclose(bm["tonnes"].iloc[0], 500.0 * 2.70)
    assert (bm["domain"] == "Porphyry").all()
    assert bm["x"].min() == 105.0  # origin + 0.5 * dx
    assert bm["z"].max() == 12.5   # 0 + 2.5 * 5.0

    # Validations for required density
    with pytest.raises(TypeError):
        create_block_model(origin=(0, 0, 0), block_size=(10, 10, 5), n_blocks=(2, 2, 2))
    with pytest.raises(ValueError, match="default_density must be positive"):
        create_block_model(origin=(0, 0, 0), block_size=(10, 10, 5), n_blocks=(2, 2, 2), default_density=0.0)
    with pytest.raises(ValueError, match="default_density must be positive"):
        create_block_model(origin=(0, 0, 0), block_size=(10, 10, 5), n_blocks=(2, 2, 2), default_density=-1.0)

    # Test ordinary_kriging_block_estimation
    samples_xyz = np.array([
        [110.0, 210.0, 5.0],
        [120.0, 220.0, 5.0],
        [130.0, 210.0, 5.0],
        [140.0, 230.0, 5.0],
        [115.0, 225.0, 10.0],
    ])
    sample_grades = np.array([1.5, 2.0, 1.2, 0.8, 1.8])

    est, var, disp_var, lagrange = ordinary_kriging_block_estimation(
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        block_model=bm,
        sill=1.0,
        range_param=50.0,
        nugget=0.1,
        discretization=(3, 3, 2),
    )

    assert len(est) == 60
    assert len(var) == 60
    assert len(lagrange) == 60
    assert np.all(np.isfinite(est))
    assert np.all(var >= 0.0)
    assert np.all(np.isfinite(lagrange))
    # Support effect: Block dispersion variance must be strictly positive and less than total sill (1.1)
    assert 0.0 < disp_var < 1.1
    # Block kriging variance should not exceed the total sill
    assert np.all(var <= 1.1)

    # Verify kriging_quality_metrics on actual block kriging outputs
    ke, sor = kriging_quality_metrics(
        kriging_variances=var,
        block_dispersion_variance=disp_var,
        lagrange_multipliers=lagrange,
    )
    assert len(ke) == 60
    assert len(sor) == 60
    assert np.all(np.isfinite(ke))
    assert np.all(np.isfinite(sor))


def test_domain_constrained_estimation_all_methods():
    # Setup two sharply contrasting domains separated at x = 50m
    # Domain A (x < 50): High grade ~10.0% Cu
    # Domain B (x >= 50): Low grade ~1.0% Cu
    samples_xy = np.array([
        [20.0, 30.0],
        [40.0, 30.0],
        [60.0, 30.0],
        [80.0, 30.0],
    ])
    sample_grades = np.array([10.0, 10.0, 1.0, 1.0])
    sample_domains = np.array(["DomainA", "DomainA", "DomainB", "DomainB"])

    # Target points right next to the boundary
    grid_points = np.array([
        [49.0, 30.0],  # Right on the Domain A side
        [51.0, 30.0],  # Right on the Domain B side
    ])
    grid_domains = np.array(["DomainA", "DomainB"])

    # 1. Test IDW with domain segregation (no smearing across x=50!)
    idw_est, _ = inverse_distance_weighting(
        samples_xy=samples_xy,
        sample_grades=sample_grades,
        grid_points=grid_points,
        sample_domains=sample_domains,
        grid_domains=grid_domains,
    )
    assert np.isclose(idw_est[0], 10.0), f"Expected 10.0 for Domain A, got {idw_est[0]}"
    assert np.isclose(idw_est[1], 1.0), f"Expected 1.0 for Domain B, got {idw_est[1]}"

    # 2. Test Nearest Neighbor with domain segregation
    nn_est, _ = nearest_neighbor_grid_estimation(
        samples_xy=samples_xy,
        sample_grades=sample_grades,
        grid_points=grid_points,
        sample_domains=sample_domains,
        grid_domains=grid_domains,
    )
    assert np.isclose(nn_est[0], 10.0)
    assert np.isclose(nn_est[1], 1.0)

    # 3. Test Ordinary Kriging with domain segregation
    ok_est, ok_var = ordinary_kriging_grid_estimation(
        samples_xy=samples_xy,
        sample_grades=sample_grades,
        grid_points=grid_points,
        sill=1.0,
        range_param=100.0,
        sample_domains=sample_domains,
        grid_domains=grid_domains,
    )
    assert np.isclose(ok_est[0], 10.0)
    assert np.isclose(ok_est[1], 1.0)

    # 4. Test Simple Kriging with domain segregation and per-domain priors
    sk_est, sk_var = simple_kriging_grid_estimation(
        samples_xy=samples_xy,
        sample_grades=sample_grades,
        grid_points=grid_points,
        mean={"DomainA": 10.0, "DomainB": 1.0},
        sill={"DomainA": 1.0, "DomainB": 0.5},
        range_param=100.0,
        sample_domains=sample_domains,
        grid_domains=grid_domains,
    )
    assert np.isclose(sk_est[0], 10.0)
    assert np.isclose(sk_est[1], 1.0)

    # 5. Test Polygonal Estimation with domain_col
    dh_df = pd.DataFrame({
        "hole_id": ["DH01", "DH02", "DH03", "DH04"],
        "x": [20.0, 40.0, 60.0, 80.0],
        "y": [30.0, 30.0, 30.0, 30.0],
        "grade": [10.0, 10.0, 1.0, 1.0],
        "thickness": [5.0, 5.0, 5.0, 5.0],
        "domain": ["DomainA", "DomainA", "DomainB", "DomainB"],
    })
    poly_df = polygonal_estimation(dh_df, bulk_density=2.7, domain_col="domain")
    assert len(poly_df) == 4
    assert "domain" in poly_df.columns
    assert set(poly_df["domain"]) == {"DomainA", "DomainB"}
    assert (poly_df[poly_df["domain"] == "DomainA"]["grade"] == 10.0).all()
    assert (poly_df[poly_df["domain"] == "DomainB"]["grade"] == 1.0).all()

    # 6. Test Block Kriging with domain segregation
    bm_domains = pd.DataFrame({
        "x": [40.0, 60.0],
        "y": [30.0, 30.0],
        "z": [0.0, 0.0],
        "dx": [10.0, 10.0],
        "dy": [10.0, 10.0],
        "dz": [5.0, 5.0],
        "domain": ["DomainA", "DomainB"],
    })
    samples_xyz = np.column_stack([samples_xy, np.zeros(len(samples_xy))])
    blk_est, blk_var, blk_disp, blk_lag = ordinary_kriging_block_estimation(
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        block_model=bm_domains,
        sill={"DomainA": 1.0, "DomainB": 0.5},
        range_param=100.0,
        domain_col="domain",
        sample_domains=sample_domains,
    )
    assert np.isclose(blk_est[0], 10.0)
    assert np.isclose(blk_est[1], 1.0)
    assert len(blk_lag) == 2


def test_simple_kriging_block_estimation():
    # Test Simple Block Kriging
    bm = create_block_model(
        origin=(0.0, 0.0, 0.0),
        block_size=(10.0, 10.0, 5.0),
        n_blocks=(3, 3, 1),
        default_density=2.70,
    )
    samples_xyz = np.array([
        [5.0, 5.0, 2.5],
        [15.0, 5.0, 2.5],
        [25.0, 5.0, 2.5],
    ])
    sample_grades = np.array([3.0, 2.0, 1.0])

    est, var, disp_var = simple_kriging_block_estimation(
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        block_model=bm,
        mean=2.0,
        sill=1.0,
        range_param=30.0,
        nugget=0.0,
        k_neighbors=8,
    )
    assert len(est) == 9
    assert len(var) == 9
    assert np.all(np.isfinite(est))
    assert np.all(var >= 0.0)
    assert 0.0 < disp_var < 1.0

    # Distant blocks outside search radius revert to prior mean
    samples_far = np.array([[500.0, 500.0, 500.0]])
    grades_far = np.array([10.0])
    est_far, var_far, _ = simple_kriging_block_estimation(
        samples_xyz=samples_far,
        sample_grades=grades_far,
        block_model=bm,
        mean=5.5,
        sill=1.0,
        range_param=30.0,
        max_radius=50.0,  # Blocks are at ~10m, samples are at 500m
        k_neighbors=8,
    )
    assert np.allclose(est_far, 5.5)

    # Multi-domain Simple Block Kriging
    bm_dom = pd.DataFrame({
        "x": [5.0, 100.0],
        "y": [5.0, 100.0],
        "z": [2.5, 2.5],
        "dx": [10.0, 10.0],
        "dy": [10.0, 10.0],
        "dz": [5.0, 5.0],
        "domain": ["DomainA", "DomainB"],
    })
    samples_dom = np.array([
        [5.0, 5.0, 2.5],
        [100.0, 100.0, 2.5],
    ])
    grades_dom = np.array([10.0, 1.0])
    s_domains = ["DomainA", "DomainB"]

    est_d, var_d, _ = simple_kriging_block_estimation(
        samples_xyz=samples_dom,
        sample_grades=grades_dom,
        block_model=bm_dom,
        mean={"DomainA": 10.0, "DomainB": 1.0},
        sill={"DomainA": 1.0, "DomainB": 0.5},
        range_param=50.0,
        domain_col="domain",
        sample_domains=s_domains,
    )
    assert np.isclose(est_d[0], 10.0)
    assert np.isclose(est_d[1], 1.0)


def test_block_model_visualization_suite():
    # Setup test block model and samples
    bm = create_block_model(
        origin=(0.0, 0.0, 0.0),
        block_size=(10.0, 10.0, 5.0),
        n_blocks=(4, 4, 3),  # 48 blocks
        default_density=2.70,
    )
    bm["estimated_grade"] = np.linspace(0.2, 2.5, len(bm))
    bm["kriging_variance"] = np.linspace(0.1, 0.9, len(bm))

    samples_xyz = np.array([
        [15.0, 15.0, 7.5],
        [25.0, 25.0, 7.5],
    ])
    sample_grades = np.array([1.5, 2.0])
    grade_bins = [0.0, 0.5, 1.0, 2.0, 3.0]

    # 1. Test Orthogonal Slices (Default: Looking North, Looking West)
    fig1, axes1 = plot_block_model_orthogonal_slices(
        block_model=bm,
        grade_col="estimated_grade",
        bench_z=7.5,
        section_y=15.0,
        section_x=15.0,
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        grade_bins=grade_bins,
        cross_section_view="north",
        long_section_view="west",
    )
    assert isinstance(fig1, plt.Figure)
    assert len(axes1) == 3
    assert "Looking North" in axes1[1].get_title()
    assert "Looking West" in axes1[2].get_title()
    plt.close(fig1)

    # 1b. Test Orthogonal Slices (Looking South, Looking East)
    fig1b, axes1b = plot_block_model_orthogonal_slices(
        block_model=bm,
        cross_section_view="south",
        long_section_view="east",
    )
    assert "Looking South" in axes1b[1].get_title()
    assert "Looking East" in axes1b[2].get_title()
    plt.close(fig1b)

    with pytest.raises(ValueError, match="cross_section_view"):
        plot_block_model_orthogonal_slices(bm, cross_section_view="invalid")
    with pytest.raises(ValueError, match="long_section_view"):
        plot_block_model_orthogonal_slices(bm, long_section_view="invalid")


    # 2. Test Bench Gallery
    fig2, axes2 = plot_block_model_bench_gallery(
        block_model=bm,
        grade_col="estimated_grade",
        bench_elevations=[2.5, 7.5, 12.5],
        n_cols=3,
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        grade_bins=grade_bins,
    )
    assert isinstance(fig2, plt.Figure)
    assert len(axes2) == 3
    plt.close(fig2)

    # 3. Test 3D Isometric View
    fig3, ax3 = plot_block_model_3d_isometric(
        block_model=bm,
        grade_col="estimated_grade",
        cutoff_grade=1.0,
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        grade_bins=grade_bins,
    )
    assert isinstance(fig3, plt.Figure)
    assert ax3 is not None
    plt.close(fig3)

    # 4. Test Grade vs Uncertainty Audit
    fig4, axes4 = plot_block_model_grade_uncertainty(
        block_model=bm,
        grade_col="estimated_grade",
        var_col="kriging_variance",
        slice_axis="z",
        slice_coord=7.5,
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        grade_bins=grade_bins,
        vmax_var=1.0,
        vmin_var=0.0,
    )
    assert isinstance(fig4, plt.Figure)
    assert len(axes4) == 2
    plt.close(fig4)

    # 5. Test Interactive 3D Explorer
    fig5, ax5, controls = plot_block_model_3d_interactive(
        block_model=bm,
        grade_col="estimated_grade",
        initial_cutoff=0.8,
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        grade_bins=grade_bins,
    )
    assert isinstance(fig5, plt.Figure)
    assert ax5 is not None
    assert "slider_cutoff" in controls
    assert "slider_elev" in controls
    assert "button_reset" in controls

    # Test slider update and reset events
    controls["slider_cutoff"].set_val(1.5)
    controls["slider_elev"].set_val(5.0)
    controls["reset_func"](None)  # trigger reset

    # Test cutoff filter resulting in 0 visible blocks
    controls["slider_cutoff"].set_val(10.0)

    plt.close(fig5)

    with pytest.raises(ValueError, match="missing required column"):
        plot_block_model_3d_interactive(pd.DataFrame({"x": [1.0]}))


def test_consistent_contained_metal_scaling():
    """Validates Item 1.5: Consistent contained metal scaling across resources and reserves.

    Tests both base metals (% Cu with metal_factor=0.01) and precious metals
    (g/t Au with metal_factor=1e-6 for tonnes of metal, or metal_unit="koz").
    """
    # 1. Base metal scenario: 1 Mt @ 2.0% Cu
    res_base = pd.DataFrame({
        "category": ["Measured"],
        "grade": [2.0],
        "tonnes": [1_000_000.0],
    })
    # In-situ contained metal = 1_000_000 * 2.0 * 0.01 = 20,000 t = 20 kt
    res_stmt_base = format_resource_statement(
        res_base,
        cutoff_grade=0.5,
        tonnage_unit="Mt",
        metal_unit="kt",
        metal_factor=0.01,
    )
    # Row 0 (Measured): Contained Metal should be 20 kt
    assert float(res_stmt_base.loc[res_stmt_base["Classification"] == "Measured", "Contained Metal (kt)"].iloc[0]) == pytest.approx(20.0)

    # Convert to reserve with 0 dilution and 100% recovery
    resv_base = convert_resource_to_reserve(
        res_base,
        mining_dilution_pct=0.0,
        mining_recovery_pct=100.0,
        cutoff_grade=0.5,
        metal_factor=0.01,
    )
    assert resv_base["contained_metal"].iloc[0] == pytest.approx(20_000.0)

    resv_stmt_base = format_reserve_statement(
        resv_base,
        cutoff_grade=0.5,
        mining_dilution_pct=0.0,
        mining_recovery_pct=100.0,
        commodity_price="$4.00/lb Cu",
        metallurgical_recovery=90.0,
        tonnage_unit="Mt",
        metal_unit="kt",
        metal_factor=0.01,
    )
    # Contained metal in reserve statement should match resource exactly: 20 kt
    assert float(resv_stmt_base.loc[resv_stmt_base["Classification"] == "Proven Reserve", "Contained Metal (kt)"].iloc[0]) == pytest.approx(20.0)

    # 2. Precious metal scenario: 5 Mt @ 2.5 g/t Au
    # 5,000,000 t * 2.5 g/t * 1e-6 = 12.5 tonnes of Au
    # In koz: 12.5 tonnes / 0.0311035 tonnes/koz = 401.88 koz Au
    res_gold = pd.DataFrame({
        "category": ["Indicated"],
        "grade": [2.5],
        "tonnes": [5_000_000.0],
    })
    res_stmt_gold = format_resource_statement(
        res_gold,
        cutoff_grade=0.5,
        grade_unit="g/t Au",
        tonnage_unit="Mt",
        metal_unit="koz",
        metal_factor=1e-6,
    )
    # Contained metal in koz should be ~402 koz
    gold_metal_res = float(res_stmt_gold.loc[res_stmt_gold["Classification"] == "Indicated", "Contained Metal (koz)"].iloc[0])
    assert gold_metal_res == pytest.approx(12.5 / 0.0311035, rel=1e-2)

    resv_gold = convert_resource_to_reserve(
        res_gold,
        mining_dilution_pct=0.0,
        mining_recovery_pct=100.0,
        cutoff_grade=0.5,
        metal_factor=1e-6,
    )
    assert resv_gold["contained_metal"].iloc[0] == pytest.approx(12.5)

    resv_stmt_gold = format_reserve_statement(
        resv_gold,
        cutoff_grade=0.5,
        mining_dilution_pct=0.0,
        mining_recovery_pct=100.0,
        commodity_price="$2,000/oz Au",
        metallurgical_recovery=92.0,
        grade_unit="g/t Au",
        tonnage_unit="Mt",
        metal_unit="koz",
        metal_factor=1e-6,
    )
    gold_metal_resv = float(resv_stmt_gold.loc[resv_stmt_gold["Classification"] == "Probable Reserve", "Contained Metal (koz)"].iloc[0])
    assert gold_metal_resv == pytest.approx(gold_metal_res, rel=1e-2)


# =============================================================================
# SPATIAL ANISOTROPY (2D / 3D STRUCTURAL CONTINUITY - CIM MRMR §6.8/§6.9)
# =============================================================================


def test_anisotropy_rotation_matrix():
    """Verifies 2D and 3D rotation matrix calculation, orthogonality, and axis projections."""
    # 1. 2D rotation matrix
    # Default (azimuth=0, North strike)
    R2_0 = compute_anisotropy_rotation_matrix(dim=2, angles=0.0)
    assert np.allclose(R2_0 @ R2_0.T, np.eye(2))
    # Along North [0, 1] projects onto major axis [1, 0]
    proj_north = R2_0 @ np.array([0.0, 1.0])
    assert np.allclose(proj_north, [1.0, 0.0])
    # Along East [1, 0] projects onto minor axis [0, 1]
    proj_east = R2_0 @ np.array([1.0, 0.0])
    assert np.allclose(proj_east, [0.0, 1.0])

    # Azimuth 90 (East strike)
    R2_90 = compute_anisotropy_rotation_matrix(dim=2, angles=90.0)
    assert np.allclose(R2_90 @ R2_90.T, np.eye(2))
    # Along East [1, 0] projects onto major axis [1, 0]
    assert np.allclose(R2_90 @ np.array([1.0, 0.0]), [1.0, 0.0])

    # Dict input in 2D
    R2_dict = compute_anisotropy_rotation_matrix(dim=2, angles={"strike": 45.0})
    assert np.allclose(R2_dict @ R2_dict.T, np.eye(2))
    v_45 = np.array([1.0, 1.0]) / np.sqrt(2.0)
    assert np.allclose(R2_dict @ v_45, [1.0, 0.0])

    # 2. 3D rotation matrix
    # Default (0, 0, 0)
    R3_0 = compute_anisotropy_rotation_matrix(dim=3, angles=None)
    assert np.allclose(R3_0 @ R3_0.T, np.eye(3))

    # Dipping and plunging structure: azimuth=45, dip=30, plunge=15
    R3_arb = compute_anisotropy_rotation_matrix(
        dim=3, angles=(45.0, 30.0, 15.0)
    )
    assert np.allclose(R3_arb @ R3_arb.T, np.eye(3))

    # Dict input in 3D
    R3_dict = compute_anisotropy_rotation_matrix(
        dim=3, angles={"azimuth": 90.0, "dip": 45.0, "plunge": 0.0}
    )
    assert np.allclose(R3_dict @ R3_dict.T, np.eye(3))

    # Invalid dimension
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        compute_anisotropy_rotation_matrix(dim=4)


def test_transform_anisotropic_coordinates():
    """Verifies coordinate mapping into isotropic equivalent metric space."""
    # 1. Isotropic coordinates return unmodified
    pts2 = np.array([[10.0, 20.0], [30.0, 40.0]])
    t_pts, r_maj = transform_anisotropic_coordinates(pts2, ranges=100.0)
    assert np.allclose(t_pts, pts2)
    assert r_maj == 100.0

    # Equal ranges return unmodified
    t_pts_eq, _ = transform_anisotropic_coordinates(pts2, ranges=(80.0, 80.0))
    assert np.allclose(t_pts_eq, pts2)

    # Empty array
    t_empty, _ = transform_anisotropic_coordinates(np.empty((0, 2)), ranges=50.0)
    assert len(t_empty) == 0

    # 2. 2D Anisotropic scaling along strike (Azimuth = 0, North)
    # Range major = 100m (along Y), Range minor = 20m (along X)
    pts = np.array([
        [0.0, 0.0],
        [0.0, 50.0],   # 50m North (along strike)
        [10.0, 0.0],   # 10m East (across strike)
    ])
    t_pts, a_maj = transform_anisotropic_coordinates(
        pts, ranges=(100.0, 20.0), angles=0.0
    )
    assert a_maj == 100.0
    # Along strike: 50m / 100m = 0.5 of range -> transformed distance is 50m
    d_along = np.linalg.norm(t_pts[1] - t_pts[0])
    assert d_along == pytest.approx(50.0)
    # Across strike: 10m / 20m = 0.5 of range -> transformed distance is 10 * (100/20) = 50m
    d_across = np.linalg.norm(t_pts[2] - t_pts[0])
    assert d_across == pytest.approx(50.0)

    # 3. 3D Anisotropic scaling
    pts3 = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 50.0, 0.0],  # Major (along North Y)
        [25.0, 0.0, 0.0],  # Semi-major (along East X)
        [0.0, 0.0, 10.0],  # Minor (along Elevation Z)
    ])
    t_pts3, a_maj3 = transform_anisotropic_coordinates(
        pts3, ranges={"major": 100.0, "semi_major": 50.0, "minor": 20.0}, angles=(0, 0, 0)
    )
    assert a_maj3 == 100.0
    # Major: 50m -> 50m
    assert np.linalg.norm(t_pts3[1] - t_pts3[0]) == pytest.approx(50.0)
    # Semi-major: 25m * (100/50) = 50m
    assert np.linalg.norm(t_pts3[2] - t_pts3[0]) == pytest.approx(50.0)
    # Minor: 10m * (100/20) = 50m
    assert np.linalg.norm(t_pts3[3] - t_pts3[0]) == pytest.approx(50.0)

    # 4. Error validations
    with pytest.raises(ValueError, match="Coordinates must have last dimension 2 or 3"):
        transform_anisotropic_coordinates(np.ones((5, 4)))
    with pytest.raises(ValueError, match="2D anisotropy requires 2 ranges"):
        transform_anisotropic_coordinates(pts2, ranges=[100.0, 20.0, 10.0][:1])
    with pytest.raises(ValueError, match="3D anisotropy requires 3 ranges"):
        transform_anisotropic_coordinates(pts3, ranges=[100.0, 50.0])


def test_compute_anisotropic_distance():
    """Verifies pairwise and difference vector anisotropic distance calculations."""
    c1 = np.array([[0.0, 0.0], [0.0, 50.0]])
    c2 = np.array([[0.0, 0.0], [10.0, 0.0]])
    # Major=100 along North (azimuth=0), Minor=20
    dists = compute_anisotropic_distance(c1, c2, ranges=(100.0, 20.0), angles=0.0)
    assert dists.shape == (2, 2)
    assert dists[0, 0] == pytest.approx(0.0)
    assert dists[1, 0] == pytest.approx(50.0)
    assert dists[0, 1] == pytest.approx(50.0)  # 10m across strike * 5 = 50m

    # Single point c2
    dists_single = compute_anisotropic_distance(c1, np.array([0.0, 0.0]), ranges=(100.0, 20.0), angles=0.0)
    assert np.allclose(dists_single, [0.0, 50.0])

    # Difference vectors
    diffs = np.array([[0.0, 50.0], [10.0, 0.0]])
    d_diffs = compute_anisotropic_distance(diffs, ranges=(100.0, 20.0), angles=0.0)
    assert np.allclose(d_diffs, [50.0, 50.0])


def test_theoretical_covariance_anisotropy():
    """Verifies directional spatial covariance honoring structural anisotropy."""
    # Coordinate differences of 30m along North (Y) and 30m along East (X)
    diff_along = np.array([[0.0, 30.0]])  # along strike
    diff_across = np.array([[30.0, 0.0]]) # across strike

    # Strike = 0 (North), Major = 100m, Minor = 25m
    cov_along = _theoretical_covariance(
        diff_along,
        model="spherical",
        sill=1.0,
        range_param=100.0,
        anisotropy_ranges=(100.0, 25.0),
        anisotropy_angles=0.0,
    )
    cov_across = _theoretical_covariance(
        diff_across,
        model="spherical",
        sill=1.0,
        range_param=100.0,
        anisotropy_ranges=(100.0, 25.0),
        anisotropy_angles=0.0,
    )

    # Along strike: 30m < 100m -> significant covariance
    assert float(cov_along[0]) > 0.5
    # Across strike: 30m > 25m (minor range) -> zero covariance
    assert float(cov_across[0]) == pytest.approx(0.0)


def test_idw_and_nn_anisotropy():
    """Verifies Inverse Distance Weighting and Nearest Neighbor honor anisotropic search ellipsoids."""
    target = np.array([[0.0, 0.0]])
    # Sample A: 50m along strike (Azimuth=0, North [0, 50]), Grade = 10.0
    # Sample B: 25m across strike (East [25, 0]), Grade = 2.0
    samples = np.array([[0.0, 50.0], [25.0, 0.0]])
    grades = np.array([10.0, 2.0])

    # 1. Isotropic IDW: Sample B is closer (25m vs 50m), so grade is closer to 2.0
    est_iso, _ = inverse_distance_weighting(samples, grades, target, power=2.0)
    assert est_iso[0] < 5.0

    # 2. Anisotropic IDW: Major range 100m (North), Minor range 20m (East)
    # Sample A equivalent distance: 50m
    # Sample B equivalent distance: 25m * (100/20) = 125m
    # Sample A is now much closer in anisotropic metric space!
    est_aniso, dist_aniso = inverse_distance_weighting(
        samples, grades, target, power=2.0, anisotropy_ranges=(100.0, 20.0), anisotropy_angles=0.0
    )
    # Heavily weighted toward Sample A (10.0)
    assert est_aniso[0] > 8.5
    assert dist_aniso[0, 0] == pytest.approx(50.0)
    assert dist_aniso[0, 1] == pytest.approx(125.0)

    # 3. Nearest Neighbor with anisotropy
    # Isotropic NN picks Sample B (grade 2.0)
    nn_iso, _ = nearest_neighbor_grid_estimation(samples, grades, target)
    assert nn_iso[0] == 2.0

    # Anisotropic NN picks Sample A (grade 10.0)
    nn_aniso, _ = nearest_neighbor_grid_estimation(
        samples, grades, target, anisotropy_ranges=(100.0, 20.0), anisotropy_angles=0.0
    )
    assert nn_aniso[0] == 10.0

    # 4. Domain-segregated IDW with anisotropy
    samples_dom = np.array([[0.0, 50.0], [25.0, 0.0], [0.0, -40.0]])
    grades_dom = np.array([10.0, 2.0, 7.0])
    s_domains = ["Dom1", "Dom1", "Dom2"]
    g_domains = ["Dom1"]
    est_dom, _ = inverse_distance_weighting(
        samples_dom,
        grades_dom,
        target,
        sample_domains=s_domains,
        grid_domains=g_domains,
        anisotropy_ranges={"Dom1": (100.0, 20.0), "Dom2": (80.0, 40.0)},
        anisotropy_angles={"Dom1": 0.0, "Dom2": 45.0},
    )
    assert est_dom[0] > 8.5


def test_kriging_grid_estimation_anisotropy():
    """Verifies Simple and Ordinary Kriging with 2D spatial anisotropy."""
    target = np.array([[0.0, 0.0]])
    # Sample 1: 40m along North (Y)
    # Sample 2: 40m along East (X)
    samples = np.array([[0.0, 40.0], [40.0, 0.0]])
    grades = np.array([10.0, 2.0])

    # 1. Isotropic Kriging: both samples are equidistant, receive equal weights (0.5 each) -> grade 6.0
    est_ok_iso, var_ok_iso = ordinary_kriging_grid_estimation(
        samples, grades, target, sill=1.0, range_param=100.0
    )
    assert est_ok_iso[0] == pytest.approx(6.0, rel=1e-3)

    # Invariance: passing equal anisotropy ranges reproduces isotropic result exactly
    est_ok_eq, var_ok_eq = ordinary_kriging_grid_estimation(
        samples, grades, target, sill=1.0, range_param=100.0, anisotropy_ranges=(100.0, 100.0)
    )
    assert est_ok_eq[0] == pytest.approx(est_ok_iso[0], rel=1e-5)
    assert var_ok_eq[0] == pytest.approx(var_ok_iso[0], rel=1e-5)

    # 2. Anisotropic Ordinary Kriging: strike=0 (North), major=100m, minor=25m
    est_ok_aniso, var_ok_aniso = ordinary_kriging_grid_estimation(
        samples, grades, target, sill=1.0, range_param=100.0,
        anisotropy_ranges=(100.0, 25.0), anisotropy_angles=0.0
    )
    # Sample 1 (North) is within major range (40m < 100m), Sample 2 (East) is beyond minor range (40m > 25m)
    # Sample 1 receives majority of weight (w1=0.716 vs w2=0.284 in OK due to sum-to-1 constraint)
    assert est_ok_aniso[0] == pytest.approx(7.728, abs=1e-2)
    assert est_ok_aniso[0] > est_ok_iso[0] + 1.5

    # 3. Anisotropic Simple Kriging
    est_sk_aniso, var_sk_aniso = simple_kriging_grid_estimation(
        samples, grades, target, mean=5.0, sill=1.0, range_param=100.0,
        anisotropy_ranges=(100.0, 25.0), anisotropy_angles=0.0
    )
    assert est_sk_aniso[0] > 7.0

    # Domain segregation in Ordinary Kriging with anisotropy
    samples_d = np.array([[0.0, 40.0], [40.0, 0.0], [0.0, 30.0]])
    grades_d = np.array([10.0, 2.0, 8.0])
    est_d, _ = ordinary_kriging_grid_estimation(
        samples_d, grades_d, target, sill=1.0, range_param=100.0,
        sample_domains=["A", "A", "B"], grid_domains=["A"],
        anisotropy_ranges={"A": (100.0, 25.0), "B": (80.0, 30.0)},
        anisotropy_angles={"A": 0.0, "B": 90.0},
    )
    assert est_d[0] == pytest.approx(est_ok_aniso[0], rel=1e-4)


def test_block_kriging_3d_anisotropy():
    """Verifies 3D Block Kriging with dipping structural anisotropy."""
    bm = create_block_model(
        origin=(0.0, 0.0, 0.0),
        block_size=(20.0, 20.0, 10.0),
        n_blocks=(2, 2, 2),
        default_density=2.70,
    )
    samples = np.array([
        [10.0, 10.0, 5.0],
        [10.0, 30.0, 5.0],
        [30.0, 10.0, 5.0],
        [30.0, 30.0, 5.0],
    ])
    grades = np.array([1.5, 2.0, 1.2, 2.5])

    # 1. Isotropic baseline
    est_iso, var_iso, disp_iso, _ = ordinary_kriging_block_estimation(
        samples, grades, bm, sill=1.0, range_param=80.0, discretization=(2, 2, 2)
    )
    assert len(est_iso) == 8
    assert np.all(np.isfinite(est_iso))
    assert np.all(var_iso >= 0.0)

    # 2. 3D Dipping Anisotropy: Azimuth=90 (East), Dip=30 deg, ranges=(100, 50, 20)
    est_aniso, var_aniso, disp_aniso, _ = ordinary_kriging_block_estimation(
        samples,
        grades,
        bm,
        sill=1.0,
        range_param=100.0,
        discretization=(2, 2, 2),
        anisotropy_ranges=(100.0, 50.0, 20.0),
        anisotropy_angles=(90.0, 30.0, 0.0),
    )
    assert len(est_aniso) == 8
    assert np.all(np.isfinite(est_aniso))
    assert np.all(var_aniso >= 0.0)
    assert disp_aniso > 0.0

    # 3. Simple Block Kriging 3D Anisotropy
    est_sk, var_sk, disp_sk = simple_kriging_block_estimation(
        samples,
        grades,
        bm,
        mean=1.8,
        sill=1.0,
        range_param=100.0,
        discretization=(2, 2, 2),
        anisotropy_ranges=(100.0, 50.0, 20.0),
        anisotropy_angles=(90.0, 30.0, 0.0),
    )
    assert len(est_sk) == 8
    assert np.all(np.isfinite(est_sk))
    assert np.all(var_sk >= 0.0)


def test_classification_by_drill_spacing_anisotropy():
    """Verifies drillhole spacing classification using anisotropic search ellipsoids."""
    grid_pts = np.array([[0.0, 0.0]])
    # 3 drillholes located 45m along strike (North Y) and spaced 10m in X
    samples = np.array([
        [-10.0, 45.0],
        [0.0, 45.0],
        [10.0, 45.0],
    ])

    # 1. Under isotropic radius max_radius_measured=30m:
    # Physical distance is ~45m > 30m, so it cannot qualify for Measured
    cats_iso = classify_resources_by_drill_spacing(
        grid_pts,
        samples,
        max_radius_measured=30.0,
        max_radius_indicated=60.0,
        min_holes_measured=3,
        min_holes_indicated=2,
    )
    assert cats_iso[0] == "Indicated"

    # 2. Under anisotropic search ellipsoid: strike=0, major=60m, minor=25m
    # 45m along strike is well within the 60m major axis
    cats_aniso = classify_resources_by_drill_spacing(
        grid_pts,
        samples,
        max_radius_measured=60.0,
        max_radius_indicated=120.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        anisotropy_ranges=(60.0, 25.0),
        anisotropy_angles=0.0,
    )
    assert cats_aniso[0] == "Measured"

    # 3. Integrated classify_mineral_resources with anisotropy
    cats_min_res = classify_mineral_resources(
        grid_points=grid_pts,
        samples_xy=samples,
        max_radius_measured=60.0,
        max_radius_indicated=120.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        anisotropy_ranges=(60.0, 25.0),
        anisotropy_angles=0.0,
    )
    assert cats_min_res[0] == "Measured"


def test_compositing_unassayed_gap_zero_treatment():
    """Verifies that unassayed gaps are filled with zero/default grade to conserve metal."""
    # Hole DH01:
    # 0.0 - 1.0m: 10.0% grade (contained metal = 1.0 * 10 = 10 units)
    # 1.0 - 2.5m: UNASSAYED GAP of 1.5m (e.g. unsampled waste or core loss)
    # 2.5 - 3.0m: 4.0% grade (contained metal = 0.5 * 4 = 2 units)
    # Total raw contained metal = 12 units
    df = pd.DataFrame({
        "hole_id": ["DH01", "DH01"],
        "from_m": [0.0, 2.5],
        "to_m": [1.0, 3.0],
        "grade": [10.0, 4.0],
        "x": [100.0, 100.0],
        "y": [200.0, 200.0],
    })

    # 1. Default unassayed_treatment="zero" with 2.0m composites:
    # Comp 1: [0.0, 2.0]: 1.0m @ 10.0% + 1.0m @ 0.0% -> grade = 5.0%
    #   Contained metal = 2.0 * 5.0 = 10 units (conserved!)
    #   sampled_length = 1.0m, coverage_ratio = 0.50
    # Comp 2: [2.0, 3.0]: 0.5m @ 0.0% + 0.5m @ 4.0% -> grade = 2.0%
    #   Contained metal = 1.0 * 2.0 = 2 units (conserved!)
    #   sampled_length = 0.5m, coverage_ratio = 0.50
    comp_df = composite_drillhole_intervals(
        df,
        composite_length=2.0,
        min_length_ratio=0.50,
        unassayed_treatment="zero",
        unassayed_grade=0.0,
    )
    assert len(comp_df) == 2
    assert comp_df.iloc[0]["grade"] == pytest.approx(5.0)
    assert comp_df.iloc[0]["sampled_length"] == pytest.approx(1.0)
    assert comp_df.iloc[0]["coverage_ratio"] == pytest.approx(0.50)
    assert comp_df.iloc[1]["grade"] == pytest.approx(2.0)
    assert comp_df.iloc[1]["sampled_length"] == pytest.approx(0.5)
    assert comp_df.iloc[1]["coverage_ratio"] == pytest.approx(0.50)

    # Total composite metal equals raw metal exactly (Zero dilution distortion)
    tot_metal = float((comp_df["length"] * comp_df["grade"]).sum())
    assert tot_metal == pytest.approx(12.0)
    assert comp_df.attrs["unassayed_gaps_count"] == 1
    assert comp_df.attrs["unassayed_gaps_total_length"] == pytest.approx(1.5)

    # 2. Custom unassayed_grade (e.g. background threshold = 0.5%)
    comp_bg = composite_drillhole_intervals(
        df,
        composite_length=2.0,
        min_length_ratio=0.50,
        unassayed_treatment="zero",
        unassayed_grade=0.5,
    )
    # Comp 1: (1.0 * 10.0 + 1.0 * 0.5) / 2.0 = 5.25%
    assert comp_bg.iloc[0]["grade"] == pytest.approx(5.25)


def test_compositing_unassayed_gap_split_treatment():
    """Verifies that unassayed gaps trigger run splits when unassayed_treatment='split'."""
    df = pd.DataFrame({
        "hole_id": ["DH01", "DH01"],
        "from_m": [0.0, 2.5],
        "to_m": [1.0, 3.5],
        "grade": [10.0, 4.0],
    })

    # Under 'split', the 1.5m gap resets the run.
    # Run 1: [0.0, 1.0] (1.0m length). With target 2.0m and min_length_ratio 0.50, kept as [0, 1] @ 10.0%
    # Run 2: [2.5, 3.5] (1.0m length). Kept as [2.5, 3.5] @ 4.0%
    # Neither composite spans the gap, and no zero-grade dilution occurs.
    comp_split = composite_drillhole_intervals(
        df,
        composite_length=2.0,
        min_length_ratio=0.50,
        unassayed_treatment="split",
    )
    assert len(comp_split) == 2
    assert comp_split.iloc[0]["from_m"] == pytest.approx(0.0)
    assert comp_split.iloc[0]["to_m"] == pytest.approx(1.0)
    assert comp_split.iloc[0]["grade"] == pytest.approx(10.0)
    assert comp_split.iloc[0]["coverage_ratio"] == pytest.approx(1.0)

    assert comp_split.iloc[1]["from_m"] == pytest.approx(2.5)
    assert comp_split.iloc[1]["to_m"] == pytest.approx(3.5)
    assert comp_split.iloc[1]["grade"] == pytest.approx(4.0)
    assert comp_split.iloc[1]["coverage_ratio"] == pytest.approx(1.0)
    assert comp_split.attrs["contiguous_runs_count"] == 2


def test_compositing_missing_grade_codes_and_nan():
    """Verifies handling of explicit NaN and negative missing assay codes."""
    df = pd.DataFrame({
        "hole_id": ["DH01", "DH01", "DH01", "DH01"],
        "from_m": [0.0, 2.0, 4.0, 6.0],
        "to_m": [2.0, 4.0, 6.0, 8.0],
        "grade": [5.0, np.nan, -999.0, 3.0],
    })

    # 1. Under "zero": missing intervals are treated as 0.0%
    comp_zero = composite_drillhole_intervals(
        df,
        composite_length=4.0,
        unassayed_treatment="zero",
        unassayed_grade=0.0,
    )
    assert len(comp_zero) == 2
    # Comp 1 [0, 4]: 2m @ 5.0% + 2m @ 0.0% -> grade = 2.5%, sampled_length = 2.0m
    assert comp_zero.iloc[0]["grade"] == pytest.approx(2.5)
    assert comp_zero.iloc[0]["sampled_length"] == pytest.approx(2.0)
    # Comp 2 [4, 8]: 2m @ 0.0% + 2m @ 3.0% -> grade = 1.5%, sampled_length = 2.0m
    assert comp_zero.iloc[1]["grade"] == pytest.approx(1.5)

    # 2. Under "split": missing rows break runs
    comp_split = composite_drillhole_intervals(
        df,
        composite_length=2.0,
        unassayed_treatment="split",
    )
    # Only interval [0, 2] and [6, 8] are valid sampled runs
    assert len(comp_split) == 2
    assert comp_split.iloc[0]["grade"] == pytest.approx(5.0)
    assert comp_split.iloc[1]["grade"] == pytest.approx(3.0)


def test_compositing_min_coverage_ratio():
    """Verifies that composites with low assay coverage are discarded when min_coverage_ratio is set."""
    # 0.5m sampled @ 8.0%, 1.5m gap -> 2m composite has coverage_ratio = 0.5 / 2.0 = 0.25
    df = pd.DataFrame({
        "hole_id": ["DH01", "DH01"],
        "from_m": [0.0, 2.0],
        "to_m": [0.5, 4.0],
        "grade": [8.0, 2.0],
    })

    # min_coverage_ratio = 0.50 -> Comp 1 ([0, 2], coverage 0.25) must be discarded!
    # Comp 2 ([2, 4], coverage 1.0) is retained.
    comp_cov = composite_drillhole_intervals(
        df,
        composite_length=2.0,
        unassayed_treatment="zero",
        min_coverage_ratio=0.50,
    )
    assert len(comp_cov) == 1
    assert comp_cov.iloc[0]["from_m"] == pytest.approx(2.0)
    assert comp_cov.attrs["discarded_low_coverage_count"] == 1


def test_compositing_non_contiguous_domain_repetition():
    """Verifies that repeating domains (A -> B -> A) are treated as distinct runs without crossing contacts."""
    df = pd.DataFrame({
        "hole_id": ["DH01", "DH01", "DH01"],
        "from_m": [0.0, 6.0, 10.0],
        "to_m": [6.0, 10.0, 16.0],
        "grade": [3.0, 0.2, 5.0],
        "domain": ["Ore", "Waste", "Ore"],
    })

    comp_dom = composite_drillhole_intervals(
        df,
        composite_length=2.0,
        domain_col="domain",
    )
    # Ore run 1: [0, 2], [2, 4], [4, 6] (3 composites)
    # Waste run: [6, 8], [8, 10] (2 composites)
    # Ore run 2: [10, 12], [12, 14], [14, 16] (3 composites)
    assert len(comp_dom) == 8
    assert list(comp_dom["domain"]) == [
        "Ore", "Ore", "Ore", "Waste", "Waste", "Ore", "Ore", "Ore"
    ]
    # Check that Ore run 1 ends at 6.0 and Ore run 2 starts at 10.0
    assert comp_dom.iloc[2]["to_m"] == pytest.approx(6.0)
    assert comp_dom.iloc[5]["from_m"] == pytest.approx(10.0)
    assert comp_dom.attrs["contiguous_runs_count"] == 3


def test_compositing_error_and_validation():
    """Verifies strict validation of invalid intervals and unassayed_treatment='error'."""
    # 1. from >= to
    df_inv = pd.DataFrame({
        "hole_id": ["DH01"],
        "from_m": [5.0],
        "to_m": [3.0],
        "grade": [1.0],
    })
    with pytest.raises(ValueError, match="from_m .* >= to_m"):
        composite_drillhole_intervals(df_inv, composite_length=2.0)

    # 2. Overlapping intervals
    df_overlap = pd.DataFrame({
        "hole_id": ["DH01", "DH01"],
        "from_m": [0.0, 2.0],
        "to_m": [3.0, 5.0],  # row 1 ends at 3.0, but row 2 starts at 2.0
        "grade": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="Overlapping intervals detected"):
        composite_drillhole_intervals(df_overlap, composite_length=2.0)

    # 3. unassayed_treatment="error" raises on unassayed gap
    df_gap = pd.DataFrame({
        "hole_id": ["DH01", "DH01"],
        "from_m": [0.0, 5.0],
        "to_m": [2.0, 7.0],
        "grade": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="Unassayed depth gap"):
        composite_drillhole_intervals(
            df_gap, composite_length=2.0, unassayed_treatment="error"
        )

    # 4. Invalid treatment or parameter
    with pytest.raises(ValueError, match="unassayed_treatment must be"):
        composite_drillhole_intervals(
            df_gap, composite_length=2.0, unassayed_treatment="invalid"
        )
    with pytest.raises(ValueError, match="min_coverage_ratio must be in"):
        composite_drillhole_intervals(
            df_gap, composite_length=2.0, min_coverage_ratio=1.5
        )


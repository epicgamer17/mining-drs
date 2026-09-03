"""Example: 2D Block Model Estimation using Simple and Ordinary Kriging.

Demonstrates geostatistical spatial interpolation of exploration drillholes across
a regular mine block grid:
1. Defines exploration drillhole assay samples and concession boundary.
2. Generates a regular 2D block grid across the deposit.
3. Compares Geostatistical Estimators:
   - Simple Kriging (SK): Uses known global prior mean (gracefully reverts in un-drilled zones).
   - Ordinary Kriging (OK): Enforces local unbiasedness (sum of weights = 1).
   - Kriging Estimation Variance (sigma^2): Maps spatial uncertainty for resource classification.
   - Inverse Distance Squared (IDW^2): Classical benchmark comparison.
4. Performs Spatial Audit: Interpolation vs. Extrapolation (Convex Hull).
5. Computes Grade-Tonnage curves across economic cutoff grades for Ordinary Kriging.
6. Generates high-resolution 4-panel plan maps (Grades, Variances, and Extrapolation Masking).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, KDTree

from drs_mining.components.estimation import (
    simple_kriging_grid_estimation,
    ordinary_kriging_grid_estimation,
    inverse_distance_weighting,
    is_within_convex_hull,
    grade_tonnage_table,
    plot_grade_tonnage_curve,
    cell_declustering,
    plot_cell_declustering_curve,
    plot_swath_analysis,
    format_resource_statement,
    calculate_cut_off_grade,
    convert_resource_to_reserve,
    format_reserve_statement,
    plot_resource_to_reserve_waterfall,
    plot_reserve_classification_map,
    plot_in_situ_vs_diluted_curves,
)


def create_sample_deposit() -> tuple[pd.DataFrame, list[tuple[float, float]]]:
    """Sample copper porphyry exploration drillholes and concession perimeter."""
    drillholes = pd.DataFrame(
        {
            "hole_id": [
                "DH01",
                "DH02",
                "DH03",
                "DH04",
                "DH05",
                "DH06",
                "DH07",
                "DH08",
                "DH09",
                "DH10",
                "DH11",
                "DH12",
                "DH13",
                "DH14",
                "DH15",
                "DH16",
            ],
            "x": [
                120.0,
                240.0,
                380.0,
                520.0,
                150.0,
                280.0,
                420.0,
                560.0,
                130.0,
                260.0,
                400.0,
                540.0,
                170.0,
                300.0,
                450.0,
                580.0,
            ],
            "y": [
                120.0,
                110.0,
                130.0,
                140.0,
                230.0,
                240.0,
                220.0,
                250.0,
                350.0,
                360.0,
                340.0,
                370.0,
                460.0,
                470.0,
                450.0,
                480.0,
            ],
            "grade": [
                0.35,
                0.65,
                1.10,
                0.45,
                0.52,
                1.55,
                1.95,
                0.85,
                0.40,
                1.30,
                1.70,
                0.72,
                0.28,
                0.60,
                0.95,
                0.40,
            ],  # % Cu
        }
    )

    boundary = [
        (80.0, 70.0),
        (620.0, 80.0),
        (640.0, 530.0),
        (380.0, 550.0),
        (90.0, 510.0),
    ]

    return drillholes, boundary


def generate_grid_points(
    boundary: list[tuple[float, float]],
    grid_resolution: float = 10.0,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[float, float, float, float],
]:
    """Generates regular 2D grid nodes clipped within the concession boundary."""
    b_arr = np.array(boundary)
    min_x, max_x = b_arr[:, 0].min(), b_arr[:, 0].max()
    min_y, max_y = b_arr[:, 1].min(), b_arr[:, 1].max()

    xs = np.arange(min_x, max_x + grid_resolution, grid_resolution)
    ys = np.arange(min_y, max_y + grid_resolution, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)

    all_grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Clip points to concession boundary polygon
    poly_path = MplPath(boundary)
    inside_mask = poly_path.contains_points(all_grid_points)
    clipped_grid = all_grid_points[inside_mask]

    extent = (float(min_x), float(max_x), float(min_y), float(max_y))
    return clipped_grid, (xx, yy, inside_mask), extent


def main():
    parser = argparse.ArgumentParser(
        description="2D Kriging Geostatistical Estimation Demo"
    )
    parser.add_argument(
        "--grid-res",
        type=float,
        default=10.0,
        help="Block model grid resolution in meters (default: 10m x 10m)",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=None,
        help="Maximum search neighborhood radius in meters (default: unlimited)",
    )
    parser.add_argument(
        "--variogram",
        type=str,
        default="spherical",
        choices=["spherical", "exponential", "gaussian"],
        help="Theoretical variogram model (default: spherical)",
    )
    parser.add_argument(
        "--nugget",
        type=float,
        default=0.08,
        help="Nugget variance c0 (micro-scale noise/sampling error, default: 0.08)",
    )
    parser.add_argument(
        "--sill",
        type=float,
        default=0.25,
        help="Partial sill variance c (default: 0.25)",
    )
    parser.add_argument(
        "--range",
        type=float,
        default=180.0,
        help="Spatial correlation range a in meters (default: 180m)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=16,
        help="Number of nearest drillhole neighbors to query per block (default: 16)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="plots/kriging_estimation_comparison.png",
        help="Output image path for multi-panel comparison figure",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable figure rendering",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("      2D GEOSTATISTICAL ESTIMATION: SIMPLE VS. ORDINARY KRIGING")
    print("=" * 70)

    # 1. Load drillhole data and define grid
    drillholes, boundary = create_sample_deposit()
    samples_xy = drillholes[["x", "y"]].to_numpy()
    sample_grades = drillholes["grade"].to_numpy()
    global_mean = float(sample_grades.mean())

    grid_points, grid_info, extent = generate_grid_points(
        boundary, grid_resolution=args.grid_res
    )

    bench_height = 12.0  # meters
    bulk_density = 2.7  # t/m^3
    block_tonnes = (args.grid_res**2) * bench_height * bulk_density

    print(f"\nLoaded {len(drillholes)} exploration drillholes.")
    print(
        f"Assays: Min={sample_grades.min():.2f}%, Max={sample_grades.max():.2f}%, Naive Mean={global_mean:.3f}% Cu, Variance={sample_grades.var():.3f}"
    )
    print(
        f"Variogram: Model={args.variogram.title()}, Nugget={args.nugget:.2f}, Sill={args.sill:.2f}, Range={args.range:.1f}m"
    )
    print(
        f"Block Model: {len(grid_points):,} blocks ({args.grid_res:.0f}m x {args.grid_res:.0f}m x {bench_height:.0f}m), Total Tonnage: {len(grid_points) * block_tonnes:,.0f} tonnes."
    )

    # 2. Cell Declustering Analysis (SME Handbook Section 4.3 & Deutsch & Journel 1998)
    declust_cell_sizes = np.linspace(20.0, 520.0, 21)
    declust_weights, declust_df, opt_cell_size = cell_declustering(
        drillholes,
        cell_sizes=declust_cell_sizes,
        grade_col="grade",
        x_col="x",
        y_col="y",
        min_mean=True,
    )
    opt_declust_mean = float(
        declust_df.loc[
            declust_df["cell_size"] == opt_cell_size, "declustered_mean"
        ].iloc[0]
    )
    opt_declust_var = float(
        declust_df.loc[
            declust_df["cell_size"] == opt_cell_size, "declustered_variance"
        ].iloc[0]
    )
    bias_removed_pct = ((global_mean - opt_declust_mean) / global_mean) * 100.0

    print("\n--- Cell Declustering Analysis (SME Handbook Section 4.3) ---")
    print(
        f"Optimal Declustering Cell Size : {opt_cell_size:.1f} m (Minimum Declustered Mean)"
    )
    print(
        f"Naive (Unweighted) Mean Assay  : {global_mean:.3f}% Cu  (Variance: {sample_grades.var():.3f})"
    )
    print(
        f"Representative Declustered Mean: {opt_declust_mean:.3f}% Cu  (Variance: {opt_declust_var:.3f})"
    )
    print(f"High-Grade Drilling Bias Removed: {bias_removed_pct:+.1f}%")

    # 3. Spatial Audit: Interpolation vs. Extrapolation
    is_interpolated = is_within_convex_hull(samples_xy, grid_points)
    n_interpolated = int(is_interpolated.sum())
    n_extrapolated = len(grid_points) - n_interpolated

    print("\n--- Spatial Audit: Interpolation vs. Extrapolation (Convex Hull) ---")
    print(
        f"Interpolated Blocks (Inside Convex Hull)  : {n_interpolated:,d} ({n_interpolated / len(grid_points) * 100:.1f}%) -> {n_interpolated * block_tonnes:,.0f} tonnes [High Confidence / Measured]"
    )
    print(
        f"Extrapolated Blocks (Outside Convex Hull) : {n_extrapolated:,d} ({n_extrapolated / len(grid_points) * 100:.1f}%) -> {n_extrapolated * block_tonnes:,.0f} tonnes [Exploration Risk / Inferred]"
    )

    # 4. Estimator 1: Ordinary Kriging (OK) - Full Domain
    print(f"\n[1/4] Running Ordinary Kriging (OK, k={args.k_neighbors})...")
    ok_grades, ok_variances = ordinary_kriging_grid_estimation(
        samples_xy,
        sample_grades,
        grid_points,
        variogram_model=args.variogram,
        nugget=args.nugget,
        sill=args.sill,
        range_param=args.range,
        k_neighbors=args.k_neighbors,
        max_radius=args.max_radius,
        mask_extrapolation=False,
    )

    # 5. Estimator 2: Ordinary Kriging (OK) - Strict Interpolation Only
    print(
        f"[2/4] Running Ordinary Kriging with Extrapolation Masked (Strict Interpolation)..."
    )
    ok_strict_grades, ok_strict_variances = ordinary_kriging_grid_estimation(
        samples_xy,
        sample_grades,
        grid_points,
        variogram_model=args.variogram,
        nugget=args.nugget,
        sill=args.sill,
        range_param=args.range,
        k_neighbors=args.k_neighbors,
        max_radius=args.max_radius,
        mask_extrapolation=True,
    )

    # 6. Estimator 3: Simple Kriging (SK) - Full Domain (Using Unbiased Declustered Mean)
    print(
        f"[3/4] Running Simple Kriging (SK, declustered prior mean={opt_declust_mean:.3f}%)..."
    )
    sk_grades, sk_variances = simple_kriging_grid_estimation(
        samples_xy,
        sample_grades,
        grid_points,
        mean=opt_declust_mean,
        variogram_model=args.variogram,
        nugget=args.nugget,
        sill=args.sill,
        range_param=args.range,
        k_neighbors=args.k_neighbors,
        max_radius=args.max_radius,
        mask_extrapolation=False,
    )

    # 7. Estimator 4: Inverse Distance Squared (IDW^2) - Benchmark
    print(f"[4/4] Running Inverse Distance Squared (IDW^2, power=2.0)...")
    idw_grades, _ = inverse_distance_weighting(
        samples_xy,
        sample_grades,
        grid_points,
        power=2.0,
        k_neighbors=args.k_neighbors,
        max_radius=args.max_radius,
        mask_extrapolation=False,
    )

    # 7. Model Comparative Statistics
    models = {
        "Ordinary Kriging (OK)": (ok_grades, ok_variances),
        "Ordinary Kriging (Strict Interp)": (ok_strict_grades, ok_strict_variances),
        "Simple Kriging (SK)": (sk_grades, sk_variances),
        "Inverse Distance Squared (IDW²)": (idw_grades, None),
    }

    print("\n--- Model Comparative Statistics (Estimates & Kriging Variances) ---")
    stats_rows = []
    for name, (g, v) in models.items():
        valid_g = g[~np.isnan(g)]
        v_mean_str = f"{v[~np.isnan(v)].mean():.3f}" if v is not None else "N/A"
        stats_rows.append(
            {
                "Model": name,
                "Mean (% Cu)": f"{valid_g.mean():.3f}",
                "Std Dev": f"{valid_g.std():.3f}",
                "Min (%)": f"{valid_g.min():.3f}",
                "Max (%)": f"{valid_g.max():.3f}",
                "Mean Variance": v_mean_str,
                "Estimated Blocks": f"{len(valid_g):,d} ({len(valid_g) / len(g) * 100.0:.1f}%)",
            }
        )
    print(pd.DataFrame(stats_rows).to_string(index=False))

    # 8. Grade-Tonnage Sensitivity for All Models (Model Audit)
    cutoffs = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    gt_models = {}
    for name, (g, _) in models.items():
        block_df = pd.DataFrame(
            {
                "grade": g[~np.isnan(g)],
                "tonnes": block_tonnes,
            }
        )
        gt_models[name] = grade_tonnage_table(block_df, cutoffs=cutoffs)

    print("\n--- Ordinary Kriging Grade–Tonnage Sensitivity Curve ---")
    gt_disp = gt_models["Ordinary Kriging (OK)"].copy()
    gt_disp["ore_tonnes"] = gt_disp["ore_tonnes"].map(lambda x: f"{x:,.0f}")
    gt_disp["ore_grade"] = gt_disp["ore_grade"].map(lambda x: f"{x:.3f}%")
    gt_disp["waste_tonnes"] = gt_disp["waste_tonnes"].map(lambda x: f"{x:,.0f}")
    gt_disp["strip_ratio"] = gt_disp["strip_ratio"].map(lambda x: f"{x:.2f}")
    gt_disp["ore_recovery_pct"] = gt_disp["ore_recovery_pct"].map(lambda x: f"{x:.1f}%")
    gt_disp["metal_recovery_pct"] = gt_disp["metal_recovery_pct"].map(
        lambda x: f"{x:.1f}%"
    )
    print(
        gt_disp[
            [
                "ore_tonnes",
                "ore_grade",
                "waste_tonnes",
                "strip_ratio",
                "ore_recovery_pct",
                "metal_recovery_pct",
            ]
        ].to_string()
    )

    # 9. Official Mineral Resource Statement (NI 43-101 / JORC Code Compliant)
    # Assign confidence categories based on spatial audit & drill spacing
    tree_samples = KDTree(samples_xy)
    d_to_samples, _ = tree_samples.query(grid_points)

    categories = np.empty(len(grid_points), dtype=object)
    for i in range(len(grid_points)):
        if is_interpolated[i]:
            if d_to_samples[i] <= 60.0:
                categories[i] = "Measured"
            else:
                categories[i] = "Indicated"
        else:
            categories[i] = "Inferred"

    block_class_df = pd.DataFrame(
        {
            "category": categories,
            "grade": ok_grades,
            "tonnes": block_tonnes,
        }
    )

    base_cutoff = 0.50
    resource_stmt = format_resource_statement(
        block_class_df,
        cutoff_grade=base_cutoff,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        metal_unit="kt",
        commodity_price="$3.80/lb Cu",
        metallurgical_recovery=88.0,
        rpeee_constraint="Constrained within Lerchs-Grossmann optimized pit shell",
    )

    print(
        f"\n--- Official Mineral Resource Statement (Base Cutoff: {base_cutoff:.2f}% Cu) ---"
    )
    print(resource_stmt.to_string(index=False))
    print("\nCompliance Footnotes:")
    for fn in resource_stmt.attrs.get("footnotes", []):
        print(f"  {fn}")

    # 10. Mineral Reserve Delineation: Applying Modifying Factors (CIM / NI 43-101)
    # Explicit deposit-specific engineering cost and recovery structure (zero hidden defaults)
    proc_cost = 11.50   # $/t ore processing
    ga_cost = 2.20      # $/t ore G&A
    mining_cost = 2.40  # $/t rock mining
    cu_price = 3.80     # $/lb Cu base commodity price
    selling_cost = 0.35 # $/lb Cu deductions (smelting, refining, transport)
    royalty_pct = 2.0   # 2% Net Smelter Return (NSR) royalty
    met_rec = 88.0      # 88% plant recovery
    lbs_per_pct_t = 22.0462  # 1% Cu = 22.0462 lbs Cu per metric tonne

    # Calculate engineering cut-off grades
    co_breakeven = calculate_cut_off_grade(
        processing_cost=proc_cost,
        ga_cost=ga_cost,
        mining_cost=mining_cost,
        commodity_price=cu_price,
        selling_cost=selling_cost,
        royalty_pct=royalty_pct,
        metallurgical_recovery=met_rec,
        metal_conversion_factor=lbs_per_pct_t,
    )
    co_marginal = calculate_cut_off_grade(
        processing_cost=proc_cost,
        ga_cost=ga_cost,
        mining_cost=None,
        commodity_price=cu_price,
        selling_cost=selling_cost,
        royalty_pct=royalty_pct,
        metallurgical_recovery=met_rec,
        metal_conversion_factor=lbs_per_pct_t,
    )

    print("\n--- Engineering Cut-Off Grade Determination (Modifying Factors) ---")
    print(f"Breakeven Economic Cut-Off : {co_breakeven:.3f}% Cu (Covers Mining, Processing, G&A, Royalties)")
    print(f"Marginal / Internal Cut-Off: {co_marginal:.3f}% Cu (Covers Processing, G&A, Royalties; Sunk Mining)")

    # Site-specific mining modifying factors
    dilution_pct = 5.0    # 5% unplanned wall rock dilution
    dilution_grade = 0.05 # 0.05% Cu grade in diluting contact rock
    recovery_pct = 95.0   # 95% mining extraction recovery (5% ore loss)

    reserve_df = convert_resource_to_reserve(
        block_class_df,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        cutoff_grade=base_cutoff,
        dilution_grade=dilution_grade,
        allow_inferred=False,  # Strict regulatory compliance: Inferred cannot be reserves!
    )

    reserve_stmt = format_reserve_statement(
        reserve_df,
        cutoff_grade=base_cutoff,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        commodity_price=f"${cu_price:.2f}/lb Cu",
        metallurgical_recovery=met_rec,
        tonnage_unit="Mt",
        grade_unit="% Cu",
        metal_unit="kt",
        rpeee_constraint="Constrained within Phase 3 engineered final pit design",
    )

    print(f"\n--- Official Mineral Reserve Statement (Run-of-Mine Feed, Cutoff: {base_cutoff:.2f}% Cu) ---")
    print(reserve_stmt.to_string(index=False))
    print("\nCompliance Footnotes:")
    for fn in reserve_stmt.attrs.get("footnotes", []):
        print(f"  {fn}")

    # 11. Spatial Visualization & Dedicated Grade-Tonnage Plots (One per Model)
    if not args.no_plot:
        plots_dir = Path(args.save_plot).parent
        plots_dir.mkdir(parents=True, exist_ok=True)
        xx, yy, inside_mask = grid_info

        # A. 5-Panel Spatial Comparison Map
        fig, axes = plt.subplots(1, 5, figsize=(30, 6), sharey=True)
        v_min, v_max = float(sample_grades.min()), float(sample_grades.max())

        b_poly = np.array(list(boundary) + [boundary[0]])

        # Compute convex hull for outline
        hull = ConvexHull(samples_xy)
        hull_pts = np.vstack([samples_xy[hull.vertices], samples_xy[hull.vertices[0]]])

        spatial_configs = [
            (
                axes[0],
                "Ordinary Kriging (OK)",
                ok_grades,
                "viridis",
                v_min,
                v_max,
                "Grade (% Cu)",
            ),
            (
                axes[1],
                "Simple Kriging (SK)",
                sk_grades,
                "viridis",
                v_min,
                v_max,
                "Grade (% Cu)",
            ),
            (
                axes[2],
                "Inverse Distance Squared (IDW²)",
                idw_grades,
                "viridis",
                v_min,
                v_max,
                "Grade (% Cu)",
            ),
            (
                axes[3],
                "OK Estimation Variance (σ²)",
                ok_variances,
                "magma",
                0.0,
                float(args.nugget + args.sill),
                "Variance (σ²)",
            ),
            (
                axes[4],
                "OK (Strict Interpolation)",
                ok_strict_grades,
                "viridis",
                v_min,
                v_max,
                "Grade (% Cu)",
            ),
        ]

        for ax, title, values, cmap, c_min, c_max, label in spatial_configs:
            img_flat = np.full(xx.size, np.nan)
            img_flat[inside_mask] = values
            img_grid = img_flat.reshape(xx.shape)

            im = ax.imshow(
                img_grid,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=c_min,
                vmax=c_max,
                alpha=0.9,
            )

            # Overlay concession boundary
            ax.plot(
                b_poly[:, 0],
                b_poly[:, 1],
                "r--",
                linewidth=1.5,
                label="Concession Boundary",
            )

            # Overlay drillhole convex hull
            ax.plot(
                hull_pts[:, 0],
                hull_pts[:, 1],
                color="cyan",
                linestyle="-.",
                linewidth=2.0,
                label="Drillhole Convex Hull",
            )

            # Overlay drillhole collars
            ax.scatter(
                samples_xy[:, 0],
                samples_xy[:, 1],
                c=sample_grades,
                cmap="viridis",
                vmin=v_min,
                vmax=v_max,
                edgecolors="white",
                linewidth=1.2,
                s=55,
                zorder=5,
                label="Drillhole Collars",
            )

            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Easting (m)", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.5)

            cbar = fig.colorbar(
                im, ax=ax, orientation="horizontal", pad=0.10, shrink=0.8
            )
            cbar.set_label(label, fontsize=9)

        axes[0].set_ylabel("Northing (m)", fontsize=10)
        axes[0].legend(loc="upper left", fontsize=8)

        plt.suptitle(
            f"Geostatistical Resource Evaluation: Spatial Estimates & Uncertainty (Variance)\n"
            f"Variogram: {args.variogram.title()} (c0={args.nugget:.2f}, c={args.sill:.2f}, a={args.range:.0f}m)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSpatial comparison map saved to: {args.save_plot}")

        # B. Individual Grade-Tonnage Curves (One Dedicated Dual-Axis Plot per Model)
        gt_file_map = {
            "Ordinary Kriging (OK)": "kriging_ok_grade_tonnage.png",
            "Simple Kriging (SK)": "kriging_sk_grade_tonnage.png",
            "Inverse Distance Squared (IDW²)": "kriging_idw_grade_tonnage.png",
        }

        print("\nGenerating individual Grade–Tonnage plots for each estimator...")
        for model_name, filename in gt_file_map.items():
            fig_gt, ax_gt = plot_grade_tonnage_curve(
                gt_models[model_name],
                grade_unit="% Cu",
                tonnage_unit="Mt",
                title=f"Grade–Tonnage Sensitivity Curve - {model_name}",
                show_metal=True,
            )
            out_file = str(plots_dir / filename)
            fig_gt.savefig(out_file, dpi=180, bbox_inches="tight")
            plt.close(fig_gt)
            print(f"  • {model_name}: saved to {out_file}")

        # C. Cell Declustering Optimization Curve (SME Handbook & NI 43-101 Standard)
        declust_plot_file = str(plots_dir / "kriging_cell_declustering_curve.png")
        fig_dec, ax_dec = plot_cell_declustering_curve(
            declust_df,
            naive_mean=global_mean,
            optimal_cell_size=opt_cell_size,
            grade_unit="% Cu",
            title=f"Cell Declustering Sensitivity Curve (Optimal: {opt_cell_size:.1f}m)",
        )
        fig_dec.savefig(declust_plot_file, dpi=180, bbox_inches="tight")
        plt.close(fig_dec)
        print(f"  • Cell Declustering curve: saved to {declust_plot_file}")

        # D. Directional Swath Plots (Local Drift Analysis - NI 43-101 / JORC Standard)
        block_model_df = pd.DataFrame(
            {
                "x": grid_points[:, 0],
                "y": grid_points[:, 1],
                "ok_grade": ok_grades,
                "idw_grade": idw_grades,
                "tonnes": block_tonnes,
            }
        )

        swath_x_file = str(plots_dir / "kriging_swath_easting.png")
        fig_swath_x, _ = plot_swath_analysis(
            block_model_df,
            drillholes=drillholes,
            axis="x",
            bin_width=40.0,
            grade_col="ok_grade",
            validation_grade_col="idw_grade",
            drillhole_grade_col="grade",
            model_name="Ordinary Kriging",
            validation_model_name="IDW² (Validation Benchmark)",
            tonnes_col="tonnes",
            grade_unit="% Cu",
            tonnage_unit="Mt",
            title="Swath Plot (Local Drift Analysis) Along Easting (X)",
        )
        fig_swath_x.savefig(swath_x_file, dpi=180, bbox_inches="tight")
        plt.close(fig_swath_x)
        print(f"  • Swath Plot (Easting): saved to {swath_x_file}")

        swath_y_file = str(plots_dir / "kriging_swath_northing.png")
        fig_swath_y, _ = plot_swath_analysis(
            block_model_df,
            drillholes=drillholes,
            axis="y",
            bin_width=40.0,
            grade_col="ok_grade",
            validation_grade_col="idw_grade",
            drillhole_grade_col="grade",
            model_name="Ordinary Kriging",
            validation_model_name="IDW² (Validation Benchmark)",
            tonnes_col="tonnes",
            grade_unit="% Cu",
            tonnage_unit="Mt",
            title="Swath Plot (Local Drift Analysis) Along Northing (Y)",
        )
        fig_swath_y.savefig(swath_y_file, dpi=180, bbox_inches="tight")
        plt.close(fig_swath_y)
        print(f"  • Swath Plot (Northing): saved to {swath_y_file}")

        # E. Resource-to-Reserve Waterfall Reconciliation (Bridge)
        waterfall_file = str(plots_dir / "kriging_reserve_waterfall.png")
        fig_wf, _ = plot_resource_to_reserve_waterfall(
            block_class_df,
            reserve_df,
            cutoff_grade=base_cutoff,
            mining_dilution_pct=dilution_pct,
            mining_recovery_pct=recovery_pct,
            dilution_grade=dilution_grade,
            grade_unit="% Cu",
            tonnage_unit="Mt",
            metal_unit="kt",
            title="Executive Resource-to-Reserve Bridge (Modifying Factors Reconciliation)",
        )
        fig_wf.savefig(waterfall_file, dpi=180, bbox_inches="tight")
        plt.close(fig_wf)
        print(f"  • Resource-to-Reserve Waterfall: saved to {waterfall_file}")

        # F. Spatial Reserve & Resource Classification Map
        status_arr = np.empty(len(grid_points), dtype=object)
        for i in range(len(grid_points)):
            cat = categories[i]
            g = ok_grades[i]
            if cat == "Inferred":
                status_arr[i] = "Inferred Resource (Excluded)"
            elif g < base_cutoff:
                status_arr[i] = "Sub-Economic / Waste"
            elif cat == "Measured":
                status_arr[i] = "Proven Reserve"
            elif cat == "Indicated":
                status_arr[i] = "Probable Reserve"
            else:
                status_arr[i] = "Sub-Economic / Waste"

        map_blocks_df = pd.DataFrame(
            {
                "x": grid_points[:, 0],
                "y": grid_points[:, 1],
                "status": status_arr,
            }
        )

        class_map_file = str(plots_dir / "kriging_reserve_classification_map.png")
        fig_map, _ = plot_reserve_classification_map(
            map_blocks_df,
            boundary=boundary,
            drillholes=drillholes,
            title="Spatial Mineral Reserve & Resource Classification Plan Map",
        )
        fig_map.savefig(class_map_file, dpi=180, bbox_inches="tight")
        plt.close(fig_map)
        print(f"  • Reserve Classification Map: saved to {class_map_file}")

        # G. In-Situ Resource vs. Diluted Reserve Grade-Tonnage Shift Curve
        cutoffs_eval = [
            0.0,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.2,
            1.4,
            1.6,
        ]
        is_mi_mask = (categories == "Measured") | (categories == "Indicated")
        insitu_mi_df = pd.DataFrame(
            {
                "grade": ok_grades[is_mi_mask],
                "tonnes": block_tonnes,
            }
        )
        insitu_gt = grade_tonnage_table(insitu_mi_df, cutoffs=cutoffs_eval)

        dil_mi_tonnes = (
            block_tonnes * (1.0 + dilution_pct / 100.0) * (recovery_pct / 100.0)
        )
        dil_mi_grades = (
            ok_grades[is_mi_mask] + (dilution_pct / 100.0) * dilution_grade
        ) / (1.0 + dilution_pct / 100.0)
        dil_res_df = pd.DataFrame(
            {
                "grade": dil_mi_grades,
                "tonnes": dil_mi_tonnes,
            }
        )
        diluted_gt = grade_tonnage_table(dil_res_df, cutoffs=cutoffs_eval)

        gt_shift_file = str(plots_dir / "kriging_in_situ_vs_diluted_gt.png")
        fig_shift, _ = plot_in_situ_vs_diluted_curves(
            insitu_gt,
            diluted_gt,
            grade_unit="% Cu",
            tonnage_unit="Mt",
            title="Grade–Tonnage Operational Shift: In-Situ Resource vs. Diluted Reserve",
        )
        fig_shift.savefig(gt_shift_file, dpi=180, bbox_inches="tight")
        plt.close(fig_shift)
        print(f"  • In-Situ vs Diluted Shift Curve: saved to {gt_shift_file}")

    print("\n" + "=" * 70)
    print("                    ESTIMATION RUN COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

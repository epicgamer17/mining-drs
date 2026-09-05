"""Example: 2D Block Model Estimation using IDW and Nearest Neighbor.

Demonstrates spatial interpolation of exploration drillholes across a regular
mine block grid:
1. Defines exploration drillhole assay samples and concession boundary.
2. Generates a regular 2D block grid across the deposit.
3. Compares three industry-standard spatial estimators:
   - Nearest Neighbor (NN): Polygonal equivalent (k=1)
   - Inverse Distance Squared (IDW²): Standard mining benchmark (alpha=2, k=8)
   - Smooth Inverse Distance (IDW¹): Moving average style (alpha=1, k=12)
4. Analyzes the spatial smoothing effect (variance reduction).
5. Computes Grade-Tonnage curves across economic cutoff grades.
6. Renders side-by-side spatial heatmaps comparing all three models.
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
    inverse_distance_weighting,
    nearest_neighbor_grid_estimation,
    is_within_convex_hull,
    grade_tonnage_table,
    plot_grade_tonnage_curve,
    cell_declustering,
    plot_cell_declustering_curve,
    plot_swath_analysis,
    classify_resources_by_drill_spacing,
    format_resource_statement,
    calculate_cut_off_grade,
    convert_resource_to_reserve,
    format_reserve_statement,
    plot_resource_to_reserve_waterfall,
    plot_reserve_classification_map,
    plot_resource_classification_map,
    plot_in_situ_vs_diluted_curves,
)


def create_sample_deposit() -> tuple[pd.DataFrame, list[tuple[float, float]]]:
    """Sample copper porphyry exploration drillholes and concession perimeter."""
    drillholes = pd.DataFrame({
        "hole_id": [
            "DH01", "DH02", "DH03", "DH04",
            "DH05", "DH06", "DH07", "DH08",
            "DH09", "DH10", "DH11", "DH12",
            "DH13", "DH14", "DH15", "DH16",
        ],
        "x": [
            120.0, 240.0, 380.0, 520.0,
            150.0, 280.0, 420.0, 560.0,
            130.0, 260.0, 400.0, 540.0,
            170.0, 300.0, 450.0, 580.0,
        ],
        "y": [
            120.0, 110.0, 130.0, 140.0,
            230.0, 240.0, 220.0, 250.0,
            350.0, 360.0, 340.0, 370.0,
            460.0, 470.0, 450.0, 480.0,
        ],
        "grade": [
            0.35, 0.65, 1.10, 0.45,
            0.52, 1.55, 1.95, 0.85,
            0.40, 1.30, 1.70, 0.72,
            0.28, 0.60, 0.95, 0.40,
        ],  # % Cu
    })

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
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
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
    parser = argparse.ArgumentParser(description="IDW & Nearest Neighbor Grid Demo")
    parser.add_argument(
        "--grid-res",
        type=float,
        default=10.0,
        help="Grid block resolution in meters (default: 10.0 m)",
    )
    parser.add_argument(
        "--max-radius",
        type=float,
        default=160.0,
        help="Search neighborhood radius in meters (default: 160.0 m)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="plots/idw_vs_nn_comparison.png",
        help="Path to save comparison figure",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable figure generation",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("       2D BLOCK ESTIMATION: NEAREST NEIGHBOR VS. IDW")
    print("=" * 70)

    drillholes, boundary = create_sample_deposit()
    samples_xy = drillholes[["x", "y"]].to_numpy()
    sample_grades = drillholes["grade"].to_numpy()

    print(f"\nLoaded {len(drillholes)} drillholes. Assays min={sample_grades.min():.2f}%, max={sample_grades.max():.2f}%, mean={sample_grades.mean():.2f}% Cu.")

    # 1. Generate grid blocks
    grid_points, grid_info, extent = generate_grid_points(
        boundary, grid_resolution=args.grid_res
    )
    block_tonnes = (args.grid_res ** 2) * 12.0 * 2.7  # 12m bench height, 2.7 t/m^3
    print(f"Generated {len(grid_points):,} grid blocks ({args.grid_res:.0f}m x {args.grid_res:.0f}m).")
    print(f"Total Block Model Tonnage: {len(grid_points) * block_tonnes:,.0f} tonnes.")

    # 2. Cell Declustering Analysis (SME Handbook Section 4.3)
    declust_weights, declust_df, opt_cell_size = cell_declustering(
        drillholes, grade_col="grade", x_col="x", y_col="y", min_mean=True
    )
    opt_declust_mean = float(
        declust_df.loc[declust_df["cell_size"] == opt_cell_size, "declustered_mean"].iloc[0]
    )
    naive_mean = float(sample_grades.mean())
    bias_removed_pct = ((naive_mean - opt_declust_mean) / naive_mean) * 100.0

    print("\n--- Cell Declustering Analysis (SME Handbook Section 4.3) ---")
    print(f"Optimal Declustering Cell Size : {opt_cell_size:.1f} m (Minimum Declustered Mean)")
    print(f"Naive (Unweighted) Mean Assay  : {naive_mean:.3f}% Cu")
    print(f"Representative Declustered Mean: {opt_declust_mean:.3f}% Cu (Bias Removed: {bias_removed_pct:+.1f}%)")

    # 3. Spatial Classification: Interpolation vs Extrapolation Audit
    is_interpolated = is_within_convex_hull(samples_xy, grid_points)
    n_interpolated = int(is_interpolated.sum())
    n_extrapolated = len(grid_points) - n_interpolated

    print("\n--- Spatial Audit: Interpolation vs. Extrapolation (Convex Hull) ---")
    print(f"Interpolated Blocks (Inside Convex Hull)  : {n_interpolated:,d} ({n_interpolated / len(grid_points) * 100:.1f}%) -> {n_interpolated * block_tonnes:,.0f} tonnes [High Confidence]")
    print(f"Extrapolated Blocks (Outside Convex Hull) : {n_extrapolated:,d} ({n_extrapolated / len(grid_points) * 100:.1f}%) -> {n_extrapolated * block_tonnes:,.0f} tonnes [Exploration Risk]")

    # 4. Run Estimator 1: Nearest Neighbor (NN)
    print(f"\n[1/4] Running Nearest Neighbor (NN)...")
    nn_grades, _ = nearest_neighbor_grid_estimation(
        samples_xy, sample_grades, grid_points, max_radius=args.max_radius
    )

    # 5. Run Estimator 2: IDW² (Standard Mining Default)
    print(f"[2/4] Running IDW² (power=2.0, k=8)...")
    idw2_grades, _ = inverse_distance_weighting(
        samples_xy, sample_grades, grid_points, power=2.0, k_neighbors=8, max_radius=args.max_radius
    )

    # 5. Run Estimator 3: IDW¹ (Smooth Moving Average)
    print(f"[3/4] Running IDW¹ (power=1.0, k=12)...")
    idw1_grades, _ = inverse_distance_weighting(
        samples_xy, sample_grades, grid_points, power=1.0, k_neighbors=12, max_radius=args.max_radius
    )

    # 6. Run Estimator 4: IDW² (Strict Interpolation Only, Mask Extrapolation)
    print(f"[4/4] Running IDW² with Extrapolation Masked (Strict Interpolation)...")
    idw2_strict, _ = inverse_distance_weighting(
        samples_xy, sample_grades, grid_points, power=2.0, k_neighbors=8, max_radius=args.max_radius, mask_extrapolation=True
    )

    # 7. Comparative Summary Statistics
    models = {
        "Nearest Neighbor (NN)": nn_grades,
        "IDW² (Full Extrapolation)": idw2_grades,
        "IDW¹ (Smooth Moving Avg)": idw1_grades,
        "IDW² (Strict Interpolation)": idw2_strict,
    }

    print("\n--- Model Comparative Statistics (Smoothing & Extrapolation Audit) ---")
    stats_rows = []
    for name, g in models.items():
        valid = g[~np.isnan(g)]
        stats_rows.append({
            "Model": name,
            "Mean (% Cu)": f"{valid.mean():.3f}",
            "Std Dev": f"{valid.std():.3f}",
            "Min (%)": f"{valid.min():.3f}",
            "Max (%)": f"{valid.max():.3f}",
            "Estimated Blocks": f"{len(valid):,d} ({len(valid) / len(g) * 100.0:.1f}%)",
        })
    print(pd.DataFrame(stats_rows).to_string(index=False))

    # 8. Cutoff Grade-Tonnage Sensitivity for All Models (Model Audit)
    cutoffs = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    gt_models = {}
    for name, g in models.items():
        block_df = pd.DataFrame({
            "grade": g[~np.isnan(g)],
            "tonnes": block_tonnes,
        })
        gt_models[name] = grade_tonnage_table(block_df, cutoffs=cutoffs)

    print("\n--- IDW² Grade–Tonnage Sensitivity Curve ---")
    gt_disp = gt_models["IDW² (Full Extrapolation)"].copy()
    gt_disp["ore_tonnes"] = gt_disp["ore_tonnes"].map(lambda x: f"{x:,.0f}")
    gt_disp["ore_grade"] = gt_disp["ore_grade"].map(lambda x: f"{x:.3f}%")
    gt_disp["internal_waste_tonnes"] = gt_disp["internal_waste_tonnes"].map(lambda x: f"{x:,.0f}")
    gt_disp["internal_waste_ratio"] = gt_disp["internal_waste_ratio"].map(lambda x: f"{x:.2f}")
    gt_disp["ore_recovery_pct"] = gt_disp["ore_recovery_pct"].map(lambda x: f"{x:.1f}%")
    gt_disp["metal_recovery_pct"] = gt_disp["metal_recovery_pct"].map(lambda x: f"{x:.1f}%")
    print(gt_disp[["ore_tonnes", "ore_grade", "internal_waste_tonnes", "internal_waste_ratio", "ore_recovery_pct", "metal_recovery_pct"]].to_string())

    # 9. Official Mineral Resource Statement (NI 43-101 / JORC Code Compliant)
    categories = classify_resources_by_drill_spacing(
        grid_points=grid_points,
        samples_xy=samples_xy,
        max_radius_measured=60.0,
        max_radius_indicated=1e9,
        min_holes_measured=1,
        min_holes_indicated=1,
        is_interpolated=is_interpolated,
    )

    block_class_df = pd.DataFrame(
        {
            "category": categories,
            "grade": idw2_grades,
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
        f"\n--- Official Mineral Resource Statement (IDW² Model, Cutoff: {base_cutoff:.2f}% Cu) ---"
    )
    print(resource_stmt.to_string(index=False))
    print("\nCompliance Footnotes:")
    for fn in resource_stmt.attrs.get("footnotes", []):
        print(f"  {fn}")

    # 10. Mineral Reserve Delineation: Applying Modifying Factors (CIM / NI 43-101)
    dilution_pct = 5.0
    dilution_grade = 0.05
    recovery_pct = 95.0

    reserve_df = convert_resource_to_reserve(
        block_class_df,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        cutoff_grade=base_cutoff,
        dilution_grade=dilution_grade,
    )

    reserve_stmt = format_reserve_statement(
        reserve_df,
        cutoff_grade=base_cutoff,
        mining_dilution_pct=dilution_pct,
        mining_recovery_pct=recovery_pct,
        commodity_price="$3.80/lb Cu",
        metallurgical_recovery=88.0,
        tonnage_unit="Mt",
        grade_unit="% Cu",
        metal_unit="kt",
        rpeee_constraint="Constrained within Lerchs-Grossmann optimized pit shell",
    )

    print(f"\n--- Official Mineral Reserve Statement (IDW² Model, Cutoff: {base_cutoff:.2f}% Cu) ---")
    print(reserve_stmt.to_string(index=False))
    print("\nCompliance Footnotes:")
    for fn in reserve_stmt.attrs.get("footnotes", []):
        print(f"  {fn}")

    # 11. Spatial Visualization & Comparative Grade-Tonnage Chart
    if not args.no_plot:
        plots_dir = Path(args.save_plot).parent
        plots_dir.mkdir(parents=True, exist_ok=True)
        xx, yy, inside_mask = grid_info

        fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
        v_min, v_max = float(sample_grades.min()), float(sample_grades.max())

        b_poly = np.array(list(boundary) + [boundary[0]])

        # Compute convex hull vertices of drillholes for drawing
        hull = ConvexHull(samples_xy)
        hull_pts = np.vstack([samples_xy[hull.vertices], samples_xy[hull.vertices[0]]])

        for ax, (name, g_vals) in zip(axes, models.items()):
            # Reconstruct 2D grid image
            img_grid_flat = np.full(xx.size, np.nan)
            img_grid_flat[inside_mask] = g_vals
            img_grid = img_grid_flat.reshape(xx.shape)

            im = ax.imshow(
                img_grid,
                origin="lower",
                extent=extent,
                cmap="viridis",
                vmin=v_min,
                vmax=v_max,
                alpha=0.9,
            )

            # Draw concession boundary
            ax.plot(b_poly[:, 0], b_poly[:, 1], "r--", linewidth=1.5, label="Boundary")

            # Draw drillhole convex hull
            ax.plot(
                hull_pts[:, 0],
                hull_pts[:, 1],
                color="orange",
                linewidth=1.8,
                linestyle="-.",
                label="Drillhole Hull",
            )

            # Draw drillhole collars
            ax.scatter(
                samples_xy[:, 0],
                samples_xy[:, 1],
                c=sample_grades,
                cmap="viridis",
                edgecolor="black",
                s=70,
                vmin=v_min,
                vmax=v_max,
                zorder=5,
            )

            for _, row in drillholes.iterrows():
                ax.annotate(
                    f"{row['grade']:.2f}%",
                    (row["x"], row["y"]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=7,
                    fontweight="bold",
                )

            ax.set_title(name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Easting (m)")
            ax.grid(True, linestyle=":", alpha=0.4)

        axes[0].set_ylabel("Northing (m)")
        fig.subplots_adjust(right=0.92, wspace=0.1)
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Copper Grade (% Cu)", fontsize=11)

        fig.savefig(args.save_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSpatial comparison figure saved to: {args.save_plot}")

        # Individual Grade-Tonnage Curves (One Dedicated Dual-Axis Plot per Model)
        gt_file_map = {
            "Nearest Neighbor (NN)": "nn_grade_tonnage.png",
            "IDW² (Full Extrapolation)": "idw2_grade_tonnage.png",
            "IDW¹ (Smooth Moving Avg)": "idw1_grade_tonnage.png",
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

        # Cell Declustering Sensitivity Curve
        declust_plot_file = str(plots_dir / "idw_cell_declustering_curve.png")
        fig_dec, ax_dec = plot_cell_declustering_curve(
            declust_df,
            naive_mean=naive_mean,
            optimal_cell_size=opt_cell_size,
            grade_unit="% Cu",
            title=f"Cell Declustering Sensitivity Curve (Optimal: {opt_cell_size:.1f}m)",
        )
        fig_dec.savefig(declust_plot_file, dpi=180, bbox_inches="tight")
        plt.close(fig_dec)
        print(f"  • Cell Declustering curve: saved to {declust_plot_file}")

        # Directional Swath Plots: IDW² vs. Nearest Neighbor (Declustered Check)
        swath_df = pd.DataFrame(
            {
                "x": grid_points[:, 0],
                "y": grid_points[:, 1],
                "idw2_grade": idw2_grades,
                "nn_grade": nn_grades,
                "tonnes": block_tonnes,
            }
        )

        swath_x_file = str(plots_dir / "idw_swath_easting.png")
        fig_swath_x, _ = plot_swath_analysis(
            swath_df,
            drillholes=drillholes,
            axis="x",
            bin_width=40.0,
            grade_col="idw2_grade",
            validation_grade_col="nn_grade",
            drillhole_grade_col="grade",
            model_name="IDW² (Estimate)",
            validation_model_name="Nearest Neighbor (Declustered Check)",
            tonnes_col="tonnes",
            grade_unit="% Cu",
            tonnage_unit="Mt",
            title="Swath Plot (Local Drift Analysis) Along Easting (X)",
        )
        fig_swath_x.savefig(swath_x_file, dpi=180, bbox_inches="tight")
        plt.close(fig_swath_x)
        print(f"  • Swath Plot (Easting): saved to {swath_x_file}")

        swath_y_file = str(plots_dir / "idw_swath_northing.png")
        fig_swath_y, _ = plot_swath_analysis(
            swath_df,
            drillholes=drillholes,
            axis="y",
            bin_width=40.0,
            grade_col="idw2_grade",
            validation_grade_col="nn_grade",
            drillhole_grade_col="grade",
            model_name="IDW² (Estimate)",
            validation_model_name="Nearest Neighbor (Declustered Check)",
            tonnes_col="tonnes",
            grade_unit="% Cu",
            tonnage_unit="Mt",
            title="Swath Plot (Local Drift Analysis) Along Northing (Y)",
        )
        fig_swath_y.savefig(swath_y_file, dpi=180, bbox_inches="tight")
        plt.close(fig_swath_y)
        print(f"  • Swath Plot (Northing): saved to {swath_y_file}")

        # Resource-to-Reserve Waterfall Reconciliation (Bridge)
        waterfall_file = str(plots_dir / "idw_reserve_waterfall.png")
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
            title="IDW² Resource-to-Reserve Bridge (Modifying Factors Reconciliation)",
        )
        fig_wf.savefig(waterfall_file, dpi=180, bbox_inches="tight")
        plt.close(fig_wf)
        print(f"  • Resource-to-Reserve Waterfall: saved to {waterfall_file}")

        # Spatial Reserve & Resource Classification Map
        status_arr = np.empty(len(grid_points), dtype=object)
        for i in range(len(grid_points)):
            cat = categories[i]
            g = idw2_grades[i]
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

        # Resource Classification Map (In-Situ Confidence)
        res_blocks_df = pd.DataFrame(
            {
                "x": grid_points[:, 0],
                "y": grid_points[:, 1],
                "category": categories,
            }
        )
        res_map_file = str(plots_dir / "idw_resource_classification_map.png")
        fig_res_map, _ = plot_resource_classification_map(
            res_blocks_df,
            boundary=boundary,
            drillholes=drillholes,
            title="IDW²: In-Situ Mineral Resource Classification Map (CIM / JORC)",
        )
        fig_res_map.savefig(res_map_file, dpi=180, bbox_inches="tight")
        plt.close(fig_res_map)
        print(f"  • Resource Classification Map: saved to {res_map_file}")

        class_map_file = str(plots_dir / "idw_reserve_classification_map.png")
        fig_map, _ = plot_reserve_classification_map(
            map_blocks_df,
            boundary=boundary,
            drillholes=drillholes,
            title="IDW² Mineral Reserve & Resource Classification Map",
        )
        fig_map.savefig(class_map_file, dpi=180, bbox_inches="tight")
        plt.close(fig_map)
        print(f"  • Reserve Classification Map: saved to {class_map_file}")

        # In-Situ Resource vs. Diluted Reserve Grade-Tonnage Shift Curve
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
                "grade": idw2_grades[is_mi_mask],
                "tonnes": block_tonnes,
            }
        )
        insitu_gt = grade_tonnage_table(insitu_mi_df, cutoffs=cutoffs_eval)

        dil_mi_tonnes = (
            block_tonnes * (1.0 + dilution_pct / 100.0) * (recovery_pct / 100.0)
        )
        dil_mi_grades = (
            idw2_grades[is_mi_mask] + (dilution_pct / 100.0) * dilution_grade
        ) / (1.0 + dilution_pct / 100.0)
        dil_res_df = pd.DataFrame(
            {
                "grade": dil_mi_grades,
                "tonnes": dil_mi_tonnes,
            }
        )
        diluted_gt = grade_tonnage_table(dil_res_df, cutoffs=cutoffs_eval)

        gt_shift_file = str(plots_dir / "idw_in_situ_vs_diluted_gt.png")
        fig_shift, _ = plot_in_situ_vs_diluted_curves(
            insitu_gt,
            diluted_gt,
            grade_unit="% Cu",
            tonnage_unit="Mt",
            title="IDW² Grade–Tonnage Operational Shift: In-Situ Resource vs. Diluted Reserve",
        )
        fig_shift.savefig(gt_shift_file, dpi=180, bbox_inches="tight")
        plt.close(fig_shift)
        print(f"  • In-Situ vs Diluted Shift Curve: saved to {gt_shift_file}")


if __name__ == "__main__":
    main()

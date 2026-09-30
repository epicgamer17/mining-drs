"""Example: 3D Block Model Estimation using Ordinary Block Kriging and Support Effect.

Demonstrates industry-standard 3D block model workflows for selective mining units (SMUs):
1. Generates a 3D exploration drillhole composite dataset (X, Y, Z) with depth intervals.
2. Constructs a regular 3D Block Model grid (dx=10m, dy=10m, dz=5m) with rock densities.
3. Performs 3D Ordinary Block Kriging with internal point discretization (4x4x2 = 32 sub-points).
4. Quantifies the Support Effect (volume-variance reduction relation):
   - Theoretical point sill C(0)
   - Block self-covariance C_bar(V, V)
   - Block dispersion variance BV = C(0) - C_bar(V, V)
5. Generates the 4 industry-standard visualization archetypes:
   - Method 1: Orthogonal Slices (Bench Plan X-Y, Cross-Section X-Z, Longitudinal Section Y-Z).
   - Method 2: Multi-Bench Depth Gallery (Faceted elevation levels through the deposit).
   - Method 3: 3D Isometric View with cut-off grade filtering to reveal internal ore shoots.
   - Method 4: Dual Grade vs. Kriging Estimation Variance Audit Panels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drs_mining.components.estimation import (
    cell_declustering,
    create_block_model,
    ordinary_kriging_block_estimation,
    inverse_distance_weighting_block_estimation,
    nearest_neighbor_block_estimation,
    kriging_cross_validation,
    plot_cross_validation_diagnostics,
    compare_model_estimation_methods,
    kriging_neighborhood_analysis,
    classify_resources_by_sor,
    classify_mineral_resources,
    smooth_resource_categories,
    convert_resource_to_reserve,
    classify_reserve_status,
    format_reserve_statement,
    plot_resource_classification_map,
    plot_resource_classification_orthogonal_slices,
    plot_resource_classification_bench_gallery,
    plot_resource_classification_3d_isometric,
    plot_resource_classification_3d_interactive,
    plot_reserve_classification_map,
    plot_reserve_classification_orthogonal_slices,
    plot_reserve_classification_bench_gallery,
    plot_reserve_classification_3d_isometric,
    plot_reserve_classification_3d_interactive,
    plot_block_model_orthogonal_slices,
    plot_block_model_bench_gallery,
    plot_block_model_3d_isometric,
    plot_block_model_3d_interactive,
    plot_block_model_grade_uncertainty,
    plot_swath_analysis_3axis,
    grade_tonnage_table,
    plot_grade_tonnage_curve,
)


def generate_3d_porphyry_drillholes(
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generates synthetic 3D exploration drillhole composites for a copper-gold deposit.

    Creates 20 drillholes on an irregular 60m-80m grid with 5m down-hole composites
    penetrating a dipping high-grade porphyry core and surrounding low-grade halo.
    """
    rng = np.random.default_rng(seed)

    # Drillhole collars across 400m x 400m exploration area
    collar_coords = [
        (80.0, 80.0),
        (160.0, 70.0),
        (240.0, 90.0),
        (320.0, 80.0),
        (70.0, 150.0),
        (150.0, 160.0),
        (230.0, 150.0),
        (310.0, 170.0),
        (90.0, 230.0),
        (170.0, 240.0),
        (250.0, 230.0),
        (330.0, 250.0),
        (80.0, 310.0),
        (160.0, 320.0),
        (240.0, 310.0),
        (320.0, 330.0),
        # Infill holes in high-grade central area
        (190.0, 190.0),
        (210.0, 210.0),
        (180.0, 220.0),
        (220.0, 180.0),
    ]

    records = []
    # Core geometry: centered at (200, 200, 40) dipping south-east
    core_center = np.array([200.0, 200.0, 40.0])

    for h_idx, (cx, cy) in enumerate(collar_coords):
        hole_id = f"DH_{h_idx + 1:02d}"
        # Downhole composites every 5m from surface (Z=100m) to depth (Z=10m)
        elevations = np.arange(97.5, 12.5, -5.0)

        for z in elevations:
            pt = np.array([cx, cy, z])
            # Distance to plunging mineralized core
            dist_core = np.sqrt(
                ((pt[0] - core_center[0]) / 80.0) ** 2
                + ((pt[1] - core_center[1]) / 70.0) ** 2
                + ((pt[2] - core_center[2]) / 35.0) ** 2
            )

            # High grade core (~2.2% Cu) decaying to background halo (~0.2% Cu)
            base_grade = 0.20 + 2.0 * np.exp(-0.5 * (dist_core**2))
            # Geostatistical log-normal assay noise
            noise = rng.lognormal(mean=0.0, sigma=0.22)
            grade = float(np.clip(base_grade * noise, 0.05, 4.50))

            records.append(
                {
                    "hole_id": hole_id,
                    "x": cx,
                    "y": cy,
                    "z": z,
                    "grade": round(grade, 3),
                }
            )

    df_dh = pd.DataFrame(records)
    samples_xyz = df_dh[["x", "y", "z"]].to_numpy()
    sample_grades = df_dh["grade"].to_numpy()
    return samples_xyz, sample_grades, df_dh


def main():
    parser = argparse.ArgumentParser(description="3D Block Model Kriging Demonstrator")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive 3D deposit explorer with cut-off and elevation sliders",
    )
    parser.add_argument(
        "--interactive-view",
        choices=["all", "grade", "resource", "reserve"],
        default="all",
        help="Interactive 3D explorer to display in interactive mode (default: all)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation and file saving",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("3D BLOCK MODEL ESTIMATION & SUPPORT EFFECT DEMONSTRATOR")
    print("=" * 80)

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Generate 3D Exploration Drillhole Composite Data
    # -------------------------------------------------------------------------
    print("\n[Step 1] Generating 3D exploration drillholes...")
    samples_xyz, sample_grades, df_dh = generate_3d_porphyry_drillholes(seed=42)
    print(
        f"  • Total composite assays: {len(df_dh)} from {df_dh['hole_id'].nunique()} drillholes"
    )
    print(
        f"  • Elevation range: Z = {df_dh['z'].min():.1f}m to {df_dh['z'].max():.1f}m"
    )
    print(
        f"  • Raw composite grade: mean={sample_grades.mean():.3f}%, std={sample_grades.std():.3f}%, max={sample_grades.max():.3f}%"
    )

    # -------------------------------------------------------------------------
    # 1b. 3D Cell Declustering (CIM MRMR §4.2 / JORC Table 1)
    # -------------------------------------------------------------------------
    print("\n[Step 1b] Running 3D Cell Declustering on Exploration Composites...")
    z_aspect = 0.5
    declust_weights, declust_summary, opt_cell = cell_declustering(
        df_dh,
        grade_col="grade",
        x_col="x",
        y_col="y",
        z_col="z",
        z_aspect_ratio=z_aspect,  # Vertical aspect ratio for 5m downhole composites vs horizontal spacing
        cell_sizes=np.linspace(20.0, 150.0, 14),
    )
    raw_mean = float(df_dh["grade"].mean())
    declust_mean = float(np.average(df_dh["grade"], weights=declust_weights))
    opt_cell_tuple = (
        (opt_cell[0], opt_cell[1], opt_cell[2])
        if isinstance(opt_cell, (tuple, list, np.ndarray))
        else (float(opt_cell), float(opt_cell), float(opt_cell) * z_aspect)
    )
    print(
        f"  • Optimal 3D cell size: {opt_cell_tuple[0]:.1f}m x {opt_cell_tuple[1]:.1f}m x {opt_cell_tuple[2]:.1f}m"
    )
    print(
        f"  • Raw composite mean: {raw_mean:.3f}% Cu -> Declustered mean: {declust_mean:.3f}% Cu (clustering bias: {raw_mean - declust_mean:+.3f}%)"
    )

    # -------------------------------------------------------------------------
    # 2. Construct 3D Block Model Grid (SMUs)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Constructing 3D Block Model Selective Mining Units (SMUs)...")
    origin = (20.0, 20.0, 10.0)
    block_size = (10.0, 10.0, 5.0)  # dx=10m, dy=10m, dz=5m bench height
    n_blocks = (36, 36, 18)  # 36 x 36 x 18 = 23,328 blocks spanning 360m x 360m x 90m

    block_model = create_block_model(
        origin=origin,
        block_size=block_size,
        n_blocks=n_blocks,
        default_density=2.70,
        default_domain="Porphyry",
    )
    total_tonnes = block_model["tonnes"].sum() / 1e6
    print(
        f"  • Block dimensions: {block_size[0]}m (E) x {block_size[1]}m (N) x {block_size[2]}m (bench)"
    )
    print(f"  • Total SMU blocks: {len(block_model):,}")
    print(f"  • Deposit in-situ tonnage: {total_tonnes:.2f} Mt @ 2.70 t/m³")

    # -------------------------------------------------------------------------
    # 3. Perform 3D Ordinary Block Kriging
    # -------------------------------------------------------------------------
    print("\n[Step 3] Running 3D Ordinary Block Kriging with Point Discretization...")
    sill = 0.85
    nugget = 0.15
    total_sill = sill + nugget
    range_param = 90.0
    discretization = (4, 4, 2)  # 32 internal discretization sub-points per SMU

    # Best-practice 3D search neighborhood configuration (SME Mining Engineering Handbook §4.5 & CIM MRMR):
    # - Rule 6: Max 4 composites per drillhole trace to prevent downhole collinear strings from dominating.
    # - Rule 7: Octant search with max 3 samples per octant and min 2 octants to enforce 3D spatial spread.
    # - Deutsch (1996): Negative kriging weight clipping and renormalization to eliminate screen-effect artifacts.
    estimates, variances, dispersion_var, lagrange_multipliers = (
        ordinary_kriging_block_estimation(
            samples_xyz=samples_xyz,
            sample_grades=sample_grades,
            block_model=block_model,
            sill=sill,
            range_param=range_param,
            nugget=nugget,
            discretization=discretization,
            variogram_model="spherical",
            k_neighbors=16,
            max_radius=110.0,
            min_samples=2,
            max_per_octant=3,
            min_sectors=2,
            max_per_drillhole=4,
            sample_hole_ids=df_dh["hole_id"].tolist(),
            clip_negative_weights=True,
        )
    )

    block_model["estimated_grade"] = estimates
    block_model["kriging_variance"] = variances

    # Kriging Neighborhood Analysis (KNA) Quality Metrics
    kriging_eff, slope_regr = kriging_neighborhood_analysis(
        kriging_variances=variances,
        block_dispersion_variance=dispersion_var,
        lagrange_multipliers=lagrange_multipliers,
    )
    block_model["kriging_efficiency"] = kriging_eff
    block_model["slope_of_regression"] = slope_regr

    # Support Effect Analysis
    bv = dispersion_var  # Block Dispersion Variance BV = C_bar(V, V) = sigma^2(V|D)
    within_block_var = (
        total_sill - bv
    )  # Within-block variance gamma_bar(V, V) = sigma^2(v|V)
    variance_reduction = within_block_var / total_sill
    print(f"  • Theoretical Point Sill C(0): {total_sill:.4f}")
    print(f"  • Block Dispersion Variance BV = C_bar(V, V): {bv:.4f}")
    print(f"  • Within-Block Variance sigma^2(v|V): {within_block_var:.4f}")
    print(f"  • Block Support Variance Retention: {(bv / total_sill) * 100:.1f}%")
    print(
        f"  • Variance Reduction from Point to Block: {variance_reduction * 100:.1f}%"
    )
    print(
        f"  • Estimated blocks: {np.sum(np.isfinite(estimates)):,} / {len(block_model):,} ({np.mean(np.isfinite(estimates))*100:.1f}%)"
    )

    # JORC / CIM Quality Diagnostics Summary
    valid_blocks = np.isfinite(kriging_eff) & np.isfinite(slope_regr)
    mean_ke = float(np.mean(kriging_eff[valid_blocks]))
    mean_sor = float(np.mean(slope_regr[valid_blocks]))
    sor_categories = classify_resources_by_sor(
        slopes_of_regression=slope_regr,
        kriging_efficiencies=kriging_eff,
        threshold_measured=0.80,
        threshold_indicated=0.50,
        min_kriging_efficiency=0.0,
    )
    pct_measured = float(np.mean(sor_categories[valid_blocks] == "Measured") * 100.0)
    pct_indicated = float(np.mean(sor_categories[valid_blocks] == "Indicated") * 100.0)
    print(f"  • Mean Kriging Efficiency (KE): {mean_ke * 100:.1f}%")
    print(f"  • Mean Slope of Regression (SoR): {mean_sor:.3f}")
    print(
        f"  • High Confidence Blocks (SoR >= 0.80, KE > 0, Measured candidate): {pct_measured:.1f}%"
    )
    print(
        f"  • Moderate Confidence Blocks (0.50 ≤ SoR < 0.80, KE > 0, Indicated candidate): {pct_indicated:.1f}%"
    )

    # Multi-criteria Resource Classification (Multi-hole independence + Spacing + SoR + KE)
    raw_resource_categories = classify_mineral_resources(
        grid_points=block_model[["x", "y", "z"]].values,
        samples_xy=samples_xyz,
        sample_hole_ids=df_dh["hole_id"].tolist(),
        max_radius_measured=45.0,
        max_radius_indicated=90.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        slopes_of_regression=slope_regr,
        kriging_efficiencies=kriging_eff,
        sor_threshold_measured=0.80,
        sor_threshold_indicated=0.50,
        min_kriging_efficiency=0.0,
    )
    # Post-processing spatial smoothing to eliminate 'spotted dog' artifacts (CIM MRMR Guidelines §6.11)
    resource_categories = smooth_resource_categories(
        grid_points=block_model[["x", "y", "z"]].values,
        categories=raw_resource_categories,
        k_neighbors=18,
        min_cluster_size=8,
        downgrade_isolated=False,
    )
    block_model["category"] = resource_categories

    # -------------------------------------------------------------------------
    # 3b. Comparative 3D Block IDW & Nearest Neighbor Benchmark Estimation
    # -------------------------------------------------------------------------
    print("\n[Step 3b] Running Comparative 3D IDW² and 3D Nearest Neighbor Benchmark Models...")
    idw_estimates, _ = inverse_distance_weighting_block_estimation(
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        block_model=block_model,
        power=2.0,
        discretization=(4, 4, 2),
        max_radius=110.0,
    )
    nn_estimates = nearest_neighbor_block_estimation(
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        block_model=block_model,
        max_radius=110.0,
    )
    block_model["idw_grade"] = idw_estimates
    block_model["nn_grade"] = nn_estimates
    print(f"  • 3D IDW² mean block grade: {np.nanmean(idw_estimates):.3f}% Cu")
    print(f"  • 3D NN declustered polygonal benchmark mean: {np.nanmean(nn_estimates):.3f}% Cu")
    print(f"  • 3D Ordinary Kriging mean block grade: {np.nanmean(estimates):.3f}% Cu")

    # -------------------------------------------------------------------------
    # 3b. Kriging Model Cross-Validation (Leave-One-Out Jackknife)
    # -------------------------------------------------------------------------
    print("\n[Step 3b] 3D Kriging Cross-Validation (Leave-One-Out Jackknife, CIM MRMR §6.10)...")
    cv_df = kriging_cross_validation(
        samples_xy=samples_xyz,
        sample_grades=sample_grades,
        variogram_model="spherical",
        sill=sill,
        range_param=range_param,
        nugget=nugget,
        method="ordinary",
        min_neighbors=2,
        max_neighbors=16,
        search_radius=110.0,
        max_per_drillhole=4,
        sample_hole_ids=df_dh["hole_id"].tolist(),
    )
    print(f"  • Cross-Validated Samples  : {cv_df.attrs['estimated_samples']} / {cv_df.attrs['total_samples']}")
    print(f"  • Mean Error (Bias)        : {cv_df.attrs['mean_error']:+.4f}% Cu  (Target: ~0.0)")
    print(f"  • Root Mean Sq Error (RMSE): {cv_df.attrs['root_mean_square_error']:.4f}% Cu")
    print(f"  • Mean Std Error (MSE)     : {cv_df.attrs['mean_standardized_error']:+.4f}  (Target: ~0.0)")
    print(f"  • Var Std Error (Var(SE))  : {cv_df.attrs['variance_standardized_error']:.4f}  (Target: ~1.0)")
    print(f"  • Pearson Correlation r    : {cv_df.attrs['pearson_correlation']:.3f}")
    print(f"  • Slope of Regression (SoR): {cv_df.attrs['slope_of_regression']:.3f}  (Target: ~1.0)")

    # -------------------------------------------------------------------------
    # 3c. Multi-Method Comparison vs. NN Benchmark (CIM MRMR §6.10)
    # -------------------------------------------------------------------------
    print("\n[Step 3c] Multi-Method Comparison vs. NN Benchmark (CIM MRMR §6.10)...")
    comp_df = compare_model_estimation_methods(
        samples_xy=samples_xyz,
        sample_grades=sample_grades,
        block_model=block_model,
        cutoff_grades=[0.0, 0.4, 0.6, 0.8, 1.0],
        grade_cols={
            "Nearest Neighbor": "nn_grade",
            "Ordinary Kriging": "estimated_grade",
            "IDW²": "idw_grade",
        },
        sample_weights=None,
        grade_unit="% Cu",
        tonnes_col="tonnes",
    )
    comp_disp = comp_df.copy()
    comp_disp["ore_tonnes"] = (comp_disp["ore_tonnes"] / 1e6).map(lambda x: f"{x:.2f} Mt")
    comp_disp["mean_grade"] = comp_disp["mean_grade"].map(lambda x: f"{x:.3f}%")
    comp_disp["contained_metal"] = (comp_disp["contained_metal"] / 1e3).map(lambda x: f"{x:.1f} kt")
    comp_disp["tonnes_diff_pct_vs_nn"] = comp_disp["tonnes_diff_pct_vs_nn"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
    )
    comp_disp["grade_diff_pct_vs_nn"] = comp_disp["grade_diff_pct_vs_nn"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
    )
    comp_disp["metal_diff_pct_vs_nn"] = comp_disp["metal_diff_pct_vs_nn"].map(
        lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"
    )
    print(comp_disp.to_string(index=False))

    print("\nGlobal Bias at Zero Cut-off (vs. Declustered Sample Mean):")
    for m_name, b_val in comp_df.attrs["global_bias_pct"].items():
        flag = "PASSED (within +/-5%)" if abs(b_val) <= 5.0 else "WARNING (>5%)"
        print(f"  • {m_name:20s}: {b_val:+.2f}% bias [{flag}]")

    # -------------------------------------------------------------------------
    # 4. Generate Grade-Tonnage Sensitivity Table
    # -------------------------------------------------------------------------
    print("\n[Step 4] Computing Grade-Tonnage Sensitivity Curve...")
    gt_df = grade_tonnage_table(
        block_model[np.isfinite(block_model["estimated_grade"])],
        grade_col="estimated_grade",
        tonnes_col="tonnes",
        cutoffs=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
    )
    print(
        gt_df[
            ["ore_tonnes", "ore_grade", "contained_metal", "ore_recovery_pct"]
        ].to_string()
    )

    # Discrete reporting cut-off intervals matching mining standards
    grade_bins = [0.0, 0.4, 0.8, 1.2, 1.6, 2.2, 3.5]

    # -------------------------------------------------------------------------
    # 4b. Convert Mineral Resources to Reserves (CIM / JORC Modifying Factors)
    # -------------------------------------------------------------------------
    print("\n[Step 4b] Converting Mineral Resources to Mineral Reserves (CIM / JORC Modifying Factors)...")
    cog_reserve = 0.40  # Economic cut-off grade (% Cu)
    reserve_df = convert_resource_to_reserve(
        resource_df=block_model,
        mining_dilution_pct=5.0,
        mining_recovery_pct=95.0,
        cutoff_grade=cog_reserve,
        dilution_grade=0.10,
        category_col="category",
        grade_col="estimated_grade",
        tonnes_col="tonnes",
    )
    reserve_df["status"] = reserve_df["reserve_category"]

    # Classify 100% of spatial blocks into regulatory reserve statuses:
    # 1. 'Proven Reserve' (Forest Green)
    # 2. 'Probable Reserve' (Royal Blue)
    # 3. 'Inferred Resource (Excluded)' (Purple / Magenta)
    # 4. 'Sub-Economic / Waste' (Light Grey)
    reserve_block_model = classify_reserve_status(
        block_model=block_model,
        cutoff_grade=cog_reserve,
        mining_dilution_pct=5.0,
        mining_recovery_pct=95.0,
        dilution_grade=0.10,
        category_col="category",
        grade_col="estimated_grade",
        tonnes_col="tonnes",
        status_col="status",
    )
    block_model["status"] = reserve_block_model["status"]

    res_stmt = format_reserve_statement(
        reserve_df,
        cutoff_grade=cog_reserve,
        mining_dilution_pct=5.0,
        mining_recovery_pct=95.0,
        commodity_price="$3.80/lb Cu",
        metallurgical_recovery=88.0,
        rpeee_constraint="Constrained within Lerchs-Grossmann optimized pit shell",
    )
    print("\n--- Official Mineral Reserve Statement (Run-of-Mine Delivered Feed) ---")
    print(res_stmt.to_string(index=False))

    # -------------------------------------------------------------------------
    # 5. Visualizations (Method 1 to 4 & Resource Map)
    # -------------------------------------------------------------------------
    if not args.no_plot:
        print("\n[Step 5] Generating Method 1: Orthogonal Slices...")
        fig1, _ = plot_block_model_orthogonal_slices(
            block_model=block_model,
            grade_col="estimated_grade",
            bench_z=47.5,  # Central mining bench
            section_y=205.0,  # Central E-W cross section
            section_x=205.0,  # Central N-S longitudinal section
            samples_xyz=samples_xyz,
            sample_grades=sample_grades,
            grade_bins=grade_bins,
            grade_unit="% Cu",
            title="Method 1: 3D Block Model Orthogonal Slices & Drillhole Reconciliation",
        )
        fig1_path = output_dir / "block_kriging_orthogonal_slices.png"
        fig1.savefig(fig1_path, dpi=200, bbox_inches="tight")
        plt.close(fig1)
        print(f"  -> Saved: {fig1_path}")

        print("\n[Step 6] Generating Method 2: Multi-Bench Depth Gallery...")
        bench_levels = [77.5, 62.5, 47.5, 32.5]  # Descending elevation benches
        fig2, _ = plot_block_model_bench_gallery(
            block_model=block_model,
            grade_col="estimated_grade",
            bench_elevations=bench_levels,
            n_cols=2,
            samples_xyz=samples_xyz,
            sample_grades=sample_grades,
            grade_bins=grade_bins,
            grade_unit="% Cu",
            title="Method 2: Multi-Bench Elevation Gallery (Z = 77.5m down to 32.5m)",
            figsize=(12.0, 10.0),
        )
        fig2_path = output_dir / "block_kriging_bench_gallery.png"
        fig2.savefig(fig2_path, dpi=200, bbox_inches="tight")
        plt.close(fig2)
        print(f"  -> Saved: {fig2_path}")

        print("\n[Step 7] Generating Method 3: 3D Isometric View...")
        fig3, _ = plot_block_model_3d_isometric(
            block_model=block_model,
            grade_col="estimated_grade",
            cutoff_grade=0.80,  # Filter out low-grade waste to reveal high-grade core
            samples_xyz=samples_xyz,
            sample_grades=sample_grades,
            grade_bins=grade_bins,
            grade_unit="% Cu",
            title="Method 3: 3D Isometric Mineralized Envelope (Cut-off ≥ 0.80% Cu)",
        )
        fig3_path = output_dir / "block_kriging_3d_isometric.png"
        fig3.savefig(fig3_path, dpi=200, bbox_inches="tight")
        plt.close(fig3)
        print(f"  -> Saved: {fig3_path}")

        print("\n[Step 8] Generating Method 4: Grade vs. Kriging Variance Audit...")
        fig4, _ = plot_block_model_grade_uncertainty(
            block_model=block_model,
            grade_col="estimated_grade",
            var_col="kriging_variance",
            slice_axis="z",
            slice_coord=47.5,
            samples_xyz=samples_xyz,
            sample_grades=sample_grades,
            grade_bins=grade_bins,
            vmax_var=total_sill,  # Anchors colorbar to theoretical point sill C(0) = 1.0
            vmin_var=0.0,
            grade_unit="% Cu",
            title="Method 4: Geostatistical Audit (Block Grade vs. Kriging Variance at Z=47.5m)",
        )
        fig4_path = output_dir / "block_kriging_grade_uncertainty.png"
        fig4.savefig(fig4_path, dpi=200, bbox_inches="tight")
        plt.close(fig4)
        print(f"  -> Saved: {fig4_path}")

        fig5, _ = plot_grade_tonnage_curve(
            gt_data=gt_df,
            grade_unit="% Cu",
            tonnage_unit="t",
            title="SMU Block Model Grade-Tonnage Curve",
        )
        fig5_path = output_dir / "block_kriging_grade_tonnage_curve.png"
        fig5.savefig(fig5_path, dpi=200, bbox_inches="tight")
        plt.close(fig5)
        print(f"  -> Saved: {fig5_path}")

        print(
            "\n[Step 8b] Generating Mineral Resource Classification Map (Central Bench Z=47.5m)..."
        )
        dh_df = pd.DataFrame(
            {"x": samples_xyz[:, 0], "y": samples_xyz[:, 1]}
        ).drop_duplicates()
        fig_res_map, _ = plot_resource_classification_map(
            block_model,
            drillholes=dh_df,
            bench_z=47.5,
            elevation_tolerance=2.5,
            title="3D Block Kriging: CIM / JORC Mineral Resource Classification (Bench Z = 47.5m)",
        )
        fig_res_path = output_dir / "block_kriging_resource_classification_map.png"
        fig_res_map.savefig(fig_res_path, dpi=200, bbox_inches="tight")
        plt.close(fig_res_map)
        print(f"  -> Saved: {fig_res_path}")

        print("\n[Step 8c] Generating 3-Axis Swath Local Bias Validation Suite...")
        fig_swath, _ = plot_swath_analysis_3axis(
            block_model=block_model,
            drillholes=df_dh,
            bin_widths=(20.0, 20.0, 10.0),
            grade_col="estimated_grade",
            validation_grade_col="nn_grade",
            drillhole_grade_col="grade",
            tonnes_col="tonnes",
            model_name="Ordinary Kriging",
            validation_model_name="Nearest Neighbor (Benchmark)",
            grade_unit="% Cu",
            tonnage_unit="Mt",
            title="3-Axis Swath Validation: Ordinary Kriging vs. Nearest Neighbor Benchmark",
        )
        fig_swath_path = output_dir / "block_kriging_swath_analysis_3axis.png"
        fig_swath.savefig(fig_swath_path, dpi=200, bbox_inches="tight")
        plt.close(fig_swath)
        print(f"  -> Saved: {fig_swath_path}")

        print("\n[Step 8d] Generating 3D Kriging Cross-Validation Diagnostics Dashboard...")
        fig_cv, _ = plot_cross_validation_diagnostics(
            cv_df,
            grade_unit="% Cu",
            title="3D Block Kriging: Cross-Validation Diagnostics (CIM MRMR §6.10, SME Ch. 4.5)",
        )
        fig_cv_path = output_dir / "block_kriging_cross_validation_diagnostics.png"
        fig_cv.savefig(fig_cv_path, dpi=200, bbox_inches="tight")
        plt.close(fig_cv)
        print(f"  -> Saved: {fig_cv_path}")

        print("\n[Step 8e] Generating 3-View Orthogonal Resource Classification Projections...")
        fig_ortho_class, _ = plot_resource_classification_orthogonal_slices(
            block_model=block_model,
            category_col="category",
            bench_z=47.5,
            section_y=205.0,
            section_x=205.0,
            samples_xyz=samples_xyz,
            title="CIM / JORC Mineral Resource Classification: Synchronized Orthogonal Projections",
        )
        fig_ortho_path = output_dir / "block_kriging_resource_orthogonal_slices.png"
        fig_ortho_class.savefig(fig_ortho_path, dpi=200, bbox_inches="tight")
        plt.close(fig_ortho_class)
        print(f"  -> Saved: {fig_ortho_path}")

        print("\n[Step 8e] Generating Multi-Bench Resource Classification Gallery...")
        fig_res_gal, _ = plot_resource_classification_bench_gallery(
            block_model=block_model,
            category_col="category",
            bench_elevations=bench_levels,
            n_cols=2,
            samples_xyz=samples_xyz,
            title="CIM / JORC Mineral Resource Classification: Multi-Bench Depth Gallery",
            figsize=(12.0, 10.0),
        )
        fig_res_gal_path = output_dir / "block_kriging_resource_bench_gallery.png"
        fig_res_gal.savefig(fig_res_gal_path, dpi=200, bbox_inches="tight")
        plt.close(fig_res_gal)
        print(f"  -> Saved: {fig_res_gal_path}")

        print("\n[Step 8f] Generating 3D Isometric Resource Classification Model...")
        fig_res_iso, _ = plot_resource_classification_3d_isometric(
            block_model=block_model,
            category_col="category",
            samples_xyz=samples_xyz,
            title="3D Isometric Mineral Resource Classification Model (CIM / JORC)",
        )
        fig_res_iso_path = output_dir / "block_kriging_resource_3d_isometric.png"
        fig_res_iso.savefig(fig_res_iso_path, dpi=200, bbox_inches="tight")
        plt.close(fig_res_iso)
        print(f"  -> Saved: {fig_res_iso_path}")

        print("\n[Step 8g] Generating Mineral Reserve Status Map (Central Bench Z=47.5m)...")
        fig_rev_map, _ = plot_reserve_classification_map(
            block_model=reserve_block_model,
            drillholes=dh_df,
            status_col="status",
            bench_z=47.5,
            elevation_tolerance=2.5,
            title="3D Block Model: Mineral Reserve Status (Bench Z = 47.5m)",
        )
        fig_rev_map_path = output_dir / "block_kriging_reserve_classification_map.png"
        fig_rev_map.savefig(fig_rev_map_path, dpi=200, bbox_inches="tight")
        plt.close(fig_rev_map)
        print(f"  -> Saved: {fig_rev_map_path}")

        print("\n[Step 8h] Generating 3-View Orthogonal Reserve Status Projections...")
        fig_rev_ortho, _ = plot_reserve_classification_orthogonal_slices(
            block_model=reserve_block_model,
            status_col="status",
            bench_z=47.5,
            section_y=205.0,
            section_x=205.0,
            samples_xyz=samples_xyz,
            title="Mineral Reserve Status: Synchronized Orthogonal Projections",
        )
        fig_rev_ortho_path = output_dir / "block_kriging_reserve_orthogonal_slices.png"
        fig_rev_ortho.savefig(fig_rev_ortho_path, dpi=200, bbox_inches="tight")
        plt.close(fig_rev_ortho)
        print(f"  -> Saved: {fig_rev_ortho_path}")

        print("\n[Step 8i] Generating Multi-Bench Mineral Reserve Status Gallery...")
        fig_rev_gal, _ = plot_reserve_classification_bench_gallery(
            block_model=reserve_block_model,
            status_col="status",
            bench_elevations=bench_levels,
            n_cols=2,
            samples_xyz=samples_xyz,
            title="Mineral Reserve Status: Multi-Bench Depth Gallery",
            figsize=(12.0, 10.0),
        )
        fig_rev_gal_path = output_dir / "block_kriging_reserve_bench_gallery.png"
        fig_rev_gal.savefig(fig_rev_gal_path, dpi=200, bbox_inches="tight")
        plt.close(fig_rev_gal)
        print(f"  -> Saved: {fig_rev_gal_path}")

        print("\n[Step 8j] Generating 3D Isometric Mineral Reserve Status Model...")
        fig_rev_iso, _ = plot_reserve_classification_3d_isometric(
            block_model=reserve_block_model,
            status_col="status",
            samples_xyz=samples_xyz,
            title="3D Isometric Mineral Reserve Status Model",
        )
        fig_rev_iso_path = output_dir / "block_kriging_reserve_3d_isometric.png"
        fig_rev_iso.savefig(fig_rev_iso_path, dpi=200, bbox_inches="tight")
        plt.close(fig_rev_iso)
        print(f"  -> Saved: {fig_rev_iso_path}")
    else:
        print("\n[Plotting skipped via --no-plot]")

    # -------------------------------------------------------------------------
    # 9. Optional Interactive 3D Explorers (Continuous Grade, Resource, Reserve)
    # -------------------------------------------------------------------------
    if args.interactive:
        print("\n[Step 9] Launching Interactive 3D Block Model Explorers...")
        active_view = getattr(args, "interactive_view", "all")
        interactive_figs = []

        if active_view in ("all", "grade"):
            print("  • Launching Continuous Grade 3D Explorer (Cut-off & Elevation Sliders)...")
            fig_grade, _, _ = plot_block_model_3d_interactive(
                block_model=block_model,
                grade_col="estimated_grade",
                initial_cutoff=0.80,
                samples_xyz=samples_xyz,
                sample_grades=sample_grades,
                grade_bins=grade_bins,
                grade_unit="% Cu",
                title="Interactive 3D Block Model Explorer (Continuous Grade Cu %)",
            )
            interactive_figs.append(fig_grade)

        if active_view in ("all", "resource"):
            print("  • Launching Mineral Resource Classification 3D Explorer (Category Checkboxes & Bench Slider)...")
            fig_res, _, _ = plot_resource_classification_3d_interactive(
                block_model=block_model,
                category_col="category",
                samples_xyz=samples_xyz,
                title="Interactive 3D Mineral Resource Classification Explorer (CIM / JORC)",
            )
            interactive_figs.append(fig_res)

        if active_view in ("all", "reserve"):
            print("  • Launching Mineral Reserve Status 3D Explorer (Reserve Checkboxes & Bench Slider)...")
            fig_rev, _, _ = plot_reserve_classification_3d_interactive(
                block_model=reserve_block_model,
                status_col="status",
                samples_xyz=samples_xyz,
                title="Interactive 3D Mineral Reserve Status Explorer (Modifying Factors Applied)",
            )
            interactive_figs.append(fig_rev)

        print("\n  • Click & drag 3D axes to rotate, scroll/right-click to zoom.")
        print("  • Use UI controls (sliders, checkboxes, reset buttons) to dynamically explore block subsets.")
        plt.show()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

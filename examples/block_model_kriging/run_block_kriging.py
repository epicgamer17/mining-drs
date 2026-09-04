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
    create_block_model,
    ordinary_kriging_block_estimation,
    kriging_quality_metrics,
    classify_resources_by_sor,
    classify_mineral_resources,
    plot_resource_classification_map,
    plot_block_model_orthogonal_slices,
    plot_block_model_bench_gallery,
    plot_block_model_3d_isometric,
    plot_block_model_3d_interactive,
    plot_block_model_grade_uncertainty,
    grade_tonnage_table,
    plot_grade_tonnage_curve,
)


def generate_3d_porphyry_drillholes(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Generates synthetic 3D exploration drillhole composites for a copper-gold deposit.

    Creates 20 drillholes on an irregular 60m-80m grid with 5m down-hole composites
    penetrating a dipping high-grade porphyry core and surrounding low-grade halo.
    """
    rng = np.random.default_rng(seed)

    # Drillhole collars across 400m x 400m exploration area
    collar_coords = [
        (80.0, 80.0), (160.0, 70.0), (240.0, 90.0), (320.0, 80.0),
        (70.0, 150.0), (150.0, 160.0), (230.0, 150.0), (310.0, 170.0),
        (90.0, 230.0), (170.0, 240.0), (250.0, 230.0), (330.0, 250.0),
        (80.0, 310.0), (160.0, 320.0), (240.0, 310.0), (320.0, 330.0),
        # Infill holes in high-grade central area
        (190.0, 190.0), (210.0, 210.0), (180.0, 220.0), (220.0, 180.0),
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
            base_grade = 0.20 + 2.0 * np.exp(-0.5 * (dist_core ** 2))
            # Geostatistical log-normal assay noise
            noise = rng.lognormal(mean=0.0, sigma=0.22)
            grade = float(np.clip(base_grade * noise, 0.05, 4.50))

            records.append({
                "hole_id": hole_id,
                "x": cx,
                "y": cy,
                "z": z,
                "grade": round(grade, 3),
            })

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
    print(f"  • Total composite assays: {len(df_dh)} from {df_dh['hole_id'].nunique()} drillholes")
    print(f"  • Elevation range: Z = {df_dh['z'].min():.1f}m to {df_dh['z'].max():.1f}m")
    print(f"  • Raw composite grade: mean={sample_grades.mean():.3f}%, std={sample_grades.std():.3f}%, max={sample_grades.max():.3f}%")

    # -------------------------------------------------------------------------
    # 2. Construct 3D Block Model Grid (SMUs)
    # -------------------------------------------------------------------------
    print("\n[Step 2] Constructing 3D Block Model Selective Mining Units (SMUs)...")
    origin = (20.0, 20.0, 10.0)
    block_size = (10.0, 10.0, 5.0)  # dx=10m, dy=10m, dz=5m bench height
    n_blocks = (36, 36, 18)         # 36 x 36 x 18 = 23,328 blocks spanning 360m x 360m x 90m

    block_model = create_block_model(
        origin=origin,
        block_size=block_size,
        n_blocks=n_blocks,
        default_density=2.70,
        default_domain="Porphyry",
    )
    total_tonnes = block_model["tonnes"].sum() / 1e6
    print(f"  • Block dimensions: {block_size[0]}m (E) x {block_size[1]}m (N) x {block_size[2]}m (bench)")
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

    estimates, variances, dispersion_var, lagrange_multipliers = ordinary_kriging_block_estimation(
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
    )

    block_model["estimated_grade"] = estimates
    block_model["kriging_variance"] = variances

    # Kriging Neighborhood Analysis (KNA) Quality Metrics
    kriging_eff, slope_regr = kriging_quality_metrics(
        kriging_variances=variances,
        block_dispersion_variance=dispersion_var,
        lagrange_multipliers=lagrange_multipliers,
    )
    block_model["kriging_efficiency"] = kriging_eff
    block_model["slope_of_regression"] = slope_regr

    # Support Effect Analysis
    bv = dispersion_var  # Block Dispersion Variance BV = C_bar(V, V) = sigma^2(V|D)
    within_block_var = total_sill - bv  # Within-block variance gamma_bar(V, V) = sigma^2(v|V)
    variance_reduction = within_block_var / total_sill
    print(f"  • Theoretical Point Sill C(0): {total_sill:.4f}")
    print(f"  • Block Dispersion Variance BV = C_bar(V, V): {bv:.4f}")
    print(f"  • Within-Block Variance sigma^2(v|V): {within_block_var:.4f}")
    print(f"  • Block Support Variance Retention: {(bv / total_sill) * 100:.1f}%")
    print(f"  • Variance Reduction from Point to Block: {variance_reduction * 100:.1f}%")
    print(f"  • Estimated blocks: {np.sum(np.isfinite(estimates)):,} / {len(block_model):,} ({np.mean(np.isfinite(estimates))*100:.1f}%)")

    # JORC / CIM Quality Diagnostics Summary
    valid_blocks = np.isfinite(kriging_eff) & np.isfinite(slope_regr)
    mean_ke = float(np.mean(kriging_eff[valid_blocks]))
    mean_sor = float(np.mean(slope_regr[valid_blocks]))
    sor_categories = classify_resources_by_sor(
        slopes_of_regression=slope_regr,
        kriging_efficiencies=kriging_eff,
        threshold_measured=0.80,
        threshold_indicated=0.50,
        max_slope_measured=1.05,
        min_kriging_efficiency=0.0,
    )
    pct_measured = float(np.mean(sor_categories[valid_blocks] == "Measured") * 100.0)
    pct_indicated = float(np.mean(sor_categories[valid_blocks] == "Indicated") * 100.0)
    print(f"  • Mean Kriging Efficiency (KE): {mean_ke * 100:.1f}%")
    print(f"  • Mean Slope of Regression (SoR): {mean_sor:.3f}")
    print(f"  • High Confidence Blocks (0.80 ≤ SoR ≤ 1.05, KE > 0, Measured candidate): {pct_measured:.1f}%")
    print(f"  • Moderate Confidence Blocks (0.50 ≤ SoR < 0.80, KE > 0, Indicated candidate): {pct_indicated:.1f}%")

    # Multi-criteria Resource Classification (Spacing + SoR + KE)
    resource_categories = classify_mineral_resources(
        grid_points=block_model[["x", "y", "z"]].values,
        samples_xy=samples_xyz,
        max_radius_measured=45.0,
        max_radius_indicated=90.0,
        min_holes_measured=3,
        min_holes_indicated=2,
        slopes_of_regression=slope_regr,
        kriging_efficiencies=kriging_eff,
        sor_threshold_measured=0.80,
        sor_threshold_indicated=0.50,
        max_slope_measured=1.05,
        min_kriging_efficiency=0.0,
    )
    block_model["category"] = resource_categories

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
    print(gt_df[["ore_tonnes", "ore_grade", "contained_metal", "ore_recovery_pct"]].to_string())

    # Discrete reporting cut-off intervals matching mining standards
    grade_bins = [0.0, 0.4, 0.8, 1.2, 1.6, 2.2, 3.5]

    # -------------------------------------------------------------------------
    # 5. Method 1: Orthogonal Slices (Bench Plan, Cross-Section, Long-Section)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Method 1: Orthogonal Slices...")
    fig1, _ = plot_block_model_orthogonal_slices(
        block_model=block_model,
        grade_col="estimated_grade",
        bench_z=47.5,      # Central mining bench
        section_y=205.0,    # Central E-W cross section
        section_x=205.0,    # Central N-S longitudinal section
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

    # -------------------------------------------------------------------------
    # 6. Method 2: Multi-Bench Elevation Gallery (Depth Slices)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 7. Method 3: 3D Isometric View (Cut-off Thresholded Ore Envelope)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 8. Method 4: Dual Grade vs. Estimation Uncertainty Audit
    # -------------------------------------------------------------------------
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

    # Grade-Tonnage Curve Plot
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

    # Method 5: Mineral Resource Classification Map (Central Bench)
    print("\n[Step 8b] Generating Mineral Resource Classification Map (Central Bench)...")
    bench_blocks = block_model[block_model["z"] == 47.5].copy()
    dh_df = pd.DataFrame({"x": samples_xyz[:, 0], "y": samples_xyz[:, 1]}).drop_duplicates()
    fig_res_map, _ = plot_resource_classification_map(
        bench_blocks,
        drillholes=dh_df,
        title="3D Block Kriging: CIM / JORC Mineral Resource Classification (Bench Z = 47.5m)",
    )
    fig_res_path = output_dir / "block_kriging_resource_classification_map.png"
    fig_res_map.savefig(fig_res_path, dpi=200, bbox_inches="tight")
    plt.close(fig_res_map)
    print(f"  -> Saved: {fig_res_path}")

    # -------------------------------------------------------------------------
    # 9. Optional Interactive 3D Explorer
    # -------------------------------------------------------------------------
    if args.interactive:
        print("\n[Step 9] Launching Interactive 3D Block Model Explorer...")
        print("  • Drag 'Cut-off Grade' slider to dynamically peel away waste blocks.")
        print("  • Drag 'Max Elevation' slider to slice downward through benches.")
        print("  • Click & drag 3D axes to rotate, scroll/right-click to zoom.")
        fig_int, ax_int, controls = plot_block_model_3d_interactive(
            block_model=block_model,
            grade_col="estimated_grade",
            initial_cutoff=0.80,
            samples_xyz=samples_xyz,
            sample_grades=sample_grades,
            grade_bins=grade_bins,
            grade_unit="% Cu",
            title="Interactive 3D Block Model Explorer (Porphyry Cu Deposit)",
        )
        plt.show()

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

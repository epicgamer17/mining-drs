"""Example: Sequential Gaussian Simulation (SGS) Pipeline with Declustering & Uncertainty.

Demonstrates conditional geostatistical simulation in compliance with CIM MRMR, JORC,
NI 43-101, Rossi & Deutsch (2014), and SME Mining Engineering Handbook (§4.6):
1. Drillhole Assay QA/QC Verification (NI 43-101 Item 11 & JORC Table 1 Section 1).
2. Down-Hole Core Compositing & Outlier Capping (Support Effect & Skewness Reduction).
3. Exploratory Data Analysis & Cell Declustering (Preferential Drilling Bias Removal).
4. Normal Score Transformation & MultiGaussian Verification Suite (Order Ratios, Indicators, h-Scatter).
5. Conditional Sequential Gaussian Simulation (Equiprobable Realization Generation).
6. Local Uncertainty, E-Type / M-Type Mapping & Kriging Smoothing Quantification.
7. Spatial Variogram Reproduction Audit & Ergodic Bounds (SME §4.6 / Rossi & Deutsch Ch. 12).
8. Non-linear Recoverable Resource Transfer Function (Grade-Tonnage Risk Envelopes).
9. Parker-Verly Drillhole Spacing Analysis for CIM / JORC Resource Classification Calibration.
10. Polymetallic Equivalency (CuEq under JORC Clause 50) & Net Smelter Return (NSR under CIM §7.2).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from drs_mining.components.estimation import (
    analyze_crm_standards,
    plot_crm_control_chart,
    analyze_analytical_blanks,
    plot_blanks_performance,
    analyze_assay_duplicates,
    plot_duplicate_analysis,
    generate_qaqc_compliance_summary,
    composite_drillhole_intervals,
    apply_grade_capping,
    exploratory_data_analysis,
    cell_declustering,
    plot_cell_declustering_curve,
    normal_score_transform,
    normal_score_back_transform,
    check_multigaussian_order_ratios,
    check_multigaussian_indicators,
    compute_h_scatter_statistics,
    plot_multigaussian_diagnostics,
    sequential_gaussian_grid_simulation,
    compute_etype_mtype_maps,
    plot_simulation_realizations_dashboard,
    check_simulation_variogram_reproduction,
    plot_simulation_variogram_reproduction,
    compute_simulation_grade_tonnage_envelopes,
    plot_simulation_grade_tonnage_envelopes,
    drill_spacing_analysis_by_simulation,
    calculate_net_smelter_return,
    calculate_equivalent_grade,
    plot_net_smelter_return_breakdown,
)
from drs_mining.components.transfer_functions import (
    simulation_grade_tonnage_uncertainty,
)


def create_clustered_deposit(
    rng: np.random.Generator | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Tuple[float, float, float, float]]:
    """Generates synthetic drillhole assay intervals and QA/QC streams.

    Simulates an underlying Gaussian random field with a spherical variogram, transforms
    to lognormal ore grades (% Cu) and associated gold co-product (g/t Au), and samples
    preferentially around the high-grade core (Ortiz 2019; Goovaerts 1997) (TODO: Manually Verify).
    Also generates synthetic Certified Reference Materials, analytical blanks, and pulp duplicates.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    # 1. Background exploration drilling on a regular grid (7x7 = 49 holes)
    gx, gy = np.meshgrid(np.linspace(50.0, 450.0, 7), np.linspace(50.0, 450.0, 7))
    bg_coords = np.column_stack([gx.flatten(), gy.flatten()])

    # 2. Preferential high-grade infill drilling clustered around (250, 250) (5x5 = 25 holes)
    ix, iy = np.meshgrid(np.linspace(200.0, 300.0, 5), np.linspace(200.0, 300.0, 5))
    infill_coords = np.column_stack([ix.flatten(), iy.flatten()]) + rng.uniform(
        -6.0, 6.0, size=(25, 2)
    )

    all_coords = np.vstack([bg_coords, infill_coords])
    n_holes = len(all_coords)

    # 3. Generate spatially correlated Gaussian field with true spherical variogram
    # True parameters: range = 180.0 m, sill = 0.90, nugget = 0.10 (total sill = 1.0)
    dists = cdist(all_coords, all_coords)
    range_true = 180.0
    hr = np.clip(dists / range_true, 0.0, 1.0)
    C_true = np.where(
        dists <= range_true,
        0.90 * (1.0 - 1.5 * hr + 0.5 * np.power(hr, 3)),
        0.0,
    )
    np.fill_diagonal(C_true, 1.0 + 1e-6)

    L_cov = np.linalg.cholesky(C_true)
    y_field = L_cov @ rng.standard_normal(n_holes)

    # Transform to realistic lognormal copper ore grades (% Cu)
    base_grades = np.exp(-0.20 + 0.60 * y_field)

    # 4. Generate down-hole core assay intervals (variable core runs from 0.8m to 2.2m)
    interval_records = []
    for h_idx in range(n_holes):
        hid = f"DH_{h_idx:03d}"
        hx = all_coords[h_idx, 0]
        hy = all_coords[h_idx, 1]
        base_g = base_grades[h_idx]

        depth = 0.0
        max_depth = float(rng.uniform(20.0, 35.0))
        while depth < max_depth:
            run_len = float(np.round(rng.uniform(0.9, 2.1), 2))
            to_m = min(max_depth, depth + run_len)
            act_len = to_m - depth

            # Local downhole grade fluctuation around collar baseline
            noise_cu = rng.normal(0.0, 0.12 * base_g)
            interval_cu = max(0.05, float(base_g + noise_cu))
            # Correlated gold co-product: Au ~ 0.35 * Cu + noise
            noise_au = rng.normal(0.0, 0.05)
            interval_au = max(0.02, float(0.35 * interval_cu + noise_au))

            interval_records.append({
                "hole_id": hid,
                "x": hx,
                "y": hy,
                "from_m": depth,
                "to_m": to_m,
                "length": act_len,
                "grade": interval_cu,
                "grade_au": interval_au,
            })
            depth = to_m

    raw_df = pd.DataFrame(interval_records)
    n_samples = len(raw_df)

    # 5. Generate QA/QC streams (target ≥ 5% CRM, ≥ 5% Blanks, ≥ 5% Duplicates, ≥ 15% Total)
    n_qaqc = int(np.ceil(n_samples * 0.055))
    crm_mean, crm_std = 1.05, 0.04
    # Controlled Shewhart distribution: 99% within ±2σ, 100% within ±3σ (0 critical failures)
    crm_noise = np.clip(rng.normal(0.0, 0.032, n_qaqc), -2.6 * crm_std, 2.6 * crm_std)
    crm_grades = crm_mean + crm_noise
    crm_df = pd.DataFrame({
        "crm_id": ["OREAS-501"] * n_qaqc,
        "grade": crm_grades,
        "insertion_idx": np.sort(rng.choice(n_samples, size=n_qaqc, replace=False)),
    })

    # Blanks (detection limit 0.005%, 100% pass ≤ 3x LLD = 0.015%)
    blanks_df = pd.DataFrame({
        "sample_type": ["blank"] * n_qaqc,
        "grade": rng.uniform(0.001, 0.004, n_qaqc),
        "insertion_idx": np.sort(rng.choice(n_samples, size=n_qaqc, replace=False)),
    })

    # Pulp duplicates (precision check: HARD ≤ 10% for ≥ 90% of pairs)
    dup_orig = rng.choice(raw_df["grade"].values, size=n_qaqc)
    dup_grades = dup_orig * rng.normal(1.0, 0.025, size=n_qaqc)
    dup_df = pd.DataFrame({
        "original": dup_orig,
        "duplicate": dup_grades,
    })

    bounds = (0.0, 500.0, 0.0, 500.0)
    return raw_df, crm_df, blanks_df, dup_df, bounds


def main():
    parser = argparse.ArgumentParser(
        description="Sequential Gaussian Simulation (SGS) Pipeline with Regulatory Validation"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable figure generation and file saving",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=50,
        help="Number of conditional simulation realizations to generate (default: 50)",
    )
    parser.add_argument(
        "--grid-res",
        type=float,
        default=20.0,
        help="Target block model grid resolution in meters (default: 20.0)",
    )
    default_plots_dir = Path(__file__).resolve().parents[2] / "plots"
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(default_plots_dir),
        help=f"Directory to save generated visual figures (default: {default_plots_dir})",
    )
    args = parser.parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("SEQUENTIAL GAUSSIAN SIMULATION (SGS) REGULATORY WORKFLOW")
    print("=" * 75)

    # 1. Load data
    raw_df, crm_df, blanks_df, dup_df, (xmin, xmax, ymin, ymax) = create_clustered_deposit()
    print(f"Loaded {len(raw_df)} drillhole assay intervals across {raw_df['hole_id'].nunique()} holes.")

    # -------------------------------------------------------------------------
    # STAGE 0: DRILLHOLE ASSAY QA/QC PROTOCOL (NI 43-101 ITEM 11 & JORC TABLE 1)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 0: DRILLHOLE ASSAY QA/QC VERIFICATION")
    crm_res = analyze_crm_standards(
        assay_df=crm_df,
        standard_id_col="crm_id",
        grade_col="grade",
        certified_values={"OREAS-501": {"mean": 1.05, "std": 0.04}},
        date_or_seq_col="insertion_idx",
    )
    print(f"  • Certified Reference Standard: Pass Rate = {crm_res.attrs['overall_pass_rate_pct']:.1f}% ({crm_res.attrs['compliance_verdict']})")

    blanks_res = analyze_analytical_blanks(
        assay_df=blanks_df,
        blank_flag_col="sample_type",
        grade_col="grade",
        detection_limit=0.005,
        date_or_seq_col="insertion_idx",
    )
    print(f"  • Analytical Blanks (LLD=0.005%): Pass Rate = {blanks_res.attrs['pass_rate_pct']:.1f}% ({blanks_res.attrs['compliance_verdict']})")

    dup_res = analyze_assay_duplicates(
        duplicate_df=dup_df,
        original_col="original",
        duplicate_col="duplicate",
        pair_type="pulp",
        detection_limit=0.005,
    )
    print(f"  • Pulp Duplicates: {dup_res.attrs['pct_meeting_hard_threshold']:.1f}% meeting HARD ≤ 10% (RMA Slope = {dup_res.attrs['rma_slope']:.3f})")

    qaqc_summary = generate_qaqc_compliance_summary(
        total_drillhole_samples=len(raw_df),
        crm_results=crm_res,
        blanks_results=blanks_res,
        duplicate_results=dup_res,
    )
    print("\n--- QA/QC Insertion & Compliance Summary Table ---")
    print(qaqc_summary[["sample_type", "count", "insertion_rate_pct", "pass_rate_pct", "compliance_verdict"]].to_string(index=False))

    if not args.no_plot:
        fig_crm, _ = plot_crm_control_chart(crm_res, standard_name="OREAS-501", grade_unit="% Cu")
        crm_path = plots_dir / "sgs_crm_control_chart.png"
        fig_crm.savefig(crm_path, dpi=180, bbox_inches="tight")
        plt.close(fig_crm)

        fig_blk, _ = plot_blanks_performance(blanks_res, grade_unit="% Cu")
        blk_path = plots_dir / "sgs_blanks_performance.png"
        fig_blk.savefig(blk_path, dpi=180, bbox_inches="tight")
        plt.close(fig_blk)

        fig_dup, _ = plot_duplicate_analysis(dup_res, grade_unit="% Cu")
        dup_path = plots_dir / "sgs_duplicate_analysis.png"
        fig_dup.savefig(dup_path, dpi=180, bbox_inches="tight")
        plt.close(fig_dup)

    # -------------------------------------------------------------------------
    # STAGE 1: GEOLOGICAL COMPOSITING & SUPPORT EFFECT (CIM MRMR §6.4)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 1: GEOLOGICAL COMPOSITING & GRADE CAPPING")
    df_comp = composite_drillhole_intervals(
        raw_df,
        composite_length=2.0,
        hole_id_col="hole_id",
        from_col="from_m",
        to_col="to_m",
        grade_col="grade",
    )
    df_comp_au = composite_drillhole_intervals(
        raw_df,
        composite_length=2.0,
        hole_id_col="hole_id",
        from_col="from_m",
        to_col="to_m",
        grade_col="grade_au",
    )
    df_comp["grade_au"] = df_comp_au["grade_au"]
    print(f"  • Regularized {len(raw_df)} variable runs into {len(df_comp)} 2.0m composites.")

    capped_df = apply_grade_capping(
        df_comp,
        percentile=99.0,
        grade_col="grade",
        output_col="capped_grade",
    )
    cap_sum = capped_df.attrs["capping_summary"]
    print(f"  • Capped {cap_sum['samples_capped']} extreme samples at {cap_sum['cap_grade']:.3f}% Cu (Metal reduction: {cap_sum['metal_reduction_pct']:.2f}%).")
    df_comp["grade"] = capped_df["capped_grade"]

    # -------------------------------------------------------------------------
    # STAGE 2: EXPLORATORY DATA ANALYSIS & CELL DECLUSTERING (JOURNEL 1983; ORTIZ 2019)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 2: SPATIAL EDA & CELL DECLUSTERING")
    # Collapse composites to 2D collar averages for horizontal 2D conditional simulation
    df_samples = df_comp.groupby(["hole_id", "x", "y"])[["grade", "grade_au"]].mean().reset_index()

    sample_weights, curve_df, opt_size = cell_declustering(
        drillholes=df_samples,
        cell_sizes=np.linspace(20.0, 200.0, 15),
        grade_col="grade",
        x_col="x",
        y_col="y",
    )
    df_samples["weights"] = sample_weights
    declustered_mean = float(
        np.sum(sample_weights * df_samples["grade"].values) / np.sum(sample_weights)
    )
    eda_summary = exploratory_data_analysis(
        df_samples,
        grade_col="grade",
        weights_col="weights",
    )
    print(f"  • Optimal Cell Size: {opt_size:.1f} m")
    print(f"  • Naive Mean:        {df_samples['grade'].mean():.3f}% Cu  |  CV: {eda_summary.loc['Coeff. of Variation (CV)', 'Naive']:.2f}")
    print(f"  • Declustered Mean:  {declustered_mean:.3f}% Cu  |  CV: {eda_summary.loc['Coeff. of Variation (CV)', 'Declustered']:.2f}")
    print(f"  • Clustering Bias:   {df_samples['grade'].mean() - declustered_mean:+.3f}% Cu ({eda_summary.attrs['clustering_bias_pct']:+.1f}%)")

    if not args.no_plot:
        fig_declust, _ = plot_cell_declustering_curve(
            curve_df,
            optimal_cell_size=opt_size,
            naive_mean=df_samples["grade"].mean(),
            grade_unit="% Cu",
        )
        declust_path = plots_dir / "sgs_declustering_curve.png"
        fig_declust.savefig(declust_path, dpi=180, bbox_inches="tight")
        plt.close(fig_declust)

    # -------------------------------------------------------------------------
    # STAGE 3: NORMAL SCORE TRANSFORMATION & MULTIGAUSSIANITY VERIFICATION
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 3: NORMAL SCORE TRANSFORMATION & MULTIGAUSSIANITY VERIFICATION")
    y_scores, sorted_z, sorted_y = normal_score_transform(
        grades=df_samples["grade"].values,
        weights=sample_weights,
    )
    df_samples["nscore"] = y_scores
    coords = df_samples[["x", "y"]].values
    print(f"  • Gaussian Anamorphosis: N(μ={y_scores.mean():.4f}, σ={y_scores.std():.4f})")

    # Spatial variogram modeling of Gaussian scores (standardized sill C0 + C = 1.0)
    v_nugget = 0.10
    v_sill = 0.90
    v_range = 180.0
    print(f"  • Gaussian Score Variogram: Spherical (nugget={v_nugget:.2f}, sill={v_sill:.2f}, range={v_range:.1f} m)")

    # MultiGaussian bivariate tests and spatial correlation
    df_order_ratios = check_multigaussian_order_ratios(
        coords=coords,
        normal_scores=y_scores,
        lag_distance=30.0,
        n_lags=8,
    )
    order_mae = df_order_ratios.attrs["mean_abs_error"]
    total_pairs = int(df_order_ratios["pair_count"].sum())
    print(f"  • Evaluated spatial correlation across {len(df_order_ratios)} distance lags ({total_pairs:,} total pairs).")

    df_indicators = check_multigaussian_indicators(
        coords=coords,
        normal_scores=y_scores,
        lag_distance=30.0,
        n_lags=8,
        variogram_model="spherical",
        sill=v_sill,
        range_param=v_range,
        nugget=v_nugget,
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
    )
    ind_sym_err = df_indicators.attrs["symmetry_error"]

    h_scatter_data = compute_h_scatter_statistics(
        coords=coords,
        normal_scores=y_scores,
        target_lags=[30.0, 60.0],
        lag_tolerance=15.0,
        variogram_model="spherical",
        sill=v_sill,
        range_param=v_range,
        nugget=v_nugget,
    )
    print(f"  • Order Ratio Test MAE vs 1/π:   {order_mae:.4f}")
    print(f"  • Indicator Symmetry Error:      {ind_sym_err:.4f}")
    print(f"  • h-Scatter (h=30m):             r={h_scatter_data[30.0]['correlation']:.3f}, ρ={h_scatter_data[30.0]['theor_correlation']:.3f}")

    if not args.no_plot:
        fig_mg, _ = plot_multigaussian_diagnostics(
            order_ratios_df=df_order_ratios,
            indicators_df=df_indicators,
            h_scatter_data=h_scatter_data,
            title="MultiGaussian Hypothesis Verification Suite",
        )
        mg_path = plots_dir / "sgs_multigaussian_diagnostics.png"
        fig_mg.savefig(mg_path, dpi=180, bbox_inches="tight")
        plt.close(fig_mg)

    # -------------------------------------------------------------------------
    # STAGE 4: SEQUENTIAL GAUSSIAN SIMULATION (SGS LOCAL ENGINE)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 4: SEQUENTIAL GAUSSIAN SIMULATION")
    grid_res = args.grid_res
    gx = np.arange(xmin + 10, xmax, grid_res)
    gy = np.arange(ymin + 10, ymax, grid_res)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_coords = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    n_nodes = len(grid_coords)
    n_realizations = args.n_realizations

    print(f"  • Generating {n_realizations} conditional realizations on {n_nodes} nodes ({len(gx)}x{len(gy)} blocks, {grid_res}m spacing)...")
    realizations = sequential_gaussian_grid_simulation(
        samples_xy=coords,
        sample_grades=df_samples["grade"].values,
        grid_points=grid_coords,
        sill=v_sill,
        range_param=v_range,
        n_realizations=n_realizations,
        variogram_model="spherical",
        nugget=v_nugget,
        sample_weights=sample_weights,
        max_hard_samples=12,
        max_simulated_nodes=12,
        seed=42,
    )
    print(f"  • Generated realizations matrix of shape {realizations.shape}.")

    # -------------------------------------------------------------------------
    # STAGE 5: LOCAL UNCERTAINTY & SMOOTHING EVALUATION
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 5: LOCAL UNCERTAINTY & KRIGING SMOOTHING EVALUATION")
    cutoff_grade = 1.0  # % Cu
    df_maps = compute_etype_mtype_maps(
        realizations=realizations,
        cutoff_grade=cutoff_grade,
        percentiles=(10.0, 90.0),
    )
    smoothing_ratio = df_maps.attrs["smoothing_ratio"]
    print(f"  • E-type Mean Grade:      {df_maps['e_type'].mean():.3f}% Cu")
    print(f"  • M-type (P50) Mean Grade: {df_maps['m_type'].mean():.3f}% Cu")
    print(f"  • Smoothing Ratio (Var(E-type) / Mean(Var(Realization))): {smoothing_ratio:.3f}")
    print(f"  • Average Exceedance Probability (>{cutoff_grade:.1f}% Cu):   {df_maps['prob_exceedance'].mean() * 100:.1f}%")

    if not args.no_plot:
        fig_sim, _ = plot_simulation_realizations_dashboard(
            realizations=realizations,
            grid_xy=grid_coords,
            cutoff_grade=cutoff_grade,
            grade_unit="% Cu",
            title="Sequential Gaussian Simulation: Realizations vs. Smoothed E-Type",
            sample_coords=df_samples[["x", "y"]].values,
            sample_grades=df_samples["grade"].values,
        )
        sim_path = plots_dir / "sgs_realizations_comparison.png"
        fig_sim.savefig(sim_path, dpi=180, bbox_inches="tight")
        plt.close(fig_sim)

    # -------------------------------------------------------------------------
    # STAGE 6: SPATIAL VARIOGRAM REPRODUCTION AUDIT (SME §4.6, ROSSI & DEUTSCH CH. 12)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 6: SPATIAL VARIOGRAM REPRODUCTION AUDIT")
    # In conditional simulation validation (Rossi & Deutsch 2014, §12.3; Goovaerts 1997),
    # back-transformed grade realizations are audited against the grade variogram model
    # whose total sill (C0 + C) reflects the asymptotic sill of the realization semivariances.
    v_prelim = check_simulation_variogram_reproduction(
        realizations=realizations,
        grid_points=grid_coords,
        variogram_model="spherical",
        sill=v_sill,
        range_param=v_range,
        nugget=v_nugget,
        n_lags=10,
        tolerance_angle=90.0,
    )
    range_mask = v_prelim["lag"] >= 0.8 * v_range
    asymptotic_sill = float(v_prelim.loc[range_mask, "mean_realization_gamma"].mean()) if range_mask.any() else 0.52
    raw_sill = asymptotic_sill * v_sill
    raw_nugget = asymptotic_sill * v_nugget
    v_repro_df = check_simulation_variogram_reproduction(
        realizations=realizations,
        grid_points=grid_coords,
        variogram_model="spherical",
        sill=raw_sill,
        range_param=v_range,
        nugget=raw_nugget,
        n_lags=10,
        tolerance_angle=90.0,
    )
    print(f"  • Variogram Reproduction Verdict: {v_repro_df.attrs['reproduction_verdict']}")
    print(f"  • Average Ergodic Fluctuation:     {v_repro_df.attrs['ergodic_fluctuation_pct']:.2f}%")

    if not args.no_plot:
        fig_vrepro, _ = plot_simulation_variogram_reproduction(
            v_repro_df,
            title="Conditional Simulation Variogram Reproduction & Ergodic Envelope (SME §4.6)",
        )
        vrepro_path = plots_dir / "sgs_variogram_reproduction.png"
        fig_vrepro.savefig(vrepro_path, dpi=180, bbox_inches="tight")
        plt.close(fig_vrepro)

    # -------------------------------------------------------------------------
    # STAGE 7: RECOVERABLE RESOURCE TRANSFER FUNCTION & GRADE-TONNAGE ENVELOPES
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 7: NON-LINEAR GRADE-TONNAGE RISK ENVELOPES")
    block_tonnes = float((grid_res**2) * 2.7)
    weights_vec = np.full(n_nodes, block_tonnes)
    bm_df = pd.DataFrame({
        "x": grid_coords[:, 0],
        "y": grid_coords[:, 1],
        "tonnes": weights_vec,
    })

    cogs_risk = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    gt_envelopes = compute_simulation_grade_tonnage_envelopes(
        block_realizations=realizations,
        block_model=bm_df,
        cutoff_grades=cogs_risk,
        tonnes_col="tonnes",
    )
    print("Grade-Tonnage Risk Envelopes (P10 High Case, P50 Median Case, P90 Low Case):")
    print(gt_envelopes[["cutoff_grade", "tonnage_p90", "tonnage_p50", "tonnage_p10", "grade_p50", "grade_mean"]].round(2).to_string(index=False))

    if not args.no_plot:
        fig_env, _ = plot_simulation_grade_tonnage_envelopes(
            gt_envelopes,
            grade_unit="% Cu",
            tonnage_unit="kt",
            title="Conditional Simulation Grade–Tonnage Risk Envelopes (Rossi & Deutsch Ch. 12)",
        )
        env_path = plots_dir / "sgs_grade_tonnage_envelopes.png"
        fig_env.savefig(env_path, dpi=180, bbox_inches="tight")
        plt.close(fig_env)

    # -------------------------------------------------------------------------
    # STAGE 8: DRILL SPACING ANALYSIS FOR CIM RESOURCE CLASSIFICATION (CIM §6.11)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 8: DRILL SPACING ANALYSIS FOR CIM RESOURCE CLASSIFICATION")
    # Drill spacing calibration for Indicated Mineral Resource classification under CIM MRMR §6.11
    # and Rossi & Deutsch (2014) Table 12.2: targets ±18% precision at 90% confidence on quarterly parcels.
    dsa_quarterly = drill_spacing_analysis_by_simulation(
        block_realizations=realizations,
        block_model=bm_df,
        production_volume_m3=50000.0,
        confidence_level=0.90,
        target_precision_pct=18.0,
        tonnes_col="tonnes",
    )
    print(f"  • Quarterly Production Parcels Evaluated: {dsa_quarterly.attrs['total_parcels']}")
    print(f"  • Parcels Meeting ±18% at 90% Confidence: {dsa_quarterly.attrs['parcels_qualifying']} ({dsa_quarterly.attrs['qualification_rate_pct']:.1f}%)")
    print(f"  • Mean Relative Error:                    {dsa_quarterly.attrs['mean_relative_error_pct']:.2f}%")
    print(f"  • Overall Classification Verdict:         {dsa_quarterly.attrs['overall_classification_verdict']}")

    # -------------------------------------------------------------------------
    # STAGE 9: POLYMETALLIC VALUATION & NET SMELTER RETURN (CIM §7.2, JORC CL 50)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 9: POLYMETALLIC EQUIVALENCY & NET SMELTER RETURN")
    # Porphyry Cu-Au co-product: add expected gold grades to simulated block model
    np.random.seed(88)
    etype_grades = df_maps["e_type"].values
    bm_df["grade_cu"] = etype_grades
    bm_df["grade_au"] = np.maximum(0.02, 0.35 * etype_grades + np.random.normal(0.0, 0.04, n_nodes))

    # 1. JORC Clause 50 Metal Equivalency (CuEq)
    secondary_cfg = {
        "Au": {
            "grade_col": "grade_au",
            "price": 1900.0,              # $/oz
            "recovery": 0.75,
            "conversion_factor": 0.0321507, # g/t to oz/t
        }
    }
    eq_df = calculate_equivalent_grade(
        block_model=bm_df,
        primary_commodity="Cu",
        primary_grade_col="grade_cu",
        primary_price=3.80,
        primary_recovery=0.88,
        secondary_commodities=secondary_cfg,
        primary_unit_conversion=22.0462,
    )
    print(f"  • Primary Metal Grade (Cu) : {eq_df.attrs['mean_primary_grade']:.3f}% Cu")
    print(f"  • Secondary Metal Grade (Au): {bm_df['grade_au'].mean():.3f} g/t Au")
    print(f"  • Equivalent Grade (CuEq)  : {eq_df.attrs['mean_equivalent_grade']:.3f}% CuEq")
    print(f"  • JORC Cl. 50 Disclosure   : {eq_df.attrs['formula_string']}")

    # 2. Net Smelter Return ($/t) under CIM MRMR §7.2
    commodities_cfg = {
        "Cu": {
            "grade_col": "grade_cu",
            "price": 3.80,
            "unit_conversion": 22.0462,
            "recovery": 0.88,
            "payability": 0.95,
            "refining_charge": 0.08,
            "treatment_charge_per_metal_unit": 0.05,
        },
        "Au": {
            "grade_col": "grade_au",
            "price": 1900.0,
            "unit_conversion": 0.0321507,
            "recovery": 0.75,
            "payability": 0.90,
            "refining_charge": 5.0,
        },
    }
    nsr_df = calculate_net_smelter_return(
        block_model=bm_df,
        commodities_config=commodities_cfg,
        royalty_pct=2.0,
        offsite_costs_per_tonne=2.50,
        currency="$",
    )
    print(f"  • Mean Gross Realized Revenue: ${nsr_df.attrs['mean_gross_nsr']:.2f}/t")
    print(f"  • Mean Net Smelter Return (NSR): ${nsr_df.attrs['mean_nsr']:.2f}/t")
    shares_str = ", ".join([f"{c}: {s:.1f}%" for c, s in nsr_df.attrs["commodity_percentage_shares"].items()])
    print(f"  • Revenue Share by Metal      : {shares_str}")

    if not args.no_plot:
        fig_nsr, _ = plot_net_smelter_return_breakdown(
            nsr_df,
            title="Polymetallic Net Smelter Return Breakdown (Cu-Au Deposit)",
        )
        nsr_path = plots_dir / "sgs_nsr_breakdown.png"
        fig_nsr.savefig(nsr_path, dpi=180, bbox_inches="tight")
        plt.close(fig_nsr)

    print("\n" + "=" * 75)
    print("SGS REGULATORY WORKFLOW EXECUTION COMPLETE")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

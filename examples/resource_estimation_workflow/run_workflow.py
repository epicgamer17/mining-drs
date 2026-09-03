"""End-to-End Functional Resource Estimation Workflow (Stages 1 through 5).

Demonstrates the complete industry-standard workflow complying with CIM / NI 43-101 / JORC:
1. Data Preparation: Down-hole compositing (equal support) and high-grade capping (top-cutting).
2. Exploratory Data Analysis (EDA): Global statistics, CV rule of thumb, and log-probability diagnostics.
3. Spatial Domain Delineation: Boundary contact analysis (Hard vs. Soft boundary evaluation).
4. Interpolation / Extrapolation: Ordinary Kriging, RPEEE resource statement, and reserve conversion.
5. Production Reconciliation: Harry Parker (2012) F1, F2, F3 Mine-to-Mill performance factors.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from drs_mining.components.estimation import (
    composite_drillhole_intervals,
    apply_grade_capping,
    exploratory_data_analysis,
    plot_eda_distributions,
    contact_profile_analysis,
    plot_contact_profile,
    ordinary_kriging_grid_estimation,
    calculate_cut_off_grade,
    convert_resource_to_reserve,
    format_resource_statement,
    format_reserve_statement,
    plot_resource_to_reserve_waterfall,
    reconcile_production_to_reserve,
    plot_production_reconciliation,
    cell_declustering,
)


def generate_synthetic_deposit() -> pd.DataFrame:
    """Generates raw exploration drill core assays across two geological domains."""
    np.random.seed(42)
    records = []
    hole_ids = [f"DH{i+1:02d}" for i in range(16)]

    # 16 drillholes arranged on a grid across a contact at x = 200m
    for idx, hid in enumerate(hole_ids):
        col_x = 80.0 + (idx % 4) * 80.0
        col_y = 80.0 + (idx // 4) * 80.0

        # Down-hole depth from 0 to 60m with variable core run lengths (0.8m to 2.2m)
        depth = 0.0
        while depth < 60.0:
            interval_len = np.random.uniform(0.8, 2.2)
            to_depth = min(60.0, depth + interval_len)

            # Geological domain: Porphyry Core (x >= 200) vs. Skarn Wall Rock (x < 200)
            is_core = col_x >= 200.0
            domain = "PorphyryCore" if is_core else "SkarnWallRock"

            # Grade distribution: Core has higher grade with log-normal skewness
            if is_core:
                base_g = np.random.lognormal(mean=0.35, sigma=0.45)
            else:
                base_g = np.random.lognormal(mean=-0.80, sigma=0.40)

            # Signed distance to north-south geological contact at x = 200m
            dist_to_contact = col_x - 200.0

            records.append({
                "hole_id": hid,
                "from_m": depth,
                "to_m": to_depth,
                "length": to_depth - depth,
                "grade": max(0.05, float(base_g)),
                "domain": domain,
                "x": col_x,
                "y": col_y,
                "dist_to_contact": dist_to_contact,
            })
            depth = to_depth

    df = pd.DataFrame(records)
    # Inject a couple of extreme outlier nugget assays to test capping
    df.loc[12, "grade"] = 18.50
    df.loc[85, "grade"] = 14.20
    return df


def main():
    print("\n" + "=" * 75)
    print("      END-TO-END RESOURCE ESTIMATION & PRODUCTION RECONCILIATION WORKFLOW")
    print("=" * 75)

    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # STAGE 1: DATA PREPARATION (COMPOSITING & CAPPING)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 1: DATA PREPARATION (Support Effect & Capping)")
    raw_df = generate_synthetic_deposit()
    print(f"  • Loaded {len(raw_df)} raw drill core assay intervals across 16 holes.")
    print(f"  • Raw interval lengths: min={raw_df['length'].min():.2f}m, max={raw_df['length'].max():.2f}m.")

    # 1. Down-hole regular compositing to 2.0m with domain boundary constraint
    comp_df = composite_drillhole_intervals(
        raw_df,
        composite_length=2.0,
        domain_col="domain",
        min_length_ratio=0.50,
        remnant_strategy="discard",
    )
    print(f"  • Created {len(comp_df)} regular 2.0m composites constrained by geological domain.")
    print(f"  • Discarded terminal remnants: {comp_df.attrs['discarded_remnants_count']}.")

    # 2. Statistical Top-Cutting (Capping) on composites at 99th percentile
    capped_df = apply_grade_capping(
        comp_df,
        percentile=99.0,
        grade_col="grade",
        output_col="capped_grade",
    )
    cap_sum = capped_df.attrs["capping_summary"]
    print(f"  • Applied grade capping at {cap_sum['cap_grade']:.2f}% Cu (P99 threshold).")
    print(f"  • Capped samples: {cap_sum['samples_capped']} ({cap_sum['samples_capped_pct']:.1f}%).")
    print(f"  • Total metal reduction: {cap_sum['metal_reduction_pct']:.2f}% (acceptable industry range: 1–5%).")
    print(f"  • CV reduction: {cap_sum['uncapped_cv']:.2f} -> {cap_sum['capped_cv']:.2f}.")

    # -------------------------------------------------------------------------
    # STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    # Run cell declustering to compute spatial weights for EDA comparison
    weights, _, _ = cell_declustering(
        capped_df,
        grade_col="capped_grade",
    )
    capped_df["declust_weight"] = weights

    eda_summary = exploratory_data_analysis(
        capped_df,
        grade_col="capped_grade",
        weights_col="declust_weight",
    )
    print(eda_summary[["Naive", "Declustered"]].round(3).to_string())
    print(f"\n  • Geostatistical CV Audit: {eda_summary.attrs['cv_status']}")
    if "clustering_bias_pct" in eda_summary.attrs:
        print(f"  • Preferential Drilling Bias: {eda_summary.attrs['clustering_bias_pct']:+.1f}%")

    # Generate 3-Panel EDA Distribution Plot
    fig_eda, _ = plot_eda_distributions(
        capped_df,
        grade_col="grade",
        capped_grade_col="capped_grade",
        cap_grade=cap_sum["cap_grade"],
        grade_unit="% Cu",
        title="Deposit Exploratory Data Analysis & Capping Validation",
    )
    eda_plot_path = plots_dir / "workflow_eda_distribution.png"
    fig_eda.savefig(eda_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_eda)
    print(f"  • EDA distribution plot saved to: {eda_plot_path}")

    # -------------------------------------------------------------------------
    # STAGE 3: SPATIAL DOMAIN DELINEATION (CONTACT ANALYSIS)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 3: SPATIAL DOMAIN DELINEATION (Contact Analysis)")
    contact_df = contact_profile_analysis(
        capped_df,
        domain_col="domain",
        grade_col="capped_grade",
        contact_surface=[(200.0, 0.0), (200.0, 500.0)],
        bin_width=20.0,
        max_distance=120.0,
        domain_a="SkarnWallRock",
        domain_b="PorphyryCore",
    )
    print(f"  • Evaluated boundary continuity: SkarnWallRock vs. PorphyryCore.")
    print(f"  • Contact Step Jump Δ: {contact_df.attrs['step_change']:.2f}% Cu ({contact_df.attrs['step_ratio']*100:.1f}% relative).")
    print(f"  • Boundary Decision: {contact_df.attrs['boundary_type']}")
    print(f"  • Recommendation: {contact_df.attrs['recommendation']}")

    fig_contact, _ = plot_contact_profile(
        contact_df,
        domain_a_name="Skarn Wall Rock",
        domain_b_name="Porphyry Core",
        grade_unit="% Cu",
        title="Boundary Contact Analysis: Skarn vs. Porphyry Contact",
    )
    contact_plot_path = plots_dir / "workflow_contact_profile.png"
    fig_contact.savefig(contact_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_contact)
    print(f"  • Contact profile plot saved to: {contact_plot_path}")

    # -------------------------------------------------------------------------
    # STAGE 4: INTERPOLATION & RESERVE CONVERSION
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 4: INTERPOLATION & RESERVE DELINEATION")
    # Multi-domain Ordinary Kriging: honors Hard Boundary diagnosed in Stage 3
    dh_all = capped_df.groupby(["hole_id", "domain"])[["x", "y", "capped_grade"]].mean().reset_index()
    samples_xy = dh_all[["x", "y"]].to_numpy()
    sample_grades = dh_all["capped_grade"].to_numpy()
    sample_domains = dh_all["domain"].to_numpy()

    # Target grid covers deposit across both domains
    grid_x = np.linspace(80.0, 360.0, 30)
    grid_y = np.linspace(60.0, 340.0, 25)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    grid_domains = np.where(grid_pts[:, 0] >= 200.0, "PorphyryCore", "SkarnWallRock")

    ok_estimates, ok_vars = ordinary_kriging_grid_estimation(
        samples_xy=samples_xy,
        sample_grades=sample_grades,
        grid_points=grid_pts,
        sill={"PorphyryCore": 0.20, "SkarnWallRock": 0.10},
        range_param={"PorphyryCore": 120.0, "SkarnWallRock": 80.0},
        nugget={"PorphyryCore": 0.05, "SkarnWallRock": 0.02},
        k_neighbors=min(16, len(samples_xy)),
        sample_domains=sample_domains,
        grid_domains=grid_domains,
    )

    # Cut-Off Grade Determination
    cog = calculate_cut_off_grade(
        processing_cost=10.50,
        ga_cost=2.20,
        commodity_price=3.80,
        metallurgical_recovery=88.0,
        mining_cost=2.40,
        selling_cost=0.35,
        metal_conversion_factor=22.0462,  # % Cu to lbs/t
    )
    print(f"  • Economic Breakeven Cut-Off Grade: {cog:.3f}% Cu.")

    block_tonnes = 5000.0  # 5,000 t per block
    block_cats = np.where(ok_vars < 0.12, "Measured", "Indicated")

    block_res_df = pd.DataFrame({
        "category": block_cats,
        "grade": ok_estimates,
        "tonnes": block_tonnes,
        "x": grid_pts[:, 0],
        "y": grid_pts[:, 1],
        "domain": grid_domains,
    })

    # Mineral Resource Statement
    res_stmt = format_resource_statement(
        block_res_df,
        cutoff_grade=cog,
        commodity_price="$3.80/lb Cu",
        metallurgical_recovery=88.0,
        rpeee_constraint="Constrained within Lerchs-Grossmann optimized pit shell",
    )
    print("\n--- Mineral Resource Statement ---")
    print(res_stmt.to_string(index=False))

    # Reserve Conversion with Modifying Factors
    reserve_df = convert_resource_to_reserve(
        block_res_df,
        mining_dilution_pct=6.0,
        mining_recovery_pct=94.0,
        cutoff_grade=cog,
        dilution_grade=0.10,
    )

    res_stmt_table = format_reserve_statement(
        reserve_df,
        cutoff_grade=cog,
        mining_dilution_pct=6.0,
        mining_recovery_pct=94.0,
        commodity_price="$3.80/lb Cu",
        metallurgical_recovery=88.0,
    )
    print("\n--- Mineral Reserve Statement (Run-of-Mine Feed) ---")
    print(res_stmt_table.to_string(index=False))

    # Save Waterfall Bridge
    fig_wf, _ = plot_resource_to_reserve_waterfall(
        block_res_df,
        reserve_df,
        cutoff_grade=cog,
        mining_dilution_pct=6.0,
        mining_recovery_pct=94.0,
        dilution_grade=0.10,
        grade_unit="% Cu",
    )
    wf_plot_path = plots_dir / "workflow_reserve_waterfall.png"
    fig_wf.savefig(wf_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_wf)
    print(f"  • Reserve reconciliation waterfall saved to: {wf_plot_path}")

    # -------------------------------------------------------------------------
    # STAGE 5: PRODUCTION RECONCILIATION (PARKER F1, F2, F3 FACTORS)
    # -------------------------------------------------------------------------
    print("\n>>> STAGE 5: PRODUCTION RECONCILIATION (Mine-to-Mill Validation)")
    # 12-Month Historical Production Reconciliation
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    tot_rom_tonnes = reserve_df["rom_tonnes"].sum()
    monthly_rom_tonnes = tot_rom_tonnes / 12.0
    avg_rom_grade = reserve_df["rom_grade"].mean()

    # Synthetic monthly production logs simulating real operational dynamics
    prod_records = []
    for m in months:
        # Long-term reserve plan for the month
        res_t = monthly_rom_tonnes * np.random.uniform(0.95, 1.05)
        res_g = avg_rom_grade * np.random.uniform(0.96, 1.04)

        # Grade control model (blastholes): slight local variation
        gc_t = res_t * np.random.uniform(0.98, 1.04)
        gc_g = res_g * np.random.uniform(0.97, 1.03)

        # Actual mill received (weightometer + mill head grade assay): operational dilution
        mill_t = gc_t * np.random.uniform(0.99, 1.05)
        mill_g = gc_g * np.random.uniform(0.96, 1.01)

        prod_records.append({
            "period": m,
            "res_t": res_t / 1e6,  # Mt
            "res_g": res_g,
            "gc_t": gc_t / 1e6,
            "gc_g": gc_g,
            "mill_t": mill_t / 1e6,
            "mill_g": mill_g,
        })
    p_df = pd.DataFrame(prod_records)

    reconcile_df = reconcile_production_to_reserve(
        reserve_data=p_df[["period", "res_t", "res_g"]].rename(columns={"res_t": "tonnes", "res_g": "grade"}),
        plant_data=p_df[["period", "mill_t", "mill_g"]].rename(columns={"mill_t": "tonnes", "mill_g": "grade"}),
        grade_control_data=p_df[["period", "gc_t", "gc_g"]].rename(columns={"gc_t": "tonnes", "gc_g": "grade"}),
        period_col="period",
        grade_unit="% Cu",
    )

    print("\n--- Harry Parker (2012) Mine-to-Mill Reconciliation Summary ---")
    cols_display = [
        "period",
        "f1_tonnes_ratio", "f1_grade_ratio", "f1_metal_factor",
        "f2_tonnes_ratio", "f2_grade_ratio", "f2_metal_factor",
        "f3_metal_factor",
    ]
    print(reconcile_df[cols_display].round(3).to_string(index=False))
    print(f"\n  • System Health Diagnosis: {reconcile_df.attrs['health_status']}")
    print(f"  • Cumulative F1 (Model to Mine) : {reconcile_df.attrs['f1_factor']:.3f}")
    print(f"  • Cumulative F2 (Mine to Mill)  : {reconcile_df.attrs['f2_factor']:.3f}")
    print(f"  • Cumulative F3 (Total Value)   : {reconcile_df.attrs['f3_factor']:.3f}")

    fig_rec, _ = plot_production_reconciliation(
        reconcile_df,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        metal_unit="kt",
        title="12-Month Mine-to-Mill Production Reconciliation Dashboard (Parker F-Factors)",
    )
    rec_plot_path = plots_dir / "workflow_production_reconciliation.png"
    fig_rec.savefig(rec_plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig_rec)
    print(f"  • Reconciliation dashboard saved to: {rec_plot_path}")

    print("\n" + "=" * 75)
    print("                 WORKFLOW EXECUTION COMPLETE")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

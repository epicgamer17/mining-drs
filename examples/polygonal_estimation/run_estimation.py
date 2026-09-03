"""Example: Polygonal Mineral Reserve Estimation & Grade-Tonnage Analysis.

Demonstrates practical resource estimation for an exploration bench using the
method of polygons of influence (Voronoi tessellation):
1. Loads exploration drillhole collar assays and intercept thicknesses.
2. Defines concession / pit perimeter boundary.
3. Computes bounded polygons of influence, volumes, and in-situ tonnages.
4. Evaluates extrapolation limits using maximum radius of influence.
5. Generates global reserve summary and cutoff grade-tonnage sensitivity curve.
6. Renders 2D spatial plan map colored by assay grade.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from drs_mining.components.estimation import (
    polygonal_estimation,
    polygonal_reserve_summary,
    format_reserve_summary,
    grade_tonnage_table,
    plot_polygonal_map,
    plot_grade_tonnage_curve,
    format_resource_statement,
)


def create_sample_drillholes() -> pd.DataFrame:
    """Create sample exploration drillhole collar assays for a porphyry copper deposit."""
    return pd.DataFrame({
        "hole_id": [
            "DH01", "DH02", "DH03", "DH04",
            "DH05", "DH06", "DH07", "DH08",
            "DH09", "DH10", "DH11", "DH12",
        ],
        "x": [
            180.0, 320.0, 460.0, 580.0,
            200.0, 350.0, 480.0, 620.0,
            220.0, 340.0, 470.0, 590.0,
        ],
        "y": [
            160.0, 150.0, 170.0, 180.0,
            280.0, 300.0, 290.0, 310.0,
            420.0, 410.0, 430.0, 440.0,
        ],
        "grade": [
            0.42, 0.78, 1.15, 0.65,
            0.55, 1.45, 1.82, 0.90,
            0.35, 0.88, 1.25, 0.50,
        ],  # Copper grade (% Cu)
        "thickness": [12.0] * 12,  # Bench height = 12 meters
    })


def get_concession_boundary() -> list[tuple[float, float]]:
    """Defines the perimeter of the mining lease / pit sector."""
    return [
        (100.0, 100.0),
        (700.0, 80.0),
        (720.0, 500.0),
        (450.0, 520.0),
        (120.0, 480.0),
    ]


def main():
    parser = argparse.ArgumentParser(description="Polygonal Reserve Estimation Demo")
    parser.add_argument(
        "--max-radius",
        type=float,
        default=None,
        help="Maximum radius of influence around drillholes in meters (e.g. 100.0)",
    )
    parser.add_argument(
        "--clip-to-convex-hull",
        action="store_true",
        help="Clip polygons strictly to drillhole convex hull (prevent perimeter extrapolation)",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=2.7,
        help="In-situ rock bulk density (t/m^3)",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default="plots/polygonal_reserve_map.png",
        help="Output image path for 2D plan map",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable map generation and display",
    )
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("      POLYGONAL MINERAL ESTIMATION: COPPER BENCH BENCHMARK")
    print("=" * 64)

    # 1. Load drillhole data
    drillholes = create_sample_drillholes()
    boundary = get_concession_boundary()

    print(f"\nLoaded {len(drillholes)} exploration drill holes.")
    print(drillholes[["hole_id", "x", "y", "grade", "thickness"]].to_string(index=False))

    # 2. Run polygonal estimation
    print(f"\nComputing Voronoi polygons (max_radius={args.max_radius} m, clip_to_convex_hull={args.clip_to_convex_hull})...")
    polygons_df = polygonal_estimation(
        drillholes,
        boundary=boundary,
        bulk_density=args.density,
        max_radius=args.max_radius,
        clip_to_convex_hull=args.clip_to_convex_hull,
    )

    # 3. Print global reserve summary
    summary = polygonal_reserve_summary(polygons_df)
    print("\n" + format_reserve_summary(summary, grade_unit="% Cu", metal_unit="tonnes Cu"))

    # 4. Detailed polygon breakdown
    print("\n--- Individual Polygon Reserves ---")
    display_cols = ["hole_id", "grade", "area_m2", "volume_m3", "tonnes", "contained_metal"]
    formatted_df = polygons_df[display_cols].copy()
    formatted_df["area_m2"] = formatted_df["area_m2"].map(lambda x: f"{x:,.0f}")
    formatted_df["volume_m3"] = formatted_df["volume_m3"].map(lambda x: f"{x:,.0f}")
    formatted_df["tonnes"] = formatted_df["tonnes"].map(lambda x: f"{x:,.0f}")
    formatted_df["contained_metal"] = formatted_df["contained_metal"].map(lambda x: f"{x:,.1f}")
    print(formatted_df.to_string(index=False))

    # 5. Grade-Tonnage curve (cutoff analysis)
    cutoffs = [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    gt_curve = grade_tonnage_table(polygons_df, cutoffs=cutoffs)

    print("\n--- Grade–Tonnage Sensitivity Curve ---")
    gt_display = gt_curve.copy()
    gt_display["ore_tonnes"] = gt_display["ore_tonnes"].map(lambda x: f"{x:,.0f}")
    gt_display["ore_grade"] = gt_display["ore_grade"].map(lambda x: f"{x:.3f}%")
    gt_display["waste_tonnes"] = gt_display["waste_tonnes"].map(lambda x: f"{x:,.0f}")
    gt_display["contained_metal"] = gt_display["contained_metal"].map(lambda x: f"{x:,.1f}")
    gt_display["strip_ratio"] = gt_display["strip_ratio"].map(lambda x: f"{x:.2f}")
    gt_display["ore_recovery_pct"] = gt_display["ore_recovery_pct"].map(lambda x: f"{x:.1f}%")
    gt_display["metal_recovery_pct"] = gt_display["metal_recovery_pct"].map(lambda x: f"{x:.1f}%")
    print(gt_display.to_string())

    # 6. Official Mineral Resource Statement (NI 43-101 / JORC Significant Figures)
    poly_class = polygons_df.copy()
    poly_cats = []
    for a in poly_class["area_m2"]:
        if a <= 25000.0:
            poly_cats.append("Measured")
        elif a <= 45000.0:
            poly_cats.append("Indicated")
        else:
            poly_cats.append("Inferred")
    poly_class["category"] = poly_cats

    base_cutoff = 0.50
    resource_stmt = format_resource_statement(
        poly_class,
        cutoff_grade=base_cutoff,
        grade_unit="% Cu",
        tonnage_unit="Mt",
        metal_unit="kt",
        commodity_price="$3.80/lb Cu",
        metallurgical_recovery=88.0,
        rpeee_constraint="Constrained within pit concession boundary",
    )

    print(
        f"\n--- Official Mineral Resource Statement (Polygonal Model, Cutoff: {base_cutoff:.2f}% Cu) ---"
    )
    print(resource_stmt.to_string(index=False))
    print("\nCompliance Footnotes:")
    for fn in resource_stmt.attrs.get("footnotes", []):
        print(f"  {fn}")

    # 7. Spatial visualization & Grade-Tonnage Curve
    if not args.no_plot:
        Path(args.save_plot).parent.mkdir(parents=True, exist_ok=True)
        fig, (ax_map, ax_gt) = plt.subplots(1, 2, figsize=(18, 7))

        # Panel 1: Plan Map
        plot_polygonal_map(
            polygons_df,
            boundary=boundary,
            title=f"Polygonal Reserves (max_radius={args.max_radius})",
            cmap="viridis",
            ax=ax_map,
        )

        # Panel 2: Standard Dual-Axis Grade-Tonnage Curve (NI 43-101)
        plot_grade_tonnage_curve(
            gt_curve,
            grade_unit="% Cu",
            tonnage_unit="Mt",
            title="Grade–Tonnage Sensitivity Curve",
            ax=ax_gt,
            show_metal=True,
        )

        fig.suptitle("Executive Polygonal Mineral Resource Report", fontsize=15, fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.save_plot, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"\nExecutive report figure saved to: {args.save_plot}")


if __name__ == "__main__":
    main()

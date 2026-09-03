"""Polygonal mineral resource and reserve estimation.

Provides functional tools for geometric polygonal estimation (method of polygons
of influence), global reserve calculations, cutoff-grade sensitivity analysis
(grade-tonnage curves), and spatial 2D plan map visualizations.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.ops import voronoi_diagram
from scipy.spatial import KDTree


def polygonal_estimation(
    drillholes: pd.DataFrame,
    boundary: Optional[Sequence[Tuple[float, float]]] = None,
    bulk_density: float = 2.7,
    max_radius: Optional[float] = None,
    grade_col: str = "grade",
    thickness_col: str = "thickness",
    x_col: str = "x",
    y_col: str = "y",
    hole_id_col: str = "hole_id",
) -> pd.DataFrame:
    """Estimates mineral reserves using the method of polygons of influence (Voronoi tessellation).

    Parameters
    ----------
    drillholes : pd.DataFrame
        DataFrame of exploration drill holes or blast holes with coordinates, assays, and intercept thicknesses.
    boundary : Sequence[Tuple[float, float]], optional
        Closed polygon coordinates [(x1, y1), (x2, y2), ...] defining concession or pit perimeter.
    bulk_density : float, default 2.7
        Specific gravity / bulk density in tonnes per cubic meter (t/m^3).
    max_radius : float, optional
        Maximum radius of influence (meters) to restrict spatial extrapolation around each drillhole.
    grade_col : str, default "grade"
        Column name for assay grade (e.g., % Cu, g/t Au, or attribute fraction).
    thickness_col : str, default "thickness"
        Column name for vertical intercept thickness or bench height in meters.
    x_col : str, default "x"
        Easting coordinate column name.
    y_col : str, default "y"
        Northing coordinate column name.
    hole_id_col : str, default "hole_id"
        Identifier column name for each drillhole.

    Returns
    -------
    pd.DataFrame
        Table with one row per drillhole polygon containing:
        - hole_id, x, y, grade, thickness
        - area_m2: Plan area of polygon of influence (m^2)
        - volume_m3: In-situ rock volume (area * thickness)
        - tonnes: Mineral mass (volume * bulk_density)
        - contained_metal: Quantity of metal (tonnes * grade)
        - vertices: List of (x, y) coordinates forming the polygon perimeter
    """
    output_cols = [
        hole_id_col,
        x_col,
        y_col,
        grade_col,
        thickness_col,
        "area_m2",
        "volume_m3",
        "tonnes",
        "contained_metal",
        "vertices",
    ]

    if drillholes.empty:
        return pd.DataFrame(columns=output_cols)

    points = [Point(x, y) for x, y in zip(drillholes[x_col], drillholes[y_col])]
    xs = drillholes[x_col].to_numpy()
    ys = drillholes[y_col].to_numpy()

    # 1. Determine outer bounding geometry
    if boundary is not None and len(boundary) >= 3:
        boundary_geom = Polygon(boundary)
        envelope = boundary_geom
    else:
        # Default envelope: data bounding box with 10% padding
        pad_x = max(float(xs.max() - xs.min()), 100) * 0.1
        pad_y = max(float(ys.max() - ys.min()), 100) * 0.1
        boundary_geom = box(
            float(xs.min() - pad_x),
            float(ys.min() - pad_y),
            float(xs.max() + pad_x),
            float(ys.max() + pad_y),
        )
        envelope = boundary_geom

    # 2. Tessellation: Single hole vs Multi-hole Voronoi
    if len(drillholes) == 1:
        voronoi_cells = [envelope]
    else:
        multi_pt = MultiPoint(points)
        # Bounded Voronoi diagram inside envelope
        voronoi_collection = voronoi_diagram(multi_pt, envelope=envelope)
        voronoi_cells = list(voronoi_collection.geoms)

    # 3. Match each drillhole to its cell and clip against boundary & max_radius
    rows = []
    for _, hole in drillholes.iterrows():
        pt = Point(hole[x_col], hole[y_col])

        # Find Voronoi cell containing this drillhole collar
        cell = next((c for c in voronoi_cells if c.intersects(pt)), None)

        if cell is None:
            # Fallback: nearest cell by centroid distance in case of floating point issues when a point is on a boundary.
            cell = min(voronoi_cells, key=lambda c: c.distance(pt))

        # Clip cell against boundary
        clipped = cell.intersection(boundary_geom)

        # Optionally clip to maximum radius of influence (circular buffer)
        if max_radius is not None and max_radius > 0.0:
            clipped = clipped.intersection(pt.buffer(max_radius))

        # Extract polygon metrics
        area = float(clipped.area) if not clipped.is_empty else 0.0
        thickness = float(hole[thickness_col])
        grade = float(hole[grade_col])
        volume = area * thickness
        tonnes = volume * bulk_density
        metal = tonnes * grade

        # Extract exterior vertices [(x1, y1), (x2, y2), ...]
        if not clipped.is_empty and hasattr(clipped, "exterior"):
            vertices = list(clipped.exterior.coords)
        elif not clipped.is_empty and hasattr(clipped, "geoms"):
            # MultiPolygon case (take largest piece or exterior of first)
            largest = max(clipped.geoms, key=lambda g: g.area)
            vertices = list(largest.exterior.coords)
        else:
            vertices = []

        rows.append(
            {
                hole_id_col: hole[hole_id_col],
                x_col: hole[x_col],
                y_col: hole[y_col],
                grade_col: grade,
                thickness_col: thickness,
                "area_m2": area,
                "volume_m3": volume,
                "tonnes": tonnes,
                "contained_metal": metal,
                "vertices": vertices,
            }
        )
    return pd.DataFrame(rows)


def polygonal_reserve_summary(
    df: pd.DataFrame,
    tonnes_col: str = "tonnes",
    grade_col: str = "grade",
    area_col: str = "area_m2",
    volume_col: str = "volume_m3",
) -> dict[str, float]:
    """Calculates global in-situ reserve metrics from an estimated polygon table.

    Parameters
    ----------
    df : pd.DataFrame
        Estimated polygon table (output of polygonal_estimation).
    tonnes_col : str, default "tonnes"
        Column name for mineral tonnage.
    grade_col : str, default "grade"
        Column name for mineral grade.
    area_col : str, default "area_m2"
        Column name for polygon surface area.
    volume_col : str, default "volume_m3"
        Column name for in-situ rock volume.

    Returns
    -------
    dict[str, float]
        Dictionary containing total_tonnes, mean_grade, contained_metal,
        total_area_m2, total_volume_m3, drillhole_count, and mean_polygon_area_m2.
    """
    if df.empty:
        return {
            "total_tonnes": 0.0,
            "mean_grade": 0.0,
            "contained_metal": 0.0,
            "total_area_m2": 0.0,
            "total_volume_m3": 0.0,
            "drillhole_count": 0,
            "mean_polygon_area_m2": 0.0,
        }

    total_tonnes = float(df[tonnes_col].sum())
    total_area = float(df[area_col].sum()) if area_col in df.columns else 0.0
    total_volume = float(df[volume_col].sum()) if volume_col in df.columns else 0.0

    contained_metal = float((df[tonnes_col] * df[grade_col]).sum())
    mean_grade = (contained_metal / total_tonnes) if total_tonnes > 0.0 else 0.0

    return {
        "total_tonnes": total_tonnes,
        "mean_grade": mean_grade,
        "contained_metal": contained_metal,
        "total_area_m2": total_area,
        "total_volume_m3": total_volume,
        "drillhole_count": len(df),
        "mean_polygon_area_m2": total_area / len(df) if len(df) > 0 else 0.0,
    }


def format_reserve_summary(
    summary: Mapping[str, float],
    grade_unit: str = "%",
    metal_unit: str = "units",
) -> str:
    """Formats the reserve summary dictionary into an executive text table."""
    lines = [
        "=" * 64,
        "             POLYGONAL MINERAL RESERVE SUMMARY",
        "=" * 64,
        f"Total Mineral Tonnage  : {summary.get('total_tonnes', 0.0):>15,.1f} tonnes",
        f"Mean Weighted Grade    : {summary.get('mean_grade', 0.0):>15.4f} {grade_unit}",
        f"Contained Metal        : {summary.get('contained_metal', 0.0):>15,.1f} {metal_unit}",
        f"Total Surface Area     : {summary.get('total_area_m2', 0.0):>15,.1f} m²",
        f"Total Rock Volume      : {summary.get('total_volume_m3', 0.0):>15,.1f} m³",
        f"Total Sample Points    : {int(summary.get('drillhole_count', 0)):>15d}",
        f"Mean Area of Influence : {summary.get('mean_polygon_area_m2', 0.0):>15,.1f} m²",
        "=" * 64,
    ]
    return "\n".join(lines)


def grade_tonnage_table(
    df: pd.DataFrame,
    cutoffs: Sequence[float],
    grade_col: str = "grade",
    tonnes_col: str = "tonnes",
) -> pd.DataFrame:
    """Computes ore-waste distribution and recovery across varying cutoff grades.

    Parameters
    ----------
    df : pd.DataFrame
        Polygonal or block reserve table containing tonnage and grade columns.
    cutoffs : Sequence[float]
        List of cutoff grades to evaluate (e.g. [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).
    grade_col : str, default "grade"
        Grade column name.
    tonnes_col : str, default "tonnes"
        Tonnage column name.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by cutoff grade with columns:
        - ore_tonnes: Tonnage of mineral exceeding cutoff
        - ore_grade: Weighted average grade of ore
        - waste_tonnes: Tonnage of sub-economic material
        - contained_metal: Metal quantity in ore
        - strip_ratio: Waste-to-ore tonnage ratio
        - ore_recovery_pct: Percentage of total deposit tonnage retained as ore
        - metal_recovery_pct: Percentage of total contained metal recovered in ore
    """
    total_tonnes = float(df[tonnes_col].sum()) if not df.empty else 0.0
    total_metal = float((df[tonnes_col] * df[grade_col]).sum()) if not df.empty else 0.0

    rows = []
    for c in sorted(cutoffs):
        ore_mask = df[grade_col] >= c
        ore_tonnes = float(df.loc[ore_mask, tonnes_col].sum())
        ore_metal = float(
            (df.loc[ore_mask, tonnes_col] * df.loc[ore_mask, grade_col]).sum()
        )
        ore_grade = (ore_metal / ore_tonnes) if ore_tonnes > 0.0 else 0.0
        waste_tonnes = max(0.0, total_tonnes - ore_tonnes)
        strip_ratio = (waste_tonnes / ore_tonnes) if ore_tonnes > 0.0 else float("inf")
        ore_rec = (ore_tonnes / total_tonnes * 100.0) if total_tonnes > 0.0 else 0.0
        metal_rec = (ore_metal / total_metal * 100.0) if total_metal > 0.0 else 0.0

        rows.append(
            {
                "cutoff": c,
                "ore_tonnes": ore_tonnes,
                "ore_grade": ore_grade,
                "waste_tonnes": waste_tonnes,
                "contained_metal": ore_metal,
                "strip_ratio": strip_ratio,
                "ore_recovery_pct": ore_rec,
                "metal_recovery_pct": metal_rec,
            }
        )

    res_df = pd.DataFrame(rows)
    return res_df.set_index("cutoff")


def plot_polygonal_map(
    df: pd.DataFrame,
    boundary: Optional[Sequence[Tuple[float, float]]] = None,
    grade_col: str = "grade",
    vertices_col: str = "vertices",
    x_col: str = "x",
    y_col: str = "y",
    hole_id_col: str = "hole_id",
    title: str = "Polygonal Mineral Reserve Estimation",
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plots 2D plan map of polygons of influence colored by grade with sample collars.

    Parameters
    ----------
    df : pd.DataFrame
        Estimated polygon table containing vertices and drillhole coordinates.
    boundary : Sequence[Tuple[float, float]], optional
        Closed polygon [(x, y), ...] representing the concession or pit boundary.
    grade_col : str, default "grade"
        Grade column name used for colormap shading.
    vertices_col : str, default "vertices"
        Column containing polygon vertex coordinate sequences.
    x_col : str, default "x"
        Easting coordinate column name.
    y_col : str, default "y"
        Northing coordinate column name.
    hole_id_col : str, default "hole_id"
        Drill hole label column name.
    title : str, default "Polygonal Mineral Reserve Estimation"
        Plot title.
    cmap : str, default "viridis"
        Matplotlib colormap name.
    ax : plt.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    plt.Axes
        Matplotlib axes with rendered polygonal map.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    grades = df[grade_col].values if not df.empty else np.array([0.0])
    norm = mcolors.Normalize(vmin=float(grades.min()), vmax=float(grades.max()))
    colormap = plt.get_cmap(cmap)

    patches = []
    colors = []

    if vertices_col in df.columns:
        for _, row in df.iterrows():
            verts = row[vertices_col]
            if verts is not None and len(verts) >= 3:
                poly = MplPolygon(verts, closed=True)
                patches.append(poly)
                colors.append(colormap(norm(row[grade_col])))

    if patches:
        p_coll = PatchCollection(
            patches,
            facecolors=colors,
            edgecolors="black",
            linewidths=0.75,
            alpha=0.85,
        )
        ax.add_collection(p_coll)

    # Plot sample collar points
    if x_col in df.columns and y_col in df.columns:
        ax.scatter(
            df[x_col],
            df[y_col],
            color="black",
            s=30,
            zorder=5,
            label="Drillhole Collars",
        )
        if hole_id_col in df.columns:
            for _, row in df.iterrows():
                label = f"{row[hole_id_col]}\n({row[grade_col]:.2f})"
                ax.annotate(
                    label,
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                    color="#212121",
                    zorder=6,
                )

    # Plot outer boundary if provided
    if boundary is not None and len(boundary) >= 3:
        b_poly = np.array(list(boundary) + [boundary[0]])
        ax.plot(
            b_poly[:, 0],
            b_poly[:, 1],
            color="red",
            linestyle="--",
            linewidth=2.0,
            label="Deposit Boundary",
            zorder=7,
        )

    # Colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Assay Grade ({grade_col})", fontsize=11)

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Easting (m)", fontsize=11)
    ax.set_ylabel("Northing (m)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")

    return ax


def inverse_distance_weighting(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    power: float = 2.0,
    k_neighbors: int = 8,
    max_radius: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse Distance Weighting (IDW) interpolation using a k-d tree.

    Parameters
    ----------
    samples_xy : np.ndarray
        Sample collar/assay coordinates of shape (N, 2) or (N, 3).
    sample_grades : np.ndarray
        Sample assay values of shape (N,).
    grid_points : np.ndarray
        Target estimation coordinates of shape (M, 2) or (M, 3).
    power : float, default 2.0
        Distance weighting exponent alpha (2.0 = standard IDW^2).
    k_neighbors : int, default 8
        Number of nearest neighbors to query (k=1 gives Nearest Neighbor).
    max_radius : float, optional
        Maximum search radius. Points with no samples within this radius are NaN.

    Returns
    -------
    estimated_grades : np.ndarray
        Interpolated grades of shape (M,).
    distances : np.ndarray
        Distances to informing samples. Shape (M,) if k=1, else (M, k).
    """
    # Step 1: Build spatial index
    tree = KDTree(samples_xy)

    # Step 2: Query k-nearest neighbors within search radius
    upper_bound = max_radius if max_radius is not None else float("inf")
    distances, indices = tree.query(
        grid_points, k=k_neighbors, distance_upper_bound=upper_bound
    )

    # Ensure uniform 2D shape (M, k) even when k_neighbors == 1
    if k_neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # Safe indexing: pad sample grades with NaN at index N for out-of-bound neighbors
    padded_grades = np.append(sample_grades, np.nan)
    neighbor_grades = padded_grades[indices]  # Shape (M, k)

    # Step 3: Identify valid neighbors and exact collocations (distance < 1e-6)
    valid_neighbors = distances <= upper_bound
    is_exact_match = distances < 1e-6
    has_any_exact = np.any(is_exact_match, axis=1)

    # Step 4: Calculate inverse distance weights (raw_weights)
    # Use 0.0 for exact matches and out-of-bounds to prevent divide-by-zero
    safe_distances = np.maximum(distances, 1e-12)
    raw_weights = np.where(
        valid_neighbors & ~is_exact_match,
        1.0 / np.power(safe_distances, power),
        0.0,
    )

    # Step 5: Normalize weights across neighbors
    total_weights = np.sum(raw_weights, axis=1, keepdims=True)  # Shape (M, 1)
    has_valid_weights = (total_weights > 0.0).ravel()

    normalized_weights = np.divide(
        raw_weights,
        total_weights,
        out=np.zeros_like(raw_weights),
        where=(total_weights > 0.0),
    )

    # Step 6: Compute weighted grade estimates
    estimated_grades = np.full(len(grid_points), np.nan)
    estimated_grades[has_valid_weights] = np.nansum(
        (normalized_weights * neighbor_grades)[has_valid_weights], axis=1
    )

    # Step 7: Overwrite exact collocations with exact drillhole grade
    if np.any(has_any_exact):
        first_exact_idx = np.argmax(is_exact_match[has_any_exact], axis=1)
        exact_sample_idxs = indices[has_any_exact, first_exact_idx]
        estimated_grades[has_any_exact] = sample_grades[exact_sample_idxs]

    # Return distances as (M,) if k=1 for convenience, else (M, k)
    dist_out = distances.ravel() if k_neighbors == 1 else distances
    return estimated_grades, dist_out


def nearest_neighbor_grid_estimation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    max_radius: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest neighbor estimation (IDW with k=1)."""
    return inverse_distance_weighting(
        samples_xy, sample_grades, grid_points, k_neighbors=1, max_radius=max_radius
    )

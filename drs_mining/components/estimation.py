"""Polygonal mineral resource and reserve estimation.

Provides functional tools for geometric polygonal estimation (method of polygons
of influence), global reserve calculations, cutoff-grade sensitivity analysis
(grade-tonnage curves), and spatial 2D plan map visualizations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.ops import voronoi_diagram
from scipy.spatial import KDTree, Delaunay
from scipy import stats


def polygonal_estimation(
    drillholes: pd.DataFrame,
    boundary: Optional[Sequence[Tuple[float, float]]] = None,
    bulk_density: float = 2.7,
    max_radius: Optional[float] = None,
    clip_to_convex_hull: bool = False,
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
    clip_to_convex_hull : bool, default False
        If True, clips all polygons strictly to the convex hull of the drillholes,
        preventing extrapolation beyond the outermost drill pattern.
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

    # TODO: Add domain_col support to execute domain-segregated polygonal estimation,
    # ensuring Voronoi polygons are strictly bounded/clipped by individual geological domain polygons.

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

    # Restrict bounding geometry to the convex hull of informing drillholes if requested
    if clip_to_convex_hull and len(points) >= 3:
        hull_geom = MultiPoint(points).convex_hull
        boundary_geom = boundary_geom.intersection(hull_geom)
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


def is_within_convex_hull(
    samples_xy: np.ndarray,
    grid_points: np.ndarray,
) -> np.ndarray:
    """Classifies target points as Interpolation (True) or Extrapolation (False).

    A target point x* is interpolated if it lies strictly within the Convex Hull
    of the informing drillhole samples S.

    Parameters
    ----------
    samples_xy : np.ndarray
        Sample collar/assay coordinates of shape (N, 2) or (N, 3).
    grid_points : np.ndarray
        Target estimation coordinates of shape (M, 2) or (M, 3).

    Returns
    -------
    np.ndarray of bool
        Boolean mask of shape (M,) where True indicates interpolation and False
        indicates extrapolation outside the drillhole envelope.
    """
    delaunay = Delaunay(samples_xy)
    return delaunay.find_simplex(grid_points) >= 0


def inverse_distance_weighting(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    power: float = 2.0,
    k_neighbors: int = 8,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
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
    mask_extrapolation : bool, default False
        If True, blocks outside the drillhole convex hull are set to NaN.

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

    # Step 8: Optionally mask out extrapolation blocks outside convex hull
    if mask_extrapolation:
        inside_hull = is_within_convex_hull(samples_xy, grid_points)
        estimated_grades[~inside_hull] = np.nan

    # Return distances as (M,) if k=1 for convenience, else (M, k)
    dist_out = distances.ravel() if k_neighbors == 1 else distances
    return estimated_grades, dist_out


def nearest_neighbor_grid_estimation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest neighbor estimation (IDW with k=1)."""
    return inverse_distance_weighting(
        samples_xy,
        sample_grades,
        grid_points,
        k_neighbors=1,
        max_radius=max_radius,
        mask_extrapolation=mask_extrapolation,
    )


def _theoretical_covariance(
    h: np.ndarray,
    model: str = "spherical",
    nugget: float = 0.0,
    sill: float = 1.0,
    range_param: float = 100.0,
) -> np.ndarray:
    """Evaluates theoretical spatial covariance C(h) = (c0 + c) - gamma(h).

    Parameters
    ----------
    h : np.ndarray
        Separation lag distance array.
    model : str, default "spherical"
        Variogram model ("spherical", "exponential", "gaussian").
    nugget : float, default 0.0
        Nugget variance c0 (micro-scale variance / noise at h=0).
    sill : float, default 1.0
        Partial sill variance c (total sill is c0 + c).
    range_param : float, default 100.0
        Practical correlation range a.

    Returns
    -------
    np.ndarray
        Covariance values C(h) of identical shape to h.
    """
    c0 = float(nugget)
    c = float(sill)
    a = max(float(range_param), 1e-6)
    total_sill = c0 + c

    h_arr = np.asarray(h, dtype=float)
    gamma = np.zeros_like(h_arr)

    # Positive lag mask (at h=0, gamma=0, C(0) = total_sill)
    pos_mask = h_arr > 1e-12

    if model.lower() == "spherical":
        hr = h_arr / a
        # Spherical variogram: c0 + c * [1.5*(h/a) - 0.5*(h/a)^3] for h <= a, else c0 + c
        within_range = (h_arr <= a) & pos_mask
        beyond_range = (h_arr > a) & pos_mask
        gamma[within_range] = c0 + c * (
            1.5 * hr[within_range] - 0.5 * np.power(hr[within_range], 3)
        )
        gamma[beyond_range] = total_sill
    elif model.lower() == "exponential":
        # Exponential variogram: c0 + c * [1 - exp(-3*h/a)]
        gamma[pos_mask] = c0 + c * (1.0 - np.exp(-3.0 * h_arr[pos_mask] / a))
    elif model.lower() == "gaussian":
        # Gaussian variogram: c0 + c * [1 - exp(-3*(h/a)^2)]
        gamma[pos_mask] = c0 + c * (
            1.0 - np.exp(-3.0 * np.power(h_arr[pos_mask] / a, 2))
        )
    else:
        raise ValueError(
            f"Unknown variogram model: '{model}'. Choose 'spherical', 'exponential', or 'gaussian'."
        )

    # Covariance relation: C(h) = C(0) - gamma(h)
    cov = total_sill - gamma
    cov[~pos_mask] = total_sill
    return cov


def simple_kriging_grid_estimation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    mean: float,
    sill: float,
    range_param: float,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    k_neighbors: int = 16,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple Kriging (SK) grid interpolation with known stationary mean.

    Estimates values at target grid points by solving the unconstrained Simple
    Kriging system K * lambda = k, where the estimate gracefully reverts to the
    known global prior mean in regions devoid of sample support.

    Parameters
    ----------
    samples_xy : np.ndarray
        Sample coordinates of shape (N, 2) or (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    grid_points : np.ndarray
        Target estimation coordinates of shape (M, 2) or (M, 3).
    mean : float
        Known stationary global mean of the domain.
    variogram_model : str, default "spherical"
        Theoretical variogram model ("spherical", "exponential", "gaussian").
    nugget : float, default 0.0
        Nugget variance c0 (measurement error / micro-scale noise).
    sill : float, default 1.0
        Partial sill variance c.
    range_param : float, default 100.0
        Spatial correlation range a.
    k_neighbors : int, default 16
        Maximum conditioning samples to query per target grid node.
    max_radius : float, optional
        Maximum search neighborhood radius.
    mask_extrapolation : bool, default False
        If True, masks blocks outside the drillhole convex hull to NaN.

    Returns
    -------
    estimated_grades : np.ndarray
        Simple Kriging grade estimates of shape (M,).
    kriging_variance : np.ndarray
        Estimation variance sigma_SK^2 of shape (M,).
    """
    # -------------------------------------------------------------------------
    n_targets = len(grid_points)
    total_sill = nugget + sill

    # Initialize with the prior mean and maximum uncertainty (total sill)
    estimates = np.full(n_targets, mean, dtype=float)
    variances = np.full(n_targets, total_sill, dtype=float)

    if len(samples_xy) == 0 or n_targets == 0:
        return estimates, variances

    # 1. Query k nearest neighbors using KDTree
    # A kriging system cannot have more conditioning samples than total available samples
    k_query = min(k_neighbors, len(samples_xy))
    tree = KDTree(samples_xy)
    upper_bound = max_radius if max_radius is not None else float("inf")
    distances, indices = tree.query(
        grid_points, k=k_query, distance_upper_bound=upper_bound
    )

    if k_query == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # 2. Solve Simple Kriging system for each target point
    for m in range(n_targets):
        # Filter to valid neighbors within upper_bound (excluding inf)
        valid_mask = np.isfinite(distances[m]) & (distances[m] <= upper_bound)
        if not np.any(valid_mask):
            continue  # No samples within range: remains prior mean and total sill

        d_m = distances[m][valid_mask]
        idx_m = indices[m][valid_mask]
        k_m = len(idx_m)

        # Exact collocation: if target lies directly on a sample point
        if d_m[0] < 1e-6:
            estimates[m] = sample_grades[idx_m[0]]
            variances[m] = 0.0
            continue

        # Build sample-to-sample covariance matrix K_m of shape (k_m, k_m)
        coords_m = samples_xy[idx_m]
        grades_m = sample_grades[idx_m]

        # Deduplicate identical sample coordinates (e.g. twin drillholes or duplicate assays)
        if k_m > 1:
            diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
            h_matrix = np.linalg.norm(diff_matrix, axis=2)
            np.fill_diagonal(h_matrix, np.inf)
            if np.any(h_matrix < 1e-6):
                unique_coords, inverse_indices = np.unique(
                    coords_m.round(decimals=5), axis=0, return_inverse=True
                )
                if len(unique_coords) < k_m:
                    new_grades = np.zeros(len(unique_coords), dtype=float)
                    for u_idx in range(len(unique_coords)):
                        new_grades[u_idx] = grades_m[inverse_indices == u_idx].mean()
                    coords_m = unique_coords
                    grades_m = new_grades
                    d_m = np.linalg.norm(coords_m - grid_points[m], axis=1)
                    k_m = len(coords_m)
                    diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
                    h_matrix = np.linalg.norm(diff_matrix, axis=2)
            np.fill_diagonal(h_matrix, 0.0)
        else:
            diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
            h_matrix = np.linalg.norm(diff_matrix, axis=2)

        K_m = _theoretical_covariance(
            h_matrix, variogram_model, nugget, sill, range_param
        )
        # Regularize diagonal to prevent singular matrices from collinear samples
        K_m[np.diag_indices(k_m)] += 1e-9

        # Build sample-to-target covariance vector k_0_m of shape (k_m,)
        k0_m = _theoretical_covariance(d_m, variogram_model, nugget, sill, range_param)

        # Solve linear system K_m * lambda_m = k0_m
        weights_m = np.linalg.solve(K_m, k0_m)

        # Simple Kriging estimate: Z*_SK = mean + sum_i lambda_i * (Z_i - mean)
        estimates[m] = mean + np.sum(weights_m * (grades_m - mean))

        # Simple Kriging variance: sigma_SK^2 = C(0) - sum_i lambda_i * k0_i
        var_val = total_sill - np.sum(weights_m * k0_m)
        variances[m] = max(0.0, float(var_val))

    # 3. Handle extrapolation masking
    if mask_extrapolation:
        inside_hull = is_within_convex_hull(samples_xy, grid_points)
        estimates[~inside_hull] = np.nan
        variances[~inside_hull] = np.nan

    return estimates, variances


def ordinary_kriging_grid_estimation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    sill: float,
    range_param: float,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    k_neighbors: int = 16,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary Kriging (OK) grid interpolation with unknown local mean.

    Estimates values at target grid points by solving the constrained Ordinary
    Kriging system with a Lagrange multiplier:
        [K   1] [lambda]   [k_0]
        [1^T 0] [  mu  ] = [ 1 ]
    Guarantees local unbiasedness (sum(lambda_i) = 1) without requiring a known
    global prior mean.

    Parameters
    ----------
    samples_xy : np.ndarray
        Sample coordinates of shape (N, 2) or (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    grid_points : np.ndarray
        Target estimation coordinates of shape (M, 2) or (M, 3).
    variogram_model : str, default "spherical"
        Theoretical variogram model ("spherical", "exponential", "gaussian").
    nugget : float, default 0.0
        Nugget variance c0.
    sill : float, default 1.0
        Partial sill variance c.
    range_param : float, default 100.0
        Spatial correlation range a.
    k_neighbors : int, default 16
        Maximum conditioning samples to query per target point.
    max_radius : float, optional
        Maximum search neighborhood radius.
    mask_extrapolation : bool, default False
        If True, masks blocks outside the drillhole convex hull to NaN.

    Returns
    -------
    estimated_grades : np.ndarray
        Ordinary Kriging grade estimates of shape (M,).
    kriging_variance : np.ndarray
        Estimation variance sigma_OK^2 of shape (M,).
    """
    # -------------------------------------------------------------------------
    # 1. Query k nearest neighbors for each grid point using KDTree(samples_xy).
    n_targets = len(grid_points)
    total_sill = nugget + sill

    # Initialize with NaN and maximum uncertainty (total sill)
    estimates = np.full(n_targets, np.nan, dtype=float)
    variances = np.full(n_targets, total_sill, dtype=float)

    if len(samples_xy) == 0 or n_targets == 0:
        return estimates, variances

    # 1. Query k nearest neighbors using KDTree
    # A kriging system cannot have more conditioning samples than total available samples
    k_query = min(k_neighbors, len(samples_xy))
    tree = KDTree(samples_xy)
    upper_bound = max_radius if max_radius is not None else float("inf")
    distances, indices = tree.query(
        grid_points, k=k_query, distance_upper_bound=upper_bound
    )

    if k_query == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # 2. For each target point m with k active neighbors:
    for m in range(n_targets):
        # Filter to valid neighbors within upper_bound (excluding inf)
        valid_mask = np.isfinite(distances[m]) & (distances[m] <= upper_bound)
        if not np.any(valid_mask):
            continue  # No samples within range: remains NaN and total sill

        d_m = distances[m][valid_mask]
        idx_m = indices[m][valid_mask]
        k_m = len(idx_m)

        # Exact collocation: if target lies directly on a sample point
        if d_m[0] < 1e-6:
            estimates[m] = sample_grades[idx_m[0]]
            variances[m] = 0.0
            continue

        # Build sample-to-sample covariance matrix K_m of shape (k_m, k_m)
        coords_m = samples_xy[idx_m]
        grades_m = sample_grades[idx_m]

        # Deduplicate identical sample coordinates (e.g. twin drillholes or duplicate assays)
        if k_m > 1:
            diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
            h_matrix = np.linalg.norm(diff_matrix, axis=2)
            np.fill_diagonal(h_matrix, np.inf)
            if np.any(h_matrix < 1e-6):
                unique_coords, inverse_indices = np.unique(
                    coords_m.round(decimals=5), axis=0, return_inverse=True
                )
                if len(unique_coords) < k_m:
                    new_grades = np.zeros(len(unique_coords), dtype=float)
                    for u_idx in range(len(unique_coords)):
                        new_grades[u_idx] = grades_m[inverse_indices == u_idx].mean()
                    coords_m = unique_coords
                    grades_m = new_grades
                    d_m = np.linalg.norm(coords_m - grid_points[m], axis=1)
                    k_m = len(coords_m)
                    diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
                    h_matrix = np.linalg.norm(diff_matrix, axis=2)
            np.fill_diagonal(h_matrix, 0.0)
        else:
            diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
            h_matrix = np.linalg.norm(diff_matrix, axis=2)

        K_m = _theoretical_covariance(
            h_matrix, variogram_model, nugget, sill, range_param
        )
        K_m[np.diag_indices(k_m)] += 1e-9  # Regularizer to guarantee invertibility

        # Build sample-to-target covariance vector k0_m of shape (k_m,)
        k0_m = _theoretical_covariance(d_m, variogram_model, nugget, sill, range_param)

        # Build augmented Ordinary Kriging matrix K_aug of shape (k_m + 1, k_m + 1):
        # [ K_m   1 ]
        # [ 1^T   0 ]
        K_aug = np.ones((k_m + 1, k_m + 1))
        K_aug[:k_m, :k_m] = K_m
        K_aug[k_m, k_m] = 0.0

        # Build augmented target vector k0_aug of shape (k_m + 1,):
        # [ k0_m ]
        # [  1   ]
        k0_aug = np.ones(k_m + 1)
        k0_aug[:k_m] = k0_m
        k0_aug[k_m] = 1.0

        # Solve linear system: K_aug * [lambda; mu_lagrange] = k0_aug
        solution = np.linalg.solve(K_aug, k0_aug)
        weights_m = solution[:k_m]
        mu_lagrange = solution[k_m]

        # Ordinary Kriging estimate: Z*_OK = sum_i lambda_i * Z(x_i)
        estimates[m] = np.sum(weights_m * grades_m)

        # Ordinary Kriging variance: sigma_OK^2 = C(0) - sum_i lambda_i * k0_i - mu_lagrange
        var_val = total_sill - np.sum(weights_m * k0_m) - mu_lagrange
        variances[m] = max(0.0, float(var_val))

    # 3. Handle extrapolation masking
    if mask_extrapolation:
        inside_hull = is_within_convex_hull(samples_xy, grid_points)
        estimates[~inside_hull] = np.nan
        variances[~inside_hull] = np.nan

    return estimates, variances


# =============================================================================
# 3D BLOCK MODELING & BLOCK KRIGING (SUPPORT EFFECT & DISCRETIZATION)
# =============================================================================


@dataclass
class SearchNeighborhood:
    """Defines 3D anisotropic search ellipsoid and drillhole sample constraints.

    Industry standard for Kriging and IDW search neighborhood control
    (Isaaks & Srivastava 1989; Armstrong 1998; SME Handbook Section 4.4).
    Prevents single-drillhole clustering bias and aligns search radii with
    geological strike, dip, and plunge.

    Attributes
    ----------
    radius_major : float
        Search radius along principal continuity axis (strike/plunge).
    radius_semi : float
        Search radius along semi-major axis (dip plane).
    radius_minor : float
        Search radius along minor axis (across-strike / thickness).
    azimuth : float, default 0.0
        Rotation angle around Z axis (strike / bearing in degrees: 0 = North, 90 = East).
    dip : float, default 0.0
        Dip angle in degrees below horizontal (-90 to +90).
    plunge : float, default 0.0
        Plunge angle along strike in degrees.
    min_samples : int, default 4
        Minimum number of samples required to inform an estimate.
    max_samples : int, default 16
        Maximum total samples used in estimation.
    max_per_hole : Optional[int], default None
        Maximum composites allowed from any single drillhole (prevents hole dominance).
    min_octants : int, default 1
        Minimum informed octants/quadrants required (ensures spatial support).
    max_per_octant : Optional[int], default None
        Maximum samples accepted per octant.
    """

    radius_major: float
    radius_semi: float
    radius_minor: float
    azimuth: float = 0.0
    dip: float = 0.0
    plunge: float = 0.0
    min_samples: int = 4
    max_samples: int = 16
    max_per_hole: Optional[int] = None
    min_octants: int = 1
    max_per_octant: Optional[int] = None


def create_block_model(
    origin: Tuple[float, float, float],
    block_size: Tuple[float, float, float],
    n_blocks: Tuple[int, int, int],
    default_density: float = 2.70,
    default_domain: str = "Default",
) -> pd.DataFrame:
    """Constructs a regular 3D Block Model DataFrame for resource estimation.

    Parameters
    ----------
    origin : tuple of (float, float, float)
        (x0, y0, z0) minimum coordinate origin (south-west-bottom corner of first block).
    block_size : tuple of (float, float, float)
        (dx, dy, dz) block dimensions in meters (SMU size).
    n_blocks : tuple of (int, int, int)
        (nx, ny, nz) number of blocks along X, Y, and Z axes.
    default_density : float, default 2.70
        Bulk density / specific gravity (t/m^3).
    default_domain : str, default "Default"
        Initial geological domain identifier.

    Returns
    -------
    pd.DataFrame
        DataFrame with block centroids (x, y, z), dimensions (dx, dy, dz),
        volume_m3, density, tonnes, and domain.
    """
    x0, y0, z0 = origin
    dx, dy, dz = block_size
    nx, ny, nz = n_blocks

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("Number of blocks in each dimension must be positive.")
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError("Block dimensions must be positive.")

    # Block centroids: origin + (i + 0.5) * d
    xc = x0 + (np.arange(nx) + 0.5) * dx
    yc = y0 + (np.arange(ny) + 0.5) * dy
    zc = z0 + (np.arange(nz) + 0.5) * dz

    # Meshgrid (X fast, Y medium, Z slow)
    grid_x, grid_y, grid_z = np.meshgrid(xc, yc, zc, indexing="ij")

    vol = float(dx * dy * dz)
    tonnes = vol * default_density

    df_blocks = pd.DataFrame(
        {
            "x": grid_x.ravel(),
            "y": grid_y.ravel(),
            "z": grid_z.ravel(),
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "volume_m3": vol,
            "density": default_density,
            "tonnes": tonnes,
            "domain": default_domain,
        }
    )
    df_blocks.attrs["origin"] = origin
    df_blocks.attrs["block_size"] = block_size
    df_blocks.attrs["n_blocks"] = n_blocks
    return df_blocks


def ordinary_kriging_block_estimation(
    samples_xyz: np.ndarray,
    sample_grades: np.ndarray,
    block_model: pd.DataFrame,
    sill: float,
    range_param: float,
    discretization: Tuple[int, int, int] = (4, 4, 2),
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    search_neighborhood: Optional[SearchNeighborhood] = None,
    domain_col: Optional[str] = None,
    sample_domain_col: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimates block grades using 3D Ordinary Block Kriging with internal discretization.

    Block Kriging (Journel & Huijbregts 1978; SME Handbook Section 4.5) accounts for the
    Support Effect by discretizing each mining block into an internal grid of
    (nx_disc * ny_disc * nz_disc) points. It calculates:
    1. Average sample-to-block covariance: C_bar(x_i, V)
    2. Block-to-block self-covariance: C_bar(V, V)
    3. Block dispersion variance: BV = C(0) - C_bar(V, V)

    Parameters
    ----------
    samples_xyz : np.ndarray
        Sample coordinates of shape (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and dimensions (dx, dy, dz).
    sill : float
        Partial sill variance.
    range_param : float
        Spatial correlation range.
    discretization : tuple of (int, int, int), default (4, 4, 2)
        Number of internal discretization points (nx_disc, ny_disc, nz_disc) per block.
    variogram_model : str, default "spherical"
        Variogram model ("spherical", "exponential", "gaussian").
    nugget : float, default 0.0
        Nugget variance.
    search_neighborhood : SearchNeighborhood, optional
        Anisotropic search ellipsoid and drillhole sharing constraints.
    domain_col : str, optional
        Geological domain column in block_model.
    sample_domain_col : str, optional
        Geological domain column for conditioning samples.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (block_estimates, block_variances, block_dispersion_variance).
    """
    # TODO: Implement 3D Block Kriging with internal point discretization (nx, ny, nz),
    # sample-to-block covariance C_bar(x_i, V), and block dispersion variance BV.
    raise NotImplementedError(
        "TODO: 3D Block Kriging with internal point discretization is planned. "
        "Use ordinary_kriging_grid_estimation() for 2D point interpolation."
    )


def plot_grade_tonnage_curve(
    gt_data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    title: str = "Grade–Tonnage Sensitivity Curve",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.5, 5.0),
    show_metal: bool = False,
) -> Union[Tuple[plt.Figure, plt.Axes], Dict[str, Tuple[plt.Figure, plt.Axes]]]:
    """Plots standard dual-axis Grade–Tonnage sensitivity curves.

    Conforms to NI 43-101 (Item 14) and JORC Table 1 mineral reporting standards:
    - Primary Y-axis (left): Ore Tonnage above cutoff (monotonically decreasing).
    - Secondary Y-axis (right): Average Grade above cutoff (monotonically increasing).
    - If a dictionary of models is provided, generates an individual, clean dual-axis
      plot for each model (avoiding multi-line overlapping clutter).

    Parameters
    ----------
    gt_data : pd.DataFrame or Mapping[str, pd.DataFrame]
        Output of `grade_tonnage_table` with cutoff index, or a dictionary mapping
        model names to their respective grade-tonnage DataFrames.
    grade_unit : str, default "% Cu"
        Unit label for cutoff and average grades.
    tonnage_unit : str, default "Mt"
        Unit for ore tonnage: "Mt" (1e6 tonnes), "kt" (1e3 tonnes), or "tonnes" (1:1).
    title : str, default "Grade–Tonnage Sensitivity Curve"
        Plot title.
    ax : plt.Axes, optional
        Primary Matplotlib Axes for the plot (only applicable when gt_data is a single DataFrame).
    figsize : Tuple[float, float], default (8.5, 5.0)
        Figure dimensions if created.
    show_metal : bool, default False
        If True, plots metal recovery percentage.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes] or Dict[str, Tuple[plt.Figure, plt.Axes]]
        If gt_data is a single DataFrame: (Figure, primary Axes).
        If gt_data is a dictionary: dict mapping model_name -> (Figure, primary Axes).
    """
    # Case A: Dictionary of Models -> Generate individual clean dual-axis plot per model
    if isinstance(gt_data, Mapping):
        figures = {}
        for model_name, df_m in gt_data.items():
            fig_m, ax_m = plot_grade_tonnage_curve(
                df_m,
                grade_unit=grade_unit,
                tonnage_unit=tonnage_unit,
                title=f"{title} - {model_name}",
                figsize=figsize,
                show_metal=show_metal,
            )
            figures[model_name] = (fig_m, ax_m)
        return figures

    # Case B: Single Model Dual-Axis Plot
    unit_divisors = {"mt": 1e6, "kt": 1e3, "tonnes": 1.0, "t": 1.0}
    t_divisor = unit_divisors.get(tonnage_unit.lower(), 1.0)

    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        ax1 = ax
        fig = ax1.figure

    ax2 = ax1.twinx()

    cutoffs = np.asarray(gt_data.index, dtype=float)
    tonnes = gt_data["ore_tonnes"].to_numpy() / t_divisor
    grades = gt_data["ore_grade"].to_numpy()

    line1 = ax1.plot(
        cutoffs,
        tonnes,
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label=f"Ore Tonnage ({tonnage_unit})",
    )
    line2 = ax2.plot(
        cutoffs,
        grades,
        color="#d62728",
        linewidth=2.5,
        marker="s",
        markersize=5,
        label=f"Average Grade ({grade_unit})",
    )

    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    if show_metal and "metal_recovery_pct" in gt_data.columns:
        metal_rec = gt_data["metal_recovery_pct"].to_numpy()
        line3 = ax1.plot(
            cutoffs,
            metal_rec * (tonnes.max() / 100.0) if tonnes.max() > 0 else metal_rec,
            color="#2ca02c",
            linewidth=1.8,
            linestyle="--",
            label="Metal Recovery (%)",
        )
        lines += line3
        labels.append("Metal Recovery (%)")

    ax1.set_xlabel(f"Cutoff Grade ({grade_unit})", fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        f"Ore Tonnage ({tonnage_unit})", fontsize=11, fontweight="bold", color="#1f77b4"
    )
    ax2.set_ylabel(
        f"Average Ore Grade ({grade_unit})",
        fontsize=11,
        fontweight="bold",
        color="#d62728",
    )
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(lines, labels, loc="center left", framealpha=0.9)

    ax1.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig, ax1


def cell_declustering(
    drillholes: pd.DataFrame,
    cell_sizes: Union[float, Sequence[float]],
    grade_col: str = "grade",
    x_col: str = "x",
    y_col: str = "y",
    n_offsets: int = 4,
    min_mean: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """Calculates spatial declustering weights and cell size sensitivity.

    In exploration drilling, geologists cluster drillholes preferentially in high-grade
    zones, causing naive sample averages to overestimate global deposit grade.
    Cell declustering (Journel 1983; Deutsch & Journel 1998; SME Handbook Section 4.3)
    superimposes a regular grid over the data and assigns weights inversely proportional
    to the number of samples sharing each cell.

    To eliminate cell boundary edge artifacts, the grid origin is shifted across
    `n_offsets` x `n_offsets` sub-grid increments and the resulting weights are averaged.

    Parameters
    ----------
    drillholes : pd.DataFrame
        Table of drillholes containing spatial coordinates and grades.
    cell_sizes : float or Sequence[float]
        Single cell dimension or sequence of cell dimensions (m) to test.
    grade_col : str, default "grade"
        Grade column name in drillholes.
    x_col : str, default "x"
        X coordinate column name.
    y_col : str, default "y"
        Y coordinate column name.
    n_offsets : int, default 4
        Number of origin offset increments per axis to average out boundary jumps.
    min_mean : bool, default True
        If True, selects the optimal cell size at the minimum declustered mean (standard
        for deposits drilled preferentially in high-grade ore). If False, selects maximum.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, float]
        - weights: 1D array of normalized sample weights for the optimal cell size (sum to 1.0).
        - sensitivity_df: DataFrame of columns ['cell_size', 'declustered_mean', 'declustered_variance'].
        - optimal_cell_size: The chosen cell dimension (float).
    """
    if len(drillholes) == 0:
        raise ValueError("Drillholes DataFrame cannot be empty.")

    x = np.asarray(drillholes[x_col], dtype=float)
    y = np.asarray(drillholes[y_col], dtype=float)
    z = np.asarray(drillholes[grade_col], dtype=float)
    n_samples = len(x)

    if isinstance(cell_sizes, (int, float)):
        cell_sizes_list = [float(cell_sizes)]
    else:
        cell_sizes_list = [float(cs) for cs in cell_sizes]

    n_offsets = max(1, int(n_offsets))
    total_shifts = n_offsets * n_offsets
    x_min, y_min = float(x.min()), float(y.min())

    weights_by_cell = []
    means = []
    variances = []

    for cs in cell_sizes_list:
        if cs <= 0:
            raise ValueError(f"Cell size must be strictly positive, got {cs}")

        weights_sum = np.zeros(n_samples, dtype=float)

        for ox in range(n_offsets):
            shift_x = x_min - (ox / n_offsets) * cs
            for oy in range(n_offsets):
                shift_y = y_min - (oy / n_offsets) * cs

                idx_x = np.floor((x - shift_x) / cs).astype(np.int64)
                idx_y = np.floor((y - shift_y) / cs).astype(np.int64)

                cell_keys = list(zip(idx_x, idx_y))
                counts = Counter(cell_keys)
                n_occupied = len(counts)

                for i, ck in enumerate(cell_keys):
                    weights_sum[i] += 1.0 / (n_occupied * counts[ck])

        w = weights_sum / total_shifts
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum

        mean_val = float(np.sum(w * z))
        var_val = float(np.sum(w * ((z - mean_val) ** 2)))

        weights_by_cell.append(w)
        means.append(mean_val)
        variances.append(var_val)

    sensitivity_df = pd.DataFrame(
        {
            "cell_size": cell_sizes_list,
            "declustered_mean": means,
            "declustered_variance": variances,
        }
    )

    if min_mean:
        opt_idx = int(sensitivity_df["declustered_mean"].idxmin())
    else:
        opt_idx = int(sensitivity_df["declustered_mean"].idxmax())

    optimal_cell_size = float(sensitivity_df.loc[opt_idx, "cell_size"])
    optimal_weights = weights_by_cell[opt_idx]

    return optimal_weights, sensitivity_df, optimal_cell_size


def plot_cell_declustering_curve(
    sensitivity_df: pd.DataFrame,
    naive_mean: float,
    optimal_cell_size: Optional[float] = None,
    grade_unit: str = "% Cu",
    title: str = "Cell Declustering Sensitivity Curve",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.5, 4.8),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the Declustered Mean vs. Cell Size optimization curve.

    Standard NI 43-101 and JORC reporting chart justifying the selected
    declustering cell size and displaying the removal of high-grade spatial bias.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        DataFrame with 'cell_size' and 'declustered_mean' columns from cell_declustering.
    naive_mean : float
        Unweighted arithmetic mean of sample assays for reference.
    optimal_cell_size : float, optional
        Selected optimal cell dimension. If None, chosen at minimum mean.
    grade_unit : str, default "% Cu"
        Unit label for mineral grade.
    title : str, default "Cell Declustering Sensitivity Curve"
        Plot title.
    ax : plt.Axes, optional
        Primary Matplotlib Axes if embedding in existing figure.
    figsize : Tuple[float, float], default (8.5, 4.8)
        Dimensions of figure if created.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib Figure and primary Axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    cell_sizes = sensitivity_df["cell_size"].to_numpy()
    means = sensitivity_df["declustered_mean"].to_numpy()

    if optimal_cell_size is None:
        opt_idx = int(sensitivity_df["declustered_mean"].idxmin())
        optimal_cell_size = float(cell_sizes[opt_idx])
        optimal_mean = float(means[opt_idx])
    else:
        opt_matches = sensitivity_df[
            np.isclose(sensitivity_df["cell_size"], optimal_cell_size)
        ]
        if len(opt_matches) > 0:
            optimal_mean = float(opt_matches["declustered_mean"].iloc[0])
        else:
            optimal_mean = float(np.interp(optimal_cell_size, cell_sizes, means))

    ax.plot(
        cell_sizes,
        means,
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Declustered Mean",
    )

    ax.axhline(
        naive_mean,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Naive Sample Mean ({naive_mean:.3f}{grade_unit})",
    )

    ax.axvline(
        optimal_cell_size,
        color="#2ca02c",
        linestyle=":",
        linewidth=2.0,
        label=f"Selected Cell Size ({optimal_cell_size:.1f} m)",
    )

    ax.scatter(
        [optimal_cell_size],
        [optimal_mean],
        color="#2ca02c",
        s=140,
        marker="*",
        zorder=6,
        label=f"Optimal Mean ({optimal_mean:.3f}{grade_unit})",
    )

    bias_pct = (
        ((naive_mean - optimal_mean) / naive_mean) * 100.0 if naive_mean != 0 else 0.0
    )
    ax.annotate(
        f"Selected: {optimal_cell_size:.1f}m\nBias Removed: {bias_pct:+.1f}%",
        xy=(optimal_cell_size, optimal_mean),
        xytext=(15, -20),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#eafaf1", ec="#2ca02c", lw=1.2),
    )

    ax.set_xlabel("Declustering Cell Size (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        f"Declustered Mean Grade ({grade_unit})", fontsize=11, fontweight="bold"
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()

    return fig, ax


def kriging_quality_metrics(
    kriging_variances: np.ndarray,
    block_dispersion_variance: float,
    lagrange_multipliers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calculates Kriging Efficiency (KE) and Slope of Regression (SoR) for Block Kriging.

    Standard conditional bias diagnostics introduced by Danie Krige (1996) and
    Isobel Clark (1983) for Kriging Neighborhood Analysis (KNA), widely used under
    JORC and SAMREC to optimize search parameters and assess geological confidence:

    1. Kriging Efficiency (KE):
       KE = (BV - sigma_OK^2) / BV
       Measures the relative percentage error reduction of the block estimate compared
       to the block dispersion variance (BV = sigma^2(V|D)).

    2. Slope of Regression (SoR):
       SoR = (BV - sigma_OK^2 + |mu|) / (BV - sigma_OK^2 + 2 * |mu|)
       Estimates the linear regression slope of true block grades on kriged estimates.
       SoR >= 0.8 is often required for Measured Resources; SoR >= 0.5 for Indicated.

    Theoretical Support Constraint (Krige 1996; SME Handbook Section 4.5):
    ----------------------------------------------------------------------
    These metrics evaluate conditional bias for mining blocks of finite volume (V)
    informed by drill core samples of point volume (v). They are strictly invalid for
    Point Kriging (V -> 0), because:
    - Point variance C(0) overstates block dispersion variance BV, distorting KE to negative
      or physically meaningless values.
    - Exact point collocation (sigma^2 = 0) falsely implies 100% confidence for an entire block.
    Therefore, this diagnostic should only be evaluated on Block Kriging estimates
    incorporating block discretization.

    Parameters
    ----------
    kriging_variances : np.ndarray
        Array of block estimation variances (sigma_OK,block^2) from Block Kriging.
    block_dispersion_variance : float
        Block dispersion variance BV = C_bar(D, D) - C_bar(V, V), representing the true
        variance of mining blocks of volume V within the deposit D.
    lagrange_multipliers : np.ndarray, optional
        Array of Lagrange multipliers (mu) from the Ordinary Block Kriging system.

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (kriging_efficiency, slope_of_regression). If lagrange_multipliers is None,
        slope_of_regression is None.
    """
    # TODO: Defer implementation until Block Kriging (with block discretization and
    # block dispersion variance BV = C_bar(D,D) - C_bar(V,V)) is implemented.
    # I want to implement the core functionality myself when block support is introduced.
    raise NotImplementedError(
        "TODO: Implement Kriging Efficiency (KE) and Slope of Regression (SoR) "
        "alongside Block Kriging (requires block dispersion variance BV)."
    )


def classify_mineral_resources(
    grid_points: np.ndarray,
    samples_xy: np.ndarray,
    kriging_variances: Optional[np.ndarray] = None,
    max_radius_measured: float = 30.0,
    max_radius_indicated: float = 60.0,
    min_holes_measured: int = 3,
    min_holes_indicated: int = 2,
    variance_threshold_measured: Optional[float] = None,
    variance_threshold_indicated: Optional[float] = None,
) -> np.ndarray:
    """Classifies mineral resources into Measured, Indicated, and Inferred.

    CIM Definition Standards (2014) and JORC Code (Clause 20-24) Regulatory Framework:
    ----------------------------------------------------------------------------------
    Mineral Resource classification categorizes estimates based on geological confidence:
    - Measured: High geological confidence; dense drilling; verified continuous geology/grades.
    - Indicated: Reasonable confidence; sufficient drilling to assume geological/grade continuity.
    - Inferred: Low confidence; limited geological evidence; continuity implied but not verified.

    The "Spotted Dog" Pitfall & Why Automated Kriging Variance Cutoffs are Discouraged:
    ----------------------------------------------------------------------------------
    Classifying blocks purely on automated block-by-block mathematical criteria (such as
    raw kriging variance cutoffs or raw nearest-hole counts) is heavily scrutinized by
    regulators (CIM Best Practice Guidelines; JORC Table 1; Stephenson et al.):
    1. Concentric Bullseyes / Spotted Dog: Kriging variance forms concentric rings around
       drillholes, producing isolated "speckles" of Measured blocks surrounded by Indicated,
       or single Inferred holes inside Measured zones that cannot be practically mined.
    2. Data Geometry vs. Geological Confidence: Kriging variance reflects sample geometry and
       the variogram, but conveys zero information about structural faulting, lithology,
       core recovery, or QA/QC assay integrity.

    Industry Best Practice:
    -----------------------
    Geometric parameters (drill spacing, hole counts) serve only as an initial guide.
    The Qualified Person (QP) constructs smoothed 3D continuous wireframe classification
    envelopes (or applies morphological majority smoothing) to delineate contiguous,
    practical mining zones alongside qualitative geological signoff.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of block centers of shape (M, 2) or (M, 3).
    samples_xy : np.ndarray
        Array of informing drillhole coordinates of shape (N, 2) or (N, 3).
    kriging_variances : np.ndarray, optional
        Array of Kriging estimation variances of shape (M,).
    max_radius_measured : float, default 30.0
        Nominal drillhole spacing radius for Measured classification.
    max_radius_indicated : float, default 60.0
        Nominal drillhole spacing radius for Indicated classification.
    min_holes_measured : int, default 3
        Minimum informing drillholes within search radius for Measured.
    min_holes_indicated : int, default 2
        Minimum informing drillholes within search radius for Indicated.
    variance_threshold_measured : float, optional
        Maximum Kriging variance allowed for Measured classification.
    variance_threshold_indicated : float, optional
        Maximum Kriging variance allowed for Indicated classification.

    Returns
    -------
    np.ndarray
        Array of category strings ("Measured", "Indicated", "Inferred", or "Unclassified") of shape (M,).
    """
    # TODO: Defer implementation until 3D continuous wireframe envelopes or morphological
    # majority smoothing are implemented to avoid the "spotted dog" artifact.
    # Raw block-by-block variance cutoffs are non-compliant under CIM / JORC guidelines.
    # I want to implement the core functionality myself when spatial domaining is introduced.
    raise NotImplementedError(
        "TODO: Implement Resource Classification using continuous spatial envelopes "
        "or morphological smoothing (deferred to prevent the 'spotted dog' artifact)."
    )


def _round_sig_figs(value: float, sig_figs: int) -> float:
    """Rounds a numeric value to a specified number of significant figures."""
    if value == 0.0 or np.isnan(value) or np.isinf(value):
        return 0.0
    return float(round(value, -int(np.floor(np.log10(abs(value)))) + (sig_figs - 1)))


def format_resource_statement(
    block_df: pd.DataFrame,
    category_col: str = "category",
    grade_col: str = "grade",
    tonnes_col: str = "tonnes",
    cutoff_grade: float = 0.0,
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    metal_unit: str = "kt",
    metal_factor: float = 0.01,
    commodity_price: Optional[str] = None,
    metallurgical_recovery: Optional[float] = None,
    rpeee_constraint: str = "Constrained within an optimized pit shell",
) -> pd.DataFrame:
    """Formats an official Mineral Resource Statement adhering to significant figures rules.

    In accordance with CIM Definition Standards (2014), JORC (Clause 25), and
    SEC S-K 1300 standards:
    - Avoid False Precision: Tonnages, grades, and contained metal must be rounded
      to reflect the relative uncertainty of each classification tier:
        * Measured: 3 significant figures (or nearest deposit-scale precision)
        * Indicated: 2-3 significant figures
        * Measured + Indicated (M&I): Reported as subtotal (raw sum rounded to 3 sig figs)
        * Inferred: 1-2 significant figures, reported strictly separately from M&I.
    - Mandatory Footnote: Clarifies rounding and non-additivity.
    - RPEEE Condition: Requires reporting base-case economic cut-off, commodity price,
      metallurgical recovery, and spatial constraint (pit shell or stope shapes).

    Parameters
    ----------
    block_df : pd.DataFrame
        Block model DataFrame with category, grade, and tonnage columns.
    category_col : str, default "category"
        Column containing classification ("Measured", "Indicated", "Inferred").
    grade_col : str, default "grade"
        Column containing block grades.
    tonnes_col : str, default "tonnes"
        Column containing block tonnages.
    cutoff_grade : float, default 0.0
        Economic cutoff grade to apply (blocks below this are excluded).
    grade_unit : str, default "% Cu"
        Unit of the mineral grade.
    tonnage_unit : str, default "Mt"
        Reporting unit for ore tonnage ("Mt" or "kt").
    metal_unit : str, default "kt"
        Reporting unit for contained metal ("kt", "t", "koz").
    metal_factor : float, default 0.01
        Multiplier to convert (grade * tonnes) to raw metal tonnes. Default 0.01 for %.
    commodity_price : str, optional
        Economic commodity price assumption (e.g. "$3.80/lb Cu").
    metallurgical_recovery : float, optional
        Assumed metallurgical recovery percentage (e.g. 88.0%).
    rpeee_constraint : str, default "Constrained within an optimized pit shell"
        Spatial constraint applied to demonstrate RPEEE.

    Returns
    -------
    pd.DataFrame
        Formatted summary table compliant with international reporting disclosure,
        with metadata and mandatory footnotes accessible via `df.attrs["footnotes"]`.
    """
    if block_df.empty:
        raise ValueError("block_df cannot be empty.")

    # Filter by economic cutoff
    valid_blocks = block_df[block_df[grade_col] >= cutoff_grade].copy()

    # Define unit scaling factors
    t_scale = 1e6 if tonnage_unit.upper() == "MT" else 1e3
    m_scale = (
        1e3
        if metal_unit.lower() == "kt"
        else (1.0 if metal_unit.lower() == "t" else 31103.5)
    )

    # Standard category groupings
    cat_names = {
        "measured": "Measured",
        "indicated": "Indicated",
        "inferred": "Inferred",
    }

    # Configuration of sig figs per category [tonnage_sigfigs, grade_sigfigs, metal_sigfigs]
    sigfig_rules = {
        "Measured": (3, 3, 3),
        "Indicated": (2, 2, 2),
        "Measured + Indicated": (3, 3, 3),
        "Inferred": (2, 2, 2),
    }

    # Extract raw data by standardized category
    raw_stats: Dict[str, Dict[str, float]] = {}
    standardized_cats = valid_blocks[category_col].astype(str).str.strip().str.lower()

    for key, formal_name in cat_names.items():
        sub = valid_blocks[standardized_cats == key]
        if len(sub) > 0:
            tonnes = float(sub[tonnes_col].sum())
            metal_raw = float((sub[tonnes_col] * sub[grade_col] * metal_factor).sum())
            grade = float(metal_raw / (tonnes * metal_factor)) if tonnes > 0 else 0.0
            raw_stats[formal_name] = {
                "tonnes": tonnes,
                "grade": grade,
                "metal": metal_raw,
            }
        else:
            raw_stats[formal_name] = {"tonnes": 0.0, "grade": 0.0, "metal": 0.0}

    # Calculate raw Measured + Indicated subtotal
    mi_tonnes = raw_stats["Measured"]["tonnes"] + raw_stats["Indicated"]["tonnes"]
    mi_metal = raw_stats["Measured"]["metal"] + raw_stats["Indicated"]["metal"]
    mi_grade = float(mi_metal / (mi_tonnes * metal_factor)) if mi_tonnes > 0 else 0.0
    raw_stats["Measured + Indicated"] = {
        "tonnes": mi_tonnes,
        "grade": mi_grade,
        "metal": mi_metal,
    }

    # Assemble formatted rows adhering to significant figures
    rows = []
    display_order = ["Measured", "Indicated", "Measured + Indicated", "Inferred"]

    for cat in display_order:
        stats = raw_stats[cat]
        t_sf, g_sf, m_sf = sigfig_rules[cat]

        t_val = stats["tonnes"] / t_scale
        g_val = stats["grade"]
        m_val = stats["metal"] / m_scale

        t_rnd = _round_sig_figs(t_val, t_sf)
        g_rnd = _round_sig_figs(g_val, g_sf)
        m_rnd = _round_sig_figs(m_val, m_sf)

        rows.append(
            {
                "Classification": cat,
                f"Cutoff ({grade_unit})": f"{cutoff_grade:.2f}",
                f"Tonnage ({tonnage_unit})": f"{t_rnd:,.3g}",
                f"Grade ({grade_unit})": f"{g_rnd:.3g}",
                f"Contained Metal ({metal_unit})": f"{m_rnd:,.3g}",
            }
        )

    statement_df = pd.DataFrame(rows)

    # Mandatory Disclosure Footnotes
    footnotes = [
        "1. Mineral Resources are reported in accordance with CIM Definition Standards (2014) / JORC Code (2012).",
        "2. Tonnages, grades, and contained metal are rounded to reflect relative estimation uncertainty. Totals may not sum due to rounding.",
        "3. Mineral Resources that are not Mineral Reserves do not have demonstrated economic viability.",
        "4. Inferred Mineral Resources have greater uncertainty and cannot be combined with Measured and Indicated Resources.",
        f"5. Reasonable Prospects for Eventual Economic Extraction (RPEEE): {rpeee_constraint}, "
        f"Cutoff Grade = {cutoff_grade:.2f}{grade_unit}"
        + (f", Commodity Price = {commodity_price}" if commodity_price else "")
        + (
            f", Metallurgical Recovery = {metallurgical_recovery:.1f}%"
            if metallurgical_recovery is not None
            else ""
        )
        + ".",
    ]
    statement_df.attrs["footnotes"] = footnotes

    return statement_df


def plot_swath_analysis(
    block_model: pd.DataFrame,
    drillholes: Optional[pd.DataFrame] = None,
    axis: str = "x",
    bin_width: float = 50.0,
    grade_col: str = "grade",
    validation_grade_col: Optional[str] = None,
    drillhole_grade_col: str = "grade",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    tonnes_col: Optional[str] = "tonnes",
    model_name: str = "Ordinary Kriging",
    validation_model_name: str = "Nearest Neighbor (Declustered Proxy)",
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10.0, 5.2),
) -> Tuple[plt.Figure, plt.Axes]:
    """Generates a directional Swath Plot (drift analysis) for local bias validation.

    Swath plots are the industry-wide validation tool mandated under NI 43-101 and JORC.
    They slice the deposit along major Cartesian corridors (Easting, Northing, or Elevation),
    comparing the estimated model against a declustered validation model (e.g., Nearest
    Neighbor / Polygonal) and raw drillhole composites, overlaid with slice tonnages on
    a secondary Y-axis to assess local unbiasedness across data-dense and sparse zones.

    Important Geostatistical Standard:
    ---------------------------------
    Directly comparing block estimates to naive drillhole composite averages can reveal
    apparent local bias because drillholes are preferentially clustered in high-grade
    sweet spots. Industry best practice compares the estimate against a declustered
    block proxy (e.g., Nearest Neighbor block grades) or block-averaged composites.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of block model estimates with coordinates and estimated grade.
    drillholes : pd.DataFrame, optional
        Table of drillhole composites with coordinates and assay grades.
    axis : str, default "x"
        Direction of slicing: "x" / "easting", "y" / "northing", or "z" / "elevation".
    bin_width : float, default 50.0
        Width of spatial swath corridor in meters.
    grade_col : str, default "grade"
        Column name in `block_model` for the primary model estimate.
    validation_grade_col : str, optional
        Column name in `block_model` for the declustered validation check (e.g. NN grade).
    drillhole_grade_col : str, default "grade"
        Grade column name in `drillholes`.
    x_col : str, default "x"
        Column name for X coordinate (Easting).
    y_col : str, default "y"
        Column name for Y coordinate (Northing).
    z_col : str, default "z"
        Column name for Z coordinate (Elevation).
    tonnes_col : str, optional, default "tonnes"
        Column name for block tonnage in `block_model`.
    model_name : str, default "Ordinary Kriging"
        Label for the primary estimator being validated.
    validation_model_name : str, default "Nearest Neighbor (Declustered Proxy)"
        Label for the declustered benchmark model.
    grade_unit : str, default "% Cu"
        Unit label for mineral grade.
    tonnage_unit : str, default "Mt"
        Unit label for slice ore tonnage.
    title : str, optional
        Plot title. If None, automatically generated.
    ax : plt.Axes, optional
        Primary Matplotlib Axes if embedding in an existing layout.
    figsize : Tuple[float, float], default (10.0, 5.2)
        Figure dimensions if a new figure is created.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Matplotlib Figure and primary Axes.
    """
    if block_model.empty:
        raise ValueError("block_model cannot be empty.")

    # Determine slice axis coordinate column and descriptive label
    ax_lower = axis.strip().lower()
    if ax_lower in ("x", "easting", "east", "e"):
        coord_col = x_col
        axis_label = "Easting (X, meters)"
        default_dir = "Easting"
    elif ax_lower in ("y", "northing", "north", "n"):
        coord_col = y_col
        axis_label = "Northing (Y, meters)"
        default_dir = "Northing"
    elif ax_lower in ("z", "elevation", "elev", "rl"):
        coord_col = z_col
        axis_label = "Elevation (Z, meters)"
        default_dir = "Elevation"
    else:
        raise ValueError(f"Unknown swath axis '{axis}'. Choose 'x', 'y', or 'z'.")

    # Establish swath bins
    valid_blocks = block_model.dropna(subset=[coord_col, grade_col]).copy()
    if valid_blocks.empty:
        raise ValueError(
            f"No valid blocks with coordinates in '{coord_col}' and '{grade_col}'."
        )

    min_coord = float(valid_blocks[coord_col].min())
    max_coord = float(valid_blocks[coord_col].max())

    bins = np.arange(min_coord, max_coord + bin_width, bin_width)
    if len(bins) < 2:
        bins = np.array([min_coord, min_coord + bin_width])

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    n_bins = len(bin_centers)

    t_scale = 1e6 if tonnage_unit.upper() == "MT" else 1e3

    # Aggregate block model slices
    est_grades = []
    val_grades = []
    slice_tonnes = []
    valid_centers = []

    has_val = (
        validation_grade_col is not None
        and validation_grade_col in valid_blocks.columns
    )
    has_tonnes = tonnes_col is not None and tonnes_col in valid_blocks.columns

    for k in range(n_bins):
        low, high = bins[k], bins[k + 1]
        in_bin = valid_blocks[
            (valid_blocks[coord_col] >= low) & (valid_blocks[coord_col] < high)
        ]
        if len(in_bin) == 0:
            continue

        valid_centers.append(bin_centers[k])

        if has_tonnes:
            t = in_bin[tonnes_col].to_numpy()
            total_t = float(t.sum())
            slice_tonnes.append(total_t / t_scale)
            if total_t > 0:
                est_g = float((in_bin[grade_col] * t).sum() / total_t)
                est_grades.append(est_g)
                if has_val:
                    val_g = float((in_bin[validation_grade_col] * t).sum() / total_t)
                    val_grades.append(val_g)
            else:
                est_grades.append(float(in_bin[grade_col].mean()))
                if has_val:
                    val_grades.append(float(in_bin[validation_grade_col].mean()))
        else:
            slice_tonnes.append(len(in_bin))
            est_grades.append(float(in_bin[grade_col].mean()))
            if has_val:
                val_grades.append(float(in_bin[validation_grade_col].mean()))

    # Aggregate drillholes slices if provided
    dh_centers = []
    dh_grades = []
    if (
        drillholes is not None
        and not drillholes.empty
        and coord_col in drillholes.columns
    ):
        valid_dh = drillholes.dropna(subset=[coord_col, drillhole_grade_col])
        for k in range(n_bins):
            low, high = bins[k], bins[k + 1]
            in_bin_dh = valid_dh[
                (valid_dh[coord_col] >= low) & (valid_dh[coord_col] < high)
            ]
            if len(in_bin_dh) > 0:
                dh_centers.append(bin_centers[k])
                dh_grades.append(float(in_bin_dh[drillhole_grade_col].mean()))

    # Initialize plot
    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        fig = ax1.figure
        ax1 = ax

    ax2 = ax1.twinx()

    # Plot slice volume / tonnage bars on secondary axis
    bar_width = bin_width * 0.75
    tonnage_label = f"Slice Tonnage ({tonnage_unit})" if has_tonnes else "Block Count"
    bars = ax2.bar(
        valid_centers,
        slice_tonnes,
        width=bar_width,
        color="#c6dbef",
        edgecolor="#9ecae1",
        alpha=0.6,
        label=tonnage_label,
        zorder=1,
    )
    ax2.set_ylabel(tonnage_label, fontsize=10, fontweight="bold", color="#6baed6")
    ax2.tick_params(axis="y", labelcolor="#6baed6")
    ax2.grid(False)

    lines = []
    labels = []

    # Plot primary model estimated grade
    line_est = ax1.plot(
        valid_centers,
        est_grades,
        color="#1f77b4",
        linewidth=2.4,
        marker="o",
        markersize=6,
        label=f"{model_name} (Estimate)",
        zorder=4,
    )
    lines += line_est
    labels.append(f"{model_name} (Estimate)")

    # Plot validation model check line
    if has_val and len(val_grades) == len(valid_centers):
        line_val = ax1.plot(
            valid_centers,
            val_grades,
            color="#ff7f0e",
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=5,
            label=validation_model_name,
            zorder=5,
        )
        lines += line_val
        labels.append(validation_model_name)

    # Plot raw drillhole composite average
    if len(dh_grades) > 0:
        line_dh = ax1.plot(
            dh_centers,
            dh_grades,
            color="#2ca02c",
            linewidth=1.8,
            linestyle=":",
            marker="^",
            markersize=6,
            label="Drillhole Composites",
            zorder=6,
        )
        lines += line_dh
        labels.append("Drillhole Composites")

    # Combine legends from both axes
    lines.append(bars)
    labels.append(tonnage_label)

    ax1.set_xlabel(axis_label, fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        f"Average Grade ({grade_unit})", fontsize=11, fontweight="bold", color="#1f77b4"
    )
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plot_title = (
        title if title else f"Swath Plot (Local Drift Analysis) Along {default_dir}"
    )
    ax1.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    return fig, ax1


# =============================================================================
# RESOURCE TO RESERVE DELINEATION (CIM / NI 43-101 / JORC MODIFYING FACTORS)
# =============================================================================


def calculate_cut_off_grade(
    processing_cost: float,
    ga_cost: float,
    commodity_price: float,
    metallurgical_recovery: float,
    mining_cost: Optional[float] = None,
    selling_cost: float = 0.0,
    royalty_pct: float = 0.0,
    metal_conversion_factor: float = 1.0,
) -> float:
    """Calculates engineering cut-off grades with zero assumed default costs or prices.

    In mining economics (Lane, 1988; Taylor, 1972):
    - Breakeven Cut-Off Grade: Covers total costs (mining + processing + G&A + selling).
      Used to delineate the ultimate economic pit limit / stope envelope.
    - Marginal / Internal Cut-Off Grade: Covers processing + G&A + selling (mining cost
      is treated as sunk because the rock must be excavated anyway to access deeper ore).

    Parameters
    ----------
    processing_cost : float
        Processing / milling cost per tonne of ore ($/t ore). Required.
    ga_cost : float
        General & Administrative (G&A) overhead cost per tonne of ore ($/t ore). Required.
    commodity_price : float
        Base commodity price per unit metal ($/lb, $/oz, $/t). Required.
    metallurgical_recovery : float
        Plant metallurgical recovery percentage (0 < recovery <= 100.0) or fraction
        (0 < recovery <= 1.0). Required.
    mining_cost : float, optional
        Mining cost per tonne of rock ($/t rock).
        If provided: calculates Breakeven Cut-Off Grade.
        If None: calculates Marginal / Internal Cut-Off Grade.
    selling_cost : float, default 0.0
        Refining, smelting, freight, and realization deduction per unit metal ($/unit).
    royalty_pct : float, default 0.0
        Net Smelter Return (NSR) or gross revenue royalty percentage (e.g., 2.0 for 2%).
    metal_conversion_factor : float, default 1.0
        Multiplier to convert grade unit into pricing unit:
        - Copper (% Cu grade to $/lb price): 1% Cu = 22.0462 lbs Cu per metric tonne -> 22.0462
        - Gold (g/t Au grade to $/oz price): 1 g/t = 1/31.1035 oz/t -> 0.0321507
        - Base metals in $/tonne metal with % grade: 1% = 0.01 tonnes metal -> 0.01

    Returns
    -------
    float
        Economic cut-off grade in the corresponding grade unit.
    """
    if processing_cost < 0 or ga_cost < 0:
        raise ValueError("Operating costs (processing, G&A) cannot be negative.")
    if commodity_price <= 0:
        raise ValueError("Commodity price must be strictly positive.")

    # Normalize metallurgical recovery
    # TODO: somewhat dangerous here.
    rec = (
        metallurgical_recovery / 100.0
        if metallurgical_recovery > 1.0
        else metallurgical_recovery
    )
    if rec <= 0.0 or rec > 1.0:
        raise ValueError(
            f"Metallurgical recovery must be in (0, 100]%, got {metallurgical_recovery}"
        )

    # Net realized revenue per unit metal after deductions and royalties
    royalty_factor = max(0.0, 1.0 - (royalty_pct / 100.0))
    net_price = (commodity_price - selling_cost) * royalty_factor
    if net_price <= 0:
        raise ValueError(
            f"Net price ({net_price:.4f}) is non-positive after selling deductions and royalties."
        )

    revenue_per_grade_unit = net_price * rec * metal_conversion_factor

    total_ore_cost = processing_cost + ga_cost
    if mining_cost is not None:
        if mining_cost < 0:
            raise ValueError("Mining cost cannot be negative.")
        total_cost = total_ore_cost + mining_cost
    else:
        total_cost = total_ore_cost

    return float(total_cost / revenue_per_grade_unit)


def convert_resource_to_reserve(
    resource_df: pd.DataFrame,
    mining_dilution_pct: float,
    mining_recovery_pct: float,
    cutoff_grade: float,
    dilution_grade: float = 0.0,
    category_col: str = "category",
    grade_col: str = "grade",
    tonnes_col: str = "tonnes",
    allow_inferred: bool = False,
) -> pd.DataFrame:
    """Converts classified Mineral Resources into Mineral Reserves applying Modifying Factors.

    Under CIM Definition Standards (2014), JORC Code (2012), and SEC S-K 1300:
    1. 'Measured' Mineral Resources convert into 'Proven' (or 'Proved') Reserves.
    2. 'Indicated' Mineral Resources convert into 'Probable' Reserves.
    3. 'Inferred' Mineral Resources CANNOT be converted into Mineral Reserves under
       any circumstances due to low geological confidence. They are excluded by default.
    4. Applies Mining Dilution: waste rock inadvertently blasted and hauled with ore,
       increasing run-of-mine tonnage and lowering delivered head grade.
    5. Applies Mining Recovery (Ore Loss): unrecovered ore due to blast scatter or
       stability pillars, reducing delivered tonnage.

    Parameters
    ----------
    resource_df : pd.DataFrame
        Classified block model or polygon table containing classification, grade, and tonnage.
    mining_dilution_pct : float
        Mining dilution percentage (e.g., 5.0 for 5% dilution). Required without default.
    mining_recovery_pct : float
        Mining extraction recovery percentage (e.g., 95.0 for 95% recovery, meaning 5% ore loss).
        Required without default.
    cutoff_grade : float
        Economic cut-off grade for reserve delineation. Material below cutoff is excluded.
        Required without default.
    dilution_grade : float, default 0.0
        Grade of the diluting waste or contact rock.
    category_col : str, default "category"
        Column containing resource classifications ('Measured', 'Indicated', 'Inferred').
    grade_col : str, default "grade"
        In-situ grade column name.
    tonnes_col : str, default "tonnes"
        In-situ tonnage column name.
    allow_inferred : bool, default False
        Strict compliance enforcement. If False (default), Inferred material is strictly
        excluded from reserves in compliance with international reporting codes.

    Returns
    -------
    pd.DataFrame
        Run-of-Mine (ROM) Mineral Reserve DataFrame with columns:
        - 'reserve_category': 'Proven Reserve' or 'Probable Reserve'
        - 'rom_tonnes': Diluted and recovered ore tonnage delivered to plant
        - 'rom_grade': Diluted head grade delivered to plant
        - 'contained_metal': Contained metal after dilution and recovery
        - 'in_situ_tonnes', 'in_situ_grade': Original resource values before modifying factors
        - Spatial coordinate columns ('x', 'y', 'z') if present in resource_df.
    """
    for col in (category_col, grade_col, tonnes_col):
        if col not in resource_df.columns:
            raise ValueError(
                f"Required column '{col}' not found in resource DataFrame."
            )

    if mining_dilution_pct < 0.0:
        raise ValueError(
            f"Mining dilution cannot be negative, got {mining_dilution_pct}%"
        )
    if mining_recovery_pct <= 0.0 or mining_recovery_pct > 100.0:
        raise ValueError(
            f"Mining recovery must be in (0, 100]%, got {mining_recovery_pct}%"
        )

    # Normalize category strings for robust matching
    cat_series = resource_df[category_col].astype(str).str.strip().str.capitalize()

    # Segregate and audit Inferred material
    is_inferred = cat_series.str.startswith("Infer")
    n_inferred = int(is_inferred.sum())
    inferred_tonnes = float(resource_df.loc[is_inferred, tonnes_col].sum())

    if n_inferred > 0 and not allow_inferred:
        # Strictly excluded from reserves
        eligible_mask = ~is_inferred
    else:
        eligible_mask = np.ones(len(resource_df), dtype=bool)

    # Filter by economic cut-off grade
    above_cutoff = eligible_mask & (resource_df[grade_col] >= cutoff_grade)
    res_subset = resource_df.loc[above_cutoff].copy()

    if len(res_subset) == 0:
        empty_df = pd.DataFrame(
            columns=[
                "reserve_category",
                "rom_tonnes",
                "rom_grade",
                "contained_metal",
                "in_situ_tonnes",
                "in_situ_grade",
            ]
        )
        empty_df.attrs["excluded_inferred_tonnes"] = inferred_tonnes
        return empty_df

    # Map categories to reserve classifications
    subset_cats = cat_series.loc[above_cutoff]
    reserve_cats = np.empty(len(res_subset), dtype=object)

    is_meas = subset_cats.str.startswith("Meas")
    is_ind = subset_cats.str.startswith("Ind")

    reserve_cats[is_meas.to_numpy()] = "Proven Reserve"
    reserve_cats[is_ind.to_numpy()] = "Probable Reserve"

    # Any remaining (e.g. Inferred if allow_inferred was true, or unclassified)
    unmapped = ~(is_meas | is_ind)
    if unmapped.any():
        reserve_cats[unmapped.to_numpy()] = "Probable Reserve"

    # In-situ values
    t_insitu = res_subset[tonnes_col].to_numpy(dtype=float)
    g_insitu = res_subset[grade_col].to_numpy(dtype=float)

    # 1. Apply Mining Dilution:
    # T_dil = T_insitu * (1 + Dilution)
    # g_dil = (T_insitu * g_insitu + T_waste * g_waste) / T_dil
    dil_frac = mining_dilution_pct / 100.0
    t_diluted = t_insitu * (1.0 + dil_frac)
    t_waste = t_diluted - t_insitu
    g_diluted = (t_insitu * g_insitu + t_waste * float(dilution_grade)) / np.maximum(
        1e-9, t_diluted
    )

    # 2. Apply Mining Recovery (Ore Loss):
    # T_rom = T_diluted * Mining_Recovery
    # g_rom = g_diluted (ore loss drops mass, but does not alter blended head grade)
    rec_frac = mining_recovery_pct / 100.0
    t_rom = t_diluted * rec_frac
    g_rom = g_diluted
    contained_metal = t_rom * (g_rom / 100.0)

    reserve_df = pd.DataFrame(
        {
            "reserve_category": reserve_cats,
            "rom_tonnes": t_rom,
            "rom_grade": g_rom,
            "contained_metal": contained_metal,
            "in_situ_tonnes": t_insitu,
            "in_situ_grade": g_insitu,
        },
        index=res_subset.index,
    )

    # Preserve spatial coordinates if present
    for coord in ("x", "y", "z", "easting", "northing", "elevation"):
        if coord in res_subset.columns:
            reserve_df[coord] = res_subset[coord]

    # Attach conversion audit metadata
    reserve_df.attrs["excluded_inferred_tonnes"] = inferred_tonnes
    reserve_df.attrs["mining_dilution_pct"] = mining_dilution_pct
    reserve_df.attrs["mining_recovery_pct"] = mining_recovery_pct
    reserve_df.attrs["cutoff_grade"] = cutoff_grade
    return reserve_df


def format_reserve_statement(
    reserve_df: pd.DataFrame,
    cutoff_grade: float,
    mining_dilution_pct: float,
    mining_recovery_pct: float,
    commodity_price: str,
    metallurgical_recovery: float,
    category_col: str = "reserve_category",
    grade_col: str = "rom_grade",
    tonnes_col: str = "rom_tonnes",
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    metal_unit: str = "kt",
    rpeee_constraint: str = "Constrained within engineered final pit design",
    sig_figs: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Formats an official NI 43-101 / JORC compliant Mineral Reserve Statement.

    Enforces:
    - Segregation into Proven, Probable, and Total Proven + Probable reserves.
    - Tiered significant figures rounding to eliminate false precision.
    - Mandatory regulatory footnotes disclosing all applied Modifying Factors.

    Parameters
    ----------
    reserve_df : pd.DataFrame
        Mineral reserve DataFrame generated by convert_resource_to_reserve.
    cutoff_grade : float
        Cut-off grade used for reserve delineation. Required without default.
    mining_dilution_pct : float
        Mining dilution percentage applied. Required without default.
    mining_recovery_pct : float
        Mining recovery percentage applied. Required without default.
    commodity_price : str
        Commodity price disclosure (e.g., "$3.80/lb Cu"). Required without default.
    metallurgical_recovery : float
        Metallurgical recovery percentage (e.g., 88.0). Required without default.
    category_col : str, default "reserve_category"
        Column indicating reserve classification.
    grade_col : str, default "rom_grade"
        Run-of-mine grade column name.
    tonnes_col : str, default "rom_tonnes"
        Run-of-mine tonnage column name.
    grade_unit : str, default "% Cu"
        Grade display unit.
    tonnage_unit : str, default "Mt"
        Tonnage display unit ("t", "kt", "Mt").
    metal_unit : str, default "kt"
        Contained metal display unit.
    rpeee_constraint : str, default "Constrained within engineered final pit design"
        Engineering design constraint statement.
    sig_figs : dict, optional
        Custom significant figures per category.

    Returns
    -------
    pd.DataFrame
        Formatted Mineral Reserve Statement with attached compliance footnotes.
    """
    for col in (category_col, grade_col, tonnes_col):
        if col not in reserve_df.columns:
            raise ValueError(f"Required column '{col}' not found in reserve DataFrame.")

    default_sig_figs = {
        "Proven Reserve": 3,
        "Probable Reserve": 2,
        "Total Proven + Probable": 3,
    }
    if sig_figs:
        default_sig_figs.update(sig_figs)

    # Unit scaling
    t_scale = 1e6 if tonnage_unit == "Mt" else (1e3 if tonnage_unit == "kt" else 1.0)
    m_scale = 1e3 if metal_unit == "kt" else (1e6 if metal_unit == "Mt" else 1.0)

    rows = []
    # 1. Proven Reserves
    prov_mask = reserve_df[category_col].astype(str).str.strip().str.startswith("Prov")
    prov_df = reserve_df.loc[prov_mask]

    t_prov = float(prov_df[tonnes_col].sum())
    g_prov = (
        float((prov_df[tonnes_col] * prov_df[grade_col]).sum() / max(1e-9, t_prov))
        if t_prov > 0
        else 0.0
    )
    m_prov = t_prov * (g_prov / 100.0) if t_prov > 0 else 0.0

    # 2. Probable Reserves
    prob_mask = reserve_df[category_col].astype(str).str.strip().str.startswith("Prob")
    prob_df = reserve_df.loc[prob_mask]

    t_prob = float(prob_df[tonnes_col].sum())
    g_prob = (
        float((prob_df[tonnes_col] * prob_df[grade_col]).sum() / max(1e-9, t_prob))
        if t_prob > 0
        else 0.0
    )
    m_prob = t_prob * (g_prob / 100.0) if t_prob > 0 else 0.0

    # 3. Total Proven + Probable
    t_tot = t_prov + t_prob
    g_tot = (
        float((t_prov * g_prov + t_prob * g_prob) / max(1e-9, t_tot))
        if t_tot > 0
        else 0.0
    )
    m_tot = t_prov * (g_prov / 100.0) + t_prob * (g_prob / 100.0)

    for cat_name, t_val, g_val, m_val in [
        ("Proven Reserve", t_prov, g_prov, m_prov),
        ("Probable Reserve", t_prob, g_prob, m_prob),
        ("Total Proven + Probable", t_tot, g_tot, m_tot),
    ]:
        n_sf = default_sig_figs.get(cat_name, 3)
        rows.append(
            {
                "Classification": cat_name,
                f"Cutoff ({grade_unit})": f"{cutoff_grade:.2f}",
                f"Tonnage ({tonnage_unit})": _round_sig_figs(t_val / t_scale, n_sf),
                f"Grade ({grade_unit})": _round_sig_figs(g_val, n_sf),
                f"Contained Metal ({metal_unit})": _round_sig_figs(
                    m_val / m_scale, n_sf
                ),
            }
        )

    statement_df = pd.DataFrame(rows)

    # Mandatory CIM / JORC Compliance Footnotes
    footnotes = [
        "1. Mineral Reserves are reported in accordance with CIM Definition Standards (2014) / JORC Code (2012).",
        "2. Tonnages, grades, and contained metal are Run-of-Mine (ROM) and rounded to reflect relative uncertainty. Totals may not sum due to rounding.",
        "3. Mineral Reserves represent the economically mineable part of Measured and Indicated Mineral Resources demonstrated by at least a Pre-Feasibility Study.",
        (
            f"4. Modifying Factors applied: Mining Dilution = {mining_dilution_pct:.1f}%, Mining Recovery = {mining_recovery_pct:.1f}%, "
            f"Metallurgical Recovery = {metallurgical_recovery:.1f}%, Base Cutoff Grade = {cutoff_grade:.2f}{grade_unit}, Commodity Price = {commodity_price}."
        ),
        f"5. {rpeee_constraint}.",
    ]
    statement_df.attrs["footnotes"] = footnotes
    return statement_df


# =============================================================================
# RESERVE VISUALIZATION (WATERFALL BRIDGES, MAPS & GRADE-TONNAGE SHIFT)
# =============================================================================


def plot_resource_to_reserve_waterfall(
    resource_df: pd.DataFrame,
    reserve_df: pd.DataFrame,
    cutoff_grade: float,
    mining_dilution_pct: float,
    mining_recovery_pct: float,
    dilution_grade: float = 0.0,
    category_col: str = "category",
    grade_col: str = "grade",
    tonnes_col: str = "tonnes",
    tonnage_unit: str = "Mt",
    metal_unit: str = "kt",
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: tuple[float, float] = (16, 7),
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Generates dual-panel waterfall reconciliation charts (Tonnage & Contained Metal).

    Reconciles in-situ Measured & Indicated Mineral Resources to Run-of-Mine (ROM)
    Mineral Reserves through each modifying factor step:
    1. In-Situ M&I Resource Inventory
    2. Sub-Economic Cutoff Truncation (< Cutoff Grade)
    3. Mining Ore Loss (Recovery < 100%)
    4. Mining Dilution Tonnage Added
    5. Final Delivered Run-of-Mine (ROM) Mineral Reserve

    Parameters
    ----------
    resource_df : pd.DataFrame
        Classified mineral resource block model / polygon table.
    reserve_df : pd.DataFrame
        Run-of-Mine mineral reserve table produced by convert_resource_to_reserve.
    cutoff_grade : float
        Economic cut-off grade applied.
    mining_dilution_pct : float
        Mining dilution percentage applied.
    mining_recovery_pct : float
        Mining recovery percentage applied.
    dilution_grade : float, default 0.0
        Grade of diluting rock.
    category_col : str, default "category"
        Column in resource_df with classifications.
    grade_col : str, default "grade"
        In-situ grade column name.
    tonnes_col : str, default "tonnes"
        In-situ tonnage column name.
    tonnage_unit : str, default "Mt"
        Tonnage scale unit ("t", "kt", "Mt").
    metal_unit : str, default "kt"
        Metal scale unit ("t", "kt", "Mt").
    grade_unit : str, default "% Cu"
        Grade display unit.
    title : str, optional
        Overall figure title.
    figsize : tuple[float, float], default (16, 7)
        Matplotlib figure dimensions.

    Returns
    -------
    tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]
        Matplotlib figure and pair of axes (tonnage_ax, metal_ax).
    """
    t_scale = 1e6 if tonnage_unit == "Mt" else (1e3 if tonnage_unit == "kt" else 1.0)
    m_scale = 1e3 if metal_unit == "kt" else (1e6 if metal_unit == "Mt" else 1.0)

    # 1. Filter for Measured & Indicated in-situ resource
    cat_s = resource_df[category_col].astype(str).str.strip().str.capitalize()
    is_mi = cat_s.str.startswith("Meas") | cat_s.str.startswith("Ind")
    mi_df = resource_df.loc[is_mi].copy()

    # Step 1: In-Situ M&I
    t_step1 = float(mi_df[tonnes_col].sum())
    m_step1 = float((mi_df[tonnes_col] * mi_df[grade_col] / 100.0).sum())

    # Step 2: Cut-Off Loss
    sub_cutoff = mi_df[mi_df[grade_col] < cutoff_grade]
    delta_t_co = -float(sub_cutoff[tonnes_col].sum())
    delta_m_co = -float((sub_cutoff[tonnes_col] * sub_cutoff[grade_col] / 100.0).sum())

    t_above_co = t_step1 + delta_t_co
    m_above_co = m_step1 + delta_m_co

    # Step 3: Mining Ore Loss
    ore_loss_frac = max(0.0, 1.0 - (mining_recovery_pct / 100.0))
    delta_t_loss = -float(t_above_co * ore_loss_frac)
    delta_m_loss = -float(m_above_co * ore_loss_frac)

    # Step 4: Mining Dilution
    dil_frac = (mining_dilution_pct / 100.0) * (mining_recovery_pct / 100.0)
    delta_t_dil = float(t_above_co * dil_frac)
    delta_m_dil = float(delta_t_dil * (dilution_grade / 100.0))

    # Step 5: Final ROM Reserve
    t_final = float(reserve_df["rom_tonnes"].sum())
    m_final = float(reserve_df["contained_metal"].sum())

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=figsize)

    steps = [
        "In-Situ M&I\nResource",
        f"Sub-Cutoff\n(<{cutoff_grade:.2f}{grade_unit})",
        f"Ore Loss\n({100-mining_recovery_pct:.1f}%)",
        f"Dilution\n(+{mining_dilution_pct:.1f}%)",
        "Run-of-Mine\nReserve",
    ]

    def _render_waterfall(ax, deltas, finals, unit, label):
        n = len(deltas)
        bottoms = np.zeros(n)
        heights = np.zeros(n)
        colors = []

        # Step 0: Initial Total
        bottoms[0] = 0.0
        heights[0] = deltas[0]
        colors.append("#1f77b4")  # Blue total

        running_total = deltas[0]
        for i in range(1, n - 1):
            d = deltas[i]
            if d < 0:
                bottoms[i] = running_total + d
                heights[i] = abs(d)
                colors.append("#d62728")  # Red deduction
            else:
                bottoms[i] = running_total
                heights[i] = d
                colors.append("#ff7f0e")  # Orange addition
            running_total += d

        # Step n-1: Final Total
        bottoms[-1] = 0.0
        heights[-1] = finals
        colors.append("#2ca02c")  # Green final reserve

        bars = ax.bar(
            range(n),
            heights,
            bottom=bottoms,
            color=colors,
            edgecolor="black",
            linewidth=1.2,
            width=0.6,
        )

        # Connecting dashed lines
        cur_lev = deltas[0]
        for i in range(1, n - 1):
            ax.plot(
                [i - 1 + 0.3, i - 0.3],
                [cur_lev, cur_lev],
                "k--",
                linewidth=1.0,
                alpha=0.6,
            )
            cur_lev += deltas[i]
        ax.plot(
            [n - 2 + 0.3, n - 1 - 0.3],
            [cur_lev, cur_lev],
            "k--",
            linewidth=1.0,
            alpha=0.6,
        )

        # Callout values
        for i, bar in enumerate(bars):
            val = heights[i]
            val_signed = (
                deltas[i] if (0 < i < n - 1) else (finals if i == n - 1 else val)
            )
            prefix = "+" if (0 < i < n - 1 and val_signed > 0) else ""
            y_pos = bottoms[i] + heights[i] + (max(deltas[0], finals) * 0.02)
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{prefix}{val_signed:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_xticks(range(n))
        ax.set_xticklabels(steps, fontsize=9.5, fontweight="bold")
        ax.set_ylabel(f"{label} ({unit})", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(deltas[0], finals) * 1.18)
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Render Tonnage Waterfall
    _render_waterfall(
        ax_t,
        [
            t_step1 / t_scale,
            delta_t_co / t_scale,
            delta_t_loss / t_scale,
            delta_t_dil / t_scale,
            t_final / t_scale,
        ],
        t_final / t_scale,
        tonnage_unit,
        "Ore Tonnage",
    )
    ax_t.set_title("Tonnage Reconciliation Waterfall", fontsize=12, fontweight="bold")

    # Render Contained Metal Waterfall
    _render_waterfall(
        ax_m,
        [
            m_step1 / m_scale,
            delta_m_co / m_scale,
            delta_m_loss / m_scale,
            delta_m_dil / m_scale,
            m_final / m_scale,
        ],
        m_final / m_scale,
        metal_unit,
        "Contained Metal",
    )
    ax_m.set_title(
        "Contained Metal Reconciliation Waterfall", fontsize=12, fontweight="bold"
    )

    fig_title = (
        title
        if title
        else "Resource-to-Reserve Bridge (Modifying Factors Reconciliation)"
    )
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig, (ax_t, ax_m)


def plot_reserve_classification_map(
    block_model: pd.DataFrame,
    boundary: Optional[Sequence[tuple[float, float]]] = None,
    drillholes: Optional[pd.DataFrame] = None,
    status_col: str = "status",
    x_col: str = "x",
    y_col: str = "y",
    title: str = "Mineral Reserve & Resource Classification Map",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Renders 2D spatial mine plan map colored by regulatory reserve status.

    Standard coloring:
    - Proven Reserve: Forest Green (#2ca02c)
    - Probable Reserve: Royal Blue (#1f77b4)
    - Inferred Resource (Excluded): Purple / Magenta (#9467bd)
    - Sub-Economic / Waste (< Cutoff): Light Grey (#d3d3d3)

    Parameters
    ----------
    block_model : pd.DataFrame
        Table with block coordinates and 'status' column indicating category.
    boundary : Sequence[tuple[float, float]], optional
        Perimeter boundary polygon coordinates.
    drillholes : pd.DataFrame, optional
        Collar coordinates table with 'x' and 'y' columns.
    status_col : str, default "status"
        Column containing status string.
    x_col, y_col : str, default "x", "y"
        Coordinate column names.
    title : str, default "Mineral Reserve & Resource Classification Map"
        Plot title.
    ax : plt.Axes, optional
        Existing axes to draw on.
    figsize : tuple[float, float], default (10, 8)
        Dimensions of figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    color_palette = {
        "Proven Reserve": "#2ca02c",  # Forest green
        "Probable Reserve": "#1f77b4",  # Blue
        "Inferred Resource (Excluded)": "#9467bd",  # Purple
        "Sub-Economic / Waste": "#d3d3d3",  # Light grey
    }

    # Draw blocks grouped by status
    for status_name, color in color_palette.items():
        sub = block_model[
            block_model[status_col].astype(str).str.strip().str.lower()
            == status_name.lower()
        ]
        if len(sub) > 0:
            ax.scatter(
                sub[x_col],
                sub[y_col],
                c=color,
                label=f"{status_name} ({len(sub):,} blocks)",
                s=35,
                marker="s",
                alpha=0.85,
                edgecolors="none",
            )

    # Draw boundary if provided
    if boundary is not None:
        b_pts = np.array(list(boundary) + [boundary[0]])
        ax.plot(
            b_pts[:, 0],
            b_pts[:, 1],
            "r--",
            linewidth=2.0,
            label="Pit / Concession Perimeter",
        )

    # Draw drillholes if provided
    if drillholes is not None and not drillholes.empty:
        ax.scatter(
            drillholes["x"],
            drillholes["y"],
            c="black",
            s=50,
            marker="^",
            label=f"Drillholes (N={len(drillholes)})",
            zorder=6,
        )

    ax.set_xlabel("Easting (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Northing (m)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    fig.tight_layout()
    return fig, ax


def plot_in_situ_vs_diluted_curves(
    in_situ_gt: pd.DataFrame,
    diluted_gt: pd.DataFrame,
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    title: str = "Grade–Tonnage Shift: In-Situ Resource vs. Diluted Reserve",
    figsize: tuple[float, float] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plots comparative Grade-Tonnage curves demonstrating the dilution grade shift.

    Visualizes the classic operational shift from in-situ resource to run-of-mine
    reserve:
    - Tonnage shifts higher (dashed to solid blue curve) due to wall rock dilution.
    - Average head grade shifts downward (dashed to solid red curve) due to low-grade dilution.

    Parameters
    ----------
    in_situ_gt : pd.DataFrame
        Grade-tonnage table of in-situ resources.
    diluted_gt : pd.DataFrame
        Grade-tonnage table of diluted/recovered reserves.
    grade_unit : str, default "% Cu"
        Grade unit label.
    tonnage_unit : str, default "Mt"
        Tonnage unit label.
    title : str
        Figure title.
    figsize : tuple[float, float], default (12, 6)
        Figure size.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    t_scale = 1e6 if tonnage_unit == "Mt" else (1e3 if tonnage_unit == "kt" else 1.0)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    cutoffs = in_situ_gt.index.to_numpy(dtype=float)

    # In-Situ Curves (Dashed)
    (line1,) = ax1.plot(
        cutoffs,
        in_situ_gt["ore_tonnes"] / t_scale,
        color="#1f77b4",
        linestyle="--",
        linewidth=2.0,
        label="In-Situ Resource Tonnage",
    )
    (line2,) = ax2.plot(
        cutoffs,
        in_situ_gt["ore_grade"],
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label="In-Situ Resource Grade",
    )

    # Diluted ROM Curves (Solid with markers)
    (line3,) = ax1.plot(
        cutoffs,
        diluted_gt["ore_tonnes"] / t_scale,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.5,
        marker="o",
        label="Diluted Reserve Tonnage (ROM)",
    )
    (line4,) = ax2.plot(
        cutoffs,
        diluted_gt["ore_grade"],
        color="#d62728",
        linestyle="-",
        linewidth=2.5,
        marker="s",
        label="Diluted Reserve Grade (ROM)",
    )

    ax1.set_xlabel(f"Cutoff Grade ({grade_unit})", fontsize=11, fontweight="bold")
    ax1.set_ylabel(
        f"Ore Tonnage ({tonnage_unit})", fontsize=11, fontweight="bold", color="#1f77b4"
    )
    ax2.set_ylabel(
        f"Average Grade ({grade_unit})", fontsize=11, fontweight="bold", color="#d62728"
    )

    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(True, linestyle=":", alpha=0.5)

    lines = [line1, line3, line2, line4]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9.5)

    ax1.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig, ax1


# =============================================================================
# DATA PREPARATION (DOWN-HOLE COMPOSITING & HIGH-GRADE CAPPING / TOP-CUTTING)
# =============================================================================


def composite_drillhole_intervals(
    assay_df: pd.DataFrame,
    composite_length: float,
    hole_id_col: str = "hole_id",
    from_col: str = "from_m",
    to_col: str = "to_m",
    grade_col: str = "grade",
    domain_col: Optional[str] = None,
    density_col: Optional[str] = None,
    min_length_ratio: float = 0.50,
    remnant_strategy: str = "discard",
) -> pd.DataFrame:
    """Down-hole regular compositing with domain-boundary constraints.

    Solves the Support Effect in mining geostatistics: raw drill core assays have
    variable sample lengths cut along geological and alteration contacts. Equal
    volume/length support is required prior to spatial statistical evaluation and
    variography.

    Parameters
    ----------
    assay_df : pd.DataFrame
        Raw drillhole assay intervals with collar/survey coordinates.
    composite_length : float
        Target composite interval length (e.g., 2.0m or 12.0m mining bench height).
        Required without default.
    hole_id_col : str, default "hole_id"
        Drillhole identifier column name.
    from_col, to_col : str, default "from_m", "to_m"
        Down-hole interval start and end depth column names (meters).
    grade_col : str, default "grade"
        Assay grade column name.
    domain_col : str, optional
        Geological / lithological / structural domain column.
        CRITICAL: If provided, compositing strictly resets at domain contacts.
        Composites never cross domain boundaries.
    density_col : str, optional
        Bulk density column for mass-weighted compositing (length * density).
    min_length_ratio : float, default 0.50
        Minimum length ratio (relative to composite_length) required to retain a
        terminal remnant interval. (e.g., 0.50 retains remnants >= 50% of target).
    remnant_strategy : {"discard", "distribute"}, default "discard"
        Strategy for terminal leftovers:
        - "discard": Discards remnants shorter than min_length_ratio * composite_length.
        - "distribute": Evenly adjusts composite lengths across the interval so all
          composites have equal support without tiny remnants.

    Returns
    -------
    pd.DataFrame
        Composited drillhole intervals with length-weighted grades and midpoint coordinates.
    """
    if composite_length <= 0:
        raise ValueError(
            f"composite_length must be strictly positive, got {composite_length}"
        )
    if remnant_strategy not in ("discard", "distribute"):
        raise ValueError(
            f"remnant_strategy must be 'discard' or 'distribute', got '{remnant_strategy}'"
        )

    for col in (hole_id_col, from_col, to_col, grade_col):
        if col not in assay_df.columns:
            raise ValueError(f"Required column '{col}' not found in assay DataFrame.")

    if domain_col is not None and domain_col not in assay_df.columns:
        raise ValueError(f"Specified domain column '{domain_col}' not found.")

    # Spatial coordinate columns to preserve and interpolate
    coord_cols = [c for c in ("x", "y", "z", "elevation") if c in assay_df.columns]

    # Grouping keys: strictly constrain by hole_id and domain (if provided)
    group_cols = [hole_id_col]
    if domain_col is not None:
        group_cols.append(domain_col)

    composite_records = []
    discarded_count = 0

    for keys, grp in assay_df.groupby(group_cols, sort=False):
        hole_id = keys[0] if isinstance(keys, tuple) else keys
        dom_val = keys[1] if (isinstance(keys, tuple) and len(keys) > 1) else None

        sub = grp.sort_values(from_col).copy()
        if len(sub) == 0:
            continue

        run_start = float(sub[from_col].min())
        run_end = float(sub[to_col].max())
        run_len = run_end - run_start

        if run_len <= 0:
            continue

        # Establish composite interval boundaries
        comp_intervals = []
        if remnant_strategy == "distribute":
            n_comp = max(1, int(round(run_len / composite_length)))
            actual_len = run_len / n_comp
            # If total run is shorter than min_length_ratio * composite_length, check discard
            if run_len < (min_length_ratio * composite_length):
                discarded_count += 1
                continue
            for i in range(n_comp):
                c_start = run_start + i * actual_len
                c_end = run_start + (i + 1) * actual_len
                comp_intervals.append((c_start, c_end))
        else:  # "discard"
            curr = run_start
            while curr + composite_length <= run_end:
                comp_intervals.append((curr, curr + composite_length))
                curr += composite_length
            remnant_len = run_end - curr
            if remnant_len >= (min_length_ratio * composite_length):
                comp_intervals.append((curr, run_end))
            elif remnant_len > 0:
                discarded_count += 1

        # Calculate length-weighted (and density-weighted) grades for each composite
        raw_from = sub[from_col].to_numpy(dtype=float)
        raw_to = sub[to_col].to_numpy(dtype=float)
        raw_grade = sub[grade_col].to_numpy(dtype=float)
        raw_dens = (
            sub[density_col].to_numpy(dtype=float)
            if density_col
            else np.ones(len(sub), dtype=float)
        )

        for c_from, c_to in comp_intervals:
            c_len = c_to - c_from
            if c_len <= 0:
                continue

            # Overlap calculation
            overlap_starts = np.maximum(c_from, raw_from)
            overlap_ends = np.minimum(c_to, raw_to)
            overlaps = np.maximum(0.0, overlap_ends - overlap_starts)

            valid_mask = overlaps > 1e-9
            if not np.any(valid_mask):
                continue

            active_lens = overlaps[valid_mask]
            active_grades = raw_grade[valid_mask]
            active_weights = active_lens * raw_dens[valid_mask]

            total_weight = float(active_weights.sum())
            if total_weight <= 0:
                continue

            weighted_grade = float(
                (active_weights * active_grades).sum() / total_weight
            )

            rec: dict = {
                hole_id_col: hole_id,
                from_col: c_from,
                to_col: c_to,
                "length": c_len,
                grade_col: weighted_grade,
            }
            if domain_col is not None:
                rec[domain_col] = dom_val

            # Interpolate spatial coordinates to composite midpoint
            c_mid = (c_from + c_to) / 2.0
            for c_name in coord_cols:
                raw_coords = sub[c_name].to_numpy(dtype=float)
                # Weighted average coordinate for midpoint
                comp_coord = float(
                    (active_lens * raw_coords[valid_mask]).sum() / active_lens.sum()
                )
                rec[c_name] = comp_coord

            composite_records.append(rec)

    out_df = pd.DataFrame(composite_records)
    out_df.attrs["composite_length"] = composite_length
    out_df.attrs["remnant_strategy"] = remnant_strategy
    out_df.attrs["discarded_remnants_count"] = discarded_count
    return out_df


def apply_grade_capping(
    composite_df: pd.DataFrame,
    cap_grade: Optional[float] = None,
    percentile: Optional[float] = None,
    grade_col: str = "grade",
    length_col: Optional[str] = "length",
    output_col: str = "capped_grade",
) -> pd.DataFrame:
    """Applies statistical top-cutting (capping) to composited drillhole intervals.

    Mitigates the Proportional Effect and prevents erratic high-grade outliers from
    smearing artificial grade balloons across neighboring mining blocks during
    spatial estimation.

    Parameters
    ----------
    composite_df : pd.DataFrame
        Composited drillhole intervals. Capping must always follow compositing.
    cap_grade : float, optional
        Explicit maximum grade threshold. Samples above this grade are clamped to cap_grade.
    percentile : float, optional
        Percentile threshold (e.g., 99.0 or 99.5) to compute cap_grade if cap_grade is not provided.
    grade_col : str, default "grade"
        Grade column name in composite_df.
    length_col : str, optional, default "length"
        Interval length column used for calculating length-weighted metal reduction.
    output_col : str, default "capped_grade"
        Column name to store the capped grades in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of composite_df with output_col containing capped grades and
        comprehensive audit metadata attached in .attrs["capping_summary"].
    """
    if cap_grade is None and percentile is None:
        raise ValueError(
            "Must specify either an explicit cap_grade or a percentile (e.g. 99.0)."
        )

    if grade_col not in composite_df.columns:
        raise ValueError(f"Grade column '{grade_col}' not found in DataFrame.")

    grades = composite_df[grade_col].to_numpy(dtype=float)
    if len(grades) == 0:
        res = composite_df.copy()
        res[output_col] = grades
        return res

    # Determine capping threshold
    if cap_grade is None:
        if not (0.0 < percentile <= 100.0):
            raise ValueError(f"Percentile must be in (0, 100], got {percentile}")
        cap_val = float(np.percentile(grades, percentile))
    else:
        if cap_grade <= 0:
            raise ValueError(f"cap_grade must be strictly positive, got {cap_grade}")
        cap_val = float(cap_grade)

    capped_grades = np.minimum(grades, cap_val)

    # Weights for metal calculation
    if length_col is not None and length_col in composite_df.columns:
        weights = composite_df[length_col].to_numpy(dtype=float)
    else:
        weights = np.ones(len(grades), dtype=float)

    uncapped_metal = float((weights * grades).sum())
    capped_metal = float((weights * capped_grades).sum())
    metal_reduction_pct = (
        ((uncapped_metal - capped_metal) / uncapped_metal) * 100.0
        if uncapped_metal > 0
        else 0.0
    )

    n_capped = int((grades > cap_val).sum())
    n_total = len(grades)

    uncapped_mean = float(grades.mean())
    capped_mean = float(capped_grades.mean())
    uncapped_std = float(grades.std())
    capped_std = float(capped_grades.std())

    uncapped_cv = float(uncapped_std / uncapped_mean) if uncapped_mean > 0 else 0.0
    capped_cv = float(capped_std / capped_mean) if capped_mean > 0 else 0.0

    result_df = composite_df.copy()
    result_df[output_col] = capped_grades

    summary = {
        "cap_grade": cap_val,
        "total_samples": n_total,
        "samples_capped": n_capped,
        "samples_capped_pct": float(n_capped / n_total * 100.0),
        "metal_reduction_pct": float(metal_reduction_pct),
        "uncapped_mean": uncapped_mean,
        "capped_mean": capped_mean,
        "uncapped_std": uncapped_std,
        "capped_std": capped_std,
        "uncapped_cv": uncapped_cv,
        "capped_cv": capped_cv,
    }
    result_df.attrs["capping_summary"] = summary
    return result_df


# =============================================================================
# EXPLORATORY DATA ANALYSIS (EDA) & DISTRIBUTION DIAGNOSTICS
# =============================================================================


def exploratory_data_analysis(
    df: pd.DataFrame,
    grade_col: str = "grade",
    weights_col: Optional[str] = None,
) -> pd.DataFrame:
    """Computes comprehensive summary statistics for mineral resource evaluation.

    Complies with NI 43-101 and JORC reporting standards for exploratory data
    analysis (EDA). Evaluates distributional symmetry, mean vs. median divergence,
    the Coefficient of Variation (CV = sigma / mu), and preferential drilling
    clustering bias if weights are provided.

    Parameters
    ----------
    df : pd.DataFrame
        Assay or composite data table.
    grade_col : str, default "grade"
        Grade column name.
    weights_col : str, optional
        Declustering or spatial weights column. If provided, computes both Naive
        and Declustered statistics alongside clustering bias %.

    Returns
    -------
    pd.DataFrame
        Summary table comparing Naive (and Declustered) metrics:
        - Count, Min, Max, Mean, Median (P50), Mean/Median Ratio
        - Variance, Standard Deviation, Coefficient of Variation (CV)
        - Skewness, Kurtosis
        - Percentiles: P10, P25, P50, P75, P90, P95, P99
        Attributes (.attrs) include 'cv_status' and 'clustering_bias_pct'.
    """
    if grade_col not in df.columns:
        raise ValueError(f"Grade column '{grade_col}' not found in DataFrame.")

    grades = df[grade_col].to_numpy(dtype=float)
    valid_mask = ~np.isnan(grades)
    grades = grades[valid_mask]
    n = len(grades)

    if n == 0:
        raise ValueError(f"No valid non-null values found in column '{grade_col}'.")

    # Naive statistics
    min_val = float(grades.min())
    max_val = float(grades.max())
    mean_val = float(grades.mean())
    median_val = float(np.median(grades))
    var_val = float(grades.var(ddof=1)) if n > 1 else 0.0
    std_val = float(np.sqrt(var_val))
    cv_val = float(std_val / mean_val) if mean_val > 0 else 0.0
    skew_val = float(stats.skew(grades)) if n > 2 else 0.0
    kurt_val = float(stats.kurtosis(grades)) if n > 3 else 0.0

    p10 = float(np.percentile(grades, 10.0))
    p25 = float(np.percentile(grades, 25.0))
    p50 = median_val
    p75 = float(np.percentile(grades, 75.0))
    p90 = float(np.percentile(grades, 90.0))
    p95 = float(np.percentile(grades, 95.0))
    p99 = float(np.percentile(grades, 99.0))

    mean_median_ratio = float(mean_val / median_val) if median_val > 0 else 1.0

    metrics = [
        "Sample Count",
        "Minimum",
        "Maximum",
        "Mean",
        "Median (P50)",
        "Mean / Median Ratio",
        "Variance",
        "Standard Deviation",
        "Coeff. of Variation (CV)",
        "Skewness",
        "Kurtosis",
        "P10",
        "P25",
        "P75",
        "P90",
        "P95",
        "P99",
    ]

    naive_vals = [
        float(n),
        min_val,
        max_val,
        mean_val,
        median_val,
        mean_median_ratio,
        var_val,
        std_val,
        cv_val,
        skew_val,
        kurt_val,
        p10,
        p25,
        p75,
        p90,
        p95,
        p99,
    ]

    data = {"Metric": metrics, "Naive": naive_vals}

    # Optional declustered statistics
    clustering_bias_pct = None
    if weights_col is not None:
        if weights_col not in df.columns:
            raise ValueError(f"Weights column '{weights_col}' not found in DataFrame.")
        w = df[weights_col].to_numpy(dtype=float)[valid_mask]
        w_sum = w.sum()
        if w_sum > 0:
            w_norm = w / w_sum
            dec_mean = float(np.sum(w_norm * grades))
            dec_var = float(np.sum(w_norm * (grades - dec_mean) ** 2))
            dec_std = float(np.sqrt(dec_var))
            dec_cv = float(dec_std / dec_mean) if dec_mean > 0 else 0.0

            # Weighted percentiles
            sort_idx = np.argsort(grades)
            sorted_g = grades[sort_idx]
            cum_w = np.cumsum(w_norm[sort_idx])

            def w_perc(pct: float) -> float:
                return float(np.interp(pct / 100.0, cum_w, sorted_g))

            dec_med = w_perc(50.0)
            dec_p10 = w_perc(10.0)
            dec_p25 = w_perc(25.0)
            dec_p75 = w_perc(75.0)
            dec_p90 = w_perc(90.0)
            dec_p95 = w_perc(95.0)
            dec_p99 = w_perc(99.0)

            clustering_bias_pct = float(
                ((mean_val - dec_mean) / dec_mean) * 100.0 if dec_mean > 0 else 0.0
            )

            dec_vals = [
                float(n),
                min_val,
                max_val,
                dec_mean,
                dec_med,
                float(dec_mean / dec_med) if dec_med > 0 else 1.0,
                dec_var,
                dec_std,
                dec_cv,
                skew_val,
                kurt_val,
                dec_p10,
                dec_p25,
                dec_p75,
                dec_p90,
                dec_p95,
                dec_p99,
            ]
            data["Declustered"] = dec_vals

    out_df = pd.DataFrame(data).set_index("Metric")

    # Geostatistical rule of thumb on CV:
    if cv_val <= 1.0:
        cv_status = "Well-behaved / Low skewness (linear geostatistics suitable)"
    elif cv_val <= 1.5:
        cv_status = (
            "Moderately skewed (monitor variogram stability and kriging weights)"
        )
    else:
        cv_status = "Highly skewed / Outlier risk (CV > 1.5: top-cutting or domain review recommended)"

    out_df.attrs["cv_status"] = cv_status
    out_df.attrs["cv"] = cv_val
    if clustering_bias_pct is not None:
        out_df.attrs["clustering_bias_pct"] = clustering_bias_pct

    return out_df


def plot_eda_distributions(
    df: pd.DataFrame,
    grade_col: str = "grade",
    capped_grade_col: Optional[str] = None,
    cap_grade: Optional[float] = None,
    bins: int = 30,
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16.0, 5.0),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates the standard 3-panel Exploratory Data Analysis (EDA) distribution figure.

    Visualizes:
    1. Linear Histogram & Density with Cumulative Frequency (showing mean, median, CV).
    2. Log-Transformed Distribution (diagnosing unimodal vs. bimodal/multimodal mixing).
    3. Log-Probability Plot (normal probability plot of log grades for capping threshold validation).

    Parameters
    ----------
    df : pd.DataFrame
        Assay or composite table.
    grade_col : str, default "grade"
        Grade column name.
    capped_grade_col : str, optional
        Capped grade column name to compare before vs. after distributions.
    cap_grade : float, optional
        Explicit capping threshold value to display as horizontal cutoff on the
        probability plot and vertical line on the histogram.
    bins : int, default 30
        Number of histogram bins.
    grade_unit : str, default "% Cu"
        Grade unit label.
    title : str, optional
        Overall figure title.
    figsize : tuple of float, default (16.0, 5.0)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Matplotlib figure and the three axes objects.
    """
    if grade_col not in df.columns:
        raise ValueError(f"Grade column '{grade_col}' not found in DataFrame.")

    raw_grades = df[grade_col].to_numpy(dtype=float)
    valid_mask = ~np.isnan(raw_grades) & (raw_grades > 0)
    grades = raw_grades[valid_mask]
    n = len(grades)

    if n == 0:
        raise ValueError("No positive non-null grade values available for plotting.")

    mean_g = float(grades.mean())
    med_g = float(np.median(grades))
    std_g = float(grades.std())
    cv_g = float(std_g / mean_g) if mean_g > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # -------------------------------------------------------------------------
    # Panel 1: Linear Histogram + Cumulative Frequency
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1_cum = ax1.twinx()

    counts, bin_edges, _ = ax1.hist(
        grades,
        bins=bins,
        color="#1f77b4",
        edgecolor="black",
        alpha=0.65,
        density=False,
        label="Raw Composites",
    )

    # Overlay capped distribution if available
    if capped_grade_col is not None and capped_grade_col in df.columns:
        capped_g = df[capped_grade_col].to_numpy(dtype=float)[valid_mask]
        ax1.hist(
            capped_g,
            bins=bins,
            color="#d62728",
            edgecolor="#d62728",
            histtype="step",
            linewidth=2.0,
            label="After Capping",
        )

    # Cumulative % curve
    sorted_g = np.sort(grades)
    cum_pct = np.linspace(0.0, 100.0, len(sorted_g))
    ax1_cum.plot(
        sorted_g,
        cum_pct,
        color="#2ca02c",
        linewidth=2.0,
        linestyle="-",
        label="Cum. Freq. (%)",
    )
    ax1_cum.set_ylabel("Cumulative Frequency (%)", color="#2ca02c", fontsize=9.5)
    ax1_cum.tick_params(axis="y", labelcolor="#2ca02c")
    ax1_cum.set_ylim(0, 105)

    # Reference lines
    ax1.axvline(
        mean_g,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_g:.2f}",
    )
    ax1.axvline(
        med_g,
        color="#ff7f0e",
        linestyle=":",
        linewidth=1.5,
        label=f"Median: {med_g:.2f}",
    )
    if cap_grade is not None:
        ax1.axvline(
            cap_grade,
            color="black",
            linestyle="-.",
            linewidth=1.5,
            label=f"Cap: {cap_grade:.2f}",
        )

    ax1.set_xlabel(f"Grade ({grade_unit})", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Sample Frequency", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"Histogram & Cumulative Frequency\n(N={n}, CV={cv_g:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_cum.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    # -------------------------------------------------------------------------
    # Panel 2: Log-Transformed Distribution (Unimodal vs. Multimodal)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    log_grades = np.log(grades)

    ax2.hist(
        log_grades,
        bins=bins,
        color="#9467bd",
        edgecolor="black",
        alpha=0.65,
        density=True,
        label="ln(Grade) Density",
    )

    # Fitted normal PDF for visual log-normality reference
    mu_log = float(log_grades.mean())
    std_log = float(log_grades.std())
    x_eval = np.linspace(log_grades.min(), log_grades.max(), 200)
    pdf_eval = stats.norm.pdf(x_eval, mu_log, std_log)
    ax2.plot(
        x_eval, pdf_eval, color="#d62728", linewidth=2.0, label="Fitted Log-Normal"
    )

    ax2.set_xlabel(f"ln(Grade {grade_unit})", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax2.set_title(
        "Log-Transformed Distribution\n(Population Modality)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Log-Probability Plot (Normal Probability Plot of Log Grades)
    # -------------------------------------------------------------------------
    ax3 = axes[2]

    # Blom plotting position: (i - 0.375) / (N + 0.25)
    i_rank = np.arange(1, n + 1)
    p_blom = (i_rank - 0.375) / (n + 0.25)
    z_scores = stats.norm.ppf(p_blom)

    ax3.scatter(
        z_scores,
        sorted_g,
        s=16,
        color="#1f77b4",
        alpha=0.75,
        edgecolors="none",
        label="Raw Composites",
    )

    if capped_grade_col is not None and capped_grade_col in df.columns:
        sorted_cap = np.sort(df[capped_grade_col].to_numpy(dtype=float)[valid_mask])
        ax3.scatter(
            z_scores,
            sorted_cap,
            s=12,
            color="#d62728",
            alpha=0.75,
            marker="x",
            label="Capped Composites",
        )

    if cap_grade is not None:
        ax3.axhline(
            cap_grade,
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            label=f"Cap Threshold: {cap_grade:.2f} {grade_unit}",
        )

    ax3.set_yscale("log")

    # Set probability-spaced ticks on x-axis
    prob_ticks = np.array(
        [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    )
    z_ticks = stats.norm.ppf(prob_ticks)
    prob_labels = [
        "0.1%",
        "1%",
        "5%",
        "10%",
        "25%",
        "50%",
        "75%",
        "90%",
        "95%",
        "99%",
        "99.9%",
    ]

    # Filter ticks within actual z range
    in_range = (z_ticks >= z_scores.min() - 0.2) & (z_ticks <= z_scores.max() + 0.2)
    ax3.set_xticks(z_ticks[in_range])
    ax3.set_xticklabels(
        [prob_labels[k] for k in range(len(prob_labels)) if in_range[k]], fontsize=8
    )

    ax3.set_xlabel("Cumulative Probability (%)", fontsize=10, fontweight="bold")
    ax3.set_ylabel(f"Grade ({grade_unit}, Log Scale)", fontsize=10, fontweight="bold")
    ax3.set_title(
        "Log-Probability Plot\n(Capping & Outlier Diagnostic)",
        fontsize=11,
        fontweight="bold",
    )
    ax3.grid(True, which="both", linestyle=":", alpha=0.5)
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.03)

    fig.tight_layout()
    return fig, axes


# =============================================================================
# SPATIAL DOMAIN DELINEATION (CONTACT PROFILE ANALYSIS: HARD VS. SOFT BOUNDARY)
# =============================================================================


def contact_profile_analysis(
    df: pd.DataFrame,
    domain_col: str,
    grade_col: str,
    distance_col: Optional[str] = None,
    contact_surface: Optional[Sequence[Tuple[float, float]]] = None,
    bin_width: float = 2.0,
    max_distance: float = 30.0,
    domain_a: Optional[Any] = None,
    domain_b: Optional[Any] = None,
) -> pd.DataFrame:
    """Evaluates grade continuity across a geological contact (Boundary Analysis).

    Determines quantitatively whether a geological, lithological, or structural
    contact is:
    - HARD BOUNDARY: A sharp step-change discontinuity at distance 0.
      Requires strict segregation during kriging/interpolation (no sample sharing).
    - SOFT BOUNDARY: A continuous gradient with no discontinuity.
      Samples from both sides can be freely shared across the contact.
    - SEMI-SOFT / TRANSITIONAL BOUNDARY: A moderate transition zone.
      Samples are shared only within a restricted buffer or with distance decay.

    Parameters
    ----------
    df : pd.DataFrame
        Composite or assay data table.
    domain_col : str
        Domain classification column (e.g. "lithology", "oxidation", "zone").
    grade_col : str
        Grade column name.
    distance_col : str, optional
        Signed perpendicular distance to the contact surface (negative for domain_a,
        positive for domain_b). If omitted, auto-detected from 'distance' or computed
        from contact_surface.
    contact_surface : sequence of tuple of float, optional
        2D contact line segment [(x1, y1), (x2, y2)] used to compute perpendicular
        signed distance if distance_col is not pre-calculated.
    bin_width : float, default 2.0
        Distance bin width (meters).
    max_distance : float, default 30.0
        Maximum distance from contact to evaluate in each direction (meters).
    domain_a, domain_b : Any, optional
        Identifiers for the two domains bordering the contact.
        If omitted, selected as the two most frequent categories in domain_col.

    Returns
    -------
    pd.DataFrame
        Contact profile binned table with columns:
        ['bin_center', 'bin_min', 'bin_max', 'domain', 'sample_count', 'mean_grade', 'std_grade', 'sem_grade']
        Audit attributes (.attrs) include 'boundary_type', 'step_change', 'step_ratio', and 'recommendation'.
    """
    if domain_col not in df.columns:
        raise ValueError(f"Domain column '{domain_col}' not found in DataFrame.")
    if grade_col not in df.columns:
        raise ValueError(f"Grade column '{grade_col}' not found in DataFrame.")
    if bin_width <= 0:
        raise ValueError(f"bin_width must be strictly positive, got {bin_width}")
    if max_distance <= 0:
        raise ValueError(f"max_distance must be strictly positive, got {max_distance}")

    # Determine domain pair
    if domain_a is None or domain_b is None:
        top_domains = df[domain_col].value_counts().index.tolist()
        if len(top_domains) < 2:
            raise ValueError(
                f"Need at least 2 unique domains in '{domain_col}' for contact analysis."
            )
        dom_a = top_domains[0] if domain_a is None else domain_a
        dom_b = top_domains[1] if domain_b is None else domain_b
    else:
        dom_a, dom_b = domain_a, domain_b

    # Filter to samples belonging to domain_a or domain_b
    sub_df = df[df[domain_col].isin([dom_a, dom_b])].copy()
    if len(sub_df) == 0:
        raise ValueError(f"No samples found for domains '{dom_a}' and '{dom_b}'.")

    # Compute or extract signed distance
    if distance_col is not None:
        if distance_col not in sub_df.columns:
            raise ValueError(f"Specified distance_col '{distance_col}' not found.")
        dists = sub_df[distance_col].to_numpy(dtype=float)
    elif "distance" in sub_df.columns:
        dists = sub_df["distance"].to_numpy(dtype=float)
    elif "dist_to_contact" in sub_df.columns:
        dists = sub_df["dist_to_contact"].to_numpy(dtype=float)
    elif contact_surface is not None:
        # Calculate 2D signed perpendicular distance to line segment
        pts = sub_df[["x", "y"]].to_numpy(dtype=float)
        p1 = np.array(contact_surface[0], dtype=float)
        p2 = np.array(contact_surface[1], dtype=float)
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len <= 1e-9:
            raise ValueError("Contact surface line segment has zero length.")
        normal = np.array([-line_vec[1], line_vec[0]]) / line_len

        # Signed distance from p1 along normal
        dists = np.dot(pts - p1, normal)
    else:
        raise ValueError(
            "Must provide either 'distance_col' or a 2D 'contact_surface' segment."
        )

    dom_vals = sub_df[domain_col].to_numpy()
    if (dists >= 0).all():
        signed_dists = np.where(dom_vals == dom_a, -dists, dists)
    else:
        # Orient signed distances so that domain_a is on the negative distance side
        mask_a = dom_vals == dom_a
        mask_b = dom_vals == dom_b
        if mask_a.any() and mask_b.any():
            if np.mean(dists[mask_a]) > np.mean(dists[mask_b]):
                dists = -dists
        signed_dists = dists.copy()

    grades = sub_df[grade_col].to_numpy(dtype=float)

    # Construct discrete distance bins with 0.0 as an exact boundary
    left_edges = np.arange(-max_distance, 0.0, bin_width)
    right_edges = np.arange(0.0, max_distance + 1e-6, bin_width)
    bin_edges = np.unique(np.concatenate([left_edges, [0.0], right_edges]))

    records = []
    for i in range(len(bin_edges) - 1):
        b_min = float(bin_edges[i])
        b_max = float(bin_edges[i + 1])
        b_center = (b_min + b_max) / 2.0
        is_domain_a = b_center < 0

        # Mask: include upper edge for last bin
        if i == len(bin_edges) - 2:
            mask = (signed_dists >= b_min) & (signed_dists <= b_max)
        else:
            mask = (signed_dists >= b_min) & (signed_dists < b_max)

        bin_grades = grades[mask]
        valid_grades = bin_grades[~np.isnan(bin_grades)]
        cnt = len(valid_grades)

        if cnt > 0:
            mean_g = float(valid_grades.mean())
            std_g = float(valid_grades.std()) if cnt > 1 else 0.0
            sem_g = float(std_g / np.sqrt(cnt))
        else:
            mean_g = np.nan
            std_g = np.nan
            sem_g = np.nan

        records.append(
            {
                "bin_center": b_center,
                "bin_min": b_min,
                "bin_max": b_max,
                "domain": dom_a if is_domain_a else dom_b,
                "sample_count": cnt,
                "mean_grade": mean_g,
                "std_grade": std_g,
                "sem_grade": sem_g,
            }
        )

    profile_df = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Decision Rule: Hard vs. Soft vs. Semi-Soft
    # -------------------------------------------------------------------------
    # Find innermost valid bins adjacent to contact (left and right of 0)
    left_sub = profile_df[profile_df["bin_center"] < 0].dropna(subset=["mean_grade"])
    right_sub = profile_df[profile_df["bin_center"] > 0].dropna(subset=["mean_grade"])

    if len(left_sub) > 0 and len(right_sub) > 0:
        # Closest bin on domain A side (max bin_center < 0)
        g_a_contact = float(left_sub.iloc[-1]["mean_grade"])
        sem_a = float(left_sub.iloc[-1]["sem_grade"])
        # Closest bin on domain B side (min bin_center > 0)
        g_b_contact = float(right_sub.iloc[0]["mean_grade"])
        sem_b = float(right_sub.iloc[0]["sem_grade"])

        step_change = abs(g_b_contact - g_a_contact)
        base_g = min(g_a_contact, g_b_contact)
        step_ratio = step_change / base_g if base_g > 0 else 1.0

        # Uncertainty threshold: step must exceed combined standard error
        combined_sem = (
            np.sqrt(sem_a**2 + sem_b**2) if (sem_a > 0 or sem_b > 0) else 0.01
        )

        if step_ratio >= 0.40 and step_change > 1.5 * combined_sem:
            boundary_type = "Hard"
            recommendation = (
                "HARD BOUNDARY: Discontinuous step change at contact. "
                "Composites must be strictly segregated during kriging (no sample sharing)."
            )
        elif step_ratio <= 0.15:
            boundary_type = "Soft"
            recommendation = (
                "SOFT BOUNDARY: Continuous transitional gradient across contact. "
                "Mineralization cross-cuts contact; samples can be freely shared across domains."
            )
        else:
            boundary_type = "Semi-Soft"
            recommendation = (
                "SEMI-SOFT / TRANSITIONAL BOUNDARY: Moderate transition zone across contact. "
                "Share samples only within a restricted buffer envelope or apply distance decay."
            )
    else:
        step_change = np.nan
        step_ratio = np.nan
        boundary_type = "Indeterminate"
        recommendation = "Insufficient data near contact to determine boundary type."

    profile_df.attrs["boundary_type"] = boundary_type
    profile_df.attrs["step_change"] = step_change
    profile_df.attrs["step_ratio"] = step_ratio
    profile_df.attrs["recommendation"] = recommendation
    profile_df.attrs["domain_a"] = dom_a
    profile_df.attrs["domain_b"] = dom_b
    profile_df.attrs["bin_width"] = bin_width
    return profile_df


def plot_contact_profile(
    contact_df: pd.DataFrame,
    domain_a_name: Optional[str] = None,
    domain_b_name: Optional[str] = None,
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 6.5),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates the industry-standard Contact Profile Plot (Grade vs. Distance from Contact).

    Mandatory deliverable for NI 43-101 / JORC Section 14 to document whether
    geological contacts are treated as Hard, Soft, or Semi-Soft boundaries.

    Parameters
    ----------
    contact_df : pd.DataFrame
        Output of contact_profile_analysis.
    domain_a_name, domain_b_name : str, optional
        Custom display names for domains. Defaults to values in contact_df.attrs.
    grade_unit : str, default "% Cu"
        Grade unit label.
    title : str, optional
        Figure title.
    figsize : tuple of float, default (11.0, 6.5)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and (ax_profile, ax_counts) axes.
    """
    dom_a = domain_a_name or str(
        contact_df.attrs.get("domain_a", "Domain A (Host Rock)")
    )
    dom_b = domain_b_name or str(contact_df.attrs.get("domain_b", "Domain B (Deposit)"))
    b_type = str(contact_df.attrs.get("boundary_type", "Indeterminate"))
    step_val = contact_df.attrs.get("step_change", np.nan)
    step_ratio = contact_df.attrs.get("step_ratio", np.nan)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Split into Domain A (d < 0) and Domain B (d > 0)
    df_a = contact_df[contact_df["bin_center"] < 0].dropna(subset=["mean_grade"])
    df_b = contact_df[contact_df["bin_center"] > 0].dropna(subset=["mean_grade"])

    # -------------------------------------------------------------------------
    # Panel 1: Grade Profile with Error Bars
    # -------------------------------------------------------------------------
    col_a = "#2ca02c"  # Forest Green
    col_b = "#d62728"  # Crimson Red

    if len(df_a) > 0:
        ax1.errorbar(
            df_a["bin_center"],
            df_a["mean_grade"],
            yerr=df_a["sem_grade"],
            fmt="-o",
            color=col_a,
            linewidth=2.0,
            markersize=6,
            capsize=3,
            label=f"{dom_a} (Host)",
        )

    if len(df_b) > 0:
        ax1.errorbar(
            df_b["bin_center"],
            df_b["mean_grade"],
            yerr=df_b["sem_grade"],
            fmt="-s",
            color=col_b,
            linewidth=2.0,
            markersize=6,
            capsize=3,
            label=f"{dom_b} (Mineralized)",
        )

    # Vertical Contact Line at x = 0
    ax1.axvline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="Geological Contact (d=0m)",
    )

    # Step-change callout annotation at contact
    if len(df_a) > 0 and len(df_b) > 0 and not np.isnan(step_val):
        g_a_end = float(df_a.iloc[-1]["mean_grade"])
        g_b_start = float(df_b.iloc[0]["mean_grade"])
        mid_y = (g_a_end + g_b_start) / 2.0

        ax1.annotate(
            f"Step Jump Δ: {step_val:.2f} {grade_unit}\n({step_ratio*100:.1f}% relative)",
            xy=(0.0, mid_y),
            xytext=(10.0, mid_y),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#ffffcc",
                edgecolor="gray",
                alpha=0.9,
            ),
            fontsize=9,
            fontweight="bold",
        )

    # Ensure generous headroom on Y-axis so neither the decision badge nor the legend
    # collides with the contact profile lines or error bars.
    y_min, y_max = ax1.get_ylim()
    y_span = max(y_max - y_min, 1e-4)
    ax1.set_ylim(max(0.0, y_min - 0.05 * y_span), y_max + 0.35 * y_span)

    # Boundary Type Decision Badge
    if b_type == "Hard":
        badge_color = "#ffcccc"
        badge_edge = "#d62728"
        badge_text = (
            "DECISION: HARD BOUNDARY\n(Strict Segregation: No Cross-Boundary Samples)"
        )
    elif b_type == "Soft":
        badge_color = "#ccffcc"
        badge_edge = "#2ca02c"
        badge_text = (
            "DECISION: SOFT BOUNDARY\n(Free Sample Sharing Permitted Across Contact)"
        )
    elif b_type == "Semi-Soft":
        badge_color = "#fff2cc"
        badge_edge = "#ff7f0e"
        badge_text = (
            "DECISION: SEMI-SOFT BOUNDARY\n(Restricted Buffer Sharing Recommended)"
        )
    else:
        badge_color = "#f0f0f0"
        badge_edge = "gray"
        badge_text = "DECISION: INDETERMINATE"

    ax1.text(
        0.03,
        0.95,
        badge_text,
        transform=ax1.transAxes,
        fontsize=9.0,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor=badge_color,
            edgecolor=badge_edge,
            linewidth=1.5,
        ),
    )

    ax1.set_ylabel(f"Average Grade ({grade_unit})", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.0)

    plot_title = title or f"Geological Contact Profile Analysis: {dom_a} vs. {dom_b}"
    ax1.set_title(plot_title, fontsize=12, fontweight="bold", pad=10)

    # -------------------------------------------------------------------------
    # Panel 2: Sample Count Data Support per Distance Bin
    # -------------------------------------------------------------------------
    bin_w = float(contact_df.attrs.get("bin_width", 2.0))
    bar_colors = [col_a if c < 0 else col_b for c in contact_df["bin_center"]]

    ax2.bar(
        contact_df["bin_center"],
        contact_df["sample_count"],
        width=bin_w * 0.85,
        color=bar_colors,
        edgecolor="black",
        alpha=0.7,
    )
    ax2.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Composites", fontsize=10, fontweight="bold")
    ax2.set_xlabel(
        "Signed Distance from Contact Surface (m)", fontsize=11, fontweight="bold"
    )
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig, (ax1, ax2)


# =============================================================================
# STAGE 5: PRODUCTION RECONCILIATION (PARKER F1, F2, F3 MINE-TO-MILL FACTORS)
# =============================================================================


def reconcile_production_to_reserve(
    reserve_data: Union[pd.DataFrame, Dict[str, float]],
    plant_data: Union[pd.DataFrame, Dict[str, float]],
    grade_control_data: Optional[Union[pd.DataFrame, Dict[str, float]]] = None,
    tonnes_col: str = "tonnes",
    grade_col: str = "grade",
    period_col: Optional[str] = None,
    grade_unit: str = "% Cu",
) -> pd.DataFrame:
    """Reconciles mine production against the long-term mineral reserve model.

    Implements the Harry Parker (2012) F1, F2, F3 reconciliation framework:
    - F1 Factor (Model to Mine / Ore Selection):
      F1 = Metal(Grade Control) / Metal(Reserve)
      Measures reserve model accuracy and local estimation bias.
    - F2 Factor (Mine to Mill / Delivery Efficiency):
      F2 = Metal(Plant Received) / Metal(Grade Control)
      Measures mining execution: unplanned dilution, ore loss, and misrouting.
    - F3 Factor (Total System Reconciliation):
      F3 = F1 * F2 = Metal(Plant Received) / Metal(Reserve)
      Measures total value chain health and cash-flow delivery.

    Also calculates component ratios for every stage:
    - Tonnage Ratio (R_T): T_actual / T_pred (high R_T flags excess dilution)
    - Grade Ratio (R_G): g_actual / g_pred (low R_G confirms dilution)
    - Metal Ratio (R_M): M_actual / M_pred = R_T * R_G

    Parameters
    ----------
    reserve_data : pd.DataFrame or dict
        Predicted reserve model feed (tonnes, grade).
    plant_data : pd.DataFrame or dict
        Actual received plant/mill feed (weightometer tonnes, assayed head grade).
    grade_control_data : pd.DataFrame or dict, optional
        Short-term grade control / blasthole model (delineated/trucked ore).
    tonnes_col : str, default "tonnes"
        Tonnage column name.
    grade_col : str, default "grade"
        Grade column name.
    period_col : str, optional
        Production period column (e.g. "month", "quarter", "year").
        If omitted or single record, treats data as a single global reconciliation.
    grade_unit : str, default "% Cu"
        Grade unit label.

    Returns
    -------
    pd.DataFrame
        Reconciliation summary table per period (plus Total row) with F1, F2, F3
        factors and component ratios. Attributes (.attrs) contain cumulative metrics
        and the value chain health diagnosis.
    """

    def _to_df(data: Union[pd.DataFrame, Dict[str, float]]) -> pd.DataFrame:
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return data.copy()

    df_res = _to_df(reserve_data)
    df_plant = _to_df(plant_data)
    has_gc = grade_control_data is not None
    df_gc = _to_df(grade_control_data) if has_gc else None

    # Handle period identifier
    if period_col is None or period_col not in df_res.columns:
        p_col = "period"
        df_res[p_col] = [
            f"P{i+1}" if len(df_res) > 1 else "Total" for i in range(len(df_res))
        ]
        df_plant[p_col] = df_res[p_col].values
        if has_gc:
            df_gc[p_col] = df_res[p_col].values
    else:
        p_col = period_col

    for df_chk, name in [(df_res, "reserve"), (df_plant, "plant")]:
        for col in (tonnes_col, grade_col):
            if col not in df_chk.columns:
                raise ValueError(f"Column '{col}' not found in {name} data.")
    if has_gc:
        for col in (tonnes_col, grade_col):
            if col not in df_gc.columns:
                raise ValueError(f"Column '{col}' not found in grade_control data.")

    grade_scale = 100.0 if "%" in grade_unit else 1.0

    # Ensure period ordering matches
    periods = df_res[p_col].tolist()
    records = []

    for p in periods:
        r_row = df_res[df_res[p_col] == p].iloc[0]
        p_row = df_plant[df_plant[p_col] == p].iloc[0]

        t_res = float(r_row[tonnes_col])
        g_res = float(r_row[grade_col])
        m_res = t_res * (g_res / grade_scale)

        t_plant = float(p_row[tonnes_col])
        g_plant = float(p_row[grade_col])
        m_plant = t_plant * (g_plant / grade_scale)

        rec = {
            "period": p,
            "reserve_tonnes": t_res,
            "reserve_grade": g_res,
            "reserve_metal": m_res,
            "plant_tonnes": t_plant,
            "plant_grade": g_plant,
            "plant_metal": m_plant,
        }

        if has_gc:
            gc_row = df_gc[df_gc[p_col] == p].iloc[0]
            t_gc = float(gc_row[tonnes_col])
            g_gc = float(gc_row[grade_col])
            m_gc = t_gc * (g_gc / grade_scale)

            rec["gc_tonnes"] = t_gc
            rec["gc_grade"] = g_gc
            rec["gc_metal"] = m_gc

            # F1: Reserve to Grade Control
            rec["f1_tonnes_ratio"] = t_gc / t_res if t_res > 0 else 1.0
            rec["f1_grade_ratio"] = g_gc / g_res if g_res > 0 else 1.0
            rec["f1_metal_factor"] = m_gc / m_res if m_res > 0 else 1.0

            # F2: Grade Control to Plant
            rec["f2_tonnes_ratio"] = t_plant / t_gc if t_gc > 0 else 1.0
            rec["f2_grade_ratio"] = g_plant / g_gc if g_gc > 0 else 1.0
            rec["f2_metal_factor"] = m_plant / m_gc if m_gc > 0 else 1.0

        # F3: Reserve to Plant (Total)
        rec["f3_tonnes_ratio"] = t_plant / t_res if t_res > 0 else 1.0
        rec["f3_grade_ratio"] = g_plant / g_res if g_res > 0 else 1.0
        rec["f3_metal_factor"] = m_plant / m_res if m_res > 0 else 1.0

        records.append(rec)

    res_df = pd.DataFrame(records)

    # Compute Overall Total Row if multi-period
    if len(res_df) > 1:
        tot_t_res = float(res_df["reserve_tonnes"].sum())
        tot_m_res = float(res_df["reserve_metal"].sum())
        tot_g_res = (tot_m_res / tot_t_res) * grade_scale if tot_t_res > 0 else 0.0

        tot_t_plant = float(res_df["plant_tonnes"].sum())
        tot_m_plant = float(res_df["plant_metal"].sum())
        tot_g_plant = (
            (tot_m_plant / tot_t_plant) * grade_scale if tot_t_plant > 0 else 0.0
        )

        tot_rec = {
            "period": "Total",
            "reserve_tonnes": tot_t_res,
            "reserve_grade": tot_g_res,
            "reserve_metal": tot_m_res,
            "plant_tonnes": tot_t_plant,
            "plant_grade": tot_g_plant,
            "plant_metal": tot_m_plant,
        }

        if has_gc:
            tot_t_gc = float(res_df["gc_tonnes"].sum())
            tot_m_gc = float(res_df["gc_metal"].sum())
            tot_g_gc = (tot_m_gc / tot_t_gc) * grade_scale if tot_t_gc > 0 else 0.0

            tot_rec["gc_tonnes"] = tot_t_gc
            tot_rec["gc_grade"] = tot_g_gc
            tot_rec["gc_metal"] = tot_m_gc

            tot_rec["f1_tonnes_ratio"] = tot_t_gc / tot_t_res if tot_t_res > 0 else 1.0
            tot_rec["f1_grade_ratio"] = tot_g_gc / tot_g_res if tot_g_res > 0 else 1.0
            tot_rec["f1_metal_factor"] = tot_m_gc / tot_m_res if tot_m_res > 0 else 1.0

            tot_rec["f2_tonnes_ratio"] = tot_t_plant / tot_t_gc if tot_t_gc > 0 else 1.0
            tot_rec["f2_grade_ratio"] = tot_g_plant / tot_g_gc if tot_g_gc > 0 else 1.0
            tot_rec["f2_metal_factor"] = tot_m_plant / tot_m_gc if tot_m_gc > 0 else 1.0

        tot_rec["f3_tonnes_ratio"] = tot_t_plant / tot_t_res if tot_t_res > 0 else 1.0
        tot_rec["f3_grade_ratio"] = tot_g_plant / tot_g_res if tot_g_res > 0 else 1.0
        tot_rec["f3_metal_factor"] = tot_m_plant / tot_m_res if tot_m_res > 0 else 1.0

        res_df = pd.concat([res_df, pd.DataFrame([tot_rec])], ignore_index=True)

    # Attach summary attributes
    final_row = res_df.iloc[-1]
    f3_tot = float(final_row["f3_metal_factor"])
    f1_tot = float(final_row["f1_metal_factor"]) if has_gc else None
    f2_tot = float(final_row["f2_metal_factor"]) if has_gc else None

    if 0.95 <= f3_tot <= 1.05:
        health_status = "EXCELLENT: Production is within +/-5% of reserve model (Bankable benchmark)."
    elif 0.90 <= f3_tot <= 1.10:
        health_status = "GOOD: Production is within +/-10% of reserve model."
    elif f3_tot < 0.90:
        health_status = "WARNING: Metal under-performance (>10% deficit vs. reserve model). Check dilution or over-smoothing."
    else:
        health_status = "WARNING: Metal over-performance (>10% surplus vs. reserve model). Check conservative bias or unmodeled ore."

    res_df.attrs["f3_factor"] = f3_tot
    if has_gc:
        res_df.attrs["f1_factor"] = f1_tot
        res_df.attrs["f2_factor"] = f2_tot
    res_df.attrs["health_status"] = health_status
    res_df.attrs["grade_unit"] = grade_unit
    return res_df


def plot_production_reconciliation(
    reconciliation_df: pd.DataFrame,
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    metal_unit: str = "kt",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14.0, 9.0),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates the industry-standard 4-panel Production Reconciliation Dashboard.

    Visualizes:
    1. Ore Tonnage Comparison (Reserve vs. Grade Control vs. Plant Feed)
    2. Head Grade Comparison
    3. Contained Metal Comparison
    4. Harry Parker F1, F2, F3 Factors tracking over time against the [0.95, 1.05] benchmark band.

    Parameters
    ----------
    reconciliation_df : pd.DataFrame
        Output of reconcile_production_to_reserve.
    grade_unit : str, default "% Cu"
        Grade unit label.
    tonnage_unit : str, default "Mt"
        Tonnage unit label.
    metal_unit : str, default "kt"
        Contained metal unit label.
    title : str, optional
        Overall dashboard title.
    figsize : tuple of float, default (14.0, 9.0)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and flattened axes array (ax_t, ax_g, ax_m, ax_f).
    """
    # Exclude "Total" row if multi-period for time plots
    has_total = "Total" in reconciliation_df["period"].values
    if has_total and len(reconciliation_df) > 1:
        plot_df = reconciliation_df[reconciliation_df["period"] != "Total"].copy()
    else:
        plot_df = reconciliation_df.copy()

    has_gc = "gc_tonnes" in plot_df.columns
    periods = plot_df["period"].astype(str).tolist()
    n_p = len(periods)
    x = np.arange(n_p)

    width = 0.26 if has_gc else 0.38

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_t, ax_g = axes[0, 0], axes[0, 1]
    ax_m, ax_f = axes[1, 0], axes[1, 1]

    col_res = "#1f77b4"  # Blue (Reserve Model)
    col_gc = "#ff7f0e"  # Orange (Grade Control)
    col_plant = "#2ca02c"  # Green (Plant Feed)

    # -------------------------------------------------------------------------
    # Panel 1: Ore Tonnage
    # -------------------------------------------------------------------------
    if has_gc:
        ax_t.bar(
            x - width,
            plot_df["reserve_tonnes"],
            width,
            label="Reserve Model",
            color=col_res,
            edgecolor="black",
            alpha=0.85,
        )
        ax_t.bar(
            x,
            plot_df["gc_tonnes"],
            width,
            label="Grade Control",
            color=col_gc,
            edgecolor="black",
            alpha=0.85,
        )
        ax_t.bar(
            x + width,
            plot_df["plant_tonnes"],
            width,
            label="Plant Feed",
            color=col_plant,
            edgecolor="black",
            alpha=0.85,
        )
    else:
        ax_t.bar(
            x - width / 2,
            plot_df["reserve_tonnes"],
            width,
            label="Reserve Model",
            color=col_res,
            edgecolor="black",
            alpha=0.85,
        )
        ax_t.bar(
            x + width / 2,
            plot_df["plant_tonnes"],
            width,
            label="Plant Feed",
            color=col_plant,
            edgecolor="black",
            alpha=0.85,
        )

    ax_t.set_ylabel(f"Ore Tonnage ({tonnage_unit})", fontsize=10, fontweight="bold")
    ax_t.set_title("Ore Tonnage Reconciliation", fontsize=11, fontweight="bold")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(periods, fontsize=9)
    ax_t.grid(True, linestyle=":", alpha=0.5)
    ax_t.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 2: Head Grade
    # -------------------------------------------------------------------------
    if has_gc:
        ax_g.bar(
            x - width,
            plot_df["reserve_grade"],
            width,
            label="Reserve Model",
            color=col_res,
            edgecolor="black",
            alpha=0.85,
        )
        ax_g.bar(
            x,
            plot_df["gc_grade"],
            width,
            label="Grade Control",
            color=col_gc,
            edgecolor="black",
            alpha=0.85,
        )
        ax_g.bar(
            x + width,
            plot_df["plant_grade"],
            width,
            label="Plant Feed",
            color=col_plant,
            edgecolor="black",
            alpha=0.85,
        )
    else:
        ax_g.bar(
            x - width / 2,
            plot_df["reserve_grade"],
            width,
            label="Reserve Model",
            color=col_res,
            edgecolor="black",
            alpha=0.85,
        )
        ax_g.bar(
            x + width / 2,
            plot_df["plant_grade"],
            width,
            label="Plant Feed",
            color=col_plant,
            edgecolor="black",
            alpha=0.85,
        )

    ax_g.set_ylabel(f"Head Grade ({grade_unit})", fontsize=10, fontweight="bold")
    ax_g.set_title("Head Grade Reconciliation", fontsize=11, fontweight="bold")
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(periods, fontsize=9)
    ax_g.grid(True, linestyle=":", alpha=0.5)
    ax_g.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Contained Metal
    # -------------------------------------------------------------------------
    if has_gc:
        ax_m.bar(
            x - width,
            plot_df["reserve_metal"],
            width,
            label="Reserve Model",
            color=col_res,
            edgecolor="black",
            alpha=0.85,
        )
        ax_m.bar(
            x,
            plot_df["gc_metal"],
            width,
            label="Grade Control",
            color=col_gc,
            edgecolor="black",
            alpha=0.85,
        )
        ax_m.bar(
            x + width,
            plot_df["plant_metal"],
            width,
            label="Plant Feed",
            color=col_plant,
            edgecolor="black",
            alpha=0.85,
        )
    else:
        ax_m.bar(
            x - width / 2,
            plot_df["reserve_metal"],
            width,
            label="Reserve Model",
            color=col_res,
            edgecolor="black",
            alpha=0.85,
        )
        ax_m.bar(
            x + width / 2,
            plot_df["plant_metal"],
            width,
            label="Plant Feed",
            color=col_plant,
            edgecolor="black",
            alpha=0.85,
        )

    ax_m.set_ylabel(f"Contained Metal ({metal_unit})", fontsize=10, fontweight="bold")
    ax_m.set_title("Contained Metal Reconciliation", fontsize=11, fontweight="bold")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(periods, fontsize=9)
    ax_m.grid(True, linestyle=":", alpha=0.5)
    ax_m.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Harry Parker F1, F2, F3 Factors Tracking
    # -------------------------------------------------------------------------
    # Benchmark target band: [0.95, 1.05] shaded green
    ax_f.axhspan(0.95, 1.05, color="#2ca02c", alpha=0.15, label="Target Band (±5%)")
    ax_f.axhline(1.00, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    if n_p > 1:
        if has_gc:
            ax_f.plot(
                x,
                plot_df["f1_metal_factor"],
                "-o",
                color=col_res,
                linewidth=2.0,
                markersize=6,
                label="F1 (Model → Mine)",
            )
            ax_f.plot(
                x,
                plot_df["f2_metal_factor"],
                "-s",
                color=col_gc,
                linewidth=2.0,
                markersize=6,
                label="F2 (Mine → Mill)",
            )
        ax_f.plot(
            x,
            plot_df["f3_metal_factor"],
            "-^",
            color=col_plant,
            linewidth=2.5,
            markersize=7,
            label="F3 (Total Value Chain)",
        )
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(periods, fontsize=9)
    else:
        # Single period bar representation
        cats = (
            ["F1 (Model→Mine)", "F2 (Mine→Mill)", "F3 (Total)"]
            if has_gc
            else ["F3 (Total)"]
        )
        vals = (
            [
                float(plot_df["f1_metal_factor"].iloc[0]),
                float(plot_df["f2_metal_factor"].iloc[0]),
                float(plot_df["f3_metal_factor"].iloc[0]),
            ]
            if has_gc
            else [float(plot_df["f3_metal_factor"].iloc[0])]
        )
        bar_c = [col_res, col_gc, col_plant] if has_gc else [col_plant]
        ax_f.bar(
            range(len(cats)),
            vals,
            width=0.45,
            color=bar_c,
            edgecolor="black",
            alpha=0.85,
        )
        ax_f.set_xticks(range(len(cats)))
        ax_f.set_xticklabels(cats, fontsize=9, fontweight="bold")
        for idx, v in enumerate(vals):
            ax_f.text(
                idx, v + 0.02, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold"
            )

    ax_f.set_ylabel("Reconciliation Factor (Ratio)", fontsize=10, fontweight="bold")
    ax_f.set_title(
        "Harry Parker F1, F2, F3 Performance Factors", fontsize=11, fontweight="bold"
    )
    ax_f.grid(True, linestyle=":", alpha=0.5)
    ax_f.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Health diagnosis annotation
    health_txt = str(reconciliation_df.attrs.get("health_status", ""))
    if health_txt:
        ax_f.text(
            0.03,
            0.06,
            health_txt,
            transform=ax_f.transAxes,
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#ffffcc",
                edgecolor="gray",
                alpha=0.9,
            ),
        )

    dashboard_title = (
        title or "Mine-to-Mill Production Reconciliation Dashboard (Parker F-Factors)"
    )
    fig.suptitle(dashboard_title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig, axes.flatten()


# =============================================================================
# GEOSTATISTICAL CONDITIONAL SIMULATION & UNCERTAINTY (E-TYPE & M-TYPE)
# =============================================================================


def sequential_gaussian_simulation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    sill: float,
    range_param: float,
    n_realizations: int = 50,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generates equiprobable conditional realizations via Sequential Gaussian Simulation (SGS).

    Sequential Gaussian Simulation (Isaaks 1990; Deutsch & Journel 1998; SME Handbook)
    models spatial uncertainty by generating multiple stochastic realizations that honor:
    1. Conditioning sample values at drillhole locations.
    2. The experimental univariate histogram (via normal-score transformation).
    3. The spatial covariance / variogram structure (retaining short-range variance).

    Unlike linear kriging, each realization reproduces the realistic spatial heterogeneity,
    nugget variance, and extreme values of the deposit without artificial smoothing.

    Parameters
    ----------
    samples_xy : np.ndarray
        Array of shape (N, 2) with conditioning sample coordinates.
    sample_grades : np.ndarray
        Array of shape (N,) with conditioning assay grades.
    grid_points : np.ndarray
        Array of shape (M, 2) with target simulation grid coordinates.
    sill : float
        Variogram sill (C) in Gaussian space.
    range_param : float
        Variogram range (a).
    n_realizations : int, default 50
        Number of equiprobable stochastic realizations to generate.
    variogram_model : str, default "spherical"
        Spatial correlation structure ("spherical", "exponential", "gaussian").
    nugget : float, default 0.0
        Nugget variance (C0).
    seed : int, optional
        Random number generator seed for reproducible simulations.

    Returns
    -------
    np.ndarray
        Array of shape (M, n_realizations) containing the simulated realizations.
    """
    raise NotImplementedError(
        "Sequential Gaussian Simulation (SGS) engine is planned. "
        "Use compute_etype_mtype_maps() on pre-computed realization arrays."
    )


def compute_etype_mtype_maps(
    realizations: np.ndarray,
    cutoff_grade: Optional[float] = None,
    percentiles: Tuple[float, float] = (10.0, 90.0),
) -> pd.DataFrame:
    """Computes E-Type (mean), M-Type (median), and spatial uncertainty metrics from realizations.

    Definitions:
    ------------
    1. E-Type (Conditional Expectation):
       e_type(x) = (1 / L) * sum_{l=1}^L Z^{(l)}(x)
       Pointwise average across all realizations. In a multi-Gaussian framework,
       the E-type map asymptotically converges to the Simple Kriging estimate.
    2. M-Type (Conditional Median / P50):
       m_type(x) = median({Z^{(1)}(x), ..., Z^{(L)}(x)})
       Pointwise 50th percentile. More robust than E-type in highly skewed,
       high-nugget deposits (e.g. epithermal gold) where extreme tail simulations
       would otherwise distort the expectation.
    3. Conditional Variance & Standard Deviation:
       Quantifies local estimation uncertainty and risk at each grid point.
    4. Probability of Exceedance:
       P(Z(x) >= cutoff) = (1 / L) * sum_{l=1}^L I(Z^{(l)}(x) >= cutoff)
       Evaluates the spatial probability that a block exceeds the economic cutoff grade.

    Theoretical Limitations of E-Type Mapping:
    -------------------------------------------
    E-type maps are mathematically smoothed conditional expectations. Because averaging
    suppresses spatial variance (Var(E-type) << Var(Realization)), E-type estimates:
    - Underestimate high grades and overestimate low grades (re-introducing Kriging smoothing).
    - Fail to reproduce extreme values and short-scale spatial variability (nugget effect).
    - Fail to reproduce higher-order geological structures (e.g., connected high-permeability
      channels or fault boundaries).
    IMPORTANT: Non-linear transfer functions (such as pit optimization, flow modeling, or
    mill blending simulations) must be evaluated individually on each realization and then
    summarized, rather than evaluated once on the smoothed E-type map.

    Parameters
    ----------
    realizations : np.ndarray
        Array of shape (M, L) where M is the number of grid points and L is the
        number of equiprobable realizations.
    cutoff_grade : float, optional
        Economic cut-off grade for calculating Probability of Exceedance P(Z >= cutoff).
    percentiles : tuple of (float, float), default (10.0, 90.0)
        Lower and upper percentiles for uncertainty confidence bounds.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - "e_type": Conditional expectation (mean).
        - "m_type": Conditional median (P50).
        - "conditional_std": Local standard deviation (uncertainty).
        - "conditional_var": Local variance.
        - "p_lower": Lower confidence percentile (default P10).
        - "p_upper": Upper confidence percentile (default P90).
        - "prob_exceedance": P(Z >= cutoff) if cutoff_grade is provided.
        Attributes (.attrs) contain the smoothing ratio and regulatory audit notes.
    """
    realizations = np.asarray(realizations, dtype=float)
    if realizations.ndim != 2:
        raise ValueError(
            f"Expected a 2D array of shape (n_points, n_realizations), got {realizations.shape}."
        )

    n_pts, n_real = realizations.shape
    if n_real < 2:
        raise ValueError(
            f"Need at least 2 realizations to compute statistics, got {n_real}."
        )

    p_low, p_high = percentiles
    e_type = np.mean(realizations, axis=1)
    m_type = np.median(realizations, axis=1)
    cond_var = np.var(realizations, axis=1, ddof=1)
    cond_std = np.sqrt(cond_var)
    p_lower_vals = np.percentile(realizations, p_low, axis=1)
    p_upper_vals = np.percentile(realizations, p_high, axis=1)

    data = {
        "e_type": e_type,
        "m_type": m_type,
        "conditional_std": cond_std,
        "conditional_var": cond_var,
        f"p{int(p_low)}": p_lower_vals,
        f"p{int(p_high)}": p_upper_vals,
    }

    if cutoff_grade is not None:
        data["prob_exceedance"] = np.mean(realizations >= cutoff_grade, axis=1)

    df_out = pd.DataFrame(data)

    # Calculate smoothing ratio: Var(E-type) / Mean(Var(Realization))
    realization_variances = np.var(realizations, axis=0, ddof=1)
    mean_real_var = (
        float(np.mean(realization_variances)) if len(realization_variances) > 0 else 1.0
    )
    etype_var = float(np.var(e_type, ddof=1))
    smoothing_ratio = etype_var / mean_real_var if mean_real_var > 0 else 1.0

    df_out.attrs["smoothing_ratio"] = smoothing_ratio
    df_out.attrs["n_realizations"] = n_real
    df_out.attrs["n_points"] = n_pts
    df_out.attrs["limitations_note"] = (
        "E-type maps are smoothed conditional expectations. They fail to reproduce extreme "
        "grades or short-range spatial variance. Non-linear processes (e.g. pit limits, "
        "flow simulation) should be run directly on individual realizations."
    )
    return df_out


def plot_simulation_realizations_dashboard(
    realizations: np.ndarray,
    grid_xy: np.ndarray,
    cutoff_grade: Optional[float] = None,
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14.0, 10.0),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates a 4-panel comparison dashboard of simulation realizations vs. E-type map.

    Visualizes:
    1. Realization 1 (showing full spatial variance, texture, and extreme values).
    2. Realization 2 (an alternative equiprobable stochastic outcome).
    3. E-Type Map (the smoothed conditional expectation).
    4. Conditional Uncertainty / Probability of Exceedance Map.

    Parameters
    ----------
    realizations : np.ndarray
        Array of shape (M, L) with simulated realizations.
    grid_xy : np.ndarray
        Array of shape (M, 2) with spatial coordinates.
    cutoff_grade : float, optional
        Cut-off grade for probability of exceedance map.
    grade_unit : str, default "% Cu"
        Grade unit label.
    title : str, optional
        Dashboard title.
    figsize : tuple of float, default (14.0, 10.0)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and axes sequence.
    """
    raise NotImplementedError(
        "Simulation realizations dashboard plotting stub. "
        "Will visualize realizations vs. smoothed E-type map."
    )

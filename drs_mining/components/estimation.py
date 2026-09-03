"""Polygonal mineral resource and reserve estimation.

Provides functional tools for geometric polygonal estimation (method of polygons
of influence), global reserve calculations, cutoff-grade sensitivity analysis
(grade-tonnage curves), and spatial 2D plan map visualizations.
"""

from __future__ import annotations

from collections import Counter
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
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    sill: float = 1.0,
    range_param: float = 100.0,
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
                unique_coords, inverse_indices = np.unique(coords_m.round(decimals=5), axis=0, return_inverse=True)
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
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    sill: float = 1.0,
    range_param: float = 100.0,
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
    # TODO: Implement Ordinary Kriging (OK) System:
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
                unique_coords, inverse_indices = np.unique(coords_m.round(decimals=5), axis=0, return_inverse=True)
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
        k0_m = _theoretical_covariance(
            d_m, variogram_model, nugget, sill, range_param
        )

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
    ax1.set_ylabel(f"Ore Tonnage ({tonnage_unit})", fontsize=11, fontweight="bold", color="#1f77b4")
    ax2.set_ylabel(f"Average Ore Grade ({grade_unit})", fontsize=11, fontweight="bold", color="#d62728")
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

    sensitivity_df = pd.DataFrame({
        "cell_size": cell_sizes_list,
        "declustered_mean": means,
        "declustered_variance": variances,
    })

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
        opt_matches = sensitivity_df[np.isclose(sensitivity_df["cell_size"], optimal_cell_size)]
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

    bias_pct = ((naive_mean - optimal_mean) / naive_mean) * 100.0 if naive_mean != 0 else 0.0
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
    ax.set_ylabel(f"Declustered Mean Grade ({grade_unit})", fontsize=11, fontweight="bold")
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
    standardized_cats = (
        valid_blocks[category_col].astype(str).str.strip().str.lower()
    )

    for key, formal_name in cat_names.items():
        sub = valid_blocks[standardized_cats == key]
        if len(sub) > 0:
            tonnes = float(sub[tonnes_col].sum())
            metal_raw = float((sub[tonnes_col] * sub[grade_col] * metal_factor).sum())
            grade = (
                float(metal_raw / (tonnes * metal_factor)) if tonnes > 0 else 0.0
            )
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
    mi_grade = (
        float(mi_metal / (mi_tonnes * metal_factor)) if mi_tonnes > 0 else 0.0
    )
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
        raise ValueError(f"No valid blocks with coordinates in '{coord_col}' and '{grade_col}'.")

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

    has_val = validation_grade_col is not None and validation_grade_col in valid_blocks.columns
    has_tonnes = tonnes_col is not None and tonnes_col in valid_blocks.columns

    for k in range(n_bins):
        low, high = bins[k], bins[k + 1]
        in_bin = valid_blocks[(valid_blocks[coord_col] >= low) & (valid_blocks[coord_col] < high)]
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
    if drillholes is not None and not drillholes.empty and coord_col in drillholes.columns:
        valid_dh = drillholes.dropna(subset=[coord_col, drillhole_grade_col])
        for k in range(n_bins):
            low, high = bins[k], bins[k + 1]
            in_bin_dh = valid_dh[(valid_dh[coord_col] >= low) & (valid_dh[coord_col] < high)]
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
    ax1.set_ylabel(f"Average Grade ({grade_unit})", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    plot_title = title if title else f"Swath Plot (Local Drift Analysis) Along {default_dir}"
    ax1.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9, fontsize=9)

    fig.tight_layout()
    return fig, ax1

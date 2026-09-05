"""Polygonal mineral resource and reserve estimation.

Provides functional tools for geometric polygonal estimation (method of polygons
of influence), global reserve calculations, cutoff-grade sensitivity analysis
(grade-tonnage curves), and spatial 2D plan map visualizations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import warnings
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.ops import voronoi_diagram
from scipy.spatial import KDTree, cKDTree, Delaunay
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy import stats


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: 2D planar Voronoi polygons of influence.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Functional approach: transparent DataFrames and NumPy arrays (no custom config/wrapper classes).
# - Domain segregation: must strictly respect domain boundaries (`domain_col`).
# - Extrapolation control: provide `clip_to_convex_hull=True` and `max_radius`.
# - Boundary vs Data Support: separate legal `boundary` from `is_within_convex_hull`.
# - No backwards-compatibility fallbacks: provide direct functional 3D block model estimator
#   (or update signature) assigning nearest composite to block centroids as the standard
#   declustered validation benchmark for comparative grade-tonnage audits and swath plots.
def polygonal_estimation(
    drillholes: pd.DataFrame,
    bulk_density: float,
    boundary: Optional[Sequence[Tuple[float, float]]] = None,
    max_radius: Optional[float] = None,
    clip_to_convex_hull: bool = False,
    grade_col: str = "grade",
    thickness_col: str = "thickness",
    x_col: str = "x",
    y_col: str = "y",
    hole_id_col: str = "hole_id",
    domain_col: Optional[str] = None,
    domain_boundaries: Optional[Mapping[Any, Sequence[Tuple[float, float]]]] = None,
    metal_factor: float = 0.01,
) -> pd.DataFrame:
    """Estimates in-situ mineral resources and volumes using the method of polygons of influence (Voronoi tessellation).

    Supports geological domain boundaries: when `domain_col` is provided, Voronoi tessellations
    are constructed strictly within each domain, ensuring polygons never cross geological contacts.

    Parameters
    ----------
    drillholes : pd.DataFrame
        DataFrame of exploration drill holes or blast holes with coordinates, assays, and intercept thicknesses.
    bulk_density : float
        Specific gravity / bulk density in tonnes per cubic meter (t/m^3).
        Site-specific parameter required without default.
    boundary : Sequence[Tuple[float, float]], optional
        Closed polygon coordinates [(x1, y1), (x2, y2), ...] defining concession or pit perimeter.
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
    domain_col : str, optional
        Column name for geological domain. When provided, polygons are strictly segregated
        by domain, honoring hard geological boundaries.
    domain_boundaries : Mapping[Any, Sequence[Tuple[float, float]]], optional
        Mapping from domain identifier to boundary polygon coordinates for that specific domain.
    metal_factor : float, default 0.01
        Multiplier converting (grade * tonnes) to raw metal quantity (e.g., tonnes of metal).
        Default is 0.01 for percentage grades (% Cu). For g/t or ppm, use 1e-6 (or 1.0 for grams).

    Returns
    -------
    pd.DataFrame
        Table with one row per drillhole polygon containing:
        - hole_id, x, y, grade, thickness
        - area_m2: Plan area of polygon of influence (m^2)
        - volume_m3: In-situ rock volume (area * thickness)
        - tonnes: Mineral mass (volume * bulk_density)
        - contained_metal: Quantity of metal (tonnes * grade * metal_factor)
        - vertices: List of (x, y) coordinates forming the polygon perimeter
        - domain: Geological domain identifier (if domain_col provided)
    """
    if bulk_density <= 0:
        raise ValueError("bulk_density must be positive.")

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

    # Handle domain-segregated polygonal estimation
    if domain_col is not None and domain_col in drillholes.columns:
        output_cols_dom = output_cols + [domain_col]
        if drillholes.empty:
            return pd.DataFrame(columns=output_cols_dom)

        dfs = []
        for dom, grp in drillholes.groupby(domain_col):
            dom_bound = domain_boundaries.get(dom) if domain_boundaries else boundary
            dom_df = polygonal_estimation(
                drillholes=grp,
                boundary=dom_bound,
                bulk_density=bulk_density,
                max_radius=max_radius,
                clip_to_convex_hull=clip_to_convex_hull,
                grade_col=grade_col,
                thickness_col=thickness_col,
                x_col=x_col,
                y_col=y_col,
                hole_id_col=hole_id_col,
                domain_col=None,
                metal_factor=metal_factor,
            )
            dom_df[domain_col] = dom
            dfs.append(dom_df)

        if not dfs:
            return pd.DataFrame(columns=output_cols_dom)
        return pd.concat(dfs, ignore_index=True)

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
        metal = tonnes * grade * metal_factor

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


def polygonal_resource_summary(
    df: pd.DataFrame,
    tonnes_col: str = "tonnes",
    grade_col: str = "grade",
    area_col: str = "area_m2",
    volume_col: str = "volume_m3",
    metal_factor: float = 0.01,
) -> dict[str, float]:
    """Calculates global in-situ resource metrics from an estimated polygon table.

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
    metal_factor : float, default 0.01
        Multiplier converting (grade * tonnes) to contained metal quantity.
        Default is 0.01 for percentage grades (% Cu).

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

    if "contained_metal" in df.columns:
        contained_metal = float(df["contained_metal"].sum())
    else:
        contained_metal = float((df[tonnes_col] * df[grade_col] * metal_factor).sum())

    mean_grade = (
        float((df[tonnes_col] * df[grade_col]).sum() / total_tonnes)
        if total_tonnes > 0.0
        else 0.0
    )

    return {
        "total_tonnes": total_tonnes,
        "mean_grade": mean_grade,
        "contained_metal": contained_metal,
        "total_area_m2": total_area,
        "total_volume_m3": total_volume,
        "drillhole_count": len(df),
        "mean_polygon_area_m2": total_area / len(df) if len(df) > 0 else 0.0,
    }


def format_polygonal_summary(
    summary: Mapping[str, float],
    grade_unit: str = "%",
    metal_unit: str = "units",
) -> str:
    """Formats the polygonal in-situ resource summary dictionary into an executive text table."""
    lines = [
        "=" * 64,
        "             POLYGONAL IN-SITU RESOURCE SUMMARY",
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
    metal_factor: float = 0.01,
    external_waste_tonnes: Optional[float] = None,
) -> pd.DataFrame:
    """Computes ore-waste distribution, internal subgrade ratio, and recovery across cutoff grades.

    In open-pit and underground mining (SME Mining Engineering Handbook Ch. 6.1 and
    CIM MRMR Best Practice Guidelines (Section 7 on Mineral Reserve Estimation and Internal Waste) (TODO: Manually Verify), sub-economic material located within the mineralized wireframe
    is internal sub-grade waste (internal_waste_ratio = internal_waste_tonnes / ore_tonnes).
    If external wall-rock or overburden tonnage (external_waste_tonnes) is provided,
    the true open-pit stripping ratio is calculated:
    strip_ratio = (external_waste_tonnes + internal_waste_tonnes) / ore_tonnes.

    Parameters
    ----------
    df : pd.DataFrame
        Polygonal or block model table containing tonnage and grade columns.
    cutoffs : Sequence[float]
        List of cutoff grades to evaluate (e.g. [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).
    grade_col : str, default "grade"
        Grade column name.
    tonnes_col : str, default "tonnes"
        Tonnage column name.
    metal_factor : float, default 0.01
        Multiplier converting (grade * tonnes) to contained metal quantity.
        Default is 0.01 for percentage grades (% Cu).
    external_waste_tonnes : float, optional
        External pit wall-rock, overburden, or interburden tonnage from pit optimization.
        If provided, computes overall open-pit strip_ratio.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by cutoff grade with columns:
        - ore_tonnes: Tonnage of mineral exceeding cutoff
        - ore_grade: Weighted average grade of ore
        - internal_waste_tonnes: Sub-grade material within wireframe
        - contained_metal: Metal quantity in ore
        - internal_waste_ratio: Internal sub-grade to ore tonnage ratio
        - ore_recovery_pct: Percentage of total deposit tonnage retained as ore
        - metal_recovery_pct: Percentage of total contained metal recovered in ore
        - strip_ratio: Overall open-pit stripping ratio (included if external_waste_tonnes is provided)
    """
    total_tonnes = float(df[tonnes_col].sum()) if not df.empty else 0.0
    total_metal = (
        float((df[tonnes_col] * df[grade_col] * metal_factor).sum())
        if not df.empty
        else 0.0
    )

    rows = []
    for c in sorted(cutoffs):
        ore_mask = df[grade_col] >= c
        ore_tonnes = float(df.loc[ore_mask, tonnes_col].sum())
        ore_metal = float(
            (
                df.loc[ore_mask, tonnes_col]
                * df.loc[ore_mask, grade_col]
                * metal_factor
            ).sum()
        )
        ore_grade = (
            float(
                (df.loc[ore_mask, tonnes_col] * df.loc[ore_mask, grade_col]).sum()
                / ore_tonnes
            )
            if ore_tonnes > 0.0
            else 0.0
        )
        int_waste_tonnes = max(0.0, total_tonnes - ore_tonnes)
        int_waste_ratio = (
            (int_waste_tonnes / ore_tonnes) if ore_tonnes > 0.0 else float("inf")
        )
        ore_rec = (ore_tonnes / total_tonnes * 100.0) if total_tonnes > 0.0 else 0.0
        metal_rec = (ore_metal / total_metal * 100.0) if total_metal > 0.0 else 0.0

        row_dict = {
            "cutoff": c,
            "ore_tonnes": ore_tonnes,
            "ore_grade": ore_grade,
            "internal_waste_tonnes": int_waste_tonnes,
            "contained_metal": ore_metal,
            "internal_waste_ratio": int_waste_ratio,
            "ore_recovery_pct": ore_rec,
            "metal_recovery_pct": metal_rec,
        }
        if external_waste_tonnes is not None:
            tot_waste = float(external_waste_tonnes) + int_waste_tonnes
            row_dict["strip_ratio"] = (
                (tot_waste / ore_tonnes) if ore_tonnes > 0.0 else float("inf")
            )

        rows.append(row_dict)

    res_df = pd.DataFrame(rows)
    return res_df.set_index("cutoff")


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: 2D plan map of Voronoi polygons.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Scope of this function: strictly 2D plan view mapping; add `bench_z: Optional[float] = None`
#   to render single horizontal mining bench slices.
# - Following codebase conventions, 3D visualizations are implemented as separate functions:
#   1. `plot_polygonal_3d_isometric`: dedicated static 3D isometric visualization (projection="3d").
#   2. `plot_polygonal_3d_interactive`: dedicated interactive 3D explorer with pan/zoom/rotation.
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


def compute_anisotropy_rotation_matrix(
    dim: int = 3,
    angles: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
) -> np.ndarray:
    """Computes an orthonormal rotation matrix for 2D or 3D spatial anisotropy.

    Following international geostatistical standards (CIM MRMR §6.8/§6.9, SME Handbook,
    Deutsch & Journel 1998, SGeMS) (TODO: Manually Verify):
    - In 2D:
      `azimuth` (degrees clockwise from North / +Y axis) or standard strike angle.
      Major axis is oriented along strike; minor axis is across strike.
      R = [[sin(a), cos(a)], [cos(a), -sin(a)]]
    - In 3D:
      1. `azimuth` (alpha): Strike angle clockwise from North (+Y) in horizontal plane.
      2. `dip` (beta): Downward inclination from horizontal (-90 to +90 deg).
      3. `plunge` (theta): Pitch/rake angle in the plane of structure (-90 to +90 deg).
      Basis vectors:
      - u1 (major): along strike (or plunging within strike plane)
      - u2 (semi-major): down-dip
      - u3 (minor): normal to structure (across thickness)
      R has rows [u1, u2, u3], so R @ x projects x onto [major, semi-major, minor].

    Parameters
    ----------
    dim : int, default 3
        Spatial dimension (2 or 3).
    angles : float, Sequence[float], or Mapping[str, float], optional
        In 2D: azimuth angle (float or dict with "azimuth"/"strike").
        In 3D: (azimuth, dip, plunge) tuple or dict with keys "azimuth", "dip", "plunge".
        Defaults to None (azimuth=0, dip=0, plunge=0).

    Returns
    -------
    np.ndarray
        Orthonormal rotation matrix of shape (dim, dim).
    """
    if dim == 2:
        if angles is None:
            azimuth = 0.0
        elif isinstance(angles, (int, float)):
            azimuth = float(angles)
        elif isinstance(angles, Mapping):
            azimuth = float(angles.get("azimuth", angles.get("strike", 0.0)))
        else:
            azimuth = float(angles[0]) if len(angles) > 0 else 0.0

        a = np.radians(azimuth)
        return np.array(
            [
                [np.sin(a), np.cos(a)],
                [np.cos(a), -np.sin(a)],
            ],
            dtype=float,
        )

    elif dim == 3:
        if angles is None:
            azimuth, dip, plunge = 0.0, 0.0, 0.0
        elif isinstance(angles, (int, float)):
            azimuth, dip, plunge = float(angles), 0.0, 0.0
        elif isinstance(angles, Mapping):
            azimuth = float(angles.get("azimuth", angles.get("strike", 0.0)))
            dip = float(angles.get("dip", 0.0))
            plunge = float(angles.get("plunge", angles.get("rake", 0.0)))
        else:
            ang = list(angles)
            azimuth = float(ang[0]) if len(ang) > 0 else 0.0
            dip = float(ang[1]) if len(ang) > 1 else 0.0
            plunge = float(ang[2]) if len(ang) > 2 else 0.0

        a = np.radians(azimuth)
        b = np.radians(dip)
        th = np.radians(plunge)

        # Strike vector (along azimuth in horizontal plane):
        u1 = np.array([np.sin(a), np.cos(a), 0.0], dtype=float)
        # Dip vector (perpendicular to strike, dipping downward by angle b):
        u2 = np.array(
            [np.cos(a) * np.cos(b), -np.sin(a) * np.cos(b), -np.sin(b)],
            dtype=float,
        )
        # Normal vector (minor axis, normal to the dipping plane):
        u3 = np.array(
            [np.cos(a) * np.sin(b), -np.sin(a) * np.sin(b), np.cos(b)],
            dtype=float,
        )

        # Plunge rotation in the (u1, u2) plane:
        if abs(th) > 1e-9:
            u1_p = np.cos(th) * u1 + np.sin(th) * u2
            u2_p = -np.sin(th) * u1 + np.cos(th) * u2
            u1, u2 = u1_p, u2_p

        return np.vstack([u1, u2, u3])
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")


def transform_anisotropic_coordinates(
    coords: np.ndarray,
    ranges: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
    angles: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
) -> tuple[np.ndarray, float]:
    """Transforms spatial coordinates into an isotropic equivalent space.

    Maps physical coordinates through rotation R and aspect-ratio scaling S such that
    Euclidean distances in the transformed space equal the anisotropic lag distance h_aniso:
    h_aniso = a_major * sqrt( (u_major / a_major)^2 + (u_semi / a_semi)^2 + (u_minor / a_minor)^2 )

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array of shape (..., 2) or (..., 3).
    ranges : float, Sequence[float], or Mapping[str, float], optional
        Directional correlation / search ranges:
        - 2D: (range_major, range_minor) or dict with "major", "minor".
        - 3D: (range_major, range_semi_major, range_minor) or dict with "major", "semi_major", "minor".
        If a single scalar float or None is provided, coordinates are treated as isotropic.
    angles : float, Sequence[float], or Mapping[str, float], optional
        Rotation angles:
        - 2D: azimuth angle (degrees clockwise from North).
        - 3D: (azimuth, dip, plunge) in degrees.

    Returns
    -------
    transformed_coords : np.ndarray
        Coordinates in isotropic equivalent metric space of same shape as coords.
    major_range : float
        The reference major range a_major.
    """
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.size == 0:
        def_range = (
            100.0
            if ranges is None
            else (float(ranges) if isinstance(ranges, (int, float)) else 100.0)
        )
        return coords_arr, def_range

    dim = coords_arr.shape[-1]
    if dim not in (2, 3):
        raise ValueError(f"Coordinates must have last dimension 2 or 3, got {dim}")

    # 1. Parse ranges
    if ranges is None:
        return coords_arr, 100.0
    if isinstance(ranges, (int, float)):
        return coords_arr, float(ranges)

    if isinstance(ranges, Mapping):
        if dim == 2:
            r_major = float(ranges.get("major", ranges.get("range_major", 100.0)))
            r_minor = float(ranges.get("minor", ranges.get("range_minor", r_major)))
            r_list = [r_major, r_minor]
        else:
            r_major = float(ranges.get("major", ranges.get("range_major", 100.0)))
            r_semi = float(
                ranges.get("semi_major", ranges.get("range_semi_major", r_major))
            )
            r_minor = float(ranges.get("minor", ranges.get("range_minor", r_semi)))
            r_list = [r_major, r_semi, r_minor]
    else:
        r_list = [float(r) for r in ranges]
        if dim == 2 and len(r_list) < 2:
            raise ValueError(
                f"2D anisotropy requires 2 ranges (major, minor), got {len(r_list)}"
            )
        if dim == 3 and len(r_list) < 3:
            raise ValueError(
                f"3D anisotropy requires 3 ranges (major, semi_major, minor), got {len(r_list)}"
            )
        r_major = r_list[0]

    # Check if isotropic
    is_iso = all(abs(r - r_major) < 1e-9 for r in r_list)
    if is_iso:
        return coords_arr, r_major

    # 2. Get rotation matrix R
    R = compute_anisotropy_rotation_matrix(dim, angles=angles)

    # 3. Scaling factors relative to r_major
    if dim == 2:
        r_minor = max(r_list[1], 1e-6)
        scale = np.array([1.0, r_major / r_minor], dtype=float)
    else:
        r_semi = max(r_list[1], 1e-6)
        r_minor = max(r_list[2], 1e-6)
        scale = np.array([1.0, r_major / r_semi, r_major / r_minor], dtype=float)

    # 4. Transform: (coords @ R.T) * scale
    transformed = (coords_arr @ R.T) * scale
    return transformed, r_major


def compute_anisotropic_distance(
    coords1: np.ndarray,
    coords2: Optional[np.ndarray] = None,
    ranges: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
    angles: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
) -> np.ndarray:
    """Computes anisotropic distance between points or coordinate lag vectors.

    Parameters
    ----------
    coords1 : np.ndarray
        Either coordinate differences of shape (..., dim) or point coordinates of shape (N, dim).
    coords2 : np.ndarray, optional
        Target coordinates of shape (M, dim) or (dim,). If None, coords1 is treated as
        coordinate difference vectors or self-distance coordinates.
    ranges : float, Sequence[float], or Mapping[str, float], optional
        Directional ranges (major, semi-major, minor) or (major, minor).
    angles : float, Sequence[float], or Mapping[str, float], optional
        Rotation angles (azimuth, dip, plunge) or azimuth.

    Returns
    -------
    np.ndarray
        Array of anisotropic distances.
    """
    c1 = np.asarray(coords1, dtype=float)
    if coords2 is not None:
        c2 = np.asarray(coords2, dtype=float)
        c1_t, _ = transform_anisotropic_coordinates(c1, ranges=ranges, angles=angles)
        c2_t, _ = transform_anisotropic_coordinates(c2, ranges=ranges, angles=angles)
        if c1_t.ndim == 2 and c2_t.ndim == 2:
            diffs = c1_t[:, None, :] - c2_t[None, :, :]
            return np.linalg.norm(diffs, axis=-1)
        elif c1_t.ndim == 2 and c2_t.ndim == 1:
            diffs = c1_t - c2_t[None, :]
            return np.linalg.norm(diffs, axis=-1)
        else:
            diffs = c1_t - c2_t
            return np.linalg.norm(diffs, axis=-1)
    else:
        # coords1 is already difference vectors
        c1_t, _ = transform_anisotropic_coordinates(c1, ranges=ranges, angles=angles)
        return np.linalg.norm(c1_t, axis=-1)


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: Point-support interpolation (V -> 0) on coordinate arrays.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Explicit Extrapolation Control: strictly enforce `mask_extrapolation=True` and `max_radius`.
# - Domain Boundaries: must respect domain boundaries (`sample_domains`, `grid_domains`).
# - Block Support: add 3D block model estimation with block dimensions (dx, dy, dz) and internal
#   point discretization (Nx x Ny x Nz) for volume-averaged block grades (SME Handbook).
# - Anisotropic search: 3D search ellipsoids with dip/azimuth/plunge without rigid config objects.
# - Validation support: block outputs must feed directly into `plot_grade_tonnage_curve` for
#   smoothing audits against NN and Kriging without indirection layers.
def inverse_distance_weighting(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    power: float = 2.0,
    k_neighbors: int = 8,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
    sample_domains: Optional[Sequence[Any]] = None,
    grid_domains: Optional[Sequence[Any]] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse Distance Weighting (IDW) interpolation using a k-d tree.

    Supports geological domain boundaries and 2D/3D spatial anisotropy (CIM MRMR §6.8/§6.9) (TODO: Manually Verify):
    when `sample_domains` and `grid_domains` are provided, estimation is strictly segregated
    so that grid nodes are only informed by samples sharing the same geological domain.
    When `anisotropy_ranges` and/or `anisotropy_angles` are supplied, search neighborhoods
    and distance weights are computed using an anisotropic search ellipsoid.

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
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples of shape (N,).
    grid_domains : Sequence[Any], optional
        Geological domain identifiers for target grid points of shape (M,).
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional search ranges (major, minor) in 2D or (major, semi-major, minor) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles: azimuth in 2D or (azimuth, dip, plunge) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to angles.

    Returns
    -------
    estimated_grades : np.ndarray
        Interpolated grades of shape (M,).
    distances : np.ndarray
        Distances to informing samples. Shape (M,) if k=1, else (M, k).
    """
    if (sample_domains is None) != (grid_domains is None):
        raise ValueError(
            "Both sample_domains and grid_domains must be provided together."
        )

    # Handle domain-segregated estimation
    if sample_domains is not None and grid_domains is not None:
        s_dom = np.asarray(sample_domains)
        g_dom = np.asarray(grid_domains)
        if len(s_dom) != len(samples_xy):
            raise ValueError(
                f"sample_domains length ({len(s_dom)}) must match samples_xy ({len(samples_xy)})."
            )
        if len(g_dom) != len(grid_points):
            raise ValueError(
                f"grid_domains length ({len(g_dom)}) must match grid_points ({len(grid_points)})."
            )

        estimated_grades = np.full(len(grid_points), np.nan)
        dist_shape = (
            (len(grid_points),) if k_neighbors == 1 else (len(grid_points), k_neighbors)
        )
        distances = np.full(dist_shape, np.nan)

        for dom in np.unique(g_dom):
            g_mask = g_dom == dom
            s_mask = s_dom == dom
            if not np.any(s_mask):
                continue
            dom_aniso_ranges = (
                anisotropy_ranges[dom]
                if isinstance(anisotropy_ranges, Mapping) and dom in anisotropy_ranges
                else anisotropy_ranges
            )
            dom_aniso_angles = (
                anisotropy_angles[dom]
                if isinstance(anisotropy_angles, Mapping) and dom in anisotropy_angles
                else anisotropy_angles
            )
            est_dom, dist_dom = inverse_distance_weighting(
                samples_xy[s_mask],
                sample_grades[s_mask],
                grid_points[g_mask],
                power=power,
                k_neighbors=k_neighbors,
                max_radius=max_radius,
                mask_extrapolation=mask_extrapolation,
                sample_domains=None,
                grid_domains=None,
                anisotropy_ranges=dom_aniso_ranges,
                anisotropy_angles=dom_aniso_angles,
            )
            estimated_grades[g_mask] = est_dom
            distances[g_mask] = dist_dom

        return estimated_grades, distances

    # Step 1: Transform coordinates to anisotropic equivalent space if specified
    samples_t, _ = transform_anisotropic_coordinates(
        samples_xy, ranges=anisotropy_ranges, angles=anisotropy_angles
    )
    grid_t, _ = transform_anisotropic_coordinates(
        grid_points, ranges=anisotropy_ranges, angles=anisotropy_angles
    )

    # Step 2: Build spatial index and query k-nearest neighbors
    tree = KDTree(samples_t)
    upper_bound = max_radius if max_radius is not None else float("inf")
    distances, indices = tree.query(
        grid_t, k=k_neighbors, distance_upper_bound=upper_bound
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


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: 2D/3D point nearest neighbor.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Core standard: 3D Nearest Neighbor on the block model is the mandatory declustered benchmark
#   proxy for swath plot local bias audits (`plot_swath_analysis`) and comparative grade-tonnage
#   audits (`plot_grade_tonnage_curve`) per JORC Table 1 and NI 43-101 (TODO: Manually Verify).
# - Domain segregation: respect domain boundaries (`sample_domains`, `grid_domains`).
# - Extrapolation control: enforce `mask_extrapolation` and `max_radius`.
# - No fallback layers: direct functional block model routine accepting DataFrame and returning DataFrame.
def nearest_neighbor_grid_estimation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
    sample_domains: Optional[Sequence[Any]] = None,
    grid_domains: Optional[Sequence[Any]] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest neighbor estimation (IDW with k=1) with optional domain segregation and anisotropy."""
    return inverse_distance_weighting(
        samples_xy,
        sample_grades,
        grid_points,
        power=1.0,
        k_neighbors=1,
        max_radius=max_radius,
        mask_extrapolation=mask_extrapolation,
        sample_domains=sample_domains,
        grid_domains=grid_domains,
        anisotropy_ranges=anisotropy_ranges,
        anisotropy_angles=anisotropy_angles,
    )


def _theoretical_covariance(
    h: np.ndarray,
    model: str = "spherical",
    nugget: float = 0.0,
    sill: float = 1.0,
    range_param: Union[float, Sequence[float], Mapping[str, float]] = 100.0,
    anisotropy_ranges: Optional[
        Union[float, Sequence[float], Mapping[str, float]]
    ] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[str, float]]
    ] = None,
) -> np.ndarray:
    """Evaluates theoretical spatial covariance C(h) = (c0 + c) - gamma(h).

    Supports 2D and 3D geometric anisotropy (CIM MRMR §6.8/§6.9) (TODO: Manually Verify): if separation vectors
    or anisotropic ranges are provided, lag distances are computed using the anisotropic metric.

    Parameters
    ----------
    h : np.ndarray
        Separation lag distance array, or separation coordinate difference array of shape (..., dim).
    model : str, default "spherical"
        Variogram model ("spherical", "exponential", "gaussian").
    nugget : float, default 0.0
        Nugget variance c0 (micro-scale variance / noise at h=0).
    sill : float, default 1.0
        Partial sill variance c (total sill is c0 + c).
    range_param : float, Sequence[float], or Mapping, default 100.0
        Practical correlation range a (or directional ranges).
    anisotropy_ranges : float, Sequence[float], or Mapping, optional
        Directional ranges (major, semi-major, minor) or (major, minor).
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth, dip, plunge) or azimuth.

    Returns
    -------
    np.ndarray
        Covariance values C(h) of identical shape to h (or leading shape if h was coordinates).
    """
    c0 = float(nugget)
    c = float(sill)
    total_sill = c0 + c

    h_arr = np.asarray(h, dtype=float)

    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (range_param if isinstance(range_param, (Sequence, Mapping)) else None)
    )
    if eff_ranges is not None and h_arr.ndim >= 1 and h_arr.shape[-1] in (2, 3):
        h_arr = compute_anisotropic_distance(
            h_arr, ranges=eff_ranges, angles=anisotropy_angles
        )
        a = (
            float(eff_ranges["major"])
            if isinstance(eff_ranges, Mapping) and "major" in eff_ranges
            else (
                float(eff_ranges[0])
                if isinstance(eff_ranges, Sequence)
                else float(eff_ranges)
            )
        )
    elif isinstance(range_param, Mapping):
        a = float(range_param.get("major", list(range_param.values())[0]))
    elif isinstance(range_param, Sequence):
        a = float(range_param[0])
    else:
        a = max(float(range_param), 1e-6)

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
    mean: Union[float, Mapping[Any, float]],
    sill: Union[float, Mapping[Any, float]],
    range_param: Union[float, Mapping[Any, float]] = 100.0,
    variogram_model: str = "spherical",
    nugget: Union[float, Mapping[Any, float]] = 0.0,
    k_neighbors: int = 16,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
    sample_domains: Optional[Sequence[Any]] = None,
    grid_domains: Optional[Sequence[Any]] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple Kriging (SK) grid interpolation with known stationary mean.

    Supports geological domain boundaries and 2D/3D spatial anisotropy (CIM MRMR §6.8/§6.9) (TODO: Manually Verify):
    when `sample_domains` and `grid_domains` are provided, estimation is strictly segregated
    so that grid nodes are only informed by samples sharing the same geological domain.
    Parameter mappings (mean, sill, range, nugget, anisotropy) can be supplied per-domain.

    Parameters
    ----------
    samples_xy : np.ndarray
        Sample coordinates of shape (N, 2) or (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    grid_points : np.ndarray
        Target estimation coordinates of shape (M, 2) or (M, 3).
    mean : float or Mapping
        Known stationary global mean of the domain (or dict mapping domain to mean).
    sill : float or Mapping, default 1.0
        Partial sill variance c (or dict mapping domain to sill).
    range_param : float or Mapping, default 100.0
        Spatial correlation range a (or dict mapping domain to range).
    variogram_model : str, default "spherical"
        Theoretical variogram model ("spherical", "exponential", "gaussian").
    nugget : float or Mapping, default 0.0
        Nugget variance c0 (or dict mapping domain to nugget).
    k_neighbors : int, default 16
        Maximum conditioning samples to query per target grid node.
    max_radius : float, optional
        Maximum search neighborhood radius.
    mask_extrapolation : bool, default False
        If True, masks blocks outside the drillhole convex hull to NaN.
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples of shape (N,).
    grid_domains : Sequence[Any], optional
        Geological domain identifiers for target grid points of shape (M,).
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional ranges (major, minor) in 2D or (major, semi-major, minor) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles: azimuth in 2D or (azimuth, dip, plunge) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to angles.

    Returns
    -------
    estimated_grades : np.ndarray
        Simple Kriging grade estimates of shape (M,).
    kriging_variance : np.ndarray
        Estimation variance sigma_SK^2 of shape (M,).
    """
    if (sample_domains is None) != (grid_domains is None):
        raise ValueError(
            "Both sample_domains and grid_domains must be provided together."
        )

    # Handle domain-segregated estimation
    if sample_domains is not None and grid_domains is not None:
        s_dom = np.asarray(sample_domains)
        g_dom = np.asarray(grid_domains)
        if len(s_dom) != len(samples_xy):
            raise ValueError(
                f"sample_domains length ({len(s_dom)}) must match samples_xy ({len(samples_xy)})."
            )
        if len(g_dom) != len(grid_points):
            raise ValueError(
                f"grid_domains length ({len(g_dom)}) must match grid_points ({len(grid_points)})."
            )

        estimates = np.full(len(grid_points), np.nan)
        variances = np.full(len(grid_points), np.nan)

        for dom in np.unique(g_dom):
            g_mask = g_dom == dom
            s_mask = s_dom == dom
            dom_mean = mean[dom] if isinstance(mean, Mapping) else mean
            dom_sill = sill[dom] if isinstance(sill, Mapping) else sill
            dom_range = (
                range_param[dom] if isinstance(range_param, Mapping) else range_param
            )
            dom_nugget = nugget[dom] if isinstance(nugget, Mapping) else nugget
            dom_aniso_ranges = (
                anisotropy_ranges[dom]
                if isinstance(anisotropy_ranges, Mapping) and dom in anisotropy_ranges
                else anisotropy_ranges
            )
            dom_aniso_angles = (
                anisotropy_angles[dom]
                if isinstance(anisotropy_angles, Mapping) and dom in anisotropy_angles
                else anisotropy_angles
            )

            if not np.any(s_mask):
                estimates[g_mask] = dom_mean
                variances[g_mask] = dom_sill + dom_nugget
                continue

            est_dom, var_dom = simple_kriging_grid_estimation(
                samples_xy=samples_xy[s_mask],
                sample_grades=sample_grades[s_mask],
                grid_points=grid_points[g_mask],
                mean=dom_mean,
                sill=dom_sill,
                range_param=dom_range,
                variogram_model=variogram_model,
                nugget=dom_nugget,
                k_neighbors=k_neighbors,
                max_radius=max_radius,
                mask_extrapolation=mask_extrapolation,
                sample_domains=None,
                grid_domains=None,
                anisotropy_ranges=dom_aniso_ranges,
                anisotropy_angles=dom_aniso_angles,
            )
            estimates[g_mask] = est_dom
            variances[g_mask] = var_dom

        return estimates, variances

    # -------------------------------------------------------------------------
    n_targets = len(grid_points)
    base_mean = mean if not isinstance(mean, Mapping) else list(mean.values())[0]
    base_sill = sill if not isinstance(sill, Mapping) else list(sill.values())[0]
    base_nugget = (
        nugget if not isinstance(nugget, Mapping) else list(nugget.values())[0]
    )
    base_range = (
        range_param
        if not isinstance(range_param, Mapping)
        else list(range_param.values())[0]
    )
    total_sill = base_nugget + base_sill

    # Initialize with the prior mean and maximum uncertainty (total sill)
    estimates = np.full(n_targets, base_mean, dtype=float)
    variances = np.full(n_targets, total_sill, dtype=float)

    if len(samples_xy) == 0 or n_targets == 0:
        return estimates, variances

    # Transform coordinates to anisotropic equivalent space if specified
    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (
            range_param if isinstance(range_param, (Sequence, Mapping)) else base_range
        )
    )
    samples_t, eff_range = transform_anisotropic_coordinates(
        samples_xy, ranges=eff_ranges, angles=anisotropy_angles
    )
    grid_t, _ = transform_anisotropic_coordinates(
        grid_points, ranges=eff_ranges, angles=anisotropy_angles
    )

    # 1. Query k nearest neighbors using KDTree in transformed space
    k_query = min(k_neighbors, len(samples_xy))
    tree = KDTree(samples_t)
    upper_bound = max_radius if max_radius is not None else float("inf")
    distances, indices = tree.query(grid_t, k=k_query, distance_upper_bound=upper_bound)

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
        coords_m = samples_t[idx_m]
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
                    d_m = np.linalg.norm(coords_m - grid_t[m], axis=1)
                    k_m = len(coords_m)
                    diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
                    h_matrix = np.linalg.norm(diff_matrix, axis=2)
            np.fill_diagonal(h_matrix, 0.0)
        else:
            diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
            h_matrix = np.linalg.norm(diff_matrix, axis=2)

        K_m = _theoretical_covariance(
            h_matrix, variogram_model, base_nugget, base_sill, eff_range
        )
        # Regularize diagonal to prevent singular matrices from collinear samples
        K_m[np.diag_indices(k_m)] += 1e-9

        # Build sample-to-target covariance vector k_0_m of shape (k_m,)
        k0_m = _theoretical_covariance(
            d_m, variogram_model, base_nugget, base_sill, eff_range
        )

        # Solve linear system K_m * lambda_m = k0_m
        weights_m = np.linalg.solve(K_m, k0_m)

        # Simple Kriging estimate: Z*_SK = mean + sum_i lambda_i * (Z_i - mean)
        estimates[m] = base_mean + np.sum(weights_m * (grades_m - base_mean))

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
    sill: Union[float, Mapping[Any, float]],
    range_param: Union[float, Mapping[Any, float]] = 100.0,
    variogram_model: str = "spherical",
    nugget: Union[float, Mapping[Any, float]] = 0.0,
    k_neighbors: int = 16,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
    sample_domains: Optional[Sequence[Any]] = None,
    grid_domains: Optional[Sequence[Any]] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary Kriging (OK) grid interpolation with unknown local mean.

    Supports geological domain boundaries and 2D/3D spatial anisotropy (CIM MRMR §6.8/§6.9) (TODO: Manually Verify):
    when `sample_domains` and `grid_domains` are provided, estimation is strictly segregated
    so that grid nodes are only informed by samples sharing the same geological domain.
    Parameter mappings (sill, range, nugget, anisotropy) can be supplied per-domain.

    Parameters
    ----------
    samples_xy : np.ndarray
        Sample coordinates of shape (N, 2) or (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    grid_points : np.ndarray
        Target estimation coordinates of shape (M, 2) or (M, 3).
    sill : float or Mapping, default 1.0
        Partial sill variance c (or dict mapping domain to sill).
    range_param : float or Mapping, default 100.0
        Spatial correlation range a (or dict mapping domain to range).
    variogram_model : str, default "spherical"
        Theoretical variogram model ("spherical", "exponential", "gaussian").
    nugget : float or Mapping, default 0.0
        Nugget variance c0 (or dict mapping domain to nugget).
    k_neighbors : int, default 16
        Maximum conditioning samples to query per target point.
    max_radius : float, optional
        Maximum search neighborhood radius.
    mask_extrapolation : bool, default False
        If True, masks blocks outside the drillhole convex hull to NaN.
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples of shape (N,).
    grid_domains : Sequence[Any], optional
        Geological domain identifiers for target grid points of shape (M,).
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional ranges (major, minor) in 2D or (major, semi-major, minor) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles: azimuth in 2D or (azimuth, dip, plunge) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to angles.

    Returns
    -------
    estimated_grades : np.ndarray
        Ordinary Kriging grade estimates of shape (M,).
    kriging_variance : np.ndarray
        Estimation variance sigma_OK^2 of shape (M,).
    """
    if (sample_domains is None) != (grid_domains is None):
        raise ValueError(
            "Both sample_domains and grid_domains must be provided together."
        )

    # Handle domain-segregated estimation
    if sample_domains is not None and grid_domains is not None:
        s_dom = np.asarray(sample_domains)
        g_dom = np.asarray(grid_domains)
        if len(s_dom) != len(samples_xy):
            raise ValueError(
                f"sample_domains length ({len(s_dom)}) must match samples_xy ({len(samples_xy)})."
            )
        if len(g_dom) != len(grid_points):
            raise ValueError(
                f"grid_domains length ({len(g_dom)}) must match grid_points ({len(grid_points)})."
            )

        estimates = np.full(len(grid_points), np.nan)
        variances = np.full(len(grid_points), np.nan)

        for dom in np.unique(g_dom):
            g_mask = g_dom == dom
            s_mask = s_dom == dom
            dom_sill = sill[dom] if isinstance(sill, Mapping) else sill
            dom_range = (
                range_param[dom] if isinstance(range_param, Mapping) else range_param
            )
            dom_nugget = nugget[dom] if isinstance(nugget, Mapping) else nugget
            dom_aniso_ranges = (
                anisotropy_ranges[dom]
                if isinstance(anisotropy_ranges, Mapping) and dom in anisotropy_ranges
                else anisotropy_ranges
            )
            dom_aniso_angles = (
                anisotropy_angles[dom]
                if isinstance(anisotropy_angles, Mapping) and dom in anisotropy_angles
                else anisotropy_angles
            )

            if not np.any(s_mask):
                continue

            est_dom, var_dom = ordinary_kriging_grid_estimation(
                samples_xy=samples_xy[s_mask],
                sample_grades=sample_grades[s_mask],
                grid_points=grid_points[g_mask],
                sill=dom_sill,
                range_param=dom_range,
                variogram_model=variogram_model,
                nugget=dom_nugget,
                k_neighbors=k_neighbors,
                max_radius=max_radius,
                mask_extrapolation=mask_extrapolation,
                sample_domains=None,
                grid_domains=None,
                anisotropy_ranges=dom_aniso_ranges,
                anisotropy_angles=dom_aniso_angles,
            )
            estimates[g_mask] = est_dom
            variances[g_mask] = var_dom

        return estimates, variances

    # -------------------------------------------------------------------------
    n_targets = len(grid_points)
    base_sill = sill if not isinstance(sill, Mapping) else list(sill.values())[0]
    base_nugget = (
        nugget if not isinstance(nugget, Mapping) else list(nugget.values())[0]
    )
    base_range = (
        range_param
        if not isinstance(range_param, Mapping)
        else list(range_param.values())[0]
    )
    total_sill = base_nugget + base_sill

    # Initialize with NaN and maximum uncertainty (total sill)
    estimates = np.full(n_targets, np.nan, dtype=float)
    variances = np.full(n_targets, total_sill, dtype=float)

    if len(samples_xy) == 0 or n_targets == 0:
        return estimates, variances

    # Transform coordinates to anisotropic equivalent space if specified
    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (
            range_param if isinstance(range_param, (Sequence, Mapping)) else base_range
        )
    )
    samples_t, eff_range = transform_anisotropic_coordinates(
        samples_xy, ranges=eff_ranges, angles=anisotropy_angles
    )
    grid_t, _ = transform_anisotropic_coordinates(
        grid_points, ranges=eff_ranges, angles=anisotropy_angles
    )

    # 1. Query k nearest neighbors using KDTree in transformed space
    k_query = min(k_neighbors, len(samples_xy))
    tree = KDTree(samples_t)
    upper_bound = max_radius if max_radius is not None else float("inf")
    distances, indices = tree.query(grid_t, k=k_query, distance_upper_bound=upper_bound)

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
        coords_m = samples_t[idx_m]
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
                    d_m = np.linalg.norm(coords_m - grid_t[m], axis=1)
                    k_m = len(coords_m)
                    diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
                    h_matrix = np.linalg.norm(diff_matrix, axis=2)
            np.fill_diagonal(h_matrix, 0.0)
        else:
            diff_matrix = coords_m[:, None, :] - coords_m[None, :, :]
            h_matrix = np.linalg.norm(diff_matrix, axis=2)

        K_m = _theoretical_covariance(
            h_matrix, variogram_model, base_nugget, base_sill, eff_range
        )
        K_m[np.diag_indices(k_m)] += 1e-9  # Regularizer to guarantee invertibility

        # Build sample-to-target covariance vector k0_m of shape (k_m,)
        k0_m = _theoretical_covariance(
            d_m, variogram_model, base_nugget, base_sill, eff_range
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


# =============================================================================
# 3D BLOCK MODELING & BLOCK KRIGING (SUPPORT EFFECT & DISCRETIZATION)
# =============================================================================


def create_block_model(
    origin: Tuple[float, float, float],
    block_size: Tuple[float, float, float],
    n_blocks: Tuple[int, int, int],
    default_density: float,
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
    default_density : float
        Bulk density / specific gravity (t/m^3).
        Site-specific parameter required without default.
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
    if default_density <= 0:
        raise ValueError("default_density must be positive.")

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


# TODO: potentially return internal_block_var
def ordinary_kriging_block_estimation(
    samples_xyz: np.ndarray,
    sample_grades: np.ndarray,
    block_model: pd.DataFrame,
    sill: Union[float, Mapping[Any, float]],
    range_param: Union[float, Mapping[Any, float]] = 100.0,
    discretization: Tuple[int, int, int] = (4, 4, 2),
    variogram_model: str = "spherical",
    nugget: Union[float, Mapping[Any, float]] = 0.0,
    k_neighbors: int = 16,
    max_radius: Optional[float] = None,
    min_samples: int = 1,
    domain_col: Optional[str] = None,
    sample_domains: Optional[Sequence[Any]] = None,
    sample_domain_col: Optional[Union[str, Sequence[Any]]] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Estimates block grades using 3D Ordinary Block Kriging with internal discretization.

    Supports geological domain boundaries and 3D spatial anisotropy (CIM MRMR §6.8/§6.9) (TODO: Manually Verify):
    when `anisotropy_ranges` and/or `anisotropy_angles` are supplied, block discretization
    points, sample points, and search ellipsoids are evaluated in the rotated anisotropic metric.

    Parameters
    ----------
    samples_xyz : np.ndarray
        Sample coordinates of shape (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and dimensions (dx, dy, dz).
    sill : float or Mapping
        Partial sill variance (or dict mapping domain to sill).
    range_param : float or Mapping, default 100.0
        Spatial correlation range (or dict mapping domain to range).
    discretization : tuple of (int, int, int), default (4, 4, 2)
        Number of internal discretization points (nx_disc, ny_disc, nz_disc) per block.
    variogram_model : str, default "spherical"
        Variogram model ("spherical", "exponential", "gaussian").
    nugget : float or Mapping, default 0.0
        Nugget variance (or dict mapping domain to nugget).
    k_neighbors : int, default 16
        Maximum number of informing samples queried per block.
    max_radius : float, optional
        Maximum search radius. Samples beyond this distance are excluded.
    min_samples : int, default 1
        Minimum number of samples required within search radius to estimate a block.
    domain_col : str, optional
        Geological domain column in block_model.
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples.
    sample_domain_col : str or Sequence[Any], optional
        Geological domain identifiers or column for conditioning samples.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional ranges (major, semi-major, minor) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth, dip, plunge) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to angles.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, np.ndarray]
        (block_estimates, block_variances, block_dispersion_variance, lagrange_multipliers).
    """
    n_blocks = len(block_model)
    if n_blocks == 0:
        return np.array([]), np.array([]), 0.0, np.array([])

    # Handle domain-segregated estimation
    s_dom = sample_domains if sample_domains is not None else sample_domain_col
    if (
        domain_col is not None
        and domain_col in block_model.columns
        and s_dom is not None
    ):
        s_dom_arr = np.asarray(s_dom)
        b_dom_arr = np.asarray(block_model[domain_col])
        if len(s_dom_arr) != len(samples_xyz):
            raise ValueError(
                f"sample_domains length ({len(s_dom_arr)}) must match samples_xyz ({len(samples_xyz)})."
            )

        block_estimates = np.full(n_blocks, np.nan, dtype=float)
        block_variances = np.full(n_blocks, np.nan, dtype=float)
        lagrange_multipliers = np.full(n_blocks, np.nan, dtype=float)
        overall_disp_var = 0.0

        for dom in np.unique(b_dom_arr):
            b_mask = b_dom_arr == dom
            s_mask = s_dom_arr == dom
            dom_sill = sill[dom] if isinstance(sill, Mapping) else sill
            dom_range = (
                range_param[dom] if isinstance(range_param, Mapping) else range_param
            )
            dom_nugget = nugget[dom] if isinstance(nugget, Mapping) else nugget
            dom_aniso_ranges = (
                anisotropy_ranges[dom]
                if isinstance(anisotropy_ranges, Mapping) and dom in anisotropy_ranges
                else anisotropy_ranges
            )
            dom_aniso_angles = (
                anisotropy_angles[dom]
                if isinstance(anisotropy_angles, Mapping) and dom in anisotropy_angles
                else anisotropy_angles
            )

            if not np.any(s_mask):
                continue

            sub_bm = block_model[b_mask].copy()
            est_dom, var_dom, disp_dom, lag_dom = ordinary_kriging_block_estimation(
                samples_xyz=samples_xyz[s_mask],
                sample_grades=sample_grades[s_mask],
                block_model=sub_bm,
                sill=dom_sill,
                range_param=dom_range,
                discretization=discretization,
                variogram_model=variogram_model,
                nugget=dom_nugget,
                k_neighbors=k_neighbors,
                max_radius=max_radius,
                min_samples=min_samples,
                domain_col=None,
                sample_domains=None,
                anisotropy_ranges=dom_aniso_ranges,
                anisotropy_angles=dom_aniso_angles,
            )
            block_estimates[b_mask] = est_dom
            block_variances[b_mask] = var_dom
            lagrange_multipliers[b_mask] = lag_dom
            overall_disp_var = disp_dom

        return block_estimates, block_variances, overall_disp_var, lagrange_multipliers

    base_sill = sill if not isinstance(sill, Mapping) else list(sill.values())[0]
    base_nugget = (
        nugget if not isinstance(nugget, Mapping) else list(nugget.values())[0]
    )
    base_range = (
        range_param
        if not isinstance(range_param, Mapping)
        else list(range_param.values())[0]
    )

    # Discretization resolution along each axis
    nx, ny, nz = discretization  # e.g., (4, 4, 2) -> 32 sub-points per block

    # Nominal block dimensions (can be read from block_model columns dx, dy, dz)
    dx = float(block_model["dx"].iloc[0])
    dy = float(block_model["dy"].iloc[0])
    dz = float(block_model["dz"].iloc[0])

    # 1. Compute 1D offset positions centered at 0 within [-dx/2, +dx/2]
    x_offsets = ((np.arange(nx) + 0.5) / nx - 0.5) * dx
    y_offsets = ((np.arange(ny) + 0.5) / ny - 0.5) * dy
    z_offsets = ((np.arange(nz) + 0.5) / nz - 0.5) * dz

    # 2. Meshgrid to create all 3D relative coordinate combinations
    xx, yy, zz = np.meshgrid(x_offsets, y_offsets, z_offsets, indexing="ij")
    disc_offsets = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # 3. Anisotropy Transformation
    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (
            range_param if isinstance(range_param, (Sequence, Mapping)) else base_range
        )
    )
    disc_offsets_t, eff_range = transform_anisotropic_coordinates(
        disc_offsets, ranges=eff_ranges, angles=anisotropy_angles
    )

    # 4. Compute Block Self-Covariance C_bar(V, V) in transformed space
    internal_diffs = disc_offsets_t[:, None, :] - disc_offsets_t[None, :, :]
    internal_dists = np.linalg.norm(internal_diffs, axis=2)
    internal_covs = _theoretical_covariance(
        internal_dists, variogram_model, base_nugget, base_sill, eff_range
    )
    c_vv = float(np.mean(internal_covs))
    block_dispersion_var = max(0.0, c_vv)

    # 5. Neighbor Search Setup & Spatial Query in transformed space
    upper_bound = max_radius if max_radius is not None else float("inf")
    block_coords = block_model[["x", "y", "z"]].to_numpy(dtype=float)

    # Initialize outputs: unestimated blocks remain NaN and receive maximum block uncertainty (c_vv)
    block_estimates = np.full(n_blocks, np.nan, dtype=float)
    block_variances = np.full(n_blocks, c_vv, dtype=float)
    lagrange_multipliers = np.full(n_blocks, np.nan, dtype=float)

    if len(samples_xyz) == 0:
        return (
            block_estimates,
            block_variances,
            block_dispersion_var,
            lagrange_multipliers,
        )

    samples_xyz_t, _ = transform_anisotropic_coordinates(
        samples_xyz, ranges=eff_ranges, angles=anisotropy_angles
    )
    block_coords_t, _ = transform_anisotropic_coordinates(
        block_coords, ranges=eff_ranges, angles=anisotropy_angles
    )

    tree = KDTree(samples_xyz_t)
    k_query = min(k_neighbors, len(samples_xyz))

    distances, indices = tree.query(
        block_coords_t,
        k=k_query,
        distance_upper_bound=upper_bound,
    )

    if k_query == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # 6. Point-to-Block Covariance & Ordinary Kriging Solution per Block
    for b in range(n_blocks):
        # a. Filter valid neighbors within upper_bound search radius
        valid_mask = np.isfinite(distances[b]) & (distances[b] <= upper_bound)
        if not np.any(valid_mask) or np.sum(valid_mask) < min_samples:
            continue  # Insufficient sample support: remains NaN and c_vv

        active_indices = indices[b][valid_mask]
        coords_active = samples_xyz_t[active_indices]  # shape (k, 3)
        grades_active = sample_grades[active_indices]  # shape (k,)
        k_active = len(active_indices)

        # b. Get absolute coordinates of the M discretization points for block b in transformed space
        block_points = block_coords_t[b] + disc_offsets_t  # shape (M, 3)

        # c. Compute Sample-to-Block covariance vector k_0 of shape (k,)
        sample_to_disc_diffs = (
            coords_active[:, None, :] - block_points[None, :, :]
        )  # (k, M, 3)
        sample_to_disc_dists = np.linalg.norm(sample_to_disc_diffs, axis=2)  # (k, M)
        sample_to_disc_covs = _theoretical_covariance(
            sample_to_disc_dists, variogram_model, base_nugget, base_sill, eff_range
        )
        # Average across the M sub-points: C_bar(x_i, V)
        k0_block = np.mean(sample_to_disc_covs, axis=1)  # shape (k,)

        # d. Build Sample-to-Sample covariance matrix K of shape (k, k)
        sample_diffs = coords_active[:, None, :] - coords_active[None, :, :]
        sample_dists = np.linalg.norm(sample_diffs, axis=2)
        K_mat = _theoretical_covariance(
            sample_dists, variogram_model, base_nugget, base_sill, eff_range
        )
        # Regularize diagonal to avoid singularity from collocated/close samples
        K_mat[np.diag_indices(k_active)] += 1e-9

        # e. Assemble (k+1) x (k+1) Ordinary Kriging system:
        A = np.ones((k_active + 1, k_active + 1), dtype=float)
        A[:k_active, :k_active] = K_mat
        A[k_active, k_active] = 0.0

        rhs = np.ones(k_active + 1, dtype=float)
        rhs[:k_active] = k0_block

        try:
            solution = np.linalg.solve(A, rhs)
            weights = solution[:k_active]
            mu = solution[k_active]
            # Block Grade Estimate: Z*(V) = sum(lambda_i * Z_i)
            block_estimates[b] = float(np.sum(weights * grades_active))
            # Block Kriging Variance: sigma_OK^2 = C_bar(V, V) - sum(lambda_i * C_bar(x_i, V)) - mu
            raw_variance = c_vv - np.sum(weights * k0_block) - mu
            block_variances[b] = max(0.0, float(raw_variance))
            lagrange_multipliers[b] = float(mu)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            continue

    return block_estimates, block_variances, block_dispersion_var, lagrange_multipliers


def simple_kriging_block_estimation(
    samples_xyz: np.ndarray,
    sample_grades: np.ndarray,
    block_model: pd.DataFrame,
    mean: Union[float, Mapping[Any, float]],
    sill: Union[float, Mapping[Any, float]],
    range_param: Union[float, Mapping[Any, float]] = 100.0,
    discretization: Tuple[int, int, int] = (4, 4, 2),
    variogram_model: str = "spherical",
    nugget: Union[float, Mapping[Any, float]] = 0.0,
    k_neighbors: int = 16,
    max_radius: Optional[float] = None,
    min_samples: int = 0,
    domain_col: Optional[str] = None,
    sample_domains: Optional[Sequence[Any]] = None,
    sample_domain_col: Optional[Union[str, Sequence[Any]]] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimates block grades using 3D Simple Block Kriging with a known stationary mean.

    Supports geological domain boundaries and 3D spatial anisotropy (CIM MRMR §6.8/§6.9) (TODO: Manually Verify):
    when `anisotropy_ranges` and/or `anisotropy_angles` are supplied, block discretization
    points, sample points, and search ellipsoids are evaluated in the rotated anisotropic metric.

    Parameters
    ----------
    samples_xyz : np.ndarray
        Sample coordinates of shape (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and dimensions (dx, dy, dz).
    mean : float or Mapping
        Known stationary mean grade m (or dict mapping domain to mean).
    sill : float or Mapping
        Partial sill variance (or dict mapping domain to sill).
    range_param : float or Mapping, default 100.0
        Spatial correlation range (or dict mapping domain to range).
    discretization : tuple of (int, int, int), default (4, 4, 2)
        Number of internal discretization points (nx_disc, ny_disc, nz_disc) per block.
    variogram_model : str, default "spherical"
        Variogram model ("spherical", "exponential", "gaussian").
    nugget : float or Mapping, default 0.0
        Nugget variance (or dict mapping domain to nugget).
    k_neighbors : int, default 16
        Maximum number of informing samples queried per block.
    max_radius : float, optional
        Maximum search radius. Samples beyond this distance are excluded.
    min_samples : int, default 0
        Minimum number of samples required. If 0 and no samples are found, reverts to mean.
    domain_col : str, optional
        Geological domain column in block_model.
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples.
    sample_domain_col : str or Sequence[Any], optional
        Geological domain identifiers or column for conditioning samples.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional ranges (major, semi-major, minor) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth, dip, plunge) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to angles.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (block_estimates, block_variances, block_dispersion_variance).
    """
    n_blocks = len(block_model)
    if n_blocks == 0:
        return np.array([]), np.array([]), 0.0

    # Handle domain-segregated estimation
    s_dom = sample_domains if sample_domains is not None else sample_domain_col
    if (
        domain_col is not None
        and domain_col in block_model.columns
        and s_dom is not None
    ):
        s_dom_arr = np.asarray(s_dom)
        b_dom_arr = np.asarray(block_model[domain_col])
        if len(s_dom_arr) != len(samples_xyz):
            raise ValueError(
                f"sample_domains length ({len(s_dom_arr)}) must match samples_xyz ({len(samples_xyz)})."
            )

        block_estimates = np.full(n_blocks, np.nan, dtype=float)
        block_variances = np.full(n_blocks, np.nan, dtype=float)
        overall_disp_var = 0.0

        for dom in np.unique(b_dom_arr):
            b_mask = b_dom_arr == dom
            s_mask = s_dom_arr == dom
            dom_mean = mean[dom] if isinstance(mean, Mapping) else mean
            dom_sill = sill[dom] if isinstance(sill, Mapping) else sill
            dom_range = (
                range_param[dom] if isinstance(range_param, Mapping) else range_param
            )
            dom_nugget = nugget[dom] if isinstance(nugget, Mapping) else nugget
            dom_aniso_ranges = (
                anisotropy_ranges[dom]
                if isinstance(anisotropy_ranges, Mapping) and dom in anisotropy_ranges
                else anisotropy_ranges
            )
            dom_aniso_angles = (
                anisotropy_angles[dom]
                if isinstance(anisotropy_angles, Mapping) and dom in anisotropy_angles
                else anisotropy_angles
            )

            if not np.any(s_mask):
                continue

            sub_bm = block_model[b_mask].copy()
            est_dom, var_dom, disp_dom = simple_kriging_block_estimation(
                samples_xyz=samples_xyz[s_mask],
                sample_grades=sample_grades[s_mask],
                block_model=sub_bm,
                mean=dom_mean,
                sill=dom_sill,
                range_param=dom_range,
                discretization=discretization,
                variogram_model=variogram_model,
                nugget=dom_nugget,
                k_neighbors=k_neighbors,
                max_radius=max_radius,
                min_samples=min_samples,
                domain_col=None,
                sample_domains=None,
                anisotropy_ranges=dom_aniso_ranges,
                anisotropy_angles=dom_aniso_angles,
            )
            block_estimates[b_mask] = est_dom
            block_variances[b_mask] = var_dom
            overall_disp_var = disp_dom

        return block_estimates, block_variances, overall_disp_var

    base_mean = mean if not isinstance(mean, Mapping) else list(mean.values())[0]
    base_sill = sill if not isinstance(sill, Mapping) else list(sill.values())[0]
    base_nugget = (
        nugget if not isinstance(nugget, Mapping) else list(nugget.values())[0]
    )
    base_range = (
        range_param
        if not isinstance(range_param, Mapping)
        else list(range_param.values())[0]
    )

    # Discretization resolution along each axis
    nx, ny, nz = discretization

    dx = float(block_model["dx"].iloc[0])
    dy = float(block_model["dy"].iloc[0])
    dz = float(block_model["dz"].iloc[0])

    x_offsets = ((np.arange(nx) + 0.5) / nx - 0.5) * dx
    y_offsets = ((np.arange(ny) + 0.5) / ny - 0.5) * dy
    z_offsets = ((np.arange(nz) + 0.5) / nz - 0.5) * dz

    xx, yy, zz = np.meshgrid(x_offsets, y_offsets, z_offsets, indexing="ij")
    disc_offsets = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (
            range_param if isinstance(range_param, (Sequence, Mapping)) else base_range
        )
    )
    disc_offsets_t, eff_range = transform_anisotropic_coordinates(
        disc_offsets, ranges=eff_ranges, angles=anisotropy_angles
    )

    internal_diffs = disc_offsets_t[:, None, :] - disc_offsets_t[None, :, :]
    internal_dists = np.linalg.norm(internal_diffs, axis=2)
    internal_covs = _theoretical_covariance(
        internal_dists, variogram_model, base_nugget, base_sill, eff_range
    )
    c_vv = float(np.mean(internal_covs))
    block_dispersion_var = max(0.0, c_vv)

    upper_bound = max_radius if max_radius is not None else float("inf")
    block_coords = block_model[["x", "y", "z"]].to_numpy(dtype=float)

    # In Simple Kriging, unestimated blocks smoothly revert to the stationary prior mean
    block_estimates = np.full(n_blocks, float(base_mean), dtype=float)
    block_variances = np.full(n_blocks, c_vv, dtype=float)

    if len(samples_xyz) == 0:
        return block_estimates, block_variances, block_dispersion_var

    samples_xyz_t, _ = transform_anisotropic_coordinates(
        samples_xyz, ranges=eff_ranges, angles=anisotropy_angles
    )
    block_coords_t, _ = transform_anisotropic_coordinates(
        block_coords, ranges=eff_ranges, angles=anisotropy_angles
    )

    tree = KDTree(samples_xyz_t)
    k_query = min(k_neighbors, len(samples_xyz))

    distances, indices = tree.query(
        block_coords_t,
        k=k_query,
        distance_upper_bound=upper_bound,
    )

    if k_query == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    for b in range(n_blocks):
        valid_mask = np.isfinite(distances[b]) & (distances[b] <= upper_bound)
        if not np.any(valid_mask) or np.sum(valid_mask) < min_samples:
            continue

        active_indices = indices[b][valid_mask]
        coords_active = samples_xyz_t[active_indices]
        grades_active = sample_grades[active_indices]
        k_active = len(active_indices)

        block_points = block_coords_t[b] + disc_offsets_t

        sample_to_disc_diffs = coords_active[:, None, :] - block_points[None, :, :]
        sample_to_disc_dists = np.linalg.norm(sample_to_disc_diffs, axis=2)
        sample_to_disc_covs = _theoretical_covariance(
            sample_to_disc_dists, variogram_model, base_nugget, base_sill, eff_range
        )
        k0_block = np.mean(sample_to_disc_covs, axis=1)

        sample_diffs = coords_active[:, None, :] - coords_active[None, :, :]
        sample_dists = np.linalg.norm(sample_diffs, axis=2)
        K_mat = _theoretical_covariance(
            sample_dists, variogram_model, base_nugget, base_sill, eff_range
        )
        K_mat[np.diag_indices(k_active)] += 1e-9

        try:
            weights = np.linalg.solve(K_mat, k0_block)
            # SK estimate: m + sum(lambda_i * (Z_i - m))
            residual = grades_active - base_mean
            block_estimates[b] = float(base_mean + np.sum(weights * residual))
            # SK block variance: C_bar(V, V) - sum(lambda_i * C_bar(x_i, V))
            raw_variance = c_vv - np.sum(weights * k0_block)
            block_variances[b] = max(0.0, float(raw_variance))
        except np.linalg.LinAlgError:
            continue

    return block_estimates, block_variances, block_dispersion_var


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

    Conforms to NI 43-101 (Item 14) and JORC Table 1 mineral reporting standards (TODO: Manually Verify):
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


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: 2D cell declustering in (X, Y) plane.
# Guidelines from CONTRIBUTING.md to adhere to:
# - SME Handbook & Geostatistical Standards: 3D exploration drillholes have dense downhole
#   sampling vs wide lateral spacing; requires 3D anisotropic cell sizing (Lx, Ly, Lz) with `z_col`.
# - Domain boundaries: declustering weights must be calculated per geological domain (`domain_col`).
# - Functional style & Simple over easy: transparent DataFrame inputs and array outputs; avoid
#   rigid config classes, use inline constants and simple function arguments.
# - No backwards-compatibility fallbacks: cleanly extend the signature rather than adding shim layers.
def cell_declustering(
    drillholes: pd.DataFrame,
    cell_sizes: Optional[Union[float, Sequence[float]]] = None,
    grade_col: str = "grade",
    x_col: str = "x",
    y_col: str = "y",
    n_offsets: int = 8,
    min_mean: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """Calculates spatial declustering weights and cell size sensitivity.

    In exploration drilling, geologists cluster drillholes preferentially in high-grade
    zones, causing naive sample averages to overestimate global deposit grade.
    Cell declustering (Journel 1983; Deutsch & Journel 1998; SME Handbook Section 4.3) (TODO: Manually Verify)
    superimposes a regular grid over the data and assigns weights inversely proportional
    to the number of samples sharing each cell.

    To eliminate cell boundary edge artifacts, the grid origin is shifted across
    `n_offsets` x `n_offsets` sub-grid increments and the resulting weights are averaged.

    Parameters
    ----------
    drillholes : pd.DataFrame
        Table of drillholes containing spatial coordinates and grades.
    cell_sizes : float or Sequence[float], optional
        Single cell dimension or sequence of cell dimensions (m) to test.
        If None, automatically determines a comprehensive range from below the minimum
        drillhole spacing up to 3.5x the deposit extent, guaranteeing a complete U-shaped curve.
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

    if cell_sizes is None:
        dx = float(x.max() - x.min())
        dy = float(y.max() - y.min())
        diag = float(np.hypot(dx, dy))
        if n_samples > 1:
            pts = np.column_stack([x, y])
            diffs = pts[:, None, :] - pts[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist = float(dists.min())
            min_cs = max(min_dist * 0.5, 1e-3)
        else:
            min_cs = 1.0
        max_cs = max(diag * 7.0, min_cs * 15.0)
        cell_sizes_list = list(np.linspace(min_cs, max_cs, 45))
    elif isinstance(cell_sizes, (int, float)):
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

    Standard NI 43-101 and JORC reporting chart (TODO: Manually Verify) justifying the selected
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


# TODO: is kriging_neighborhood_analysis a better name?
def kriging_quality_metrics(
    kriging_variances: np.ndarray,
    block_dispersion_variance: float,
    lagrange_multipliers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calculates Kriging Efficiency (KE) and Slope of Regression (SoR) for Block Kriging.

    Standard conditional bias diagnostics introduced by Danie Krige (1996) and
    Isobel Clark (1983) for Kriging Neighborhood Analysis (KNA), widely used under
    JORC and SAMREC (TODO: Manually Verify) to optimize search parameters and assess geological confidence:

    1. Kriging Efficiency (KE):
       KE = (BV - sigma_OK^2) / BV
       Measures the relative percentage error reduction of the block estimate compared
       to the block dispersion variance (BV = sigma^2(V|D)).

    2. Slope of Regression (SoR):
       SoR = Cov(Z_V, Z_V*) / Var(Z_V*) = (BV - sigma_OK^2 - mu) / (BV - sigma_OK^2 - 2 * mu)
       Estimates the linear regression slope of true block grades on kriged estimates.
       SoR >= 0.8 is often required for Measured Resources; SoR >= 0.5 for Indicated.
       Note: The minus signs on mu correspond to the augmented matrix system [K 1; 1^T 0]
       where K * lambda + mu * 1 = k0.

    Theoretical Support Constraint (Krige 1996; SME Handbook Section 4.5) (TODO: Manually Verify):
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
    vars_arr = np.asarray(kriging_variances, dtype=float)

    if not isinstance(
        block_dispersion_variance, (int, float, np.floating, np.integer)
    ) or np.isnan(block_dispersion_variance):
        raise ValueError("block_dispersion_variance must be a valid numeric scalar.")
    if block_dispersion_variance <= 0.0:
        raise ValueError(
            f"block_dispersion_variance must be strictly positive (> 0), got {block_dispersion_variance}."
        )

    if vars_arr.size == 0:
        empty_sor = (
            np.array([], dtype=float) if lagrange_multipliers is not None else None
        )
        return np.array([], dtype=float), empty_sor

    kriging_efficiency = (
        block_dispersion_variance - vars_arr
    ) / block_dispersion_variance

    if lagrange_multipliers is None:
        slope_of_regression = None
    else:
        mu_arr = np.asarray(lagrange_multipliers, dtype=float)
        if mu_arr.shape != vars_arr.shape:
            raise ValueError(
                f"Shape mismatch: lagrange_multipliers shape {mu_arr.shape} must match "
                f"kriging_variances shape {vars_arr.shape}."
            )
        # Cov(Z_V, Z_V*) = BV - sigma_OK^2 - mu
        # Var(Z_V*) = BV - sigma_OK^2 - 2*mu
        num = block_dispersion_variance - vars_arr - mu_arr
        den = block_dispersion_variance - vars_arr - 2.0 * mu_arr

        with np.errstate(divide="ignore", invalid="ignore"):
            slope_of_regression = np.where(np.abs(den) > 1e-12, num / den, np.nan)

    return kriging_efficiency, slope_of_regression


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: Euclidean k-d tree point distance queries.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Reporting Standards (CIM 2014/2019, JORC 2012, NI 43-101) (TODO: Manually Verify): multi-hole confidence criteria
#   for Measured/Indicated must demonstrate continuity between INDEPENDENT drillholes.
#   In 3D block models informed by downhole composites, multiple points along the same borehole
#   must not falsely satisfy multi-hole requirements. Accept `hole_id_col` or collar locations.
# - Domain boundaries: spacing classification must be segregated by geological domain.
# - Data Support Separation: separate legal concession `boundary` from drillhole support envelope.
# - Functional design: transparent DataFrame/array inputs and string category outputs.
def classify_resources_by_drill_spacing(
    grid_points: np.ndarray,
    samples_xy: np.ndarray,
    max_radius_measured: float,
    max_radius_indicated: float,
    min_holes_measured: int = 3,
    min_holes_indicated: int = 2,
    max_radius_inferred: Optional[float] = None,
    is_interpolated: Optional[np.ndarray] = None,
    anisotropy_ranges: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
    anisotropy_angles: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
) -> np.ndarray:
    """Classifies resource blocks into confidence categories based on drillhole spacing and hole counts.

    In accordance with CIM Definition Standards (2014) and JORC Code (2012) principles (TODO: Manually Verify):
    - Measured: Dense drillhole spacing where at least `min_holes_measured` informing
      drillholes lie within `max_radius_measured`, demonstrating high geological and grade continuity.
    - Indicated: Moderate drillhole spacing where at least `min_holes_indicated` informing
      drillholes lie within `max_radius_indicated`, supporting reasonable assumptions of continuity.
    - Inferred: Wider spacing or extrapolated blocks where geological continuity can be implied
      (within `max_radius_inferred` if specified).
    - Unclassified: Blocks beyond the maximum inferred search radius or lacking sufficient data.

    Supports 2D and 3D geometric anisotropy (CIM MRMR §6.8/§6.9) (TODO: Manually Verify): when `anisotropy_ranges`
    and/or `anisotropy_angles` are supplied, search distances are evaluated along the
    oriented search ellipsoid.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of block centers of shape (M, 2) or (M, 3).
    samples_xy : np.ndarray
        Array of informing drillhole coordinates of shape (N, 2) or (N, 3).
    max_radius_measured : float
        Maximum search radius to qualify for Measured classification.
        Site-specific parameter required without default.
    max_radius_indicated : float
        Maximum search radius to qualify for Indicated classification.
        Site-specific parameter required without default.
    min_holes_measured : int, default 3
        Minimum informing drillholes within `max_radius_measured` for Measured.
    min_holes_indicated : int, default 2
        Minimum informing drillholes within `max_radius_indicated` for Indicated.
    max_radius_inferred : float, optional
        Maximum search radius to qualify for Inferred classification. If None,
        all blocks not meeting Measured or Indicated default to Inferred.
    is_interpolated : np.ndarray of bool, optional
        Boolean mask of shape (M,) indicating whether each block lies within the
        convex hull / domain interpolation volume. Blocks where `is_interpolated`
        is False cannot be classified as Measured or Indicated (capped at Inferred).
    anisotropy_ranges : float, Sequence[float], or Mapping, optional
        Directional ranges (major, minor) in 2D or (major, semi-major, minor) in 3D.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth in 2D, or azimuth, dip, plunge in 3D).

    Returns
    -------
    np.ndarray
        Array of category strings ("Measured", "Indicated", "Inferred", or "Unclassified") of shape (M,).
    """
    pts = np.asarray(grid_points, dtype=float)
    if pts.size == 0:
        return np.empty(0, dtype=object)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(
            f"grid_points must have shape (M, 2) or (M, 3), got shape {pts.shape}."
        )

    samples = np.asarray(samples_xy, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != pts.shape[1]:
        raise ValueError(
            f"samples_xy must have shape (N, {pts.shape[1]}) matching grid_points, got shape {samples.shape}."
        )

    if max_radius_measured <= 0.0 or max_radius_indicated <= 0.0:
        raise ValueError("Search radii must be strictly positive.")
    if max_radius_measured > max_radius_indicated:
        raise ValueError(
            f"max_radius_measured ({max_radius_measured}) must be <= max_radius_indicated ({max_radius_indicated})."
        )
    if max_radius_inferred is not None and max_radius_inferred < max_radius_indicated:
        raise ValueError(
            f"max_radius_inferred ({max_radius_inferred}) must be >= max_radius_indicated ({max_radius_indicated})."
        )
    if min_holes_measured < 1 or min_holes_indicated < 1:
        raise ValueError("min_holes_measured and min_holes_indicated must be >= 1.")
    if min_holes_measured < min_holes_indicated:
        raise ValueError(
            f"min_holes_measured ({min_holes_measured}) must be >= min_holes_indicated ({min_holes_indicated})."
        )

    M = len(pts)
    N = len(samples)

    if is_interpolated is not None:
        interp_mask = np.asarray(is_interpolated, dtype=bool)
        if len(interp_mask) != M:
            raise ValueError(
                f"is_interpolated length ({len(interp_mask)}) must match grid_points length ({M})."
            )
    else:
        interp_mask = None

    if N == 0:
        return np.full(M, "Unclassified", dtype=object)

    samples_t, _ = transform_anisotropic_coordinates(
        samples, ranges=anisotropy_ranges, angles=anisotropy_angles
    )
    pts_t, _ = transform_anisotropic_coordinates(
        pts, ranges=anisotropy_ranges, angles=anisotropy_angles
    )

    tree = KDTree(samples_t)

    # Check Measured criteria
    if N >= min_holes_measured:
        d_meas, _ = tree.query(pts_t, k=min_holes_measured)
        dist_meas = d_meas if min_holes_measured == 1 else d_meas[:, -1]
        has_measured = dist_meas <= max_radius_measured
    else:
        has_measured = np.zeros(M, dtype=bool)

    # Check Indicated criteria
    if N >= min_holes_indicated:
        d_ind, _ = tree.query(pts_t, k=min_holes_indicated)
        dist_ind = d_ind if min_holes_indicated == 1 else d_ind[:, -1]
        has_indicated = dist_ind <= max_radius_indicated
    else:
        has_indicated = np.zeros(M, dtype=bool)

    # Check Inferred criteria
    if max_radius_inferred is not None:
        d_inf, _ = tree.query(pts_t, k=1)
        has_inferred = d_inf <= max_radius_inferred
    else:
        has_inferred = np.ones(M, dtype=bool)

    categories = np.full(M, "Unclassified", dtype=object)

    if interp_mask is not None:
        categories[interp_mask & has_measured] = "Measured"
        categories[interp_mask & (~has_measured) & has_indicated] = "Indicated"
        categories[interp_mask & (~has_measured) & (~has_indicated) & has_inferred] = (
            "Inferred"
        )
        categories[(~interp_mask) & has_inferred] = "Inferred"
    else:
        categories[has_measured] = "Measured"
        categories[(~has_measured) & has_indicated] = "Indicated"
        categories[(~has_measured) & (~has_indicated) & has_inferred] = "Inferred"

    return categories


# TODO: possibly remove and inline in classify_mineral_resources
def classify_resources_by_sor(
    slopes_of_regression: np.ndarray,
    threshold_measured: float,
    threshold_indicated: float,
    kriging_efficiencies: Optional[np.ndarray] = None,
    max_slope_measured: float = 1.05,
    min_kriging_efficiency: Optional[float] = 0.0,
) -> np.ndarray:
    """Classifies resource blocks based on Kriging Neighborhood Analysis (KNA) Slope of Regression (SoR).

    In geostatistical best practice (Vann et al., 2003; Armstrong, 1998) (TODO: Manually Verify):
    - Slope of Regression (SoR) gauges conditional unbiasedness of block estimates.
    - Measured: High conditional unbiasedness (typically SoR >= 0.80 and <= 1.05, with positive KE).
    - Indicated: Moderate conditional unbiasedness (typically 0.50 <= SoR < 0.80, with positive KE).
    - Inferred: Low conditional unbiasedness (SoR < 0.50) or negative efficiency.
    - Unclassified: Non-finite or unestimated blocks (NaN / Inf).

    Parameters
    ----------
    slopes_of_regression : np.ndarray
        Array of Slope of Regression values of shape (M,).
    threshold_measured : float
        Minimum Slope of Regression for Measured classification.
        Site-specific parameter required without default.
    threshold_indicated : float
        Minimum Slope of Regression for Indicated classification.
        Site-specific parameter required without default.
    kriging_efficiencies : np.ndarray, optional
        Array of Kriging Efficiency values of shape (M,). If provided, blocks
        with KE < min_kriging_efficiency are disqualified from Measured/Indicated.
    max_slope_measured : float, default 1.05
        Maximum Slope of Regression for Measured classification (slopes > 1.05
        frequently reflect numerical instability or severe conditional bias).
    min_kriging_efficiency : float, optional, default 0.0
        Minimum Kriging Efficiency required for Measured and Indicated.
        If None, Kriging Efficiency is not evaluated.

    Returns
    -------
    np.ndarray
        Array of category strings ("Measured", "Indicated", "Inferred", or "Unclassified") of shape (M,).
    """
    sor = np.asarray(slopes_of_regression, dtype=float)
    if sor.size == 0:
        return np.empty(0, dtype=object)

    if threshold_measured <= threshold_indicated:
        raise ValueError(
            f"threshold_measured ({threshold_measured}) must be > threshold_indicated ({threshold_indicated})."
        )
    if max_slope_measured < threshold_measured:
        raise ValueError(
            f"max_slope_measured ({max_slope_measured}) must be >= threshold_measured ({threshold_measured})."
        )

    if kriging_efficiencies is not None:
        ke = np.asarray(kriging_efficiencies, dtype=float)
        if ke.shape != sor.shape:
            raise ValueError(
                f"Shape mismatch: kriging_efficiencies shape {ke.shape} must match slopes_of_regression shape {sor.shape}."
            )
        if min_kriging_efficiency is not None:
            ke_ok = np.isfinite(ke) & (ke >= min_kriging_efficiency)
        else:
            ke_ok = np.isfinite(ke)
    else:
        ke_ok = np.ones(len(sor), dtype=bool)

    valid_sor = np.isfinite(sor)
    categories = np.full(len(sor), "Unclassified", dtype=object)

    is_meas = (
        valid_sor & (sor >= threshold_measured) & (sor <= max_slope_measured) & ke_ok
    )
    is_ind = (
        valid_sor & (sor >= threshold_indicated) & (sor < threshold_measured) & ke_ok
    )
    is_inf = valid_sor & (~is_meas) & (~is_ind)

    categories[is_meas] = "Measured"
    categories[is_ind] = "Indicated"
    categories[is_inf] = "Inferred"

    return categories


def smooth_resource_categories(
    grid_points: np.ndarray,
    categories: Sequence[str] | np.ndarray,
    smoothing_radius: Optional[float] = None,
    k_neighbors: Optional[int] = None,
    min_cluster_size: int = 1,
    downgrade_isolated: bool = True,
) -> np.ndarray:
    """Eliminates the 'spotted dog' artifact by spatially smoothing block classifications.

    Regulatory & Geostatistical Context (CIM MRMR Guidelines §6.11) (TODO: Manually Verify):
    ----------------------------------------------------------------
    Automated numeric classification criteria (such as Kriging estimation variance or
    drill spacing cut-offs) frequently produce isolated, disjointed blocks of high confidence
    surrounded by lower confidence (the 'spotted dog' effect, Stephenson et al., 2006).
    Under CIM MRMR Best Practice Guidelines (§6.11, pp. 25–26) (TODO: Manually Verify), numeric cut-offs must only be
    an initial guide; computer-based 'categorization smoothers' or wireframe envelopes must be
    applied to ensure resource categories form coherent, operationally mineable zones.

    This function applies:
    1. Spatial Majority / Mode Filtering:
       For each block, queries neighbors within `smoothing_radius` (or `k_neighbors`).
       Replaces isolated classifications with the majority class of its neighborhood.
       Under the Conservative Downgrade Principle (`downgrade_isolated=True`), isolated
       high-confidence blocks ('Measured') surrounded by lower confidence are downgraded
       to 'Indicated', but low-confidence blocks are not artificially upgraded into Measured
       without supporting drill data.
    2. Minimum Contiguous Volume / Cluster Filtering:
       If `min_cluster_size > 1`, connected components of blocks belonging to 'Measured'
       or 'Indicated' with fewer than `min_cluster_size` contiguous blocks are downgraded
       to the next lower category, ensuring that reported resources meet minimum selective
       mining unit (SMU) or stope panel volume requirements.

    Parameters
    ----------
    grid_points : np.ndarray
        Coordinates of block centers of shape (M, 2) or (M, 3).
    categories : Sequence[str] or np.ndarray
        Array of category strings ("Measured", "Indicated", "Inferred", or "Unclassified") of shape (M,).
    smoothing_radius : float, optional
        Spatial search radius for neighborhood majority voting. If None and `k_neighbors` is None,
        only `min_cluster_size` filtering is performed.
    k_neighbors : int, optional
        Number of nearest neighbors to evaluate if `smoothing_radius` is not specified.
    min_cluster_size : int, default 1
        Minimum number of contiguous blocks required to sustain a confidence category.
        Clusters smaller than this threshold are downgraded.
    downgrade_isolated : bool, default True
        If True (conservative principle), isolated high-confidence blocks are downgraded,
        preventing un-drilled areas from being upgraded. If False, standard majority vote is applied.

    Returns
    -------
    np.ndarray
        Smoothed category array of shape (M,).
    """
    pts = np.asarray(grid_points, dtype=float)
    cats = np.asarray(categories, dtype=object).copy()
    M = len(pts)
    if M == 0:
        return np.empty(0, dtype=object)
    if len(cats) != M:
        raise ValueError(
            f"Shape mismatch: grid_points length ({M}) does not match categories length ({len(cats)})."
        )

    if smoothing_radius is not None and smoothing_radius <= 0.0:
        raise ValueError(
            f"smoothing_radius must be strictly positive, got {smoothing_radius}"
        )
    if k_neighbors is not None and k_neighbors <= 0:
        raise ValueError(f"k_neighbors must be strictly positive, got {k_neighbors}")
    if min_cluster_size < 1:
        raise ValueError(f"min_cluster_size must be >= 1, got {min_cluster_size}")

    if smoothing_radius is None and k_neighbors is None and min_cluster_size <= 1:
        return cats

    inv_ranks = {0: "Unclassified", 1: "Inferred", 2: "Indicated", 3: "Measured"}

    numeric_ranks = np.zeros(M, dtype=int)
    for i, c in enumerate(cats):
        norm_c = str(c).strip().capitalize()
        if norm_c.startswith("Meas"):
            numeric_ranks[i] = 3
        elif norm_c.startswith("Ind"):
            numeric_ranks[i] = 2
        elif norm_c.startswith("Infer"):
            numeric_ranks[i] = 1
        else:
            numeric_ranks[i] = 0

    tree = cKDTree(pts)

    # 1. Spatial majority filter
    if smoothing_radius is not None or k_neighbors is not None:
        smoothed_ranks = numeric_ranks.copy()
        for i in range(M):
            if smoothing_radius is not None:
                nbr_indices = tree.query_ball_point(pts[i], r=smoothing_radius)
            else:
                _, nbr_indices = tree.query(pts[i], k=min(M, k_neighbors or 5))
                if isinstance(nbr_indices, (int, np.integer)):
                    nbr_indices = [int(nbr_indices)]
                else:
                    nbr_indices = list(nbr_indices)

            if len(nbr_indices) == 0:
                continue

            nbr_ranks = numeric_ranks[nbr_indices]
            counts = np.bincount(nbr_ranks, minlength=4)
            max_c = np.max(counts)
            candidates = np.where(counts == max_c)[0]
            # Conservative tie-breaking: choose lower rank when downgrade_isolated=True
            mode_rank = (
                int(np.min(candidates))
                if downgrade_isolated
                else int(np.max(candidates))
            )

            if downgrade_isolated:
                smoothed_ranks[i] = min(numeric_ranks[i], mode_rank)
            else:
                smoothed_ranks[i] = mode_rank

        numeric_ranks = smoothed_ranks

    # 2. Minimum cluster size filter (connected components)
    if min_cluster_size > 1:
        if smoothing_radius is not None:
            r_conn = smoothing_radius
        else:
            dists, _ = tree.query(pts, k=min(2, M))
            if dists.ndim == 2 and dists.shape[1] > 1:
                r_conn = 1.5 * float(np.median(dists[:, 1]))
            else:
                r_conn = 1.0

        for cat_rank, downgrade_to in [(3, 2), (2, 1)]:
            mask = numeric_ranks == cat_rank
            indices = np.where(mask)[0]
            if len(indices) > 0:
                if len(indices) < min_cluster_size:
                    numeric_ranks[indices] = downgrade_to
                else:
                    sub_pts = pts[indices]
                    sub_tree = cKDTree(sub_pts)
                    pairs = sub_tree.query_pairs(r=r_conn)
                    adj = lil_matrix((len(indices), len(indices)), dtype=bool)
                    for u, v in pairs:
                        adj[u, v] = True
                        adj[v, u] = True
                    n_comp, labels = connected_components(adj, directed=False)
                    comp_sizes = np.bincount(labels)
                    small_comp_labels = np.where(comp_sizes < min_cluster_size)[0]
                    to_downgrade = indices[np.isin(labels, small_comp_labels)]
                    numeric_ranks[to_downgrade] = downgrade_to

    return np.array(
        [inv_ranks.get(r, "Unclassified") for r in numeric_ranks], dtype=object
    )


def classify_resources_by_kriging_variance(
    kriging_variances: np.ndarray,
    variance_threshold_measured: float,
    variance_threshold_indicated: float,
    variance_threshold_inferred: Optional[float] = None,
    grid_points: Optional[np.ndarray] = None,
    smoothing_radius: Optional[float] = None,
    k_neighbors: Optional[int] = None,
    min_cluster_size: int = 1,
    downgrade_isolated: bool = True,
    warn_standalone: bool = True,
) -> np.ndarray:
    """Classifies resource blocks based on Kriging estimation variance thresholds.

    Regulatory & Geostatistical Best Practice Warning (CIM MRMR Guidelines §6.11) (TODO: Manually Verify):
    ------------------------------------------------------------------------------
    Kriging estimation variance is a purely geometric metric reflecting data spacing and
    variogram structure, independent of actual grade values or geological complexity.
    Under CIM MRMR Best Practice Guidelines (§6.11, pp. 25–26) (TODO: Manually Verify), relying solely on Kriging
    variance produces the 'spotted dog' artifact (isolated high-confidence blocks surrounded
    by lower confidence). Standalone numeric variance classification must be post-processed
    using spatial smoothing (`grid_points` + `smoothing_radius`) or embedded in multi-criteria
    classification (`classify_mineral_resources`).

    Parameters
    ----------
    kriging_variances : np.ndarray
        Array of Kriging estimation variances of shape (M,).
    variance_threshold_measured : float
        Maximum Kriging variance allowed for Measured classification.
    variance_threshold_indicated : float
        Maximum Kriging variance allowed for Indicated classification.
    variance_threshold_inferred : float, optional
        Maximum Kriging variance allowed for Inferred classification.
        Variances exceeding this threshold are assigned "Unclassified".
    grid_points : np.ndarray, optional
        Block coordinates of shape (M, 2) or (M, 3) for spatial smoothing.
    smoothing_radius : float, optional
        Spatial search radius for neighborhood majority smoothing to eliminate spotted dog artifacts.
    k_neighbors : int, optional
        Number of nearest neighbors for smoothing if smoothing_radius is not specified.
    min_cluster_size : int, default 1
        Minimum contiguous cluster size required to sustain Measured/Indicated confidence.
    downgrade_isolated : bool, default True
        If True, applies conservative downgrade to isolated high-confidence blocks.
    warn_standalone : bool, default True
        If True and no spatial smoothing is applied, issues a regulatory warning under CIM MRMR §6.11 (TODO: Manually Verify).

    Returns
    -------
    np.ndarray
        Array of category strings ("Measured", "Indicated", "Inferred", or "Unclassified") of shape (M,).
    """
    vars_arr = np.asarray(kriging_variances, dtype=float)
    if vars_arr.size == 0:
        return np.empty(0, dtype=object)

    if variance_threshold_measured <= 0.0 or variance_threshold_indicated <= 0.0:
        raise ValueError("Variance thresholds must be strictly positive.")
    if variance_threshold_measured >= variance_threshold_indicated:
        raise ValueError(
            f"variance_threshold_measured ({variance_threshold_measured}) must be < variance_threshold_indicated ({variance_threshold_indicated})."
        )
    if (
        variance_threshold_inferred is not None
        and variance_threshold_inferred <= variance_threshold_indicated
    ):
        raise ValueError(
            f"variance_threshold_inferred ({variance_threshold_inferred}) must be > variance_threshold_indicated ({variance_threshold_indicated})."
        )

    # Issue regulatory compliance warning if used as an unsmoothed standalone metric
    if warn_standalone and (
        grid_points is None
        or (smoothing_radius is None and k_neighbors is None and min_cluster_size <= 1)
    ):
        warnings.warn(
            "CIM MRMR Best Practice Guidelines (§6.11) (TODO: Manually Verify) explicitly caution against classifying "
            "mineral resources based solely on Kriging estimation variance. Kriging variance is a purely "
            "geometric measure independent of grade values and produces isolated, unmineable 'spotted dog' "
            "artifacts. Best practice requires multi-criteria classification (via 'classify_mineral_resources') "
            "or applying spatial smoothing (via 'smooth_resource_categories').",
            UserWarning,
            stacklevel=2,
        )

    valid_var = np.isfinite(vars_arr) & (vars_arr >= 0.0)
    categories = np.full(len(vars_arr), "Unclassified", dtype=object)

    is_meas = valid_var & (vars_arr <= variance_threshold_measured)
    is_ind = (
        valid_var
        & (vars_arr > variance_threshold_measured)
        & (vars_arr <= variance_threshold_indicated)
    )
    if variance_threshold_inferred is not None:
        is_inf = (
            valid_var
            & (vars_arr > variance_threshold_indicated)
            & (vars_arr <= variance_threshold_inferred)
        )
    else:
        is_inf = valid_var & (vars_arr > variance_threshold_indicated)

    categories[is_meas] = "Measured"
    categories[is_ind] = "Indicated"
    categories[is_inf] = "Inferred"

    if grid_points is not None and (
        smoothing_radius is not None or k_neighbors is not None or min_cluster_size > 1
    ):
        categories = smooth_resource_categories(
            grid_points=grid_points,
            categories=categories,
            smoothing_radius=smoothing_radius,
            k_neighbors=k_neighbors,
            min_cluster_size=min_cluster_size,
            downgrade_isolated=downgrade_isolated,
        )

    return categories


def classify_mineral_resources(
    grid_points: np.ndarray,
    samples_xy: Optional[np.ndarray] = None,
    kriging_variances: Optional[np.ndarray] = None,
    max_radius_measured: Optional[float] = None,
    max_radius_indicated: Optional[float] = None,
    min_holes_measured: int = 3,
    min_holes_indicated: int = 2,
    variance_threshold_measured: Optional[float] = None,
    variance_threshold_indicated: Optional[float] = None,
    variance_threshold_inferred: Optional[float] = None,
    slopes_of_regression: Optional[np.ndarray] = None,
    kriging_efficiencies: Optional[np.ndarray] = None,
    sor_threshold_measured: Optional[float] = None,
    sor_threshold_indicated: Optional[float] = None,
    max_slope_measured: float = 1.05,
    min_kriging_efficiency: Optional[float] = 0.0,
    max_radius_inferred: Optional[float] = None,
    is_interpolated: Optional[np.ndarray] = None,
    smoothing_radius: Optional[float] = None,
    k_neighbors_smoothing: Optional[int] = None,
    min_cluster_size: int = 1,
    downgrade_isolated: bool = True,
    anisotropy_ranges: Optional[
        Union[float, Sequence[float], Mapping[str, float]]
    ] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[str, float]]
    ] = None,
) -> np.ndarray:
    """Classifies mineral resources into Measured, Indicated, and Inferred using multi-criteria standard practice.

    CIM Definition Standards (2014) and JORC Code (Clause 20-24) Regulatory Framework (TODO: Manually Verify):
    ----------------------------------------------------------------------------------
    Mineral Resource classification categorizes estimates based on geological confidence:
    - Measured: High geological confidence; dense drilling; verified continuous geology/grades.
    - Indicated: Reasonable confidence; sufficient drilling to assume geological/grade continuity.
    - Inferred: Low confidence; limited geological evidence; continuity implied but not verified.

    Multi-Criteria Best Practice & The Conservative Downgrade Principle:
    -------------------------------------------------------------------
    Rather than relying on a single simplistic metric, this function integrates:
    1. Drill spacing and informing hole count (`classify_resources_by_drill_spacing`),
    2. Kriging estimation variance thresholds (`classify_resources_by_kriging_variance`),
    3. Kriging Neighborhood Analysis Slope of Regression / Efficiency (`classify_resources_by_sor`),
    4. Categorization Smoothing (`smooth_resource_categories`) to eliminate spotted dog artifacts.

    When multiple criteria are provided, classification follows the conservative
    minimum-confidence principle: a block only attains "Measured" status if it satisfies
    all active criteria (spacing, estimation variance, and slope of regression). If any
    criterion is not met, the block is downgraded accordingly.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of block centers of shape (M, 2) or (M, 3).
    samples_xy : np.ndarray, optional
        Array of informing drillhole coordinates of shape (N, 2) or (N, 3).
        If provided, drillhole spacing and hole count criteria are evaluated.
    kriging_variances : np.ndarray, optional
        Array of Kriging estimation variances of shape (M,).
    max_radius_measured : float, optional
        Nominal drillhole spacing radius for Measured classification. Required if samples_xy is provided.
    max_radius_indicated : float, optional
        Nominal drillhole spacing radius for Indicated classification. Required if samples_xy is provided.
    min_holes_measured : int, default 3
        Minimum informing drillholes within search radius for Measured.
    min_holes_indicated : int, default 2
        Minimum informing drillholes within search radius for Indicated.
    variance_threshold_measured : float, optional
        Maximum Kriging variance allowed for Measured classification.
    variance_threshold_indicated : float, optional
        Maximum Kriging variance allowed for Indicated classification.
    variance_threshold_inferred : float, optional
        Maximum Kriging variance allowed for Inferred classification.
    slopes_of_regression : np.ndarray, optional
        Array of Slope of Regression values of shape (M,).
    kriging_efficiencies : np.ndarray, optional
        Array of Kriging Efficiency values of shape (M,).
    sor_threshold_measured : float, optional
        Minimum Slope of Regression for Measured classification. Required if slopes_of_regression is provided.
    sor_threshold_indicated : float, optional
        Minimum Slope of Regression for Indicated classification. Required if slopes_of_regression is provided.
    max_slope_measured : float, default 1.05
        Maximum Slope of Regression for Measured classification.
    min_kriging_efficiency : float, optional, default 0.0
        Minimum Kriging Efficiency required for Measured/Indicated.
    max_radius_inferred : float, optional
        Maximum search radius for Inferred classification.
    is_interpolated : np.ndarray of bool, optional
        Boolean mask indicating whether each block lies within the interpolation hull.
        Blocks outside the hull are restricted to Inferred or Unclassified.
    smoothing_radius : float, optional
        Search radius for spatial majority smoothing to remove isolated 'spotted dog' blocks.
    k_neighbors_smoothing : int, optional
        Number of nearest neighbors for spatial smoothing if smoothing_radius is not set.
    min_cluster_size : int, default 1
        Minimum contiguous cluster size required to sustain Measured/Indicated confidence.
    downgrade_isolated : bool, default True
        Conservative principle: isolated high-confidence blocks are downgraded.
    anisotropy_ranges : float, Sequence[float], or Mapping, optional
        Directional ranges (major, minor) in 2D or (major, semi-major, minor) in 3D.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth in 2D, or azimuth, dip, plunge in 3D).

    Returns
    -------
    np.ndarray
        Array of category strings ("Measured", "Indicated", "Inferred", or "Unclassified") of shape (M,).
    """
    pts = np.asarray(grid_points)
    if pts.size == 0:
        return np.empty(0, dtype=object)

    M = len(pts)
    criteria_results = []

    # 1. Drill spacing criterion
    if samples_xy is not None:
        if max_radius_measured is None or max_radius_indicated is None:
            raise ValueError(
                "When classifying by drillhole spacing (samples_xy provided), "
                "max_radius_measured and max_radius_indicated must be explicitly specified."
            )
        cats_spacing = classify_resources_by_drill_spacing(
            grid_points=pts,
            samples_xy=samples_xy,
            max_radius_measured=max_radius_measured,
            max_radius_indicated=max_radius_indicated,
            min_holes_measured=min_holes_measured,
            min_holes_indicated=min_holes_indicated,
            max_radius_inferred=max_radius_inferred,
            is_interpolated=is_interpolated,
            anisotropy_ranges=anisotropy_ranges,
            anisotropy_angles=anisotropy_angles,
        )
        criteria_results.append(cats_spacing)

    # 2. Kriging variance criterion
    if kriging_variances is not None:
        if len(kriging_variances) != M:
            raise ValueError(
                f"Shape mismatch: kriging_variances length ({len(kriging_variances)}) must match grid_points length ({M})."
            )
        if variance_threshold_measured is None or variance_threshold_indicated is None:
            raise ValueError(
                "variance_threshold_measured and variance_threshold_indicated must be specified when passing kriging_variances."
            )
        # Check standalone variance usage in classify_mineral_resources
        if samples_xy is None and slopes_of_regression is None:
            warnings.warn(
                "CIM MRMR Best Practice Guidelines (§6.11) (TODO: Manually Verify) caution against classifying mineral resources "
                "based solely on Kriging estimation variance. Kriging variance is a geometric measure "
                "independent of actual grades and produces unmineable 'spotted dog' artifacts. Multi-criteria "
                "classification with drillhole spacing ('samples_xy') and spatial smoothing ('smoothing_radius') "
                "is strongly recommended.",
                UserWarning,
                stacklevel=2,
            )
        cats_var = classify_resources_by_kriging_variance(
            kriging_variances=kriging_variances,
            variance_threshold_measured=variance_threshold_measured,
            variance_threshold_indicated=variance_threshold_indicated,
            variance_threshold_inferred=variance_threshold_inferred,
            warn_standalone=False,
        )
        criteria_results.append(cats_var)
    elif (
        variance_threshold_measured is not None
        or variance_threshold_indicated is not None
    ):
        raise ValueError(
            "kriging_variances must be provided when variance thresholds are passed."
        )

    # 3. Slope of regression criterion
    if slopes_of_regression is not None:
        if sor_threshold_measured is None or sor_threshold_indicated is None:
            raise ValueError(
                "When classifying by Slope of Regression (slopes_of_regression provided), "
                "sor_threshold_measured and sor_threshold_indicated must be explicitly specified."
            )
        if len(slopes_of_regression) != M:
            raise ValueError(
                f"Shape mismatch: slopes_of_regression length ({len(slopes_of_regression)}) must match grid_points length ({M})."
            )
        cats_sor = classify_resources_by_sor(
            slopes_of_regression=slopes_of_regression,
            threshold_measured=sor_threshold_measured,
            threshold_indicated=sor_threshold_indicated,
            kriging_efficiencies=kriging_efficiencies,
            max_slope_measured=max_slope_measured,
            min_kriging_efficiency=min_kriging_efficiency,
        )
        criteria_results.append(cats_sor)

    if len(criteria_results) == 0:
        raise ValueError(
            "At least one classification criterion must be provided "
            "(e.g. samples_xy for drill spacing, kriging_variances with thresholds, or slopes_of_regression)."
        )

    category_ranks = {
        "Unclassified": 0,
        "Inferred": 1,
        "Indicated": 2,
        "Measured": 3,
    }
    rank_to_category = {0: "Unclassified", 1: "Inferred", 2: "Indicated", 3: "Measured"}

    ranks_matrix = np.zeros((len(criteria_results), M), dtype=int)
    for idx, cats in enumerate(criteria_results):
        ranks_matrix[idx] = [category_ranks.get(c, 0) for c in cats]

    combined_ranks = np.min(ranks_matrix, axis=0)

    if is_interpolated is not None:
        interp_mask = np.asarray(is_interpolated, dtype=bool)
        if len(interp_mask) != M:
            raise ValueError(
                f"is_interpolated length ({len(interp_mask)}) must match grid_points length ({M})."
            )
        combined_ranks = np.where(
            interp_mask, combined_ranks, np.minimum(combined_ranks, 1)
        )

    final_cats = np.array([rank_to_category[r] for r in combined_ranks], dtype=object)

    # 4. Apply Spatial Categorization Smoothing (CIM MRMR §6.11) (TODO: Manually Verify)
    if (
        smoothing_radius is not None
        or k_neighbors_smoothing is not None
        or min_cluster_size > 1
    ):
        final_cats = smooth_resource_categories(
            grid_points=pts,
            categories=final_cats,
            smoothing_radius=smoothing_radius,
            k_neighbors=k_neighbors_smoothing,
            min_cluster_size=min_cluster_size,
            downgrade_isolated=downgrade_isolated,
        )

    return final_cats


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
    reporting_basis: str = "exclusive",
    point_of_reference: str = "In situ",
) -> pd.DataFrame:
    """Formats an official Mineral Resource Statement adhering to significant figures rules.

    In accordance with CIM Definition Standards (2014), JORC Code (2012), and
    SEC S-K 1300 standards (TODO: Manually Verify):
    - Avoid False Precision: Tonnages, grades, and contained metal must be rounded
      to reflect the relative uncertainty of each classification tier:
        * Measured: 3 significant figures (or nearest deposit-scale precision)
        * Indicated: 2-3 significant figures
        * Measured + Indicated (M&I): Reported as subtotal (raw sum rounded to 3 sig figs)
        * Inferred: 1-2 significant figures, reported strictly separately from M&I.
    - Mandatory Footnote: Clarifies rounding and non-additivity.
    - RPEEE Condition: Requires reporting base-case economic cut-off, commodity price,
      metallurgical recovery, and spatial constraint (pit shell or stope shapes).
    - Inclusive vs. Exclusive Declaration: Declares whether Mineral Resources are reported
      exclusive of (additional to) or inclusive of Mineral Reserves as mandated by
      SEC Regulation S-K 1300 (§229.1303(b)(2)(ii)), JORC Code (2012, Clauses on Resource/Reserve Reporting), and CRIRSCO Template (Clause 39) (TODO: Manually Verify).
    - Point of Reference: Mandatory declaration under SEC Regulation S-K 1300
      (§229.1303(b)(1) / §229.1304(d)(1)), JORC Code (Clause 35), and CRIRSCO Template (Clause 38) (TODO: Manually Verify)
      specifying the reference point (e.g., "In situ", "Run-of-Mine", "Plant feed").

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
    reporting_basis : {"exclusive", "inclusive"}, default "exclusive"
        Mandatory statutory declaration under SEC S-K 1300 (§229.1303(b)(2)(ii)),
        JORC Code (2012, Clauses on Resource/Reserve Reporting), and CRIRSCO Template (Clause 39) (TODO: Manually Verify).
        - "exclusive": Mineral Resources are reported exclusive of (additional to) Mineral Reserves.
        - "inclusive": Mineral Resources are reported inclusive of Mineral Reserves.
    point_of_reference : str, default "In situ"
        Mandatory reference point under SEC S-K 1300 (§229.1303(b)(1)) and
        CRIRSCO Template (Clause 38) / JORC Code (Clause 35) (TODO: Manually Verify). Defaults to "In situ".

    Returns
    -------
    pd.DataFrame
        Formatted summary table compliant with international reporting disclosure,
        with metadata and mandatory footnotes accessible via `df.attrs["footnotes"]`.
    """
    if block_df.empty:
        raise ValueError("block_df cannot be empty.")

    basis = str(reporting_basis).strip().lower()
    if basis not in ("exclusive", "inclusive"):
        raise ValueError(
            f"Invalid reporting_basis '{reporting_basis}'. Must be either 'exclusive' or 'inclusive' "
            "under SEC S-K 1300 (§229.1303(b)(2)(ii)), JORC (Clause 19/38), and CRIRSCO (Clause 39)."
        )

    if not isinstance(point_of_reference, str) or not point_of_reference.strip():
        raise ValueError(
            "point_of_reference must be a non-empty string specifying the reference point "
            "(e.g., 'In situ', 'Run-of-Mine', or 'Plant feed') in compliance with "
            "SEC Regulation S-K 1300 (§229.1303(b)(1)) and CRIRSCO / JORC standards."
        )
    ref_point = point_of_reference.strip()

    # Filter by economic cutoff
    valid_blocks = block_df[block_df[grade_col] >= cutoff_grade].copy()

    # Define unit scaling factors
    t_scale = (
        1e6
        if tonnage_unit.upper() == "MT"
        else (1e3 if tonnage_unit.upper() == "KT" else 1.0)
    )
    m_unit = metal_unit.lower().strip()
    if m_unit == "kt":
        m_scale = 1e3
    elif m_unit == "mt":
        m_scale = 1e6
    elif m_unit == "t":
        m_scale = 1.0
    elif "koz" in m_unit:
        m_scale = 0.0311034768  # 1 koz = 0.0311034768 tonnes
    elif "moz" in m_unit:
        m_scale = 31.1034768  # 1 Moz = 31.1034768 tonnes
    elif "oz" in m_unit:
        m_scale = 31.1034768e-6
    else:
        m_scale = 1.0

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
    basis_fn = (
        "6. Mineral Resources are reported exclusive of Mineral Reserves in accordance with "
        "SEC Regulation S-K 1300 (§229.1303(b)(2)(ii)) and JORC Code (2012, Clauses on Resource/Reserve Reporting) (TODO: Manually Verify)."
        if basis == "exclusive"
        else "6. Mineral Resources are reported inclusive of Mineral Reserves (Measured and Indicated "
        "resources include material modified to produce reserves)."
    )

    footnotes = [
        "1. Mineral Resources are reported in accordance with CIM Definition Standards (2014) / JORC Code (2012) (TODO: Manually Verify).",
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
        basis_fn,
        f"7. Point of Reference: {ref_point} in compliance with SEC Regulation S-K 1300 (§229.1303(b)(1)) and CRIRSCO / JORC standards (TODO: Manually Verify).",
    ]
    statement_df.attrs["footnotes"] = footnotes
    statement_df.attrs["reporting_basis"] = basis
    statement_df.attrs["point_of_reference"] = ref_point

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
    """Validates block model estimates against informing drillhole composites across spatial swaths.

    Swath plots are the industry-wide validation tool mandated under NI 43-101 and JORC (TODO: Manually Verify).
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
# RESOURCE TO RESERVE DELINEATION (CIM / NI 43-101 / JORC MODIFYING FACTORS) (TODO: Manually Verify)
# =============================================================================


def calculate_cut_off_grade(
    processing_cost: float,
    ga_cost: float,
    commodity_price: float,
    metallurgical_recovery: float,
    payable_metal_factor: float,
    mining_cost: Optional[float] = None,
    sustaining_capital: float = 0.0,
    selling_cost: float = 0.0,
    royalty_pct: float = 0.0,
    metal_conversion_factor: float = 1.0,
    mining_dilution_pct: float = 0.0,
    dilution_grade: float = 0.0,
) -> float:
    """Calculates engineering and regulatory compliant cut-off grades.

    In accordance with CIM MRMR Best Practice Guidelines (§7.2.2 & Table 7-1),
    Kenneth Lane (1988), and Taylor (1972) (TODO: Manually Verify):
    - Breakeven Cut-Off Grade: Covers total costs (mining + processing + G&A + realization).
      Used to delineate the ultimate economic pit limit / stope envelope.
    - Marginal / Internal Cut-Off Grade: Covers processing + G&A + realization (mining cost
      is treated as sunk because rock must be excavated anyway to access deeper ore).
    - Mineral Reserve Cut-Off Grade: Must include sustaining capital (tailings expansions,
      equipment replacement, ongoing development) directly associated with extraction and milling.
    - In-Situ Equivalent Cut-Off: Adjusted for planned mining dilution prior to processing.

    Parameters
    ----------
    processing_cost : float
        Processing / milling operating cost per tonne of ore ($/t ore). Required.
    ga_cost : float
        General & Administrative (G&A) overhead cost per tonne of ore ($/t ore). Required.
    commodity_price : float
        Base commodity price per unit metal ($/lb, $/oz, $/t). Required.
    metallurgical_recovery : float
        Plant metallurgical recovery percentage (0 < recovery <= 100.0) or fraction
        (0 < recovery <= 1.0). Required.
    payable_metal_factor : float
        Smelter / refiner payable metal fraction (e.g., 0.95 for 95%) or percentage (95.0).
        Required without default because off-take terms vary significantly across metals
        (e.g., copper concentrate ~95%, zinc concentrate ~85%, gold doré ~99.5%).
    mining_cost : float, optional
        Ore mining cost per tonne of rock ($/t rock).
        If provided: calculates Breakeven Cut-Off Grade.
        If None: calculates Marginal / Internal Cut-Off Grade.
    sustaining_capital : float, default 0.0
        Sustaining capital cost per tonne of ore ($/t ore). Under CIM MRMR Table 7-1 (TODO: Manually Verify),
        mandatory for Mineral Reserve cut-off grades; excluded from Mineral Resources.
    selling_cost : float, default 0.0
        Smelter refining charges, treatment deductions (TC/RCs), freight, and realization
        deduction per unit metal ($/unit metal).
    royalty_pct : float, default 0.0
        Net Smelter Return (NSR) or gross revenue royalty percentage (e.g., 2.0 for 2%).
    metal_conversion_factor : float, default 1.0
        Multiplier converting grade unit into pricing unit:
        - Copper (% Cu grade to $/lb price): 1% Cu = 22.0462 lbs Cu per metric tonne -> 22.0462
        - Gold (g/t Au grade to $/oz price): 1 g/t = 1/31.1035 oz/t -> 0.0321507
        - Base metals in $/tonne metal with % grade: 1% = 0.01 tonnes metal -> 0.01
    mining_dilution_pct : float, default 0.0
        Anticipated mining dilution percentage (e.g., 10.0 for 10% dilution).
        If > 0, returns the in-situ equivalent cut-off grade required to achieve
        the economic mill-feed cut-off grade after dilution.
    dilution_grade : float, default 0.0
        Average grade of diluting wall-rock / backfill material.

    Returns
    -------
    float
        Economic cut-off grade in the corresponding grade unit.
    """
    if processing_cost < 0 or ga_cost < 0:
        raise ValueError("Operating costs (processing, G&A) cannot be negative.")
    if commodity_price <= 0:
        raise ValueError("Commodity price must be strictly positive.")
    if sustaining_capital < 0:
        raise ValueError("Sustaining capital cannot be negative.")
    if selling_cost < 0:
        raise ValueError("Selling / realization cost cannot be negative.")
    if royalty_pct < 0 or royalty_pct >= 100.0:
        raise ValueError(f"Royalty percentage must be in [0, 100)%, got {royalty_pct}")
    if metal_conversion_factor <= 0:
        raise ValueError("metal_conversion_factor must be strictly positive.")
    if mining_dilution_pct < 0:
        raise ValueError("mining_dilution_pct cannot be negative.")
    if dilution_grade < 0:
        raise ValueError("dilution_grade cannot be negative.")

    # Normalize payable metal factor
    pay_factor = (
        payable_metal_factor / 100.0
        if payable_metal_factor > 1.0
        else payable_metal_factor
    )
    if pay_factor <= 0.0 or pay_factor > 1.0:
        raise ValueError(
            f"payable_metal_factor must be in (0, 100]% or (0, 1.0], got {payable_metal_factor}"
        )

    # Normalize metallurgical recovery
    rec = (
        metallurgical_recovery / 100.0
        if metallurgical_recovery > 1.0
        else metallurgical_recovery
    )
    if rec <= 0.0 or rec > 1.0:
        raise ValueError(
            f"Metallurgical recovery must be in (0, 100]%, got {metallurgical_recovery}"
        )

    # Net realized revenue per unit metal after payability, selling/refining costs, and royalties
    royalty_factor = max(0.0, 1.0 - (royalty_pct / 100.0))
    payable_price = commodity_price * pay_factor
    net_price = (payable_price - selling_cost) * royalty_factor
    if net_price <= 0:
        raise ValueError(
            f"Net price ({net_price:.4f}) is non-positive after payability, selling deductions, and royalties."
        )

    revenue_per_grade_unit = net_price * rec * metal_conversion_factor

    total_cost = processing_cost + ga_cost + sustaining_capital
    if mining_cost is not None:
        if mining_cost < 0:
            raise ValueError("Mining cost cannot be negative.")
        total_cost += mining_cost

    mill_feed_cog = total_cost / revenue_per_grade_unit

    if mining_dilution_pct > 0.0:
        d = (
            mining_dilution_pct / 100.0
            if mining_dilution_pct > 1.0
            else mining_dilution_pct
        )
        if d >= 1.0:
            raise ValueError("mining_dilution_pct cannot be 100% or greater.")
        in_situ_cog = max(0.0, (mill_feed_cog - d * dilution_grade) / (1.0 - d))
        return float(in_situ_cog)

    return float(mill_feed_cog)


def cut_off_grade_breakdown(
    processing_cost: float,
    ga_cost: float,
    commodity_price: float,
    metallurgical_recovery: float,
    payable_metal_factor: float,
    mining_cost: Optional[float] = None,
    sustaining_capital: float = 0.0,
    selling_cost: float = 0.0,
    royalty_pct: float = 0.0,
    metal_conversion_factor: float = 1.0,
    mining_dilution_pct: float = 0.0,
    dilution_grade: float = 0.0,
) -> dict[str, Any]:
    """Provides a detailed engineering and regulatory breakdown of cut-off grade economics.

    In accordance with CIM MRMR Best Practice Guidelines (§7.2.2 & Table 7-1) (TODO: Manually Verify),
    distinguishes Resource RPEEE operating parameters from Reserve LOM feasibility parameters.

    Returns
    -------
    dict[str, Any]
        Financial and engineering audit dictionary:
        - 'gross_price': Base commodity price.
        - 'payable_price': Commodity price after smelter payability.
        - 'net_realized_price': Net price per unit metal after TC/RCs and royalties.
        - 'revenue_per_grade_unit': Net realized revenue per 1.0 unit of grade per tonne.
        - 'operating_cost_per_tonne': Milling + G&A (+ mining) cash operating cost.
        - 'sustaining_capital_per_tonne': Capital required to sustain Life-of-Mine production.
        - 'total_cost_per_tonne': Total cost applied to cut-off grade calculation.
        - 'mill_feed_cutoff_grade': Cut-off grade at process plant feed / ROM.
        - 'in_situ_cutoff_grade': Dilution-adjusted in-situ resource cut-off grade.
        - 'cutoff_type': Classification category description.
    """
    # Validate via calculate_cut_off_grade
    _ = calculate_cut_off_grade(
        processing_cost=processing_cost,
        ga_cost=ga_cost,
        commodity_price=commodity_price,
        metallurgical_recovery=metallurgical_recovery,
        payable_metal_factor=payable_metal_factor,
        mining_cost=mining_cost,
        sustaining_capital=sustaining_capital,
        selling_cost=selling_cost,
        royalty_pct=royalty_pct,
        metal_conversion_factor=metal_conversion_factor,
        mining_dilution_pct=mining_dilution_pct,
        dilution_grade=dilution_grade,
    )

    pay_factor = (
        payable_metal_factor / 100.0
        if payable_metal_factor > 1.0
        else payable_metal_factor
    )
    rec = (
        metallurgical_recovery / 100.0
        if metallurgical_recovery > 1.0
        else metallurgical_recovery
    )
    royalty_factor = max(0.0, 1.0 - (royalty_pct / 100.0))
    payable_price = commodity_price * pay_factor
    net_price = (payable_price - selling_cost) * royalty_factor
    revenue_per_grade_unit = net_price * rec * metal_conversion_factor

    op_cost = processing_cost + ga_cost + (mining_cost if mining_cost is not None else 0.0)
    total_cost = op_cost + sustaining_capital
    mill_feed_cog = total_cost / revenue_per_grade_unit

    if mining_dilution_pct > 0.0:
        d = (
            mining_dilution_pct / 100.0
            if mining_dilution_pct > 1.0
            else mining_dilution_pct
        )
        in_situ_cog = max(0.0, (mill_feed_cog - d * dilution_grade) / (1.0 - d))
    else:
        in_situ_cog = mill_feed_cog

    if mining_cost is not None:
        c_type = (
            "Breakeven Reserve (LOM)"
            if sustaining_capital > 0.0
            else "Breakeven Resource (RPEEE)"
        )
    else:
        c_type = (
            "Marginal Reserve (Internal)"
            if sustaining_capital > 0.0
            else "Marginal Resource (Internal)"
        )

    return {
        "gross_price": float(commodity_price),
        "payable_price": float(payable_price),
        "net_realized_price": float(net_price),
        "revenue_per_grade_unit": float(revenue_per_grade_unit),
        "operating_cost_per_tonne": float(op_cost),
        "sustaining_capital_per_tonne": float(sustaining_capital),
        "total_cost_per_tonne": float(total_cost),
        "mill_feed_cutoff_grade": float(mill_feed_cog),
        "in_situ_cutoff_grade": float(in_situ_cog),
        "cutoff_type": c_type,
    }


def convert_resource_to_reserve(
    resource_df: pd.DataFrame,
    mining_dilution_pct: float,
    mining_recovery_pct: float,
    cutoff_grade: float,
    dilution_grade: float = 0.0,
    category_col: str = "category",
    grade_col: str = "grade",
    tonnes_col: str = "tonnes",
    metal_factor: float = 0.01,
    cutoff_type: Literal["rom", "in_situ"] = "rom",
) -> pd.DataFrame:
    """Converts classified Mineral Resources into Mineral Reserves applying Modifying Factors.

    Strict Regulatory Compliance Framework:
    ---------------------------------------
    Under CRIRSCO International Reporting Template (Clause 28), CIM Definition Standards (2014),
    CIM MRMR Best Practice Guidelines (§7.1–§7.7 on Modifying Factors), JORC Code (2012, Clause 29 on Modifying Factors and Ore Reserves), and
    SEC Regulation S-K 1300 (§229.1302(d)(4)) (TODO: Manually Verify):
    1. 'Measured' Mineral Resources convert into 'Proven' (or 'Proved') Mineral Reserves.
    2. 'Indicated' Mineral Resources convert into 'Probable' Mineral Reserves.
    3. 'Inferred' Mineral Resources CANNOT be converted into Mineral Reserves under any
       circumstances due to low geological confidence. Inferred material is strictly excluded.
    4. 'Unclassified' or waste blocks cannot be converted into Mineral Reserves.
    5. Applies Mining Dilution: waste or contact rock inadvertently blasted and hauled with ore,
       increasing run-of-mine tonnage and lowering delivered head grade.
    6. Applies Mining Recovery (Ore Loss): unrecovered ore due to blast scatter or
       stability pillars, reducing delivered tonnage.
    7. Cut-off Grade Application (CIM MRMR §7.2.1 & SME Mining Engineering Handbook Ch. 6.1) (TODO: Manually Verify):
       The economic cut-off grade represents the marginal economic threshold of the delivered
       Run-of-Mine (ROM) product at the processing plant gate. When mining dilution is added,
       wall-rock dilution lowers the head grade of the delivered ore. Under standard practice
       (`cutoff_type='rom'`, default), the economic cut-off criterion is tested against the
       delivered ROM grade (rom_grade >= cutoff_grade). Blocks whose diluted head grade falls
       below the cut-off cannot pay for processing and delivery, and are excluded from the reserve
       and audited in `attrs['excluded_subeconomic_diluted_tonnes']`. Alternatively, if the practitioner
       has pre-adjusted the in-situ cut-off for dilution (COG_insitu = COG_rom * (1 + Dilution)),
       `cutoff_type='in_situ'` applies the filter directly to in-situ grades.

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
        Economic cut-off grade for reserve delineation. Required without default.
    dilution_grade : float, default 0.0
        Grade of the diluting waste or contact rock.
    category_col : str, default "category"
        Column containing resource classifications ('Measured', 'Indicated', 'Inferred').
    grade_col : str, default "grade"
        In-situ grade column name.
    tonnes_col : str, default "tonnes"
        In-situ tonnage column name.
    metal_factor : float, default 0.01
        Multiplier converting (grade * tonnes) to contained metal quantity.
        Default is 0.01 for percentage grades (% Cu). For g/t or ppm, use 1e-6 (or 1.0 for grams).
    cutoff_type : Literal["rom", "in_situ"], default "rom"
        Basis for economic cut-off grade evaluation:
        - "rom" (default, industry standard): Economic cut-off is applied to the delivered Run-of-Mine
          (ROM) diluted head grade (rom_grade >= cutoff_grade). Diluted blocks falling below cutoff
          are excluded as sub-economic.
        - "in_situ": Economic cut-off is applied directly to in-situ resource grades
          (in_situ_grade >= cutoff_grade). Use only if cutoff_grade has already been adjusted for dilution.

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
    if cutoff_grade < 0.0:
        raise ValueError(f"cutoff_grade cannot be negative, got {cutoff_grade}")
    if cutoff_type not in ("rom", "in_situ"):
        raise ValueError(f"cutoff_type must be 'rom' or 'in_situ', got '{cutoff_type}'")

    # Normalize category strings for robust matching
    cat_series = resource_df[category_col].astype(str).str.strip().str.capitalize()

    # Segregate and audit Inferred and Unclassified material
    is_meas = cat_series.str.startswith("Meas")
    is_ind = cat_series.str.startswith("Ind")
    is_inferred = cat_series.str.startswith("Infer")
    is_unclassified = ~(is_meas | is_ind | is_inferred)

    inferred_tonnes = float(resource_df.loc[is_inferred, tonnes_col].sum())
    unclassified_tonnes = float(resource_df.loc[is_unclassified, tonnes_col].sum())

    # Only Measured and Indicated resources are eligible for conversion
    eligible_mask = is_meas | is_ind

    # Candidate blocks: must be eligible classification and part of mineral resource (>= cutoff in-situ)
    above_insitu = eligible_mask & (resource_df[grade_col] >= cutoff_grade)
    res_subset = resource_df.loc[above_insitu].copy()

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
        empty_df.attrs["excluded_unclassified_tonnes"] = unclassified_tonnes
        empty_df.attrs["excluded_subeconomic_diluted_tonnes"] = 0.0
        empty_df.attrs["excluded_subeconomic_diluted_metal"] = 0.0
        empty_df.attrs["excluded_subeconomic_insitu_tonnes"] = 0.0
        empty_df.attrs["excluded_subeconomic_insitu_metal"] = 0.0
        empty_df.attrs["mining_dilution_pct"] = mining_dilution_pct
        empty_df.attrs["mining_recovery_pct"] = mining_recovery_pct
        empty_df.attrs["cutoff_grade"] = cutoff_grade
        empty_df.attrs["cutoff_type"] = cutoff_type
        empty_df.attrs["metal_factor"] = metal_factor
        return empty_df

    # In-situ values of candidate blocks
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
    contained_metal = t_rom * g_rom * metal_factor

    # 3. Apply Economic Cut-off Grade Criterion (CIM MRMR §7.2.1 / SME Handbook Ch. 6.1) (TODO: Manually Verify)
    if cutoff_type == "rom":
        economic_mask = g_rom >= cutoff_grade
    else:
        economic_mask = np.ones(len(res_subset), dtype=bool)

    # Audit candidate blocks that fell below cutoff after dilution
    subecon_mask = ~economic_mask
    excluded_subecon_diluted_tonnes = float(np.sum(t_rom[subecon_mask]))
    excluded_subecon_diluted_metal = float(np.sum(contained_metal[subecon_mask]))
    excluded_subecon_insitu_tonnes = float(np.sum(t_insitu[subecon_mask]))
    excluded_subecon_insitu_metal = float(
        np.sum((t_insitu * g_insitu * metal_factor)[subecon_mask])
    )

    if not np.any(economic_mask):
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
        empty_df.attrs["excluded_unclassified_tonnes"] = unclassified_tonnes
        empty_df.attrs["excluded_subeconomic_diluted_tonnes"] = (
            excluded_subecon_diluted_tonnes
        )
        empty_df.attrs["excluded_subeconomic_diluted_metal"] = (
            excluded_subecon_diluted_metal
        )
        empty_df.attrs["excluded_subeconomic_insitu_tonnes"] = (
            excluded_subecon_insitu_tonnes
        )
        empty_df.attrs["excluded_subeconomic_insitu_metal"] = (
            excluded_subecon_insitu_metal
        )
        empty_df.attrs["mining_dilution_pct"] = mining_dilution_pct
        empty_df.attrs["mining_recovery_pct"] = mining_recovery_pct
        empty_df.attrs["cutoff_grade"] = cutoff_grade
        empty_df.attrs["cutoff_type"] = cutoff_type
        empty_df.attrs["metal_factor"] = metal_factor
        return empty_df

    # Map categories strictly to reserve classifications for economic blocks
    subset_cats = cat_series.loc[above_insitu].iloc[economic_mask]
    reserve_cats = np.empty(int(np.sum(economic_mask)), dtype=object)

    subset_meas = subset_cats.str.startswith("Meas")
    subset_ind = subset_cats.str.startswith("Ind")

    reserve_cats[subset_meas.to_numpy()] = "Proven Reserve"
    reserve_cats[subset_ind.to_numpy()] = "Probable Reserve"

    econ_res_subset = res_subset.iloc[economic_mask].copy()

    reserve_df = pd.DataFrame(
        {
            "reserve_category": reserve_cats,
            "rom_tonnes": t_rom[economic_mask],
            "rom_grade": g_rom[economic_mask],
            "contained_metal": contained_metal[economic_mask],
            "in_situ_tonnes": t_insitu[economic_mask],
            "in_situ_grade": g_insitu[economic_mask],
        },
        index=econ_res_subset.index,
    )

    # Preserve spatial coordinates if present
    for coord in ("x", "y", "z", "easting", "northing", "elevation"):
        if coord in econ_res_subset.columns:
            reserve_df[coord] = econ_res_subset[coord]

    # Attach conversion audit metadata
    reserve_df.attrs["excluded_inferred_tonnes"] = inferred_tonnes
    reserve_df.attrs["excluded_unclassified_tonnes"] = unclassified_tonnes
    reserve_df.attrs["excluded_subeconomic_diluted_tonnes"] = (
        excluded_subecon_diluted_tonnes
    )
    reserve_df.attrs["excluded_subeconomic_diluted_metal"] = (
        excluded_subecon_diluted_metal
    )
    reserve_df.attrs["excluded_subeconomic_insitu_tonnes"] = (
        excluded_subecon_insitu_tonnes
    )
    reserve_df.attrs["excluded_subeconomic_insitu_metal"] = (
        excluded_subecon_insitu_metal
    )
    reserve_df.attrs["mining_dilution_pct"] = mining_dilution_pct
    reserve_df.attrs["mining_recovery_pct"] = mining_recovery_pct
    reserve_df.attrs["cutoff_grade"] = cutoff_grade
    reserve_df.attrs["cutoff_type"] = cutoff_type
    reserve_df.attrs["metal_factor"] = metal_factor
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
    metal_factor: float = 0.01,
    rpeee_constraint: str = "Constrained within engineered final pit design",
    point_of_reference: str = "Run-of-Mine (ROM) delivered to processing facility",
    sig_figs: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Formats an official NI 43-101 / JORC compliant Mineral Reserve Statement (TODO: Manually Verify).

    Enforces:
    - Segregation into Proven, Probable, and Total Proven + Probable reserves.
    - Tiered significant figures rounding to eliminate false precision.
    - Mandatory regulatory footnotes disclosing all applied Modifying Factors.
    - Mandatory Point of Reference: Discloses the reference point (e.g., Run-of-Mine delivered to
      processing facility, plant feed, or marketable product) under SEC Regulation S-K 1300
      (§229.1303(b)(1)), JORC Code (Clause 35), and CRIRSCO Template (Clause 38) (TODO: Manually Verify).

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
    metal_factor : float, default 0.01
        Multiplier to convert (grade * tonnes) to raw metal tonnes. Default 0.01 for %.
    rpeee_constraint : str, default "Constrained within engineered final pit design"
        Engineering design constraint statement.
    point_of_reference : str, default "Run-of-Mine (ROM) delivered to processing facility"
        Mandatory reference point under SEC S-K 1300 (§229.1303(b)(1)) and
        CRIRSCO Template (Clause 38) / JORC Code (Clause 35) (TODO: Manually Verify).
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

    if not isinstance(point_of_reference, str) or not point_of_reference.strip():
        raise ValueError(
            "point_of_reference must be a non-empty string specifying the reference point "
            "(e.g., 'Run-of-Mine (ROM) delivered to processing facility', 'Plant feed', or 'Marketable product') "
            "in compliance with SEC Regulation S-K 1300 (§229.1303(b)(1)) and CRIRSCO / JORC standards."
        )
    ref_point = point_of_reference.strip()

    default_sig_figs = {
        "Proven Reserve": 3,
        "Probable Reserve": 2,
        "Total Proven + Probable": 3,
    }
    if sig_figs:
        default_sig_figs.update(sig_figs)

    # Unit scaling
    t_scale = (
        1e6
        if tonnage_unit.upper() == "MT"
        else (1e3 if tonnage_unit.upper() == "KT" else 1.0)
    )
    m_unit = metal_unit.lower().strip()
    if m_unit == "kt":
        m_scale = 1e3
    elif m_unit == "mt":
        m_scale = 1e6
    elif m_unit == "t":
        m_scale = 1.0
    elif "koz" in m_unit:
        m_scale = 0.0311034768  # 1 koz = 0.0311034768 tonnes
    elif "moz" in m_unit:
        m_scale = 31.1034768  # 1 Moz = 31.1034768 tonnes
    elif "oz" in m_unit:
        m_scale = 31.1034768e-6
    else:
        m_scale = 1.0

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
    m_prov = t_prov * g_prov * metal_factor if t_prov > 0 else 0.0

    # 2. Probable Reserves
    prob_mask = reserve_df[category_col].astype(str).str.strip().str.startswith("Prob")
    prob_df = reserve_df.loc[prob_mask]

    t_prob = float(prob_df[tonnes_col].sum())
    g_prob = (
        float((prob_df[tonnes_col] * prob_df[grade_col]).sum() / max(1e-9, t_prob))
        if t_prob > 0
        else 0.0
    )
    m_prob = t_prob * g_prob * metal_factor if t_prob > 0 else 0.0

    # 3. Total Proven + Probable
    t_tot = t_prov + t_prob
    g_tot = (
        float((t_prov * g_prov + t_prob * g_prob) / max(1e-9, t_tot))
        if t_tot > 0
        else 0.0
    )
    m_tot = m_prov + m_prob

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

    # Mandatory CIM / JORC / SEC S-K 1300 Compliance Footnotes (TODO: Manually Verify)
    footnotes = [
        "1. Mineral Reserves are reported in accordance with CIM Definition Standards (2014) / JORC Code (2012) (TODO: Manually Verify).",
        "2. Tonnages, grades, and contained metal are rounded to reflect relative uncertainty. Totals may not sum due to rounding.",
        "3. Mineral Reserves represent the economically mineable part of Measured and Indicated Mineral Resources demonstrated by at least a Pre-Feasibility Study.",
        (
            f"4. Modifying Factors applied: Mining Dilution = {mining_dilution_pct:.1f}%, Mining Recovery = {mining_recovery_pct:.1f}%, "
            f"Metallurgical Recovery = {metallurgical_recovery:.1f}%, Base Cutoff Grade = {cutoff_grade:.2f}{grade_unit}, Commodity Price = {commodity_price}."
        ),
        f"5. {rpeee_constraint}.",
        f"6. Point of Reference: {ref_point} in compliance with SEC Regulation S-K 1300 (§229.1303(b)(1)) and CRIRSCO / JORC standards (TODO: Manually Verify).",
    ]
    statement_df.attrs["footnotes"] = footnotes
    statement_df.attrs["point_of_reference"] = ref_point
    statement_df.attrs["metal_factor"] = metal_factor
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
    metal_factor: float = 0.01,
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
        Metal scale unit ("t", "kt", "Mt", "koz").
    metal_factor : float, default 0.01
        Multiplier converting (grade * tonnes) to raw metal tonnes. Default 0.01 for %.
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
    t_scale = (
        1e6
        if tonnage_unit.upper() == "MT"
        else (1e3 if tonnage_unit.upper() == "KT" else 1.0)
    )
    m_unit = metal_unit.lower().strip()
    if m_unit == "kt":
        m_scale = 1e3
    elif m_unit == "mt":
        m_scale = 1e6
    elif m_unit == "t":
        m_scale = 1.0
    elif "koz" in m_unit:
        m_scale = 0.0311034768  # 1 koz = 0.0311034768 tonnes
    elif "moz" in m_unit:
        m_scale = 31.1034768  # 1 Moz = 31.1034768 tonnes
    elif "oz" in m_unit:
        m_scale = 31.1034768e-6
    else:
        m_scale = 1.0

    # 1. Filter for Measured & Indicated in-situ resource
    cat_s = resource_df[category_col].astype(str).str.strip().str.capitalize()
    is_mi = cat_s.str.startswith("Meas") | cat_s.str.startswith("Ind")
    mi_df = resource_df.loc[is_mi].copy()

    # Step 1: In-Situ M&I
    t_step1 = float(mi_df[tonnes_col].sum())
    m_step1 = float((mi_df[tonnes_col] * mi_df[grade_col] * metal_factor).sum())

    # Step 2: Cut-Off Loss
    sub_cutoff = mi_df[mi_df[grade_col] < cutoff_grade]
    delta_t_co = -float(sub_cutoff[tonnes_col].sum())
    delta_m_co = -float(
        (sub_cutoff[tonnes_col] * sub_cutoff[grade_col] * metal_factor).sum()
    )

    t_above_co = t_step1 + delta_t_co
    m_above_co = m_step1 + delta_m_co

    # Step 3: Mining Ore Loss
    ore_loss_frac = max(0.0, 1.0 - (mining_recovery_pct / 100.0))
    delta_t_loss = -float(t_above_co * ore_loss_frac)
    delta_m_loss = -float(m_above_co * ore_loss_frac)

    # Step 4: Mining Dilution
    dil_frac = (mining_dilution_pct / 100.0) * (mining_recovery_pct / 100.0)
    delta_t_dil = float(t_above_co * dil_frac)
    delta_m_dil = float(delta_t_dil * dilution_grade * metal_factor)

    # Step 5: Final ROM Reserve
    t_final = float(reserve_df["rom_tonnes"].sum())
    m_final = float(reserve_df["contained_metal"].sum())

    # Check for dilution sub-cutoff loss (material degraded below cutoff by dilution)
    t_running = t_above_co + delta_t_loss + delta_t_dil
    m_running = m_above_co + delta_m_loss + delta_m_dil
    delta_t_spoilage = t_final - t_running
    delta_m_spoilage = m_final - m_running

    has_spoilage = (abs(delta_t_spoilage / max(1e-9, t_step1)) > 1e-4) or (
        abs(delta_m_spoilage / max(1e-9, m_step1)) > 1e-4
    )

    if has_spoilage:
        steps = [
            "In-Situ M&I\nResource",
            f"Sub-Cutoff\n(<{cutoff_grade:.2f}{grade_unit})",
            f"Ore Loss\n({100-mining_recovery_pct:.1f}%)",
            f"Dilution\n(+{mining_dilution_pct:.1f}%)",
            f"Dilution Spoilage\n(<{cutoff_grade:.2f}{grade_unit})",
            "Run-of-Mine\nReserve",
        ]
        t_deltas = [
            t_step1 / t_scale,
            delta_t_co / t_scale,
            delta_t_loss / t_scale,
            delta_t_dil / t_scale,
            delta_t_spoilage / t_scale,
            t_final / t_scale,
        ]
        m_deltas = [
            m_step1 / m_scale,
            delta_m_co / m_scale,
            delta_m_loss / m_scale,
            delta_m_dil / m_scale,
            delta_m_spoilage / m_scale,
            m_final / m_scale,
        ]
    else:
        steps = [
            "In-Situ M&I\nResource",
            f"Sub-Cutoff\n(<{cutoff_grade:.2f}{grade_unit})",
            f"Ore Loss\n({100-mining_recovery_pct:.1f}%)",
            f"Dilution\n(+{mining_dilution_pct:.1f}%)",
            "Run-of-Mine\nReserve",
        ]
        t_deltas = [
            t_step1 / t_scale,
            delta_t_co / t_scale,
            delta_t_loss / t_scale,
            delta_t_dil / t_scale,
            t_final / t_scale,
        ]
        m_deltas = [
            m_step1 / m_scale,
            delta_m_co / m_scale,
            delta_m_loss / m_scale,
            delta_m_dil / m_scale,
            m_final / m_scale,
        ]

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=figsize)

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
        t_deltas,
        t_final / t_scale,
        tonnage_unit,
        "Ore Tonnage",
    )
    ax_t.set_title("Tonnage Reconciliation Waterfall", fontsize=12, fontweight="bold")

    # Render Contained Metal Waterfall
    _render_waterfall(
        ax_m,
        m_deltas,
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


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: 2D plan scatter plot of block centroids.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Scope of this function: strictly 2D plan view mapping; add `bench_z: Optional[float] = None`
#   and `elevation_tolerance` to render a single horizontal mining bench slice without overplotting.
# - Keep property `boundary` distinct from drillhole data envelope (`is_within_convex_hull`).
# - Transparent DataFrames: directly read block model DataFrame without custom wrapper objects.
# - Following codebase conventions, 3D visualizations are implemented as separate functions:
#   1. `plot_reserve_classification_3d_isometric`: dedicated static 3D isometric view.
#   2. `plot_reserve_classification_3d_interactive`: dedicated interactive 3D explorer (sliders, HUD).
#   3. `plot_reserve_classification_bench_gallery`: dedicated multi-bench elevation gallery.
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


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: 2D plan scatter plot of block centroids.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Scope of this function: strictly 2D plan view mapping; add `bench_z: Optional[float] = None`
#   and `elevation_tolerance` to render a single horizontal mining bench slice without overplotting.
# - Keep property `boundary` distinct from drillhole data envelope (`is_within_convex_hull`).
# - Transparent DataFrames: directly read block model DataFrame without custom wrapper objects.
# - Following codebase conventions, 3D visualizations are implemented as separate functions:
#   1. `plot_resource_classification_3d_isometric`: dedicated static 3D isometric view (following `plot_block_model_3d_isometric`).
#   2. `plot_resource_classification_3d_interactive`: dedicated interactive 3D explorer (following `plot_block_model_3d_interactive` with category toggles and bench sliders).
#   3. `plot_resource_classification_bench_gallery`: dedicated multi-bench elevation gallery (following `plot_block_model_bench_gallery`).
#   4. `plot_resource_classification_orthogonal_slices`: dedicated 3-view orthogonal slice viewer (following `plot_block_model_orthogonal_slices`).
def plot_resource_classification_map(
    block_model: pd.DataFrame,
    category_col: str = "category",
    boundary: Optional[Sequence[tuple[float, float]]] = None,
    drillholes: Optional[pd.DataFrame] = None,
    x_col: str = "x",
    y_col: str = "y",
    title: str = "CIM / JORC Mineral Resource Classification Map",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Renders 2D spatial mine plan map colored by regulatory mineral resource category.

    In accordance with CIM Definition Standards (2014) and JORC Code (2012) (TODO: Manually Verify):
    - Measured Resource: Forest Green (#2ca02c)
    - Indicated Resource: Royal Blue (#1f77b4)
    - Inferred Resource: Amber / Orange (#ff7f0e)
    - Unclassified: Light Grey (#d3d3d3)

    Parameters
    ----------
    block_model : pd.DataFrame
        Table with block coordinates and category column (e.g. 'category').
    category_col : str, default "category"
        Column containing resource classification string ("Measured", "Indicated", "Inferred", "Unclassified").
    boundary : Sequence[tuple[float, float]], optional
        Perimeter boundary polygon coordinates.
    drillholes : pd.DataFrame, optional
        Collar coordinates table with 'x' and 'y' columns.
    x_col, y_col : str, default "x", "y"
        Coordinate column names.
    title : str, default "CIM / JORC Mineral Resource Classification Map"
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
        "Measured": "#2ca02c",  # Forest green
        "Indicated": "#1f77b4",  # Royal Blue
        "Inferred": "#ff7f0e",  # Amber / Orange
        "Unclassified": "#d3d3d3",  # Light grey
    }

    # Normalize categories
    cats_lower = block_model[category_col].astype(str).str.strip().str.lower()

    # Draw blocks grouped by category
    for cat_name, color in color_palette.items():
        mask = cats_lower == cat_name.lower()
        sub = block_model[mask]
        if len(sub) > 0:
            ax.scatter(
                sub[x_col],
                sub[y_col],
                c=color,
                label=f"{cat_name} ({len(sub):,} blocks)",
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
            label="Concession Perimeter",
        )

    # Draw drillholes if provided
    if drillholes is not None and not drillholes.empty:
        dh_x = (
            drillholes["x"]
            if "x" in drillholes
            else (drillholes[x_col] if x_col in drillholes else drillholes.iloc[:, 0])
        )
        dh_y = (
            drillholes["y"]
            if "y" in drillholes
            else (drillholes[y_col] if y_col in drillholes else drillholes.iloc[:, 1])
        )
        ax.scatter(
            dh_x,
            dh_y,
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
    unassayed_treatment: str = "zero",
    unassayed_grade: float = 0.0,
    max_gap_length: float = 0.0,
    min_coverage_ratio: float = 0.0,
    missing_grade_values: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Down-hole regular compositing with domain-boundary constraints and unassayed gap handling.

    Solves the Support Effect in mining geostatistics: raw drill core assays have
    variable sample lengths cut along geological and alteration contacts. Equal
    volume/length support is required prior to spatial statistical evaluation and
    variography.

    In accordance with CIM Exploration Best Practice Guidelines (2018, §4.3 & §5.1)
    and JORC Code (2012, Table 1) (TODO: Manually Verify), unassayed core loss intervals, cavities, or missing
    assays must never be assumed to carry the grade of adjacent mineralization without
    explicit justification.

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
        CRITICAL: Compositing strictly resets at domain contacts.
        Composites never cross domain boundaries. Furthermore, non-contiguous
        intercepts of the same domain (e.g. A -> B -> A) are treated as distinct runs.
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
    unassayed_treatment : {"zero", "split", "ignore", "error"}, default "zero"
        Strategy for handling unsampled gaps and missing/NaN assay intervals:
        - "zero": Replaces unassayed/missing intervals with unassayed_grade (typically 0.0),
          conserving physical contained metal over the full composite length (CIM best practice) (TODO: Manually Verify).
          If max_gap_length > 0 and a gap exceeds max_gap_length, the run is split instead.
        - "split": Treats any unassayed gap exceeding max_gap_length (and missing assay rows)
          as a hard boundary, splitting the drillhole run so intervals before and after
          the gap are composited independently.
        - "ignore": Length-weights only the assayed segments (legacy behavior), but records
          true sampled_length and coverage_ratio in the output.
        - "error": Raises ValueError if any unassayed gap or missing assay is detected.
    unassayed_grade : float, default 0.0
        Grade assigned to unassayed/missing core intervals when unassayed_treatment="zero".
    max_gap_length : float, default 0.0
        Maximum allowable depth gap before triggering gap treatment or splitting.
    min_coverage_ratio : float, default 0.0
        Minimum ratio of assayed core length to total composite length (sampled_length / length)
        required to retain a composite. Composites with coverage below this threshold are discarded.
    missing_grade_values : Sequence[float], optional
        Numeric codes treated as missing assays (e.g., [-999.0, -1.0]). If None, NaN/null and
        negative values are treated as missing.

    Returns
    -------
    pd.DataFrame
        Composited drillhole intervals with length-weighted grades, sampled lengths,
        coverage ratios, and midpoint coordinates.
    """
    if composite_length <= 0:
        raise ValueError(
            f"composite_length must be strictly positive, got {composite_length}"
        )
    if remnant_strategy not in ("discard", "distribute"):
        raise ValueError(
            f"remnant_strategy must be 'discard' or 'distribute', got '{remnant_strategy}'"
        )
    if unassayed_treatment not in ("zero", "split", "ignore", "error"):
        raise ValueError(
            f"unassayed_treatment must be 'zero', 'split', 'ignore', or 'error', got '{unassayed_treatment}'"
        )
    if not (0.0 <= min_length_ratio <= 1.0):
        raise ValueError(
            f"min_length_ratio must be in [0, 1], got {min_length_ratio}"
        )
    if not (0.0 <= min_coverage_ratio <= 1.0):
        raise ValueError(
            f"min_coverage_ratio must be in [0, 1], got {min_coverage_ratio}"
        )
    if max_gap_length < 0.0:
        raise ValueError(
            f"max_gap_length must be non-negative, got {max_gap_length}"
        )

    for col in (hole_id_col, from_col, to_col, grade_col):
        if col not in assay_df.columns:
            raise ValueError(f"Required column '{col}' not found in assay DataFrame.")

    if domain_col is not None and domain_col not in assay_df.columns:
        raise ValueError(f"Specified domain column '{domain_col}' not found.")

    if density_col is not None and density_col not in assay_df.columns:
        raise ValueError(f"Specified density column '{density_col}' not found.")

    coord_cols = [c for c in ("x", "y", "z", "elevation") if c in assay_df.columns]

    if assay_df.empty:
        out_cols = [
            hole_id_col,
            from_col,
            to_col,
            "length",
            grade_col,
            "sampled_length",
            "coverage_ratio",
        ]
        if domain_col is not None:
            out_cols.append(domain_col)
        out_cols.extend(coord_cols)
        out_empty = pd.DataFrame(columns=out_cols)
        out_empty.attrs["composite_length"] = composite_length
        out_empty.attrs["remnant_strategy"] = remnant_strategy
        out_empty.attrs["unassayed_treatment"] = unassayed_treatment
        out_empty.attrs["unassayed_grade"] = unassayed_grade
        out_empty.attrs["discarded_remnants_count"] = 0
        out_empty.attrs["discarded_low_coverage_count"] = 0
        out_empty.attrs["unassayed_gaps_count"] = 0
        out_empty.attrs["unassayed_gaps_total_length"] = 0.0
        out_empty.attrs["contiguous_runs_count"] = 0
        return out_empty

    def is_missing_grade(val) -> bool:
        if pd.isna(val):
            return True
        try:
            fval = float(val)
        except (ValueError, TypeError):
            return True
        if missing_grade_values is not None:
            return any(np.isclose(fval, float(m)) for m in missing_grade_values)
        return fval < 0.0

    composite_records = []
    discarded_remnants_count = 0
    discarded_low_cov_count = 0
    total_gaps_count = 0
    total_gaps_length = 0.0
    total_runs_count = 0

    # Group by hole_id only to maintain sequential down-hole continuity
    for hole_id, grp in assay_df.groupby(hole_id_col, sort=False):
        sub = grp.sort_values([from_col, to_col]).copy()
        if len(sub) == 0:
            continue

        raw_from = sub[from_col].to_numpy(dtype=float)
        raw_to = sub[to_col].to_numpy(dtype=float)
        raw_grade = sub[grade_col].to_numpy()
        raw_domain = sub[domain_col].to_numpy() if domain_col is not None else None
        raw_density = (
            sub[density_col].to_numpy(dtype=float)
            if density_col is not None
            else np.ones(len(sub), dtype=float)
        )
        mean_density = float(np.nanmean(raw_density)) if len(raw_density) > 0 else 1.0
        if not np.isfinite(mean_density) or mean_density <= 0:
            mean_density = 1.0

        raw_coords = {c: sub[c].to_numpy(dtype=float) for c in coord_cols}

        # 1. Validation: from < to and strictly no overlapping intervals
        n_rows = len(raw_from)
        for i in range(n_rows):
            f_i = raw_from[i]
            t_i = raw_to[i]
            if t_i <= f_i:
                raise ValueError(
                    f"Invalid interval in hole '{hole_id}' at row {i}: "
                    f"from_m ({f_i}) >= to_m ({t_i})"
                )
            if i > 0 and f_i < raw_to[i - 1] - 1e-6:
                raise ValueError(
                    f"Overlapping intervals detected in drillhole '{hole_id}': "
                    f"interval [{f_i}, {t_i}] overlaps with preceding interval "
                    f"[{raw_from[i-1]}, {raw_to[i-1]}]"
                )

        # 2. Build segments and partition into contiguous runs
        # A run is a contiguous list of segments sharing the same domain and unbroken by splits.
        runs: list[list[dict]] = []
        current_run: list[dict] = []
        current_dom = None

        for i in range(n_rows):
            f_i = raw_from[i]
            t_i = raw_to[i]
            val_i = raw_grade[i]
            dom_i = raw_domain[i] if raw_domain is not None else None
            dens_i = (
                raw_density[i]
                if (np.isfinite(raw_density[i]) and raw_density[i] > 0)
                else mean_density
            )
            c_dict = {c: raw_coords[c][i] for c in coord_cols}

            missing_i = is_missing_grade(val_i)

            # Check gap between previous interval and current interval
            if i > 0:
                prev_to = raw_to[i - 1]
                gap_len = f_i - prev_to
                if gap_len > 1e-6:
                    total_gaps_count += 1
                    total_gaps_length += gap_len

                    if unassayed_treatment == "error":
                        raise ValueError(
                            f"Unassayed depth gap of {gap_len:.2f}m detected in hole '{hole_id}' "
                            f"between {prev_to}m and {f_i}m."
                        )

                    gap_causes_split = (unassayed_treatment == "split") or (
                        unassayed_treatment == "zero"
                        and max_gap_length > 0.0
                        and gap_len > max_gap_length
                    )

                    if gap_causes_split:
                        if current_run:
                            runs.append(current_run)
                            current_run = []
                            current_dom = None
                    else:
                        # Materialize gap segment (treated as unassayed rock)
                        gap_coords = {}
                        prev_mid = (raw_from[i - 1] + prev_to) / 2.0
                        curr_mid = (f_i + t_i) / 2.0
                        gap_mid = (prev_to + f_i) / 2.0
                        denom = curr_mid - prev_mid
                        for c in coord_cols:
                            p_val = raw_coords[c][i - 1]
                            c_val = raw_coords[c][i]
                            if np.isfinite(p_val) and np.isfinite(c_val) and abs(denom) > 1e-9:
                                gap_coords[c] = float(
                                    p_val + ((gap_mid - prev_mid) / denom) * (c_val - p_val)
                                )
                            elif np.isfinite(p_val):
                                gap_coords[c] = float(p_val)
                            else:
                                gap_coords[c] = float(c_val)

                        current_run.append({
                            "from": prev_to,
                            "to": f_i,
                            "length": gap_len,
                            "grade": unassayed_grade,
                            "density": mean_density,
                            "is_sampled": False,
                            "domain": current_dom,
                            "coords": gap_coords,
                        })

            # Check domain transition
            if domain_col is not None and current_dom is not None and dom_i != current_dom:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                    current_dom = None

            # Handle current interval
            if missing_i:
                total_gaps_count += 1
                total_gaps_length += (t_i - f_i)
                if unassayed_treatment == "error":
                    raise ValueError(
                        f"Missing or unassayed grade detected in hole '{hole_id}' "
                        f"between {f_i}m and {t_i}m."
                    )
                if unassayed_treatment == "split":
                    if current_run:
                        runs.append(current_run)
                        current_run = []
                        current_dom = None
                    continue

                # For "zero" or "ignore", keep interval marked as unsampled
                seg = {
                    "from": f_i,
                    "to": t_i,
                    "length": t_i - f_i,
                    "grade": unassayed_grade,
                    "density": dens_i,
                    "is_sampled": False,
                    "domain": dom_i,
                    "coords": c_dict,
                }
            else:
                seg = {
                    "from": f_i,
                    "to": t_i,
                    "length": t_i - f_i,
                    "grade": float(val_i),
                    "density": dens_i,
                    "is_sampled": True,
                    "domain": dom_i,
                    "coords": c_dict,
                }

            current_run.append(seg)
            current_dom = dom_i

        if current_run:
            runs.append(current_run)

        # 3. Composite each contiguous run independently
        for run in runs:
            if not run:
                continue
            total_runs_count += 1

            run_start = run[0]["from"]
            run_end = run[-1]["to"]
            run_len = run_end - run_start
            run_dom = run[0]["domain"]

            if run_len <= 1e-9:
                continue

            # Establish composite interval boundaries
            comp_intervals = []
            if remnant_strategy == "distribute":
                n_comp = max(1, int(round(run_len / composite_length)))
                actual_len = run_len / n_comp
                if run_len < (min_length_ratio * composite_length):
                    discarded_remnants_count += 1
                    continue
                for ci in range(n_comp):
                    comp_intervals.append((
                        run_start + ci * actual_len,
                        run_start + (ci + 1) * actual_len,
                    ))
            else:  # "discard"
                curr = run_start
                while curr + composite_length <= run_end + 1e-9:
                    comp_intervals.append((curr, curr + composite_length))
                    curr += composite_length
                remnant_len = run_end - curr
                if remnant_len >= (min_length_ratio * composite_length) - 1e-9:
                    comp_intervals.append((curr, run_end))
                elif remnant_len > 1e-9:
                    discarded_remnants_count += 1

            # Extract numpy arrays for the run's segments
            seg_from = np.array([s["from"] for s in run], dtype=float)
            seg_to = np.array([s["to"] for s in run], dtype=float)
            seg_grade = np.array([s["grade"] for s in run], dtype=float)
            seg_dens = np.array([s["density"] for s in run], dtype=float)
            seg_sampled = np.array([s["is_sampled"] for s in run], dtype=bool)

            for c_from, c_to in comp_intervals:
                c_len = c_to - c_from
                if c_len <= 1e-9:
                    continue

                # Overlap calculation
                overlap_starts = np.maximum(c_from, seg_from)
                overlap_ends = np.minimum(c_to, seg_to)
                overlaps = np.maximum(0.0, overlap_ends - overlap_starts)

                valid_mask = overlaps > 1e-9
                if not np.any(valid_mask):
                    continue

                active_lens = overlaps[valid_mask]
                active_grades = seg_grade[valid_mask]
                active_dens = seg_dens[valid_mask]
                active_sampled = seg_sampled[valid_mask]

                sampled_len = float(active_lens[active_sampled].sum())
                coverage_ratio = float(sampled_len / c_len) if c_len > 0 else 0.0

                # Check coverage threshold
                if coverage_ratio < min_coverage_ratio - 1e-9:
                    discarded_low_cov_count += 1
                    continue

                # Compute weighted grade based on unassayed_treatment
                if unassayed_treatment == "ignore":
                    if not np.any(active_sampled):
                        continue
                    sample_weights = active_lens[active_sampled] * active_dens[active_sampled]
                    tot_w = float(sample_weights.sum())
                    if tot_w <= 0:
                        continue
                    weighted_grade = float(
                        (sample_weights * active_grades[active_sampled]).sum() / tot_w
                    )
                else:
                    weights = active_lens * active_dens
                    tot_w = float(weights.sum())
                    if tot_w <= 0:
                        continue
                    weighted_grade = float(
                        (weights * active_grades).sum() / tot_w
                    )

                rec: dict = {
                    hole_id_col: hole_id,
                    from_col: c_from,
                    to_col: c_to,
                    "length": c_len,
                    grade_col: weighted_grade,
                    "sampled_length": sampled_len,
                    "coverage_ratio": coverage_ratio,
                }
                if domain_col is not None:
                    rec[domain_col] = run_dom

                # Interpolate spatial coordinates to composite midpoint
                c_mid = (c_from + c_to) / 2.0
                for c_name in coord_cols:
                    raw_c = np.array([s["coords"].get(c_name, np.nan) for s in run], dtype=float)
                    active_c = raw_c[valid_mask]
                    c_valid = np.isfinite(active_c)
                    if np.any(c_valid):
                        c_weights = active_lens[c_valid]
                        comp_coord = float((c_weights * active_c[c_valid]).sum() / c_weights.sum())
                    else:
                        comp_coord = np.nan
                    rec[c_name] = comp_coord

                composite_records.append(rec)

    out_df = pd.DataFrame(composite_records)
    out_df.attrs["composite_length"] = composite_length
    out_df.attrs["remnant_strategy"] = remnant_strategy
    out_df.attrs["unassayed_treatment"] = unassayed_treatment
    out_df.attrs["unassayed_grade"] = unassayed_grade
    out_df.attrs["discarded_remnants_count"] = discarded_remnants_count
    out_df.attrs["discarded_low_coverage_count"] = discarded_low_cov_count
    out_df.attrs["unassayed_gaps_count"] = total_gaps_count
    out_df.attrs["unassayed_gaps_total_length"] = total_gaps_length
    out_df.attrs["contiguous_runs_count"] = total_runs_count
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

    Complies with NI 43-101 and JORC reporting standards (TODO: Manually Verify) for exploratory data
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

    Mandatory deliverable for NI 43-101 / JORC Section 14 (TODO: Manually Verify) to document whether
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
    stockpile_data: Optional[Union[pd.DataFrame, Dict[str, float]]] = None,
    tonnes_col: str = "tonnes",
    grade_col: str = "grade",
    period_col: Optional[str] = None,
    grade_unit: str = "% Cu",
) -> pd.DataFrame:
    """Reconciles mine production against the long-term mineral reserve model.

    Implements the Harry Parker (2012) F1, F2, F3 reconciliation framework (TODO: Manually Verify):
    - F1 Factor (Model to Mine / Ore Selection):
      F1 = Metal(Grade Control) / Metal(Reserve)
      Measures reserve model accuracy and local estimation bias.
    - F2 Factor (Mine to Mill / Delivery Efficiency):
      F2 = Metal(Plant Received) / Metal(Grade Control)
      Measures mining execution: unplanned dilution, ore loss, and misrouting.
    - F3 Factor (Total System Reconciliation):
      F3 = F1 * F2 = Metal(Plant Received) / Metal(Reserve)
      Measures total value chain health and cash-flow delivery.

    Stockpile Accounting (Harry Parker 2012 §4 & SME Mining Engineering Handbook §13) (TODO: Manually Verify):
    In operating mines, ore mined does not necessarily equal ore processed in the same period
    due to ROM stockpiling, low-grade blending pads, and stockpile drawdowns.
    Without explicit stockpile accounting, stockpiling produces false-alarm drops in F2 (apparent ore loss),
    while reclaim produces false-alarm surges in F2 (phantom metal creation).
    When `stockpile_data` is provided, conservation of mass is applied:
      Delta Stockpile = Closing Inventory - Opening Inventory = Additions - Reclaims
      Adjusted Plant Delivery = Plant Received + Delta Stockpile
      F2_adj = Metal(Adjusted Plant) / Metal(Grade Control)
      F3_adj = Metal(Adjusted Plant) / Metal(Reserve)

    Parameters
    ----------
    reserve_data : pd.DataFrame or dict
        Predicted reserve model feed (tonnes, grade).
    plant_data : pd.DataFrame or dict
        Actual received plant/mill feed (weightometer tonnes, assayed head grade).
    grade_control_data : pd.DataFrame or dict, optional
        Short-term grade control / blasthole model (delineated/trucked ore).
    stockpile_data : pd.DataFrame or dict, optional
        Intermediate stockpile inventory or movement data. If None, assumes direct
        mine-to-mill delivery (no stockpiling). Accepts any of 3 standard industry formats:
        1. Opening/closing balances: 'opening_tonnes', 'opening_grade', 'closing_tonnes', 'closing_grade'
        2. Movement flows: 'added_tonnes', 'added_grade', 'reclaimed_tonnes', 'reclaimed_grade'
        3. Direct net deltas: 'delta_tonnes', and ('delta_grade' or 'delta_metal')
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
        factors and component ratios. If stockpile_data is provided, includes both
        unadjusted and stockpile-adjusted factors. Attributes (.attrs) contain cumulative metrics
        and the value chain health diagnosis.
    """

    def _to_df(data: Union[pd.DataFrame, Dict[str, float]]) -> pd.DataFrame:
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return data.copy()

    def _extract_stockpile_delta(sp_row: pd.Series, g_scale: float) -> Tuple[float, float]:
        """Extracts (delta_tonnes, delta_metal) from a stockpile record."""
        # Format 1: Opening & closing inventory balances
        open_t = next((c for c in sp_row.index if c in ("opening_tonnes", "open_tonnes", "opening_t", "open_t")), None)
        close_t = next((c for c in sp_row.index if c in ("closing_tonnes", "close_tonnes", "closing_t", "close_t")), None)
        if open_t is not None and close_t is not None:
            open_g = next((c for c in sp_row.index if c in ("opening_grade", "open_grade", "opening_g", "open_g")), None)
            close_g = next((c for c in sp_row.index if c in ("closing_grade", "close_grade", "closing_g", "close_g")), None)
            if open_g is None or close_g is None:
                raise ValueError("Stockpile opening/closing balances require both tonnes and grade columns.")
            t_o, g_o = float(sp_row[open_t]), float(sp_row[open_g])
            t_c, g_c = float(sp_row[close_t]), float(sp_row[close_g])
            d_t = t_c - t_o
            d_m = (t_c * g_c - t_o * g_o) / g_scale
            return d_t, d_m

        # Format 2: Movement receipts and reclaims
        add_t = next((c for c in sp_row.index if c in ("added_tonnes", "deposit_tonnes", "inflow_tonnes", "feed_tonnes", "add_tonnes")), None)
        rec_t = next((c for c in sp_row.index if c in ("reclaimed_tonnes", "reclaim_tonnes", "drawdown_tonnes", "outflow_tonnes")), None)
        if add_t is not None and rec_t is not None:
            add_g = next((c for c in sp_row.index if c in ("added_grade", "deposit_grade", "inflow_grade", "feed_grade", "add_grade")), None)
            rec_g = next((c for c in sp_row.index if c in ("reclaimed_grade", "reclaim_grade", "drawdown_grade", "outflow_grade")), None)
            if add_g is None or rec_g is None:
                raise ValueError("Stockpile movement flows require both added/reclaimed tonnes and grade columns.")
            t_a, g_a = float(sp_row[add_t]), float(sp_row[add_g])
            t_r, g_r = float(sp_row[rec_t]), float(sp_row[rec_g])
            d_t = t_a - t_r
            d_m = (t_a * g_a - t_r * g_r) / g_scale
            return d_t, d_m

        # Format 3: Direct deltas
        d_t_col = next((c for c in sp_row.index if c in ("delta_tonnes", "delta_t", "stockpile_delta_tonnes", "tonnes")), None)
        if d_t_col is not None:
            d_m_col = next((c for c in sp_row.index if c in ("delta_metal", "delta_m", "stockpile_delta_metal", "metal")), None)
            if d_m_col is not None:
                return float(sp_row[d_t_col]), float(sp_row[d_m_col])
            d_g_col = next((c for c in sp_row.index if c in ("delta_grade", "delta_g", "stockpile_delta_grade", "grade")), None)
            if d_g_col is not None:
                d_t = float(sp_row[d_t_col])
                d_g = float(sp_row[d_g_col])
                return d_t, d_t * (d_g / g_scale)

        raise ValueError(
            "stockpile_data must contain either opening/closing balances ('opening_tonnes', 'opening_grade', 'closing_tonnes', 'closing_grade'), "
            "movement flows ('added_tonnes', 'added_grade', 'reclaimed_tonnes', 'reclaimed_grade'), "
            "or net deltas ('delta_tonnes', 'delta_grade' / 'delta_metal')."
        )

    df_res = _to_df(reserve_data)
    df_plant = _to_df(plant_data)
    has_gc = grade_control_data is not None
    df_gc = _to_df(grade_control_data) if has_gc else None
    has_stockpile = stockpile_data is not None
    df_sp = _to_df(stockpile_data) if has_stockpile else None

    # Handle period identifier
    if period_col is None or period_col not in df_res.columns:
        p_col = "period"
        df_res[p_col] = [
            f"P{i+1}" if len(df_res) > 1 else "Total" for i in range(len(df_res))
        ]
        df_plant[p_col] = df_res[p_col].values
        if has_gc:
            df_gc[p_col] = df_res[p_col].values
        if has_stockpile:
            df_sp[p_col] = df_res[p_col].values
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
    if has_stockpile and p_col not in df_sp.columns:
        if len(df_sp) == len(df_res):
            df_sp[p_col] = df_res[p_col].values
        else:
            raise ValueError(f"Period column '{p_col}' not found in stockpile_data and length does not match reserve data.")

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

            # F2: Grade Control to Plant (Direct)
            rec["f2_tonnes_ratio"] = t_plant / t_gc if t_gc > 0 else 1.0
            rec["f2_grade_ratio"] = g_plant / g_gc if g_gc > 0 else 1.0
            rec["f2_metal_factor"] = m_plant / m_gc if m_gc > 0 else 1.0

        # F3: Reserve to Plant (Direct)
        rec["f3_tonnes_ratio"] = t_plant / t_res if t_res > 0 else 1.0
        rec["f3_grade_ratio"] = g_plant / g_res if g_res > 0 else 1.0
        rec["f3_metal_factor"] = m_plant / m_res if m_res > 0 else 1.0

        # Stockpile-adjusted reconciliation
        if has_stockpile:
            sp_match = df_sp[df_sp[p_col] == p]
            if sp_match.empty:
                raise ValueError(f"Period '{p}' not found in stockpile_data.")
            d_t, d_m = _extract_stockpile_delta(sp_match.iloc[0], grade_scale)
            d_g = (d_m / d_t) * grade_scale if d_t != 0 else 0.0

            t_plant_adj = t_plant + d_t
            m_plant_adj = m_plant + d_m
            g_plant_adj = (m_plant_adj / t_plant_adj) * grade_scale if t_plant_adj > 0 else 0.0

            rec["stockpile_delta_tonnes"] = d_t
            rec["stockpile_delta_grade"] = d_g
            rec["stockpile_delta_metal"] = d_m
            rec["plant_adj_tonnes"] = t_plant_adj
            rec["plant_adj_grade"] = g_plant_adj
            rec["plant_adj_metal"] = m_plant_adj

            if has_gc:
                rec["f2_adj_tonnes_ratio"] = t_plant_adj / t_gc if t_gc > 0 else 1.0
                rec["f2_adj_grade_ratio"] = g_plant_adj / g_gc if g_gc > 0 else 1.0
                rec["f2_adj_metal_factor"] = m_plant_adj / m_gc if m_gc > 0 else 1.0

            rec["f3_adj_tonnes_ratio"] = t_plant_adj / t_res if t_res > 0 else 1.0
            rec["f3_adj_grade_ratio"] = g_plant_adj / g_res if g_res > 0 else 1.0
            rec["f3_adj_metal_factor"] = m_plant_adj / m_res if m_res > 0 else 1.0

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

        if has_stockpile:
            tot_d_t = float(res_df["stockpile_delta_tonnes"].sum())
            tot_d_m = float(res_df["stockpile_delta_metal"].sum())
            tot_d_g = (tot_d_m / tot_d_t) * grade_scale if tot_d_t != 0 else 0.0
            tot_t_plant_adj = tot_t_plant + tot_d_t
            tot_m_plant_adj = tot_m_plant + tot_d_m
            tot_g_plant_adj = (tot_m_plant_adj / tot_t_plant_adj) * grade_scale if tot_t_plant_adj > 0 else 0.0

            tot_rec["stockpile_delta_tonnes"] = tot_d_t
            tot_rec["stockpile_delta_grade"] = tot_d_g
            tot_rec["stockpile_delta_metal"] = tot_d_m
            tot_rec["plant_adj_tonnes"] = tot_t_plant_adj
            tot_rec["plant_adj_grade"] = tot_g_plant_adj
            tot_rec["plant_adj_metal"] = tot_m_plant_adj

            if has_gc:
                tot_rec["f2_adj_tonnes_ratio"] = tot_t_plant_adj / tot_t_gc if tot_t_gc > 0 else 1.0
                tot_rec["f2_adj_grade_ratio"] = tot_g_plant_adj / tot_g_gc if tot_g_gc > 0 else 1.0
                tot_rec["f2_adj_metal_factor"] = tot_m_plant_adj / tot_m_gc if tot_m_gc > 0 else 1.0

            tot_rec["f3_adj_tonnes_ratio"] = tot_t_plant_adj / tot_t_res if tot_t_res > 0 else 1.0
            tot_rec["f3_adj_grade_ratio"] = tot_g_plant_adj / tot_g_res if tot_g_res > 0 else 1.0
            tot_rec["f3_adj_metal_factor"] = tot_m_plant_adj / tot_m_res if tot_m_res > 0 else 1.0

        res_df = pd.concat([res_df, pd.DataFrame([tot_rec])], ignore_index=True)

    # Attach summary attributes
    final_row = res_df.iloc[-1]
    f3_tot = float(final_row["f3_metal_factor"])
    f1_tot = float(final_row["f1_metal_factor"]) if has_gc else None
    f2_tot = float(final_row["f2_metal_factor"]) if has_gc else None

    if has_stockpile:
        res_df.attrs["stockpile_mode"] = "Inventory adjusted"
        f3_adj_tot = float(final_row["f3_adj_metal_factor"])
        res_df.attrs["f3_factor"] = f3_adj_tot
        res_df.attrs["f3_adjusted_factor"] = f3_adj_tot
        res_df.attrs["f3_unadjusted_factor"] = f3_tot
        if has_gc:
            f2_adj_tot = float(final_row["f2_adj_metal_factor"])
            res_df.attrs["f2_factor"] = f2_adj_tot
            res_df.attrs["f2_adjusted_factor"] = f2_adj_tot
            res_df.attrs["f2_unadjusted_factor"] = f2_tot
            res_df.attrs["f1_factor"] = f1_tot

        eval_f3 = f3_adj_tot
        suffix = " (stockpile-adjusted)"
    else:
        res_df.attrs["stockpile_mode"] = "Direct feed (no intermediate stockpiling)"
        res_df.attrs["f3_factor"] = f3_tot
        if has_gc:
            res_df.attrs["f1_factor"] = f1_tot
            res_df.attrs["f2_factor"] = f2_tot
        eval_f3 = f3_tot
        suffix = ""

    if 0.95 <= eval_f3 <= 1.05:
        health_status = f"EXCELLENT: Production is within +/-5% of reserve model (Bankable benchmark{suffix})."
    elif 0.90 <= eval_f3 <= 1.10:
        health_status = f"GOOD: Production is within +/-10% of reserve model{suffix}."
    elif eval_f3 < 0.90:
        health_status = f"WARNING: Metal under-performance (>10% deficit vs. reserve model{suffix}). Check dilution or over-smoothing."
    else:
        health_status = f"WARNING: Metal over-performance (>10% surplus vs. reserve model{suffix}). Check conservative bias or unmodeled ore."

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
    1. Ore Tonnage Comparison (Reserve vs. Grade Control vs. Plant Direct vs. Stockpile Adjusted)
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
    has_sp = "plant_adj_tonnes" in plot_df.columns
    periods = plot_df["period"].astype(str).tolist()
    n_p = len(periods)
    x = np.arange(n_p)

    # Calculate bar width based on number of series
    num_series = 1 + (1 if has_gc else 0) + 1 + (1 if has_sp else 0)
    width = 0.80 / num_series

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_t, ax_g = axes[0, 0], axes[0, 1]
    ax_m, ax_f = axes[1, 0], axes[1, 1]

    col_res = "#1f77b4"      # Blue (Reserve Model)
    col_gc = "#ff7f0e"       # Orange (Grade Control)
    col_plant = "#2ca02c"    # Green (Plant Direct Feed)
    col_plant_adj = "#9467bd"# Purple (Plant + Stockpile Delta)

    # Offsets helper
    offsets = np.linspace(-width * (num_series - 1) / 2, width * (num_series - 1) / 2, num_series)

    # -------------------------------------------------------------------------
    # Panel 1: Ore Tonnage
    # -------------------------------------------------------------------------
    s_idx = 0
    ax_t.bar(
        x + offsets[s_idx], plot_df["reserve_tonnes"], width,
        label="Reserve Model", color=col_res, edgecolor="black", alpha=0.85
    )
    s_idx += 1
    if has_gc:
        ax_t.bar(
            x + offsets[s_idx], plot_df["gc_tonnes"], width,
            label="Grade Control", color=col_gc, edgecolor="black", alpha=0.85
        )
        s_idx += 1
    ax_t.bar(
        x + offsets[s_idx], plot_df["plant_tonnes"], width,
        label="Plant (Direct)", color=col_plant, edgecolor="black", alpha=0.85
    )
    s_idx += 1
    if has_sp:
        ax_t.bar(
            x + offsets[s_idx], plot_df["plant_adj_tonnes"], width,
            label="Plant (Stockpile-Adj)", color=col_plant_adj, edgecolor="black", alpha=0.85
        )

    ax_t.set_ylabel(f"Ore Tonnage ({tonnage_unit})", fontsize=10, fontweight="bold")
    ax_t.set_title("Ore Tonnage Reconciliation", fontsize=11, fontweight="bold")
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(periods, fontsize=9)
    ax_t.grid(True, linestyle=":", alpha=0.5)
    ax_t.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 2: Head Grade
    # -------------------------------------------------------------------------
    s_idx = 0
    ax_g.bar(
        x + offsets[s_idx], plot_df["reserve_grade"], width,
        label="Reserve Model", color=col_res, edgecolor="black", alpha=0.85
    )
    s_idx += 1
    if has_gc:
        ax_g.bar(
            x + offsets[s_idx], plot_df["gc_grade"], width,
            label="Grade Control", color=col_gc, edgecolor="black", alpha=0.85
        )
        s_idx += 1
    ax_g.bar(
        x + offsets[s_idx], plot_df["plant_grade"], width,
        label="Plant (Direct)", color=col_plant, edgecolor="black", alpha=0.85
    )
    s_idx += 1
    if has_sp:
        ax_g.bar(
            x + offsets[s_idx], plot_df["plant_adj_grade"], width,
            label="Plant (Stockpile-Adj)", color=col_plant_adj, edgecolor="black", alpha=0.85
        )

    ax_g.set_ylabel(f"Head Grade ({grade_unit})", fontsize=10, fontweight="bold")
    ax_g.set_title("Head Grade Reconciliation", fontsize=11, fontweight="bold")
    ax_g.set_xticks(x)
    ax_g.set_xticklabels(periods, fontsize=9)
    ax_g.grid(True, linestyle=":", alpha=0.5)
    ax_g.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Contained Metal
    # -------------------------------------------------------------------------
    s_idx = 0
    ax_m.bar(
        x + offsets[s_idx], plot_df["reserve_metal"], width,
        label="Reserve Model", color=col_res, edgecolor="black", alpha=0.85
    )
    s_idx += 1
    if has_gc:
        ax_m.bar(
            x + offsets[s_idx], plot_df["gc_metal"], width,
            label="Grade Control", color=col_gc, edgecolor="black", alpha=0.85
        )
        s_idx += 1
    ax_m.bar(
        x + offsets[s_idx], plot_df["plant_metal"], width,
        label="Plant (Direct)", color=col_plant, edgecolor="black", alpha=0.85
    )
    s_idx += 1
    if has_sp:
        ax_m.bar(
            x + offsets[s_idx], plot_df["plant_adj_metal"], width,
            label="Plant (Stockpile-Adj)", color=col_plant_adj, edgecolor="black", alpha=0.85
        )

    ax_m.set_ylabel(f"Contained Metal ({metal_unit})", fontsize=10, fontweight="bold")
    ax_m.set_title("Contained Metal Reconciliation", fontsize=11, fontweight="bold")
    ax_m.set_xticks(x)
    ax_m.set_xticklabels(periods, fontsize=9)
    ax_m.grid(True, linestyle=":", alpha=0.5)
    ax_m.legend(loc="upper right", fontsize=8.0, framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Harry Parker F1, F2, F3 Factors Tracking
    # -------------------------------------------------------------------------
    ax_f.axhspan(0.95, 1.05, color="#2ca02c", alpha=0.15, label="Target Band (±5%)")
    ax_f.axhline(1.00, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    if n_p > 1:
        if has_gc:
            ax_f.plot(
                x, plot_df["f1_metal_factor"], "-o",
                color=col_res, linewidth=2.0, markersize=6, label="F1 (Model → Mine)"
            )
            ax_f.plot(
                x, plot_df["f2_metal_factor"], "--s" if has_sp else "-s",
                color=col_gc, linewidth=1.8 if has_sp else 2.0, markersize=5 if has_sp else 6,
                label="F2 (Direct Mine → Mill)" if has_sp else "F2 (Mine → Mill)"
            )
            if has_sp:
                ax_f.plot(
                    x, plot_df["f2_adj_metal_factor"], "-s",
                    color="#d62728", linewidth=2.2, markersize=6, label="F2 (Stockpile-Adj)"
                )
        ax_f.plot(
            x, plot_df["f3_metal_factor"], "--^" if has_sp else "-^",
            color=col_plant, linewidth=1.8 if has_sp else 2.5, markersize=6 if has_sp else 7,
            label="F3 (Direct Total)" if has_sp else "F3 (Total Value Chain)"
        )
        if has_sp:
            ax_f.plot(
                x, plot_df["f3_adj_metal_factor"], "-^",
                color=col_plant_adj, linewidth=2.5, markersize=7, label="F3 (Stockpile-Adj Total)"
            )
        ax_f.set_xticks(x)
        ax_f.set_xticklabels(periods, fontsize=9)
    else:
        # Single period bar representation
        cats = ["F1 (Model→Mine)"] if has_gc else []
        vals = [float(plot_df["f1_metal_factor"].iloc[0])] if has_gc else []
        bar_c = [col_res] if has_gc else []

        if has_gc:
            cats.append("F2 (Direct)")
            vals.append(float(plot_df["f2_metal_factor"].iloc[0]))
            bar_c.append(col_gc)
            if has_sp:
                cats.append("F2 (Adj)")
                vals.append(float(plot_df["f2_adj_metal_factor"].iloc[0]))
                bar_c.append("#d62728")

        cats.append("F3 (Direct)")
        vals.append(float(plot_df["f3_metal_factor"].iloc[0]))
        bar_c.append(col_plant)
        if has_sp:
            cats.append("F3 (Adj)")
            vals.append(float(plot_df["f3_adj_metal_factor"].iloc[0]))
            bar_c.append(col_plant_adj)

        ax_f.bar(
            range(len(cats)), vals, width=0.45,
            color=bar_c, edgecolor="black", alpha=0.85
        )
        ax_f.set_xticks(range(len(cats)))
        ax_f.set_xticklabels(cats, fontsize=8.5, fontweight="bold")
        for idx, v in enumerate(vals):
            ax_f.text(idx, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")

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


# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Current status: 2D conditioning data and 2D grid nodes.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Respect Domain Boundaries: conditional simulation must be executed independently per domain.
# - Explicit Extrapolation Control: confine simulation paths and conditioning to data envelopes.
# - 3D Anisotropic Covariance: support 3D variogram structures (major, semi-major, minor ranges).
# - Functional approach: transparent NumPy arrays/DataFrames without dataclass configuration wrappers.
# - Following codebase conventions, 3D visualizations are implemented as separate functions:
#   1. `plot_simulation_3d_isometric`: dedicated 3D isometric view of simulation realizations / exceedance.
#   2. `plot_simulation_3d_interactive`: dedicated interactive 3D realization explorer.
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

    Sequential Gaussian Simulation (Isaaks 1990; Deutsch & Journel 1998; SME Handbook) (TODO: Manually Verify)
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


# =============================================================================
# 3D BLOCK MODEL VISUALIZATION SUITE (SLICES, GALLERIES, ISOMETRIC, UNCERTAINTY)
# =============================================================================
# TODO (3D Block Model Extension - CONTRIBUTING.md Standards):
# Maintain clean separation of concerns by adding dedicated functions to this suite:
# - `plot_resource_classification_3d_isometric`: Static 3D isometric projection of CIM/JORC resource categories (TODO: Manually Verify).
# - `plot_resource_classification_3d_interactive`: Interactive 3D explorer with category toggles, bench slider, and HUD.
# - `plot_resource_classification_bench_gallery`: Multi-panel elevation gallery of classification slices.
# - `plot_resource_classification_orthogonal_slices`: 3-view orthogonal slices (XY plan, XZ cross, YZ long).
# - `plot_reserve_classification_3d_isometric` & `plot_reserve_classification_3d_interactive`: Reserve status 3D viewers.
# - `plot_polygonal_3d_isometric` & `plot_polygonal_3d_interactive`: 3D polygonal/NN domain viewers.
# - `plot_simulation_3d_isometric` & `plot_simulation_3d_interactive`: 3D conditional simulation realization viewers.


def _get_block_grade_norm_and_cmap(
    block_grades: np.ndarray,
    sample_grades: Optional[np.ndarray] = None,
    grade_bins: Optional[Sequence[float]] = None,
    cmap_name: str = "viridis",
) -> Tuple[mcolors.Normalize, Any]:
    """Ensures identical colormap and normalization between blocks and drillholes."""
    all_vals = []
    valid_blk = block_grades[np.isfinite(block_grades)]
    if len(valid_blk) > 0:
        all_vals.append(valid_blk)
    if sample_grades is not None:
        valid_smp = sample_grades[np.isfinite(sample_grades)]
        if len(valid_smp) > 0:
            all_vals.append(valid_smp)

    combined = np.concatenate(all_vals) if all_vals else np.array([0.0, 1.0])
    vmin = float(np.min(combined))
    vmax = float(np.max(combined))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#e0e0e0")

    if grade_bins is not None:
        bins = np.sort(np.asarray(grade_bins, dtype=float))
        norm = mcolors.BoundaryNorm(bins, ncolors=cmap.N, extend="both")
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    return norm, cmap


def _render_2d_block_patches(
    ax: plt.Axes,
    df_slice: pd.DataFrame,
    u_col: str,
    v_col: str,
    du_col: str,
    dv_col: str,
    val_col: str,
    norm: mcolors.Normalize,
    cmap: Any,
    edgecolor: str = "none",
    linewidth: float = 0.2,
) -> PatchCollection:
    """Renders 2D block model slices using vector PatchCollection at true support coordinates."""
    patches = []
    vals = df_slice[val_col].to_numpy()
    u = df_slice[u_col].to_numpy()
    v = df_slice[v_col].to_numpy()
    du = df_slice[du_col].to_numpy()
    dv = df_slice[dv_col].to_numpy()

    for i in range(len(df_slice)):
        rect = MplRectangle(
            (u[i] - 0.5 * du[i], v[i] - 0.5 * dv[i]),
            du[i],
            dv[i],
        )
        patches.append(rect)

    pc = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )
    pc.set_array(vals)
    ax.add_collection(pc)
    ax.autoscale_view()
    return pc


def plot_block_model_orthogonal_slices(
    block_model: pd.DataFrame,
    grade_col: str = "estimated_grade",
    bench_z: Optional[float] = None,
    section_y: Optional[float] = None,
    section_x: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    sample_grades: Optional[np.ndarray] = None,
    grade_bins: Optional[Sequence[float]] = None,
    cross_section_view: str = "north",
    long_section_view: str = "west",
    cmap: str = "viridis",
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (19.0, 5.8),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots 3-panel orthogonal slices (Bench Plan, Cross-Section, Longitudinal Section).

    Standard reporting practice in NI 43-101 and JORC technical reports (TODO: Manually Verify):
    1. Bench Plan (X-Y at elevation Z = bench_z): Shows horizontal grade continuity looking down.
    2. Cross-Section (X-Z at northing Y = section_y): Shows vertical depth continuity across strike.
       - Looking North (default): West is left, East is right.
       - Looking South: East is left, West is right (X inverted).
    3. Longitudinal Section (Y-Z at easting X = section_x): Shows plunge along strike.
       - Looking West (default): South is left, North is right.
       - Looking East: North is left, South is right (Y inverted).

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z), dimensions (dx, dy, dz),
        and the estimated grade column.
    grade_col : str, default "estimated_grade"
        Grade column to plot.
    bench_z : float, optional
        Target elevation for horizontal bench slice. If None, uses median block elevation.
    section_y : float, optional
        Target northing for cross-section slice. If None, uses median block northing.
    section_x : float, optional
        Target easting for longitudinal slice. If None, uses median block easting.
    samples_xyz : np.ndarray, optional
        Exploration composite coordinates of shape (N, 3) for overlay.
    sample_grades : np.ndarray, optional
        Composite grades of shape (N,) for overlay.
    grade_bins : Sequence[float], optional
        Discrete cutoff thresholds for discrete grade intervals.
    cross_section_view : str, default "north"
        Viewing direction for cross-section ("north" or "south").
    long_section_view : str, default "west"
        Viewing direction for longitudinal section ("west" or "east").
    cmap : str, default "viridis"
        Colormap name.
    grade_unit : str, default "% Cu"
        Unit label for grade colorbar.
    title : str, optional
        Figure title.
    figsize : tuple of float, default (19.0, 5.8)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Matplotlib figure and sequence of 3 axes.
    """
    cross_section_view = cross_section_view.lower()
    if cross_section_view not in ["north", "south"]:
        raise ValueError("cross_section_view must be 'north' or 'south'.")
    long_section_view = long_section_view.lower()
    if long_section_view not in ["west", "east"]:
        raise ValueError("long_section_view must be 'west' or 'east'.")

    for req in ["x", "y", "z", "dx", "dy", "dz"]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")
    if grade_col not in block_model.columns:
        raise ValueError(f"block_model is missing grade column '{grade_col}'.")

    # Pick exact centroid coordinate slices
    unique_z = np.sort(block_model["z"].unique())
    unique_y = np.sort(block_model["y"].unique())
    unique_x = np.sort(block_model["x"].unique())

    target_z = (
        unique_z[np.argmin(np.abs(unique_z - bench_z))]
        if bench_z is not None
        else unique_z[len(unique_z) // 2]
    )
    target_y = (
        unique_y[np.argmin(np.abs(unique_y - section_y))]
        if section_y is not None
        else unique_y[len(unique_y) // 2]
    )
    target_x = (
        unique_x[np.argmin(np.abs(unique_x - section_x))]
        if section_x is not None
        else unique_x[len(unique_x) // 2]
    )

    # Shared normalization and colormap ensuring IDENTICAL scaling between blocks and drillholes
    norm, color_map = _get_block_grade_norm_and_cmap(
        block_grades=block_model[grade_col].to_numpy(),
        sample_grades=sample_grades,
        grade_bins=grade_bins,
        cmap_name=cmap,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # 1. Bench Plan (X-Y at target_z)
    slice_bench = block_model[np.isclose(block_model["z"], target_z)]
    dz_val = float(slice_bench["dz"].iloc[0]) if len(slice_bench) > 0 else 5.0
    pc1 = _render_2d_block_patches(
        axes[0], slice_bench, "x", "y", "dx", "dy", grade_col, norm, color_map
    )
    axes[0].set_title(
        f"Bench Plan (Z = {target_z:.1f} m) [Looking Down]",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_xlabel("Easting (X) [m]")
    axes[0].set_ylabel("Northing (Y) [m]")
    axes[0].set_aspect("equal")

    # 2. Cross-Section (X-Z at target_y)
    slice_sec = block_model[np.isclose(block_model["y"], target_y)]
    dy_val = float(slice_sec["dy"].iloc[0]) if len(slice_sec) > 0 else 10.0
    _render_2d_block_patches(
        axes[1], slice_sec, "x", "z", "dx", "dz", grade_col, norm, color_map
    )
    view_cs_label = (
        "Looking North" if cross_section_view == "north" else "Looking South"
    )
    x_dir_label = (
        "Easting (X) [m] (← West | East →)"
        if cross_section_view == "north"
        else "Easting (X) [m] (← East | West →)"
    )
    axes[1].set_title(
        f"Cross-Section (Y = {target_y:.1f} m) [{view_cs_label}]",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_xlabel(x_dir_label)
    axes[1].set_ylabel("Elevation (Z) [m]")
    axes[1].set_aspect("equal")
    if cross_section_view == "south":
        axes[1].invert_xaxis()

    # 3. Longitudinal Section (Y-Z at target_x)
    slice_long = block_model[np.isclose(block_model["x"], target_x)]
    dx_val = float(slice_long["dx"].iloc[0]) if len(slice_long) > 0 else 10.0
    _render_2d_block_patches(
        axes[2], slice_long, "y", "z", "dy", "dz", grade_col, norm, color_map
    )
    view_ls_label = "Looking West" if long_section_view == "west" else "Looking East"
    y_dir_label = (
        "Northing (Y) [m] (← South | North →)"
        if long_section_view == "west"
        else "Northing (Y) [m] (← North | South →)"
    )
    axes[2].set_title(
        f"Longitudinal Section (X = {target_x:.1f} m) [{view_ls_label}]",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].set_xlabel(y_dir_label)
    axes[2].set_ylabel("Elevation (Z) [m]")
    axes[2].set_aspect("equal")
    if long_section_view == "east":
        axes[2].invert_xaxis()

    # Drillhole composite overlay within corridor with EXACT MATCHING colormap and norm
    if samples_xyz is not None and sample_grades is not None:
        # Bench corridor
        mask_bench = np.abs(samples_xyz[:, 2] - target_z) <= 0.5 * dz_val
        if np.any(mask_bench):
            axes[0].scatter(
                samples_xyz[mask_bench, 0],
                samples_xyz[mask_bench, 1],
                c=sample_grades[mask_bench],
                cmap=color_map,
                norm=norm,
                edgecolor="black",
                linewidth=0.8,
                s=35,
                zorder=4,
                label=f"Composites (±{0.5 * dz_val:.1f}m)",
            )
            axes[0].legend(loc="upper right", framealpha=0.8, fontsize=8)

        # Cross-section corridor
        mask_sec = np.abs(samples_xyz[:, 1] - target_y) <= 0.5 * dy_val
        if np.any(mask_sec):
            axes[1].scatter(
                samples_xyz[mask_sec, 0],
                samples_xyz[mask_sec, 2],
                c=sample_grades[mask_sec],
                cmap=color_map,
                norm=norm,
                edgecolor="black",
                linewidth=0.8,
                s=35,
                zorder=4,
                label=f"Composites (±{0.5 * dy_val:.1f}m)",
            )
            axes[1].legend(loc="upper right", framealpha=0.8, fontsize=8)

        # Longitudinal section corridor
        mask_long = np.abs(samples_xyz[:, 0] - target_x) <= 0.5 * dx_val
        if np.any(mask_long):
            axes[2].scatter(
                samples_xyz[mask_long, 1],
                samples_xyz[mask_long, 2],
                c=sample_grades[mask_long],
                cmap=color_map,
                norm=norm,
                edgecolor="black",
                linewidth=0.8,
                s=35,
                zorder=4,
                label=f"Composites (±{0.5 * dx_val:.1f}m)",
            )
            axes[2].legend(loc="upper right", framealpha=0.8, fontsize=8)

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)

    cbar = fig.colorbar(pc1, ax=axes, orientation="vertical", shrink=0.85, pad=0.015)
    cbar.set_label(f"Block Grade ({grade_unit})", fontsize=10, fontweight="bold")

    fig_title = title or "3D Block Model Orthogonal Slices & Drillhole Reconciliation"
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")
    return fig, axes


def plot_block_model_bench_gallery(
    block_model: pd.DataFrame,
    grade_col: str = "estimated_grade",
    bench_elevations: Optional[Sequence[float]] = None,
    n_cols: int = 3,
    samples_xyz: Optional[np.ndarray] = None,
    sample_grades: Optional[np.ndarray] = None,
    grade_bins: Optional[Sequence[float]] = None,
    cmap: str = "viridis",
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots a multi-bench elevation gallery showing deposit morphology from surface to depth."""
    for req in ["x", "y", "z", "dx", "dy", "dz", grade_col]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")

    unique_z = np.sort(block_model["z"].unique())
    if bench_elevations is None:
        if len(unique_z) <= 6:
            chosen_z = unique_z
        else:
            indices = np.linspace(0, len(unique_z) - 1, 6, dtype=int)
            chosen_z = unique_z[indices]
    else:
        chosen_z = [
            unique_z[np.argmin(np.abs(unique_z - bz))] for bz in bench_elevations
        ]

    n_plots = len(chosen_z)
    n_cols = max(1, min(n_cols, n_plots))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig_size = figsize or (5.5 * n_cols + 1.5, 4.8 * n_rows)
    fig, axes_flat = plt.subplots(
        n_rows, n_cols, figsize=fig_size, constrained_layout=True, squeeze=False
    )
    axes = axes_flat.ravel()

    norm, color_map = _get_block_grade_norm_and_cmap(
        block_grades=block_model[grade_col].to_numpy(),
        sample_grades=sample_grades,
        grade_bins=grade_bins,
        cmap_name=cmap,
    )

    last_pc = None
    for i, bz in enumerate(chosen_z):
        ax = axes[i]
        df_bench = block_model[np.isclose(block_model["z"], bz)]
        dz_val = float(df_bench["dz"].iloc[0]) if len(df_bench) > 0 else 5.0
        last_pc = _render_2d_block_patches(
            ax, df_bench, "x", "y", "dx", "dy", grade_col, norm, color_map
        )
        ax.set_title(f"Level Z = {bz:.1f} m", fontsize=10, fontweight="bold")
        ax.set_xlabel("Easting (X) [m]")
        ax.set_ylabel("Northing (Y) [m]")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)

        if samples_xyz is not None and sample_grades is not None:
            mask_b = np.abs(samples_xyz[:, 2] - bz) <= 0.5 * dz_val
            if np.any(mask_b):
                ax.scatter(
                    samples_xyz[mask_b, 0],
                    samples_xyz[mask_b, 1],
                    c=sample_grades[mask_b],
                    cmap=color_map,
                    norm=norm,
                    edgecolor="black",
                    linewidth=0.8,
                    s=30,
                    zorder=4,
                )

    # Hide unused subplots
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    if last_pc is not None:
        cbar = fig.colorbar(
            last_pc, ax=axes[:n_plots], orientation="vertical", shrink=0.85, pad=0.015
        )
        cbar.set_label(f"Block Grade ({grade_unit})", fontsize=10, fontweight="bold")

    fig_title = title or "3D Block Model Bench Depth Gallery (Elevation Slices)"
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")
    return fig, axes[:n_plots]


def plot_block_model_3d_isometric(
    block_model: pd.DataFrame,
    grade_col: str = "estimated_grade",
    cutoff_grade: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    sample_grades: Optional[np.ndarray] = None,
    grade_bins: Optional[Sequence[float]] = None,
    cmap: str = "viridis",
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    elev: float = 28.0,
    azim: float = -55.0,
    figsize: Tuple[float, float] = (10.0, 8.5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plots 3D Isometric View with cut-off filtering to reveal internal ore shoots."""
    for req in ["x", "y", "z", grade_col]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")

    # Filter by cutoff grade to avoid waste block occlusion
    valid_mask = np.isfinite(block_model[grade_col])
    if cutoff_grade is not None:
        valid_mask &= block_model[grade_col] >= cutoff_grade
    df_plot = block_model[valid_mask]

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    norm, color_map = _get_block_grade_norm_and_cmap(
        block_grades=block_model[grade_col].to_numpy(),
        sample_grades=sample_grades,
        grade_bins=grade_bins,
        cmap_name=cmap,
    )

    if len(df_plot) > 0:
        p = ax.scatter(
            df_plot["x"],
            df_plot["y"],
            df_plot["z"],
            c=df_plot[grade_col],
            cmap=color_map,
            norm=norm,
            marker="s",
            s=45,
            alpha=0.75,
            edgecolors="none",
            label=(
                f"Ore Blocks (≥ {cutoff_grade:.2f} {grade_unit})"
                if cutoff_grade
                else "Blocks"
            ),
        )
        cbar = fig.colorbar(p, ax=ax, shrink=0.7, pad=0.08)
        cbar.set_label(f"Grade ({grade_unit})", fontsize=10, fontweight="bold")

    if samples_xyz is not None:
        c_samples = sample_grades if sample_grades is not None else "red"
        ax.scatter(
            samples_xyz[:, 0],
            samples_xyz[:, 1],
            samples_xyz[:, 2],
            c=c_samples,
            cmap=color_map if sample_grades is not None else None,
            norm=norm if sample_grades is not None else None,
            s=20,
            edgecolor="black",
            linewidth=0.6,
            label="Drillholes",
            zorder=5,
        )

    ax.set_xlabel("Easting (X) [m]", labelpad=8)
    ax.set_ylabel("Northing (Y) [m]", labelpad=8)
    ax.set_zlabel("Elevation (Z) [m]", labelpad=8)
    ax.view_init(elev=elev, azim=azim)

    # Set physical aspect ratio scaling (1:1:1 physical dimensions)
    x_min, x_max = float(block_model["x"].min()), float(block_model["x"].max())
    y_min, y_max = float(block_model["y"].min()), float(block_model["y"].max())
    z_min, z_max = float(block_model["z"].min()), float(block_model["z"].max())
    span_x = max(1.0, x_max - x_min)
    span_y = max(1.0, y_max - y_min)
    span_z = max(1.0, z_max - z_min)
    ax.set_box_aspect((span_x, span_y, span_z))

    cutoff_str = (
        f" [Cut-off ≥ {cutoff_grade:.2f} {grade_unit}]"
        if cutoff_grade is not None
        else ""
    )
    fig_title = title or f"3D Block Model Isometric View{cutoff_str}"
    ax.set_title(fig_title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.8)

    return fig, ax


def plot_block_model_3d_interactive(
    block_model: pd.DataFrame,
    grade_col: str = "estimated_grade",
    initial_cutoff: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    sample_grades: Optional[np.ndarray] = None,
    grade_bins: Optional[Sequence[float]] = None,
    tonnes_col: Optional[str] = "tonnes",
    cmap: str = "viridis",
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    elev: float = 28.0,
    azim: float = -55.0,
    figsize: Tuple[float, float] = (11.0, 9.0),
) -> Tuple[plt.Figure, Axes3D, Dict[str, Any]]:
    """Interactive 3D Block Model explorer with real-time sliders and metrics.

    Provides a live desktop exploration interface with:
    - 3D physical 1:1:1 aspect scaling, click-and-drag 360° rotation, and scroll zoom.
    - Interactive Cut-off Grade Slider to dynamically peel away waste blocks.
    - Interactive Max Elevation Slider to slice downward through benches / strip overburden.
    - Dynamic mining reconciliation HUD showing visible ore count, mean grade, and tonnage.
    - Reset View button to restore initial parameters.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and estimated grade.
    grade_col : str, default "estimated_grade"
        Grade column to visualize.
    initial_cutoff : float, optional
        Starting cut-off grade for the slider. Defaults to 25th percentile of ore grades.
    samples_xyz : np.ndarray, optional
        Exploration drillhole coordinates of shape (N, 3).
    sample_grades : np.ndarray, optional
        Drillhole composite grades for overlay.
    grade_bins : Sequence[float], optional
        Discrete cutoff thresholds for discrete color intervals.
    tonnes_col : str, optional, default "tonnes"
        Tonnage column for real-time mass calculations.
    cmap : str, default "viridis"
        Colormap name.
    grade_unit : str, default "% Cu"
        Unit label for grades and colorbars.
    title : str, optional
        Window and plot title.
    elev : float, default 28.0
        Initial camera elevation angle in degrees.
    azim : float, default -55.0
        Initial camera azimuth angle in degrees.
    figsize : tuple of float, default (11.0, 9.0)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Axes3D, Dict[str, Any]]
        Matplotlib Figure, 3D Axes, and dictionary containing references to interactive
        controls ('slider_cutoff', 'slider_elev', 'button_reset', 'update_func') to prevent
        garbage collection of widget callbacks.
    """
    for req in ["x", "y", "z", grade_col]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")

    x_all = block_model["x"].to_numpy()
    y_all = block_model["y"].to_numpy()
    z_all = block_model["z"].to_numpy()
    grades_all = block_model[grade_col].to_numpy()
    tonnes_all = (
        block_model[tonnes_col].to_numpy()
        if (tonnes_col and tonnes_col in block_model.columns)
        else None
    )
    valid_finite = np.isfinite(grades_all)

    valid_grades = grades_all[valid_finite]
    min_g = float(np.min(valid_grades)) if len(valid_grades) > 0 else 0.0
    max_g = float(np.max(valid_grades)) if len(valid_grades) > 0 else 1.0
    if initial_cutoff is None:
        init_c = (
            float(np.percentile(valid_grades, 25)) if len(valid_grades) > 0 else min_g
        )
    else:
        init_c = float(initial_cutoff)

    min_z = float(np.min(z_all))
    max_z = float(np.max(z_all))

    norm, color_map = _get_block_grade_norm_and_cmap(
        block_grades=valid_grades,
        sample_grades=sample_grades,
        grade_bins=grade_bins,
        cmap_name=cmap,
    )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.20, top=0.93, left=0.05, right=0.88)

    init_mask = valid_finite & (grades_all >= init_c) & (z_all <= max_z)
    p = ax.scatter(
        x_all[init_mask],
        y_all[init_mask],
        z_all[init_mask],
        c=grades_all[init_mask],
        cmap=color_map,
        norm=norm,
        marker="s",
        s=45,
        alpha=0.75,
        edgecolors="none",
        label="Ore Blocks",
    )

    if samples_xyz is not None:
        c_dh = sample_grades if sample_grades is not None else "red"
        ax.scatter(
            samples_xyz[:, 0],
            samples_xyz[:, 1],
            samples_xyz[:, 2],
            c=c_dh,
            cmap=color_map if sample_grades is not None else None,
            norm=norm if sample_grades is not None else None,
            s=20,
            edgecolor="black",
            linewidth=0.6,
            zorder=5,
            label="Drillholes",
        )

    # Physical aspect ratio scaling
    span_x = max(1.0, float(np.ptp(x_all)))
    span_y = max(1.0, float(np.ptp(y_all)))
    span_z = max(1.0, float(np.ptp(z_all)))
    ax.set_box_aspect((span_x, span_y, span_z))
    ax.set_xlabel("Easting (X) [m]", labelpad=8)
    ax.set_ylabel("Northing (Y) [m]", labelpad=8)
    ax.set_zlabel("Elevation (Z) [m]", labelpad=8)
    ax.view_init(elev=elev, azim=azim)

    fig_title = title or "Interactive 3D Block Model Exploration"
    ax.set_title(fig_title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="upper left", framealpha=0.8)

    cbar_ax = fig.add_axes([0.90, 0.28, 0.025, 0.58])
    cbar = fig.colorbar(p, cax=cbar_ax)
    cbar.set_label(f"Block Grade ({grade_unit})", fontsize=10, fontweight="bold")

    info_box = ax.text2D(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )

    escaped_unit = grade_unit.replace("%", "%%")
    ax_slider_c = fig.add_axes([0.18, 0.09, 0.52, 0.035])
    slider_cutoff = Slider(
        ax=ax_slider_c,
        label="Cut-off Grade",
        valmin=min_g,
        valmax=max_g,
        valinit=init_c,
        valfmt=f"%.2f {escaped_unit}",
        color="#2b5c8f",
    )

    ax_slider_z = fig.add_axes([0.18, 0.03, 0.52, 0.035])
    slider_elev = Slider(
        ax=ax_slider_z,
        label="Max Elevation",
        valmin=min_z,
        valmax=max_z,
        valinit=max_z,
        valfmt="%.1f m",
        color="#4a7c59",
    )

    ax_btn = fig.add_axes([0.76, 0.03, 0.12, 0.095])
    btn_reset = Button(ax_btn, "Reset\nControls", hovercolor="#e0e0e0")

    def update_plot(val=None):
        c_val = slider_cutoff.val
        z_val = slider_elev.val
        mask = valid_finite & (grades_all >= c_val) & (z_all <= z_val)
        n_vis = int(np.sum(mask))
        n_tot = int(np.sum(valid_finite))
        pct = (n_vis / n_tot * 100.0) if n_tot > 0 else 0.0

        if n_vis > 0:
            p._offsets3d = (x_all[mask], y_all[mask], z_all[mask])
            p.set_array(grades_all[mask])
            mean_g = float(np.mean(grades_all[mask]))
            tonnes_str = ""
            if tonnes_all is not None:
                t_vis = float(np.sum(tonnes_all[mask])) / 1e6
                tonnes_str = f"\nTonnage:    {t_vis:.2f} Mt"
            info_box.set_text(
                f"Visible:    {n_vis:,} / {n_tot:,} ({pct:.1f}%)\n"
                f"Mean Grade: {mean_g:.2f} {grade_unit}"
                f"{tonnes_str}"
            )
        else:
            p._offsets3d = (np.array([]), np.array([]), np.array([]))
            p.set_array(np.array([]))
            info_box.set_text(
                f"Visible: 0 / {n_tot:,} (0.0%)\n[No blocks above cut-off]"
            )

        fig.canvas.draw_idle()

    slider_cutoff.on_changed(update_plot)
    slider_elev.on_changed(update_plot)

    def reset_controls(event):
        slider_cutoff.reset()
        slider_elev.reset()
        ax.view_init(elev=elev, azim=azim)

    btn_reset.on_clicked(reset_controls)
    update_plot()

    controls = {
        "slider_cutoff": slider_cutoff,
        "slider_elev": slider_elev,
        "button_reset": btn_reset,
        "update_func": update_plot,
        "reset_func": reset_controls,
    }
    # Attach to figure to prevent Python garbage collection of callback references
    fig._interactive_controls = controls  # type: ignore[attr-defined]

    return fig, ax, controls


def plot_block_model_grade_uncertainty(
    block_model: pd.DataFrame,
    grade_col: str = "estimated_grade",
    var_col: str = "kriging_variance",
    slice_axis: str = "z",
    slice_coord: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    sample_grades: Optional[np.ndarray] = None,
    grade_bins: Optional[Sequence[float]] = None,
    vmax_var: Optional[float] = None,
    vmin_var: float = 0.0,
    grade_cmap: str = "viridis",
    var_cmap: str = "magma_r",
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (15.0, 6.0),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots side-by-side audit comparing Estimated Grade vs. Kriging Estimation Variance.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates, block dimensions, grade, and variance.
    grade_col : str, default "estimated_grade"
        Column name for estimated block grade.
    var_col : str, default "kriging_variance"
        Column name for block kriging estimation variance.
    slice_axis : str, default "z"
        Axis orthogonal to slice plane ('x', 'y', or 'z').
    slice_coord : float, optional
        Target coordinate along slice_axis. Defaults to median coordinate.
    samples_xyz : np.ndarray, optional
        Sample composite coordinates for pierce-point overlay.
    sample_grades : np.ndarray, optional
        Sample composite grades for pierce-point overlay.
    grade_bins : Sequence[float], optional
        Discrete cutoff bins for grade coloring.
    vmax_var : float, optional
        Upper bound for kriging variance normalization. Setting vmax_var = total_sill (or C(0))
        anchors the variance scale to theoretical maximum uncertainty, preventing misleading
        color stretches across well-informed slices.
    vmin_var : float, default 0.0
        Lower bound for kriging variance normalization (collocated samples have theoretical variance 0).
    grade_cmap : str, default "viridis"
        Colormap name for estimated grade panel.
    var_cmap : str, default "magma_r"
        Colormap name for kriging variance panel.
    grade_unit : str, default "% Cu"
        Unit label for grade colorbar.
    title : str, optional
        Overall figure title.
    figsize : tuple of float, default (15.0, 6.0)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Matplotlib figure and sequence of 2 axes (grade and variance).
    """
    slice_axis = slice_axis.lower()
    if slice_axis not in ["x", "y", "z"]:
        raise ValueError("slice_axis must be 'x', 'y', or 'z'.")
    for req in ["x", "y", "z", "dx", "dy", "dz", grade_col, var_col]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")

    u_col, v_col = [c for c in ["x", "y", "z"] if c != slice_axis]
    du_col, dv_col = f"d{u_col}", f"d{v_col}"

    unique_s = np.sort(block_model[slice_axis].unique())
    target_coord = (
        unique_s[np.argmin(np.abs(unique_s - slice_coord))]
        if slice_coord is not None
        else unique_s[len(unique_s) // 2]
    )
    df_slice = block_model[np.isclose(block_model[slice_axis], target_coord)]
    d_slice = float(df_slice[f"d{slice_axis}"].iloc[0]) if len(df_slice) > 0 else 5.0

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # 1. Grade Panel
    norm_grade, cmap_grade = _get_block_grade_norm_and_cmap(
        block_grades=block_model[grade_col].to_numpy(),
        sample_grades=sample_grades,
        grade_bins=grade_bins,
        cmap_name=grade_cmap,
    )
    pc_grade = _render_2d_block_patches(
        axes[0],
        df_slice,
        u_col,
        v_col,
        du_col,
        dv_col,
        grade_col,
        norm_grade,
        cmap_grade,
    )
    axes[0].set_title(
        f"Estimated Grade Z*(V) [{slice_axis.upper()} = {target_coord:.1f} m]",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_xlabel(f"{u_col.upper()} Coordinate [m]")
    axes[0].set_ylabel(f"{v_col.upper()} Coordinate [m]")
    axes[0].set_aspect("equal")
    axes[0].grid(True, linestyle=":", alpha=0.5, zorder=1)
    cb_g = fig.colorbar(
        pc_grade, ax=axes[0], orientation="vertical", shrink=0.85, pad=0.02
    )
    cb_g.set_label(f"Grade ({grade_unit})", fontsize=10, fontweight="bold")

    # 2. Variance Panel with theoretical scale anchoring
    valid_var = df_slice[var_col][np.isfinite(df_slice[var_col])]
    v_min_eff = float(vmin_var)
    if vmax_var is not None:
        v_max_eff = float(vmax_var)
    else:
        v_max_eff = float(np.max(valid_var)) if len(valid_var) > 0 else 1.0
    norm_var = mcolors.Normalize(vmin=v_min_eff, vmax=max(v_min_eff + 1e-6, v_max_eff))
    cmap_v = plt.get_cmap(var_cmap).copy()
    cmap_v.set_bad(color="#e0e0e0")

    pc_var = _render_2d_block_patches(
        axes[1], df_slice, u_col, v_col, du_col, dv_col, var_col, norm_var, cmap_v
    )
    axes[1].set_title(
        f"Kriging Estimation Variance σ_OK^2 [{slice_axis.upper()} = {target_coord:.1f} m]",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_xlabel(f"{u_col.upper()} Coordinate [m]")
    axes[1].set_ylabel(f"{v_col.upper()} Coordinate [m]")
    axes[1].set_aspect("equal")
    axes[1].grid(True, linestyle=":", alpha=0.5, zorder=1)
    cb_v = fig.colorbar(
        pc_var, ax=axes[1], orientation="vertical", shrink=0.85, pad=0.02
    )
    cb_v.set_label("Estimation Variance (σ²)", fontsize=10, fontweight="bold")

    # Drillhole pierce point overlay
    if samples_xyz is not None:
        axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
        u_idx = {"x": 0, "y": 1, "z": 2}[u_col]
        v_idx = {"x": 0, "y": 1, "z": 2}[v_col]
        mask_s = np.abs(samples_xyz[:, axis_idx] - target_coord) <= 0.5 * d_slice
        if np.any(mask_s):
            # Left: grade colormap
            c_g = sample_grades[mask_s] if sample_grades is not None else "black"
            axes[0].scatter(
                samples_xyz[mask_s, u_idx],
                samples_xyz[mask_s, v_idx],
                c=c_g,
                cmap=cmap_grade if sample_grades is not None else None,
                norm=norm_grade if sample_grades is not None else None,
                edgecolor="black",
                linewidth=0.8,
                s=35,
                zorder=4,
                label=f"Composites (±{0.5 * d_slice:.1f}m)",
            )
            axes[0].legend(loc="upper right", framealpha=0.8, fontsize=8)

            # Right: black markers indicating sampling locations
            axes[1].scatter(
                samples_xyz[mask_s, u_idx],
                samples_xyz[mask_s, v_idx],
                color="cyan",
                edgecolor="black",
                linewidth=0.8,
                s=35,
                zorder=4,
                label=f"Drillholes (±{0.5 * d_slice:.1f}m)",
            )
            axes[1].legend(loc="upper right", framealpha=0.8, fontsize=8)

    fig_title = (
        title
        or f"Block Model Grade vs. Geostatistical Estimation Uncertainty [{slice_axis.upper()} Slice]"
    )
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")
    return fig, axes

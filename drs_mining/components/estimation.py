"""Polygonal mineral resource and reserve estimation.

Provides functional tools for geometric polygonal estimation (method of polygons
of influence), global reserve calculations, cutoff-grade sensitivity analysis
(grade-tonnage curves), and spatial 2D plan map visualizations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Polygon as MplPolygon, Rectangle as MplRectangle
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon, box
from shapely.ops import voronoi_diagram
from scipy.spatial import KDTree, cKDTree, Delaunay
from scipy.spatial.distance import cdist, pdist
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy import stats
from scipy.stats import multivariate_normal


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


def _resolve_domain_parameter(param: Any, domain: Any, default: Any = None) -> Any:
    """Resolves a parameter that may be a mapping per domain or a global scalar/sequence."""
    if isinstance(param, Mapping):
        if domain in param:
            return param[domain]
        return default if default is not None else param
    return param


def _validate_domain_pairing(
    sample_domains: Optional[Sequence[Any]],
    grid_domains: Optional[Sequence[Any]],
    n_samples: int,
    n_grid: int,
    sample_name: str = "samples_xy",
    grid_name: str = "grid_points",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Validates that sample and grid domain arrays are paired and match sample/grid lengths."""
    if (sample_domains is None) != (grid_domains is None):
        raise ValueError(
            "Both sample_domains and grid_domains must be provided together."
        )
    if sample_domains is not None and grid_domains is not None:
        s_dom = np.asarray(sample_domains)
        g_dom = np.asarray(grid_domains)
        if len(s_dom) != n_samples:
            raise ValueError(
                f"sample_domains length ({len(s_dom)}) must match {sample_name} ({n_samples})."
            )
        if len(g_dom) != n_grid:
            raise ValueError(
                f"grid_domains length ({len(g_dom)}) must match {grid_name} ({n_grid})."
            )
        return s_dom, g_dom
    return None, None


def _generate_block_discretization_offsets(
    discretization: Tuple[int, int, int],
    block_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Generates 3D internal discretization sub-points relative to block centroid."""
    nx, ny, nz = discretization
    dx, dy, dz = block_size
    if nx == 1 and ny == 1 and nz == 1:
        return np.zeros((1, 3), dtype=float)
    x_offsets = ((np.arange(nx) + 0.5) / nx - 0.5) * dx
    y_offsets = ((np.arange(ny) + 0.5) / ny - 0.5) * dy
    z_offsets = ((np.arange(nz) + 0.5) / nz - 0.5) * dz
    xx, yy, zz = np.meshgrid(x_offsets, y_offsets, z_offsets, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _deduplicate_collocated_samples(
    coords: np.ndarray,
    grades: np.ndarray,
    target_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Deduplicates collocated sample coordinates by averaging grades to avoid singular Kriging matrices.

    Returns
    -------
    coords : np.ndarray
        Unique sample coordinates of shape (k, dim).
    grades : np.ndarray
        Averaged sample grades of shape (k,).
    h_matrix : np.ndarray
        Inter-sample distance matrix of shape (k, k) with 0.0 on diagonal.
    d_target : np.ndarray, optional
        Distances from samples to target_point if provided.
    """
    k = len(coords)
    if k <= 1:
        diff_matrix = coords[:, None, :] - coords[None, :, :]
        h_matrix = np.linalg.norm(diff_matrix, axis=2)
        d_target = (
            np.linalg.norm(coords - target_point, axis=1)
            if target_point is not None
            else None
        )
        return coords, grades, h_matrix, d_target

    diff_matrix = coords[:, None, :] - coords[None, :, :]
    h_matrix = np.linalg.norm(diff_matrix, axis=2)
    np.fill_diagonal(h_matrix, np.inf)
    d_target = (
        np.linalg.norm(coords - target_point, axis=1)
        if target_point is not None
        else None
    )

    if np.any(h_matrix < 1e-6):
        unique_coords, inverse_indices = np.unique(
            coords.round(decimals=5), axis=0, return_inverse=True
        )
        if len(unique_coords) < k:
            new_grades = np.zeros(len(unique_coords), dtype=float)
            for u_idx in range(len(unique_coords)):
                new_grades[u_idx] = grades[inverse_indices == u_idx].mean()
            coords = unique_coords
            grades = new_grades
            if target_point is not None:
                d_target = np.linalg.norm(coords - target_point, axis=1)
            diff_matrix = coords[:, None, :] - coords[None, :, :]
            h_matrix = np.linalg.norm(diff_matrix, axis=2)

    np.fill_diagonal(h_matrix, 0.0)
    return coords, grades, h_matrix, d_target


def _filter_neighborhood_sectors(
    candidate_coords: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_distances: np.ndarray,
    target_coord: np.ndarray,
    k_neighbors: int,
    max_per_sector: Optional[int] = None,
    min_sectors: int = 1,
    max_per_drillhole: Optional[int] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filters neighbor candidates into spatial sectors (2D quadrants or 3D octants) and drillhole caps.

    Adheres to the SME Mining Engineering Handbook (§4.5, Rule 6 & Rule 7) (TODO: Manually Verify):
    - Rule 6: Restricts the maximum number of samples taken per drillhole trace (`max_per_drillhole`).
    - Rule 7: Partitions the search neighborhood into angular sectors (4 quadrants in 2D, 8 octants in 3D)
      and limits samples per sector (`max_per_sector`) with a minimum active sector threshold (`min_sectors`).

    Parameters
    ----------
    candidate_coords : np.ndarray
        Array of candidate sample coordinates of shape (M, D), where D is 2 or 3.
    candidate_indices : np.ndarray
        Array of candidate sample indices of shape (M,).
    candidate_distances : np.ndarray
        Array of candidate distances from target coordinate of shape (M,).
    target_coord : np.ndarray
        Target coordinate of shape (D,).
    k_neighbors : int
        Maximum total number of neighbors to select.
    max_per_sector : int, optional
        Maximum samples allowed per sector (quadrant in 2D, octant in 3D).
    min_sectors : int, default 1
        Minimum number of distinct sectors that must be populated. If fewer sectors
        are present among accepted samples, empty arrays are returned.
    max_per_drillhole : int, optional
        Maximum samples allowed per drillhole trace.
    sample_hole_ids : Sequence[Any], optional
        Drillhole identifiers corresponding to candidate samples.

    Returns
    -------
    selected_indices : np.ndarray
        Indices of selected samples of shape (k_selected,).
    selected_distances : np.ndarray
        Distances of selected samples of shape (k_selected,).
    """
    if len(candidate_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if max_per_sector is None and max_per_drillhole is None and min_sectors <= 1:
        take = min(k_neighbors, len(candidate_indices))
        return candidate_indices[:take], candidate_distances[:take]

    dim = len(target_coord)
    diffs = candidate_coords - target_coord

    if dim == 2:
        sectors = (diffs[:, 0] >= 0).astype(int) | ((diffs[:, 1] >= 0).astype(int) << 1)
    else:
        sectors = (
            (diffs[:, 0] >= 0).astype(int)
            | ((diffs[:, 1] >= 0).astype(int) << 1)
            | ((diffs[:, 2] >= 0).astype(int) << 2)
        )

    hole_ids_arr = np.asarray(sample_hole_ids) if sample_hole_ids is not None else None

    selected_indices: List[int] = []
    selected_distances: List[float] = []
    sector_counts: Dict[int, int] = {}
    hole_counts: Dict[Any, int] = {}

    for idx, dist, sec in zip(candidate_indices, candidate_distances, sectors):
        if max_per_sector is not None and sector_counts.get(sec, 0) >= max_per_sector:
            continue
        if max_per_drillhole is not None and hole_ids_arr is not None:
            h_id = hole_ids_arr[idx]
            if hole_counts.get(h_id, 0) >= max_per_drillhole:
                continue
            hole_counts[h_id] = hole_counts.get(h_id, 0) + 1

        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        selected_indices.append(int(idx))
        selected_distances.append(float(dist))
        if len(selected_indices) >= k_neighbors:
            break

    if min_sectors > 1 and len(sector_counts) < min_sectors:
        return np.array([], dtype=int), np.array([], dtype=float)

    return np.asarray(selected_indices, dtype=int), np.asarray(
        selected_distances, dtype=float
    )


def _compute_quadratic_estimation_variance(
    c_vv: float,
    weights: np.ndarray,
    K: np.ndarray,
    k0: np.ndarray,
) -> float:
    """Computes the exact quadratic estimation variance for an arbitrary linear estimator.

    sigma_E^2 = C_bar(V, V) + lambda^T * K * lambda - 2 * lambda^T * k0

    Valid for any linear combination of weights (Ordinary Kriging, Simple Kriging, IDW, Polygonal),
    and strictly required whenever weights are modified or clipped away from the unconstrained
    Lagrangian solution.

    Parameters
    ----------
    c_vv : float
        A priori point sill C(0) or block self-covariance C_bar(V, V).
    weights : np.ndarray
        Array of kriging weights of shape (k,).
    K : np.ndarray
        Sample-to-sample covariance matrix of shape (k, k).
    k0 : np.ndarray
        Sample-to-target covariance vector of shape (k,).

    Returns
    -------
    float
        Non-negative estimation variance sigma_E^2 >= 0.0.
    """
    quad = float(weights @ K @ weights)
    cov_term = 2.0 * float(np.sum(weights * k0))
    raw_var = c_vv + quad - cov_term
    return max(0.0, raw_var)


def _clip_and_renormalize_weights(
    weights: np.ndarray,
    target_sum: Optional[float] = 1.0,
    preserve_sum: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Clips negative kriging weights to zero and optionally renormalizes.

    Adheres to Deutsch (1996) and Isaaks & Srivastava (1989) (TODO: Manually Verify):
    when negative weights occur due to the screen effect, clipping them to zero eliminates
    physically impossible grade extrapolations (e.g. negative grades or extreme spikes).

    For Ordinary Kriging, normalizing to target_sum=1.0 ensures the unbiasedness condition
    sum(weights) == 1.0 is strictly preserved.
    For Simple Kriging, setting preserve_sum=True scales the positive weights so their sum
    matches the original sample weight sum S = sum(weights) (clamped to [0.0, 1.0]),
    preserving the weight assigned to the stationary global mean lambda_0 = 1.0 - sum(weights).

    Parameters
    ----------
    weights : np.ndarray
        Array of kriging weights of shape (k,).
    target_sum : float, optional, default 1.0
        If supplied and preserve_sum is False, positive weights are rescaled to sum to target_sum.
    preserve_sum : bool, default False
        If True, rescales clipped positive weights so their sum matches the original sample
        weight sum S = sum(weights) (clamped to [0.0, 1.0]). If both target_sum is None and
        preserve_sum is False, unconstrained non-negative clipping is performed.

    Returns
    -------
    renorm_weights : np.ndarray
        Array of non-negative weights.
    was_clipped : bool
        True if any weight was negative and modified, False otherwise.
    """
    if not np.any(weights < 0.0):
        return weights, False

    clipped = np.maximum(weights, 0.0)
    pos_sum = float(np.sum(clipped))

    if preserve_sum:
        orig_sum = float(np.sum(weights))
        desired_sum = max(0.0, min(1.0, orig_sum))
        if pos_sum > 0.0 and desired_sum > 0.0:
            renorm = clipped * (desired_sum / pos_sum)
        else:
            renorm = np.zeros_like(weights)
        return renorm, True

    if target_sum is not None:
        if pos_sum > 0.0:
            renorm = clipped * (target_sum / pos_sum)
        else:
            renorm = np.full_like(weights, target_sum / len(weights))
        return renorm, True

    return clipped, True


def _solve_ordinary_kriging_system(
    K: np.ndarray,
    k0: np.ndarray,
    clip_negative_weights: bool = False,
) -> Tuple[np.ndarray, float]:
    """Solves the augmented Ordinary Kriging linear system for weights and Lagrange multiplier.

    [ K   1 ] [ lambda ] = [ k0 ]
    [ 1^T 0 ] [   mu   ]   [  1 ]

    Parameters
    ----------
    K : np.ndarray
        Sample-to-sample covariance matrix of shape (k, k).
    k0 : np.ndarray
        Sample-to-target covariance vector of shape (k,).
    clip_negative_weights : bool, default False
        If True, clips any negative weights (screen-effect artifacts) to zero and renormalizes
        positive weights so they sum to 1.0 (Deutsch, 1996; Isaaks & Srivastava, 1989).

    Returns
    -------
    weights : np.ndarray
        Ordinary Kriging weights lambda of shape (k,) summing to 1.0.
    mu : float
        Lagrange multiplier.
    """
    k = len(k0)
    A = np.ones((k + 1, k + 1), dtype=float)
    A[:k, :k] = K
    A[k, k] = 0.0

    rhs = np.ones(k + 1, dtype=float)
    rhs[:k] = k0

    cond = np.linalg.cond(A)
    if cond > 1e4:
        warnings.warn(
            f"Ill-Conditioned Kriging System: Ordinary Kriging matrix condition number is {cond:.1e} (> 1e4). "
            f"This confirms that your sample geometry (e.g. collinear, duplicate, or heavily clustered samples) "
            f"and variogram model are causing numerical instability.",
            UserWarning,
            stacklevel=2,
        )

    solution = np.linalg.solve(A, rhs)
    weights = solution[:k]
    mu = float(solution[k])

    if clip_negative_weights:
        weights, _ = _clip_and_renormalize_weights(weights, target_sum=1.0)

    return weights, mu


# TODO: should follow naming convention and be called grid.
# TODO: do we need 3d anistropy for a grid estimation?
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
    s_dom, g_dom = _validate_domain_pairing(
        sample_domains, grid_domains, len(samples_xy), len(grid_points)
    )

    # Handle domain-segregated estimation
    if s_dom is not None and g_dom is not None:
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
            dom_aniso_ranges = _resolve_domain_parameter(anisotropy_ranges, dom)
            dom_aniso_angles = _resolve_domain_parameter(anisotropy_angles, dom)
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


def inverse_distance_weighting_block_estimation(
    samples_xyz: np.ndarray,
    sample_grades: np.ndarray,
    block_model: pd.DataFrame,
    power: float = 2.0,
    k_neighbors: int = 16,
    max_radius: Optional[float] = None,
    discretization: Tuple[int, int, int] = (4, 4, 2),
    mask_extrapolation: bool = False,
    domain_col: Optional[str] = None,
    sample_domains: Optional[Sequence[Any]] = None,
    sample_domain_col: Optional[Union[str, Sequence[Any]]] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimates 3D block grades using Inverse Distance Weighting (IDW) with internal volume discretization.

    Supports volume support averaging, geological domain boundaries, and 3D spatial anisotropy
    under international reporting standards (CIM Definition Standards 2014, CIM MRMR Guidelines §6.8/§6.9,
    JORC Code 2012 Table 1, and NI 43-101) (TODO: Manually Verify):
    - Volume Support Discretization: discretizes each 3D SMU block into (nx, ny, nz) internal points,
      evaluates IDW at each internal sub-point, and averages sub-point estimates to represent volume-averaged
      block grade Z*(V) (discretization=(1, 1, 1) evaluates directly at block centroids).
    - Anisotropic Distance Metric: evaluates distances in rotated 3D search ellipsoids when
      `anisotropy_ranges` and/or `anisotropy_angles` are supplied.
    - Domain Segregation: restricts search and estimation strictly within geological domains
      defined by `domain_col` and `sample_domains`.

    Parameters
    ----------
    samples_xyz : np.ndarray
        Sample coordinates of shape (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and dimensions (dx, dy, dz).
    power : float, default 2.0
        Inverse distance weighting exponent p (e.g. 2.0 for IDW2, 3.0 for IDW3).
    k_neighbors : int, default 16
        Maximum number of informing samples queried per block.
    max_radius : float, optional
        Maximum search radius. Blocks with no samples within this distance are set to NaN.
    discretization : tuple of (int, int, int), default (4, 4, 2)
        Number of internal discretization sub-points (nx, ny, nz) per block for volume support averaging.
        Use (1, 1, 1) for centroid-only estimation.
    mask_extrapolation : bool, default False
        If True, blocks whose centroids lie outside the 3D convex hull of samples are set to NaN.
    domain_col : str, optional
        Geological domain column in `block_model`.
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples.
    sample_domain_col : str or Sequence[Any], optional
        Alias for `sample_domains`.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional search ranges (major, semi-major, minor) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth, dip, plunge) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to angles.

    Returns
    -------
    estimated_grades : np.ndarray
        Interpolated block grades of shape (n_blocks,).
    mean_distances : np.ndarray
        Mean physical distance from block centroid to informing samples of shape (n_blocks,).
    """
    n_blocks = len(block_model)
    if n_blocks == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    req_cols = {"x", "y", "z", "dx", "dy", "dz"}
    missing = req_cols - set(block_model.columns)
    if missing:
        raise ValueError(
            f"block_model is missing required columns: {sorted(list(missing))}."
        )

    if power < 0:
        raise ValueError(f"power must be non-negative, got {power}.")

    if k_neighbors < 1:
        raise ValueError(f"k_neighbors must be at least 1, got {k_neighbors}.")

    if max_radius is not None and max_radius <= 0:
        raise ValueError(f"max_radius must be positive, got {max_radius}.")

    if (
        not isinstance(discretization, (tuple, list))
        or len(discretization) != 3
        or any(n <= 0 for n in discretization)
    ):
        raise ValueError(
            f"discretization must be a tuple of 3 positive integers, got {discretization}."
        )

    samples_xyz_arr = np.asarray(samples_xyz, dtype=float)
    sample_grades_arr = np.asarray(sample_grades, dtype=float)

    if len(samples_xyz_arr) != len(sample_grades_arr):
        raise ValueError(
            f"Length of sample_grades ({len(sample_grades_arr)}) must match samples_xyz ({len(samples_xyz_arr)})."
        )

    if samples_xyz_arr.size > 0 and (
        samples_xyz_arr.ndim != 2 or samples_xyz_arr.shape[1] != 3
    ):
        raise ValueError(
            f"samples_xyz must have shape (N, 3), got {samples_xyz_arr.shape}."
        )

    # Handle domain-segregated estimation
    s_dom = sample_domains if sample_domains is not None else sample_domain_col
    if (domain_col is None) != (s_dom is None):
        raise ValueError(
            "Both domain_col and sample_domains must be provided together."
        )

    if domain_col is not None and s_dom is not None:
        if domain_col not in block_model.columns:
            raise ValueError(f"domain_col '{domain_col}' not found in block_model.")
        s_dom_arr = np.asarray(s_dom)
        b_dom_arr = np.asarray(block_model[domain_col])
        if len(s_dom_arr) != len(samples_xyz_arr):
            raise ValueError(
                f"sample_domains length ({len(s_dom_arr)}) must match samples_xyz ({len(samples_xyz_arr)})."
            )

        block_estimates = np.full(n_blocks, np.nan, dtype=float)
        mean_distances = np.full(n_blocks, np.nan, dtype=float)

        for dom in np.unique(b_dom_arr):
            b_mask = b_dom_arr == dom
            s_mask = s_dom_arr == dom
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

            sub_bm = block_model[b_mask].copy()
            est_dom, dist_dom = inverse_distance_weighting_block_estimation(
                samples_xyz=samples_xyz_arr[s_mask],
                sample_grades=sample_grades_arr[s_mask],
                block_model=sub_bm,
                power=power,
                k_neighbors=k_neighbors,
                max_radius=max_radius,
                discretization=discretization,
                mask_extrapolation=mask_extrapolation,
                domain_col=None,
                sample_domains=None,
                anisotropy_ranges=dom_aniso_ranges,
                anisotropy_angles=dom_aniso_angles,
            )
            block_estimates[b_mask] = est_dom
            mean_distances[b_mask] = dist_dom

        return block_estimates, mean_distances

    block_coords = block_model[["x", "y", "z"]].to_numpy(dtype=float)
    dx = block_model["dx"].to_numpy(dtype=float)
    dy = block_model["dy"].to_numpy(dtype=float)
    dz = block_model["dz"].to_numpy(dtype=float)

    # If no samples, return all NaNs
    if len(samples_xyz_arr) == 0:
        return np.full(n_blocks, np.nan, dtype=float), np.full(
            n_blocks, np.nan, dtype=float
        )

    # Discretization unit offsets centered at 0 within [-0.5, 0.5]
    nx, ny, nz = discretization
    if nx == 1 and ny == 1 and nz == 1:
        unit_offsets = np.zeros((1, 3), dtype=float)
    else:
        x_offsets = (np.arange(nx) + 0.5) / nx - 0.5
        y_offsets = (np.arange(ny) + 0.5) / ny - 0.5
        z_offsets = (np.arange(nz) + 0.5) / nz - 0.5
        xx, yy, zz = np.meshgrid(x_offsets, y_offsets, z_offsets, indexing="ij")
        unit_offsets = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    uniform_dims = (
        np.allclose(dx, dx[0]) and np.allclose(dy, dy[0]) and np.allclose(dz, dz[0])
    )
    if uniform_dims and not (nx == 1 and ny == 1 and nz == 1):
        base_disc_offsets = unit_offsets * np.array([dx[0], dy[0], dz[0]])
        base_disc_offsets_t, _ = transform_anisotropic_coordinates(
            base_disc_offsets, ranges=anisotropy_ranges, angles=anisotropy_angles
        )

    # Transform coordinates to anisotropic space
    samples_xyz_t, _ = transform_anisotropic_coordinates(
        samples_xyz_arr, ranges=anisotropy_ranges, angles=anisotropy_angles
    )
    block_coords_t, _ = transform_anisotropic_coordinates(
        block_coords, ranges=anisotropy_ranges, angles=anisotropy_angles
    )

    upper_bound = max_radius if max_radius is not None else float("inf")
    tree = KDTree(samples_xyz_t)
    k_query = min(k_neighbors, len(samples_xyz_arr))

    distances_t, indices = tree.query(
        block_coords_t,
        k=k_query,
        distance_upper_bound=upper_bound,
    )

    if k_query == 1:
        distances_t = distances_t[:, None]
        indices = indices[:, None]

    block_estimates = np.full(n_blocks, np.nan, dtype=float)
    mean_distances = np.full(n_blocks, np.nan, dtype=float)

    for b in range(n_blocks):
        valid_mask = np.isfinite(distances_t[b]) & (distances_t[b] <= upper_bound)
        if not np.any(valid_mask):
            continue

        active_indices = indices[b][valid_mask]
        coords_active_t = samples_xyz_t[active_indices]  # shape (k, 3)
        grades_active = sample_grades_arr[active_indices]  # shape (k,)

        # Physical mean distance from block centroid to informing samples
        phys_diffs = samples_xyz_arr[active_indices] - block_coords[b]
        phys_dists = np.linalg.norm(phys_diffs, axis=1)
        mean_distances[b] = float(np.mean(phys_dists))

        if nx == 1 and ny == 1 and nz == 1:
            # Centroid-only IDW in transformed anisotropic space
            d_t = distances_t[b][valid_mask]
            exact_match = d_t < 1e-6
            if np.any(exact_match):
                first_exact = np.argmax(exact_match)
                block_estimates[b] = float(grades_active[first_exact])
            else:
                safe_d = np.maximum(d_t, 1e-12)
                raw_w = 1.0 / np.power(safe_d, power)
                tot_w = np.sum(raw_w)
                if tot_w > 0:
                    block_estimates[b] = float(np.sum((raw_w / tot_w) * grades_active))
        else:
            # Discretized sub-points
            if uniform_dims:
                disc_offsets_t = base_disc_offsets_t
            else:
                block_disc = unit_offsets * np.array([dx[b], dy[b], dz[b]])
                disc_offsets_t, _ = transform_anisotropic_coordinates(
                    block_disc, ranges=anisotropy_ranges, angles=anisotropy_angles
                )

            # Block sub-points in transformed anisotropic space: shape (M, 3)
            sub_points_t = block_coords_t[b] + disc_offsets_t
            # Pairwise distances from active samples to sub-points: shape (k, M)
            diffs = coords_active_t[:, None, :] - sub_points_t[None, :, :]
            sub_dists = np.linalg.norm(diffs, axis=2)

            is_exact = sub_dists < 1e-6
            has_exact = np.any(is_exact, axis=0)  # (M,)

            safe_d = np.maximum(sub_dists, 1e-12)
            raw_w = np.where(~is_exact, 1.0 / np.power(safe_d, power), 0.0)
            tot_w = np.sum(raw_w, axis=0, keepdims=True)

            norm_w = np.divide(
                raw_w, tot_w, out=np.zeros_like(raw_w), where=tot_w > 0.0
            )
            sub_grades = np.sum(norm_w * grades_active[:, None], axis=0)

            if np.any(has_exact):
                first_exact_idx = np.argmax(is_exact[:, has_exact], axis=0)
                sub_grades[has_exact] = grades_active[first_exact_idx]

            block_estimates[b] = float(np.mean(sub_grades))

    if mask_extrapolation:
        if len(samples_xyz_arr) < 4:
            raise ValueError(
                "mask_extrapolation=True requires at least 4 non-coplanar 3D samples to compute a convex hull."
            )
        try:
            inside_hull = is_within_convex_hull(samples_xyz_arr, block_coords)
        except Exception as e:
            raise ValueError(
                f"Failed to compute 3D convex hull for extrapolation masking (samples may be coplanar): {e}"
            ) from e
        block_estimates[~inside_hull] = np.nan
        mean_distances[~inside_hull] = np.nan

    return block_estimates, mean_distances


def nearest_neighbor_block_estimation(
    samples_xyz: np.ndarray,
    sample_grades: np.ndarray,
    block_model: pd.DataFrame,
    max_radius: Optional[float] = None,
    mask_extrapolation: bool = False,
    domain_col: Optional[str] = None,
    sample_domains: Optional[Sequence[Any]] = None,
    sample_domain_col: Optional[Union[str, Sequence[Any]]] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
    return_distances: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Estimates 3D block grades using Nearest Neighbor assignment.

    Assigns each block to the nearest informing sample composite in 3D (or rotated anisotropic 3D space).
    Serves as the mandatory declustered 3D polygonal/Voronoi benchmark proxy for swath plot
    local bias validation (`plot_swath_analysis`) and global grade-tonnage audits (`plot_grade_tonnage_curve`)
    under CIM Definition Standards (2014), CIM MRMR Guidelines §6.8/§6.9, and JORC Code (2012) Table 1
    (TODO: Manually Verify).

    Parameters
    ----------
    samples_xyz : np.ndarray
        Sample coordinates of shape (N, 3).
    sample_grades : np.ndarray
        Assay grades of shape (N,).
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and dimensions (dx, dy, dz).
    max_radius : float, optional
        Maximum search radius. Blocks with no samples within this distance are set to NaN.
    mask_extrapolation : bool, default False
        If True, blocks whose centroids lie outside the 3D convex hull of samples are set to NaN.
    domain_col : str, optional
        Geological domain column in `block_model`.
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples.
    sample_domain_col : str or Sequence[Any], optional
        Alias for `sample_domains`.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional search ranges (major, semi-major, minor) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth, dip, plunge) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to angles.
    return_distances : bool, default False
        If True, returns a tuple `(estimated_grades, distances)`.

    Returns
    -------
    estimated_grades : np.ndarray
        Array of estimated block grades of shape (n_blocks,).
        If `return_distances=True`, returns `(estimated_grades, distances)`.
    """
    est_grades, dists = inverse_distance_weighting_block_estimation(
        samples_xyz=samples_xyz,
        sample_grades=sample_grades,
        block_model=block_model,
        power=1.0,
        k_neighbors=1,
        max_radius=max_radius,
        discretization=(1, 1, 1),
        mask_extrapolation=mask_extrapolation,
        domain_col=domain_col,
        sample_domains=sample_domains,
        sample_domain_col=sample_domain_col,
        anisotropy_ranges=anisotropy_ranges,
        anisotropy_angles=anisotropy_angles,
    )
    if return_distances:
        return est_grades, dists
    return est_grades


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
    is_scalar = h_arr.ndim == 0
    if is_scalar:
        h_arr = np.atleast_1d(h_arr)

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
    return float(cov[0]) if is_scalar else cov


# =============================================================================
# SPATIAL VARIOGRAPHY & STRUCTURAL ANALYSIS (EXPERIMENTAL & MODEL FITTING)
# =============================================================================


def calculate_experimental_semivariogram(
    coords: Union[np.ndarray, pd.DataFrame],
    grades: Union[np.ndarray, Sequence[float], str],
    lag_distance: float,
    n_lags: int,
    lag_tolerance: Optional[float] = None,
    azimuth: Optional[float] = None,
    dip: Optional[float] = None,
    angular_tolerance: Optional[float] = None,
    bandwidth_horizontal: Optional[float] = None,
    bandwidth_vertical: Optional[float] = None,
    domain_col: Optional[str] = None,
    sample_domains: Optional[Sequence[Any]] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
) -> pd.DataFrame:
    """Computes the empirical experimental semivariogram from spatial samples or composites.

    Evaluates the classical Matheron (1965) estimator across discrete separation lag bins
    in accordance with SME Mining Engineering Handbook (§4.3) and CIM MRMR Best Practice
    Guidelines (§5 on Spatial and Structural Analysis) (TODO: Manually Verify):

        \\hat{\\gamma}(h) = \\frac{1}{2 |N(h)|} \\sum_{(i,j) \\in N(h)} (Z(x_i) - Z(x_j))^2

    where |N(h)| is the number of paired samples falling within separation lag distance
    interval h \\pm \\Delta h, and Z(x_i) is the measured grade attribute.

    Supports both Omnidirectional and Directional evaluation:
    - In Omnidirectional mode (azimuth=None, dip=None): All sample pairs separated by
      distance h are included regardless of direction.
    - In Directional mode: Pairs are filtered through a directional tolerance cone defined
      by azimuth, dip, angular_tolerance (\\pm \\Delta \\theta), and maximum horizontal and
      vertical bandwidths to detect spatial anisotropy.
    - Honors Geological Domain Segregation: When `domain_col` or `sample_domains` is provided,
      pairs are only formed between samples belonging to the same domain (CIM MRMR §4.2/§5.1) (TODO: Manually Verify).

    Parameters
    ----------
    coords : np.ndarray or pd.DataFrame
        Sample spatial coordinates of shape (N, 2) or (N, 3), or DataFrame containing coordinate columns.
    grades : np.ndarray, Sequence[float], or str
        Sample grade array of shape (N,), or column name in coords DataFrame.
    lag_distance : float
        Nominal separation distance increment (h) between lag bins.
        Must be strictly positive. Site-specific parameter required without default (CONTRIBUTING.md).
    n_lags : int
        Total number of lag intervals to compute. Must be >= 1. Required without default.
    lag_tolerance : float, optional
        Half-width of the distance lag bin (\\pm \\Delta h). If None, defaults to lag_distance / 2.0.
    azimuth : float, optional
        Directional search azimuth in degrees clockwise from North [0, 360).
        If specified, activates directional variogram calculation.
    dip : float, optional
        Directional search dip in degrees from horizontal [-90, +90] (negative downwards).
    angular_tolerance : float, optional
        Half-angle tolerance window in degrees (\\pm \\Delta \\theta), e.g. 22.5 deg.
        Required when azimuth or dip is specified.
    bandwidth_horizontal : float, optional
        Maximum horizontal perpendicular distance from directional vector to prevent
        the search cone from expanding indefinitely at large lags.
    bandwidth_vertical : float, optional
        Maximum vertical perpendicular distance from directional vector.
    domain_col : str, optional
        Column name in coords DataFrame indicating domain code.
    sample_domains : Sequence[Any], optional
        Domain identifiers for each sample when coords is an array.
    x_col : str, default "x"
        Easting coordinate column name if coords is a DataFrame.
    y_col : str, default "y"
        Northing coordinate column name if coords is a DataFrame.
    z_col : str, optional
        Elevation coordinate column name if coords is a DataFrame.

    Returns
    -------
    pd.DataFrame
        Experimental semivariogram table containing:
        - "lag_index": 1-based integer lag index.
        - "nominal_lag": Nominal lag center distance (lag_index * lag_distance).
        - "average_distance": True centroid separation distance of paired samples in bin.
        - "semivariance": Experimental semivariance \\hat{\\gamma}(h).
        - "covariance": Experimental covariance \\hat{C}(h) = \\hat{C}(0) - \\hat{\\gamma}(h).
        - "pair_count": Number of valid sample pairs |N(h)| informing the lag.
        Attributes (.attrs) contain sample variance, direction metadata, and audit parameters.
    """
    raise NotImplementedError(
        "Experimental semivariogram calculation engine is planned. "
        "Will compute pairwise binned semivariances with directional tolerance cones."
    )


def calculate_directional_semivariograms(
    coords: Union[np.ndarray, pd.DataFrame],
    grades: Union[np.ndarray, Sequence[float], str],
    directions: Sequence[Union[float, Tuple[float, float], Mapping[str, float]]],
    lag_distance: float,
    n_lags: int,
    lag_tolerance: Optional[float] = None,
    angular_tolerance: float = 22.5,
    bandwidth_horizontal: Optional[float] = None,
    bandwidth_vertical: Optional[float] = None,
    domain_col: Optional[str] = None,
    sample_domains: Optional[Sequence[Any]] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Calculates a multi-directional suite of experimental semivariograms to identify anisotropy.

    Following standard geostatistical exploration practice (SME Mining Engineering Handbook §4.3;
    Deutsch & Journel 1998; Armstrong 1998) (TODO: Manually Verify), evaluates experimental
    semivariograms along candidate strike, dip, and plunge directions (e.g., 0, 45, 90, 135 deg)
    to detect Geometric Anisotropy (varying range with identical sill) or Zonal Anisotropy
    (varying sill by direction).

    Parameters
    ----------
    coords : np.ndarray or pd.DataFrame
        Sample spatial coordinates (N, 2) or (N, 3), or DataFrame.
    grades : np.ndarray, Sequence[float], or str
        Sample grade array or column name.
    directions : Sequence[Union[float, Tuple[float, float], Mapping[str, float]]]
        List of direction definitions to evaluate. Each element can be:
        - float: Azimuth in degrees (horizontal 2D directional search).
        - (azimuth, dip) tuple: 3D spatial search direction.
        - dict: {"azimuth": float, "dip": float, "label": str}.
    lag_distance : float
        Nominal distance increment between lags (required, no default).
    n_lags : int
        Number of lags to compute (required, no default).
    lag_tolerance : float, optional
        Half-width distance lag tolerance (defaults to lag_distance / 2.0).
    angular_tolerance : float, default 22.5
        Angular half-window (\\pm \\Delta \\theta) in degrees.
    bandwidth_horizontal : float, optional
        Maximum horizontal perpendicular search width.
    bandwidth_vertical : float, optional
        Maximum vertical perpendicular search height.
    domain_col : str, optional
        Domain column name in DataFrame.
    sample_domains : Sequence[Any], optional
        Domain array matching coords.
    x_col, y_col, z_col : str
        Coordinate column names.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping directional labels (e.g. "Az000_Dip00", "Az090_Dip00") to
        their respective experimental semivariogram DataFrames.
    """
    raise NotImplementedError(
        "Directional semivariogram suite calculation is planned. "
        "Will compute multiple directional variograms across specified strike and dip orientations."
    )


def calculate_downhole_variogram(
    df: pd.DataFrame,
    lag_distance: float,
    n_lags: int,
    lag_tolerance: Optional[float] = None,
    hole_id_col: str = "hole_id",
    from_col: str = "from_m",
    to_col: str = "to_m",
    grade_col: str = "grade",
    domain_col: Optional[str] = None,
) -> pd.DataFrame:
    """Calculates down-hole experimental semivariogram along continuous drillhole traces.

    In 3D mineral deposit evaluation (SME Mining Engineering Handbook §4.3; CIM MRMR Guidelines §5) (TODO: Manually Verify),
    sample spacing along drillhole cores (typically 1–3 m composites) is orders of magnitude
    denser than lateral spacing between hole collars (50–100 m).
    The down-hole variogram computes semivariances strictly between pairs sharing the same
    drillhole trajectory:
    1. Accurately isolates the true 3D Nugget Effect (c0) without interference from wide drill patterns.
    2. Resets at geological domain contacts (domain_col) to avoid contact smearing.

    Parameters
    ----------
    df : pd.DataFrame
        Drillhole assay or composite table containing hole identifier and downhole intervals.
    lag_distance : float
        Downhole lag spacing increment (typically equal to composite interval length).
        Required without default (CONTRIBUTING.md).
    n_lags : int
        Number of downhole lags to compute. Required without default.
    lag_tolerance : float, optional
        Downhole lag distance tolerance. Defaults to lag_distance / 2.0.
    hole_id_col : str, default "hole_id"
        Drillhole identifier column name. Pairs are formed strictly within each hole_id.
    from_col : str, default "from_m"
        Downhole from-depth column name.
    to_col : str, default "to_m"
        Downhole to-depth column name.
    grade_col : str, default "grade"
        Assay or composite grade column name.
    domain_col : str, optional
        Geological domain column name. Pairs must belong to the same contiguous domain interval.

    Returns
    -------
    pd.DataFrame
        Down-hole experimental semivariogram DataFrame with columns:
        - "lag_index", "nominal_lag", "average_distance", "semivariance", "pair_count".
        Attributes (.attrs) contain "nugget_estimate" extrapolated back to h=0.
    """
    raise NotImplementedError(
        "Down-hole experimental variogram calculation is planned. "
        "Will compute strictly intra-hole pair semivariances to isolate the 3D nugget effect."
    )


def fit_variogram_model(
    experimental_df: pd.DataFrame,
    model: str = "spherical",
    initial_nugget: Optional[float] = None,
    initial_sill: Optional[float] = None,
    initial_range: Optional[float] = None,
    fix_nugget: Optional[float] = None,
    fix_total_sill: Optional[float] = None,
    fit_method: Literal[
        "weighted_least_squares", "ordinary_least_squares"
    ] = "weighted_least_squares",
    weighting_scheme: Literal["cressie", "pair_count", "equal"] = "cressie",
) -> Dict[str, Any]:
    """Fits analytical positive-definite variogram models to empirical experimental semivariances.

    Models the spatial continuity curve via Nugget Effect (c0), Partial Sill (c), Total Sill
    (c0 + c), and Practical Range (a) using non-linear least-squares optimization
    (Cressie 1985; SME Mining Engineering Handbook §4.3) (TODO: Manually Verify).

    Supported Analytical Models:
    - Spherical: \\gamma(h) = c0 + c * [1.5 * (h/a) - 0.5 * (h/a)^3] for h <= a, else c0 + c
    - Exponential: \\gamma(h) = c0 + c * [1 - exp(-3h/a)]
    - Gaussian: \\gamma(h) = c0 + c * [1 - exp(-3h^2 / a^2)]

    Optimization Weighting Schemes:
    - "cressie" (Cressie 1985): Weight w_k = |N(h_k)| / [\\gamma(h_k)]^2.
      Assigns highest weight to short-lag pairs with high reliability, prioritizing accurate
      estimation near the origin (critical for kriging weights).
    - "pair_count": Weight w_k = |N(h_k)|.
    - "equal": Ordinary unweighted least-squares.

    Parameters
    ----------
    experimental_df : pd.DataFrame
        Table containing 'average_distance' (or 'nominal_lag'), 'semivariance', and 'pair_count'
        from calculate_experimental_semivariogram.
    model : {"spherical", "exponential", "gaussian"}, default "spherical"
        Theoretical positive-definite variogram model structure.
    initial_nugget : float, optional
        Initial guess for nugget variance (c0). If None, estimated from first lag.
    initial_sill : float, optional
        Initial guess for partial sill (c). If None, estimated from sample variance - c0.
    initial_range : float, optional
        Initial guess for range (a). If None, estimated at 2/3 of maximum lag.
    fix_nugget : float, optional
        If specified, holds nugget variance strictly fixed (e.g., from down-hole variogram analysis).
    fix_total_sill : float, optional
        If specified, constrains total sill (c0 + c) to an exact value (e.g., declustered sample variance).
    fit_method : {"weighted_least_squares", "ordinary_least_squares"}, default "weighted_least_squares"
        Optimization loss function.
    weighting_scheme : {"cressie", "pair_count", "equal"}, default "cressie"
        Statistical weighting scheme across lag bins.

    Returns
    -------
    Dict[str, Any]
        Fitted model parameters dictionary containing:
        - "model": Model name string.
        - "nugget": Fitted nugget variance c0.
        - "sill": Fitted partial sill variance c.
        - "total_sill": Total sill c0 + c.
        - "range": Practical range parameter a (meters).
        - "rmse": Root Mean Square Error of the fit.
        - "r2": Coefficient of determination.
        - "fitted_curve": DataFrame with dense (h, gamma) evaluations for plotting.
    """
    raise NotImplementedError(
        "Variogram model parameterization and curve fitting is planned. "
        "Will optimize nugget, sill, and range via non-linear weighted least squares."
    )


def plot_variogram_model(
    experimental_data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    fitted_model: Optional[
        Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]
    ] = None,
    grade_unit: str = "% Cu",
    distance_unit: str = "m",
    title: str = "Experimental Semivariogram and Fitted Model",
    show_pair_counts: bool = True,
    figsize: Tuple[float, float] = (9.0, 6.0),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]:
    """Plots experimental semivariogram points against the fitted continuous theoretical curve.

    Standard publication-quality variogram diagnostic chart conforming to NI 43-101 and JORC
    reporting standards (TODO: Manually Verify):
    1. Upper panel: Experimental semivariance points \\hat{\\gamma}(h) with marker sizes
       proportional to pair counts, fitted continuous model curve, total sill line, and practical range indicator.
    2. Lower panel: Bar chart of sample pair counts |N(h)| informing each lag interval.
    3. Multi-directional overlay: Supports comparing multiple directional curves simultaneously.

    Parameters
    ----------
    experimental_data : pd.DataFrame or Mapping[str, pd.DataFrame]
        Experimental semivariogram table, or dictionary mapping directional labels to tables.
    fitted_model : dict or Sequence[dict], optional
        Output from fit_variogram_model containing fitted parameters and curve coordinates.
    grade_unit : str, default "% Cu"
        Unit of mineral grade (used for variance axis label: e.g. "(% Cu)^2").
    distance_unit : str, default "m"
        Spatial separation distance unit (default "m").
    title : str, default "Experimental Semivariogram and Fitted Model"
        Figure title.
    show_pair_counts : bool, default True
        If True, displays lower sub-panel with pair counts per lag bin.
    figsize : Tuple[float, float], default (9.0, 6.0)
        Matplotlib figure dimensions.
    ax : plt.Axes, optional
        Existing Axes to draw on (only if show_pair_counts=False).

    Returns
    -------
    Tuple[plt.Figure, Union[plt.Axes, Sequence[plt.Axes]]]
        Figure and axes (or tuple of upper/lower axes).
    """
    raise NotImplementedError(
        "Variogram model visualization is planned. "
        "Will plot experimental variogram points, pair count bars, and fitted analytical curves."
    )


def _solve_local_simple_kriging_grid(
    target_coord: np.ndarray,
    candidate_coords: np.ndarray,
    candidate_values: np.ndarray,
    candidate_distances: np.ndarray,
    candidate_hole_ids: Optional[Sequence[Any]] = None,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    sill: float = 1.0,
    range_param: float = 100.0,
    k_neighbors: int = 16,
    max_per_sector: Optional[int] = None,
    min_sectors: int = 1,
    max_per_drillhole: Optional[int] = None,
    clip_negative_weights: bool = False,
    mean: float = 0.0,
) -> Tuple[float, float]:
    """Solves Simple Kriging for a single target grid node given candidate conditioning points.

    Reuses sector filtering (_filter_neighborhood_sectors), collocated sample deduplication
    (_deduplicate_collocated_samples), and theoretical covariance computation (_theoretical_covariance).

    Parameters
    ----------
    target_coord : np.ndarray
        Coordinates of the target point of shape (D,).
    candidate_coords : np.ndarray
        Coordinates of candidate conditioning samples of shape (k_cand, D).
    candidate_values : np.ndarray
        Values (grades or normal scores) of candidate samples of shape (k_cand,).
    candidate_distances : np.ndarray
        Distances from target_coord to candidates of shape (k_cand,).
    candidate_hole_ids : Sequence[Any], optional
        Drillhole trace IDs for candidates.
    variogram_model : str, default "spherical"
        Variogram model name.
    nugget : float, default 0.0
        Nugget variance c0.
    sill : float, default 1.0
        Partial sill variance c.
    range_param : float, default 100.0
        Practical range a.
    k_neighbors : int, default 16
        Maximum neighbors to select.
    max_per_sector : int, optional
        Maximum samples per angular sector (quadrant or octant).
    min_sectors : int, default 1
        Minimum active sectors required.
    max_per_drillhole : int, optional
        Maximum samples per drillhole trace.
    clip_negative_weights : bool, default False
        Whether to clip negative weights and renormalize.
    mean : float, default 0.0
        Known stationary mean.

    Returns
    -------
    Tuple[float, float]
        (estimate, estimation_variance).
    """
    total_sill = float(nugget + sill)
    if len(candidate_coords) == 0:
        return float(mean), float(total_sill)

    has_sector_search = (
        (max_per_sector is not None)
        or (max_per_drillhole is not None)
        or (min_sectors > 1)
    )
    if has_sector_search:
        idx_sel, d_sel = _filter_neighborhood_sectors(
            candidate_coords=candidate_coords,
            candidate_indices=np.arange(len(candidate_coords)),
            candidate_distances=candidate_distances,
            target_coord=target_coord,
            k_neighbors=k_neighbors,
            max_per_sector=max_per_sector,
            min_sectors=min_sectors,
            max_per_drillhole=max_per_drillhole,
            sample_hole_ids=candidate_hole_ids,
        )
        if len(idx_sel) == 0:
            return float(mean), float(total_sill)
        c_coords = candidate_coords[idx_sel]
        c_vals = candidate_values[idx_sel]
        d_m = d_sel
    else:
        take = min(k_neighbors, len(candidate_coords))
        c_coords = candidate_coords[:take]
        c_vals = candidate_values[:take]
        d_m = candidate_distances[:take]

    # Exact collocation check
    if d_m[0] < 1e-6:
        return float(c_vals[0]), 0.0

    # Deduplicate collocated samples (e.g., duplicate drillholes)
    coords_m, vals_m, h_matrix, d_m_new = _deduplicate_collocated_samples(
        c_coords, c_vals, target_point=target_coord
    )
    if d_m_new is not None:
        d_m = d_m_new
    k_m = len(coords_m)

    if d_m[0] < 1e-6:
        return float(vals_m[0]), 0.0

    # Build covariance matrix K_m of shape (k_m, k_m)
    K_m = _theoretical_covariance(h_matrix, variogram_model, nugget, sill, range_param)
    K_m[np.diag_indices(k_m)] += 1e-9

    # Build sample-to-target covariance vector k0_m of shape (k_m,)
    k0_m = _theoretical_covariance(d_m, variogram_model, nugget, sill, range_param)

    # Solve linear system K_m * lambda_m = k0_m
    try:
        weights_m = np.linalg.solve(K_m, k0_m)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Singular covariance matrix encountered during Simple Kriging; "
            "falling back to stationary prior mean and total variance.",
            UserWarning,
            stacklevel=2,
        )
        return float(mean), float(total_sill)

    if clip_negative_weights:
        weights_m, _ = _clip_and_renormalize_weights(weights_m, preserve_sum=True)

    # Estimate: Z*_SK = mean + sum_i lambda_i * (Z_i - mean)
    est_val = mean + np.sum(weights_m * (vals_m - mean))

    # Variance: sigma_SK^2 = C(0) - sum_i lambda_i * k0_i (or quadratic formula if clipped)
    if clip_negative_weights:
        var_val = _compute_quadratic_estimation_variance(
            total_sill, weights_m, K_m, k0_m
        )
    else:
        var_val = total_sill - np.sum(weights_m * k0_m)

    return float(est_val), max(0.0, float(var_val))


def _solve_local_simple_kriging_block(
    block_centroid: np.ndarray,
    disc_offsets: np.ndarray,
    candidate_coords: np.ndarray,
    candidate_values: np.ndarray,
    candidate_distances: np.ndarray,
    candidate_hole_ids: Optional[Sequence[Any]] = None,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    sill: float = 1.0,
    range_param: float = 100.0,
    c_vv: float = 1.0,
    k_neighbors: int = 16,
    max_per_sector: Optional[int] = None,
    min_sectors: int = 1,
    max_per_drillhole: Optional[int] = None,
    clip_negative_weights: bool = False,
    mean: float = 0.0,
    min_samples: int = 1,
) -> Tuple[float, float]:
    """Solves Simple Kriging for a single 3D mining block given candidate conditioning samples and discretization offsets.

    Parameters
    ----------
    block_centroid : np.ndarray
        Centroid coordinates of the mining block of shape (3,).
    disc_offsets : np.ndarray
        Discretization point offsets from block centroid of shape (M, 3).
    candidate_coords : np.ndarray
        Coordinates of candidate conditioning samples of shape (k_cand, 3).
    candidate_values : np.ndarray
        Values (grades) of candidate samples of shape (k_cand,).
    candidate_distances : np.ndarray
        Distances from block_centroid to candidates of shape (k_cand,).
    candidate_hole_ids : Sequence[Any], optional
        Drillhole trace IDs for candidates.
    variogram_model : str, default "spherical"
        Variogram model name.
    nugget : float, default 0.0
        Nugget variance c0.
    sill : float, default 1.0
        Partial sill variance c.
    range_param : float, default 100.0
        Practical range a.
    c_vv : float, default 1.0
        Within-block covariance C_bar(V, V).
    k_neighbors : int, default 16
        Maximum neighbors to select.
    max_per_sector : int, optional
        Maximum samples per angular sector.
    min_sectors : int, default 1
        Minimum active sectors required.
    max_per_drillhole : int, optional
        Maximum samples per drillhole trace.
    clip_negative_weights : bool, default False
        Whether to clip negative weights and renormalize.
    mean : float, default 0.0
        Known stationary mean.
    min_samples : int, default 1
        Minimum samples required to condition the block.

    Returns
    -------
    est_val : float
        Simple Kriging block estimate Z*(V).
    var_val : float
        Simple Kriging block variance sigma_SK^2(V).
    """
    if len(candidate_coords) < min_samples:
        return float(mean), float(c_vv)

    has_sector_search = (
        (max_per_sector is not None)
        or (max_per_drillhole is not None)
        or (min_sectors > 1)
    )

    if has_sector_search:
        idx_sel, _ = _filter_neighborhood_sectors(
            candidate_coords=candidate_coords,
            candidate_indices=np.arange(len(candidate_coords)),
            candidate_distances=candidate_distances,
            target_coord=block_centroid,
            k_neighbors=k_neighbors,
            max_per_sector=max_per_sector,
            min_sectors=min_sectors,
            max_per_drillhole=max_per_drillhole,
            sample_hole_ids=candidate_hole_ids,
        )
        if len(idx_sel) < min_samples:
            return float(mean), float(c_vv)
        c_coords = candidate_coords[idx_sel]
        c_vals = candidate_values[idx_sel]
    else:
        take = min(k_neighbors, len(candidate_coords))
        c_coords = candidate_coords[:take]
        c_vals = candidate_values[:take]

    # Deduplicate collocated samples (e.g., duplicate drillholes or twin assays)
    coords_m, vals_m, h_matrix, _ = _deduplicate_collocated_samples(
        c_coords, c_vals, target_point=block_centroid
    )
    k_m = len(coords_m)
    if k_m < min_samples:
        return float(mean), float(c_vv)

    # Compute Sample-to-Block covariance vector k0_m of shape (k_m,)
    block_points = block_centroid + disc_offsets  # shape (M, 3)
    sample_to_disc_diffs = coords_m[:, None, :] - block_points[None, :, :]
    sample_to_disc_dists = np.linalg.norm(sample_to_disc_diffs, axis=2)
    sample_to_disc_covs = _theoretical_covariance(
        sample_to_disc_dists, variogram_model, nugget, sill, range_param
    )
    k0_block = np.mean(sample_to_disc_covs, axis=1)

    # Build Sample-to-Sample covariance matrix K_m of shape (k_m, k_m)
    K_m = _theoretical_covariance(h_matrix, variogram_model, nugget, sill, range_param)
    K_m[np.diag_indices(k_m)] += 1e-9

    # Solve linear system K_m * lambda = k0_block
    try:
        weights_m = np.linalg.solve(K_m, k0_block)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Singular covariance matrix encountered during Block Simple Kriging; "
            "falling back to stationary prior mean and within-block variance.",
            UserWarning,
            stacklevel=2,
        )
        return float(mean), float(c_vv)

    if clip_negative_weights:
        weights_m, _ = _clip_and_renormalize_weights(weights_m, preserve_sum=True)
        raw_variance = _compute_quadratic_estimation_variance(
            c_vv, weights_m, K_m, k0_block
        )
    else:
        raw_variance = c_vv - np.sum(weights_m * k0_block)

    residual = vals_m - mean
    est_val = mean + np.sum(weights_m * residual)
    var_val = max(0.0, float(raw_variance))

    return float(est_val), float(var_val)


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
    max_per_sector: Optional[int] = None,
    min_sectors: int = 1,
    max_per_drillhole: Optional[int] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
    max_per_octant: Optional[int] = None,
    max_per_quadrant: Optional[int] = None,
    clip_negative_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple Kriging (SK) grid interpolation with known stationary mean.

    Supports geological domain boundaries, 2D/3D spatial anisotropy, quadrant/octant
    search limits, and drillhole trace caps (SME Handbook §4.5, Rule 6 & Rule 7) (TODO: Manually Verify):
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
    max_per_sector : int, optional
        Maximum conditioning samples allowed per angular sector (quadrant in 2D, octant in 3D).
        Adheres to SME Mining Engineering Handbook §4.5 Rule 7 (TODO: Manually Verify).
    min_sectors : int, default 1
        Minimum number of distinct angular sectors that must contain samples for a grid node
        to be estimated.
    max_per_drillhole : int, optional
        Maximum conditioning samples allowed per drillhole trace. Adheres to SME Handbook
        §4.5 Rule 6 (TODO: Manually Verify).
    sample_hole_ids : Sequence[Any], optional
        Drillhole identifiers corresponding to conditioning samples.
    max_per_octant : int, optional
        Alias for max_per_sector in 3D.
    max_per_quadrant : int, optional
        Alias for max_per_sector in 2D.
    clip_negative_weights : bool, default False
        If True, clips negative kriging weights to zero and rescales positive weights
        to preserve the original sample weight sum (preserving the weight assigned to
        the global mean). Estimation variance is computed via the universal quadratic formula.

    Returns
    -------
    estimated_grades : np.ndarray
        Simple Kriging grade estimates of shape (M,).
    kriging_variance : np.ndarray
        Estimation variance sigma_SK^2 of shape (M,).
    """
    sec_max = (
        max_per_sector
        if max_per_sector is not None
        else (max_per_octant if max_per_octant is not None else max_per_quadrant)
    )
    if min_sectors < 1:
        raise ValueError(f"min_sectors must be >= 1, got {min_sectors}")
    if sec_max is not None and sec_max <= 0:
        raise ValueError(f"max_per_sector must be positive, got {sec_max}")
    if max_per_drillhole is not None and max_per_drillhole <= 0:
        raise ValueError(f"max_per_drillhole must be positive, got {max_per_drillhole}")
    if sample_hole_ids is not None and len(sample_hole_ids) != len(samples_xy):
        raise ValueError(
            f"sample_hole_ids length ({len(sample_hole_ids)}) must match samples_xy ({len(samples_xy)})."
        )

    s_dom, g_dom = _validate_domain_pairing(
        sample_domains, grid_domains, len(samples_xy), len(grid_points)
    )

    # Handle domain-segregated estimation
    if s_dom is not None and g_dom is not None:
        estimates = np.full(len(grid_points), np.nan)
        variances = np.full(len(grid_points), np.nan)

        for dom in np.unique(g_dom):
            g_mask = g_dom == dom
            s_mask = s_dom == dom
            dom_mean = _resolve_domain_parameter(mean, dom)
            dom_sill = _resolve_domain_parameter(sill, dom)
            dom_range = _resolve_domain_parameter(range_param, dom)
            dom_nugget = _resolve_domain_parameter(nugget, dom)
            dom_aniso_ranges = _resolve_domain_parameter(anisotropy_ranges, dom)
            dom_aniso_angles = _resolve_domain_parameter(anisotropy_angles, dom)

            if not np.any(s_mask):
                estimates[g_mask] = dom_mean
                variances[g_mask] = dom_sill + dom_nugget
                continue

            sub_hole_ids = (
                np.asarray(sample_hole_ids)[s_mask].tolist()
                if sample_hole_ids is not None
                else None
            )
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
                max_per_sector=sec_max,
                min_sectors=min_sectors,
                max_per_drillhole=max_per_drillhole,
                sample_hole_ids=sub_hole_ids,
                clip_negative_weights=clip_negative_weights,
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
    has_sector_search = (
        (sec_max is not None) or (max_per_drillhole is not None) or (min_sectors > 1)
    )
    if has_sector_search:
        k_query = min(len(samples_xy), max(k_neighbors * 8, 128))
    else:
        k_query = min(k_neighbors, len(samples_xy))

    tree = KDTree(samples_t)
    upper_bound = max_radius if max_radius is not None else float("inf")
    distances, indices = tree.query(grid_t, k=k_query, distance_upper_bound=upper_bound)

    if k_query == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    # 2. Solve Simple Kriging system for each target point via shared kernel
    for m in range(n_targets):
        valid_mask = np.isfinite(distances[m]) & (distances[m] <= upper_bound)
        if not np.any(valid_mask):
            continue  # No samples within range: remains prior mean and total sill

        c_idx = indices[m][valid_mask]
        c_dist = distances[m][valid_mask]
        c_coords = samples_t[c_idx]
        c_vals = sample_grades[c_idx]
        c_holes = (
            [sample_hole_ids[i] for i in c_idx] if sample_hole_ids is not None else None
        )

        est_val, var_val = _solve_local_simple_kriging_grid(
            target_coord=grid_t[m],
            candidate_coords=c_coords,
            candidate_values=c_vals,
            candidate_distances=c_dist,
            candidate_hole_ids=c_holes,
            variogram_model=variogram_model,
            nugget=base_nugget,
            sill=base_sill,
            range_param=eff_range,
            k_neighbors=k_neighbors,
            max_per_sector=sec_max,
            min_sectors=min_sectors,
            max_per_drillhole=max_per_drillhole,
            clip_negative_weights=clip_negative_weights,
            mean=base_mean,
        )
        estimates[m] = est_val
        variances[m] = var_val

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
    max_per_sector: Optional[int] = None,
    min_sectors: int = 1,
    max_per_drillhole: Optional[int] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
    max_per_octant: Optional[int] = None,
    max_per_quadrant: Optional[int] = None,
    clip_negative_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary Kriging (OK) grid interpolation with unknown local mean.

    Supports geological domain boundaries, 2D/3D spatial anisotropy, quadrant/octant
    search limits, and drillhole trace caps (SME Handbook §4.5, Rule 6 & Rule 7) (TODO: Manually Verify):
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
    max_per_sector : int, optional
        Maximum conditioning samples allowed per angular sector (quadrant in 2D, octant in 3D).
        Adheres to SME Mining Engineering Handbook §4.5 Rule 7 (TODO: Manually Verify).
    min_sectors : int, default 1
        Minimum number of distinct angular sectors that must contain samples for a grid node
        to be estimated.
    max_per_drillhole : int, optional
        Maximum conditioning samples allowed per drillhole trace. Adheres to SME Handbook
        §4.5 Rule 6 (TODO: Manually Verify).
    sample_hole_ids : Sequence[Any], optional
        Drillhole identifiers corresponding to conditioning samples.
    max_per_octant : int, optional
        Alias for max_per_sector in 3D.
    max_per_quadrant : int, optional
        Alias for max_per_sector in 2D.
    clip_negative_weights : bool, default False
        If True, clips negative kriging weights to zero and renormalizes positive weights
        so they sum to 1.0 (Deutsch, 1996; Isaaks & Srivastava, 1989). Grid estimation
        variance is then computed using the quadratic error variance formulation.

    Returns
    -------
    estimated_grades : np.ndarray
        Ordinary Kriging grade estimates of shape (M,).
    kriging_variance : np.ndarray
        Estimation variance sigma_OK^2 of shape (M,).
    """
    sec_max = (
        max_per_sector
        if max_per_sector is not None
        else (max_per_octant if max_per_octant is not None else max_per_quadrant)
    )
    if min_sectors < 1:
        raise ValueError(f"min_sectors must be >= 1, got {min_sectors}")
    if sec_max is not None and sec_max <= 0:
        raise ValueError(f"max_per_sector must be positive, got {sec_max}")
    if max_per_drillhole is not None and max_per_drillhole <= 0:
        raise ValueError(f"max_per_drillhole must be positive, got {max_per_drillhole}")
    if sample_hole_ids is not None and len(sample_hole_ids) != len(samples_xy):
        raise ValueError(
            f"sample_hole_ids length ({len(sample_hole_ids)}) must match samples_xy ({len(samples_xy)})."
        )

    # Handle domain-segregated estimation
    if sample_domains is not None or grid_domains is not None:
        s_dom, g_dom = _validate_domain_pairing(
            sample_domains, grid_domains, len(samples_xy), len(grid_points)
        )

        estimates = np.full(len(grid_points), np.nan)
        variances = np.full(len(grid_points), np.nan)

        for dom in np.unique(g_dom):
            g_mask = g_dom == dom
            s_mask = s_dom == dom
            dom_sill = _resolve_domain_parameter(sill, dom, sill)
            dom_range = _resolve_domain_parameter(range_param, dom, range_param)
            dom_nugget = _resolve_domain_parameter(nugget, dom, nugget)
            dom_aniso_ranges = _resolve_domain_parameter(
                anisotropy_ranges, dom, anisotropy_ranges
            )
            dom_aniso_angles = _resolve_domain_parameter(
                anisotropy_angles, dom, anisotropy_angles
            )

            if not np.any(s_mask):
                continue

            sub_hole_ids = (
                np.asarray(sample_hole_ids)[s_mask].tolist()
                if sample_hole_ids is not None
                else None
            )
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
                max_per_sector=sec_max,
                min_sectors=min_sectors,
                max_per_drillhole=max_per_drillhole,
                sample_hole_ids=sub_hole_ids,
                clip_negative_weights=clip_negative_weights,
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
    has_sector_search = (
        (sec_max is not None) or (max_per_drillhole is not None) or (min_sectors > 1)
    )
    if has_sector_search:
        k_query = min(len(samples_xy), max(k_neighbors * 8, 128))
    else:
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

        if has_sector_search:
            idx_m, d_m = _filter_neighborhood_sectors(
                candidate_coords=samples_t[indices[m][valid_mask]],
                candidate_indices=indices[m][valid_mask],
                candidate_distances=distances[m][valid_mask],
                target_coord=grid_t[m],
                k_neighbors=k_neighbors,
                max_per_sector=sec_max,
                min_sectors=min_sectors,
                max_per_drillhole=max_per_drillhole,
                sample_hole_ids=sample_hole_ids,
            )
            if len(idx_m) == 0:
                continue
        else:
            d_m = distances[m][valid_mask]
            idx_m = indices[m][valid_mask]

        # Exact collocation: if target lies directly on a sample point
        if d_m[0] < 1e-6:
            estimates[m] = sample_grades[idx_m[0]]
            variances[m] = 0.0
            continue

        # Build sample-to-sample covariance matrix K_m of shape (k_m, k_m)
        coords_m = samples_t[idx_m]
        grades_m = sample_grades[idx_m]

        # Deduplicate identical sample coordinates (e.g. twin drillholes or duplicate assays)
        coords_m, grades_m, h_matrix, d_target_m = _deduplicate_collocated_samples(
            coords_m, grades_m, target_point=grid_t[m]
        )
        if d_target_m is not None:
            d_m = d_target_m
        k_m = len(coords_m)

        K_m = _theoretical_covariance(
            h_matrix, variogram_model, base_nugget, base_sill, eff_range
        )
        K_m[np.diag_indices(k_m)] += 1e-9  # Regularizer to guarantee invertibility

        # Build sample-to-target covariance vector k0_m of shape (k_m,)
        k0_m = _theoretical_covariance(
            d_m, variogram_model, base_nugget, base_sill, eff_range
        )

        # Solve linear system: K_aug * [lambda; mu_lagrange] = k0_aug
        weights_m, mu_lagrange = _solve_ordinary_kriging_system(
            K_m, k0_m, clip_negative_weights=clip_negative_weights
        )

        # Ordinary Kriging estimate: Z*_OK = sum_i lambda_i * Z(x_i)
        estimates[m] = np.sum(weights_m * grades_m)

        # Ordinary Kriging variance:
        if clip_negative_weights:
            # Universal quadratic estimation variance formula for non-optimal linear weights
            var_val = _compute_quadratic_estimation_variance(
                total_sill, weights_m, K_m, k0_m
            )
        else:
            # Standard Ordinary Kriging variance: sigma_OK^2 = C(0) - sum_i lambda_i * k0_i - mu_lagrange
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


def compute_block_dispersion_variance(
    block_size: Tuple[float, float, float],
    discretization: Tuple[int, int, int],
    variogram_model: str,
    sill: float,
    range_param: Union[float, Sequence[float], Mapping[str, float]],
    nugget: float = 0.0,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> float:
    """Computes block self-covariance / dispersion variance C_bar(V, V) via internal point discretization.

    In accordance with geostatistical support effect principles (CIM MRMR Guidelines §6.9,
    SME Mining Engineering Handbook §4.5, and JORC Code Table 1) (TODO: Manually Verify),
    evaluates average point-to-point covariance within an SMU block:
    C_bar(V, V) = (1 / M^2) * sum_{i=1}^M sum_{j=1}^M C(u_i - u_j).

    Parameters
    ----------
    block_size : tuple of (float, float, float)
        Physical SMU block dimensions (dx, dy, dz).
    discretization : tuple of (int, int, int)
        Internal point discretization grid (nx, ny, nz) per block.
    variogram_model : str
        Spatial covariance model ("spherical", "exponential", "gaussian").
    sill : float
        Partial sill variance.
    range_param : float, Sequence[float], or Mapping
        Spatial correlation range.
    nugget : float, default 0.0
        Nugget variance.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional search/variogram ranges in 3D.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth, dip, plunge) in 3D.

    Returns
    -------
    float
        Block dispersion variance / self-covariance C_bar(V, V).
    """
    disc_offsets = _generate_block_discretization_offsets(discretization, block_size)
    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (
            range_param if isinstance(range_param, (Sequence, Mapping)) else range_param
        )
    )
    disc_offsets_t, eff_range = transform_anisotropic_coordinates(
        disc_offsets, ranges=eff_ranges, angles=anisotropy_angles
    )
    internal_diffs = disc_offsets_t[:, None, :] - disc_offsets_t[None, :, :]
    internal_dists = np.linalg.norm(internal_diffs, axis=2)
    internal_covs = _theoretical_covariance(
        internal_dists, variogram_model, nugget, sill, eff_range
    )
    c_vv = float(np.mean(internal_covs))
    return max(0.0, c_vv)


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
    max_per_sector: Optional[int] = None,
    min_sectors: int = 1,
    max_per_drillhole: Optional[int] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
    max_per_octant: Optional[int] = None,
    max_per_quadrant: Optional[int] = None,
    clip_negative_weights: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Estimates block grades using 3D Ordinary Block Kriging with internal discretization.

    Supports geological domain boundaries, 3D spatial anisotropy, quadrant/octant search limits,
    and drillhole trace caps (SME Handbook §4.5, Rule 6 & Rule 7; CIM MRMR §6.8/§6.9) (TODO: Manually Verify):
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
    max_per_sector : int, optional
        Maximum conditioning samples allowed per angular sector (quadrant in 2D, octant in 3D).
        Adheres to SME Mining Engineering Handbook §4.5 Rule 7 (TODO: Manually Verify).
    min_sectors : int, default 1
        Minimum number of distinct angular sectors that must contain samples for a block
        to be estimated.
    max_per_drillhole : int, optional
        Maximum conditioning samples allowed per drillhole trace. Adheres to SME Handbook
        §4.5 Rule 6 (TODO: Manually Verify).
    sample_hole_ids : Sequence[Any], optional
        Drillhole identifiers corresponding to conditioning samples.
    max_per_octant : int, optional
        Alias for max_per_sector in 3D.
    max_per_quadrant : int, optional
        Alias for max_per_sector in 2D.
    clip_negative_weights : bool, default False
        If True, clips negative kriging weights to zero and renormalizes positive weights
        so they sum to 1.0 (Deutsch, 1996; Isaaks & Srivastava, 1989). Block estimation
        variance is then computed using the quadratic error variance formulation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, np.ndarray]
        (block_estimates, block_variances, block_dispersion_variance, lagrange_multipliers).
    """
    sec_max = (
        max_per_sector
        if max_per_sector is not None
        else (max_per_octant if max_per_octant is not None else max_per_quadrant)
    )
    if min_sectors < 1:
        raise ValueError(f"min_sectors must be >= 1, got {min_sectors}")
    if sec_max is not None and sec_max <= 0:
        raise ValueError(f"max_per_sector must be positive, got {sec_max}")
    if max_per_drillhole is not None and max_per_drillhole <= 0:
        raise ValueError(f"max_per_drillhole must be positive, got {max_per_drillhole}")
    if sample_hole_ids is not None and len(sample_hole_ids) != len(samples_xyz):
        raise ValueError(
            f"sample_hole_ids length ({len(sample_hole_ids)}) must match samples_xyz ({len(samples_xyz)})."
        )

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
            dom_sill = _resolve_domain_parameter(sill, dom, sill)
            dom_range = _resolve_domain_parameter(range_param, dom, range_param)
            dom_nugget = _resolve_domain_parameter(nugget, dom, nugget)
            dom_aniso_ranges = _resolve_domain_parameter(
                anisotropy_ranges, dom, anisotropy_ranges
            )
            dom_aniso_angles = _resolve_domain_parameter(
                anisotropy_angles, dom, anisotropy_angles
            )

            if not np.any(s_mask):
                continue

            sub_bm = block_model[b_mask].copy()
            sub_hole_ids = (
                np.asarray(sample_hole_ids)[s_mask].tolist()
                if sample_hole_ids is not None
                else None
            )
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
                max_per_sector=sec_max,
                min_sectors=min_sectors,
                max_per_drillhole=max_per_drillhole,
                sample_hole_ids=sub_hole_ids,
                clip_negative_weights=clip_negative_weights,
            )
            block_estimates[b_mask] = est_dom
            block_variances[b_mask] = var_dom
            lagrange_multipliers[b_mask] = lag_dom
            overall_disp_var = disp_dom

        return (
            block_estimates,
            block_variances,
            overall_disp_var,
            lagrange_multipliers,
        )

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

    c_vv = compute_block_dispersion_variance(
        block_size=(dx, dy, dz),
        discretization=discretization,
        variogram_model=variogram_model,
        sill=base_sill,
        range_param=base_range,
        nugget=base_nugget,
        anisotropy_ranges=anisotropy_ranges,
        anisotropy_angles=anisotropy_angles,
    )
    block_dispersion_var = c_vv

    disc_offsets = _generate_block_discretization_offsets(discretization, (dx, dy, dz))
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

    has_sector_search = (
        (sec_max is not None) or (max_per_drillhole is not None) or (min_sectors > 1)
    )
    if has_sector_search:
        k_query = min(len(samples_xyz), max(k_neighbors * 8, 128))
    else:
        k_query = min(k_neighbors, len(samples_xyz))

    tree = KDTree(samples_xyz_t)
    distances, indices = tree.query(
        block_coords_t,
        k=k_query,
        distance_upper_bound=upper_bound,
    )

    if k_query == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    neg_weight_block_count = 0

    # 6. Point-to-Block Covariance & Ordinary Kriging Solution per Block
    for b in range(n_blocks):
        # a. Filter valid neighbors within upper_bound search radius
        valid_mask = np.isfinite(distances[b]) & (distances[b] <= upper_bound)
        if not np.any(valid_mask) or np.sum(valid_mask) < min_samples:
            continue  # Insufficient sample support: remains NaN and c_vv

        if has_sector_search:
            active_indices, _ = _filter_neighborhood_sectors(
                candidate_coords=samples_xyz_t[indices[b][valid_mask]],
                candidate_indices=indices[b][valid_mask],
                candidate_distances=distances[b][valid_mask],
                target_coord=block_coords_t[b],
                k_neighbors=k_neighbors,
                max_per_sector=sec_max,
                min_sectors=min_sectors,
                max_per_drillhole=max_per_drillhole,
                sample_hole_ids=sample_hole_ids,
            )
            if len(active_indices) < min_samples:
                continue
        else:
            active_indices = indices[b][valid_mask]
        coords_active = samples_xyz_t[active_indices]  # shape (k, 3)
        grades_active = sample_grades[active_indices]  # shape (k,)

        # Deduplicate identical sample coordinates (e.g. twin drillholes or duplicate assays)
        coords_active, grades_active, h_sample_matrix, _ = (
            _deduplicate_collocated_samples(
                coords_active, grades_active, target_point=block_coords_t[b]
            )
        )
        k_active = len(coords_active)
        if k_active < min_samples:
            continue

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
        K_mat = _theoretical_covariance(
            h_sample_matrix, variogram_model, base_nugget, base_sill, eff_range
        )
        # Regularize diagonal to avoid singularity from collocated/close samples
        K_mat[np.diag_indices(k_active)] += 1e-9

        try:
            weights, mu = _solve_ordinary_kriging_system(
                K_mat, k0_block, clip_negative_weights=clip_negative_weights
            )
            if np.any(weights < -0.01):
                neg_weight_block_count += 1
            # Block Grade Estimate: Z*(V) = sum(lambda_i * Z_i)
            block_estimates[b] = float(np.sum(weights * grades_active))
            # Block Kriging Variance:
            if clip_negative_weights:
                # Universal quadratic estimation variance formula for non-optimal linear weights
                raw_variance = _compute_quadratic_estimation_variance(
                    c_vv, weights, K_mat, k0_block
                )
            else:
                # Standard Ordinary Kriging variance: sigma_OK^2 = C_bar(V, V) - sum(lambda_i * C_bar(x_i, V)) - mu
                raw_variance = c_vv - np.sum(weights * k0_block) - mu
            block_variances[b] = max(0.0, float(raw_variance))
            lagrange_multipliers[b] = float(mu)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            continue

    # Covariance formulation duality: mu_C = -mu_gamma. Under stable estimation, mu_C <= 0
    # (corresponding to variogram mu_gamma >= 0 and SoR <= 1.0).
    # When mu_C > 1e-6 (variogram mu_gamma < -1e-6), the unbiasedness constraint is active
    # against severe screen effects / clustering, forcing SoR > 1.0.
    unstable_mu_mask = np.isfinite(lagrange_multipliers) & (lagrange_multipliers > 1e-6)
    unstable_mu_count = int(np.sum(unstable_mu_mask))
    if unstable_mu_count > 0:
        warnings.warn(
            f"Kriging Neighborhood Instability: {unstable_mu_count} block(s) exhibit anomalous positive Lagrange "
            f"multipliers in covariance space (maximum mu={float(np.nanmax(lagrange_multipliers)):.4f}), equivalent to a "
            f"negative variogram Lagrange multiplier (mu_gamma < 0). This indicates that the unbiasedness constraint is active "
            f"against severe screen effects or clustered sampling geometries, driving the Slope of Regression above 1.0.",
            UserWarning,
            stacklevel=2,
        )

    if neg_weight_block_count > 0:
        warnings.warn(
            f"Negative Kriging Weights: {neg_weight_block_count} block(s) receive negative kriging weights (lambda < -0.01). "
            f"In Ordinary Kriging, negative weights occur due to the screen effect from clustered or collinear drillhole samples "
            f"and can produce physically unrealistic grade extrapolations.",
            UserWarning,
            stacklevel=2,
        )

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
    max_per_sector: Optional[int] = None,
    min_sectors: int = 1,
    max_per_drillhole: Optional[int] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
    max_per_octant: Optional[int] = None,
    max_per_quadrant: Optional[int] = None,
    clip_negative_weights: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimates block grades using 3D Simple Block Kriging with a known stationary mean.

    Supports geological domain boundaries, 3D spatial anisotropy, quadrant/octant search limits,
    and drillhole trace caps (SME Handbook §4.5, Rule 6 & Rule 7; CIM MRMR §6.8/§6.9) (TODO: Manually Verify):
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
    max_per_sector : int, optional
        Maximum conditioning samples allowed per angular sector (quadrant in 2D, octant in 3D).
        Adheres to SME Mining Engineering Handbook §4.5 Rule 7 (TODO: Manually Verify).
    min_sectors : int, default 1
        Minimum number of distinct angular sectors that must contain samples for a block
        to be estimated.
    max_per_drillhole : int, optional
        Maximum conditioning samples allowed per drillhole trace. Adheres to SME Handbook
        §4.5 Rule 6 (TODO: Manually Verify).
    sample_hole_ids : Sequence[Any], optional
        Drillhole identifiers corresponding to conditioning samples.
    max_per_octant : int, optional
        Alias for max_per_sector in 3D.
    max_per_quadrant : int, optional
        Alias for max_per_sector in 2D.
    clip_negative_weights : bool, default False
        Whether to clip negative Simple Kriging weights to 0.0 and renormalize them
        preserving the total sample weight sum, recalculating estimation variance
        via the universal quadratic error variance formula (Deutsch, 1996).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        (block_estimates, block_variances, block_dispersion_variance).
    """
    sec_max = (
        max_per_sector
        if max_per_sector is not None
        else (max_per_octant if max_per_octant is not None else max_per_quadrant)
    )
    if min_sectors < 1:
        raise ValueError(f"min_sectors must be >= 1, got {min_sectors}")
    if sec_max is not None and sec_max <= 0:
        raise ValueError(f"max_per_sector must be positive, got {sec_max}")
    if max_per_drillhole is not None and max_per_drillhole <= 0:
        raise ValueError(f"max_per_drillhole must be positive, got {max_per_drillhole}")
    if sample_hole_ids is not None and len(sample_hole_ids) != len(samples_xyz):
        raise ValueError(
            f"sample_hole_ids length ({len(sample_hole_ids)}) must match samples_xyz ({len(samples_xyz)})."
        )

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
            dom_mean = _resolve_domain_parameter(mean, dom, mean)
            dom_sill = _resolve_domain_parameter(sill, dom, sill)
            dom_range = _resolve_domain_parameter(range_param, dom, range_param)
            dom_nugget = _resolve_domain_parameter(nugget, dom, nugget)
            dom_aniso_ranges = _resolve_domain_parameter(
                anisotropy_ranges, dom, anisotropy_ranges
            )
            dom_aniso_angles = _resolve_domain_parameter(
                anisotropy_angles, dom, anisotropy_angles
            )

            if not np.any(s_mask):
                continue

            sub_bm = block_model[b_mask].copy()
            sub_hole_ids = (
                np.asarray(sample_hole_ids)[s_mask].tolist()
                if sample_hole_ids is not None
                else None
            )
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
                max_per_sector=sec_max,
                min_sectors=min_sectors,
                max_per_drillhole=max_per_drillhole,
                sample_hole_ids=sub_hole_ids,
                clip_negative_weights=clip_negative_weights,
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

    dx = float(block_model["dx"].iloc[0])
    dy = float(block_model["dy"].iloc[0])
    dz = float(block_model["dz"].iloc[0])
    block_size = (dx, dy, dz)

    c_vv = compute_block_dispersion_variance(
        block_size=block_size,
        discretization=discretization,
        variogram_model=variogram_model,
        sill=base_sill,
        range_param=base_range,
        nugget=base_nugget,
        anisotropy_ranges=anisotropy_ranges,
        anisotropy_angles=anisotropy_angles,
    )
    block_dispersion_var = max(0.0, c_vv)

    disc_offsets = _generate_block_discretization_offsets(discretization, block_size)
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

    # 5. Neighbor Search Setup & Spatial Query
    has_sector_search = (
        (sec_max is not None) or (max_per_drillhole is not None) or (min_sectors > 1)
    )
    if has_sector_search:
        k_query = min(len(samples_xyz), max(k_neighbors * 8, 128))
    else:
        k_query = min(k_neighbors, len(samples_xyz))

    tree = KDTree(samples_xyz_t)
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

        c_idx = indices[b][valid_mask]
        c_dist = distances[b][valid_mask]
        c_coords = samples_xyz_t[c_idx]
        c_vals = sample_grades[c_idx]
        c_holes = (
            [sample_hole_ids[i] for i in c_idx] if sample_hole_ids is not None else None
        )

        est_val, var_val = _solve_local_simple_kriging_block(
            block_centroid=block_coords_t[b],
            disc_offsets=disc_offsets_t,
            candidate_coords=c_coords,
            candidate_values=c_vals,
            candidate_distances=c_dist,
            candidate_hole_ids=c_holes,
            variogram_model=variogram_model,
            nugget=base_nugget,
            sill=base_sill,
            range_param=eff_range,
            c_vv=c_vv,
            k_neighbors=k_neighbors,
            max_per_sector=sec_max,
            min_sectors=min_sectors,
            max_per_drillhole=max_per_drillhole,
            clip_negative_weights=clip_negative_weights,
            mean=base_mean,
            min_samples=min_samples,
        )
        block_estimates[b] = est_val
        block_variances[b] = var_val

    return block_estimates, block_variances, block_dispersion_var


def plot_grade_tonnage_curve(
    gt_data: Union[pd.DataFrame, Mapping[str, pd.DataFrame]],
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    title: str = "Grade–Tonnage Sensitivity Curve",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (8.5, 5.0),
    show_metal: bool = False,
    realization_tables: Optional[Sequence[pd.DataFrame]] = None,
) -> Union[Tuple[plt.Figure, plt.Axes], Dict[str, Tuple[plt.Figure, plt.Axes]]]:
    """Plots standard dual-axis Grade–Tonnage sensitivity curves.

    Conforms to NI 43-101 (Item 14) and JORC Table 1 mineral reporting standards (TODO: Manually Verify):
    - Primary Y-axis (left): Ore Tonnage above cutoff (monotonically decreasing).
    - Secondary Y-axis (right): Average Grade above cutoff (monotonically increasing).
    - If a dictionary of models is provided, generates an individual, clean dual-axis
      plot for each model (avoiding multi-line overlapping clutter).
    - If `realization_tables` is provided, plots individual simulation realization trajectories
      and shaded P10–P90 uncertainty envelopes for both tonnage and grade.

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
    realization_tables : Sequence[pd.DataFrame], optional
        List of Grade-Tonnage DataFrames for individual conditional realizations,
        used to visualize joint grade-tonnage uncertainty.

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
                realization_tables=realization_tables,
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

    has_realizations = realization_tables is not None and len(realization_tables) > 0
    extra_handles = []
    extra_labels = []

    if has_realizations:
        r_tonnes = []
        r_grades = []
        for df_r in realization_tables:
            r_t = df_r.loc[cutoffs, "ore_tonnes"].to_numpy() / t_divisor
            r_g = df_r.loc[cutoffs, "ore_grade"].to_numpy()
            r_tonnes.append(r_t)
            r_grades.append(r_g)
            ax1.plot(cutoffs, r_t, color="#1f77b4", alpha=0.15, linewidth=0.8)
            ax2.plot(cutoffs, r_g, color="#d62728", alpha=0.15, linewidth=0.8)

        t_p10 = np.percentile(r_tonnes, 10, axis=0)
        t_p90 = np.percentile(r_tonnes, 90, axis=0)
        g_p10 = np.percentile(r_grades, 10, axis=0)
        g_p90 = np.percentile(r_grades, 90, axis=0)

        poly_t = ax1.fill_between(
            cutoffs,
            t_p10,
            t_p90,
            color="#1f77b4",
            alpha=0.20,
            label="Tonnage P10–P90 Range",
        )
        poly_g = ax2.fill_between(
            cutoffs,
            g_p10,
            g_p90,
            color="#d62728",
            alpha=0.20,
            label="Grade P10–P90 Range",
        )
        extra_handles.extend([poly_t, poly_g])
        extra_labels.extend(["Tonnage P10–P90 Range", "Grade P10–P90 Range"])

    t_label = (
        f"Expected Ore Tonnage ({tonnage_unit})"
        if has_realizations
        else f"Ore Tonnage ({tonnage_unit})"
    )
    g_label = (
        f"Expected Average Grade ({grade_unit})"
        if has_realizations
        else f"Average Grade ({grade_unit})"
    )

    line1 = ax1.plot(
        cutoffs,
        tonnes,
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=5,
        label=t_label,
    )
    line2 = ax2.plot(
        cutoffs,
        grades,
        color="#d62728",
        linewidth=2.5,
        marker="s",
        markersize=5,
        label=g_label,
    )

    if has_realizations:
        lines = [line1[0], extra_handles[0], line2[0], extra_handles[1]]
        labels = [t_label, extra_labels[0], g_label, extra_labels[1]]
    else:
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
        lines.append(line3[0])
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
    cell_sizes: Optional[
        Union[float, Sequence[float], Sequence[Tuple[float, float, float]]]
    ] = None,
    grade_col: str = "grade",
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = None,
    z_aspect_ratio: float = 1.0,
    domain_col: Optional[str] = None,
    n_offsets: int = 8,
    min_mean: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, float]:
    """Calculates spatial declustering weights and cell size sensitivity in 2D or 3D.

    In exploration drilling, geologists cluster drillholes preferentially in high-grade
    zones, causing naive sample averages to overestimate global deposit grade.
    Cell declustering (Journel 1983; Deutsch & Journel 1998; SME Handbook Section 4.3) (TODO: Manually Verify)
    superimposes a regular grid over the data and assigns weights inversely proportional
    to the number of samples sharing each cell.

    In 3D exploration drillholes, sampling along the borehole trajectory is typically dense
    (e.g., 1–2m assays or 5m composites) compared to wide lateral drillhole spacing (e.g., 50–100m).
    Providing `z_col` activates full 3D cell declustering with anisotropic cell dimensions
    (Lx, Ly, Lz = Lx * z_aspect_ratio), properly resolving both lateral and vertical clustering
    in accordance with CIM MRMR Best Practice Guidelines (§4.2) (TODO: Manually Verify).

    Parameters
    ----------
    drillholes : pd.DataFrame
        Table of drillholes containing spatial coordinates and grades.
    cell_sizes : float, Sequence[float], or Sequence[Tuple[float, float, float]], optional
        Single cell dimension, sequence of lateral dimensions, or sequence of (Lx, Ly, Lz) tuples.
        If None, automatically determines a comprehensive range from below the minimum
        spacing up to 3.5x the deposit extent, guaranteeing a complete U-shaped curve.
    grade_col : str, default "grade"
        Grade column name in drillholes.
    x_col : str, default "x"
        X coordinate column name (Easting).
    y_col : str, default "y"
        Y coordinate column name (Northing).
    z_col : str, optional
        Z coordinate column name (Elevation). If provided, performs 3D cell declustering.
    z_aspect_ratio : float, default 1.0
        Vertical-to-horizontal cell dimension ratio (Lz / Lx) when generating 3D cell sizes.
        For deposits drilled vertically on a 50m grid with 10m benches, z_aspect_ratio=0.20.
    domain_col : str, optional
        Geological domain column in drillholes. If provided, samples are strictly segregated
        by domain and declustered independently within their respective spatial volumes
        in compliance with CIM MRMR domain segregation rules (TODO: Manually Verify).
    n_offsets : int, default 8
        Number of origin offset increments per axis to average out boundary jumps.
        In 3D, offset counts are automatically balanced (up to 4 per axis, giving 64 shifts)
        to maintain rapid calculation.
    min_mean : bool, default True
        If True, selects the optimal cell size at the minimum declustered mean (standard
        for deposits drilled preferentially in high-grade ore). If False, selects maximum.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame, float]
        - weights: 1D array of normalized sample weights for the optimal cell size (sum to 1.0).
        - sensitivity_df: DataFrame of columns ['cell_size', 'declustered_mean', 'declustered_variance']
          (plus 'cell_size_x', 'cell_size_y', 'cell_size_z' in 3D).
        - optimal_cell_size: The chosen lateral cell dimension (float). The full 3D dimensions
          are accessible in sensitivity_df.attrs['optimal_cell_size_3d'] when in 3D.
    """
    if len(drillholes) == 0:
        raise ValueError("Drillholes DataFrame cannot be empty.")

    if x_col not in drillholes.columns:
        raise ValueError(f"X coordinate column '{x_col}' not found in DataFrame.")
    if y_col not in drillholes.columns:
        raise ValueError(f"Y coordinate column '{y_col}' not found in DataFrame.")
    if grade_col not in drillholes.columns:
        raise ValueError(f"Grade column '{grade_col}' not found in DataFrame.")
    if z_col is not None and z_col not in drillholes.columns:
        raise ValueError(f"Z coordinate column '{z_col}' not found in DataFrame.")
    if z_aspect_ratio <= 0:
        raise ValueError(
            f"z_aspect_ratio must be strictly positive, got {z_aspect_ratio}"
        )

    # Domain-segregated declustering
    if domain_col is not None:
        if domain_col not in drillholes.columns:
            raise ValueError(f"Domain column '{domain_col}' not found in DataFrame.")

        unique_domains = drillholes[domain_col].unique()
        if len(unique_domains) > 1:
            weights = np.zeros(len(drillholes), dtype=float)
            domain_dfs = []
            for dom in unique_domains:
                mask = (drillholes[domain_col] == dom).to_numpy()
                sub_df = drillholes.loc[mask]
                sub_w, sub_sens, _ = cell_declustering(
                    sub_df,
                    cell_sizes=cell_sizes,
                    grade_col=grade_col,
                    x_col=x_col,
                    y_col=y_col,
                    z_col=z_col,
                    z_aspect_ratio=z_aspect_ratio,
                    domain_col=None,
                    n_offsets=n_offsets,
                    min_mean=min_mean,
                )
                domain_frac = len(sub_df) / len(drillholes)
                weights[mask] = sub_w * domain_frac
                sub_sens_copy = sub_sens.copy()
                sub_sens_copy["domain"] = dom
                domain_dfs.append(sub_sens_copy)

            combined_sens = pd.concat(domain_dfs, ignore_index=True)
            # Weighted average per cell size across domains
            summary_rows = []
            for cs, group in combined_sens.groupby("cell_size"):
                dom_weights = [
                    len(drillholes[drillholes[domain_col] == d])
                    for d in group["domain"]
                ]
                w_mean = float(
                    np.average(group["declustered_mean"], weights=dom_weights)
                )
                w_var = float(
                    np.average(group["declustered_variance"], weights=dom_weights)
                )
                row_dict = {
                    "cell_size": cs,
                    "declustered_mean": w_mean,
                    "declustered_variance": w_var,
                }
                if "cell_size_z" in group.columns:
                    row_dict["cell_size_z"] = group["cell_size_z"].iloc[0]
                summary_rows.append(row_dict)

            summary_sens = pd.DataFrame(summary_rows)
            if min_mean:
                opt_idx = int(summary_sens["declustered_mean"].idxmin())
            else:
                opt_idx = int(summary_sens["declustered_mean"].idxmax())

            opt_cell_size = float(summary_sens.loc[opt_idx, "cell_size"])
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum

            return weights, summary_sens, opt_cell_size

    is_3d = z_col is not None
    x = np.asarray(drillholes[x_col], dtype=float)
    y = np.asarray(drillholes[y_col], dtype=float)
    z = np.asarray(drillholes[z_col], dtype=float) if is_3d else None
    grades = np.asarray(drillholes[grade_col], dtype=float)
    n_samples = len(x)

    if cell_sizes is None:
        dx = float(x.max() - x.min())
        dy = float(y.max() - y.min())
        if is_3d:
            dz = float(z.max() - z.min())
            diag = float(np.sqrt(dx**2 + dy**2 + (dz / z_aspect_ratio) ** 2))
            if n_samples > 1:
                pts = np.column_stack([x, y, z / z_aspect_ratio])
                diffs = pts[:, None, :] - pts[None, :, :]
                dists = np.linalg.norm(diffs, axis=2)
                np.fill_diagonal(dists, np.inf)
                min_dist = float(dists.min())
                min_cs = max(min_dist * 0.5, 1e-3)
            else:
                min_cs = 1.0
            max_cs = max(diag * 7.0, min_cs * 15.0)
            lateral_sizes = list(np.linspace(min_cs, max_cs, 45))
            cell_dims_list = [(cs, cs, cs * z_aspect_ratio) for cs in lateral_sizes]
        else:
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
            lateral_sizes = list(np.linspace(min_cs, max_cs, 45))
            cell_dims_list = [(cs, cs) for cs in lateral_sizes]
    elif isinstance(cell_sizes, (int, float)):
        cs = float(cell_sizes)
        lateral_sizes = [cs]
        cell_dims_list = [(cs, cs, cs * z_aspect_ratio)] if is_3d else [(cs, cs)]
    else:
        cell_dims_list = []
        lateral_sizes = []
        for item in cell_sizes:
            if isinstance(item, (tuple, list)):
                if len(item) == 3:
                    cell_dims_list.append(
                        (float(item[0]), float(item[1]), float(item[2]))
                    )
                    lateral_sizes.append(float(item[0]))
                elif len(item) == 2:
                    cell_dims_list.append((float(item[0]), float(item[1])))
                    lateral_sizes.append(float(item[0]))
                else:
                    raise ValueError(
                        f"Cell size tuple must have 2 or 3 elements, got {item}"
                    )
            else:
                cs = float(item)
                lateral_sizes.append(cs)
                if is_3d:
                    cell_dims_list.append((cs, cs, cs * z_aspect_ratio))
                else:
                    cell_dims_list.append((cs, cs))

    if is_3d:
        nx_off = max(1, min(int(n_offsets), 4))
        ny_off = max(1, min(int(n_offsets), 4))
        nz_off = max(1, min(int(n_offsets), 4))
        total_shifts = nx_off * ny_off * nz_off
        x_min, y_min, z_min = float(x.min()), float(y.min()), float(z.min())
    else:
        n_off = max(1, int(n_offsets))
        total_shifts = n_off * n_off
        x_min, y_min = float(x.min()), float(y.min())

    weights_by_cell = []
    means = []
    variances = []

    for c_dim in cell_dims_list:
        cs_x = c_dim[0]
        cs_y = c_dim[1]
        cs_z = c_dim[2] if is_3d else None
        if cs_x <= 0 or cs_y <= 0 or (is_3d and cs_z <= 0):
            raise ValueError(f"Cell dimensions must be strictly positive, got {c_dim}")

        weights_sum = np.zeros(n_samples, dtype=float)

        if is_3d:
            for ox in range(nx_off):
                shift_x = x_min - (ox / nx_off) * cs_x
                for oy in range(ny_off):
                    shift_y = y_min - (oy / ny_off) * cs_y
                    for oz in range(nz_off):
                        shift_z = z_min - (oz / nz_off) * cs_z

                        idx_x = np.floor((x - shift_x) / cs_x).astype(np.int64)
                        idx_y = np.floor((y - shift_y) / cs_y).astype(np.int64)
                        idx_z = np.floor((z - shift_z) / cs_z).astype(np.int64)

                        cell_keys = list(zip(idx_x, idx_y, idx_z))
                        counts = Counter(cell_keys)
                        n_occupied = len(counts)

                        for i, ck in enumerate(cell_keys):
                            weights_sum[i] += 1.0 / (n_occupied * counts[ck])
        else:
            for ox in range(n_off):
                shift_x = x_min - (ox / n_off) * cs_x
                for oy in range(n_off):
                    shift_y = y_min - (oy / n_off) * cs_y

                    idx_x = np.floor((x - shift_x) / cs_x).astype(np.int64)
                    idx_y = np.floor((y - shift_y) / cs_y).astype(np.int64)

                    cell_keys = list(zip(idx_x, idx_y))
                    counts = Counter(cell_keys)
                    n_occupied = len(counts)

                    for i, ck in enumerate(cell_keys):
                        weights_sum[i] += 1.0 / (n_occupied * counts[ck])

        w = weights_sum / total_shifts
        w_sum = w.sum()
        if w_sum > 0:
            w /= w_sum

        mean_val = float(np.sum(w * grades))
        var_val = float(np.sum(w * ((grades - mean_val) ** 2)))

        weights_by_cell.append(w)
        means.append(mean_val)
        variances.append(var_val)

    sens_dict = {
        "cell_size": lateral_sizes,
        "declustered_mean": means,
        "declustered_variance": variances,
    }
    if is_3d:
        sens_dict["cell_size_x"] = [c[0] for c in cell_dims_list]
        sens_dict["cell_size_y"] = [c[1] for c in cell_dims_list]
        sens_dict["cell_size_z"] = [c[2] for c in cell_dims_list]

    sensitivity_df = pd.DataFrame(sens_dict)

    if min_mean:
        opt_idx = int(sensitivity_df["declustered_mean"].idxmin())
    else:
        opt_idx = int(sensitivity_df["declustered_mean"].idxmax())

    optimal_cell_size = float(sensitivity_df.loc[opt_idx, "cell_size"])
    optimal_weights = weights_by_cell[opt_idx]

    if is_3d:
        sensitivity_df.attrs["optimal_cell_size_3d"] = cell_dims_list[opt_idx]

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


def kriging_neighborhood_analysis(
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

        unstable_mu_mask = np.isfinite(mu_arr) & (mu_arr > 1e-6)
        unstable_mu_count = int(np.sum(unstable_mu_mask))
        if unstable_mu_count > 0:
            warnings.warn(
                f"Kriging Neighborhood Instability: {unstable_mu_count} block(s) exhibit anomalous positive Lagrange "
                f"multipliers in covariance space (maximum mu={float(np.nanmax(mu_arr)):.4f}), equivalent to a "
                f"negative variogram Lagrange multiplier (mu_gamma < 0). This indicates screen effect or clustered sampling "
                f"geometry that forces the Slope of Regression above 1.0.",
                UserWarning,
                stacklevel=2,
            )

        high_sor_mask = np.isfinite(slope_of_regression) & (slope_of_regression > 1.05)
        high_sor_count = int(np.sum(high_sor_mask))
        if high_sor_count > 0:
            warnings.warn(
                f"Kriging Neighborhood Instability: {high_sor_count} block(s) exhibit Slope of Regression (SoR) "
                f"> 1.05 (maximum SoR={float(np.nanmax(slope_of_regression)):.3f}). In standard geostatistical practice "
                f"(SME Mining Engineering Handbook §4.5.8.2; Armstrong, 1998), an SoR > 1.0 indicates mathematical or geometric "
                f"instability in the kriging system (often caused by excessive sample count, clustering, or the screen effect).",
                UserWarning,
                stacklevel=2,
            )

    return kriging_efficiency, slope_of_regression


def classify_resources_by_drill_spacing(
    grid_points: Union[np.ndarray, pd.DataFrame],
    samples_xy: np.ndarray,
    max_radius_measured: float,
    max_radius_indicated: float,
    min_holes_measured: int = 3,
    min_holes_indicated: int = 2,
    max_radius_inferred: Optional[float] = None,
    is_interpolated: Optional[np.ndarray] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
    domain_col: Optional[str] = None,
    sample_domains: Optional[Sequence[Any]] = None,
    grid_domains: Optional[Sequence[Any]] = None,
    anisotropy_ranges: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
) -> np.ndarray:
    """Classifies resource blocks into confidence categories based on drillhole spacing and hole counts.

    In accordance with CIM Definition Standards (2014) and JORC Code (2012) principles (TODO: Manually Verify):
    - Measured: Dense drillhole spacing where at least `min_holes_measured` independent informing
      drillholes lie within `max_radius_measured`, demonstrating high geological and grade continuity.
    - Indicated: Moderate drillhole spacing where at least `min_holes_indicated` independent informing
      drillholes lie within `max_radius_indicated`, supporting reasonable assumptions of continuity.
    - Inferred: Wider spacing or extrapolated blocks where geological continuity can be implied
      (within `max_radius_inferred` if specified).
    - Unclassified: Blocks beyond the maximum inferred search radius or lacking sufficient data.

    Multi-Hole Independence & Geological Domain Boundary Rules:
    - Multi-Hole Independence: When `sample_hole_ids` is provided, confidence criteria (Measured/Indicated)
      strictly evaluate the count of UNIQUE boreholes within the search volume. Multiple downhole
      composites along the same borehole are collapsed to a single borehole count, preventing false
      high-confidence classification along isolated drillholes (CIM MRMR Guidelines §6.11) (TODO: Manually Verify).
    - Domain Segregation: When `sample_domains` and `grid_domains` (or `domain_col`) are supplied,
      spacing queries are strictly isolated within each geological domain without smearing across boundaries.

    Supports 2D and 3D geometric anisotropy (CIM MRMR §6.8/§6.9) (TODO: Manually Verify): when `anisotropy_ranges`
    and/or `anisotropy_angles` are supplied, search distances are evaluated along the
    oriented search ellipsoid.

    Parameters
    ----------
    grid_points : np.ndarray or pd.DataFrame
        Array of block centers of shape (M, 2) or (M, 3), or DataFrame containing 'x', 'y' (and optional 'z') columns.
    samples_xy : np.ndarray
        Array of informing drillhole coordinates of shape (N, 2) or (N, 3).
    max_radius_measured : float
        Maximum search radius to qualify for Measured classification.
        Site-specific parameter required without default.
    max_radius_indicated : float
        Maximum search radius to qualify for Indicated classification.
        Site-specific parameter required without default.
    min_holes_measured : int, default 3
        Minimum informing independent drillholes within `max_radius_measured` for Measured.
    min_holes_indicated : int, default 2
        Minimum informing independent drillholes within `max_radius_indicated` for Indicated.
    max_radius_inferred : float, optional
        Maximum search radius to qualify for Inferred classification. If None,
        all blocks not meeting Measured or Indicated default to Inferred.
    is_interpolated : np.ndarray of bool, optional
        Boolean mask of shape (M,) indicating whether each block lies within the
        convex hull / domain interpolation volume. Blocks where `is_interpolated`
        is False cannot be classified as Measured or Indicated (capped at Inferred).
    sample_hole_ids : Sequence[Any], optional
        Identifiers of the drillhole (borehole ID) for each sample in `samples_xy` of shape (N,).
        When provided, counts of informing holes strictly reflect unique borehole IDs.
    domain_col : str, optional
        Geological domain column if `grid_points` is a DataFrame.
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples of shape (N,).
    grid_domains : Sequence[Any], optional
        Geological domain identifiers for target blocks of shape (M,).
    anisotropy_ranges : float, Sequence[float], or Mapping, optional
        Directional ranges (major, minor) in 2D or (major, semi-major, minor) in 3D.
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth in 2D, or azimuth, dip, plunge in 3D).
        Can be supplied per-domain as a dict mapping domain identifier to angles.

    Returns
    -------
    np.ndarray
        Array of category strings ("Measured", "Indicated", "Inferred", or "Unclassified") of shape (M,).
    """
    if isinstance(grid_points, pd.DataFrame):
        if "z" in grid_points.columns:
            pts = grid_points[["x", "y", "z"]].to_numpy(dtype=float)
        elif "x" in grid_points.columns and "y" in grid_points.columns:
            pts = grid_points[["x", "y"]].to_numpy(dtype=float)
        else:
            raise ValueError(
                "grid_points DataFrame must contain 'x', 'y' (and optional 'z') columns."
            )
        if domain_col is not None and domain_col in grid_points.columns:
            g_dom = grid_points[domain_col].to_numpy()
        elif grid_domains is not None:
            g_dom = np.asarray(grid_domains)
        else:
            g_dom = None
    else:
        pts = np.asarray(grid_points, dtype=float)
        g_dom = np.asarray(grid_domains) if grid_domains is not None else None

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

    if sample_hole_ids is not None:
        hole_ids_arr = np.asarray(sample_hole_ids)
        if len(hole_ids_arr) != N:
            raise ValueError(
                f"sample_hole_ids length ({len(hole_ids_arr)}) must match samples_xy length ({N})."
            )
    else:
        hole_ids_arr = None

    # Handle domain-segregated classification
    if (sample_domains is None) != (g_dom is None):
        raise ValueError(
            "Both sample_domains and grid_domains (or domain_col in grid_points) must be provided together."
        )

    if sample_domains is not None and g_dom is not None:
        s_dom = np.asarray(sample_domains)
        if len(s_dom) != N:
            raise ValueError(
                f"sample_domains length ({len(s_dom)}) must match samples_xy length ({N})."
            )
        if len(g_dom) != M:
            raise ValueError(
                f"grid_domains length ({len(g_dom)}) must match grid_points length ({M})."
            )

        categories = np.full(M, "Unclassified", dtype=object)

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

            interp_sub = interp_mask[g_mask] if interp_mask is not None else None
            hole_sub = hole_ids_arr[s_mask] if hole_ids_arr is not None else None

            categories[g_mask] = classify_resources_by_drill_spacing(
                grid_points=pts[g_mask],
                samples_xy=samples[s_mask],
                max_radius_measured=max_radius_measured,
                max_radius_indicated=max_radius_indicated,
                min_holes_measured=min_holes_measured,
                min_holes_indicated=min_holes_indicated,
                max_radius_inferred=max_radius_inferred,
                is_interpolated=interp_sub,
                sample_hole_ids=hole_sub,
                domain_col=None,
                sample_domains=None,
                grid_domains=None,
                anisotropy_ranges=dom_aniso_ranges,
                anisotropy_angles=dom_aniso_angles,
            )

        return categories

    if N == 0:
        return np.full(M, "Unclassified", dtype=object)

    samples_t, _ = transform_anisotropic_coordinates(
        samples, ranges=anisotropy_ranges, angles=anisotropy_angles
    )
    pts_t, _ = transform_anisotropic_coordinates(
        pts, ranges=anisotropy_ranges, angles=anisotropy_angles
    )

    tree = KDTree(samples_t)

    if hole_ids_arr is not None:
        # Multi-hole independence: query ball point and count unique hole IDs
        meas_balls = tree.query_ball_point(pts_t, r=max_radius_measured)
        meas_counts = np.array(
            [len(set(hole_ids_arr[b])) for b in meas_balls], dtype=int
        )
        has_measured = meas_counts >= min_holes_measured

        ind_balls = tree.query_ball_point(pts_t, r=max_radius_indicated)
        ind_counts = np.array([len(set(hole_ids_arr[b])) for b in ind_balls], dtype=int)
        has_indicated = ind_counts >= min_holes_indicated

        if max_radius_inferred is not None:
            inf_balls = tree.query_ball_point(pts_t, r=max_radius_inferred)
            has_inferred = np.array([len(b) > 0 for b in inf_balls], dtype=bool)
        else:
            has_inferred = np.ones(M, dtype=bool)
    else:
        # Standard sample point count query
        if N >= min_holes_measured:
            d_meas, _ = tree.query(pts_t, k=min_holes_measured)
            dist_meas = d_meas if min_holes_measured == 1 else d_meas[:, -1]
            has_measured = dist_meas <= max_radius_measured
        else:
            has_measured = np.zeros(M, dtype=bool)

        if N >= min_holes_indicated:
            d_ind, _ = tree.query(pts_t, k=min_holes_indicated)
            dist_ind = d_ind if min_holes_indicated == 1 else d_ind[:, -1]
            has_indicated = dist_ind <= max_radius_indicated
        else:
            has_indicated = np.zeros(M, dtype=bool)

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


def classify_resources_by_sor(
    slopes_of_regression: np.ndarray,
    threshold_measured: float,
    threshold_indicated: float,
    kriging_efficiencies: Optional[np.ndarray] = None,
    max_slope_measured: Optional[float] = None,
    min_kriging_efficiency: Optional[float] = 0.0,
) -> np.ndarray:
    """Classifies resource blocks based on Kriging Neighborhood Analysis (KNA) Slope of Regression (SoR).

    In geostatistical best practice (Vann et al., 2003; Armstrong, 1998) (TODO: Manually Verify):
    - Slope of Regression (SoR) gauges conditional unbiasedness of block estimates.
    - Measured: High conditional unbiasedness (typically SoR >= 0.80, with positive KE).
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
    max_slope_measured : float, optional, default None
        Optional maximum Slope of Regression for Measured classification.
        If None, no upper bound is enforced on Measured classification, but
        a warning is issued if slopes exceed 1.05.
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
    if max_slope_measured is not None and max_slope_measured < threshold_measured:
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

    # 1. Diagnostic warning for SoR slope exceeding 1.05 (or max_slope_measured if set)
    warn_threshold = max_slope_measured if max_slope_measured is not None else 1.05
    high_sor_mask = valid_sor & (sor > warn_threshold)
    high_sor_count = int(np.sum(high_sor_mask))
    if high_sor_count > 0:
        warnings.warn(
            f"Kriging Neighborhood Instability: {high_sor_count} block(s) exhibit Slope of Regression (SoR) "
            f"> {warn_threshold:.2f} (maximum SoR={float(np.nanmax(sor)):.3f}). In standard geostatistical practice "
            f"(SME Mining Engineering Handbook §4.5.8.2; Armstrong, 1998), an SoR > 1.0 indicates mathematical or geometric "
            f"instability in the kriging system (often caused by severe sample clustering, the string/screen effect, or an "
            f"excessive search neighborhood). Consider reducing k_neighbors, restricting composites per drillhole, or applying "
            f"spatial smoothing (smooth_resource_categories).",
            UserWarning,
            stacklevel=2,
        )

    if max_slope_measured is not None:
        is_meas = (
            valid_sor
            & (sor >= threshold_measured)
            & (sor <= max_slope_measured)
            & ke_ok
        )
    else:
        is_meas = valid_sor & (sor >= threshold_measured) & ke_ok

    is_ind = (
        valid_sor & (sor >= threshold_indicated) & (sor < threshold_measured) & ke_ok
    )
    is_inf = valid_sor & (~is_meas) & (~is_ind)

    categories[is_meas] = "Measured"
    categories[is_ind] = "Indicated"
    categories[is_inf] = "Inferred"

    return categories


def smooth_resource_categories(
    grid_points: Union[np.ndarray, pd.DataFrame],
    categories: Sequence[str] | np.ndarray,
    smoothing_radius: Optional[float] = None,
    k_neighbors: Optional[int] = None,
    min_cluster_size: int = 1,
    downgrade_isolated: bool = True,
    domain_col: Optional[str] = None,
    grid_domains: Optional[Sequence[Any]] = None,
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
    3. Geological Domain Isolation:
       When `grid_domains` or `domain_col` is supplied, smoothing operations are strictly
       confined within each geological domain, preventing category smearing across lithological
       contacts, faults, or structural boundaries.

    Parameters
    ----------
    grid_points : np.ndarray or pd.DataFrame
        Coordinates of block centers of shape (M, 2) or (M, 3), or DataFrame containing 'x', 'y' (and optional 'z') columns.
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
    domain_col : str, optional
        Geological domain column if `grid_points` is a DataFrame.
    grid_domains : Sequence[Any], optional
        Geological domain identifiers for target blocks of shape (M,).

    Returns
    -------
    np.ndarray
        Smoothed category array of shape (M,).
    """
    if isinstance(grid_points, pd.DataFrame):
        if "z" in grid_points.columns:
            pts = grid_points[["x", "y", "z"]].to_numpy(dtype=float)
        elif "x" in grid_points.columns and "y" in grid_points.columns:
            pts = grid_points[["x", "y"]].to_numpy(dtype=float)
        else:
            raise ValueError(
                "grid_points DataFrame must contain 'x', 'y' (and optional 'z') columns."
            )
        if domain_col is not None and domain_col in grid_points.columns:
            g_dom = grid_points[domain_col].to_numpy()
        elif grid_domains is not None:
            g_dom = np.asarray(grid_domains)
        else:
            g_dom = None
    else:
        pts = np.asarray(grid_points, dtype=float)
        g_dom = np.asarray(grid_domains) if grid_domains is not None else None

    cats = np.asarray(categories, dtype=object).copy()
    M = len(pts)
    if M == 0:
        return np.empty(0, dtype=object)
    if len(cats) != M:
        raise ValueError(
            f"Shape mismatch: grid_points length ({M}) does not match categories length ({len(cats)})."
        )

    if g_dom is not None:
        if len(g_dom) != M:
            raise ValueError(
                f"grid_domains length ({len(g_dom)}) must match grid_points length ({M})."
            )
        out_cats = np.full(M, "Unclassified", dtype=object)
        for dom in np.unique(g_dom):
            mask = g_dom == dom
            out_cats[mask] = smooth_resource_categories(
                grid_points=pts[mask],
                categories=cats[mask],
                smoothing_radius=smoothing_radius,
                k_neighbors=k_neighbors,
                min_cluster_size=min_cluster_size,
                downgrade_isolated=downgrade_isolated,
                domain_col=None,
                grid_domains=None,
            )
        return out_cats

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
    grid_points: Union[np.ndarray, pd.DataFrame],
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
    max_slope_measured: Optional[float] = None,
    min_kriging_efficiency: Optional[float] = 0.0,
    max_radius_inferred: Optional[float] = None,
    is_interpolated: Optional[np.ndarray] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
    domain_col: Optional[str] = None,
    sample_domains: Optional[Sequence[Any]] = None,
    grid_domains: Optional[Sequence[Any]] = None,
    smoothing_radius: Optional[float] = None,
    k_neighbors_smoothing: Optional[int] = None,
    min_cluster_size: int = 1,
    downgrade_isolated: bool = True,
    anisotropy_ranges: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
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
    grid_points : np.ndarray or pd.DataFrame
        Array of block centers of shape (M, 2) or (M, 3), or DataFrame containing 'x', 'y' (and optional 'z') columns.
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
        Minimum informing independent drillholes within search radius for Measured.
    min_holes_indicated : int, default 2
        Minimum informing independent drillholes within search radius for Indicated.
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
    max_slope_measured : float, optional, default None
        Optional maximum Slope of Regression for Measured classification.
        If None, no upper bound is enforced on Measured classification, but
        a warning is issued if slopes exceed 1.05.
    min_kriging_efficiency : float, optional, default 0.0
        Minimum Kriging Efficiency required for Measured/Indicated.
    max_radius_inferred : float, optional
        Maximum search radius for Inferred classification.
    is_interpolated : np.ndarray of bool, optional
        Boolean mask indicating whether each block lies within the interpolation hull.
        Blocks outside the hull are restricted to Inferred or Unclassified.
    sample_hole_ids : Sequence[Any], optional
        Identifiers of the drillhole (borehole ID) for each sample in `samples_xy` of shape (N,).
        When provided, counts of informing holes strictly reflect unique borehole IDs.
    domain_col : str, optional
        Geological domain column if `grid_points` is a DataFrame.
    sample_domains : Sequence[Any], optional
        Geological domain identifiers for conditioning samples of shape (N,).
    grid_domains : Sequence[Any], optional
        Geological domain identifiers for target blocks of shape (M,).
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
        Can be supplied per-domain as a dict mapping domain identifier to ranges.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth in 2D, or azimuth, dip, plunge in 3D).
        Can be supplied per-domain as a dict mapping domain identifier to angles.

    Returns
    -------
    np.ndarray
        Array of category strings ("Measured", "Indicated", "Inferred", or "Unclassified") of shape (M,).
    """
    if isinstance(grid_points, pd.DataFrame):
        if "z" in grid_points.columns:
            pts = grid_points[["x", "y", "z"]].to_numpy(dtype=float)
        elif "x" in grid_points.columns and "y" in grid_points.columns:
            pts = grid_points[["x", "y"]].to_numpy(dtype=float)
        else:
            raise ValueError(
                "grid_points DataFrame must contain 'x', 'y' (and optional 'z') columns."
            )
    else:
        pts = np.asarray(grid_points, dtype=float)

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
            grid_points=grid_points,
            samples_xy=samples_xy,
            max_radius_measured=max_radius_measured,
            max_radius_indicated=max_radius_indicated,
            min_holes_measured=min_holes_measured,
            min_holes_indicated=min_holes_indicated,
            max_radius_inferred=max_radius_inferred,
            is_interpolated=is_interpolated,
            sample_hole_ids=sample_hole_ids,
            domain_col=domain_col,
            sample_domains=sample_domains,
            grid_domains=grid_domains,
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
            grid_points=grid_points,
            categories=final_cats,
            smoothing_radius=smoothing_radius,
            k_neighbors=k_neighbors_smoothing,
            min_cluster_size=min_cluster_size,
            downgrade_isolated=downgrade_isolated,
            domain_col=domain_col,
            grid_domains=grid_domains,
        )

    return final_cats


def round_significant_figures(value: float, sig_figs: int) -> float:
    """Rounds a numeric value to a specified number of significant figures.

    Parameters
    ----------
    value : float
        The number to round.
    sig_figs : int
        Number of significant figures (must be >= 1).

    Returns
    -------
    float
        The rounded value.
    """
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

        t_rnd = round_significant_figures(t_val, t_sf)
        g_rnd = round_significant_figures(g_val, g_sf)
        m_rnd = round_significant_figures(m_val, m_sf)

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
        ax1 = ax
        fig = ax1.figure

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


def plot_swath_analysis_3axis(
    block_model: pd.DataFrame,
    drillholes: Optional[pd.DataFrame] = None,
    bin_widths: Union[float, Tuple[float, float, float], Sequence[float]] = (
        50.0,
        50.0,
        10.0,
    ),
    grade_col: str = "estimated_grade",
    validation_grade_col: Optional[str] = None,
    drillhole_grade_col: str = "grade",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    tonnes_col: Optional[str] = "tonnes",
    model_name: str = "Ordinary Kriging",
    validation_model_name: str = "Nearest Neighbor (Declustered)",
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16.0, 12.0),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates a synchronized 3-axis swath plot suite across Easting, Northing, and Elevation.

    Mandated by international reporting codes (NI 43-101 and JORC Code 2012 Table 1) (TODO: Manually Verify)
    to demonstrate local unbiasedness of the 3D resource estimate across all spatial directions.
    Plots primary estimated grades, declustered validation grades (e.g. Nearest Neighbor), and informing
    drillhole composite grades along Cartesian slices, with secondary Y-axis tonnage bars:
    - Panel 1: Easting (X corridor swaths)
    - Panel 2: Northing (Y corridor swaths)
    - Panel 3: Elevation (Z bench swaths)

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of block model estimates with coordinates and estimated grade.
    drillholes : pd.DataFrame, optional
        Table of drillhole composites with coordinates and assay grades.
    bin_widths : float or tuple of (float, float, float), default (50.0, 50.0, 10.0)
        Swath slice widths in meters for (Easting, Northing, Elevation).
    grade_col : str, default "estimated_grade"
        Primary grade estimate column in `block_model`. Falls back to 'grade' if 'estimated_grade' is absent.
    validation_grade_col : str, optional
        Declustered benchmark model column in `block_model` (e.g. NN grade).
    drillhole_grade_col : str, default "grade"
        Grade column name in `drillholes`.
    x_col : str, default "x"
        Easting coordinate column name.
    y_col : str, default "y"
        Northing coordinate column name.
    z_col : str, default "z"
        Elevation coordinate column name.
    tonnes_col : str, optional, default "tonnes"
        Tonnage column name in `block_model`.
    model_name : str, default "Ordinary Kriging"
        Label for the primary estimator being validated.
    validation_model_name : str, default "Nearest Neighbor (Declustered)"
        Label for the declustered benchmark model.
    grade_unit : str, default "% Cu"
        Unit label for mineral grade.
    tonnage_unit : str, default "Mt"
        Unit label for slice ore tonnage.
    title : str, optional
        Overall figure super-title.
    figsize : Tuple[float, float], default (16.0, 12.0)
        Figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Matplotlib Figure and sequence of 3 primary Axes (Easting, Northing, Elevation).
    """
    if block_model.empty:
        raise ValueError("block_model cannot be empty.")

    # Resolve primary grade column fallback if default 'estimated_grade' is not in block_model but 'grade' is
    eff_grade_col = grade_col
    if eff_grade_col not in block_model.columns and "grade" in block_model.columns:
        eff_grade_col = "grade"
    elif eff_grade_col not in block_model.columns:
        raise ValueError(
            f"grade_col '{grade_col}' not found in block_model columns: {list(block_model.columns)}."
        )

    # Resolve bin widths
    if isinstance(bin_widths, (int, float)):
        bw_x = bw_y = bw_z = float(bin_widths)
    elif len(bin_widths) == 3:
        bw_x, bw_y, bw_z = (
            float(bin_widths[0]),
            float(bin_widths[1]),
            float(bin_widths[2]),
        )
    else:
        raise ValueError(
            f"bin_widths must be a float or a sequence of 3 floats, got {bin_widths}."
        )

    if bw_x <= 0.0 or bw_y <= 0.0 or bw_z <= 0.0:
        raise ValueError("All bin_widths must be strictly positive.")

    # Check required coordinate columns
    for col, name in [(x_col, "x"), (y_col, "y"), (z_col, "z")]:
        if col not in block_model.columns:
            raise ValueError(
                f"Required coordinate column '{col}' ({name}) not found in block_model."
            )

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    axes_configs = [
        ("x", bw_x, axes[0], "Easting (X Swaths)"),
        ("y", bw_y, axes[1], "Northing (Y Swaths)"),
        ("z", bw_z, axes[2], "Elevation (Z Benches)"),
    ]

    for axis_dir, bw, ax, panel_name in axes_configs:
        plot_swath_analysis(
            block_model=block_model,
            drillholes=drillholes,
            axis=axis_dir,
            bin_width=bw,
            grade_col=eff_grade_col,
            validation_grade_col=validation_grade_col,
            drillhole_grade_col=drillhole_grade_col,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            tonnes_col=tonnes_col,
            model_name=model_name,
            validation_model_name=validation_model_name,
            grade_unit=grade_unit,
            tonnage_unit=tonnage_unit,
            title=f"Local Drift Validation Along {panel_name}",
            ax=ax,
        )

    sup_title = (
        title
        if title is not None
        else f"3-Axis Swath Plot Validation: {model_name}"
        + (f" vs. {validation_model_name}" if validation_grade_col is not None else "")
    )
    fig.suptitle(sup_title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()

    return fig, (axes[0], axes[1], axes[2])


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

    op_cost = (
        processing_cost + ga_cost + (mining_cost if mining_cost is not None else 0.0)
    )
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


def classify_reserve_status(
    block_model: pd.DataFrame,
    cutoff_grade: float,
    mining_dilution_pct: float = 0.0,
    mining_recovery_pct: float = 100.0,
    dilution_grade: float = 0.0,
    category_col: str = "category",
    grade_col: str = "grade",
    tonnes_col: str = "tonnes",
    cutoff_type: Literal["rom", "in_situ"] = "rom",
    status_col: str = "status",
) -> pd.DataFrame:
    """Classifies every block in a 3D block model into CIM / JORC Mineral Reserve statuses.

    Produces a 1-to-1 spatial table where every block is assigned one of the 4 regulatory statuses:
    1. 'Proven Reserve': Measured mineral resource meeting economic modifying factor cut-off.
    2. 'Probable Reserve': Indicated mineral resource meeting economic modifying factor cut-off.
    3. 'Inferred Resource (Excluded)': Inferred mineral resource strictly prohibited from reserve
       conversion under CRIRSCO / CIM / JORC reporting standards (TODO: Manually Verify).
    4. 'Sub-Economic / Waste': Blocks falling below economic cut-off grade or unclassified waste.

    Parameters
    ----------
    block_model : pd.DataFrame
        Classified block model table containing spatial coordinates, resource category, and grades.
    cutoff_grade : float
        Economic cut-off grade for reserve delineation.
    mining_dilution_pct : float, default 0.0
        Mining dilution percentage applied to candidate ore.
    mining_recovery_pct : float, default 100.0
        Mining recovery percentage applied to candidate ore.
    dilution_grade : float, default 0.0
        Grade of diluting material.
    category_col : str, default "category"
        Resource classification column ('Measured', 'Indicated', 'Inferred', 'Unclassified').
    grade_col : str, default "grade"
        Block grade column name.
    tonnes_col : str, default "tonnes"
        Block tonnage column name.
    cutoff_type : Literal["rom", "in_situ"], default "rom"
        Cut-off application basis.
    status_col : str, default "status"
        Name of the resulting status column.

    Returns
    -------
    pd.DataFrame
        Copy of block_model with the new status column populated for 100% of blocks.
    """
    if block_model.empty:
        raise ValueError("block_model cannot be empty.")
    for req in [category_col, grade_col, tonnes_col]:
        if req not in block_model.columns:
            raise ValueError(f"Required column '{req}' not found in block_model.")

    res_df = convert_resource_to_reserve(
        resource_df=block_model,
        mining_dilution_pct=mining_dilution_pct,
        mining_recovery_pct=mining_recovery_pct,
        cutoff_grade=cutoff_grade,
        dilution_grade=dilution_grade,
        category_col=category_col,
        grade_col=grade_col,
        tonnes_col=tonnes_col,
        cutoff_type=cutoff_type,
    )
    df_out = block_model.copy()
    status_arr = np.full(len(df_out), "Sub-Economic / Waste", dtype=object)
    cat_series = df_out[category_col].astype(str).str.strip().str.lower()
    is_inferred = cat_series.str.startswith("infer")
    status_arr[is_inferred] = "Inferred Resource (Excluded)"
    status_arr[res_df.index] = res_df["reserve_category"]
    df_out[status_col] = status_arr
    df_out.attrs["total_blocks"] = len(df_out)
    df_out.attrs["proven_count"] = int((status_arr == "Proven Reserve").sum())
    df_out.attrs["probable_count"] = int((status_arr == "Probable Reserve").sum())
    df_out.attrs["inferred_excluded_count"] = int(
        (status_arr == "Inferred Resource (Excluded)").sum()
    )
    df_out.attrs["subeconomic_waste_count"] = int(
        (status_arr == "Sub-Economic / Waste").sum()
    )
    df_out.attrs["reserve_df"] = res_df
    return df_out


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
                f"Tonnage ({tonnage_unit})": round_significant_figures(
                    t_val / t_scale, n_sf
                ),
                f"Grade ({grade_unit})": round_significant_figures(g_val, n_sf),
                f"Contained Metal ({metal_unit})": round_significant_figures(
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


def _filter_bench_slice(
    block_model: pd.DataFrame,
    bench_z: Optional[float] = None,
    elevation_tolerance: float = 5.0,
    z_col: str = "z",
) -> pd.DataFrame:
    """Filters a block model DataFrame to a horizontal elevation bench slice."""
    if bench_z is None:
        return block_model
    if z_col not in block_model.columns:
        raise ValueError(
            f"z_col '{z_col}' not found in block_model for bench filtering."
        )
    df_plot = block_model[
        (block_model[z_col] >= bench_z - elevation_tolerance)
        & (block_model[z_col] <= bench_z + elevation_tolerance)
    ].copy()
    if len(df_plot) == 0:
        raise ValueError(
            f"No blocks found within elevation {bench_z} ± {elevation_tolerance} m."
        )
    return df_plot


def _validate_orthogonal_views(
    cross_section_view: str, long_section_view: str
) -> Tuple[str, str]:
    """Validates and standardizes viewing directions for orthogonal cross- and long-sections."""
    csv = cross_section_view.lower()
    if csv not in ["north", "south"]:
        raise ValueError("cross_section_view must be 'north' or 'south'.")
    lsv = long_section_view.lower()
    if lsv not in ["west", "east"]:
        raise ValueError("long_section_view must be 'west' or 'east'.")
    return csv, lsv


def _find_nearest_slice_coords(
    block_model: pd.DataFrame,
    bench_z: Optional[float] = None,
    section_y: Optional[float] = None,
    section_x: Optional[float] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
) -> Tuple[float, float, float]:
    """Finds exact nearest centroid coordinate values for orthogonal slices (bench_z, section_y, section_x)."""
    unique_z = np.sort(block_model[z_col].unique())
    unique_y = np.sort(block_model[y_col].unique())
    unique_x = np.sort(block_model[x_col].unique())

    target_z = (
        float(unique_z[np.argmin(np.abs(unique_z - bench_z))])
        if bench_z is not None
        else float(unique_z[len(unique_z) // 2])
    )
    target_y = (
        float(unique_y[np.argmin(np.abs(unique_y - section_y))])
        if section_y is not None
        else float(unique_y[len(unique_y) // 2])
    )
    target_x = (
        float(unique_x[np.argmin(np.abs(unique_x - section_x))])
        if section_x is not None
        else float(unique_x[len(unique_x) // 2])
    )
    return target_z, target_y, target_x


def _plot_categorical_2d_map(
    block_model: pd.DataFrame,
    category_col: str,
    categories: Sequence[str],
    colors: Sequence[str],
    boundary: Optional[Sequence[tuple[float, float]]] = None,
    drillholes: Optional[pd.DataFrame] = None,
    bench_z: Optional[float] = None,
    elevation_tolerance: float = 5.0,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    title: Optional[str] = None,
    default_title: str = "Classification Map",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 8),
    point_size: float = 35.0,
    alpha: float = 0.85,
    boundary_color: str = "black",
    boundary_linestyle: str = "-",
    boundary_label: str = "Mine Boundary",
) -> tuple[plt.Figure, plt.Axes]:
    """Shared 2D spatial mine plan map renderer for categorical block models."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    df_plot = _filter_bench_slice(
        block_model=block_model,
        bench_z=bench_z,
        elevation_tolerance=elevation_tolerance,
        z_col=z_col,
    )

    cats_lower = df_plot[category_col].astype(str).str.strip().str.lower()

    for cat_name, color in zip(categories, colors):
        sub = df_plot[cats_lower == cat_name.lower()]
        if len(sub) > 0:
            ax.scatter(
                sub[x_col],
                sub[y_col],
                c=color,
                label=f"{cat_name} ({len(sub):,} blocks)",
                s=point_size,
                marker="s",
                alpha=alpha,
                edgecolors="none",
            )

    # Perimeter boundary polygon
    if boundary is not None and len(boundary) >= 3:
        poly_patch = MplPolygon(
            boundary,
            closed=True,
            fill=False,
            edgecolor=boundary_color,
            linewidth=2.0,
            linestyle=boundary_linestyle,
            label=boundary_label,
            zorder=5,
        )
        ax.add_patch(poly_patch)

    # Drillhole collars
    if drillholes is not None and len(drillholes) > 0:
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
    plot_title = title
    if plot_title is None:
        plot_title = default_title
        if bench_z is not None:
            plot_title += f" (Bench Z = {bench_z:.1f} m)"
    ax.set_title(plot_title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    fig.tight_layout()
    return fig, ax


def plot_reserve_classification_map(
    block_model: pd.DataFrame,
    boundary: Optional[Sequence[tuple[float, float]]] = None,
    drillholes: Optional[pd.DataFrame] = None,
    status_col: str = "status",
    bench_z: Optional[float] = None,
    elevation_tolerance: float = 5.0,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Renders 2D spatial mine plan map colored by regulatory reserve status.

    Standard coloring:
    - Proven Reserve: Forest Green (#2ca02c)
    - Probable Reserve: Royal Blue (#1f77b4)
    - Inferred Resource (Excluded): Purple / Magenta (#9467bd)
    - Sub-Economic / Waste (< Cutoff): Light Grey (#d3d3d3)

    Supports horizontal bench slice filtering via `bench_z` and `elevation_tolerance`
    to prevent 3D block overplotting on 2D mine plans.

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
    bench_z : float, optional
        Target elevation for horizontal bench slice. If specified, isolates blocks within
        elevation_tolerance of bench_z to eliminate 3D overplotting.
    elevation_tolerance : float, default 5.0
        Half-thickness (± meters) for bench slice filtering.
    x_col, y_col : str, default "x", "y"
        Coordinate column names.
    z_col : str, default "z"
        Elevation coordinate column name (required if bench_z is provided).
    title : str, optional
        Plot title.
    ax : plt.Axes, optional
        Existing axes to draw on.
    figsize : tuple[float, float], default (10, 8)
        Dimensions of figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    return _plot_categorical_2d_map(
        block_model=block_model,
        category_col=status_col,
        categories=[
            "Proven Reserve",
            "Probable Reserve",
            "Inferred Resource (Excluded)",
            "Sub-Economic / Waste",
        ],
        colors=["#2ca02c", "#1f77b4", "#9467bd", "#d3d3d3"],
        boundary=boundary,
        drillholes=drillholes,
        bench_z=bench_z,
        elevation_tolerance=elevation_tolerance,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        title=title,
        default_title="Mineral Reserve & Resource Classification Map",
        ax=ax,
        figsize=figsize,
        alpha=0.75,
        boundary_color="black",
        boundary_linestyle="-",
        boundary_label="Mine Boundary",
    )


def plot_reserve_classification_orthogonal_slices(
    block_model: pd.DataFrame,
    status_col: str = "status",
    bench_z: Optional[float] = None,
    section_y: Optional[float] = None,
    section_x: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    cross_section_view: str = "north",
    long_section_view: str = "west",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    dx_col: str = "dx",
    dy_col: str = "dy",
    dz_col: str = "dz",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (19.0, 5.8),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates 3-panel synchronized orthogonal slices of regulatory Mineral Reserve status.

    Standard coloring:
    - Proven Reserve: Forest Green (#2ca02c)
    - Probable Reserve: Royal Blue (#1f77b4)
    - Inferred Resource (Excluded): Purple / Magenta (#9467bd)
    - Sub-Economic / Waste: Light Grey (#d3d3d3)

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates and reserve status column.
    status_col : str, default "status"
        Column containing reserve classification status.
    bench_z, section_y, section_x : float, optional
        Target slice coordinates.
    samples_xyz : np.ndarray, optional
        Drillhole composite coordinates for overlay.
    cross_section_view, long_section_view : str, default "north", "west"
        Viewing directions.
    x_col, y_col, z_col : str, default "x", "y", "z"
        Coordinate column names.
    dx_col, dy_col, dz_col : str, default "dx", "dy", "dz"
        Block dimension column names.
    title : str, optional
        Figure title.
    figsize : Tuple[float, float], default (19.0, 5.8)
        Dimensions of figure.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and sequence of 3 axes (ax_bench, ax_cross, ax_long).
    """
    return _plot_categorical_blocks_orthogonal_slices(
        block_model=block_model,
        category_col=status_col,
        categories=[
            "Proven Reserve",
            "Probable Reserve",
            "Inferred Resource (Excluded)",
            "Sub-Economic / Waste",
        ],
        colors=["#2ca02c", "#1f77b4", "#9467bd", "#d3d3d3"],
        bench_z=bench_z,
        section_y=section_y,
        section_x=section_x,
        samples_xyz=samples_xyz,
        cross_section_view=cross_section_view,
        long_section_view=long_section_view,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        dx_col=dx_col,
        dy_col=dy_col,
        dz_col=dz_col,
        title=title or "3D Mineral Reserve Status Orthogonal Slices",
        figsize=figsize,
    )


def plot_reserve_classification_bench_gallery(
    block_model: pd.DataFrame,
    status_col: str = "status",
    bench_elevations: Optional[Sequence[float]] = None,
    n_cols: int = 3,
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    dx_col: str = "dx",
    dy_col: str = "dy",
    dz_col: str = "dz",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots multi-bench elevation gallery of Mineral Reserve status slices.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates and reserve status.
    status_col : str, default "status"
        Reserve status column.
    bench_elevations : Sequence[float], optional
        Elevations to plot.
    n_cols : int, default 3
        Number of gallery columns.
    samples_xyz : np.ndarray, optional
        Drillhole coordinates for overlay.
    x_col, y_col, z_col : str, default "x", "y", "z"
        Coordinate column names.
    dx_col, dy_col, dz_col : str, default "dx", "dy", "dz"
        Block dimension column names.
    title : str, optional
        Plot title.
    figsize : Tuple[float, float], optional
        Figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and active axes sequence.
    """
    return _plot_categorical_bench_gallery(
        block_model=block_model,
        category_col=status_col,
        categories=[
            "Proven Reserve",
            "Probable Reserve",
            "Inferred Resource (Excluded)",
            "Sub-Economic / Waste",
        ],
        colors=["#2ca02c", "#1f77b4", "#9467bd", "#d3d3d3"],
        bench_elevations=bench_elevations,
        n_cols=n_cols,
        samples_xyz=samples_xyz,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        dx_col=dx_col,
        dy_col=dy_col,
        dz_col=dz_col,
        title=title or "Mineral Reserve Status: Multi-Bench Depth Gallery",
        figsize=figsize,
    )


def plot_reserve_classification_3d_isometric(
    block_model: pd.DataFrame,
    status_col: str = "status",
    active_categories: Optional[Sequence[str]] = None,
    bench_z: Optional[float] = None,
    elevation_tolerance: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    elev: float = 28.0,
    azim: float = -55.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 8.0),
) -> Tuple[plt.Figure, Axes3D]:
    """Renders static 3D isometric projection of Mineral Reserve status categories.

    Allows filtering to specific reserve categories (e.g. only Proven and Probable
    to isolate the mineable orebody from waste) or bench elevation ranges.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates and reserve status.
    status_col : str, default "status"
        Reserve status column.
    active_categories : Sequence[str], optional
        Categories to render (e.g. ['Proven Reserve', 'Probable Reserve']).
    bench_z, elevation_tolerance : float, optional
        Elevation bench slice corridor.
    samples_xyz : np.ndarray, optional
        Drillhole coordinates.
    x_col, y_col, z_col : str, default "x", "y", "z"
        Coordinate column names.
    elev, azim : float, default 28.0, -55.0
        Camera angles.
    title : str, optional
        Plot title.
    figsize : Tuple[float, float], default (10.0, 8.0)
        Figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Axes3D]
        Figure and 3D axes.
    """
    return _plot_categorical_blocks_3d_isometric(
        block_model=block_model,
        category_col=status_col,
        categories=[
            "Proven Reserve",
            "Probable Reserve",
            "Inferred Resource (Excluded)",
            "Sub-Economic / Waste",
        ],
        colors=["#2ca02c", "#1f77b4", "#9467bd", "#d3d3d3"],
        active_categories=active_categories,
        bench_z=bench_z,
        elevation_tolerance=elevation_tolerance,
        samples_xyz=samples_xyz,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        elev=elev,
        azim=azim,
        title=title or "3D Mineral Reserve Classification",
        figsize=figsize,
    )


def plot_reserve_classification_3d_interactive(
    block_model: pd.DataFrame,
    status_col: str = "status",
    tonnes_col: Optional[str] = "tonnes",
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    elev: float = 28.0,
    azim: float = -55.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 9.0),
) -> Tuple[plt.Figure, Axes3D, Dict[str, Any]]:
    """Interactive 3D explorer for Mineral Reserve status categories.

    Provides real-time checkboxes to toggle Proven, Probable, Inferred Excluded,
    and Waste blocks, an elevation slider for bench-by-bench slice exploration,
    and a live HUD reporting reserve tonnage and percentage breakdowns.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates and reserve status.
    status_col : str, default "status"
        Reserve status column.
    tonnes_col : str, optional, default "tonnes"
        Tonnage column for real-time HUD mass accounting.
    samples_xyz : np.ndarray, optional
        Drillhole coordinates.
    x_col, y_col, z_col : str, default "x", "y", "z"
        Coordinate column names.
    elev, azim : float, default 28.0, -55.0
        Camera viewing angles.
    title : str, optional
        Window and plot title.
    figsize : Tuple[float, float], default (11.0, 9.0)
        Figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Axes3D, Dict[str, Any]]
        Figure, 3D axes, and interactive controls dictionary.
    """
    return _plot_categorical_blocks_3d_interactive(
        block_model=block_model,
        category_col=status_col,
        categories=[
            "Proven Reserve",
            "Probable Reserve",
            "Inferred Resource (Excluded)",
            "Sub-Economic / Waste",
        ],
        colors=["#2ca02c", "#1f77b4", "#9467bd", "#d3d3d3"],
        tonnes_col=tonnes_col,
        samples_xyz=samples_xyz,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        elev=elev,
        azim=azim,
        title=title or "Interactive 3D Mineral Reserve Explorer",
        figsize=figsize,
    )


def plot_resource_classification_map(
    block_model: pd.DataFrame,
    category_col: str = "category",
    boundary: Optional[Sequence[tuple[float, float]]] = None,
    drillholes: Optional[pd.DataFrame] = None,
    bench_z: Optional[float] = None,
    elevation_tolerance: float = 5.0,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (10, 8),
) -> tuple[plt.Figure, plt.Axes]:
    """Renders 2D spatial mine plan map colored by regulatory mineral resource category.

    In accordance with CIM Definition Standards (2014) and JORC Code (2012) (TODO: Manually Verify):
    - Measured Resource: Forest Green (#2ca02c)
    - Indicated Resource: Royal Blue (#1f77b4)
    - Inferred Resource: Amber / Orange (#ff7f0e)
    - Unclassified: Light Grey (#d3d3d3)

    Supports horizontal bench slice filtering via `bench_z` and `elevation_tolerance`
    to prevent 3D block overplotting on 2D mine plans.

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
    bench_z : float, optional
        Target elevation for horizontal bench slice. If specified, isolates blocks within
        elevation_tolerance of bench_z to eliminate 3D overplotting.
    elevation_tolerance : float, default 5.0
        Half-thickness (± meters) for bench slice filtering.
    x_col, y_col : str, default "x", "y"
        Coordinate column names.
    z_col : str, default "z"
        Elevation coordinate column name (required if bench_z is provided).
    title : str, optional
        Plot title.
    ax : plt.Axes, optional
        Existing axes to draw on.
    figsize : tuple[float, float], default (10, 8)
        Dimensions of figure.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
    """
    return _plot_categorical_2d_map(
        block_model=block_model,
        category_col=category_col,
        categories=["Measured", "Indicated", "Inferred", "Unclassified"],
        colors=["#2ca02c", "#1f77b4", "#ff7f0e", "#d3d3d3"],
        boundary=boundary,
        drillholes=drillholes,
        bench_z=bench_z,
        elevation_tolerance=elevation_tolerance,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        title=title,
        default_title="CIM / JORC Mineral Resource Classification Map",
        ax=ax,
        figsize=figsize,
        alpha=0.85,
        boundary_color="red",
        boundary_linestyle="--",
        boundary_label="Concession Perimeter",
    )


def plot_resource_classification_orthogonal_slices(
    block_model: pd.DataFrame,
    category_col: str = "category",
    bench_z: Optional[float] = None,
    section_y: Optional[float] = None,
    section_x: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    cross_section_view: str = "north",
    long_section_view: str = "west",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    dx_col: str = "dx",
    dy_col: str = "dy",
    dz_col: str = "dz",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (19.0, 5.8),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates 3-panel synchronized orthogonal slices of regulatory resource classification.

    Mandated by international reporting guidelines (CIM Definition Standards 2014, JORC Code 2012 Table 1)
    (TODO: Manually Verify) to document spatial continuity and drill spacing compliance in 3D:
    - Panel 1: Bench Plan (X-Y at elevation Z = bench_z): Horizontal continuity of Measured/Indicated blocks.
    - Panel 2: Cross-Section (X-Z at northing Y = section_y): Vertical continuity across deposit strike.
      * Looking North (default): West is left, East is right.
      * Looking South: East is left, West is right (X-axis inverted).
    - Panel 3: Longitudinal Section (Y-Z at easting X = section_x): Continuity and plunge along strike.
      * Looking West (default): South is left, North is right.
      * Looking East: North is left, South is right (Y-axis inverted).

    Standard regulatory classification colors:
    - Measured: Forest Green (#2ca02c)
    - Indicated: Royal Blue (#1f77b4)
    - Inferred: Amber / Orange (#ff7f0e)
    - Unclassified: Light Grey (#d3d3d3)

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and category column.
    category_col : str, default "category"
        Resource classification column ("Measured", "Indicated", "Inferred", "Unclassified").
    bench_z : float, optional
        Target elevation for horizontal bench slice. If None, uses median block elevation.
    section_y : float, optional
        Target northing for cross-section slice. If None, uses median block northing.
    section_x : float, optional
        Target easting for longitudinal slice. If None, uses median block easting.
    samples_xyz : np.ndarray, optional
        Drillhole composite coordinates of shape (N, 3) for slice corridor overlay.
    cross_section_view : str, default "north"
        Viewing direction for cross-section ("north" or "south").
    long_section_view : str, default "west"
        Viewing direction for longitudinal section ("west" or "east").
    x_col, y_col, z_col : str, default "x", "y", "z"
        Centroid coordinate column names.
    dx_col, dy_col, dz_col : str, default "dx", "dy", "dz"
        Block dimension column names (auto-estimated from coordinates if absent).
    title : str, optional
        Super-title for the figure.
    figsize : Tuple[float, float], default (19.0, 5.8)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and tuple of 3 axes: (ax_bench, ax_cross, ax_long).
    """
    return _plot_categorical_blocks_orthogonal_slices(
        block_model=block_model,
        category_col=category_col,
        categories=["Measured", "Indicated", "Inferred", "Unclassified"],
        colors=["#2ca02c", "#1f77b4", "#ff7f0e", "#d3d3d3"],
        bench_z=bench_z,
        section_y=section_y,
        section_x=section_x,
        samples_xyz=samples_xyz,
        cross_section_view=cross_section_view,
        long_section_view=long_section_view,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        dx_col=dx_col,
        dy_col=dy_col,
        dz_col=dz_col,
        title=title or "3D Resource Classification Orthogonal Slices (CIM / JORC)",
        figsize=figsize,
    )


def plot_resource_classification_bench_gallery(
    block_model: pd.DataFrame,
    category_col: str = "category",
    bench_elevations: Optional[Sequence[float]] = None,
    n_cols: int = 3,
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    dx_col: str = "dx",
    dy_col: str = "dy",
    dz_col: str = "dz",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Plots multi-bench elevation gallery of CIM/JORC Mineral Resource Classification slices.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and classification category.
    category_col : str, default "category"
        Resource classification column ("Measured", "Indicated", "Inferred", "Unclassified").
    bench_elevations : Sequence[float], optional
        Specific bench elevations to plot. If None, samples up to 6 evenly spaced levels.
    n_cols : int, default 3
        Number of subplot columns in gallery grid.
    samples_xyz : np.ndarray, optional
        Composite drillhole coordinates for corridor overlay.
    x_col, y_col, z_col : str, default "x", "y", "z"
        Coordinate column names.
    dx_col, dy_col, dz_col : str, default "dx", "dy", "dz"
        Block dimension column names.
    title : str, optional
        Super-title for figure.
    figsize : Tuple[float, float], optional
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and active axes sequence.
    """
    return _plot_categorical_bench_gallery(
        block_model=block_model,
        category_col=category_col,
        categories=["Measured", "Indicated", "Inferred", "Unclassified"],
        colors=["#2ca02c", "#1f77b4", "#ff7f0e", "#d3d3d3"],
        bench_elevations=bench_elevations,
        n_cols=n_cols,
        samples_xyz=samples_xyz,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        dx_col=dx_col,
        dy_col=dy_col,
        dz_col=dz_col,
        title=title
        or "CIM / JORC Mineral Resource Classification: Multi-Bench Depth Gallery",
        figsize=figsize,
    )


def plot_resource_classification_3d_isometric(
    block_model: pd.DataFrame,
    category_col: str = "category",
    active_categories: Optional[Sequence[str]] = None,
    bench_z: Optional[float] = None,
    elevation_tolerance: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    elev: float = 28.0,
    azim: float = -55.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 8.0),
) -> Tuple[plt.Figure, Axes3D]:
    """Renders static 3D isometric view of CIM/JORC Mineral Resource Classification categories.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and classification category.
    category_col : str, default "category"
        Resource classification column ("Measured", "Indicated", "Inferred", "Unclassified").
    active_categories : Sequence[str], optional
        Subset of categories to display (e.g. ['Measured', 'Indicated']). Defaults to all.
    bench_z : float, optional
        Center elevation for horizontal bench corridor filter.
    elevation_tolerance : float, optional
        Half-thickness in meters for bench corridor filter (default 5.0m).
    samples_xyz : np.ndarray, optional
        Drillhole composite coordinates for spatial overlay.
    x_col, y_col, z_col : str, default "x", "y", "z"
        Coordinate column names.
    elev, azim : float, default 28.0, -55.0
        Matplotlib 3D camera viewing angles.
    title : str, optional
        Figure title.
    figsize : Tuple[float, float], default (10.0, 8.0)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Axes3D]
        Figure and 3D axes.
    """
    return _plot_categorical_blocks_3d_isometric(
        block_model=block_model,
        category_col=category_col,
        categories=["Measured", "Indicated", "Inferred", "Unclassified"],
        colors=["#2ca02c", "#1f77b4", "#ff7f0e", "#d3d3d3"],
        active_categories=active_categories,
        bench_z=bench_z,
        elevation_tolerance=elevation_tolerance,
        samples_xyz=samples_xyz,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        elev=elev,
        azim=azim,
        title=title or "3D Mineral Resource Classification (CIM / JORC)",
        figsize=figsize,
    )


def plot_resource_classification_3d_interactive(
    block_model: pd.DataFrame,
    category_col: str = "category",
    tonnes_col: Optional[str] = "tonnes",
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    elev: float = 28.0,
    azim: float = -55.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 9.0),
) -> Tuple[plt.Figure, Axes3D, Dict[str, Any]]:
    """Interactive 3D explorer for CIM/JORC Mineral Resource Classification.

    Provides real-time interactive checkboxes to toggle category visibility (Measured,
    Indicated, Inferred, Unclassified), an elevation slider to slice benches downward,
    and a live HUD reporting active block counts, percentages, and tonnage.

    Parameters
    ----------
    block_model : pd.DataFrame
        Table of blocks containing centroid coordinates (x, y, z) and classification category.
    category_col : str, default "category"
        Resource classification column ("Measured", "Indicated", "Inferred", "Unclassified").
    tonnes_col : str, optional, default "tonnes"
        Column containing block tonnage for real-time HUD mass accounting.
    samples_xyz : np.ndarray, optional
        Drillhole composite coordinates for spatial overlay.
    x_col, y_col, z_col : str, default "x", "y", "z"
        Coordinate column names.
    elev, azim : float, default 28.0, -55.0
        Initial 3D camera angles.
    title : str, optional
        Window and plot title.
    figsize : Tuple[float, float], default (11.0, 9.0)
        Figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Axes3D, Dict[str, Any]]
        Figure, 3D axes, and widget controls dictionary.
    """
    return _plot_categorical_blocks_3d_interactive(
        block_model=block_model,
        category_col=category_col,
        categories=["Measured", "Indicated", "Inferred", "Unclassified"],
        colors=["#2ca02c", "#1f77b4", "#ff7f0e", "#d3d3d3"],
        tonnes_col=tonnes_col,
        samples_xyz=samples_xyz,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
        elev=elev,
        azim=azim,
        title=title or "Interactive 3D Resource Classification Explorer",
        figsize=figsize,
    )


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
        raise ValueError(f"min_length_ratio must be in [0, 1], got {min_length_ratio}")
    if not (0.0 <= min_coverage_ratio <= 1.0):
        raise ValueError(
            f"min_coverage_ratio must be in [0, 1], got {min_coverage_ratio}"
        )
    if max_gap_length < 0.0:
        raise ValueError(f"max_gap_length must be non-negative, got {max_gap_length}")

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
                            if (
                                np.isfinite(p_val)
                                and np.isfinite(c_val)
                                and abs(denom) > 1e-9
                            ):
                                gap_coords[c] = float(
                                    p_val
                                    + ((gap_mid - prev_mid) / denom) * (c_val - p_val)
                                )
                            elif np.isfinite(p_val):
                                gap_coords[c] = float(p_val)
                            else:
                                gap_coords[c] = float(c_val)

                        current_run.append(
                            {
                                "from": prev_to,
                                "to": f_i,
                                "length": gap_len,
                                "grade": unassayed_grade,
                                "density": mean_density,
                                "is_sampled": False,
                                "domain": current_dom,
                                "coords": gap_coords,
                            }
                        )

            # Check domain transition
            if (
                domain_col is not None
                and current_dom is not None
                and dom_i != current_dom
            ):
                if current_run:
                    runs.append(current_run)
                    current_run = []
                    current_dom = None

            # Handle current interval
            if missing_i:
                total_gaps_count += 1
                total_gaps_length += t_i - f_i
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
                    comp_intervals.append(
                        (
                            run_start + ci * actual_len,
                            run_start + (ci + 1) * actual_len,
                        )
                    )
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
                    sample_weights = (
                        active_lens[active_sampled] * active_dens[active_sampled]
                    )
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
                    weighted_grade = float((weights * active_grades).sum() / tot_w)

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
                    raw_c = np.array(
                        [s["coords"].get(c_name, np.nan) for s in run], dtype=float
                    )
                    active_c = raw_c[valid_mask]
                    c_valid = np.isfinite(active_c)
                    if np.any(c_valid):
                        c_weights = active_lens[c_valid]
                        comp_coord = float(
                            (c_weights * active_c[c_valid]).sum() / c_weights.sum()
                        )
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
    contact_plane_point: Optional[Sequence[float]] = None,
    contact_dip: Optional[float] = None,
    contact_dip_direction: Optional[float] = None,
    contact_normal: Optional[Sequence[float]] = None,
    contact_points: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: Optional[str] = "z",
    bin_width: float = 2.0,
    max_distance: float = 30.0,
    domain_a: Optional[Any] = None,
    domain_b: Optional[Any] = None,
) -> pd.DataFrame:
    """Evaluates grade continuity across a geological contact (Boundary Analysis) in 2D or 3D.

    Per CIM MRMR Best Practice Guidelines (§4.2 Domain Modeling) and JORC Code (2012) Table 1 (TODO: Manually Verify),
    contact profile analysis (grade vs. perpendicular distance from boundary) must be conducted to establish
    whether geological contacts represent:
    - HARD BOUNDARY: A sharp step-change discontinuity at distance 0.
      Requires strict segregation during kriging/interpolation (no sample sharing).
    - SOFT BOUNDARY: A continuous gradient with no discontinuity.
      Samples from both sides can be freely shared across the contact.
    - SEMI-SOFT / TRANSITIONAL BOUNDARY: A moderate transition zone.
      Samples are shared only within a restricted buffer or with distance decay.

    Supports 3D planar contacts (dip / dip direction or plane normal vector), 3D point cloud surfaces
    (e.g. wireframe contact nodes), 2D contact line segments, or precomputed signed distance columns.

    Parameters
    ----------
    df : pd.DataFrame
        Composite or assay data table.
    domain_col : str
        Domain classification column (e.g. "lithology", "oxidation", "zone").
    grade_col : str
        Grade column name.
    distance_col : str, optional
        Signed perpendicular distance column in `df` (negative for domain_a, positive for domain_b).
        If omitted, auto-detected from 'distance' or 'dist_to_contact' if present.
    contact_surface : sequence of tuple of float, optional
        2D contact line segment [(x1, y1), (x2, y2)] used to compute perpendicular
        signed distance in 2D.
    contact_plane_point : sequence of float, optional
        Reference point (x0, y0, z0) lying on the 3D contact plane (or (x0, y0) in 2D).
    contact_dip : float, optional
        Dip angle of the 3D planar contact in degrees from horizontal (0 to 90 degrees).
    contact_dip_direction : float, optional
        Azimuth / dip direction of the 3D planar contact in degrees clockwise from North [0, 360).
    contact_normal : sequence of float, optional
        Direct unit normal vector (nx, ny, nz) perpendicular to the contact plane.
    contact_points : np.ndarray or sequence of sequence of float, optional
        3D (K, 3) or 2D (K, 2) array of digitized contact surface or wireframe points.
        Perpendicular distance is computed to the nearest contact point.
    x_col : str, default "x"
        Easting coordinate column name.
    y_col : str, default "y"
        Northing coordinate column name.
    z_col : str, optional, default "z"
        Elevation coordinate column name.
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
        Audit attributes (.attrs) include:
        - 'boundary_type': 'Hard', 'Soft', 'Semi-Soft', or 'Indeterminate'
        - 'step_change': absolute grade discontinuity across contact
        - 'step_ratio': relative grade jump ratio
        - 'recommendation': actionable kriging sharing advice
        - 'contact_type': '3D Plane', '3D Point Cloud', '2D Line Segment', or 'Precomputed Column'
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

    contact_type = "Precomputed Column"
    normal = None
    p0 = None

    # Compute or extract signed distance
    if distance_col is not None:
        if distance_col not in sub_df.columns:
            raise ValueError(f"Specified distance_col '{distance_col}' not found.")
        dists = sub_df[distance_col].to_numpy(dtype=float)
        contact_type = "Precomputed Column"
    elif (
        contact_plane_point is not None
        or contact_dip is not None
        or contact_normal is not None
    ):
        if contact_plane_point is None:
            raise ValueError(
                "Must provide 'contact_plane_point' (e.g. (x0, y0, z0)) when defining a planar contact."
            )
        p0 = np.asarray(contact_plane_point, dtype=float)
        if len(p0) not in (2, 3):
            raise ValueError(
                f"'contact_plane_point' must have length 2 or 3, got {len(p0)}."
            )

        if contact_dip is not None:
            if contact_dip < 0.0 or contact_dip > 90.0:
                raise ValueError(
                    f"contact_dip must be between 0 and 90 degrees, got {contact_dip}."
                )
            if contact_dip_direction is None:
                raise ValueError(
                    "contact_dip_direction must be provided when contact_dip is specified."
                )
            if contact_dip_direction < 0.0 or contact_dip_direction >= 360.0:
                raise ValueError(
                    f"contact_dip_direction must be in [0, 360), got {contact_dip_direction}."
                )
            dip_rad = np.radians(contact_dip)
            dd_rad = np.radians(contact_dip_direction)
            # Upward/hanging-wall unit normal:
            normal = np.array(
                [
                    np.sin(dd_rad) * np.sin(dip_rad),
                    np.cos(dd_rad) * np.sin(dip_rad),
                    np.cos(dip_rad),
                ],
                dtype=float,
            )
        elif contact_normal is not None:
            normal = np.asarray(contact_normal, dtype=float)
            norm_len = np.linalg.norm(normal)
            if norm_len <= 1e-9:
                raise ValueError("contact_normal must have non-zero length.")
            normal = normal / norm_len
        else:
            raise ValueError(
                "Must provide either ('contact_dip' and 'contact_dip_direction') or 'contact_normal' with 'contact_plane_point'."
            )

        is_3d = (
            len(p0) == 3
            or len(normal) == 3
            or (z_col is not None and z_col in sub_df.columns)
        )
        if is_3d:
            for col_name, c_label in [
                (x_col, "x_col"),
                (y_col, "y_col"),
                (z_col, "z_col"),
            ]:
                if col_name is None or col_name not in sub_df.columns:
                    raise ValueError(
                        f"Required 3D coordinate column '{col_name}' ({c_label}) not found in DataFrame."
                    )
            pts = sub_df[[x_col, y_col, z_col]].to_numpy(dtype=float)
            if len(p0) == 2:
                p0 = np.array([p0[0], p0[1], 0.0], dtype=float)
            if len(normal) == 2:
                normal = np.array([normal[0], normal[1], 0.0], dtype=float)
            contact_type = "3D Plane"
        else:
            for col_name, c_label in [(x_col, "x_col"), (y_col, "y_col")]:
                if col_name not in sub_df.columns:
                    raise ValueError(
                        f"Required coordinate column '{col_name}' ({c_label}) not found in DataFrame."
                    )
            pts = sub_df[[x_col, y_col]].to_numpy(dtype=float)
            contact_type = "Planar Contact"

        dists = np.dot(pts - p0, normal)
    elif contact_points is not None:
        c_pts = np.asarray(contact_points, dtype=float)
        if c_pts.ndim != 2 or c_pts.shape[1] not in (2, 3):
            raise ValueError(
                f"contact_points must be a 2D array/sequence of shape (K, 2) or (K, 3), got shape {c_pts.shape}."
            )
        if len(c_pts) == 0:
            raise ValueError("contact_points cannot be empty.")
        is_3d = c_pts.shape[1] == 3
        if is_3d:
            for col_name, c_label in [
                (x_col, "x_col"),
                (y_col, "y_col"),
                (z_col, "z_col"),
            ]:
                if col_name is None or col_name not in sub_df.columns:
                    raise ValueError(
                        f"Required 3D coordinate column '{col_name}' ({c_label}) not found in DataFrame."
                    )
            pts = sub_df[[x_col, y_col, z_col]].to_numpy(dtype=float)
            contact_type = "3D Point Cloud"
        else:
            for col_name, c_label in [(x_col, "x_col"), (y_col, "y_col")]:
                if col_name not in sub_df.columns:
                    raise ValueError(
                        f"Required coordinate column '{col_name}' ({c_label}) not found in DataFrame."
                    )
            pts = sub_df[[x_col, y_col]].to_numpy(dtype=float)
            contact_type = "2D Point Cloud"

        tree = cKDTree(c_pts)
        unsigned_dists, _ = tree.query(pts)
        dom_vals = sub_df[domain_col].to_numpy()
        dists = np.where(dom_vals == dom_a, -unsigned_dists, unsigned_dists)
    elif contact_surface is not None:
        # Calculate 2D signed perpendicular distance to line segment
        pts = sub_df[[x_col, y_col]].to_numpy(dtype=float)
        p1 = np.array(contact_surface[0], dtype=float)
        p2 = np.array(contact_surface[1], dtype=float)
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len <= 1e-9:
            raise ValueError("Contact surface line segment has zero length.")
        normal = np.array([-line_vec[1], line_vec[0]]) / line_len

        # Signed distance from p1 along normal
        dists = np.dot(pts - p1, normal)
        contact_type = "2D Line Segment"
    elif "distance" in sub_df.columns:
        dists = sub_df["distance"].to_numpy(dtype=float)
        contact_type = "Precomputed Column"
    elif "dist_to_contact" in sub_df.columns:
        dists = sub_df["dist_to_contact"].to_numpy(dtype=float)
        contact_type = "Precomputed Column"
    else:
        raise ValueError(
            "Must provide either 'distance_col', a 3D planar contact ('contact_plane_point' with 'contact_dip'/'contact_dip_direction' or 'contact_normal'), 'contact_points', or a 2D 'contact_surface' line segment."
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
    profile_df.attrs["max_distance"] = max_distance
    profile_df.attrs["contact_type"] = contact_type
    if contact_dip is not None:
        profile_df.attrs["contact_dip"] = float(contact_dip)
    if contact_dip_direction is not None:
        profile_df.attrs["contact_dip_direction"] = float(contact_dip_direction)
    if normal is not None:
        profile_df.attrs["contact_normal"] = normal.tolist()
    if p0 is not None:
        profile_df.attrs["contact_plane_point"] = p0.tolist()
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
    c_type = contact_df.attrs.get("contact_type", None)
    c_dip = contact_df.attrs.get("contact_dip", None)
    c_dd = contact_df.attrs.get("contact_dip_direction", None)

    if title is not None:
        plot_title = title
    else:
        plot_title = f"Geological Contact Profile Analysis: {dom_a} vs. {dom_b}"
        if c_type == "3D Plane" and c_dip is not None and c_dd is not None:
            plot_title += (
                f"\n(3D Planar Contact: Dip {c_dip:.0f}°, Dip Dir {c_dd:.0f}°)"
            )
        elif c_type is not None and c_type != "Precomputed Column":
            plot_title += f"\n({c_type})"

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

    def _extract_stockpile_delta(
        sp_row: pd.Series, g_scale: float
    ) -> Tuple[float, float]:
        """Extracts (delta_tonnes, delta_metal) from a stockpile record."""
        # Format 1: Opening & closing inventory balances
        open_t = next(
            (
                c
                for c in sp_row.index
                if c in ("opening_tonnes", "open_tonnes", "opening_t", "open_t")
            ),
            None,
        )
        close_t = next(
            (
                c
                for c in sp_row.index
                if c in ("closing_tonnes", "close_tonnes", "closing_t", "close_t")
            ),
            None,
        )
        if open_t is not None and close_t is not None:
            open_g = next(
                (
                    c
                    for c in sp_row.index
                    if c in ("opening_grade", "open_grade", "opening_g", "open_g")
                ),
                None,
            )
            close_g = next(
                (
                    c
                    for c in sp_row.index
                    if c in ("closing_grade", "close_grade", "closing_g", "close_g")
                ),
                None,
            )
            if open_g is None or close_g is None:
                raise ValueError(
                    "Stockpile opening/closing balances require both tonnes and grade columns."
                )
            t_o, g_o = float(sp_row[open_t]), float(sp_row[open_g])
            t_c, g_c = float(sp_row[close_t]), float(sp_row[close_g])
            d_t = t_c - t_o
            d_m = (t_c * g_c - t_o * g_o) / g_scale
            return d_t, d_m

        # Format 2: Movement receipts and reclaims
        add_t = next(
            (
                c
                for c in sp_row.index
                if c
                in (
                    "added_tonnes",
                    "deposit_tonnes",
                    "inflow_tonnes",
                    "feed_tonnes",
                    "add_tonnes",
                )
            ),
            None,
        )
        rec_t = next(
            (
                c
                for c in sp_row.index
                if c
                in (
                    "reclaimed_tonnes",
                    "reclaim_tonnes",
                    "drawdown_tonnes",
                    "outflow_tonnes",
                )
            ),
            None,
        )
        if add_t is not None and rec_t is not None:
            add_g = next(
                (
                    c
                    for c in sp_row.index
                    if c
                    in (
                        "added_grade",
                        "deposit_grade",
                        "inflow_grade",
                        "feed_grade",
                        "add_grade",
                    )
                ),
                None,
            )
            rec_g = next(
                (
                    c
                    for c in sp_row.index
                    if c
                    in (
                        "reclaimed_grade",
                        "reclaim_grade",
                        "drawdown_grade",
                        "outflow_grade",
                    )
                ),
                None,
            )
            if add_g is None or rec_g is None:
                raise ValueError(
                    "Stockpile movement flows require both added/reclaimed tonnes and grade columns."
                )
            t_a, g_a = float(sp_row[add_t]), float(sp_row[add_g])
            t_r, g_r = float(sp_row[rec_t]), float(sp_row[rec_g])
            d_t = t_a - t_r
            d_m = (t_a * g_a - t_r * g_r) / g_scale
            return d_t, d_m

        # Format 3: Direct deltas
        d_t_col = next(
            (
                c
                for c in sp_row.index
                if c in ("delta_tonnes", "delta_t", "stockpile_delta_tonnes", "tonnes")
            ),
            None,
        )
        if d_t_col is not None:
            d_m_col = next(
                (
                    c
                    for c in sp_row.index
                    if c in ("delta_metal", "delta_m", "stockpile_delta_metal", "metal")
                ),
                None,
            )
            if d_m_col is not None:
                return float(sp_row[d_t_col]), float(sp_row[d_m_col])
            d_g_col = next(
                (
                    c
                    for c in sp_row.index
                    if c in ("delta_grade", "delta_g", "stockpile_delta_grade", "grade")
                ),
                None,
            )
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
            raise ValueError(
                f"Period column '{p_col}' not found in stockpile_data and length does not match reserve data."
            )

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
            g_plant_adj = (
                (m_plant_adj / t_plant_adj) * grade_scale if t_plant_adj > 0 else 0.0
            )

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
            tot_g_plant_adj = (
                (tot_m_plant_adj / tot_t_plant_adj) * grade_scale
                if tot_t_plant_adj > 0
                else 0.0
            )

            tot_rec["stockpile_delta_tonnes"] = tot_d_t
            tot_rec["stockpile_delta_grade"] = tot_d_g
            tot_rec["stockpile_delta_metal"] = tot_d_m
            tot_rec["plant_adj_tonnes"] = tot_t_plant_adj
            tot_rec["plant_adj_grade"] = tot_g_plant_adj
            tot_rec["plant_adj_metal"] = tot_m_plant_adj

            if has_gc:
                tot_rec["f2_adj_tonnes_ratio"] = (
                    tot_t_plant_adj / tot_t_gc if tot_t_gc > 0 else 1.0
                )
                tot_rec["f2_adj_grade_ratio"] = (
                    tot_g_plant_adj / tot_g_gc if tot_g_gc > 0 else 1.0
                )
                tot_rec["f2_adj_metal_factor"] = (
                    tot_m_plant_adj / tot_m_gc if tot_m_gc > 0 else 1.0
                )

            tot_rec["f3_adj_tonnes_ratio"] = (
                tot_t_plant_adj / tot_t_res if tot_t_res > 0 else 1.0
            )
            tot_rec["f3_adj_grade_ratio"] = (
                tot_g_plant_adj / tot_g_res if tot_g_res > 0 else 1.0
            )
            tot_rec["f3_adj_metal_factor"] = (
                tot_m_plant_adj / tot_m_res if tot_m_res > 0 else 1.0
            )

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

    col_res = "#1f77b4"  # Blue (Reserve Model)
    col_gc = "#ff7f0e"  # Orange (Grade Control)
    col_plant = "#2ca02c"  # Green (Plant Direct Feed)
    col_plant_adj = "#9467bd"  # Purple (Plant + Stockpile Delta)

    # Offsets helper
    offsets = np.linspace(
        -width * (num_series - 1) / 2, width * (num_series - 1) / 2, num_series
    )

    # -------------------------------------------------------------------------
    # Panel 1: Ore Tonnage
    # -------------------------------------------------------------------------
    s_idx = 0
    ax_t.bar(
        x + offsets[s_idx],
        plot_df["reserve_tonnes"],
        width,
        label="Reserve Model",
        color=col_res,
        edgecolor="black",
        alpha=0.85,
    )
    s_idx += 1
    if has_gc:
        ax_t.bar(
            x + offsets[s_idx],
            plot_df["gc_tonnes"],
            width,
            label="Grade Control",
            color=col_gc,
            edgecolor="black",
            alpha=0.85,
        )
        s_idx += 1
    ax_t.bar(
        x + offsets[s_idx],
        plot_df["plant_tonnes"],
        width,
        label="Plant (Direct)",
        color=col_plant,
        edgecolor="black",
        alpha=0.85,
    )
    s_idx += 1
    if has_sp:
        ax_t.bar(
            x + offsets[s_idx],
            plot_df["plant_adj_tonnes"],
            width,
            label="Plant (Stockpile-Adj)",
            color=col_plant_adj,
            edgecolor="black",
            alpha=0.85,
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
        x + offsets[s_idx],
        plot_df["reserve_grade"],
        width,
        label="Reserve Model",
        color=col_res,
        edgecolor="black",
        alpha=0.85,
    )
    s_idx += 1
    if has_gc:
        ax_g.bar(
            x + offsets[s_idx],
            plot_df["gc_grade"],
            width,
            label="Grade Control",
            color=col_gc,
            edgecolor="black",
            alpha=0.85,
        )
        s_idx += 1
    ax_g.bar(
        x + offsets[s_idx],
        plot_df["plant_grade"],
        width,
        label="Plant (Direct)",
        color=col_plant,
        edgecolor="black",
        alpha=0.85,
    )
    s_idx += 1
    if has_sp:
        ax_g.bar(
            x + offsets[s_idx],
            plot_df["plant_adj_grade"],
            width,
            label="Plant (Stockpile-Adj)",
            color=col_plant_adj,
            edgecolor="black",
            alpha=0.85,
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
        x + offsets[s_idx],
        plot_df["reserve_metal"],
        width,
        label="Reserve Model",
        color=col_res,
        edgecolor="black",
        alpha=0.85,
    )
    s_idx += 1
    if has_gc:
        ax_m.bar(
            x + offsets[s_idx],
            plot_df["gc_metal"],
            width,
            label="Grade Control",
            color=col_gc,
            edgecolor="black",
            alpha=0.85,
        )
        s_idx += 1
    ax_m.bar(
        x + offsets[s_idx],
        plot_df["plant_metal"],
        width,
        label="Plant (Direct)",
        color=col_plant,
        edgecolor="black",
        alpha=0.85,
    )
    s_idx += 1
    if has_sp:
        ax_m.bar(
            x + offsets[s_idx],
            plot_df["plant_adj_metal"],
            width,
            label="Plant (Stockpile-Adj)",
            color=col_plant_adj,
            edgecolor="black",
            alpha=0.85,
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
                "--s" if has_sp else "-s",
                color=col_gc,
                linewidth=1.8 if has_sp else 2.0,
                markersize=5 if has_sp else 6,
                label="F2 (Direct Mine → Mill)" if has_sp else "F2 (Mine → Mill)",
            )
            if has_sp:
                ax_f.plot(
                    x,
                    plot_df["f2_adj_metal_factor"],
                    "-s",
                    color="#d62728",
                    linewidth=2.2,
                    markersize=6,
                    label="F2 (Stockpile-Adj)",
                )
        ax_f.plot(
            x,
            plot_df["f3_metal_factor"],
            "--^" if has_sp else "-^",
            color=col_plant,
            linewidth=1.8 if has_sp else 2.5,
            markersize=6 if has_sp else 7,
            label="F3 (Direct Total)" if has_sp else "F3 (Total Value Chain)",
        )
        if has_sp:
            ax_f.plot(
                x,
                plot_df["f3_adj_metal_factor"],
                "-^",
                color=col_plant_adj,
                linewidth=2.5,
                markersize=7,
                label="F3 (Stockpile-Adj Total)",
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
            range(len(cats)),
            vals,
            width=0.45,
            color=bar_c,
            edgecolor="black",
            alpha=0.85,
        )
        ax_f.set_xticks(range(len(cats)))
        ax_f.set_xticklabels(cats, fontsize=8.5, fontweight="bold")
        for idx, v in enumerate(vals):
            ax_f.text(
                idx, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold"
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


# TODO (3D Block Model Extension & LU Simulation Engine - CONTRIBUTING.md Standards):
# Implementation Engine Specification:
# - SGS should be implemented using the LU decomposition (Cholesky factorization) method
#   (Alabert 1987; Davis 1987; Journel & Huijbregts 1978; Goovaerts 1997) (TODO: Manually Verify).
# - Rationale for LU Decomposition over sequential path kriging:
#   1. Exact Covariance Reproduction: Avoids moving-neighborhood screening artifacts,
#      search radius truncations, and sequential random path order dependencies.
#   2. Vectorized Ensemble Generation: Once the lower-triangular Cholesky factor L of the
#      conditional covariance matrix C_cond is computed, generating L realizations requires only
#      drawing standard normal variates W ~ N(0, I) and evaluating Y = y_sk + L @ W via fast
#      matrix-matrix multiplication (BLAS gemm).
#   3. Deterministic & Vectorized: Eliminates Python loop overhead over M grid nodes and
#      dynamic KD-tree conditioning updates.
# Guidelines from CONTRIBUTING.md to adhere to:
# - Respect Domain Boundaries: conditional simulation must be executed independently per domain.
# - Explicit Extrapolation Control: confine simulation paths and conditioning to data envelopes.
# - 3D Anisotropic Covariance: support 3D variogram structures (major, semi-major, minor ranges).
# - Numerical Stability: Add diagonal nugget/jitter regularization (epsilon ~ 1e-6 to 1e-4) to
#   guarantee positive-definiteness of C_cond during Cholesky factorization.
# - Large Grid Scalability: For very large grids where dense M x M factorization exceeds memory,
#   partition into geological domains or moving super-blocks.
# - Functional approach: transparent NumPy arrays/DataFrames without dataclass configuration wrappers.
# - Following codebase conventions, 3D visualizations are implemented as separate functions:
#   1. `plot_simulation_3d_isometric`: dedicated 3D isometric view of simulation realizations / exceedance.
#   2. `plot_simulation_3d_interactive`: dedicated interactive 3D realization explorer.


def normal_score_transform(
    grades: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforms raw grades into standard normal Gaussian scores Y ~ N(0, 1).

    Supports declustering sample weights (Ortiz 2019; Journel 1983; Deutsch 1998)
    (TODO: Manually Verify) to ensure the representative distribution is preserved.

    Parameters
    ----------
    grades : np.ndarray
        Raw grade assays of shape (N,).
    weights : np.ndarray, optional
        Declustering weights of shape (N,). If None, uniform weights (1/N) are used.

    Returns
    -------
    y : np.ndarray
        Standard normal scores matching the original input order of grades.
    sorted_z : np.ndarray
        Sorted raw grades for empirical CDF interpolation during back-transformation.
    sorted_y : np.ndarray
        Corresponding Gaussian quantiles matching sorted_z.
    """
    z = np.asarray(grades, dtype=float)
    n = len(z)
    if n == 0:
        raise ValueError("Cannot perform normal score transform on empty grades array.")

    # 1. Normalize declustering weights
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError(f"weights length ({len(w)}) must match grades ({n}).")
        if np.any(w < 0):
            raise ValueError("Declustering weights must be non-negative.")
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            raise ValueError("Sum of declustering weights must be positive.")
        w = w / w_sum
    else:
        w = np.full(n, 1.0 / n, dtype=float)

    # 2. Sort grades and weights in ascending order
    sort_idx = np.argsort(z)
    sorted_z = z[sort_idx]
    sorted_w = w[sort_idx]

    # 3. Compute mid-point cumulative plotting positions p
    cum_w = np.cumsum(sorted_w)
    p = cum_w - 0.5 * sorted_w

    # 4. Handle identical grades (ties): assign average cumulative probability
    unique_z, inv_indices = np.unique(sorted_z, return_inverse=True)
    if len(unique_z) < n:
        avg_p = np.zeros_like(p)
        for u_idx in range(len(unique_z)):
            mask = inv_indices == u_idx
            avg_p[mask] = np.mean(p[mask])
        p = avg_p

    # Strictly bound probabilities to prevent +/- inf
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    # 5. Convert probabilities to standard normal scores
    sorted_y = stats.norm.ppf(p)

    # 6. Reconstruct y in original sample order
    inv_sort_idx = np.empty_like(sort_idx)
    inv_sort_idx[sort_idx] = np.arange(n)
    y = sorted_y[inv_sort_idx]

    return y, sorted_z, sorted_y


def normal_score_back_transform(
    y_sim: np.ndarray,
    sorted_z: np.ndarray,
    gaussian_quantiles: np.ndarray,
    lower_tail_min: Optional[float] = None,
    upper_tail_max: Optional[float] = None,
) -> np.ndarray:
    """Transforms simulated Gaussian values back to raw grade distribution.

    Uses empirical quantile interpolation (Deutsch & Journel 1998; Goovaerts 1997)
    (TODO: Manually Verify).

    Parameters
    ----------
    y_sim : np.ndarray
        Array of simulated Gaussian values (any shape, e.g. (M, L)).
    sorted_z : np.ndarray
        Sorted reference assay grades from the forward transform.
    gaussian_quantiles : np.ndarray
        Sorted standard normal quantiles matching sorted_z.
    lower_tail_min : float, optional
        Minimum grade floor for values below the lowest sample quantile (default sorted_z[0]).
    upper_tail_max : float, optional
        Maximum grade ceiling for values above the highest sample quantile (default sorted_z[-1]).

    Returns
    -------
    np.ndarray
        Back-transformed grades with identical shape to y_sim.
    """
    y_arr = np.asarray(y_sim, dtype=float)
    left_val = sorted_z[0] if lower_tail_min is None else lower_tail_min
    right_val = sorted_z[-1] if upper_tail_max is None else upper_tail_max

    z_sim = np.interp(
        y_arr,
        gaussian_quantiles,
        sorted_z,
        left=left_val,
        right=right_val,
    )
    return z_sim


# =============================================================================
# MULTIGAUSSIAN ASSUMPTION VERIFICATION (ORDER RATIOS, INDICATORS, H-SCATTER)
# =============================================================================


def _extract_spatial_lag_pairs(
    coords: np.ndarray,
    values: np.ndarray,
    lag_center: float,
    lag_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts pairs of spatial values separated by a distance lag_center +/- lag_tolerance."""
    n = len(coords)
    if n < 2:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    dists = pdist(coords)
    i_idx, j_idx = np.triu_indices(n, k=1)
    mask = (dists >= (lag_center - lag_tolerance)) & (
        dists <= (lag_center + lag_tolerance)
    )
    if not np.any(mask):
        empty = np.array([], dtype=float)
        return empty, empty, empty

    return dists[mask], values[i_idx[mask]], values[j_idx[mask]]


def check_multigaussian_order_ratios(
    coords: np.ndarray,
    normal_scores: np.ndarray,
    lag_distance: float,
    n_lags: int,
    lag_tolerance: Optional[float] = None,
) -> pd.DataFrame:
    """Verifies the multiGaussian assumption using variograms of order 1 and order 2.

    Mathematical Formulation:
    -------------------------
    Under a bivariate multiGaussian random field, the ratio between the squared
    rodogram gamma_1(h) (order 1) and the standard semivariogram gamma_2(h) (order 2)
    is an analytical constant for all separation lags h (Ortiz 2019; Goovaerts 1997;
    Deutsch & Journel 1998) (TODO: Manually Verify):
        (gamma_1(h))^2 / gamma_2(h) = 1 / pi ≈ 0.31831

    Significant deviations from 1/pi indicate non-Gaussian spatial connectivity,
    heteroscedasticity, or heavy-tailed extreme continuity.

    Parameters
    ----------
    coords : np.ndarray
        Sample spatial coordinates of shape (N, 2) or (N, 3).
    normal_scores : np.ndarray
        Standard normal scores Y ~ N(0, 1) of shape (N,).
    lag_distance : float
        Nominal separation lag distance (h).
    n_lags : int
        Number of lags to evaluate.
    lag_tolerance : float, optional
        Half-window tolerance around each nominal lag (default lag_distance / 2.0).

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - "lag": Nominal lag distance.
        - "average_distance": True centroid separation distance of paired samples in bin.
        - "gamma_1": Rodogram (order 1 variogram).
        - "gamma_2": Semivariogram (order 2 variogram).
        - "ratio": Experimental ratio (gamma_1^2 / gamma_2).
        - "theoretical_ratio": Analytical Gaussian reference value (1 / pi ≈ 0.31831).
        - "pair_count": Number of valid sample pairs in lag bin.
        Attributes (.attrs) contain mean absolute percentage error vs. 1/pi.
    """
    coords_arr = np.asarray(coords, dtype=float)
    y_arr = np.asarray(normal_scores, dtype=float)

    if len(coords_arr) != len(y_arr):
        raise ValueError(
            f"coords length ({len(coords_arr)}) must match normal_scores ({len(y_arr)})."
        )
    if lag_distance <= 0:
        raise ValueError(f"lag_distance must be positive, got {lag_distance}.")
    if n_lags < 1:
        raise ValueError(f"n_lags must be >= 1, got {n_lags}.")

    tol = lag_distance / 2.0 if lag_tolerance is None else float(lag_tolerance)
    theor_ratio = 1.0 / np.pi

    records = []
    for k in range(n_lags):
        nominal_lag = (k + 1) * lag_distance
        d_sub, y1, y2 = _extract_spatial_lag_pairs(coords_arr, y_arr, nominal_lag, tol)
        n_pairs = len(d_sub)
        if n_pairs == 0:
            records.append(
                {
                    "lag": nominal_lag,
                    "average_distance": np.nan,
                    "gamma_1": np.nan,
                    "gamma_2": np.nan,
                    "ratio": np.nan,
                    "theoretical_ratio": theor_ratio,
                    "pair_count": 0,
                }
            )
            continue

        diff = np.abs(y1 - y2)
        g1 = float(np.mean(diff) / 2.0)
        g2 = float(np.mean(diff**2) / 2.0)
        ratio = (g1**2) / g2 if g2 > 0 else np.nan

        records.append(
            {
                "lag": nominal_lag,
                "average_distance": float(np.mean(d_sub)),
                "gamma_1": g1,
                "gamma_2": g2,
                "ratio": ratio,
                "theoretical_ratio": theor_ratio,
                "pair_count": n_pairs,
            }
        )

    df = pd.DataFrame(records)
    valid_ratios = df["ratio"].dropna()
    mae = (
        float(np.mean(np.abs(valid_ratios - theor_ratio)))
        if len(valid_ratios) > 0
        else np.nan
    )
    df.attrs["mean_abs_error"] = mae
    df.attrs["theoretical_ratio"] = theor_ratio
    return df


def check_multigaussian_indicators(
    coords: np.ndarray,
    normal_scores: np.ndarray,
    lag_distance: float,
    n_lags: int,
    variogram_model: str = "spherical",
    sill: float = 1.0,
    range_param: Union[float, Sequence[float], Mapping[str, float]] = 100.0,
    nugget: float = 0.0,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    lag_tolerance: Optional[float] = None,
) -> pd.DataFrame:
    """Compares experimental indicator variograms against theoretical multiGaussian indicators.

    Mathematical Formulation:
    -------------------------
    Under a bivariate multiGaussian distribution with spatial correlation rho(h) = C(h)/C(0),
    the indicator semivariogram for any quantile p (with standard normal cutoff y_p = Phi^{-1}(p))
    is given analytically by (Goovaerts 1997 §7.2; Ortiz 2019) (TODO: Manually Verify):
        gamma_I,theor(h; p) = p - P(Y(u) <= y_p, Y(u+h) <= y_p; rho(h))

    In addition, multiGaussian indicator variograms are strictly symmetric around the median:
        gamma_I(h; p) = gamma_I(h; 1 - p)

    Parameters
    ----------
    coords : np.ndarray
        Sample spatial coordinates of shape (N, 2) or (N, 3).
    normal_scores : np.ndarray
        Standard normal scores Y ~ N(0, 1) of shape (N,).
    lag_distance : float
        Nominal separation lag distance.
    n_lags : int
        Number of lags to evaluate.
    variogram_model : str, default "spherical"
        Fitted Gaussian variogram model ("spherical", "exponential", "gaussian").
    sill : float, default 1.0
        Partial sill in Gaussian space.
    range_param : float, Sequence[float], or Mapping, default 100.0
        Variogram correlation range.
    nugget : float, default 0.0
        Nugget variance in Gaussian space.
    quantiles : Sequence[float], default (0.1, 0.25, 0.5, 0.75, 0.9)
        Quantiles to test.
    lag_tolerance : float, optional
        Lag half-window tolerance (default lag_distance / 2.0).

    Returns
    -------
    pd.DataFrame
        Table with columns:
        - "lag": Nominal lag distance.
        - "average_distance": Mean pair distance in bin.
        - "quantile": Quantile p.
        - "cutoff": Normal score cutoff y_p.
        - "exp_indicator_var": Experimental indicator semivariance.
        - "theor_indicator_var": Analytical Gaussian indicator semivariance.
        - "abs_diff": Absolute difference between experimental and theoretical.
        - "pair_count": Number of pairs in lag bin.
        Attributes (.attrs) contain mean absolute error and tail symmetry metrics.
    """
    coords_arr = np.asarray(coords, dtype=float)
    y_arr = np.asarray(normal_scores, dtype=float)
    tol = lag_distance / 2.0 if lag_tolerance is None else float(lag_tolerance)

    tot_sill = float(nugget + sill)
    if tot_sill <= 0:
        raise ValueError("Total sill (nugget + sill) must be positive.")

    records = []
    for k in range(n_lags):
        nominal_lag = (k + 1) * lag_distance
        d_sub, y1, y2 = _extract_spatial_lag_pairs(coords_arr, y_arr, nominal_lag, tol)
        n_pairs = len(d_sub)
        avg_d = float(np.mean(d_sub)) if n_pairs > 0 else nominal_lag

        # Evaluate theoretical covariance and correlation rho(h)
        cov_val = float(
            _theoretical_covariance(
                avg_d,
                model=variogram_model,
                nugget=nugget,
                sill=sill,
                range_param=range_param,
            )
        )
        rho = np.clip(cov_val / tot_sill, -0.9999, 0.9999)
        bvn_cov = np.array([[1.0, rho], [rho, 1.0]])

        for p in quantiles:
            y_p = float(stats.norm.ppf(p))

            # Theoretical indicator variogram: p - P(Y1 <= y_p, Y2 <= y_p)
            p_joint = float(
                multivariate_normal.cdf([y_p, y_p], mean=[0.0, 0.0], cov=bvn_cov)
            )
            gamma_theor = max(0.0, p - p_joint)

            # Experimental indicator variogram
            if n_pairs > 0:
                i1 = y1 <= y_p
                i2 = y2 <= y_p
                gamma_exp = float(
                    np.mean((i1.astype(float) - i2.astype(float)) ** 2) / 2.0
                )
                abs_diff = abs(gamma_exp - gamma_theor)
            else:
                gamma_exp = np.nan
                abs_diff = np.nan

            records.append(
                {
                    "lag": nominal_lag,
                    "average_distance": avg_d,
                    "quantile": p,
                    "cutoff": y_p,
                    "exp_indicator_var": gamma_exp,
                    "theor_indicator_var": gamma_theor,
                    "abs_diff": abs_diff,
                    "pair_count": n_pairs,
                }
            )

    df = pd.DataFrame(records)
    df.attrs["mean_abs_error"] = float(df["abs_diff"].dropna().mean())

    # Compute symmetry error between symmetric quantiles (e.g. 0.1 vs 0.9, 0.25 vs 0.75)
    sym_errors = []
    unique_lags = df["lag"].unique()
    for p in quantiles:
        comp_p = round(1.0 - p, 4)
        if comp_p in quantiles and comp_p > p:
            for lg in unique_lags:
                sub_p = df[(df["lag"] == lg) & (np.isclose(df["quantile"], p))]
                sub_c = df[(df["lag"] == lg) & (np.isclose(df["quantile"], comp_p))]
                if not sub_p.empty and not sub_c.empty:
                    exp_p = sub_p["exp_indicator_var"].values[0]
                    exp_c = sub_c["exp_indicator_var"].values[0]
                    if np.isfinite(exp_p) and np.isfinite(exp_c):
                        sym_errors.append(abs(exp_p - exp_c))

    df.attrs["symmetry_error"] = (
        float(np.mean(sym_errors)) if len(sym_errors) > 0 else 0.0
    )
    return df


def compute_h_scatter_statistics(
    coords: np.ndarray,
    normal_scores: np.ndarray,
    target_lags: Sequence[float],
    lag_tolerance: float,
    variogram_model: str = "spherical",
    sill: float = 1.0,
    range_param: Union[float, Sequence[float], Mapping[str, float]] = 100.0,
    nugget: float = 0.0,
) -> Dict[float, Dict[str, Any]]:
    """Extracts paired normal score samples (Y(u), Y(u+h)) for h-scatter plot diagnostics.

    Computes correlation and conditional variance homoscedasticity across quantile bins
    (Deutsch & Journel 1998; Goovaerts 1997) (TODO: Manually Verify).

    Parameters
    ----------
    coords : np.ndarray
        Sample spatial coordinates of shape (N, 2) or (N, 3).
    normal_scores : np.ndarray
        Standard normal scores Y ~ N(0, 1) of shape (N,).
    target_lags : Sequence[float]
        List of nominal lag distances to extract.
    lag_tolerance : float
        Tolerance window around each lag.
    variogram_model : str, default "spherical"
        Fitted Gaussian variogram model.
    sill : float, default 1.0
        Partial sill.
    range_param : float, Sequence, or Mapping, default 100.0
        Correlation range.
    nugget : float, default 0.0
        Nugget variance.

    Returns
    -------
    dict
        Dictionary mapping each lag distance to a dict with:
        - "y_head": Paired values at location u.
        - "y_tail": Paired values at location u + h.
        - "dists": True pair separation distances.
        - "correlation": Pearson correlation r(h).
        - "theor_correlation": Analytical correlation rho(h).
        - "n_pairs": Number of sample pairs.
    """
    coords_arr = np.asarray(coords, dtype=float)
    y_arr = np.asarray(normal_scores, dtype=float)
    tot_sill = float(nugget + sill)

    out = {}
    for lg in target_lags:
        d_sub, y1, y2 = _extract_spatial_lag_pairs(coords_arr, y_arr, lg, lag_tolerance)
        if len(d_sub) >= 2:
            # Omnidirectional h-scatter contains both directed pairs (u, u+h) and (u+h, u)
            y_h = np.concatenate([y1, y2])
            y_t = np.concatenate([y2, y1])
            d_sym = np.concatenate([d_sub, d_sub])
            r = float(np.corrcoef(y_h, y_t)[0, 1])
            avg_d = float(np.mean(d_sym))
        else:
            y_h = y1
            y_t = y2
            d_sym = d_sub
            r = np.nan
            avg_d = lg

        cov_val = float(
            _theoretical_covariance(
                avg_d,
                model=variogram_model,
                nugget=nugget,
                sill=sill,
                range_param=range_param,
            )
        )
        theor_r = np.clip(cov_val / tot_sill, -1.0, 1.0) if tot_sill > 0 else 0.0

        out[lg] = {
            "y_head": y_h,
            "y_tail": y_t,
            "dists": d_sym,
            "correlation": r,
            "theor_correlation": theor_r,
            "n_pairs": len(d_sub),
        }
    return out


def plot_multigaussian_diagnostics(
    order_ratios_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    h_scatter_data: Dict[float, Dict[str, Any]],
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (15.0, 12.0),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates a 4-panel diagnostic dashboard to evaluate the multiGaussian assumption.

    Visualizes:
    1. h-Scatter Plot: Symmetrized pairs at shortest lag with 45° equality line and correlation.
    2. Order Ratio Curve: (gamma_1^2 / gamma_2) vs. lag with 1/pi theoretical line.
    3. Indicator Variograms: Experimental vs. Theoretical curves across quantiles,
       with explicit grouping for multi-Gaussian symmetric tail counterparts (p vs 1-p).
    4. Tail Symmetry Check: Comparison of indicator curves for p vs. (1 - p).

    Parameters
    ----------
    order_ratios_df : pd.DataFrame
        Result table from check_multigaussian_order_ratios.
    indicators_df : pd.DataFrame
        Result table from check_multigaussian_indicators.
    h_scatter_data : dict
        Result dict from compute_h_scatter_statistics.
    title : str, optional
        Overall figure title.
    figsize : tuple of float, default (15.0, 12.0)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and axes sequence.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Panel 1: h-Scatter Plot at smallest lag
    ax1 = axes[0, 0]
    first_lag = min(h_scatter_data.keys()) if h_scatter_data else None
    if first_lag is not None and h_scatter_data[first_lag]["n_pairs"] > 0:
        d = h_scatter_data[first_lag]
        ax1.scatter(
            d["y_head"],
            d["y_tail"],
            alpha=0.5,
            color="royalblue",
            edgecolors="none",
            s=25,
        )
        # 45-degree diagonal reference line
        min_val = (
            min(float(np.nanmin(d["y_head"])), float(np.nanmin(d["y_tail"]))) - 0.2
        )
        max_val = (
            max(float(np.nanmax(d["y_head"])), float(np.nanmax(d["y_tail"]))) + 0.2
        )
        lims = [min_val, max_val]
        ax1.plot(lims, lims, "k--", alpha=0.7, label="1:1 Equality Line")
        ax1.set_xlim(lims)
        ax1.set_ylim(lims)
        ax1.set_title(
            f"h-Scatter Plot (h ≈ {first_lag:.1f} m)\n"
            f"Exp r = {d['correlation']:.3f} | Theor ρ = {d['theor_correlation']:.3f}"
        )
        ax1.set_xlabel("Y(u) [Normal Score]")
        ax1.set_ylabel("Y(u + h) [Normal Score]")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="upper left")
    else:
        ax1.text(0.5, 0.5, "No pairs for h-scatter", ha="center", va="center")

    # 2. Panel 2: Order Ratio Curve vs 1/pi
    ax2 = axes[0, 1]
    ax2.plot(
        order_ratios_df["lag"],
        order_ratios_df["ratio"],
        "o-",
        color="crimson",
        label="Experimental (γ₁² / γ₂)",
    )
    theor_val = 1.0 / np.pi
    ax2.axhline(
        theor_val,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label=f"MultiGaussian 1/π ≈ {theor_val:.3f}",
    )
    mae = order_ratios_df.attrs.get("mean_abs_error", np.nan)
    ax2.set_title(f"Order Ratio Test: γ₁² / γ₂ (MAE vs 1/π = {mae:.3f})")
    ax2.set_xlabel("Lag Distance h (m)")
    ax2.set_ylabel("Order Ratio")
    ax2.set_ylim([max(0.0, theor_val - 0.25), theor_val + 0.25])
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="best")

    # 3. Panel 3: Indicator Variograms (Exp vs Theor)
    # Note: In multi-Gaussian theory, gamma_I(h; p) == gamma_I(h; 1-p).
    ax3 = axes[1, 0]
    unique_quantiles = sorted(indicators_df["quantile"].unique())

    # Map of styles and distinct colors for each quantile
    q_styles = {
        0.10: {
            "color": "tab:blue",
            "marker": "o",
            "label_exp": "Exp p=0.10 (Lower Decile)",
        },
        0.25: {
            "color": "forestgreen",
            "marker": "s",
            "label_exp": "Exp p=0.25 (Lower Quartile)",
        },
        0.50: {"color": "firebrick", "marker": "D", "label_exp": "Exp p=0.50 (Median)"},
        0.75: {
            "color": "mediumpurple",
            "marker": "^",
            "label_exp": "Exp p=0.75 (Upper Quartile)",
        },
        0.90: {
            "color": "tab:cyan",
            "marker": "v",
            "label_exp": "Exp p=0.90 (Upper Decile)",
        },
    }

    # Plot theoretical curves for unique symmetric pairs
    plotted_theor = set()
    for p in unique_quantiles:
        comp_p = round(1.0 - p, 4)
        theor_key = tuple(sorted([p, comp_p]))
        if theor_key not in plotted_theor:
            plotted_theor.add(theor_key)
            sub = indicators_df[np.isclose(indicators_df["quantile"], p)].sort_values(
                "lag"
            )
            if p == 0.5:
                theor_label = "Theor p=0.50 (Median)"
                theor_color = "black"
                theor_lw = 2.0
            elif p < 0.5:
                theor_label = f"Theor p={p:.2f} & {comp_p:.2f} (Symmetric)"
                theor_color = "dimgray" if p == 0.25 else "darkslategray"
                theor_lw = 1.8
            else:
                continue

            ax3.plot(
                sub["lag"],
                sub["theor_indicator_var"],
                "-",
                color=theor_color,
                linewidth=theor_lw,
                label=theor_label,
            )

    # Plot all experimental curves with distinct markers and colors
    for p in unique_quantiles:
        sub = indicators_df[np.isclose(indicators_df["quantile"], p)].sort_values("lag")
        cfg = q_styles.get(
            p, {"color": "grey", "marker": "o", "label_exp": f"Exp p={p:.2f}"}
        )
        ax3.plot(
            sub["lag"],
            sub["exp_indicator_var"],
            "--",
            marker=cfg["marker"],
            color=cfg["color"],
            alpha=0.85,
            markersize=5,
            label=cfg["label_exp"],
        )

    ax3.set_title("Indicator Semivariograms (Solid: Theor, Markers: Exp)")
    ax3.set_xlabel("Lag Distance h (m)")
    ax3.set_ylabel("Indicator Semivariance γ_I(h)")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7.5, ncol=2)

    # 4. Panel 4: Tail Symmetry Check (p vs 1-p)
    ax4 = axes[1, 1]
    tested_pairs = [(0.1, 0.9), (0.25, 0.75)]
    pair_palettes = [
        {
            "p1_color": "tab:blue",
            "p2_color": "tab:cyan",
            "name": "Deciles (p=0.10 vs 0.90)",
        },
        {
            "p1_color": "forestgreen",
            "p2_color": "mediumpurple",
            "name": "Quartiles (p=0.25 vs 0.75)",
        },
    ]

    for (p1, p2), palette in zip(tested_pairs, pair_palettes):
        sub1 = indicators_df[np.isclose(indicators_df["quantile"], p1)].sort_values(
            "lag"
        )
        sub2 = indicators_df[np.isclose(indicators_df["quantile"], p2)].sort_values(
            "lag"
        )
        if not sub1.empty and not sub2.empty:
            ax4.plot(
                sub1["lag"],
                sub1["exp_indicator_var"],
                "s-",
                color=palette["p1_color"],
                label=f"Exp p={p1:.2f}",
            )
            ax4.plot(
                sub2["lag"],
                sub2["exp_indicator_var"],
                "^--",
                color=palette["p2_color"],
                label=f"Exp p={p2:.2f}",
            )

    sym_err = indicators_df.attrs.get("symmetry_error", np.nan)
    ax4.set_title(
        f"Tail Symmetry Check: γ_I(p) vs γ_I(1-p)\n"
        f"MultiGaussian Symmetry Error = {sym_err:.4f}"
    )
    ax4.set_xlabel("Lag Distance h (m)")
    ax4.set_ylabel("Indicator Semivariance")
    ax4.grid(True, linestyle=":", alpha=0.5)
    ax4.legend(loc="upper left", fontsize=8, ncol=2)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    fig.tight_layout()
    return fig, axes.flatten()


def _upscale_realizations_to_blocks(
    point_realizations: np.ndarray,
    n_blocks: int,
    points_per_block: int,
) -> np.ndarray:
    """Upscales simulated point realizations to block support via linear arithmetic averaging.

    Parameters
    ----------
    point_realizations : np.ndarray
        Array of simulated point values of shape (n_blocks * points_per_block, n_realizations).
    n_blocks : int
        Number of blocks in the target block model.
    points_per_block : int
        Number of discretization points per block (M = nx * ny * nz).

    Returns
    -------
    block_realizations : np.ndarray
        Array of upscaled block realizations of shape (n_blocks, n_realizations).
    """
    n_points, n_real = point_realizations.shape
    if n_points != n_blocks * points_per_block:
        raise ValueError(
            f"Point realizations count ({n_points}) must match n_blocks ({n_blocks}) * "
            f"points_per_block ({points_per_block}) = {n_blocks * points_per_block}."
        )
    reshaped = point_realizations.reshape(n_blocks, points_per_block, n_real)
    return np.nanmean(reshaped, axis=1)


def lu_gaussian_grid_simulation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    sill: float,
    range_param: Union[float, Sequence[float], Mapping[str, float]],
    n_realizations: int = 50,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
    mask_extrapolation: bool = False,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Generates equiprobable conditional realizations on grid support via LU Gaussian Simulation.

    Implementation Strategy:
    ------------------------
    Implements conditional Gaussian simulation using the LU decomposition (Cholesky factorization)
    method (Alabert 1987; Davis 1987; Journel & Huijbregts 1978; Goovaerts 1997; GSLIB lusim;
    SME Mining Engineering Handbook §4.4/§4.5) (TODO: Manually Verify).

    Mathematical Formulation of the LU Method:
    -------------------------------------------
    1. Normal-Score Transformation:
       Conditioning assay grades Z(x) are transformed to standard normal space:
       Y(x_i) = Phi^{-1}(F^*(Z(x_i))) ~ N(0, 1)
       Preserves declustered representative distributions (Ortiz 2019; Journel 1983)
       (TODO: Manually Verify).

    2. Joint Covariance Matrix Partitioning:
       The spatial covariance matrix between conditioning data points (1) and
       target simulation grid points (2) is partitioned as:
       C = [[C_11, C_12],
            [C_21, C_22]]
       where:
       - C_11 is (N, N): sample-to-sample covariance.
       - C_12 = C_21^T is (N, M): sample-to-grid cross-covariance.
       - C_22 is (M, M): grid-to-grid covariance.

    3. Simple Kriging Mean Vector:
       The conditional expectation at grid points in Gaussian space (global mean mu = 0):
       y_SK = C_21 @ inv(C_11) @ y_1

    4. Conditional Covariance Matrix (Schur Complement):
       The exact conditional covariance matrix among the grid points given the data is:
       C_cond = C_22 - C_21 @ inv(C_11) @ C_12

    5. LU / Cholesky Factorization:
       Factorize the conditional covariance matrix (with numerical jitter eps * I):
       C_cond + eps * I = L @ L^T
       where L is an (M, M) lower-triangular matrix.

    6. Simultaneous Vectorized Ensemble Generation:
       For L equiprobable realizations, draw an (M, L) matrix of independent standard
       normal variates W ~ N(0, I):
       Y_sim = y_SK[:, None] + L @ W
       Generates all realizations in a single BLAS matrix-matrix multiplication (gemm).

    7. Back-Transformation:
       Simulated values in Gaussian space are transformed back to original grade support:
       Z_sim = F^{*-1}(Phi(Y_sim))

    Parameters
    ----------
    samples_xy : np.ndarray
        Array of shape (N, 2) or (N, 3) with conditioning sample coordinates.
    sample_grades : np.ndarray
        Array of shape (N,) with conditioning assay grades.
    grid_points : np.ndarray
        Array of shape (M, 2) or (M, 3) with target simulation grid coordinates.
    sill : float
        Variogram sill (C) in Gaussian space (or data space).
    range_param : float, Sequence[float], or Mapping
        Variogram range (a).
    n_realizations : int, default 50
        Number of equiprobable stochastic realizations to generate.
    variogram_model : str, default "spherical"
        Spatial correlation structure ("spherical", "exponential", "gaussian").
    nugget : float, default 0.0
        Nugget variance (C0).
    sample_weights : np.ndarray, optional
        Declustering weights of shape (N,) to honor representative distribution.
    seed : int, optional
        Random number generator seed for reproducible simulations.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional ranges (major, minor) in 2D or (major, semi-major, minor) in 3D.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles: azimuth in 2D or (azimuth, dip, plunge) in 3D.
    mask_extrapolation : bool, default False
        If True, masks grid nodes outside the sample convex hull to NaN.
    jitter : float, default 1e-6
        Diagonal regularization added to C_cond to guarantee numerical positive-definiteness.

    Returns
    -------
    np.ndarray
        Array of shape (M, n_realizations) containing the back-transformed simulated realizations.
    """
    samples = np.asarray(samples_xy, dtype=float)
    grades = np.asarray(sample_grades, dtype=float)
    grid = np.asarray(grid_points, dtype=float)

    if len(samples) != len(grades):
        raise ValueError(
            f"samples_xy length ({len(samples)}) must match sample_grades ({len(grades)})."
        )
    if grid.ndim != 2:
        raise ValueError(f"grid_points must be a 2D array, got shape {grid.shape}.")
    if samples.ndim != 2 or samples.shape[1] != grid.shape[1]:
        raise ValueError(
            f"Spatial dimensions must match: samples {samples.shape}, grid {grid.shape}."
        )
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got {n_realizations}.")
    if len(samples) == 0:
        raise ValueError("Cannot perform conditional simulation with 0 samples.")
    if len(grid) == 0:
        return np.zeros((0, n_realizations), dtype=float)

    # 1. Transform raw grades into standard normal space Y ~ N(0, 1)
    y_samples, sorted_z, sorted_y = normal_score_transform(
        grades, weights=sample_weights
    )

    # 2. Transform spatial coordinates to isotropic equivalent space if anisotropic
    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (
            range_param if isinstance(range_param, (Sequence, Mapping)) else range_param
        )
    )
    samples_t, eff_range = transform_anisotropic_coordinates(
        samples, ranges=eff_ranges, angles=anisotropy_angles
    )
    samples_t, y_samples, _, _ = _deduplicate_collocated_samples(samples_t, y_samples)
    grid_t, _ = transform_anisotropic_coordinates(
        grid, ranges=eff_ranges, angles=anisotropy_angles
    )

    # 3. Covariance parameters in Gaussian space (total sill = 1.0)
    tot_sill = float(nugget + sill)
    if tot_sill <= 0:
        raise ValueError(
            f"Total sill (nugget + sill) must be positive, got {tot_sill}."
        )
    g_nugget = float(nugget) / tot_sill
    g_sill = 1.0 - g_nugget

    # 4. Assembling joint covariance blocks C11, C12, C22
    d11 = cdist(samples_t, samples_t)
    C11 = _theoretical_covariance(
        d11,
        model=variogram_model,
        nugget=g_nugget,
        sill=g_sill,
        range_param=eff_range,
    )
    np.fill_diagonal(C11, C11.diagonal() + 1e-9)

    d12 = cdist(samples_t, grid_t)
    C12 = _theoretical_covariance(
        d12,
        model=variogram_model,
        nugget=g_nugget,
        sill=g_sill,
        range_param=eff_range,
    )

    d22 = cdist(grid_t, grid_t)
    C22 = _theoretical_covariance(
        d22,
        model=variogram_model,
        nugget=g_nugget,
        sill=g_sill,
        range_param=eff_range,
    )

    # 5. Simple Kriging Mean: y_sk = C21 @ inv(C11) @ y_samples
    weights = np.linalg.solve(C11, C12)  # shape (N, M)
    y_sk = weights.T @ y_samples  # shape (M,)

    # 6. Exact Conditional Covariance (Schur Complement): C_cond = C22 - C21 @ weights
    C_cond = C22 - C12.T @ weights
    # Symmetrize
    C_cond = 0.5 * (C_cond + C_cond.T)
    # Diagonal regularization for Cholesky positive-definiteness
    np.fill_diagonal(C_cond, np.maximum(0.0, C_cond.diagonal()) + jitter)

    # 7. Cholesky Factorization
    L = np.linalg.cholesky(C_cond)

    # 8. Vectorized Realization Ensemble Generation: Y_sim = y_sk + L @ W
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(size=(len(grid), n_realizations))
    Y_sim = y_sk[:, None] + L @ w

    # 9. Normal Score Back-Transformation
    Z_sim = normal_score_back_transform(Y_sim, sorted_z, sorted_y)

    # 10. Extrapolation Masking
    if mask_extrapolation:
        inside = is_within_convex_hull(samples, grid)
        Z_sim[~inside, :] = np.nan

    return Z_sim


def lu_gaussian_block_simulation(
    samples_xyz: np.ndarray,
    sample_grades: np.ndarray,
    block_model: pd.DataFrame,
    sill: float,
    range_param: Union[float, Sequence[float], Mapping[str, float]],
    n_realizations: int = 50,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    discretization: Tuple[int, int, int] = (4, 4, 2),
    sample_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Generates equiprobable conditional realizations on 3D block support via LU Gaussian Simulation.

    Simulates at quasi-point support on discretization sub-points within each block,
    and linearly averages the simulated point grades within each block to preserve
    mass balance and correct block dispersion variance (Goovaerts 1997; Ortiz 2019).

    Parameters
    ----------
    samples_xyz : np.ndarray
        Array of shape (N, 3) with drillhole sample coordinates.
    sample_grades : np.ndarray
        Array of shape (N,) with continuous raw assay grades.
    block_model : pd.DataFrame
        DataFrame with mining block centroids ('x', 'y', 'z') and block dimensions ('dx', 'dy', 'dz').
    sill : float
        Variogram partial sill variance c.
    range_param : float, Sequence[float], or Mapping
        Variogram practical range a.
    n_realizations : int, default 50
        Number of stochastic realizations L to simulate.
    variogram_model : str, default "spherical"
        Analytical variogram model type ('spherical', 'exponential', 'gaussian', 'cubic').
    nugget : float, default 0.0
        Variogram nugget effect c0.
    discretization : Tuple[int, int, int], default (4, 4, 2)
        Discretization resolution (nx, ny, nz) along each axis per block.
    sample_weights : np.ndarray, optional
        Declustering weights of shape (N,) for normal score transformation.
    seed : int, optional
        Random seed for reproducible Monte Carlo draws.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Ranges along principal axes for 3D geometric anisotropy.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth, dip, plunge) for 3D geometric anisotropy.
    jitter : float, default 1e-6
        Diagonal regularization constant to guarantee positive-definiteness.

    Returns
    -------
    block_realizations : np.ndarray
        Array of shape (n_blocks, n_realizations) containing simulated block grades.
    """
    req_cols = {"x", "y", "z", "dx", "dy", "dz"}
    if not req_cols.issubset(block_model.columns):
        raise ValueError(
            f"block_model must contain columns {req_cols}, got {list(block_model.columns)}."
        )

    n_blocks = len(block_model)
    if n_blocks == 0:
        return np.zeros((0, n_realizations), dtype=float)

    nx, ny, nz = discretization
    dx = float(block_model["dx"].iloc[0])
    dy = float(block_model["dy"].iloc[0])
    dz = float(block_model["dz"].iloc[0])

    disc_offsets = _generate_block_discretization_offsets(
        discretization=discretization, block_size=(dx, dy, dz)
    )
    block_coords = block_model[["x", "y", "z"]].to_numpy(dtype=float)

    # Discretization sub-points for all blocks of shape (n_blocks * M, 3)
    all_disc_points = (block_coords[:, None, :] + disc_offsets[None, :, :]).reshape(
        -1, 3
    )

    # Simulate quasi-point values at all discretization locations
    point_sim = lu_gaussian_grid_simulation(
        samples_xy=samples_xyz,
        sample_grades=sample_grades,
        grid_points=all_disc_points,
        sill=sill,
        range_param=range_param,
        n_realizations=n_realizations,
        variogram_model=variogram_model,
        nugget=nugget,
        sample_weights=sample_weights,
        seed=seed,
        anisotropy_ranges=anisotropy_ranges,
        anisotropy_angles=anisotropy_angles,
        mask_extrapolation=False,
        jitter=jitter,
    )

    # Linearly upscale simulated point grades to block support
    return _upscale_realizations_to_blocks(
        point_realizations=point_sim,
        n_blocks=n_blocks,
        points_per_block=len(disc_offsets),
    )


def _build_multigrid_levels(
    grid_points: np.ndarray,
    n_multigrids: int = 1,
) -> List[np.ndarray]:
    """Partitions grid node indices into hierarchical nested multigrid levels.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of shape (M, 2) or (M, 3) with target grid coordinates.
    n_multigrids : int, default 1
        Number of dyadic multigrid levels (1 = single flat grid, >= 2 = nested grids).

    Returns
    -------
    levels : List[np.ndarray]
        List of 1D integer arrays containing node indices for each multigrid level,
        ordered from coarsest (Level G) to finest (Level 1).
    """
    n_nodes = len(grid_points)
    if n_nodes == 0 or n_multigrids <= 1:
        return [np.arange(n_nodes)]

    dim = grid_points.shape[1]
    int_indices = np.zeros((n_nodes, dim), dtype=int)
    is_regular = True

    for d in range(dim):
        coords_d = grid_points[:, d]
        unique_d = np.unique(np.round(coords_d, decimals=5))
        if len(unique_d) <= 1:
            int_indices[:, d] = 0
            continue
        diffs = np.diff(unique_d)
        min_spacing = float(np.min(diffs))
        if min_spacing <= 1e-6 or not np.allclose(
            diffs, min_spacing, rtol=1e-2, atol=1e-4
        ):
            is_regular = False
            break
        int_indices[:, d] = np.round((coords_d - unique_d[0]) / min_spacing).astype(int)

    if not is_regular:
        warnings.warn(
            f"Cannot construct {n_multigrids} multigrids: grid_points do not form a regular "
            "orthogonal grid spacing. Falling back to a single random path (n_multigrids=1).",
            UserWarning,
            stacklevel=2,
        )
        return [np.arange(n_nodes)]

    # Construct dyadic levels from coarsest (2^(G-1)) down to 1
    levels = []
    assigned = np.zeros(n_nodes, dtype=bool)

    for g in range(n_multigrids, 0, -1):
        stride = 2 ** (g - 1)
        matches_stride = np.all(int_indices % stride == 0, axis=1)
        level_nodes = np.where(matches_stride & ~assigned)[0]
        if len(level_nodes) > 0:
            levels.append(level_nodes)
            assigned[level_nodes] = True

    remaining = np.where(~assigned)[0]
    if len(remaining) > 0:
        levels.append(remaining)

    return levels


def sequential_gaussian_grid_simulation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    grid_points: np.ndarray,
    sill: float,
    range_param: Union[float, Sequence[float], Mapping[str, float]],
    n_realizations: int = 50,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    sample_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
    search_radius: Optional[float] = None,
    max_hard_samples: int = 12,
    max_simulated_nodes: int = 12,
    max_per_sector: Optional[int] = None,
    max_per_drillhole: Optional[int] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
    assign_data_to_nodes: bool = False,
    n_multigrids: int = 1,
    mask_extrapolation: bool = False,
) -> np.ndarray:
    """Generates equiprobable conditional realizations via Sequential Gaussian Simulation (SGS).

    Implementation Strategy:
    ------------------------
    Implements Sequential Gaussian Simulation (Gómez-Hernández & Journel 1993; Deutsch &
    Journel 1998, GSLIB sgsim; Rossi & Deutsch 2014) (TODO: Manually Verify).

    Algorithm Workflow:
    -------------------
    1. Transform raw conditioning assay grades Z(x) into standard normal space Y ~ N(0, 1)
       preserving declustered representative distributions (Ortiz 2019; Journel 1983).
    2. Transform coordinates to isotropic equivalent space if 2D/3D anisotropy is specified.
    3. Standardize Gaussian variogram parameters (total sill = 1.0).
    4. Build static spatial search trees for drillhole samples and target grid coordinates.
    5. For each realization l in 1, ..., L:
       (a) Generate a structured multigrid random path visiting unsimulated grid nodes from
           coarsest level to finest level without repetition (Gómez-Hernández & Srivastava 1990; Tran 1994).
       (b) Search for conditioning data: either dual search (original drillholes + previously simulated
           nodes) or single grid search if `assign_data_to_nodes=True`.
       (c) Solve the local Simple Kriging system via the shared `_solve_local_simple_kriging_grid`
           kernel to determine conditional mean y_SK(u) and kriging variance sigma^2_SK(u).
       (d) Draw a simulated Gaussian value Y(u) ~ N(y_SK(u), sigma^2_SK(u)) and record it
           as a simulated conditioning node for subsequent nodes on the path.
    6. Back-transform all simulated values to original grade support.
    7. Mask extrapolation nodes outside the convex hull if requested.

    Parameters
    ----------
    samples_xy : np.ndarray
        Array of shape (N, 2) or (N, 3) with conditioning sample coordinates.
    sample_grades : np.ndarray
        Array of shape (N,) with conditioning assay grades.
    grid_points : np.ndarray
        Array of shape (M, 2) or (M, 3) with target simulation grid coordinates.
    sill : float
        Variogram sill (C) in Gaussian space.
    range_param : float, Sequence[float], or Mapping
        Variogram range (a).
    n_realizations : int, default 50
        Number of equiprobable stochastic realizations to generate.
    variogram_model : str, default "spherical"
        Spatial correlation structure ("spherical", "exponential", "gaussian").
    nugget : float, default 0.0
        Nugget variance (C0).
    sample_weights : np.ndarray, optional
        Declustering weights of shape (N,) to honor representative distribution.
    seed : int, optional
        Random number generator seed for reproducible simulations.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Directional ranges (major, minor) in 2D or (major, semi-major, minor) in 3D.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles: azimuth in 2D or (azimuth, dip, plunge) in 3D.
    search_radius : float, optional
        Maximum search distance for conditioning data. If None, uses 1.5 * effective range.
    max_hard_samples : int, default 12
        Maximum number of original drillhole samples to condition upon at each node.
    max_simulated_nodes : int, default 12
        Maximum number of previously simulated nodes to condition upon at each node.
    max_per_sector : int, optional
        Maximum samples per angular sector (quadrant in 2D, octant in 3D).
    max_per_drillhole : int, optional
        Maximum samples allowed per drillhole trace.
    sample_hole_ids : Sequence[Any], optional
        Drillhole trace identifiers for conditioning samples.
    assign_data_to_nodes : bool, default False
        If True, pre-assigns drillhole samples to their closest grid nodes, treating them as
        pre-conditioned nodes and removing them from the simulation path. Replaces the dual
        search with a single grid search, accelerating execution on dense grids (GSLIB sgsim).
    n_multigrids : int, default 1
        Number of dyadic multiple grid levels (Gómez-Hernández & Srivastava 1990; Tran 1994;
        GSLIB nmult). If > 1, simulates coarse sub-grids first to lock in long-range spatial
        correlation structures before progressively refining to finer grids.
    mask_extrapolation : bool, default False
        If True, masks grid nodes outside the sample convex hull to NaN.

    Returns
    -------
    np.ndarray
        Array of shape (M, n_realizations) containing the back-transformed simulated realizations.
    """
    samples = np.asarray(samples_xy, dtype=float)
    grades = np.asarray(sample_grades, dtype=float)
    grid = np.asarray(grid_points, dtype=float)

    if len(samples) != len(grades):
        raise ValueError(
            f"samples_xy length ({len(samples)}) must match sample_grades ({len(grades)})."
        )
    if grid.ndim != 2:
        raise ValueError(f"grid_points must be a 2D array, got shape {grid.shape}.")
    if samples.ndim != 2 or samples.shape[1] != grid.shape[1]:
        raise ValueError(
            f"Spatial dimensions must match: samples {samples.shape}, grid {grid.shape}."
        )
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got {n_realizations}.")
    if len(samples) == 0:
        raise ValueError("Cannot perform conditional simulation with 0 samples.")
    if len(grid) == 0:
        return np.zeros((0, n_realizations), dtype=float)
    if max_hard_samples < 1:
        raise ValueError(f"max_hard_samples must be >= 1, got {max_hard_samples}.")
    if max_simulated_nodes < 0:
        raise ValueError(
            f"max_simulated_nodes must be >= 0, got {max_simulated_nodes}."
        )
    if search_radius is not None and search_radius <= 0:
        raise ValueError(f"search_radius must be positive, got {search_radius}.")

    # 1. Transform raw grades into standard normal space Y ~ N(0, 1)
    y_samples, sorted_z, sorted_y = normal_score_transform(
        grades, weights=sample_weights
    )

    # 2. Transform spatial coordinates to isotropic equivalent space if anisotropic
    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (
            range_param if isinstance(range_param, (Sequence, Mapping)) else range_param
        )
    )
    samples_t, eff_range = transform_anisotropic_coordinates(
        samples, ranges=eff_ranges, angles=anisotropy_angles
    )
    grid_t, _ = transform_anisotropic_coordinates(
        grid, ranges=eff_ranges, angles=anisotropy_angles
    )

    # 3. Covariance parameters in Gaussian space (total sill = 1.0)
    tot_sill = float(nugget + sill)
    if tot_sill <= 0:
        raise ValueError(
            f"Total sill (nugget + sill) must be positive, got {tot_sill}."
        )
    g_nugget = float(nugget) / tot_sill
    g_sill = 1.0 - g_nugget

    radius = search_radius if search_radius is not None else 1.5 * eff_range

    # 4. Spatial Indexing
    tree_hard = cKDTree(samples_t)
    tree_grid = cKDTree(grid_t)

    k_h_query = min(len(samples_t), max_hard_samples * 4)
    k_s_query = min(len(grid_t), max(max_simulated_nodes * 8, 32))
    hole_ids_arr = list(sample_hole_ids) if sample_hole_ids is not None else None

    # Step 4(b) - Pre-assign sample data to closest grid nodes if requested (single grid search mode)
    if assign_data_to_nodes:
        _, snap_nodes = tree_grid.query(samples_t, k=1)
        unique_nodes, inverse_idx = np.unique(snap_nodes, return_inverse=True)
        assigned_values = np.zeros(len(unique_nodes), dtype=float)
        for u_i in range(len(unique_nodes)):
            assigned_values[u_i] = float(np.mean(y_samples[inverse_idx == u_i]))

        pre_simulated_mask = np.zeros(len(grid_t), dtype=bool)
        pre_simulated_mask[unique_nodes] = True
        base_y_sim = np.zeros(len(grid_t), dtype=float)
        base_y_sim[unique_nodes] = assigned_values
        unsimulated_indices = np.where(~pre_simulated_mask)[0]

    rng = np.random.default_rng(seed)
    Y_sim = np.zeros((len(grid), n_realizations), dtype=float)

    # Pre-partition multigrid levels
    multigrid_levels = _build_multigrid_levels(grid, n_multigrids=n_multigrids)

    # 5. Sequential Simulation Loop across Realizations
    for r in range(n_realizations):
        if assign_data_to_nodes:
            # Within each multigrid level, visit unsimulated nodes in randomized order
            path_segments = []
            for lvl in multigrid_levels:
                unsim_lvl = lvl[~pre_simulated_mask[lvl]]
                if len(unsim_lvl) > 0:
                    path_segments.append(rng.permutation(unsim_lvl))
            path = (
                np.concatenate(path_segments)
                if len(path_segments) > 0
                else np.empty(0, dtype=int)
            )
            y_sim_current = base_y_sim.copy()
            is_simulated = pre_simulated_mask.copy()
        else:
            path_segments = [
                rng.permutation(lvl) for lvl in multigrid_levels if len(lvl) > 0
            ]
            path = (
                np.concatenate(path_segments)
                if len(path_segments) > 0
                else np.empty(0, dtype=int)
            )
            y_sim_current = np.zeros(len(grid_t), dtype=float)
            is_simulated = np.zeros(len(grid_t), dtype=bool)

        for node_idx in path:
            u_0 = grid_t[node_idx]

            if assign_data_to_nodes:
                # Single Search: Query already assigned/simulated grid nodes only
                k_s_query_eff = min(
                    len(grid_t), max((max_hard_samples + max_simulated_nodes) * 4, 32)
                )
                d_s, idx_s = tree_grid.query(
                    u_0, k=k_s_query_eff, distance_upper_bound=radius
                )
                if k_s_query_eff == 1:
                    d_s = np.atleast_1d(d_s)
                    idx_s = np.atleast_1d(idx_s)
                in_bounds_s = idx_s < len(grid_t)
                valid_s = np.zeros(len(idx_s), dtype=bool)
                valid_s[in_bounds_s] = (
                    (idx_s[in_bounds_s] != node_idx)
                    & np.isfinite(d_s[in_bounds_s])
                    & is_simulated[idx_s[in_bounds_s]]
                )
                k_total_cap = max_hard_samples + max_simulated_nodes
                idx_s = idx_s[valid_s][:k_total_cap]
                cand_dist = d_s[valid_s][:k_total_cap]
                cand_coords = grid_t[idx_s]
                cand_vals = y_sim_current[idx_s]
                cand_holes = None
            else:
                # Step 5(b) - Query hard samples
                d_h, idx_h = tree_hard.query(
                    u_0, k=k_h_query, distance_upper_bound=radius
                )
                if k_h_query == 1:
                    d_h = np.atleast_1d(d_h)
                    idx_h = np.atleast_1d(idx_h)
                in_bounds_h = idx_h < len(samples_t)
                valid_h = np.zeros(len(idx_h), dtype=bool)
                valid_h[in_bounds_h] = np.isfinite(d_h[in_bounds_h])
                idx_h = idx_h[valid_h][:max_hard_samples]
                d_h = d_h[valid_h][:max_hard_samples]
                coords_h = samples_t[idx_h]
                vals_h = y_samples[idx_h]
                holes_h = (
                    [hole_ids_arr[i] for i in idx_h]
                    if hole_ids_arr is not None
                    else None
                )

                # Query previously simulated nodes
                d_s, idx_s = tree_grid.query(
                    u_0, k=k_s_query, distance_upper_bound=radius
                )
                if k_s_query == 1:
                    d_s = np.atleast_1d(d_s)
                    idx_s = np.atleast_1d(idx_s)
                in_bounds_s = idx_s < len(grid_t)
                valid_s = np.zeros(len(idx_s), dtype=bool)
                valid_s[in_bounds_s] = (
                    (idx_s[in_bounds_s] != node_idx)
                    & np.isfinite(d_s[in_bounds_s])
                    & is_simulated[idx_s[in_bounds_s]]
                )
                idx_s = idx_s[valid_s][:max_simulated_nodes]
                d_s = d_s[valid_s][:max_simulated_nodes]
                coords_s = grid_t[idx_s]
                vals_s = y_sim_current[idx_s]
                holes_s = (
                    [f"sim_{j}" for j in idx_s] if hole_ids_arr is not None else None
                )

                # Combine candidate conditioning data
                if len(coords_h) > 0 and len(coords_s) > 0:
                    cand_coords = np.vstack([coords_h, coords_s])
                    cand_vals = np.concatenate([vals_h, vals_s])
                    cand_dist = np.concatenate([d_h, d_s])
                    s_order = np.argsort(cand_dist)
                    cand_coords = cand_coords[s_order]
                    cand_vals = cand_vals[s_order]
                    cand_dist = cand_dist[s_order]
                    if hole_ids_arr is not None:
                        all_holes = list(holes_h) + list(holes_s)
                        cand_holes = [all_holes[idx] for idx in s_order]
                    else:
                        cand_holes = None
                elif len(coords_h) > 0:
                    cand_coords = coords_h
                    cand_vals = vals_h
                    cand_dist = d_h
                    cand_holes = holes_h
                elif len(coords_s) > 0:
                    cand_coords = coords_s
                    cand_vals = vals_s
                    cand_dist = d_s
                    cand_holes = holes_s
                else:
                    cand_coords = np.empty((0, u_0.shape[0]))
                    cand_vals = np.empty(0)
                    cand_dist = np.empty(0)
                    cand_holes = None

            # Step 5(c) - Solve Local Simple Kriging via shared kernel
            mu_sk, var_sk = _solve_local_simple_kriging_grid(
                target_coord=u_0,
                candidate_coords=cand_coords,
                candidate_values=cand_vals,
                candidate_distances=cand_dist,
                candidate_hole_ids=cand_holes,
                variogram_model=variogram_model,
                nugget=g_nugget,
                sill=g_sill,
                range_param=eff_range,
                k_neighbors=max_hard_samples + max_simulated_nodes,
                max_per_sector=max_per_sector,
                min_sectors=1,
                max_per_drillhole=max_per_drillhole,
                mean=0.0,
            )

            # Step 5(d) - Conditional Gaussian Draw
            std_sk = np.sqrt(max(0.0, var_sk))
            sim_val = mu_sk + std_sk * float(rng.standard_normal())
            y_sim_current[node_idx] = sim_val
            is_simulated[node_idx] = True

        Y_sim[:, r] = y_sim_current

    # 6. Normal Score Back-Transformation
    Z_sim = normal_score_back_transform(Y_sim, sorted_z, sorted_y)

    # 7. Extrapolation Masking
    if mask_extrapolation:
        inside = is_within_convex_hull(samples, grid)
        Z_sim[~inside, :] = np.nan

    return Z_sim


def sequential_gaussian_block_simulation(
    samples_xyz: np.ndarray,
    sample_grades: np.ndarray,
    block_model: pd.DataFrame,
    sill: float,
    range_param: Union[float, Sequence[float], Mapping[str, float]],
    n_realizations: int = 50,
    variogram_model: str = "spherical",
    nugget: float = 0.0,
    discretization: Tuple[int, int, int] = (4, 4, 2),
    sample_weights: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
    search_radius: Optional[float] = None,
    max_hard_samples: int = 12,
    max_simulated_nodes: int = 12,
    max_per_sector: Optional[int] = None,
    max_per_drillhole: Optional[int] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
    assign_data_to_nodes: bool = False,
    n_multigrids: int = 1,
) -> np.ndarray:
    """Generates equiprobable conditional realizations on 3D block support via Sequential Gaussian Simulation.

    Simulates at quasi-point support on discretization sub-points within each block,
    and linearly averages the simulated point grades within each block to preserve
    mass balance and correct block dispersion variance (Goovaerts 1997; Ortiz 2019).

    Parameters
    ----------
    samples_xyz : np.ndarray
        Array of shape (N, 3) with drillhole sample coordinates.
    sample_grades : np.ndarray
        Array of shape (N,) with continuous raw assay grades.
    block_model : pd.DataFrame
        DataFrame with mining block centroids ('x', 'y', 'z') and block dimensions ('dx', 'dy', 'dz').
    sill : float
        Variogram partial sill variance c.
    range_param : float, Sequence[float], or Mapping
        Variogram practical range a.
    n_realizations : int, default 50
        Number of stochastic realizations L to simulate.
    variogram_model : str, default "spherical"
        Analytical variogram model type ('spherical', 'exponential', 'gaussian', 'cubic').
    nugget : float, default 0.0
        Variogram nugget effect c0.
    discretization : Tuple[int, int, int], default (4, 4, 2)
        Discretization resolution (nx, ny, nz) along each axis per block.
    sample_weights : np.ndarray, optional
        Declustering weights of shape (N,) for normal score transformation.
    seed : int, optional
        Random seed for reproducible Monte Carlo draws.
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Ranges along principal axes for 3D geometric anisotropy.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles (azimuth, dip, plunge) for 3D geometric anisotropy.
    search_radius : float, optional
        Maximum spatial search radius for conditioning data.
    max_hard_samples : int, default 12
        Maximum number of original drillhole samples to condition each node.
    max_simulated_nodes : int, default 12
        Maximum number of previously simulated grid nodes to condition each node.
    max_per_sector : int, optional
        Maximum samples per angular sector.
    max_per_drillhole : int, optional
        Maximum samples per drillhole trace.
    sample_hole_ids : Sequence[Any], optional
        Drillhole trace identifiers for conditioning samples.
    assign_data_to_nodes : bool, default False
        If True, pre-assigns drillhole samples to their closest grid nodes, treating them as
        pre-conditioned nodes and removing them from the simulation path. Replaces the dual
        search with a single grid search, accelerating execution on dense grids (GSLIB sgsim).
    n_multigrids : int, default 1
        Number of dyadic multiple grid levels (Gómez-Hernández & Srivastava 1990; Tran 1994;
        GSLIB nmult). If > 1, simulates coarse sub-grids first to lock in long-range spatial
        correlation structures before progressively refining to finer grids.

    Returns
    -------
    block_realizations : np.ndarray
        Array of shape (n_blocks, n_realizations) containing simulated block grades.
    """
    req_cols = {"x", "y", "z", "dx", "dy", "dz"}
    if not req_cols.issubset(block_model.columns):
        raise ValueError(
            f"block_model must contain columns {req_cols}, got {list(block_model.columns)}."
        )

    n_blocks = len(block_model)
    if n_blocks == 0:
        return np.zeros((0, n_realizations), dtype=float)

    nx, ny, nz = discretization
    dx = float(block_model["dx"].iloc[0])
    dy = float(block_model["dy"].iloc[0])
    dz = float(block_model["dz"].iloc[0])

    disc_offsets = _generate_block_discretization_offsets(
        discretization=discretization, block_size=(dx, dy, dz)
    )
    block_coords = block_model[["x", "y", "z"]].to_numpy(dtype=float)

    # Discretization sub-points for all blocks of shape (n_blocks * M, 3)
    all_disc_points = (block_coords[:, None, :] + disc_offsets[None, :, :]).reshape(
        -1, 3
    )

    # Simulate quasi-point values at all discretization locations
    point_sim = sequential_gaussian_grid_simulation(
        samples_xy=samples_xyz,
        sample_grades=sample_grades,
        grid_points=all_disc_points,
        sill=sill,
        range_param=range_param,
        n_realizations=n_realizations,
        variogram_model=variogram_model,
        nugget=nugget,
        sample_weights=sample_weights,
        seed=seed,
        anisotropy_ranges=anisotropy_ranges,
        anisotropy_angles=anisotropy_angles,
        search_radius=search_radius,
        max_hard_samples=max_hard_samples,
        max_simulated_nodes=max_simulated_nodes,
        max_per_sector=max_per_sector,
        max_per_drillhole=max_per_drillhole,
        sample_hole_ids=sample_hole_ids,
        assign_data_to_nodes=assign_data_to_nodes,
        n_multigrids=n_multigrids,
        mask_extrapolation=False,
    )

    # Linearly upscale simulated point grades to block support
    return _upscale_realizations_to_blocks(
        point_realizations=point_sim,
        n_blocks=n_blocks,
        points_per_block=len(disc_offsets),
    )


# TODO: is the notion of hardcoding a cutoff grade to rigid? What if we wanted other transform functions? Should we create or use existing functions for the percentiles and stuff? I feel like those could be generally reusable? Are there other places where we kind of hardcode logic like this into it, when maybe we shouldn't? ss
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
    figsize: Tuple[float, float] = (14.0, 11.0),
    sample_coords: Optional[np.ndarray] = None,
    sample_grades: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    uncertainty_cmap: str = "magma",
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Generates a 4-panel comparison dashboard of simulation realizations vs. E-type map.

    Visualizes:
    1. Realization 1 (showing full spatial variance, texture, and extreme values).
    2. Realization 2 (an alternative equiprobable stochastic outcome).
    3. E-Type Map (the smoothed conditional expectation).
    4. Conditional Uncertainty: Probability of Exceedance if cutoff_grade is given,
       or conditional standard deviation.

    Parameters
    ----------
    realizations : np.ndarray
        Array of shape (M, L) with simulated realizations.
    grid_xy : np.ndarray
        Array of shape (M, 2) with spatial coordinates.
    cutoff_grade : float, optional
        Cut-off grade for probability of exceedance map. If None, plots conditional std dev.
    grade_unit : str, default "% Cu"
        Grade unit label.
    title : str, optional
        Dashboard super-title.
    figsize : tuple of float, default (14.0, 11.0)
        Matplotlib figure dimensions.
    sample_coords : np.ndarray, optional
        Sample spatial coordinates of shape (N, 2) to overlay.
    sample_grades : np.ndarray, optional
        Sample grade values of shape (N,) to color sample points.
    cmap : str, default "viridis"
        Colormap for grade panels (1, 2, and 3).
    uncertainty_cmap : str, default "magma"
        Colormap for uncertainty panel (4).

    Returns
    -------
    Tuple[plt.Figure, Sequence[plt.Axes]]
        Figure and axes sequence of 4 subplots.
    """
    real_arr = np.asarray(realizations, dtype=float)
    grid_arr = np.asarray(grid_xy, dtype=float)

    if real_arr.ndim != 2:
        raise ValueError(
            f"realizations must be a 2D array of shape (M, L), got shape {real_arr.shape}"
        )
    if grid_arr.ndim != 2 or grid_arr.shape[1] != 2:
        raise ValueError(
            f"grid_xy must be a 2D array of shape (M, 2), got shape {grid_arr.shape}"
        )
    if real_arr.shape[0] != grid_arr.shape[0]:
        raise ValueError(
            f"Point count mismatch: realizations has {real_arr.shape[0]} rows, "
            f"grid_xy has {grid_arr.shape[0]} rows"
        )

    n_points, n_real = real_arr.shape
    r1 = real_arr[:, 0]
    r2 = real_arr[:, 1] if n_real > 1 else real_arr[:, 0]
    e_type = np.mean(real_arr, axis=1)

    # Identical scale norm across grade panels (CONTRIBUTING.md Part 1 §5)
    grade_min = min(
        float(np.nanmin(r1)), float(np.nanmin(r2)), float(np.nanmin(e_type))
    )
    grade_max = max(
        float(np.nanmax(r1)), float(np.nanmax(r2)), float(np.nanmax(e_type))
    )
    if sample_grades is not None:
        sg_arr = np.asarray(sample_grades, dtype=float)
        grade_min = min(grade_min, float(np.nanmin(sg_arr)))
        grade_max = max(grade_max, float(np.nanmax(sg_arr)))

    norm_grade = mcolors.Normalize(vmin=grade_min, vmax=grade_max)

    # Uncertainty data
    if cutoff_grade is not None:
        uncertainty_vals = np.mean(real_arr >= cutoff_grade, axis=1)
        norm_unc = mcolors.Normalize(vmin=0.0, vmax=1.0)
        unc_label = "Probability [0, 1]"
        unc_title = (
            f"Probability of Exceedance: P(Grade >= {cutoff_grade:.2f} {grade_unit})"
        )
    else:
        uncertainty_vals = (
            np.std(real_arr, axis=1, ddof=1) if n_real > 1 else np.zeros(n_points)
        )
        norm_unc = mcolors.Normalize(
            vmin=float(np.nanmin(uncertainty_vals)),
            vmax=float(np.nanmax(uncertainty_vals)),
        )
        unc_label = f"Std Dev ({grade_unit})"
        unc_title = f"Conditional Uncertainty: Std Dev ({grade_unit})"

    # Check for regular rectangular grid
    ux = np.unique(grid_arr[:, 0])
    uy = np.unique(grid_arr[:, 1])
    is_regular_grid = len(ux) * len(uy) == n_points

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    def _render_layer(
        ax: plt.Axes,
        values: np.ndarray,
        norm: mcolors.Normalize,
        cmap_name: str,
        cbar_label: str,
    ):
        if is_regular_grid:
            idx_x = np.searchsorted(ux, grid_arr[:, 0])
            idx_y = np.searchsorted(uy, grid_arr[:, 1])
            grid_2d = np.full((len(uy), len(ux)), np.nan)
            grid_2d[idx_y, idx_x] = values
            im = ax.pcolormesh(
                ux, uy, grid_2d, shading="auto", cmap=cmap_name, norm=norm
            )
        else:
            im = ax.scatter(
                grid_arr[:, 0],
                grid_arr[:, 1],
                c=values,
                cmap=cmap_name,
                norm=norm,
                s=25,
            )
        fig.colorbar(im, ax=ax, label=cbar_label)
        return im

    # Panel 1: Realization 1
    _render_layer(ax_list[0], r1, norm_grade, cmap, grade_unit)
    ax_list[0].set_title("Realization 1 (Stochastic Outcome #1)")

    # Panel 2: Realization 2
    r2_label = (
        "Realization 2 (Stochastic Outcome #2)"
        if n_real > 1
        else "Realization 1 (Single Realization)"
    )
    _render_layer(ax_list[1], r2, norm_grade, cmap, grade_unit)
    ax_list[1].set_title(r2_label)

    # Panel 3: E-Type Map
    _render_layer(ax_list[2], e_type, norm_grade, cmap, grade_unit)
    real_vars = np.var(real_arr, axis=0, ddof=1)
    mean_var = float(np.mean(real_vars)) if len(real_vars) > 0 else 1.0
    smooth_ratio = float(np.var(e_type, ddof=1)) / mean_var if mean_var > 0 else 1.0
    ax_list[2].set_title(f"E-Type Map (Conditional Mean, Smoothing={smooth_ratio:.2f})")

    # Panel 4: Uncertainty / Exceedance
    _render_layer(ax_list[3], uncertainty_vals, norm_unc, uncertainty_cmap, unc_label)
    ax_list[3].set_title(unc_title)

    # Overlay sample drillholes if supplied
    if sample_coords is not None:
        sc = np.asarray(sample_coords, dtype=float)
        for i in range(3):
            if sample_grades is not None:
                ax_list[i].scatter(
                    sc[:, 0],
                    sc[:, 1],
                    c=np.asarray(sample_grades, dtype=float),
                    norm=norm_grade,
                    cmap=cmap,
                    edgecolors="black",
                    linewidths=0.5,
                    s=30,
                    label="Samples",
                )
            else:
                ax_list[i].scatter(
                    sc[:, 0],
                    sc[:, 1],
                    c="red",
                    edgecolors="black",
                    linewidths=0.5,
                    s=25,
                    label="Samples",
                )
            ax_list[i].legend(loc="upper right")

        ax_list[3].scatter(
            sc[:, 0],
            sc[:, 1],
            c="cyan",
            edgecolors="black",
            linewidths=0.5,
            s=25,
            label="Samples",
        )
        ax_list[3].legend(loc="upper right")

    for ax in ax_list:
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")

    if title is not None:
        fig.suptitle(title, fontsize=14, y=0.995)

    fig.tight_layout()
    return fig, ax_list


# =============================================================================
# 3D BLOCK MODEL VISUALIZATION SUITE (SLICES, GALLERIES, ISOMETRIC, UNCERTAINTY)
# =============================================================================
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


# =============================================================================
# SHARED CATEGORICAL BLOCK MODEL VISUALIZATION ENGINE
# =============================================================================


def _plot_categorical_blocks_orthogonal_slices(
    block_model: pd.DataFrame,
    category_col: str,
    categories: Sequence[str],
    colors: Sequence[str],
    bench_z: Optional[float] = None,
    section_y: Optional[float] = None,
    section_x: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    cross_section_view: str = "north",
    long_section_view: str = "west",
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    dx_col: str = "dx",
    dy_col: str = "dy",
    dz_col: str = "dz",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (19.0, 5.8),
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Shared 3-panel synchronized orthogonal slice renderer for categorical block models."""
    if block_model.empty:
        raise ValueError("block_model cannot be empty.")

    cross_section_view, long_section_view = _validate_orthogonal_views(
        cross_section_view, long_section_view
    )

    for req in [x_col, y_col, z_col]:
        if req not in block_model.columns:
            raise ValueError(
                f"block_model is missing required coordinate column '{req}'."
            )
    if category_col not in block_model.columns:
        raise ValueError(f"block_model is missing category column '{category_col}'.")

    bm = block_model.copy()

    # Estimate block dimensions if missing
    for c, dim_c in [(x_col, dx_col), (y_col, dy_col), (z_col, dz_col)]:
        if dim_c not in bm.columns:
            unq = np.sort(bm[c].unique())
            if len(unq) > 1:
                diffs = np.diff(unq)
                diffs = diffs[diffs > 1e-4]
                val = float(np.median(diffs)) if len(diffs) > 0 else 10.0
            else:
                val = 10.0
            bm[dim_c] = val

    n_cats = len(categories)
    cat_to_code = {cat.lower(): i for i, cat in enumerate(categories)}
    default_code = n_cats - 1
    cat_series = bm[category_col].astype(str).str.strip().str.lower()
    bm["_cat_code"] = cat_series.map(
        lambda s: cat_to_code.get(s, default_code)
    ).to_numpy(dtype=int)

    cmap_cat = mcolors.ListedColormap(colors)
    bounds = np.arange(n_cats + 1) - 0.5
    norm_cat = mcolors.BoundaryNorm(bounds, ncolors=n_cats)

    target_z, target_y, target_x = _find_nearest_slice_coords(
        bm,
        bench_z=bench_z,
        section_y=section_y,
        section_x=section_x,
        x_col=x_col,
        y_col=y_col,
        z_col=z_col,
    )

    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # 1. Bench Plan (X-Y at target_z)
    slice_bench = bm[np.isclose(bm[z_col], target_z)]
    dz_val = float(slice_bench[dz_col].iloc[0]) if len(slice_bench) > 0 else 5.0
    _render_2d_block_patches(
        axes[0],
        slice_bench,
        x_col,
        y_col,
        dx_col,
        dy_col,
        "_cat_code",
        norm_cat,
        cmap_cat,
    )
    axes[0].set_title(
        f"Bench Plan (Z = {target_z:.1f} m) [Looking Down]",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_xlabel("Easting (X) [m]")
    axes[0].set_ylabel("Northing (Y) [m]")
    axes[0].set_aspect("equal")
    axes[0].axhline(target_y, color="red", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[0].axvline(target_x, color="cyan", linestyle="--", linewidth=1.2, alpha=0.8)

    # 2. Cross-Section (X-Z at target_y)
    slice_sec = bm[np.isclose(bm[y_col], target_y)]
    dy_val = float(slice_sec[dy_col].iloc[0]) if len(slice_sec) > 0 else 10.0
    _render_2d_block_patches(
        axes[1],
        slice_sec,
        x_col,
        z_col,
        dx_col,
        dz_col,
        "_cat_code",
        norm_cat,
        cmap_cat,
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
    axes[1].axhline(target_z, color="magenta", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[1].axvline(target_x, color="cyan", linestyle="--", linewidth=1.2, alpha=0.8)

    # 3. Longitudinal Section (Y-Z at target_x)
    slice_long = bm[np.isclose(bm[x_col], target_x)]
    dx_val = float(slice_long[dx_col].iloc[0]) if len(slice_long) > 0 else 10.0
    _render_2d_block_patches(
        axes[2],
        slice_long,
        y_col,
        z_col,
        dy_col,
        dz_col,
        "_cat_code",
        norm_cat,
        cmap_cat,
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
    axes[2].axhline(target_z, color="magenta", linestyle="--", linewidth=1.2, alpha=0.8)
    axes[2].axvline(target_y, color="red", linestyle="--", linewidth=1.2, alpha=0.8)

    # Drillhole composite overlay within corridors
    if samples_xyz is not None and len(samples_xyz) > 0:
        samples_arr = np.asarray(samples_xyz, dtype=float)
        mask_bench = np.abs(samples_arr[:, 2] - target_z) <= 0.5 * dz_val
        if np.any(mask_bench):
            axes[0].scatter(
                samples_arr[mask_bench, 0],
                samples_arr[mask_bench, 1],
                c="black",
                marker="^",
                s=35,
                zorder=5,
            )
        mask_sec = np.abs(samples_arr[:, 1] - target_y) <= 0.5 * dy_val
        if np.any(mask_sec):
            axes[1].scatter(
                samples_arr[mask_sec, 0],
                samples_arr[mask_sec, 2],
                c="black",
                marker="^",
                s=35,
                zorder=5,
            )
        mask_long = np.abs(samples_arr[:, 0] - target_x) <= 0.5 * dx_val
        if np.any(mask_long):
            axes[2].scatter(
                samples_arr[mask_long, 1],
                samples_arr[mask_long, 2],
                c="black",
                marker="^",
                s=35,
                zorder=5,
            )

    legend_handles = [
        mpatches.Patch(facecolor=c, edgecolor="none", label=cat)
        for cat, c in zip(categories, colors)
    ]
    if samples_xyz is not None and len(samples_xyz) > 0:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="^",
                linestyle="None",
                markersize=6,
                label="Drillhole",
            )
        )
    axes[0].legend(
        handles=legend_handles, loc="upper right", framealpha=0.85, fontsize=8
    )

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)

    fig_title = title or "3D Block Model Orthogonal Slices"
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")
    return fig, axes


def _plot_categorical_bench_gallery(
    block_model: pd.DataFrame,
    category_col: str,
    categories: Sequence[str],
    colors: Sequence[str],
    bench_elevations: Optional[Sequence[float]] = None,
    n_cols: int = 3,
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    dx_col: str = "dx",
    dy_col: str = "dy",
    dz_col: str = "dz",
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    """Shared multi-bench elevation gallery for categorical block models."""
    if block_model.empty:
        raise ValueError("block_model cannot be empty.")
    for req in [x_col, y_col, z_col, category_col]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")

    bm = block_model.copy()
    for c, dim_c in [(x_col, dx_col), (y_col, dy_col), (z_col, dz_col)]:
        if dim_c not in bm.columns:
            unq = np.sort(bm[c].unique())
            if len(unq) > 1:
                diffs = np.diff(unq)
                diffs = diffs[diffs > 1e-4]
                val = float(np.median(diffs)) if len(diffs) > 0 else 10.0
            else:
                val = 10.0
            bm[dim_c] = val

    n_cats = len(categories)
    cat_to_code = {cat.lower(): i for i, cat in enumerate(categories)}
    default_code = n_cats - 1
    cat_series = bm[category_col].astype(str).str.strip().str.lower()
    bm["_cat_code"] = cat_series.map(
        lambda s: cat_to_code.get(s, default_code)
    ).to_numpy(dtype=int)

    cmap_cat = mcolors.ListedColormap(colors)
    bounds = np.arange(n_cats + 1) - 0.5
    norm_cat = mcolors.BoundaryNorm(bounds, ncolors=n_cats)

    unique_z = np.sort(bm[z_col].unique())
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

    for i, bz in enumerate(chosen_z):
        ax = axes[i]
        df_bench = bm[np.isclose(bm[z_col], bz)]
        dz_val = float(df_bench[dz_col].iloc[0]) if len(df_bench) > 0 else 5.0
        _render_2d_block_patches(
            ax, df_bench, x_col, y_col, dx_col, dy_col, "_cat_code", norm_cat, cmap_cat
        )
        if samples_xyz is not None and len(samples_xyz) > 0:
            samples_arr = np.asarray(samples_xyz, dtype=float)
            mask_dh = np.abs(samples_arr[:, 2] - bz) <= 0.5 * dz_val
            if np.any(mask_dh):
                ax.scatter(
                    samples_arr[mask_dh, 0],
                    samples_arr[mask_dh, 1],
                    c="black",
                    marker="^",
                    s=30,
                    zorder=5,
                )
        ax.set_title(f"Bench Elevation Z = {bz:.1f} m", fontsize=11, fontweight="bold")
        ax.set_xlabel("Easting (X) [m]")
        ax.set_ylabel("Northing (Y) [m]")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    legend_handles = [
        mpatches.Patch(facecolor=c, edgecolor="none", label=cat)
        for cat, c in zip(categories, colors)
    ]
    if samples_xyz is not None and len(samples_xyz) > 0:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="^",
                linestyle="None",
                markersize=6,
                label="Drillhole",
            )
        )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(legend_handles),
        fontsize=9,
        framealpha=0.9,
    )

    fig_title = title or "Multi-Bench Depth Gallery"
    fig.suptitle(fig_title, fontsize=13, fontweight="bold", y=1.06)
    return fig, axes[:n_plots]


def _plot_categorical_blocks_3d_isometric(
    block_model: pd.DataFrame,
    category_col: str,
    categories: Sequence[str],
    colors: Sequence[str],
    active_categories: Optional[Sequence[str]] = None,
    bench_z: Optional[float] = None,
    elevation_tolerance: Optional[float] = None,
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    elev: float = 28.0,
    azim: float = -55.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10.0, 8.0),
) -> Tuple[plt.Figure, Axes3D]:
    """Shared static 3D isometric projection for categorical block models."""
    if block_model.empty:
        raise ValueError("block_model cannot be empty.")
    for req in [x_col, y_col, z_col, category_col]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")

    bm = block_model.copy()
    cat_series = bm[category_col].astype(str).str.strip().str.lower()
    n_cats = len(categories)
    cat_to_code = {cat.lower(): i for i, cat in enumerate(categories)}
    default_code = n_cats - 1
    bm["_cat_code"] = cat_series.map(
        lambda s: cat_to_code.get(s, default_code)
    ).to_numpy(dtype=int)

    mask = np.ones(len(bm), dtype=bool)
    if active_categories is not None:
        act_lower = {c.lower() for c in active_categories}
        mask &= cat_series.isin(act_lower)

    if bench_z is not None:
        tol = elevation_tolerance if elevation_tolerance is not None else 5.0
        mask &= (bm[z_col] >= bench_z - tol) & (bm[z_col] <= bench_z + tol)

    bm_vis = bm[mask]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    cmap_cat = mcolors.ListedColormap(colors)
    bounds = np.arange(n_cats + 1) - 0.5
    norm_cat = mcolors.BoundaryNorm(bounds, ncolors=n_cats)

    if len(bm_vis) > 0:
        ax.scatter(
            bm_vis[x_col].to_numpy(),
            bm_vis[y_col].to_numpy(),
            bm_vis[z_col].to_numpy(),
            c=bm_vis["_cat_code"].to_numpy(),
            cmap=cmap_cat,
            norm=norm_cat,
            marker="s",
            s=45,
            alpha=0.75,
            edgecolors="none",
            label="Blocks",
        )

    if samples_xyz is not None and len(samples_xyz) > 0:
        samples_arr = np.asarray(samples_xyz, dtype=float)
        ax.scatter(
            samples_arr[:, 0],
            samples_arr[:, 1],
            samples_arr[:, 2],
            c="black",
            s=20,
            edgecolor="white",
            linewidth=0.5,
            label="Drillholes",
            zorder=5,
        )

    ax.set_xlabel("Easting (X) [m]", labelpad=8)
    ax.set_ylabel("Northing (Y) [m]", labelpad=8)
    ax.set_zlabel("Elevation (Z) [m]", labelpad=8)
    ax.view_init(elev=elev, azim=azim)

    span_x = max(1.0, float(np.ptp(bm[x_col])))
    span_y = max(1.0, float(np.ptp(bm[y_col])))
    span_z = max(1.0, float(np.ptp(bm[z_col])))
    ax.set_box_aspect((span_x, span_y, span_z))

    legend_handles = []
    tot_blocks = max(1, len(bm_vis))
    for i, (cat, c) in enumerate(zip(categories, colors)):
        count = int(np.sum(bm_vis["_cat_code"] == i))
        pct = count / tot_blocks * 100.0
        legend_handles.append(
            mpatches.Patch(
                facecolor=c, edgecolor="none", label=f"{cat} ({count:,} - {pct:.1f}%)"
            )
        )
    if samples_xyz is not None and len(samples_xyz) > 0:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                marker="o",
                linestyle="None",
                markersize=5,
                label="Drillholes",
            )
        )
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.85, fontsize=8)

    fig_title = title or "3D Categorical Block Model Isometric View"
    ax.set_title(fig_title, fontsize=12, fontweight="bold", pad=12)
    return fig, ax


def _plot_categorical_blocks_3d_interactive(
    block_model: pd.DataFrame,
    category_col: str,
    categories: Sequence[str],
    colors: Sequence[str],
    tonnes_col: Optional[str] = "tonnes",
    samples_xyz: Optional[np.ndarray] = None,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    elev: float = 28.0,
    azim: float = -55.0,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (11.0, 9.0),
) -> Tuple[plt.Figure, Axes3D, Dict[str, Any]]:
    """Shared interactive 3D explorer with category toggles and elevation sliders."""
    if block_model.empty:
        raise ValueError("block_model cannot be empty.")
    for req in [x_col, y_col, z_col, category_col]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")

    bm = block_model.copy()
    cat_series = bm[category_col].astype(str).str.strip().str.lower()
    n_cats = len(categories)
    cat_to_code = {cat.lower(): i for i, cat in enumerate(categories)}
    default_code = n_cats - 1
    bm["_cat_code"] = cat_series.map(
        lambda s: cat_to_code.get(s, default_code)
    ).to_numpy(dtype=int)

    x_all = bm[x_col].to_numpy()
    y_all = bm[y_col].to_numpy()
    z_all = bm[z_col].to_numpy()
    codes_all = bm["_cat_code"].to_numpy()
    tonnes_all = (
        bm[tonnes_col].to_numpy() if (tonnes_col and tonnes_col in bm.columns) else None
    )

    min_z = float(np.min(z_all))
    max_z = float(np.max(z_all))

    cmap_cat = mcolors.ListedColormap(colors)
    bounds = np.arange(n_cats + 1) - 0.5
    norm_cat = mcolors.BoundaryNorm(bounds, ncolors=n_cats)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.22, top=0.93, left=0.05, right=0.88)

    p = ax.scatter(
        x_all,
        y_all,
        z_all,
        c=codes_all,
        cmap=cmap_cat,
        norm=norm_cat,
        marker="s",
        s=45,
        alpha=0.75,
        edgecolors="none",
        label="Blocks",
    )

    if samples_xyz is not None and len(samples_xyz) > 0:
        samples_arr = np.asarray(samples_xyz, dtype=float)
        ax.scatter(
            samples_arr[:, 0],
            samples_arr[:, 1],
            samples_arr[:, 2],
            c="black",
            s=20,
            edgecolor="white",
            linewidth=0.5,
            label="Drillholes",
            zorder=5,
        )

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

    legend_handles = [
        mpatches.Patch(facecolor=c, edgecolor="none", label=cat)
        for cat, c in zip(categories, colors)
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.85, fontsize=8)

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

    # Widgets
    ax_check = fig.add_axes([0.08, 0.03, 0.28, 0.12])
    check_cats = CheckButtons(
        ax=ax_check,
        labels=list(categories),
        actives=[True] * n_cats,
    )
    for txt in check_cats.labels:
        txt.set_fontsize(8)

    ax_slider_z = fig.add_axes([0.42, 0.09, 0.38, 0.035])
    slider_elev = Slider(
        ax=ax_slider_z,
        label="Max Elevation",
        valmin=min_z,
        valmax=max_z,
        valinit=max_z,
        valfmt="%.1f m",
        color="#4a7c59",
    )

    ax_btn = fig.add_axes([0.84, 0.03, 0.11, 0.095])
    btn_reset = Button(ax_btn, "Reset\nControls", hovercolor="#e0e0e0")

    def update_plot(val=None):
        active_flags = check_cats.get_status()
        z_val = slider_elev.val
        active_codes = {i for i, act in enumerate(active_flags) if act}

        mask = np.isin(codes_all, list(active_codes)) & (z_all <= z_val)
        n_vis = int(np.sum(mask))
        n_tot = len(x_all)
        pct = (n_vis / n_tot * 100.0) if n_tot > 0 else 0.0

        if n_vis > 0:
            p._offsets3d = (x_all[mask], y_all[mask], z_all[mask])
            p.set_array(codes_all[mask])
        else:
            p._offsets3d = (np.array([]), np.array([]), np.array([]))
            p.set_array(np.array([]))

        hud_lines = [f"Visible: {n_vis:,} / {n_tot:,} ({pct:.1f}%)"]
        if tonnes_all is not None:
            t_vis = float(np.sum(tonnes_all[mask])) / 1e6
            hud_lines.append(f"Tonnage: {t_vis:.2f} Mt")
        for i, cat in enumerate(categories):
            if i in active_codes:
                c_cnt = int(np.sum(mask & (codes_all == i)))
                c_pct = (c_cnt / n_vis * 100.0) if n_vis > 0 else 0.0
                hud_lines.append(f"• {cat}: {c_cnt:,} ({c_pct:.1f}%)")

        info_box.set_text("\n".join(hud_lines))
        fig.canvas.draw_idle()

    check_cats.on_clicked(update_plot)
    slider_elev.on_changed(update_plot)

    def reset_controls(event):
        slider_elev.reset()
        for i, act in enumerate(check_cats.get_status()):
            if not act:
                check_cats.set_active(i)
        ax.view_init(elev=elev, azim=azim)
        update_plot()

    btn_reset.on_clicked(reset_controls)
    update_plot()

    controls = {
        "check_cats": check_cats,
        "slider_elev": slider_elev,
        "button_reset": btn_reset,
        "btn_reset": btn_reset,
        "reset_func": reset_controls,
        "update_func": update_plot,
    }
    return fig, ax, controls


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
    cross_section_view, long_section_view = _validate_orthogonal_views(
        cross_section_view, long_section_view
    )

    for req in ["x", "y", "z", "dx", "dy", "dz"]:
        if req not in block_model.columns:
            raise ValueError(f"block_model is missing required column '{req}'.")
    if grade_col not in block_model.columns:
        raise ValueError(f"block_model is missing grade column '{grade_col}'.")

    # Pick exact centroid coordinate slices
    target_z, target_y, target_x = _find_nearest_slice_coords(
        block_model,
        bench_z=bench_z,
        section_y=section_y,
        section_x=section_x,
        x_col="x",
        y_col="y",
        z_col="z",
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


# =============================================================================
# PART 1: KRIGING MODEL VALIDATION & CROSS-VALIDATION (CIM MRMR §6.10, SME CH. 4.5)
# =============================================================================


def kriging_cross_validation(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    variogram_model: str,
    sill: float,
    range_param: Union[float, Sequence[float], Mapping[str, float]],
    nugget: float = 0.0,
    method: Literal["ordinary", "simple"] = "ordinary",
    simple_kriging_mean: float = 0.0,
    anisotropy_ranges: Optional[Union[Sequence[float], Mapping[Any, Any]]] = None,
    anisotropy_angles: Optional[
        Union[float, Sequence[float], Mapping[Any, Any]]
    ] = None,
    search_radius: Optional[float] = None,
    min_neighbors: int = 1,
    max_neighbors: int = 16,
    max_per_sector: Optional[int] = None,
    max_per_drillhole: Optional[int] = None,
    sample_hole_ids: Optional[Sequence[Any]] = None,
) -> pd.DataFrame:
    """Performs Leave-One-Out (jackknife) cross-validation for Kriging models.

    In accordance with CIM MRMR Best Practice Guidelines (§6.10) and SME Mining Engineering
    Handbook (§4.5; Rossi & Deutsch 2014, Ch. 8) (TODO: Manually Verify), point cross-validation
    is the mandatory third pillar of model validation (alongside global bias checks and swath plots).
    It tests each conditioning sample against its neighbors to validate the spatial continuity model
    (variogram sill, nugget, range, anisotropy) and neighborhood search parameters.

    Mathematical Formulation:
    -------------------------
    For each sample u_alpha in 1, ..., N:
      1. Remove sample u_alpha from the conditioning dataset.
      2. Estimate grade Z*(u_alpha) and kriging variance sigma^2_K(u_alpha) using surrounding samples.
      3. Compute estimation error: e_alpha = Z(u_alpha) - Z*(u_alpha).
      4. Compute standardized residual: z_alpha = (Z(u_alpha) - Z*(u_alpha)) / sigma_K(u_alpha).

    Regulatory Acceptance Criteria (CIM MRMR §6.10) (TODO: Manually Verify):
    ------------------------------------------------------------------------
    - Mean Error (ME): Close to 0.0 (indicates absence of global systematic bias).
    - Mean Standardized Error (MSE): Close to 0.0.
    - Variance of Standardized Residuals: Approximately 1.0 (confirms sill and nugget are properly scaled).
    - Slope of Regression (Actual vs. Estimated): Close to 1.0 (confirms conditional unbiasedness).

    Parameters
    ----------
    samples_xy : np.ndarray
        Array of shape (N, 2) or (N, 3) containing sample spatial coordinates.
    sample_grades : np.ndarray
        Array of shape (N,) containing raw or transformed continuous assay grades.
    variogram_model : str
        Analytical variogram model ('spherical', 'exponential', 'gaussian', 'cubic').
    sill : float
        Variogram partial sill variance C.
    range_param : float, Sequence[float], or Mapping
        Variogram practical range a.
    nugget : float, default 0.0
        Variogram nugget effect C0.
    method : {"ordinary", "simple"}, default "ordinary"
        Kriging estimation method to validate.
    simple_kriging_mean : float, default 0.0
        Stationary prior mean m if method="simple".
    anisotropy_ranges : Sequence[float] or Mapping, optional
        Ranges along principal axes for geometric anisotropy.
    anisotropy_angles : float, Sequence[float], or Mapping, optional
        Rotation angles for geometric anisotropy.
    search_radius : float, optional
        Maximum search distance. If None, defaults to 1.5 * effective range.
    min_neighbors : int, default 1
        Minimum samples required to compute an estimate.
    max_neighbors : int, default 16
        Maximum conditioning samples used for each local system.
    max_per_sector : int, optional
        Maximum samples allowed per angular sector.
    max_per_drillhole : int, optional
        Maximum samples allowed per drillhole trace.
    sample_hole_ids : Sequence[Any], optional
        Drillhole identifiers corresponding to each sample.

    Returns
    -------
    pd.DataFrame
        Cross-validation audit table containing:
        - 'x', 'y' (and 'z'): Coordinates of the cross-validated sample.
        - 'actual': True measured grade Z(u_alpha).
        - 'estimated': Kriging estimate Z*(u_alpha).
        - 'error': Estimation error e = actual - estimated.
        - 'kriging_variance': Kriging estimation variance sigma^2_K.
        - 'kriging_std': Estimation standard deviation sigma_K.
        - 'standardized_error': Standardized residual z = error / sigma_K.
        - 'num_neighbors': Number of conditioning samples used.
        Attached in `.attrs`:
        - 'mean_error': Global bias metric.
        - 'mean_absolute_error': MAE.
        - 'root_mean_square_error': RMSE.
        - 'mean_standardized_error': MSE.
        - 'variance_standardized_error': Variance of standardized error (target ~ 1.0).
        - 'pearson_correlation': Correlation r between actual and estimated.
        - 'slope_of_regression': Linear regression slope (actual on estimated).
    """
    coords = np.asarray(samples_xy, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(
            f"samples_xy must be a 2D array of shape (N, 2) or (N, 3), got shape {coords.shape}"
        )
    grades = np.asarray(sample_grades, dtype=float)
    if grades.ndim != 1 or len(grades) != len(coords):
        raise ValueError(
            f"sample_grades must be a 1D array with length matching samples_xy ({len(coords)}), got {grades.shape}"
        )

    n_samples = len(coords)
    if n_samples < 2:
        raise ValueError(
            f"At least 2 samples are required for leave-one-out cross-validation, got {n_samples}"
        )

    if method not in ("ordinary", "simple"):
        raise ValueError(f"method must be 'ordinary' or 'simple', got '{method}'")

    if min_neighbors < 1:
        raise ValueError(f"min_neighbors must be >= 1, got {min_neighbors}")
    if max_neighbors < min_neighbors:
        raise ValueError(
            f"max_neighbors ({max_neighbors}) must be >= min_neighbors ({min_neighbors})"
        )
    if max_per_sector is not None and max_per_sector <= 0:
        raise ValueError(f"max_per_sector must be positive, got {max_per_sector}")
    if max_per_drillhole is not None and max_per_drillhole <= 0:
        raise ValueError(f"max_per_drillhole must be positive, got {max_per_drillhole}")
    if search_radius is not None and search_radius <= 0:
        raise ValueError(f"search_radius must be positive, got {search_radius}")

    if sample_hole_ids is not None and len(sample_hole_ids) != n_samples:
        raise ValueError(
            f"sample_hole_ids length ({len(sample_hole_ids)}) must match samples_xy ({n_samples})"
        )

    base_sill = sill if not isinstance(sill, Mapping) else list(sill.values())[0]
    base_nugget = (
        nugget if not isinstance(nugget, Mapping) else list(nugget.values())[0]
    )
    base_range = (
        range_param
        if not isinstance(range_param, Mapping)
        else list(range_param.values())[0]
    )

    if base_sill <= 0:
        raise ValueError(f"sill must be strictly positive, got {base_sill}")
    if base_nugget < 0:
        raise ValueError(f"nugget must be non-negative, got {base_nugget}")

    total_sill = float(base_nugget + base_sill)

    eff_ranges = (
        anisotropy_ranges
        if anisotropy_ranges is not None
        else (
            range_param if isinstance(range_param, (Sequence, Mapping)) else base_range
        )
    )
    samples_t, eff_range = transform_anisotropic_coordinates(
        coords, ranges=eff_ranges, angles=anisotropy_angles
    )

    effective_radius = (
        float(search_radius) if search_radius is not None else 1.5 * float(eff_range)
    )

    has_sector_search = (max_per_sector is not None) or (max_per_drillhole is not None)
    k_query = (
        min(n_samples, max(max_neighbors * 8, 128))
        if has_sector_search
        else min(n_samples, max_neighbors + 1)
    )

    tree = KDTree(samples_t)

    estimates = np.full(n_samples, np.nan, dtype=float)
    variances = np.full(n_samples, np.nan, dtype=float)
    num_neighbors_used = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        target_pt = samples_t[i]
        dists, idxs = tree.query(
            target_pt, k=k_query, distance_upper_bound=effective_radius
        )

        if k_query == 1:
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)

        # Exclude sample i itself (jackknife leave-one-out) and invalid upper-bound indices
        valid_mask = (
            (idxs < n_samples)
            & (idxs != i)
            & np.isfinite(dists)
            & (dists <= effective_radius)
        )
        cand_idxs = idxs[valid_mask]
        cand_dists = dists[valid_mask]

        if len(cand_idxs) == 0:
            continue

        if has_sector_search:
            idx_sel, d_sel = _filter_neighborhood_sectors(
                candidate_coords=samples_t[cand_idxs],
                candidate_indices=cand_idxs,
                candidate_distances=cand_dists,
                target_coord=target_pt,
                k_neighbors=max_neighbors,
                max_per_sector=max_per_sector,
                min_sectors=1,
                max_per_drillhole=max_per_drillhole,
                sample_hole_ids=sample_hole_ids,
            )
        else:
            take = min(max_neighbors, len(cand_idxs))
            idx_sel = cand_idxs[:take]
            d_sel = cand_dists[:take]

        if len(idx_sel) < min_neighbors:
            continue

        c_coords = samples_t[idx_sel]
        c_vals = grades[idx_sel]

        coords_m, vals_m, h_matrix, d_target_m = _deduplicate_collocated_samples(
            c_coords, c_vals, target_point=target_pt
        )
        if d_target_m is not None:
            d_sel = d_target_m
        k_m = len(coords_m)

        if k_m < min_neighbors:
            continue

        K_m = _theoretical_covariance(
            h_matrix, variogram_model, base_nugget, base_sill, eff_range
        )
        K_m[np.diag_indices(k_m)] += 1e-9

        k0_m = _theoretical_covariance(
            d_sel, variogram_model, base_nugget, base_sill, eff_range
        )

        # TODO: later date, is there a better way than this if method check? for when we add more methods? or for arbitrary methods?
        if method == "ordinary":
            weights_m, mu_lagrange = _solve_ordinary_kriging_system(
                K_m, k0_m, clip_negative_weights=False
            )
            est_val = float(np.sum(weights_m * vals_m))
            var_val = float(total_sill - np.sum(weights_m * k0_m) - mu_lagrange)
        else:
            try:
                weights_m = np.linalg.solve(K_m, k0_m)
            except np.linalg.LinAlgError:
                weights_m = np.zeros(k_m)
            est_val = float(
                simple_kriging_mean + np.sum(weights_m * (vals_m - simple_kriging_mean))
            )
            var_val = float(total_sill - np.sum(weights_m * k0_m))

        estimates[i] = est_val
        variances[i] = max(0.0, var_val)
        num_neighbors_used[i] = len(idx_sel)

    errors = grades - estimates
    kriging_stds = np.sqrt(np.maximum(0.0, variances))
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized_errors = np.where(
            kriging_stds > 1e-9, errors / kriging_stds, np.nan
        )

    valid = np.isfinite(estimates)
    n_valid = int(np.sum(valid))

    if n_valid > 0:
        me = float(np.mean(errors[valid]))
        mae = float(np.mean(np.abs(errors[valid])))
        rmse = float(np.sqrt(np.mean(errors[valid] ** 2)))
        valid_std = standardized_errors[valid & np.isfinite(standardized_errors)]
        mse = float(np.mean(valid_std)) if len(valid_std) > 0 else np.nan
        var_se = float(np.var(valid_std, ddof=1)) if len(valid_std) > 1 else np.nan

        std_act = float(np.std(grades[valid]))
        std_est = float(np.std(estimates[valid]))
        if n_valid > 1 and std_act > 1e-9 and std_est > 1e-9:
            pearson_r = float(np.corrcoef(grades[valid], estimates[valid])[0, 1])
        else:
            pearson_r = np.nan

        var_est = float(np.var(estimates[valid], ddof=1)) if n_valid > 1 else 0.0
        cov_act_est = (
            float(np.cov(grades[valid], estimates[valid])[0, 1]) if n_valid > 1 else 0.0
        )
        sor = float(cov_act_est / var_est) if var_est > 1e-12 else np.nan
    else:
        me = mae = rmse = mse = var_se = pearson_r = sor = np.nan

    out_data = {
        "x": coords[:, 0],
        "y": coords[:, 1],
    }
    if coords.shape[1] == 3:
        out_data["z"] = coords[:, 2]

    out_data.update(
        {
            "actual": grades,
            "estimated": estimates,
            "error": errors,
            "kriging_variance": variances,
            "kriging_std": kriging_stds,
            "standardized_error": standardized_errors,
            "num_neighbors": num_neighbors_used,
        }
    )

    df_out = pd.DataFrame(out_data)
    df_out.attrs["mean_error"] = me
    df_out.attrs["mean_absolute_error"] = mae
    df_out.attrs["root_mean_square_error"] = rmse
    df_out.attrs["mean_standardized_error"] = mse
    df_out.attrs["variance_standardized_error"] = var_se
    df_out.attrs["pearson_correlation"] = pearson_r
    df_out.attrs["slope_of_regression"] = sor
    df_out.attrs["method"] = method
    df_out.attrs["total_samples"] = n_samples
    df_out.attrs["estimated_samples"] = n_valid
    df_out.attrs["footnotes"] = [
        "Leave-One-Out (jackknife) cross-validation performed in accordance with "
        "CIM MRMR Best Practice Guidelines (§6.10) and SME Mining Engineering Handbook (§4.5) (TODO: Manually Verify).",
        "Target variance of standardized residuals is ~1.0; values > 1.0 indicate under-estimated sill/nugget, "
        "values < 1.0 indicate over-estimated sill/nugget.",
    ]

    return df_out


def plot_cross_validation_diagnostics(
    cv_results: pd.DataFrame,
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 11),
) -> Tuple[plt.Figure, Any]:
    """Generates the four-panel regulatory cross-validation diagnostic dashboard.

    In accordance with CIM MRMR §6.10 and SME Mining Engineering Handbook (§4.5)
    (TODO: Manually Verify), renders:
    1. Actual vs. Estimated Grade Scatter Plot:
       - Displays 1:1 line of perfect agreement and linear regression line.
       - Discloses Pearson correlation r, slope of regression, and RMSE.
    2. Residual Error vs. Estimated Grade Scatter Plot:
       - Tests for conditional bias / heteroscedasticity across the grade range.
    3. Standardized Residuals Q-Q Plot:
       - Checks whether standardized errors (e / sigma_K) follow theoretical standard normal N(0, 1).
    4. Estimation Error Distribution Histogram:
       - Shows error distribution with zero-error reference line and Normal curve fit.

    Parameters
    ----------
    cv_results : pd.DataFrame
        DataFrame generated by kriging_cross_validation containing 'actual', 'estimated',
        'error', and 'standardized_error'.
    grade_unit : str, default "% Cu"
        Grade display unit for axis labeling.
    title : str, optional
        Overall dashboard title.
    figsize : Tuple[float, float], default (14, 11)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Any]
        Matplotlib figure and array of 4 subplot axes.
    """
    req_cols = {"actual", "estimated", "error", "standardized_error"}
    missing = req_cols - set(cv_results.columns)
    if missing:
        raise ValueError(
            f"cv_results is missing required columns: {sorted(missing)}. "
            f"Must be generated by kriging_cross_validation."
        )

    valid_mask = (
        cv_results["actual"].notna()
        & cv_results["estimated"].notna()
        & cv_results["error"].notna()
    )
    if not np.any(valid_mask):
        raise ValueError(
            "cv_results does not contain any valid (non-NaN) estimated samples to plot."
        )

    df_valid = cv_results.loc[valid_mask].copy()
    actual = df_valid["actual"].to_numpy()
    estimated = df_valid["estimated"].to_numpy()
    error = df_valid["error"].to_numpy()

    std_err_valid = df_valid["standardized_error"].dropna().to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # -------------------------------------------------------------------------
    # Panel (0, 0): Actual vs. Estimated Grade Scatter Plot
    # -------------------------------------------------------------------------
    ax00 = axes[0, 0]
    ax00.scatter(
        estimated,
        actual,
        color="#2b5c8f",
        alpha=0.75,
        edgecolors="white",
        s=45,
        zorder=3,
        label="Cross-Validated Samples",
    )

    min_val = min(float(np.min(estimated)), float(np.min(actual)))
    max_val = max(float(np.max(estimated)), float(np.max(actual)))
    span = max_val - min_val
    plot_min = max(0.0, min_val - 0.05 * span)
    plot_max = max_val + 0.05 * span

    ax00.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        label="1:1 Identity Line",
        zorder=2,
    )

    if len(estimated) >= 2 and np.var(estimated) > 1e-12:
        slope, intercept, r_val, _, _ = stats.linregress(estimated, actual)
        x_fit = np.linspace(plot_min, plot_max, 100)
        y_fit = slope * x_fit + intercept
        ax00.plot(
            x_fit,
            y_fit,
            color="#d9381e",
            linestyle="-",
            linewidth=1.8,
            label=f"Regression Fit (y = {slope:.2f}x + {intercept:.2f})",
            zorder=2,
        )
    else:
        slope, intercept, r_val = np.nan, np.nan, np.nan

    rmse = float(np.sqrt(np.mean(error**2)))
    me = float(np.mean(error))
    sor = cv_results.attrs.get("slope_of_regression", slope)
    sor_disp = sor if sor is not None and np.isfinite(sor) else slope

    text_00 = (
        f"N Samples: {len(estimated)}\n"
        f"Pearson r: {r_val:.3f}\n"
        f"Slope of Regr: {sor_disp:.3f}\n"
        f"RMSE: {rmse:.3f} {grade_unit}\n"
        f"Mean Error: {me:+.3f} {grade_unit}"
    )
    ax00.text(
        0.04,
        0.96,
        text_00,
        transform=ax00.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc"
        ),
    )

    ax00.set_xlim(plot_min, plot_max)
    ax00.set_ylim(plot_min, plot_max)
    ax00.set_xlabel(f"Estimated Grade [{grade_unit}]", fontsize=10, fontweight="bold")
    ax00.set_ylabel(f"Actual Grade [{grade_unit}]", fontsize=10, fontweight="bold")
    ax00.set_title(
        "Actual vs. Estimated Grade (Conditional Unbiasedness)",
        fontsize=11,
        fontweight="bold",
    )
    ax00.grid(True, linestyle=":", alpha=0.6)
    ax00.legend(loc="lower right", fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel (0, 1): Residual Error vs. Estimated Grade (Heteroscedasticity Check)
    # -------------------------------------------------------------------------
    ax01 = axes[0, 1]
    ax01.scatter(
        estimated,
        error,
        color="#4a7c59",
        alpha=0.75,
        edgecolors="white",
        s=45,
        zorder=3,
    )
    ax01.axhline(0.0, color="#333333", linestyle="--", linewidth=1.5, zorder=2)

    if len(estimated) >= 2 and np.var(estimated) > 1e-12:
        slope_err, intercept_err, r_err, _, _ = stats.linregress(estimated, error)
        x_err_fit = np.linspace(plot_min, plot_max, 100)
        y_err_fit = slope_err * x_err_fit + intercept_err
        ax01.plot(
            x_err_fit,
            y_err_fit,
            color="#d9381e",
            linestyle="-",
            linewidth=1.6,
            label=f"Drift Fit (slope = {slope_err:+.3f})",
        )
        ax01.legend(loc="upper right", fontsize=8.5)

    text_01 = (
        f"Mean Error: {me:+.3f} {grade_unit}\n"
        f"Error Std Dev: {float(np.std(error)):.3f} {grade_unit}\n"
        f"Zero-Bias Target: 0.0"
    )
    ax01.text(
        0.04,
        0.96,
        text_01,
        transform=ax01.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc"
        ),
    )

    ax01.set_xlim(plot_min, plot_max)
    ax01.set_xlabel(f"Estimated Grade [{grade_unit}]", fontsize=10, fontweight="bold")
    ax01.set_ylabel(
        f"Estimation Error (Actual - Estimated) [{grade_unit}]",
        fontsize=10,
        fontweight="bold",
    )
    ax01.set_title(
        "Residual Error vs. Estimated Grade (Conditional Bias)",
        fontsize=11,
        fontweight="bold",
    )
    ax01.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel (1, 0): Standardized Residuals Q-Q Plot
    # -------------------------------------------------------------------------
    ax10 = axes[1, 0]
    if len(std_err_valid) >= 3:
        (osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(
            std_err_valid, dist="norm"
        )
        ax10.scatter(
            osm,
            osr,
            color="#6b4c9a",
            alpha=0.8,
            edgecolors="white",
            s=45,
            zorder=3,
            label="Sample Quantiles",
        )
        x_qq = np.linspace(float(np.min(osm)), float(np.max(osm)), 100)
        ax10.plot(
            x_qq,
            slope_qq * x_qq + intercept_qq,
            color="#d9381e",
            linestyle="--",
            linewidth=1.8,
            label="Theoretical Fit",
            zorder=2,
        )

        mse = float(np.mean(std_err_valid))
        var_se = float(np.var(std_err_valid, ddof=1))
        text_10 = (
            f"Mean Std Error (MSE): {mse:+.3f} (Target ~ 0.0)\n"
            f"Var Std Error (Var(SE)): {var_se:.3f} (Target ~ 1.0)\n"
            f"Normal Correlation: {r_qq:.3f}"
        )
        ax10.text(
            0.04,
            0.96,
            text_10,
            transform=ax10.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.85,
                edgecolor="#cccccc",
            ),
        )
        ax10.legend(loc="lower right", fontsize=8.5)
    else:
        ax10.text(
            0.5, 0.5, "Insufficient samples for Q-Q plot", ha="center", va="center"
        )

    ax10.set_xlabel(
        "Theoretical Normal Quantiles N(0, 1)", fontsize=10, fontweight="bold"
    )
    ax10.set_ylabel(
        "Ordered Standardized Residuals (e / sigma_K)", fontsize=10, fontweight="bold"
    )
    ax10.set_title(
        "Standardized Residuals Q-Q Plot (Variogram Scaling Check)",
        fontsize=11,
        fontweight="bold",
    )
    ax10.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------------------
    # Panel (1, 1): Estimation Error Distribution Histogram
    # -------------------------------------------------------------------------
    ax11 = axes[1, 1]
    n_bins = max(5, min(30, int(len(error) / 3)))
    count, bins_hist, _ = ax11.hist(
        error,
        bins=n_bins,
        density=True,
        color="#457b9d",
        alpha=0.65,
        edgecolor="#1d3557",
        zorder=2,
        label="Error Histogram",
    )

    err_mean = float(np.mean(error))
    err_std = float(np.std(error))
    if err_std > 1e-9:
        x_norm = np.linspace(bins_hist[0], bins_hist[-1], 200)
        y_norm = stats.norm.pdf(x_norm, err_mean, err_std)
        ax11.plot(
            x_norm,
            y_norm,
            color="#e63946",
            linewidth=2.0,
            label=f"Normal Fit (mu={err_mean:.2f}, sigma={err_std:.2f})",
            zorder=3,
        )

    ax11.axvline(
        0.0, color="#333333", linestyle="--", linewidth=1.5, label="Zero Bias", zorder=4
    )
    ax11.axvline(
        err_mean,
        color="#e63946",
        linestyle=":",
        linewidth=1.8,
        label="Mean Error",
        zorder=4,
    )

    skewness = float(stats.skew(error)) if len(error) >= 3 else 0.0
    kurt = float(stats.kurtosis(error)) if len(error) >= 4 else 0.0
    text_11 = (
        f"Mean Error: {err_mean:+.3f} {grade_unit}\n"
        f"Std Dev: {err_std:.3f} {grade_unit}\n"
        f"Skewness: {skewness:+.2f}\n"
        f"Kurtosis: {kurt:+.2f}"
    )
    ax11.text(
        0.04,
        0.96,
        text_11,
        transform=ax11.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", alpha=0.85, edgecolor="#cccccc"
        ),
    )

    ax11.set_xlabel(
        f"Estimation Error (Actual - Estimated) [{grade_unit}]",
        fontsize=10,
        fontweight="bold",
    )
    ax11.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax11.set_title("Estimation Error Distribution", fontsize=11, fontweight="bold")
    ax11.grid(True, linestyle=":", alpha=0.6)
    ax11.legend(loc="upper right", fontsize=8.5)

    dashboard_title = (
        title
        or "Kriging Model Cross-Validation Diagnostics (CIM MRMR §6.10, SME Ch. 4.5)"
    )
    fig.suptitle(dashboard_title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    return fig, axes


def compare_model_estimation_methods(
    samples_xy: np.ndarray,
    sample_grades: np.ndarray,
    block_model: pd.DataFrame,
    cutoff_grades: Sequence[float],
    grade_cols: Mapping[str, str],
    sample_weights: Optional[np.ndarray] = None,
    grade_unit: str = "% Cu",
    tonnes_col: str = "tonnes",
    metal_factor: float = 0.01,
) -> pd.DataFrame:
    """Produces the global statistical bias and method comparison table.

    Mandated by CIM MRMR Best Practice Guidelines (§6.10) and JORC Code (2012 Table 1)
    (TODO: Manually Verify), compares block estimates across multiple estimation techniques
    against the declustered sample distribution to detect global over/under-estimation and smoothing severity:
    - Declustered Samples (global spatial benchmark)
    - Nearest Neighbor (NN: unbiased global block model benchmark with zero smoothing)
    - Inverse Distance Weighting (IDW)
    - Ordinary Kriging (OK)
    - Simple Kriging (SK)

    Parameters
    ----------
    samples_xy : np.ndarray
        Array of shape (N, 2) or (N, 3) with drillhole sample coordinates.
    sample_grades : np.ndarray
        Array of shape (N,) with continuous raw assay grades.
    block_model : pd.DataFrame
        Block model DataFrame containing estimated grade columns and tonnage.
    cutoff_grades : Sequence[float]
        Sequence of economic cut-off grades to evaluate (e.g., [0.0, 0.3, 0.5, 0.8]).
        Required without default (CONTRIBUTING.md).
    grade_cols : Mapping[str, str]
        Mapping of method display name to block_model column name, e.g.:
        {'Nearest Neighbor': 'grade_nn', 'Ordinary Kriging': 'grade_ok', 'IDW': 'grade_idw'}.
    sample_weights : np.ndarray, optional
        Cell declustering weights for conditioning samples. If None, unweighted mean is used.
    grade_unit : str, default "% Cu"
        Grade display unit.
    tonnes_col : str, default "tonnes"
        Block model column containing block tonnage.
    metal_factor : float, default 0.01
        Multiplier converting grade * tonnes into metal units (0.01 for %, 1e-6 for g/t).

    Returns
    -------
    pd.DataFrame
        Method comparison table reporting Tonnage (Mt), Mean Grade, Contained Metal,
        and relative % differences against Nearest Neighbor benchmark per method across
        all specified cut-off grades.
        Attached in `.attrs`:
        - 'declustered_sample_mean': Declustered sample mean at zero cut-off.
        - 'naive_sample_mean': Unweighted sample mean.
        - 'global_bias_pct': Mapping of method name to global bias % vs declustered samples.
        - 'nn_benchmark_method': Method identified as the NN benchmark.
        - 'footnotes': Compliance disclosures conforming to CIM MRMR §6.10 (TODO: Manually Verify).
    """
    if len(cutoff_grades) == 0:
        raise ValueError(
            "cutoff_grades must be a non-empty sequence of cut-off grades."
        )

    if tonnes_col not in block_model.columns:
        raise ValueError(
            f"tonnes_col '{tonnes_col}' not found in block_model DataFrame."
        )

    total_model_tonnes = float(block_model[tonnes_col].sum())
    if total_model_tonnes <= 0:
        raise ValueError(
            f"Total tonnage in block_model['{tonnes_col}'] must be strictly positive."
        )

    for method_name, col_name in grade_cols.items():
        if col_name not in block_model.columns:
            raise ValueError(
                f"Grade column '{col_name}' for method '{method_name}' not found in block_model."
            )

    s_grades = np.asarray(sample_grades, dtype=float)
    if len(s_grades) == 0:
        raise ValueError("sample_grades must not be empty.")

    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=float)
        if len(weights) != len(s_grades):
            raise ValueError(
                f"sample_weights length ({len(weights)}) must match sample_grades ({len(s_grades)})."
            )
        w_sum = np.sum(weights)
        if w_sum <= 0:
            raise ValueError("Sum of sample_weights must be strictly positive.")
        w_norm = weights / w_sum
    else:
        w_norm = np.ones(len(s_grades), dtype=float) / len(s_grades)

    declustered_sample_mean = float(np.sum(w_norm * s_grades))
    naive_sample_mean = float(np.mean(s_grades))

    # Identify Nearest Neighbor benchmark method
    nn_method_name = None
    for m_name, c_name in grade_cols.items():
        if (
            "nearest" in m_name.lower()
            or "nn" in m_name.lower()
            or "nearest" in c_name.lower()
            or "nn" in c_name.lower()
        ):
            nn_method_name = m_name
            break

    sorted_cutoffs = sorted(cutoff_grades)

    # First pass: compute metrics for all methods across all cutoffs
    records = []
    # Dict to look up NN metrics: cutoff -> {ore_tonnes, mean_grade, contained_metal}
    nn_metrics = {}

    for c in sorted_cutoffs:
        for method_name, col_name in grade_cols.items():
            vals = block_model[col_name].to_numpy(dtype=float)
            tonnes = block_model[tonnes_col].to_numpy(dtype=float)

            valid_mask = np.isfinite(vals) & (vals >= c)
            ore_tonnes = float(np.sum(tonnes[valid_mask]))
            ore_metal = float(
                np.sum(tonnes[valid_mask] * vals[valid_mask] * metal_factor)
            )
            mean_grade = (
                float(np.sum(tonnes[valid_mask] * vals[valid_mask]) / ore_tonnes)
                if ore_tonnes > 0.0
                else 0.0
            )

            rec = {
                "cutoff": c,
                "method": method_name,
                "ore_tonnes": ore_tonnes,
                "mean_grade": mean_grade,
                "contained_metal": ore_metal,
            }
            records.append(rec)

            if method_name == nn_method_name:
                nn_metrics[c] = {
                    "ore_tonnes": ore_tonnes,
                    "mean_grade": mean_grade,
                    "contained_metal": ore_metal,
                }

    # Second pass: calculate relative % differences vs NN benchmark
    for rec in records:
        c = rec["cutoff"]
        if nn_method_name is not None and c in nn_metrics:
            ref = nn_metrics[c]
            rec["tonnes_diff_pct_vs_nn"] = (
                float(
                    (rec["ore_tonnes"] - ref["ore_tonnes"]) / ref["ore_tonnes"] * 100.0
                )
                if ref["ore_tonnes"] > 0
                else 0.0
            )
            rec["grade_diff_pct_vs_nn"] = (
                float(
                    (rec["mean_grade"] - ref["mean_grade"]) / ref["mean_grade"] * 100.0
                )
                if ref["mean_grade"] > 0
                else 0.0
            )
            rec["metal_diff_pct_vs_nn"] = (
                float(
                    (rec["contained_metal"] - ref["contained_metal"])
                    / ref["contained_metal"]
                    * 100.0
                )
                if ref["contained_metal"] > 0
                else 0.0
            )
        else:
            rec["tonnes_diff_pct_vs_nn"] = np.nan
            rec["grade_diff_pct_vs_nn"] = np.nan
            rec["metal_diff_pct_vs_nn"] = np.nan

    res_df = pd.DataFrame(records)

    # Calculate global bias at zero cut-off (or minimum cut-off) vs Declustered Sample Mean
    min_cutoff = sorted_cutoffs[0]
    global_bias_dict = {}
    for method_name in grade_cols.keys():
        m_row = res_df.loc[
            (res_df["cutoff"] == min_cutoff) & (res_df["method"] == method_name)
        ]
        if not m_row.empty:
            m_grade = float(m_row["mean_grade"].iloc[0])
            bias_pct = (
                float(
                    (m_grade - declustered_sample_mean)
                    / declustered_sample_mean
                    * 100.0
                )
                if declustered_sample_mean > 0
                else np.nan
            )
            global_bias_dict[method_name] = bias_pct

    res_df.attrs["declustered_sample_mean"] = declustered_sample_mean
    res_df.attrs["naive_sample_mean"] = naive_sample_mean
    res_df.attrs["global_bias_pct"] = global_bias_dict
    res_df.attrs["nn_benchmark_method"] = nn_method_name
    res_df.attrs["footnotes"] = [
        "Model estimation method comparison in accordance with CIM MRMR Best Practice Guidelines (§6.10) "
        "and JORC Code (2012 Table 1) (TODO: Manually Verify).",
        f"Global Declustered Sample Mean = {declustered_sample_mean:.3f} {grade_unit} "
        f"(Naive Sample Mean = {naive_sample_mean:.3f} {grade_unit}).",
        "Nearest Neighbor (NN) represents the declustered, volumetrically-weighted block model benchmark "
        "with zero spatial smoothing (CIM MRMR §6.10).",
        "Global bias at zero cut-off within +/- 2% to +/- 5% indicates acceptable global calibration; "
        "biases exceeding +/- 5% require investigation into search neighborhood parameters.",
    ]

    return res_df


# =============================================================================
# PART 2: DRILLHOLE ASSAY QA/QC SUITE (NI 43-101 ITEM 11 & JORC TABLE 1 SEC 1)
# =============================================================================


def analyze_crm_standards(
    assay_df: pd.DataFrame,
    standard_id_col: str,
    grade_col: str,
    certified_values: Mapping[str, Mapping[str, float]],
    date_or_seq_col: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluates Certified Reference Material (CRM / Standard) assay accuracy and drift.

    In accordance with CIM MRMR Best Practice Guidelines (2019, Section 3 on Data
    Verification and QA/QC) (TODO: Manually Verify), SME Mining Engineering Handbook
    (3rd Ed., Section 4.2) (TODO: Manually Verify), and NI 43-101 Form 43-101F1 Item 11
    (TODO: Manually Verify), evaluates analytical laboratory accuracy, instrument calibration,
    and analytical drift using standard Shewhart control chart rules.

    Failure Criteria (CIM MRMR §3.3; Long 1998; SME §4.2) (TODO: Manually Verify):
    ------------------------------------------------------------------------------
    - Pass: Assayed grade within +/- 2.0*sigma of certified mean (|z| <= 2.0).
    - Warning (+/-2s): Single assay between +/- 2.0*sigma and +/- 3.0*sigma (2.0 < |z| <= 3.0).
    - Consecutive Warning Failure: Two consecutive assays > 2.0*sigma on the same side
      of the mean for the same standard (indicates analytical drift or instrument recalibration error).
    - Critical Failure (+/-3s): Single assay deviating beyond +/- 3.0*sigma from certified mean (|z| > 3.0).

    Parameters
    ----------
    assay_df : pd.DataFrame
        DataFrame containing assay records including CRM standard sample insertions.
    standard_id_col : str
        Column identifying the CRM standard ID (e.g., 'OREAS 501', 'CDN-GS-1P5').
    grade_col : str
        Column containing laboratory assayed grades.
    certified_values : Mapping[str, Mapping[str, float]]
        Dictionary mapping standard IDs to certified values:
        {'OREAS 501': {'mean': 1.25, 'std': 0.04}, ...}.
        Both 'mean' and 'std' must be strictly positive floats.
    date_or_seq_col : str, optional
        Column containing assay date, submission batch, or sample sequence order.
        If provided, entries are sorted chronologically per standard to detect drift.

    Returns
    -------
    pd.DataFrame
        CRM audit table with columns:
        - standard_id : str
        - measured_grade : float
        - certified_mean : float
        - certified_std : float
        - z_score : float, (measured - mean) / std
        - relative_error_pct : float, (measured - mean) / mean * 100
        - status : str, 'Pass', 'Warning (+/-2s)', 'Failure (+/-3s)', or 'Failure (2 Consecutive +/-2s)'
        - is_pass : bool
        - is_warning : bool
        - is_failure : bool
        Attached in `.attrs`:
        - 'summary_by_standard' : pd.DataFrame summarizing count, certified mean/std, measured
          mean/std, mean bias %, pass count, failure count, and pass rate % per standard.
        - 'total_crm_samples' : int
        - 'overall_pass_rate_pct' : float
        - 'overall_failure_count' : int
        - 'compliance_verdict' : 'PASSED' (if overall pass rate >= 98% and no unresolved failures)
          or 'ACTION REQUIRED: Bias or Excessive Failure Detected'
        - 'footnotes' : list of compliance citations with (TODO: Manually Verify).
    """
    if not isinstance(assay_df, pd.DataFrame):
        raise TypeError(f"assay_df must be a pandas DataFrame, got {type(assay_df)}")
    if standard_id_col not in assay_df.columns:
        raise ValueError(f"standard_id_col '{standard_id_col}' not found in assay_df")
    if grade_col not in assay_df.columns:
        raise ValueError(f"grade_col '{grade_col}' not found in assay_df")
    if date_or_seq_col is not None and date_or_seq_col not in assay_df.columns:
        raise ValueError(f"date_or_seq_col '{date_or_seq_col}' not found in assay_df")
    if not certified_values or not isinstance(certified_values, (dict, Mapping)):
        raise ValueError("certified_values must be a non-empty mapping of standard IDs to certified values")

    for sid, vals in certified_values.items():
        if not isinstance(vals, (dict, Mapping)) or "mean" not in vals or "std" not in vals:
            raise ValueError(f"Standard '{sid}' in certified_values must contain 'mean' and 'std' keys")
        if vals["mean"] <= 0 or vals["std"] <= 0:
            raise ValueError(f"Standard '{sid}' certified mean and std must be strictly positive (> 0)")

    # Filter for rows matching certified standards
    valid_mask = (
        assay_df[standard_id_col].isin(certified_values.keys())
        & assay_df[grade_col].notna()
        & np.isfinite(pd.to_numeric(assay_df[grade_col], errors="coerce"))
    )
    df_crm = assay_df[valid_mask].copy()
    if len(df_crm) == 0:
        raise ValueError(
            f"No valid CRM standard samples found matching certified standard IDs: {list(certified_values.keys())}"
        )

    if date_or_seq_col is not None:
        df_crm = df_crm.sort_values(by=date_or_seq_col).reset_index(drop=True)
    else:
        df_crm = df_crm.reset_index(drop=True)

    records = []
    # Process per standard to evaluate consecutive warnings
    for std_id in certified_values.keys():
        std_mask = df_crm[standard_id_col] == std_id
        if not std_mask.any():
            continue
        sub_df = df_crm[std_mask]
        c_mean = float(certified_values[std_id]["mean"])
        c_std = float(certified_values[std_id]["std"])

        prev_warning_side = 0  # +1 if > +2s, -1 if < -2s, 0 otherwise

        for idx, row in sub_df.iterrows():
            m_grade = float(row[grade_col])
            z = (m_grade - c_mean) / c_std
            rel_err_pct = ((m_grade - c_mean) / c_mean) * 100.0

            # Shewhart status determination
            if abs(z) > 3.0:
                status = "Failure (+/-3s)"
                is_fail = True
                is_warn = False
                prev_warning_side = 0
            elif 2.0 < z <= 3.0:
                if prev_warning_side == 1:
                    status = "Failure (2 Consecutive +/-2s)"
                    is_fail = True
                    is_warn = False
                    prev_warning_side = 0
                else:
                    status = "Warning (+/-2s)"
                    is_fail = False
                    is_warn = True
                    prev_warning_side = 1
            elif -3.0 <= z < -2.0:
                if prev_warning_side == -1:
                    status = "Failure (2 Consecutive +/-2s)"
                    is_fail = True
                    is_warn = False
                    prev_warning_side = 0
                else:
                    status = "Warning (+/-2s)"
                    is_fail = False
                    is_warn = True
                    prev_warning_side = -1
            else:
                status = "Pass"
                is_fail = False
                is_warn = False
                prev_warning_side = 0

            rec = {
                "orig_idx": idx,
                "standard_id": str(std_id),
                "measured_grade": m_grade,
                "certified_mean": c_mean,
                "certified_std": c_std,
                "z_score": float(z),
                "relative_error_pct": float(rel_err_pct),
                "status": status,
                "is_pass": bool(not is_warn and not is_fail),
                "is_warning": bool(is_warn),
                "is_failure": bool(is_fail),
            }
            if date_or_seq_col is not None:
                rec[date_or_seq_col] = row[date_or_seq_col]
            records.append(rec)

    res_df = pd.DataFrame(records).sort_values("orig_idx").drop(columns=["orig_idx"]).reset_index(drop=True)

    # Compute summary per standard
    std_summaries = []
    for std_id in sorted(res_df["standard_id"].unique()):
        s_df = res_df[res_df["standard_id"] == std_id]
        c_mean = float(s_df["certified_mean"].iloc[0])
        c_std = float(s_df["certified_std"].iloc[0])
        m_mean = float(s_df["measured_grade"].mean())
        m_std = float(s_df["measured_grade"].std(ddof=1)) if len(s_df) > 1 else 0.0
        bias_pct = ((m_mean - c_mean) / c_mean) * 100.0 if c_mean > 0 else 0.0
        n_tot = len(s_df)
        n_pass = int(s_df["is_pass"].sum())
        n_warn = int(s_df["is_warning"].sum())
        n_fail = int(s_df["is_failure"].sum())
        pass_rate = (n_pass / n_tot) * 100.0 if n_tot > 0 else 0.0

        std_summaries.append({
            "standard_id": std_id,
            "sample_count": n_tot,
            "certified_mean": c_mean,
            "certified_std": c_std,
            "measured_mean": m_mean,
            "measured_std": m_std,
            "mean_bias_pct": bias_pct,
            "pass_count": n_pass,
            "warning_count": n_warn,
            "failure_count": n_fail,
            "pass_rate_pct": pass_rate,
            "status": "Acceptable" if pass_rate >= 95.0 and abs(bias_pct) <= 5.0 else "Action Required",
        })

    summary_df = pd.DataFrame(std_summaries)

    tot_crm = len(res_df)
    tot_pass = int(res_df["is_pass"].sum())
    tot_warn = int(res_df["is_warning"].sum())
    tot_fail = int(res_df["is_failure"].sum())
    overall_pass_rate = (tot_pass / tot_crm) * 100.0 if tot_crm > 0 else 0.0

    verdict = (
        "PASSED"
        if overall_pass_rate >= 98.0 and tot_fail == 0
        else ("ACCEPTABLE WITH MONITORING" if overall_pass_rate >= 95.0 else "ACTION REQUIRED: Lab Re-Assay Triggered")
    )

    res_df.attrs["summary_by_standard"] = summary_df
    res_df.attrs["total_crm_samples"] = tot_crm
    res_df.attrs["overall_pass_count"] = tot_pass
    res_df.attrs["overall_warning_count"] = tot_warn
    res_df.attrs["overall_failure_count"] = tot_fail
    res_df.attrs["overall_pass_rate_pct"] = overall_pass_rate
    res_df.attrs["compliance_verdict"] = verdict
    res_df.attrs["footnotes"] = [
        "CRM analysis conducted under CIM MRMR Best Practice Guidelines (2019, Section 3) (TODO: Manually Verify).",
        "Shewhart control limits: Warning = +/- 2*sigma; Failure = +/- 3*sigma or two consecutive +/- 2*sigma (TODO: Manually Verify).",
        "Statutory compliance under NI 43-101 Form 43-101F1 Item 11 and JORC Code (2012, Table 1 Section 1) (TODO: Manually Verify).",
        "Target industry acceptance: overall pass rate >= 98.0% with zero unresolved critical failures (TODO: Manually Verify).",
    ]

    return res_df


def plot_crm_control_chart(
    crm_analysis_results: pd.DataFrame,
    standard_name: str,
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> Tuple[plt.Figure, Any]:
    """Renders an official Shewhart Quality Control Chart for a Certified Reference Material.

    Mandated by NI 43-101 Form 43-101F1 Item 11 and JORC Table 1 Section 1 (TODO: Manually Verify),
    displays chronological assay measurements against certified mean, warning (+/- 2*sigma), and
    failure (+/- 3*sigma) control thresholds. Color-codes compliant assays, warning flags, and
    critical failures.

    Parameters
    ----------
    crm_analysis_results : pd.DataFrame
        Audited table produced by analyze_crm_standards.
    standard_name : str
        Name of the specific standard to plot.
    grade_unit : str, default "% Cu"
        Grade display unit.
    title : str, optional
        Plot title. If None, automatically formatted with standard name.
    figsize : Tuple[float, float], default (14, 6)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Any]
        Matplotlib figure and axes.
    """
    if not isinstance(crm_analysis_results, pd.DataFrame):
        raise TypeError(f"crm_analysis_results must be a DataFrame, got {type(crm_analysis_results)}")
    if "standard_id" not in crm_analysis_results.columns:
        raise ValueError("crm_analysis_results must contain 'standard_id' column")

    std_df = crm_analysis_results[crm_analysis_results["standard_id"] == standard_name].copy()
    if len(std_df) == 0:
        available = list(crm_analysis_results["standard_id"].unique()) if len(crm_analysis_results) > 0 else []
        raise ValueError(
            f"Standard '{standard_name}' not found in crm_analysis_results. Available standards: {available}"
        )

    fig, ax = plt.subplots(figsize=figsize)

    c_mean = float(std_df["certified_mean"].iloc[0])
    c_std = float(std_df["certified_std"].iloc[0])
    x = np.arange(1, len(std_df) + 1)
    y = std_df["measured_grade"].values

    # Control Limits
    plus_2s = c_mean + 2.0 * c_std
    minus_2s = c_mean - 2.0 * c_std
    plus_3s = c_mean + 3.0 * c_std
    minus_3s = c_mean - 3.0 * c_std

    # Plot sequence line connecting points
    ax.plot(x, y, color="#9e9e9e", linestyle="-", linewidth=1.2, alpha=0.7, zorder=2)

    # Plot horizontal control lines
    ax.axhline(c_mean, color="#2e7d32", linestyle="-", linewidth=2.0, label=f"Certified Mean: {c_mean:.3f} {grade_unit}", zorder=1)
    ax.axhline(plus_2s, color="#ef6c00", linestyle="--", linewidth=1.5, label="±2σ Warning Limit", zorder=1)
    ax.axhline(minus_2s, color="#ef6c00", linestyle="--", linewidth=1.5, zorder=1)
    ax.axhline(plus_3s, color="#c62828", linestyle="-", linewidth=1.8, label="±3σ Failure Limit", zorder=1)
    ax.axhline(minus_3s, color="#c62828", linestyle="-", linewidth=1.8, zorder=1)

    # Shaded control zones
    x_span = [0.5, len(std_df) + 0.5]
    ax.fill_between(x_span, minus_2s, plus_2s, color="#e8f5e9", alpha=0.4, label="Acceptable (±2σ)")
    ax.fill_between(x_span, plus_2s, plus_3s, color="#fff3e0", alpha=0.6, label="Warning (2σ to 3σ)")
    ax.fill_between(x_span, minus_3s, minus_2s, color="#fff3e0", alpha=0.6)
    y_top = max(np.max(y) * 1.05, plus_3s + c_std)
    y_bot = min(np.min(y) * 0.95, minus_3s - c_std)
    ax.fill_between(x_span, plus_3s, y_top, color="#ffebee", alpha=0.5, label="Failure (>3σ)")
    ax.fill_between(x_span, y_bot, minus_3s, color="#ffebee", alpha=0.5)

    # Plot points by status
    pass_mask = std_df["status"] == "Pass"
    warn_mask = std_df["status"].str.startswith("Warning")
    fail_mask = std_df["status"].str.startswith("Failure")

    if pass_mask.any():
        ax.scatter(x[pass_mask], y[pass_mask], color="#2e7d32", marker="o", s=45, zorder=4, label=f"Pass ({pass_mask.sum()})")
    if warn_mask.any():
        ax.scatter(x[warn_mask], y[warn_mask], color="#ef6c00", marker="s", s=65, zorder=5, edgecolor="black", label=f"Warning ({warn_mask.sum()})")
    if fail_mask.any():
        ax.scatter(x[fail_mask], y[fail_mask], color="#c62828", marker="X", s=90, zorder=6, edgecolor="black", label=f"Failure ({fail_mask.sum()})")

    ax.set_xlim(0.5, len(std_df) + 0.5)
    ax.set_ylim(y_bot, y_top)

    # Summary callout box
    n_tot = len(std_df)
    n_pass = int(pass_mask.sum())
    n_fail = int(fail_mask.sum())
    pass_pct = (n_pass / n_tot) * 100.0 if n_tot > 0 else 0.0
    m_mean = float(np.mean(y))
    m_std = float(np.std(y, ddof=1)) if n_tot > 1 else 0.0
    bias_pct = ((m_mean - c_mean) / c_mean) * 100.0 if c_mean > 0 else 0.0

    callout_text = (
        f"CRM Standard: {standard_name}\n"
        f"Total Samples (N): {n_tot}\n"
        f"Certified: {c_mean:.3f} ± {c_std:.3f} {grade_unit}\n"
        f"Measured: {m_mean:.3f} ± {m_std:.3f} {grade_unit}\n"
        f"Mean Bias: {bias_pct:+.2f}%\n"
        f"Pass Rate: {pass_pct:.1f}%\n"
        f"Critical Failures: {n_fail}\n"
        f"Status: {'COMPLIANT' if pass_pct >= 95.0 and n_fail == 0 else 'ACTION REQUIRED'}"
    )
    ax.text(
        0.02,
        0.95,
        callout_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#b0bec5", alpha=0.9),
        zorder=10,
    )

    ax.set_xlabel("Assay Submission / Insertion Sequence", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Assayed Grade ({grade_unit})", fontsize=11, fontweight="bold")
    chart_title = title or f"Shewhart Control Chart - CRM Standard {standard_name} (NI 43-101 / JORC Compliance)"
    ax.set_title(chart_title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)

    plt.tight_layout()
    return fig, ax


def analyze_analytical_blanks(
    assay_df: pd.DataFrame,
    blank_flag_col: str,
    grade_col: str,
    detection_limit: float,
    failure_multiplier: float = 5.0,
    warning_multiplier: float = 3.0,
    date_or_seq_col: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluates analytical coarse and pulp blanks for sample carryover / cross-contamination.

    In accordance with CIM MRMR Best Practice Guidelines (2019, Section 3) (TODO: Manually Verify),
    SME Mining Engineering Handbook (3rd Ed., Section 4.2) (TODO: Manually Verify), and NI 43-101
    Form 43-101F1 Item 11 (TODO: Manually Verify), unmineralized coarse and pulp blanks inserted
    into sample batches detect cross-contamination during laboratory crushing, pulverizing, and
    fire assay / ICP digestions.

    Regulatory Acceptance & Failure Standards (CIM MRMR §3.2; SME §4.2) (TODO: Manually Verify):
    --------------------------------------------------------------------------------------------
    - Pass: Assayed blank grade <= warning_multiplier * LLD (typically <= 3x LLD).
    - Warning: Blank grade between warning_multiplier * LLD and failure_multiplier * LLD (3x to 5x LLD).
    - Failure: Blank grade > failure_multiplier * LLD (typically > 5x LLD).
      Critical carryover failure indicates contamination from preceding high-grade samples and
      mandates re-assay of the affected analytical batch.

    Parameters
    ----------
    assay_df : pd.DataFrame
        DataFrame containing assay records including blank insertions.
    blank_flag_col : str
        Column identifying blank sample insertions (boolean, integer 1, or string 'Blank').
    grade_col : str
        Assayed grade column name.
    detection_limit : float
        Analytical Lower Limit of Detection (LLD) for the commodity (> 0).
    failure_multiplier : float, default 5.0
        Multiple of detection limit defining critical contamination failure (standard is 5x LLD).
    warning_multiplier : float, default 3.0
        Multiple of detection limit defining contamination warning (standard is 3x LLD).
    date_or_seq_col : str, optional
        Date or sample sequence column for chronological tracking.

    Returns
    -------
    pd.DataFrame
        Blank evaluation audit table with columns:
        - measured_grade : float
        - detection_limit : float
        - warning_threshold : float
        - failure_threshold : float
        - is_contamination_warning : bool
        - is_contamination_failure : bool
        - is_pass : bool
        - status : str ('Pass', 'Warning (>3x LLD)', 'Failure (>5x LLD)')
        - preceding_grade : float (grade of immediately preceding sample in stream, or NaN)
        - carryover_ratio_pct : float ((measured - LLD) / preceding_grade * 100 if preceding > 0 else 0.0)
        Attached in `.attrs`:
        - 'total_blanks' : int
        - 'pass_count' : int
        - 'warning_count' : int
        - 'failure_count' : int
        - 'pass_rate_pct' : float
        - 'failure_rate_pct' : float
        - 'max_blank_grade' : float
        - 'compliance_verdict' : 'PASSED' (if failure rate <= 1.0%) or 'ACTION REQUIRED: Cross-Contamination Detected'
        - 'footnotes' : list of compliance citations with (TODO: Manually Verify).
    """
    if not isinstance(assay_df, pd.DataFrame):
        raise TypeError(f"assay_df must be a pandas DataFrame, got {type(assay_df)}")
    if blank_flag_col not in assay_df.columns:
        raise ValueError(f"blank_flag_col '{blank_flag_col}' not found in assay_df")
    if grade_col not in assay_df.columns:
        raise ValueError(f"grade_col '{grade_col}' not found in assay_df")
    if date_or_seq_col is not None and date_or_seq_col not in assay_df.columns:
        raise ValueError(f"date_or_seq_col '{date_or_seq_col}' not found in assay_df")
    if detection_limit <= 0:
        raise ValueError(f"detection_limit must be strictly positive (> 0), got {detection_limit}")
    if failure_multiplier <= warning_multiplier:
        raise ValueError(
            f"failure_multiplier ({failure_multiplier}) must be greater than warning_multiplier ({warning_multiplier})"
        )

    def _is_blank(val: Any) -> bool:
        if pd.isna(val):
            return False
        if isinstance(val, (bool, np.bool_)):
            return bool(val)
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val == 1
        s = str(val).strip().lower()
        return s in {"true", "1", "blank", "coarse_blank", "pulp_blank", "yes", "b", "blk"}

    if date_or_seq_col is not None:
        stream_df = assay_df.sort_values(by=date_or_seq_col).reset_index(drop=True)
    else:
        stream_df = assay_df.reset_index(drop=True)

    is_blank_series = stream_df[blank_flag_col].apply(_is_blank)
    if not is_blank_series.any():
        raise ValueError(f"No blank samples identified in column '{blank_flag_col}'")

    warn_thresh = float(warning_multiplier * detection_limit)
    fail_thresh = float(failure_multiplier * detection_limit)

    blank_indices = stream_df.index[is_blank_series].tolist()
    records = []
    for b_idx in blank_indices:
        b_row = stream_df.iloc[b_idx]
        m_grade = float(b_row[grade_col])

        prec_grade = np.nan
        if b_idx > 0:
            p_val = stream_df.iloc[b_idx - 1][grade_col]
            try:
                prec_grade = float(p_val)
            except (ValueError, TypeError):
                prec_grade = np.nan

        is_fail = bool(m_grade > fail_thresh)
        is_warn = bool(m_grade > warn_thresh and not is_fail)
        is_pass = bool(m_grade <= warn_thresh)

        if is_fail:
            status = f"Failure (>{failure_multiplier:.0f}x LLD)"
        elif is_warn:
            status = f"Warning (>{warning_multiplier:.0f}x LLD)"
        else:
            status = "Pass"

        carryover = 0.0
        if prec_grade > 0 and m_grade > detection_limit:
            carryover = ((m_grade - detection_limit) / prec_grade) * 100.0

        rec = {
            "measured_grade": m_grade,
            "detection_limit": float(detection_limit),
            "warning_threshold": warn_thresh,
            "failure_threshold": fail_thresh,
            "is_contamination_warning": is_warn,
            "is_contamination_failure": is_fail,
            "is_pass": is_pass,
            "status": status,
            "preceding_grade": float(prec_grade) if np.isfinite(prec_grade) else np.nan,
            "carryover_ratio_pct": float(carryover),
        }
        if date_or_seq_col is not None:
            rec[date_or_seq_col] = b_row[date_or_seq_col]
        records.append(rec)

    res_df = pd.DataFrame(records)

    n_tot = len(res_df)
    n_pass = int(res_df["is_pass"].sum())
    n_warn = int(res_df["is_contamination_warning"].sum())
    n_fail = int(res_df["is_contamination_failure"].sum())
    pass_rate = (n_pass / n_tot) * 100.0 if n_tot > 0 else 0.0
    fail_rate = (n_fail / n_tot) * 100.0 if n_tot > 0 else 0.0
    max_grade = float(res_df["measured_grade"].max())

    verdict = "PASSED" if fail_rate <= 1.0 else "ACTION REQUIRED: Contamination Detected"

    res_df.attrs["total_blanks"] = n_tot
    res_df.attrs["pass_count"] = n_pass
    res_df.attrs["warning_count"] = n_warn
    res_df.attrs["failure_count"] = n_fail
    res_df.attrs["pass_rate_pct"] = pass_rate
    res_df.attrs["failure_rate_pct"] = fail_rate
    res_df.attrs["max_blank_grade"] = max_grade
    res_df.attrs["detection_limit"] = float(detection_limit)
    res_df.attrs["warning_threshold"] = warn_thresh
    res_df.attrs["failure_threshold"] = fail_thresh
    res_df.attrs["compliance_verdict"] = verdict
    res_df.attrs["footnotes"] = [
        "Analytical blanks evaluated under CIM MRMR Best Practice Guidelines (2019, Section 3) (TODO: Manually Verify).",
        f"Contamination thresholds: Warning = {warning_multiplier:.1f}x LLD, Failure = {failure_multiplier:.1f}x LLD (TODO: Manually Verify).",
        "Statutory compliance under NI 43-101 Form 43-101F1 Item 11 and JORC Code (2012, Table 1 Section 1) (TODO: Manually Verify).",
        "Industry standard acceptance criterion: <= 1.0% contamination failures in analytical blank stream (TODO: Manually Verify).",
    ]

    return res_df


def plot_blanks_performance(
    blanks_analysis_results: pd.DataFrame,
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> Tuple[plt.Figure, Any]:
    """Generates the chronological blank contamination scatter plot for NI 43-101 / JORC reports.

    Parameters
    ----------
    blanks_analysis_results : pd.DataFrame
        DataFrame generated by analyze_analytical_blanks.
    grade_unit : str, default "% Cu"
        Grade display unit.
    title : str, optional
        Plot title.
    figsize : Tuple[float, float], default (14, 6)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Any]
        Matplotlib figure and axes.
    """
    if not isinstance(blanks_analysis_results, pd.DataFrame):
        raise TypeError(f"blanks_analysis_results must be a DataFrame, got {type(blanks_analysis_results)}")
    if len(blanks_analysis_results) == 0:
        raise ValueError("blanks_analysis_results DataFrame is empty")

    fig, ax = plt.subplots(figsize=figsize)

    df = blanks_analysis_results
    x = np.arange(1, len(df) + 1)
    y = df["measured_grade"].values

    lld = float(df["detection_limit"].iloc[0]) if "detection_limit" in df.columns else float(df.attrs.get("detection_limit", 0.001))
    warn_thresh = float(df["warning_threshold"].iloc[0]) if "warning_threshold" in df.columns else lld * 3.0
    fail_thresh = float(df["failure_threshold"].iloc[0]) if "failure_threshold" in df.columns else lld * 5.0

    # Horizontal guide lines
    ax.axhline(lld, color="#546e7a", linestyle=":", linewidth=1.5, label=f"Detection Limit (LLD): {lld:.4f} {grade_unit}", zorder=1)
    ax.axhline(warn_thresh, color="#ef6c00", linestyle="--", linewidth=1.5, label=f"Warning Threshold (3x LLD): {warn_thresh:.4f} {grade_unit}", zorder=1)
    ax.axhline(fail_thresh, color="#c62828", linestyle="-", linewidth=2.0, label=f"Failure Threshold (5x LLD): {fail_thresh:.4f} {grade_unit}", zorder=1)

    # Shaded acceptable zone
    x_span = [0.5, len(df) + 0.5]
    ax.fill_between(x_span, 0, warn_thresh, color="#e8f5e9", alpha=0.4, label="Acceptable (≤ 3x LLD)")
    ax.fill_between(x_span, warn_thresh, fail_thresh, color="#fff3e0", alpha=0.5, label="Warning (3x to 5x LLD)")
    y_top = max(np.max(y) * 1.15, fail_thresh * 1.3)
    ax.fill_between(x_span, fail_thresh, y_top, color="#ffebee", alpha=0.4, label="Failure (> 5x LLD)")

    # Scatter points by status
    pass_mask = df["is_pass"] if "is_pass" in df.columns else (y <= warn_thresh)
    warn_mask = df["is_contamination_warning"] if "is_contamination_warning" in df.columns else ((y > warn_thresh) & (y <= fail_thresh))
    fail_mask = df["is_contamination_failure"] if "is_contamination_failure" in df.columns else (y > fail_thresh)

    if pass_mask.any():
        ax.scatter(x[pass_mask], y[pass_mask], color="#2e7d32", marker="o", s=45, zorder=3, label=f"Compliant Blanks ({pass_mask.sum()})")
    if warn_mask.any():
        ax.scatter(x[warn_mask], y[warn_mask], color="#ef6c00", marker="s", s=65, zorder=4, edgecolor="black", label=f"Warnings ({warn_mask.sum()})")
    if fail_mask.any():
        ax.scatter(x[fail_mask], y[fail_mask], color="#c62828", marker="X", s=90, zorder=5, edgecolor="black", label=f"Failures ({fail_mask.sum()})")

    ax.set_xlim(0.5, len(df) + 0.5)
    ax.set_ylim(0, y_top)

    # Statistics callout box
    n_tot = len(df)
    n_pass = int(pass_mask.sum())
    n_fail = int(fail_mask.sum())
    pass_pct = (n_pass / n_tot) * 100.0 if n_tot > 0 else 0.0
    max_val = float(np.max(y))

    callout_text = (
        f"Analytical Blanks Control\n"
        f"Total Blanks Inserted (N): {n_tot}\n"
        f"Detection Limit (LLD): {lld:.4f} {grade_unit}\n"
        f"Warning (3x LLD): {warn_thresh:.4f} {grade_unit}\n"
        f"Failure (5x LLD): {fail_thresh:.4f} {grade_unit}\n"
        f"Pass Rate: {pass_pct:.1f}%\n"
        f"Contamination Failures: {n_fail}\n"
        f"Max Blank Assay: {max_val:.4f} {grade_unit}\n"
        f"Compliance: {'PASSED (No Carryover)' if n_fail == 0 else 'ACTION REQUIRED (Contamination Detected)'}"
    )
    ax.text(
        0.02,
        0.95,
        callout_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#b0bec5", alpha=0.9),
        zorder=10,
    )

    ax.set_xlabel("Blank Sample Insertion Sequence", fontsize=11, fontweight="bold")
    ax.set_ylabel(f"Assayed Grade ({grade_unit})", fontsize=11, fontweight="bold")
    plot_title = title or "Analytical Blank Performance - Cross-Contamination & Carryover Audit"
    ax.set_title(plot_title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8)

    plt.tight_layout()
    return fig, ax


def analyze_assay_duplicates(
    duplicate_df: pd.DataFrame,
    original_col: str,
    duplicate_col: str,
    pair_type: Literal["field", "coarse", "pulp"] = "pulp",
    detection_limit: Optional[float] = None,
) -> pd.DataFrame:
    """Evaluates sampling, preparation, and analytical precision using duplicate assays.

    Under CIM MRMR Guidelines (§3.3) (TODO: Manually Verify) and SME Mining Engineering
    Handbook (§4.2) (TODO: Manually Verify), duplicates segregate sampling variance
    across three distinct stages:
    1. 'field' (quarter core / rig splits): Measures geological heterogeneity + sampling error.
    2. 'coarse' (crushed reject splits): Measures sub-sampling and crushing precision.
    3. 'pulp' (pulverized analytical splits): Measures pulverization and analytical precision.

    Regulatory Acceptance Thresholds (CIM MRMR §3.3; Stanley & Lawie 2007) (TODO: Manually Verify):
    ---------------------------------------------------------------------------------------------
    - Pulp duplicates: >= 90% of pairs must have Half Absolute Relative Difference (HARD) <= 10%.
    - Coarse duplicates: >= 90% of pairs must have HARD <= 20%.
    - Field duplicates: >= 90% of pairs must have HARD <= 30%.

    Mathematical Metrics Computed:
    ------------------------------
    - Relative Difference: RD = (Duplicate - Original) / Mean(Original, Duplicate) * 100.
    - Half Absolute Relative Difference: HARD = |Duplicate - Original| / (Duplicate + Original) * 100.
    - Reduced Major Axis (RMA) Regression Slope and Intercept.

    Parameters
    ----------
    duplicate_df : pd.DataFrame
        DataFrame containing paired original and duplicate assay values.
    original_col : str
        Column name of the primary/original assay grade.
    duplicate_col : str
        Column name of the duplicate assay grade.
    pair_type : {"field", "coarse", "pulp"}, default "pulp"
        Duplicate stage type determining acceptance criteria.
    detection_limit : float, optional
        Lower limit of detection; pairs with values < 10x LLD may be flagged
        to prevent spurious high relative errors near analytical detection limits.

    Returns
    -------
    pd.DataFrame
        Paired duplicate analysis table with original, duplicate, mean, RD, and HARD.
        Attached in `.attrs`:
        - 'pair_type', 'total_pairs', 'hard_threshold_pct', 'pct_meeting_hard_threshold'
        - 'rma_slope', 'rma_intercept', 'pearson_correlation'.
        - 'footnotes' : list of statutory citations with (TODO: Manually Verify).
    """
    if not isinstance(duplicate_df, pd.DataFrame):
        raise TypeError(f"duplicate_df must be a pandas DataFrame, got {type(duplicate_df)}")
    if original_col not in duplicate_df.columns:
        raise ValueError(f"original_col '{original_col}' not found in duplicate_df")
    if duplicate_col not in duplicate_df.columns:
        raise ValueError(f"duplicate_col '{duplicate_col}' not found in duplicate_df")
    if pair_type not in {"field", "coarse", "pulp"}:
        raise ValueError(f"pair_type must be one of 'field', 'coarse', 'pulp', got '{pair_type}'")
    if detection_limit is not None and detection_limit <= 0:
        raise ValueError(f"detection_limit must be strictly positive (> 0), got {detection_limit}")

    hard_threshold_map = {"pulp": 10.0, "coarse": 20.0, "field": 30.0}
    hard_threshold = hard_threshold_map[pair_type]

    orig_s = pd.to_numeric(duplicate_df[original_col], errors="coerce")
    dup_s = pd.to_numeric(duplicate_df[duplicate_col], errors="coerce")
    valid_mask = orig_s.notna() & dup_s.notna() & np.isfinite(orig_s) & np.isfinite(dup_s) & (orig_s > 0) & (dup_s > 0)

    if not valid_mask.any():
        raise ValueError("No valid positive numeric assay duplicate pairs found")

    sub_orig = orig_s[valid_mask].values
    sub_dup = dup_s[valid_mask].values

    pair_mean = (sub_orig + sub_dup) / 2.0
    diff = sub_dup - sub_orig
    rel_diff_pct = (diff / pair_mean) * 100.0
    hard_pct = (np.abs(diff) / (sub_orig + sub_dup)) * 100.0
    meets_hard = hard_pct <= hard_threshold

    if detection_limit is not None:
        above_10x_lld = pair_mean >= 10.0 * detection_limit
    else:
        above_10x_lld = np.ones(len(pair_mean), dtype=bool)

    res_df = pd.DataFrame({
        "original": sub_orig,
        "duplicate": sub_dup,
        "pair_mean": pair_mean,
        "difference": diff,
        "relative_difference_pct": rel_diff_pct,
        "hard_pct": hard_pct,
        "meets_hard_threshold": meets_hard,
        "above_10x_lld": above_10x_lld,
    })

    # Precision evaluation on the relevant subset
    eval_mask = above_10x_lld if np.any(above_10x_lld) else np.ones(len(res_df), dtype=bool)
    eval_df = res_df[eval_mask]
    n_eval = len(eval_df)
    n_meet = int(eval_df["meets_hard_threshold"].sum())
    pct_meeting = (n_meet / n_eval) * 100.0 if n_eval > 0 else 0.0

    # Reduced Major Axis (RMA) Regression:
    # b_rma = sign(r) * (s_y / s_x), a_rma = y_bar - b_rma * x_bar
    x_vals = eval_df["original"].values
    y_vals = eval_df["duplicate"].values
    s_x = float(np.std(x_vals, ddof=1)) if n_eval > 1 else 0.0
    s_y = float(np.std(y_vals, ddof=1)) if n_eval > 1 else 0.0
    mean_x = float(np.mean(x_vals)) if n_eval > 0 else 0.0
    mean_y = float(np.mean(y_vals)) if n_eval > 0 else 0.0

    if n_eval > 1 and s_x > 1e-12 and s_y > 1e-12:
        r_corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
        b_rma = float(np.sign(r_corr) * (s_y / s_x))
        a_rma = float(mean_y - b_rma * mean_x)
    else:
        r_corr = 1.0
        b_rma = 1.0
        a_rma = 0.0

    verdict = "PASSED" if pct_meeting >= 90.0 else "ACTION REQUIRED: Precision Below 90% Target"

    res_df.attrs["pair_type"] = pair_type
    res_df.attrs["total_pairs"] = len(res_df)
    res_df.attrs["evaluated_pairs"] = n_eval
    res_df.attrs["hard_threshold_pct"] = hard_threshold
    res_df.attrs["pairs_meeting_threshold"] = n_meet
    res_df.attrs["pct_meeting_hard_threshold"] = pct_meeting
    res_df.attrs["pearson_correlation"] = r_corr
    res_df.attrs["rma_slope"] = b_rma
    res_df.attrs["rma_intercept"] = a_rma
    res_df.attrs["mean_original"] = mean_x
    res_df.attrs["mean_duplicate"] = mean_y
    res_df.attrs["std_original"] = s_x
    res_df.attrs["std_duplicate"] = s_y
    res_df.attrs["compliance_verdict"] = verdict
    res_df.attrs["footnotes"] = [
        f"{pair_type.capitalize()} duplicate precision evaluated under CIM MRMR Guidelines (§3.3) (TODO: Manually Verify).",
        f"Statutory precision threshold: >= 90.0% of pairs must achieve HARD <= {hard_threshold:.0f}% (TODO: Manually Verify).",
        "Linear trend modeled via Reduced Major Axis (RMA) regression to account for equal error variance (TODO: Manually Verify).",
        "Statutory compliance under NI 43-101 Form 43-101F1 Item 11 and JORC Code (2012, Table 1 Section 1) (TODO: Manually Verify).",
    ]

    return res_df


def plot_duplicate_analysis(
    duplicate_results: pd.DataFrame,
    grade_unit: str = "% Cu",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> Tuple[plt.Figure, Any]:
    """Generates dual-panel duplicate precision plots (Scatter + Cumulative HARD Curve).

    In accordance with NI 43-101 Form 43-101F1 Item 11 and JORC Code Table 1 (TODO: Manually Verify):
    - Left Panel: Original vs. Duplicate Scatter Plot with 1:1 line, RMA regression,
      and +/- 10% / 20% / 30% error envelopes.
    - Right Panel: Cumulative Frequency Plot of HARD values with the 90th percentile target line.

    Parameters
    ----------
    duplicate_results : pd.DataFrame
        DataFrame generated by analyze_assay_duplicates.
    grade_unit : str, default "% Cu"
        Grade display unit.
    title : str, optional
        Plot title.
    figsize : Tuple[float, float], default (14, 6)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Any]
        Matplotlib figure and pair of axes.
    """
    if not isinstance(duplicate_results, pd.DataFrame):
        raise TypeError(f"duplicate_results must be a DataFrame, got {type(duplicate_results)}")
    if len(duplicate_results) == 0:
        raise ValueError("duplicate_results DataFrame is empty")

    pair_type = duplicate_results.attrs.get("pair_type", "pulp")
    hard_thresh = float(duplicate_results.attrs.get("hard_threshold_pct", 10.0))
    pct_meet = float(duplicate_results.attrs.get("pct_meeting_hard_threshold", 90.0))
    rma_slope = float(duplicate_results.attrs.get("rma_slope", 1.0))
    rma_icept = float(duplicate_results.attrs.get("rma_intercept", 0.0))
    r_corr = float(duplicate_results.attrs.get("pearson_correlation", 1.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    orig = duplicate_results["original"].values
    dup = duplicate_results["duplicate"].values
    meets = duplicate_results["meets_hard_threshold"].values

    # Left Panel: Original vs Duplicate Scatter
    max_val = max(np.max(orig), np.max(dup)) * 1.08
    axis_line = np.linspace(0, max_val, 100)

    # 1:1 line
    ax1.plot(axis_line, axis_line, color="#212121", linestyle="--", linewidth=1.5, label="1:1 Equality", zorder=1)

    # Error envelope (+/- HARD threshold)
    env_upper = axis_line * (1.0 + 2.0 * (hard_thresh / 100.0) / (1.0 - (hard_thresh / 100.0)))
    env_lower = axis_line * max(0.0, 1.0 - 2.0 * (hard_thresh / 100.0) / (1.0 + (hard_thresh / 100.0)))
    ax1.plot(axis_line, env_upper, color="#ef6c00", linestyle=":", linewidth=1.2, label=f"+{hard_thresh:.0f}% HARD Envelope", zorder=1)
    ax1.plot(axis_line, env_lower, color="#ef6c00", linestyle=":", linewidth=1.2, label=f"-{hard_thresh:.0f}% HARD Envelope", zorder=1)
    ax1.fill_between(axis_line, env_lower, env_upper, color="#fff3e0", alpha=0.3, zorder=0)

    # RMA regression line
    rma_y = rma_slope * axis_line + rma_icept
    ax1.plot(axis_line, rma_y, color="#1565c0", linestyle="-", linewidth=2.0, label=f"RMA: y = {rma_slope:.3f}x + {rma_icept:.3f}", zorder=2)

    # Scatter points
    ax1.scatter(orig[meets], dup[meets], color="#2e7d32", alpha=0.6, s=35, zorder=3, label=f"Within HARD ({np.sum(meets)})")
    if np.sum(~meets) > 0:
        ax1.scatter(orig[~meets], dup[~meets], color="#c62828", alpha=0.7, s=45, marker="^", zorder=4, label=f"Exceeding HARD ({np.sum(~meets)})")

    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.set_xlabel(f"Original Assay ({grade_unit})", fontsize=11, fontweight="bold")
    ax1.set_ylabel(f"Duplicate Assay ({grade_unit})", fontsize=11, fontweight="bold")
    ax1.set_title(f"{pair_type.capitalize()} Duplicates: Original vs. Duplicate Scatter", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=8)

    # Right Panel: Cumulative Frequency of HARD (%)
    hard_vals = np.sort(duplicate_results["hard_pct"].values)
    cum_freq = np.linspace(0.0, 100.0, len(hard_vals))

    ax2.plot(hard_vals, cum_freq, color="#303f9f", linewidth=2.2, label="Empirical HARD Distribution", zorder=2)

    # 90% horizontal target line
    ax2.axhline(90.0, color="#c62828", linestyle="--", linewidth=1.8, label="90% Regulatory Threshold", zorder=1)
    # Target HARD vertical line
    ax2.axvline(hard_thresh, color="#ef6c00", linestyle="--", linewidth=1.8, label=f"{hard_thresh:.0f}% Target HARD", zorder=1)

    # Highlight intersection
    ax2.scatter([hard_thresh], [pct_meet], color="#c62828", s=90, zorder=5)
    ax2.annotate(
        f"  {pct_meet:.1f}% ≤ {hard_thresh:.0f}% HARD",
        xy=(hard_thresh, pct_meet),
        xytext=(hard_thresh + 2, pct_meet - 8),
        fontsize=9,
        fontweight="bold",
        color="#c62828",
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2),
        zorder=6,
    )

    x_max = max(50.0, float(np.percentile(hard_vals, 98)))
    ax2.set_xlim(0, x_max)
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Half Absolute Relative Difference - HARD (%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cumulative Frequency (%)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Cumulative Frequency of HARD ({pair_type.capitalize()} Precision)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=8)

    # Summary callout box on right panel
    callout_text = (
        f"{pair_type.capitalize()} Duplicate Precision\n"
        f"Total Pairs (N): {len(duplicate_results)}\n"
        f"Pearson Correlation (r): {r_corr:.3f}\n"
        f"RMA Slope: {rma_slope:.3f}\n"
        f"HARD Threshold: ≤ {hard_thresh:.0f}%\n"
        f"Pairs Meeting: {pct_meet:.1f}% (Target: ≥ 90%)\n"
        f"Verdict: {'PASSED' if pct_meet >= 90.0 else 'ACTION REQUIRED'}"
    )
    ax2.text(
        0.03,
        0.95,
        callout_text,
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#b0bec5", alpha=0.9),
        zorder=10,
    )

    sup_title = title or f"Drillhole Assay Duplicate Precision Assessment ({pair_type.capitalize()} Splits - NI 43-101 / JORC)"
    fig.suptitle(sup_title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig, (ax1, ax2)


def generate_qaqc_compliance_summary(
    total_drillhole_samples: int,
    crm_results: Optional[pd.DataFrame] = None,
    blanks_results: Optional[pd.DataFrame] = None,
    duplicate_results: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Generates the official QA/QC insertion rate and compliance summary table for NI 43-101 / JORC reports.

    Mandated by NI 43-101 Form 43-101F1 Item 11 and JORC Code Table 1 (Section 1) (TODO: Manually Verify),
    summarizes total control samples inserted, insertion rates (% of stream), pass rates,
    and failure counts:
    - Standards (target insertion rate: 5% of stream; target pass rate: >= 98%)
    - Blanks (target insertion rate: 5% of stream; target pass rate: >= 99%)
    - Duplicates (target insertion rate: 5% of stream; target pass rate: >= 90%)
    - Total QA/QC Control Samples (recommended total insertion rate: 15–20% of total assay stream).

    Parameters
    ----------
    total_drillhole_samples : int
        Total number of regular drillhole core/RC samples in the database (> 0).
    crm_results : pd.DataFrame, optional
        Audited CRM DataFrame from analyze_crm_standards.
    blanks_results : pd.DataFrame, optional
        Audited Blanks DataFrame from analyze_analytical_blanks.
    duplicate_results : pd.DataFrame, optional
        Audited Duplicates DataFrame from analyze_assay_duplicates.

    Returns
    -------
    pd.DataFrame
        Publication-ready NI 43-101 Chapter 11 compliance table reporting Sample Type,
        Count, Insertion Rate (%), Pass Count, Fail Count, and Pass Rate (%).
    """
    if total_drillhole_samples <= 0:
        raise ValueError(f"total_drillhole_samples must be strictly positive (> 0), got {total_drillhole_samples}")

    rows = []

    # 1. CRM Standards
    if crm_results is not None and len(crm_results) > 0:
        c_cnt = len(crm_results)
        c_ins = (c_cnt / total_drillhole_samples) * 100.0
        c_pass = int(crm_results.attrs.get("overall_pass_count", (crm_results["status"] == "Pass").sum()))
        c_fail = int(crm_results.attrs.get("overall_failure_count", c_cnt - c_pass))
        c_rate = (c_pass / c_cnt) * 100.0 if c_cnt > 0 else 0.0
        c_verdict = "PASSED" if (c_rate >= 98.0 and c_ins >= 5.0 and c_fail == 0) else "ACTION REQUIRED"
        rows.append({
            "sample_type": "Certified Reference Materials (CRMs)",
            "count": c_cnt,
            "insertion_rate_pct": c_ins,
            "target_insertion_rate_pct": 5.0,
            "insertion_status": "Adequate (≥ 5%)" if c_ins >= 5.0 else "Deficient (< 5%)",
            "pass_count": c_pass,
            "fail_count": c_fail,
            "pass_rate_pct": c_rate,
            "target_pass_rate_pct": 98.0,
            "compliance_verdict": c_verdict,
        })
    else:
        c_cnt = 0
        c_pass = 0
        c_fail = 0
        rows.append({
            "sample_type": "Certified Reference Materials (CRMs)",
            "count": 0,
            "insertion_rate_pct": 0.0,
            "target_insertion_rate_pct": 5.0,
            "insertion_status": "Deficient (< 5%)",
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate_pct": 0.0,
            "target_pass_rate_pct": 98.0,
            "compliance_verdict": "NOT PROVIDED",
        })

    # 2. Analytical Blanks
    if blanks_results is not None and len(blanks_results) > 0:
        b_cnt = len(blanks_results)
        b_ins = (b_cnt / total_drillhole_samples) * 100.0
        b_pass = int(blanks_results.attrs.get("pass_count", blanks_results["is_pass"].sum() if "is_pass" in blanks_results.columns else len(blanks_results)))
        b_fail = int(blanks_results.attrs.get("failure_count", b_cnt - b_pass))
        b_rate = (b_pass / b_cnt) * 100.0 if b_cnt > 0 else 0.0
        b_verdict = "PASSED" if (b_rate >= 99.0 and b_ins >= 5.0 and b_fail == 0) else "ACTION REQUIRED"
        rows.append({
            "sample_type": "Coarse & Pulp Blanks",
            "count": b_cnt,
            "insertion_rate_pct": b_ins,
            "target_insertion_rate_pct": 5.0,
            "insertion_status": "Adequate (≥ 5%)" if b_ins >= 5.0 else "Deficient (< 5%)",
            "pass_count": b_pass,
            "fail_count": b_fail,
            "pass_rate_pct": b_rate,
            "target_pass_rate_pct": 99.0,
            "compliance_verdict": b_verdict,
        })
    else:
        b_cnt = 0
        b_pass = 0
        b_fail = 0
        rows.append({
            "sample_type": "Coarse & Pulp Blanks",
            "count": 0,
            "insertion_rate_pct": 0.0,
            "target_insertion_rate_pct": 5.0,
            "insertion_status": "Deficient (< 5%)",
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate_pct": 0.0,
            "target_pass_rate_pct": 99.0,
            "compliance_verdict": "NOT PROVIDED",
        })

    # 3. Duplicate Assays
    if duplicate_results is not None and len(duplicate_results) > 0:
        d_cnt = len(duplicate_results)
        d_ins = (d_cnt / total_drillhole_samples) * 100.0
        pair_type = duplicate_results.attrs.get("pair_type", "pulp")
        d_pass = int(duplicate_results.attrs.get("pairs_meeting_threshold", duplicate_results["meets_hard_threshold"].sum()))
        d_fail = d_cnt - d_pass
        d_rate = (d_pass / d_cnt) * 100.0 if d_cnt > 0 else 0.0
        d_verdict = "PASSED" if (d_rate >= 90.0 and d_ins >= 5.0) else "ACTION REQUIRED"
        rows.append({
            "sample_type": f"Duplicate Assays ({pair_type.capitalize()})",
            "count": d_cnt,
            "insertion_rate_pct": d_ins,
            "target_insertion_rate_pct": 5.0,
            "insertion_status": "Adequate (≥ 5%)" if d_ins >= 5.0 else "Deficient (< 5%)",
            "pass_count": d_pass,
            "fail_count": d_fail,
            "pass_rate_pct": d_rate,
            "target_pass_rate_pct": 90.0,
            "compliance_verdict": d_verdict,
        })
    else:
        d_cnt = 0
        d_pass = 0
        d_fail = 0
        rows.append({
            "sample_type": "Duplicate Assays",
            "count": 0,
            "insertion_rate_pct": 0.0,
            "target_insertion_rate_pct": 5.0,
            "insertion_status": "Deficient (< 5%)",
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate_pct": 0.0,
            "target_pass_rate_pct": 90.0,
            "compliance_verdict": "NOT PROVIDED",
        })

    # 4. Total QA/QC Control
    tot_cnt = c_cnt + b_cnt + d_cnt
    tot_ins = (tot_cnt / total_drillhole_samples) * 100.0
    tot_pass = c_pass + b_pass + d_pass
    tot_fail = c_fail + b_fail + d_fail
    tot_rate = (tot_pass / tot_cnt) * 100.0 if tot_cnt > 0 else 0.0
    tot_verdict = (
        "PASSED"
        if (tot_ins >= 15.0 and tot_rate >= 95.0 and all(r["compliance_verdict"] == "PASSED" for r in rows))
        else ("ACTION REQUIRED" if tot_cnt > 0 else "NO QA/QC DATA")
    )

    rows.append({
        "sample_type": "Total QA/QC Control Insertion",
        "count": tot_cnt,
        "insertion_rate_pct": tot_ins,
        "target_insertion_rate_pct": 15.0,
        "insertion_status": "Adequate (≥ 15%)" if tot_ins >= 15.0 else "Deficient (< 15%)",
        "pass_count": tot_pass,
        "fail_count": tot_fail,
        "pass_rate_pct": tot_rate,
        "target_pass_rate_pct": 95.0,
        "compliance_verdict": tot_verdict,
    })

    res_df = pd.DataFrame(rows)

    res_df.attrs["total_drillhole_samples"] = total_drillhole_samples
    res_df.attrs["total_qaqc_samples"] = tot_cnt
    res_df.attrs["total_stream_samples"] = total_drillhole_samples + tot_cnt
    res_df.attrs["overall_insertion_rate_pct"] = tot_ins
    res_df.attrs["overall_compliance_verdict"] = tot_verdict
    res_df.attrs["footnotes"] = [
        "QA/QC control stream compiled in accordance with NI 43-101 Form 43-101F1 Item 11 (TODO: Manually Verify).",
        "Compliance verified against CIM MRMR Best Practice Guidelines (2019, Section 3 on Data Verification) (TODO: Manually Verify).",
        "Standard industry insertion rate guidelines: >= 5% CRMs, >= 5% Blanks, >= 5% Duplicates; Total QA/QC >= 15% (TODO: Manually Verify).",
        "All data verified suitable for Mineral Resource estimation under JORC Code (2012, Table 1 Section 1) (TODO: Manually Verify).",
    ]

    return res_df


# =============================================================================
# PART 3: SIMULATION SPATIAL VALIDATION & UNCERTAINTY (SME CH. 4.6, ROSSI CH. 12)
# =============================================================================


def check_simulation_variogram_reproduction(
    realizations: np.ndarray,
    grid_points: np.ndarray,
    variogram_model: str,
    sill: float,
    range_param: Union[float, Sequence[float], Mapping[str, float]],
    nugget: float = 0.0,
    lags: Optional[Sequence[float]] = None,
    n_lags: int = 15,
    max_lag: Optional[float] = None,
    azimuth: float = 0.0,
    dip: float = 0.0,
    tolerance_angle: float = 22.5,
) -> pd.DataFrame:
    """Verifies that conditional simulation realizations reproduce the input variogram model.

    In accordance with SME Mining Engineering Handbook (3rd Ed., Section 4.6 on Conditional
    Simulation) (TODO: Manually Verify) and Rossi & Deutsch (2014, Chapter 12 on Simulation
    Validation) (TODO: Manually Verify), unlike Kriging (which produces an artificially smoothed,
    variance-reduced estimate), conditional simulation must reproduce the full spatial variance
    across all separation distances. Evaluates experimental semivariograms across all realizations
    l in 1, ..., L along principal continuity directions to verify reproduction within expected
    Monte Carlo ergodic fluctuation bounds.

    Parameters
    ----------
    realizations : np.ndarray
        Array of shape (M, n_realizations) with simulated grid or block values.
    grid_points : np.ndarray
        Array of shape (M, 2) or (M, 3) with target grid coordinates.
    variogram_model : str
        Analytical model type ('spherical', 'exponential', 'gaussian', 'cubic').
    sill : float
        Variogram partial sill variance C (> 0).
    range_param : float, Sequence[float], or Mapping
        Variogram practical correlation range a (> 0).
    nugget : float, default 0.0
        Variogram nugget effect C0 (>= 0).
    lags : Sequence[float], optional
        Explicit lag distance centers.
    n_lags : int, default 15
        Number of lag intervals if lags is None (> 0).
    max_lag : float, optional
        Maximum lag distance. If None, defaults to 1.2 * range_param.
    azimuth : float, default 0.0
        Directional azimuth in degrees clockwise from North.
    dip : float, default 0.0
        Directional dip in degrees.
    tolerance_angle : float, default 22.5
        Angular half-window tolerance in degrees. If >= 90.0, omnidirectional.

    Returns
    -------
    pd.DataFrame
        Variogram reproduction table containing:
        - lag : Separation distance h
        - model_gamma : Theoretical model semivariance gamma(h)
        - mean_realization_gamma : Ensemble mean semivariance across realizations
        - p10_gamma, p50_gamma, p90_gamma : Percentile envelopes of realization semivariances
        - individual realization semivariance columns 'realization_1', ..., 'realization_L'.
        Attached in `.attrs`:
        - 'n_realizations' : L
        - 'ergodic_fluctuation_pct' : Mean relative % difference between ensemble mean and model.
        - 'reproduction_verdict' : 'VALIDATED' (if ergodic fluctuation <= 20%) or 'INVESTIGATION REQUIRED'.
        - 'footnotes' : list of compliance citations with (TODO: Manually Verify).
    """
    reals = np.asarray(realizations, dtype=float)
    if reals.ndim == 1:
        reals = reals[:, np.newaxis]
    elif reals.ndim != 2:
        raise ValueError(f"realizations must be a 1D or 2D array, got shape {reals.shape}")

    pts = np.asarray(grid_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        raise ValueError(f"grid_points must have shape (M, 2) or (M, 3), got shape {pts.shape}")

    m_pts, n_reals = reals.shape
    if len(pts) != m_pts:
        raise ValueError(f"Length of grid_points ({len(pts)}) does not match realizations ({m_pts})")

    if sill <= 0:
        raise ValueError(f"sill must be strictly positive (> 0), got {sill}")
    if nugget < 0:
        raise ValueError(f"nugget must be non-negative (>= 0), got {nugget}")
    if n_lags < 2:
        raise ValueError(f"n_lags must be at least 2, got {n_lags}")

    # Extract scalar range representation for default lag bounds
    if isinstance(range_param, (dict, Mapping)):
        r_val = float(range_param.get("major", list(range_param.values())[0]))
    elif isinstance(range_param, (list, tuple, np.ndarray)):
        r_val = float(range_param[0])
    else:
        r_val = float(range_param)

    if r_val <= 0:
        raise ValueError(f"range_param must be strictly positive (> 0), got {r_val}")

    if max_lag is None:
        max_lag = float(r_val * 1.2)
    elif max_lag <= 0:
        raise ValueError(f"max_lag must be strictly positive (> 0), got {max_lag}")

    if lags is not None:
        lag_centers = np.sort(np.asarray(lags, dtype=float))
        n_bins = len(lag_centers)
        step = float(lag_centers[1] - lag_centers[0]) if n_bins > 1 else float(lag_centers[0])
        half_step = step / 2.0
    else:
        n_bins = int(n_lags)
        step = float(max_lag / n_bins)
        half_step = step / 2.0
        lag_centers = np.linspace(step, max_lag, n_bins)

    # Subsample if grid is large to preserve fast sub-second performance
    max_sub = 800
    if m_pts > max_sub:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(m_pts, size=max_sub, replace=False)
        sub_pts = pts[sub_idx]
        sub_reals = reals[sub_idx, :]
    else:
        sub_pts = pts
        sub_reals = reals

    n_sub = len(sub_pts)

    # Compute coordinate differences and Euclidean distances
    diff_coords = sub_pts[:, np.newaxis, :] - sub_pts[np.newaxis, :, :]  # (N, N, dim)
    dists = np.linalg.norm(diff_coords, axis=-1)  # (N, N)

    # Directional filtering
    if tolerance_angle < 90.0:
        rad_az = np.radians(azimuth)
        rad_dip = np.radians(dip)
        if sub_pts.shape[1] == 2:
            dir_vec = np.array([np.sin(rad_az) * np.cos(rad_dip), np.cos(rad_az) * np.cos(rad_dip)])
        else:
            dir_vec = np.array([
                np.sin(rad_az) * np.cos(rad_dip),
                np.cos(rad_az) * np.cos(rad_dip),
                -np.sin(rad_dip),
            ])
        norm_dir = np.linalg.norm(dir_vec)
        if norm_dir > 1e-9:
            dir_vec = dir_vec / norm_dir

        dot_p = np.abs(np.sum(diff_coords * dir_vec, axis=-1))
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_a = np.where(dists > 1e-9, dot_p / dists, 1.0)
        angle_mask = cos_a >= np.cos(np.radians(tolerance_angle))
    else:
        angle_mask = np.ones((n_sub, n_sub), dtype=bool)

    tri_mask = np.triu(np.ones((n_sub, n_sub), dtype=bool), k=1) & angle_mask
    pair_dists = dists[tri_mask]
    pair_i, pair_j = np.where(tri_mask)

    gamma_reals = np.full((n_bins, n_reals), np.nan, dtype=float)
    pair_counts = np.zeros(n_bins, dtype=int)

    # Compute experimental semivariance across realizations
    if len(pair_dists) > 0:
        for k, h_k in enumerate(lag_centers):
            in_bin = (pair_dists >= h_k - half_step) & (pair_dists < h_k + half_step)
            pair_counts[k] = int(np.sum(in_bin))
            if pair_counts[k] > 0:
                for l in range(n_reals):
                    z_diff = sub_reals[pair_i[in_bin], l] - sub_reals[pair_j[in_bin], l]
                    gamma_reals[k, l] = 0.5 * float(np.mean(z_diff ** 2))

    # Compute theoretical model semivariance
    total_sill = float(nugget + sill)
    cov_vals = _theoretical_covariance(
        lag_centers,
        model=variogram_model,
        nugget=nugget,
        sill=sill,
        range_param=range_param,
    )
    model_gamma = np.maximum(0.0, total_sill - cov_vals)

    # Realization ensemble metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_real_gamma = np.nanmean(gamma_reals, axis=1)
        p10_gamma = np.nanpercentile(gamma_reals, 10, axis=1)
        p50_gamma = np.nanpercentile(gamma_reals, 50, axis=1)
        p90_gamma = np.nanpercentile(gamma_reals, 90, axis=1)

    records = {
        "lag": lag_centers,
        "model_gamma": model_gamma,
        "mean_realization_gamma": mean_real_gamma,
        "p10_gamma": p10_gamma,
        "p50_gamma": p50_gamma,
        "p90_gamma": p90_gamma,
        "pair_count": pair_counts,
    }
    for l in range(n_reals):
        records[f"realization_{l + 1}"] = gamma_reals[:, l]

    res_df = pd.DataFrame(records)

    # Quantify ergodic fluctuation within practical range
    valid_lags = (lag_centers <= r_val) & np.isfinite(mean_real_gamma) & (model_gamma > 1e-6)
    if np.any(valid_lags):
        ergodic_fluc = float(
            np.mean(np.abs(mean_real_gamma[valid_lags] - model_gamma[valid_lags]) / model_gamma[valid_lags]) * 100.0
        )
    else:
        ergodic_fluc = 0.0

    verdict = (
        "VALIDATED (Accurate Variogram Reproduction)"
        if ergodic_fluc <= 20.0
        else "INVESTIGATION REQUIRED (Ergodic Fluctuation / Divergence Detected)"
    )

    res_df.attrs["n_realizations"] = n_reals
    res_df.attrs["variogram_model"] = variogram_model
    res_df.attrs["sill"] = float(sill)
    res_df.attrs["nugget"] = float(nugget)
    res_df.attrs["total_sill"] = total_sill
    res_df.attrs["range_param"] = r_val
    res_df.attrs["ergodic_fluctuation_pct"] = ergodic_fluc
    res_df.attrs["reproduction_verdict"] = verdict
    res_df.attrs["footnotes"] = [
        "Conditional simulation spatial reproduction evaluated under SME Mining Engineering Handbook (§4.6) (TODO: Manually Verify).",
        "Spatial variability verification complies with Rossi & Deutsch (2014, Chapter 12) (TODO: Manually Verify).",
        "Unlike smoothed Kriging estimates, conditional simulation realizations must reproduce the input semivariance model (TODO: Manually Verify).",
        "Discrepancies within +/- 15–20% reflect acceptable Monte Carlo ergodic fluctuations in finite domains (TODO: Manually Verify).",
    ]

    return res_df


def plot_simulation_variogram_reproduction(
    variogram_reproduction_results: pd.DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 6),
) -> Tuple[plt.Figure, Any]:
    """Renders the variogram reproduction validation chart for conditional simulations.

    Displays:
    - Theoretical input variogram model curve (thick solid dark red line).
    - Individual realization semivariograms (thin semi-transparent blue lines).
    - Realization ensemble mean curve (dashed navy line) and P10-P90 uncertainty band.

    Parameters
    ----------
    variogram_reproduction_results : pd.DataFrame
        DataFrame produced by check_simulation_variogram_reproduction.
    title : str, optional
        Plot title.
    figsize : Tuple[float, float], default (12, 6)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Any]
        Matplotlib figure and axes.
    """
    if not isinstance(variogram_reproduction_results, pd.DataFrame):
        raise TypeError(
            f"variogram_reproduction_results must be a DataFrame, got {type(variogram_reproduction_results)}"
        )
    if "lag" not in variogram_reproduction_results.columns:
        raise ValueError("variogram_reproduction_results must contain 'lag' column")

    df = variogram_reproduction_results
    fig, ax = plt.subplots(figsize=figsize)

    lags = df["lag"].values
    model_g = df["model_gamma"].values
    mean_g = df["mean_realization_gamma"].values
    p10_g = df["p10_gamma"].values
    p90_g = df["p90_gamma"].values

    # Plot individual realization curves
    real_cols = [c for c in df.columns if c.startswith("realization_")]
    first_real = True
    for c in real_cols:
        lbl = f"Realizations (N={len(real_cols)})" if first_real else None
        ax.plot(lags, df[c].values, color="#90caf9", alpha=0.35, linewidth=1.0, label=lbl, zorder=2)
        first_real = False

    # Shaded P10-P90 uncertainty envelope
    ax.fill_between(lags, p10_g, p90_g, color="#bbdefb", alpha=0.45, label="P10–P90 Simulation Envelope", zorder=3)

    # Realization ensemble mean curve
    ax.plot(lags, mean_g, color="#0d47a1", linestyle="--", linewidth=2.0, label="Simulation Ensemble Mean", zorder=4)

    # Theoretical model curve
    ax.plot(lags, model_g, color="#b71c1c", linestyle="-", linewidth=2.5, label="Theoretical Variogram Model γ(h)", zorder=5)

    # Total Sill line
    total_sill = float(df.attrs.get("total_sill", np.nanmax(model_g)))
    ax.axhline(total_sill, color="#424242", linestyle=":", linewidth=1.5, label=f"Total Sill ($C_0 + C$) = {total_sill:.3f}", zorder=1)

    # Practical Range vertical line
    r_val = float(df.attrs.get("range_param", np.nan))
    if np.isfinite(r_val) and r_val > 0:
        ax.axvline(r_val, color="#ef6c00", linestyle="--", linewidth=1.2, label=f"Practical Range $a$ = {r_val:.1f} m", zorder=1)

    # Summary callout box
    fluc = float(df.attrs.get("ergodic_fluctuation_pct", 0.0))
    verdict = df.attrs.get("reproduction_verdict", "N/A")
    v_model = df.attrs.get("variogram_model", "spherical")

    callout_text = (
        f"Simulation Spatial Validation\n"
        f"Realizations: {len(real_cols)}\n"
        f"Variogram Model: {v_model.capitalize()}\n"
        f"Total Sill: {total_sill:.3f}\n"
        f"Practical Range: {r_val:.1f} m\n"
        f"Ergodic Fluctuation: {fluc:.1f}%\n"
        f"Status: {verdict}"
    )
    ax.text(
        0.02,
        0.95,
        callout_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#b0bec5", alpha=0.9),
        zorder=10,
    )

    ax.set_xlabel("Separation Distance $h$ (m)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Semivariance $\\gamma(h)$", fontsize=11, fontweight="bold")
    chart_title = title or "Conditional Simulation Variogram Reproduction & Ergodic Fluctuation (SME §4.6)"
    ax.set_title(chart_title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, float(np.max(lags) * 1.02))
    ax.set_ylim(0, max(total_sill * 1.25, float(np.nanmax(p90_g) * 1.15) if len(p90_g) > 0 else total_sill * 1.25))
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)

    plt.tight_layout()
    return fig, ax


def compute_simulation_grade_tonnage_envelopes(
    block_realizations: np.ndarray,
    block_model: pd.DataFrame,
    cutoff_grades: Sequence[float],
    tonnes_col: str = "tonnes",
    metal_factor: float = 0.01,
) -> pd.DataFrame:
    """Evaluates geological risk envelopes (P10, P50, P90) across simulated realizations.

    In accordance with SME Mining Engineering Handbook (§4.6) (TODO: Manually Verify) and
    Rossi & Deutsch (2014, Chapter 12) (TODO: Manually Verify), evaluating Grade-Tonnage curves
    across all L simulated realizations reveals the true non-linear dispersion variance and provides
    defensible confidence bounds:
    - P10: Optimistic / high case (90th percentile of realization distribution).
    - P50: Median / expected case (50th percentile).
    - P90: Conservative / low case (10th percentile of realization distribution).

    Parameters
    ----------
    block_realizations : np.ndarray
        Array of shape (n_blocks, n_realizations) with simulated block grades.
    block_model : pd.DataFrame
        Block model DataFrame containing block tonnages.
    cutoff_grades : Sequence[float]
        Sequence of economic cut-off grades to evaluate.
    tonnes_col : str, default "tonnes"
        Column containing block tonnage.
    metal_factor : float, default 0.01
        Multiplier to calculate contained metal (0.01 for %, 1e-6 for g/t).

    Returns
    -------
    pd.DataFrame
        Risk envelope table containing for each cut-off grade:
        - cutoff_grade : float
        - tonnage_p10, tonnage_p50, tonnage_p90, tonnage_mean : float
        - grade_p10, grade_p50, grade_p90, grade_mean : float
        - metal_p10, metal_p50, metal_p90, metal_mean : float
        Attached in `.attrs`:
        - 'n_realizations' : L
        - 'n_blocks' : N
        - 'footnotes' : list of compliance citations with (TODO: Manually Verify).
    """
    reals = np.asarray(block_realizations, dtype=float)
    if reals.ndim == 1:
        reals = reals[:, np.newaxis]
    elif reals.ndim != 2:
        raise ValueError(f"block_realizations must be a 1D or 2D array, got shape {reals.shape}")

    if not isinstance(block_model, pd.DataFrame):
        raise TypeError(f"block_model must be a pandas DataFrame, got {type(block_model)}")
    if tonnes_col not in block_model.columns:
        raise ValueError(f"tonnes_col '{tonnes_col}' not found in block_model")

    n_blocks, n_reals = reals.shape
    if len(block_model) != n_blocks:
        raise ValueError(f"block_model rows ({len(block_model)}) must match block_realizations rows ({n_blocks})")

    if not isinstance(cutoff_grades, (list, tuple, np.ndarray, Sequence)) or len(cutoff_grades) == 0:
        raise ValueError("cutoff_grades must be a non-empty sequence of cut-off values")

    tonnes = block_model[tonnes_col].values.astype(float)
    c_grades = np.sort(np.asarray(cutoff_grades, dtype=float))
    if np.any(c_grades < 0):
        raise ValueError("Cut-off grades must be non-negative (>= 0)")

    rows = []
    for cog in c_grades:
        t_reals = np.zeros(n_reals, dtype=float)
        g_reals = np.zeros(n_reals, dtype=float)
        m_reals = np.zeros(n_reals, dtype=float)

        for l in range(n_reals):
            mask = reals[:, l] >= cog
            ore_t = float(np.sum(tonnes[mask]))
            if ore_t > 0:
                ore_g = float(np.sum(tonnes[mask] * reals[mask, l]) / ore_t)
                ore_m = float(ore_t * ore_g * metal_factor)
            else:
                ore_g = 0.0
                ore_m = 0.0
            t_reals[l] = ore_t
            g_reals[l] = ore_g
            m_reals[l] = ore_m

        rows.append({
            "cutoff_grade": cog,
            # In mining risk conventions: P10 is optimistic (high), P90 is conservative (low)
            "tonnage_p10": float(np.percentile(t_reals, 90)),
            "tonnage_p50": float(np.percentile(t_reals, 50)),
            "tonnage_p90": float(np.percentile(t_reals, 10)),
            "tonnage_mean": float(np.mean(t_reals)),
            "grade_p10": float(np.percentile(g_reals, 90)),
            "grade_p50": float(np.percentile(g_reals, 50)),
            "grade_p90": float(np.percentile(g_reals, 10)),
            "grade_mean": float(np.mean(g_reals)),
            "metal_p10": float(np.percentile(m_reals, 90)),
            "metal_p50": float(np.percentile(m_reals, 50)),
            "metal_p90": float(np.percentile(m_reals, 10)),
            "metal_mean": float(np.mean(m_reals)),
        })

    res_df = pd.DataFrame(rows)

    res_df.attrs["n_realizations"] = n_reals
    res_df.attrs["n_blocks"] = n_blocks
    res_df.attrs["tonnes_col"] = tonnes_col
    res_df.attrs["metal_factor"] = metal_factor
    res_df.attrs["footnotes"] = [
        "Grade-Tonnage risk envelopes evaluated across conditional simulation realizations (SME §4.6) (TODO: Manually Verify).",
        "P10 represents high case (90th percentile), P50 median case, and P90 conservative low case (10th percentile) (TODO: Manually Verify).",
        "Simulation grade-tonnage distributions quantify non-linear recoverable resource uncertainty (Rossi & Deutsch 2014) (TODO: Manually Verify).",
    ]

    return res_df


def plot_simulation_grade_tonnage_envelopes(
    envelopes_results: pd.DataFrame,
    grade_unit: str = "% Cu",
    tonnage_unit: str = "Mt",
    metal_unit: str = "kt",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 6),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Renders dual-panel Grade-Tonnage uncertainty envelope charts (Tonnage & Grade).

    Parameters
    ----------
    envelopes_results : pd.DataFrame
        DataFrame produced by compute_simulation_grade_tonnage_envelopes.
    grade_unit : str, default "% Cu"
        Grade display unit.
    tonnage_unit : str, default "Mt"
        Tonnage scale unit.
    metal_unit : str, default "kt"
        Contained metal scale unit.
    title : str, optional
        Figure title.
    figsize : Tuple[float, float], default (14, 6)
        Matplotlib figure dimensions.

    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
        Matplotlib figure and pair of axes (tonnage_ax, grade_ax).
    """
    if not isinstance(envelopes_results, pd.DataFrame):
        raise TypeError(f"envelopes_results must be a DataFrame, got {type(envelopes_results)}")
    if "cutoff_grade" not in envelopes_results.columns:
        raise ValueError("envelopes_results must contain 'cutoff_grade' column")

    df = envelopes_results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    cogs = df["cutoff_grade"].values

    # Determine automatic scaling factor for tonnages if expressed in raw tonnes
    raw_max_t = df["tonnage_p10"].max()
    t_scale = 1e-6 if raw_max_t > 1e5 and tonnage_unit == "Mt" else (1e-3 if raw_max_t > 1e4 and tonnage_unit == "kt" else 1.0)

    t_p10 = df["tonnage_p10"].values * t_scale
    t_p50 = df["tonnage_p50"].values * t_scale
    t_p90 = df["tonnage_p90"].values * t_scale
    t_mean = df["tonnage_mean"].values * t_scale

    g_p10 = df["grade_p10"].values
    g_p50 = df["grade_p50"].values
    g_p90 = df["grade_p90"].values
    g_mean = df["grade_mean"].values

    # Left Panel: Ore Tonnage Risk Envelope vs Cut-off Grade
    ax1.plot(cogs, t_p50, color="#1565c0", linestyle="-", linewidth=2.2, label="P50 Expected Tonnage", zorder=3)
    ax1.plot(cogs, t_p10, color="#0d47a1", linestyle="--", linewidth=1.4, label="P10 High / P90 Low Case", zorder=2)
    ax1.plot(cogs, t_p90, color="#0d47a1", linestyle="--", linewidth=1.4, zorder=2)
    ax1.fill_between(cogs, t_p90, t_p10, color="#bbdefb", alpha=0.5, label="P10–P90 Uncertainty Envelope", zorder=1)
    ax1.plot(cogs, t_mean, color="#757575", linestyle=":", linewidth=1.5, label="Mean Tonnage", zorder=2)

    ax1.set_xlabel(f"Cut-off Grade ({grade_unit})", fontsize=11, fontweight="bold")
    ax1.set_ylabel(f"Ore Tonnage ({tonnage_unit})", fontsize=11, fontweight="bold")
    ax1.set_title("Ore Tonnage Risk Envelope vs. Cut-off", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=8)

    # Right Panel: Average Ore Grade Risk Envelope vs Cut-off Grade
    ax2.plot(cogs, g_p50, color="#2e7d32", linestyle="-", linewidth=2.2, label="P50 Expected Grade", zorder=3)
    ax2.plot(cogs, g_p10, color="#1b5e20", linestyle="--", linewidth=1.4, label="P10 High / P90 Low Case", zorder=2)
    ax2.plot(cogs, g_p90, color="#1b5e20", linestyle="--", linewidth=1.4, zorder=2)
    ax2.fill_between(cogs, g_p90, g_p10, color="#c8e6c9", alpha=0.5, label="P10–P90 Uncertainty Envelope", zorder=1)
    ax2.plot(cogs, g_mean, color="#757575", linestyle=":", linewidth=1.5, label="Mean Grade", zorder=2)

    ax2.set_xlabel(f"Cut-off Grade ({grade_unit})", fontsize=11, fontweight="bold")
    ax2.set_ylabel(f"Average Ore Grade ({grade_unit})", fontsize=11, fontweight="bold")
    ax2.set_title("Average Ore Grade Risk Envelope vs. Cut-off", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=8)

    sup_title = title or "Conditional Simulation Grade–Tonnage Risk Envelopes (SME §4.6 / Rossi & Deutsch Ch. 12)"
    fig.suptitle(sup_title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig, (ax1, ax2)


def drill_spacing_analysis_by_simulation(
    block_realizations: np.ndarray,
    block_model: pd.DataFrame,
    production_volume_m3: float,
    confidence_level: float = 0.90,
    target_precision_pct: float = 15.0,
    tonnes_col: str = "tonnes",
) -> pd.DataFrame:
    """Calibrates CIM / JORC drillhole spacing criteria using conditional simulation.

    In accordance with CIM MRMR Guidelines (§6.11) (TODO: Manually Verify) and industry
    standard practices (Verly, Postolski & Parker 2014; Rossi & Deutsch 2014) (TODO: Manually Verify):
    - Measured Mineral Resource: Quarterly production volume known to within +/- 15%
      at a 90% confidence level (or annual volume with higher precision).
    - Indicated Mineral Resource: Annual production volume known to within +/- 15%
      at a 90% confidence level.
    Aggregates simulated blocks into contiguous mining production parcels matching the production
    volume to measure the relative estimation error as a function of drill spacing.

    Parameters
    ----------
    block_realizations : np.ndarray
        Array of shape (n_blocks, n_realizations) containing simulated grades.
    block_model : pd.DataFrame
        Block model with coordinates and tonnages.
    production_volume_m3 : float
        Volume of the production period (e.g. quarterly or annual production parcel) (> 0).
    confidence_level : float, default 0.90
        Target statistical confidence level (standard is 0.90 for 90%).
    target_precision_pct : float, default 15.0
        Target relative error percentage (standard is +/- 15%).
    tonnes_col : str, default "tonnes"
        Tonnage column name.

    Returns
    -------
    pd.DataFrame
        Drill spacing calibration table with parcel_id, block_count, parcel_tonnes,
        mean_grade, std_grade, p_lower, p_upper, relative_error_pct, status,
        and classification_qualification ('Measured Candidate', 'Indicated Candidate', etc.).
    """
    reals = np.asarray(block_realizations, dtype=float)
    if reals.ndim == 1:
        reals = reals[:, np.newaxis]
    elif reals.ndim != 2:
        raise ValueError(f"block_realizations must be a 1D or 2D array, got shape {reals.shape}")

    if not isinstance(block_model, pd.DataFrame):
        raise TypeError(f"block_model must be a pandas DataFrame, got {type(block_model)}")
    if tonnes_col not in block_model.columns:
        raise ValueError(f"tonnes_col '{tonnes_col}' not found in block_model")

    n_blocks, n_reals = reals.shape
    if len(block_model) != n_blocks:
        raise ValueError(f"block_model rows ({len(block_model)}) must match block_realizations rows ({n_blocks})")

    if production_volume_m3 <= 0:
        raise ValueError(f"production_volume_m3 must be strictly positive (> 0), got {production_volume_m3}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be between 0 and 1, got {confidence_level}")
    if target_precision_pct <= 0:
        raise ValueError(f"target_precision_pct must be strictly positive (> 0), got {target_precision_pct}")

    tonnes = block_model[tonnes_col].values.astype(float)

    # Estimate block volume in m3 (assuming standard 2.7 t/m3 if no volume column provided)
    if "block_volume" in block_model.columns:
        b_vol = block_model["block_volume"].values.astype(float)
    elif "density" in block_model.columns and np.all(block_model["density"] > 0):
        b_vol = tonnes / block_model["density"].values.astype(float)
    else:
        b_vol = tonnes / 2.70

    mean_b_vol = float(np.mean(b_vol)) if len(b_vol) > 0 else 1.0
    blocks_per_parcel = max(1, int(round(production_volume_m3 / max(1e-3, mean_b_vol))))

    # Spatial sorting if coordinates present
    coord_cols = [c for c in ["z", "y", "x"] if c in block_model.columns]
    if len(coord_cols) > 0:
        sort_order = block_model.sort_values(by=coord_cols).index.values
    else:
        sort_order = np.arange(n_blocks)

    sorted_reals = reals[sort_order, :]
    sorted_tonnes = tonnes[sort_order]
    sorted_vols = b_vol[sort_order]

    # Partition blocks into contiguous parcels
    alpha = 1.0 - confidence_level
    p_low_pct = 100.0 * (alpha / 2.0)
    p_high_pct = 100.0 * (1.0 - alpha / 2.0)

    records = []
    n_parcels = int(np.ceil(n_blocks / blocks_per_parcel))

    for p_idx in range(n_parcels):
        start_i = p_idx * blocks_per_parcel
        end_i = min(n_blocks, (p_idx + 1) * blocks_per_parcel)
        if start_i >= end_i:
            continue

        p_t = sorted_tonnes[start_i:end_i]
        p_v = sorted_vols[start_i:end_i]
        tot_p_t = float(np.sum(p_t))
        tot_p_v = float(np.sum(p_v))

        # Realization parcel grades
        if tot_p_t > 0:
            p_grades_reals = np.sum(sorted_reals[start_i:end_i, :] * p_t[:, np.newaxis], axis=0) / tot_p_t
        else:
            p_grades_reals = np.mean(sorted_reals[start_i:end_i, :], axis=0)

        mean_g = float(np.mean(p_grades_reals))
        std_g = float(np.std(p_grades_reals, ddof=1)) if n_reals > 1 else 0.0
        q_low = float(np.percentile(p_grades_reals, p_low_pct))
        q_high = float(np.percentile(p_grades_reals, p_high_pct))

        # Relative error % = (Q_high - Q_low) / (2 * Mean) * 100
        denom = max(1e-6, mean_g)
        rel_err = float(((q_high - q_low) / (2.0 * denom)) * 100.0)

        qualifies = rel_err <= target_precision_pct
        status = f"Qualifies (≤ {target_precision_pct:.0f}%)" if qualifies else f"Exceeds Limit (> {target_precision_pct:.0f}%)"

        # Classification candidate
        if qualifies:
            qualification = "Measured Candidate" if production_volume_m3 <= 50000.0 else "Indicated Candidate"
        else:
            qualification = "Inferred / Requires Infill Drilling"

        records.append({
            "parcel_id": f"Parcel_{p_idx + 1:02d}",
            "block_count": end_i - start_i,
            "parcel_volume_m3": tot_p_v,
            "parcel_tonnes": tot_p_t,
            "mean_grade": mean_g,
            "std_grade": std_g,
            "p_lower": q_low,
            "p_upper": q_high,
            "relative_error_pct": rel_err,
            "status": status,
            "classification_qualification": qualification,
            "meets_target_precision": qualifies,
        })

    res_df = pd.DataFrame(records)

    tot_p = len(res_df)
    n_qual = int(res_df["meets_target_precision"].sum()) if tot_p > 0 else 0
    qual_rate = (n_qual / tot_p) * 100.0 if tot_p > 0 else 0.0
    avg_rel_err = float(res_df["relative_error_pct"].mean()) if tot_p > 0 else 0.0

    overall_verdict = (
        f"CONFIRMED: Meets ±{target_precision_pct:.0f}% Precision at {confidence_level*100:.0f}% Confidence"
        if qual_rate >= 85.0
        else f"UNCONFIRMED: Requires Tighter Spacing ({qual_rate:.1f}% parcels meet target)"
    )

    res_df.attrs["total_parcels"] = tot_p
    res_df.attrs["parcels_qualifying"] = n_qual
    res_df.attrs["qualification_rate_pct"] = qual_rate
    res_df.attrs["mean_relative_error_pct"] = avg_rel_err
    res_df.attrs["production_volume_m3"] = production_volume_m3
    res_df.attrs["confidence_level"] = confidence_level
    res_df.attrs["target_precision_pct"] = target_precision_pct
    res_df.attrs["overall_classification_verdict"] = overall_verdict
    res_df.attrs["footnotes"] = [
        "Drillhole spacing calibration conducted using conditional simulation (CIM MRMR §6.11) (TODO: Manually Verify).",
        "Measured resource criterion: quarterly production known to +/- 15% at 90% confidence (Verly et al. 2014) (TODO: Manually Verify).",
        "Indicated resource criterion: annual production known to +/- 15% at 90% confidence (Rossi & Deutsch 2014) (TODO: Manually Verify).",
    ]

    return res_df


# =============================================================================
# PART 4: POLYMETALLIC EQUIVALENCY & NET SMELTER RETURN (CIM MRMR §7.2, JORC CL 50)
# =============================================================================


def calculate_net_smelter_return(
    block_model: pd.DataFrame,
    commodities_config: Mapping[str, Mapping[str, float]],
    royalty_pct: float = 0.0,
    offsite_costs_per_tonne: float = 0.0,
    currency: str = "$",
) -> pd.DataFrame:
    """Calculates block-by-block Net Smelter Return (NSR in $/t) for polymetallic deposits.

    In accordance with CIM MRMR Best Practice Guidelines (§7.2) (TODO: Manually Verify) and SME Mining
    Engineering Handbook (Ch. 6.1; Rossi & Deutsch 2014) (TODO: Manually Verify), in multi-element
    deposits (e.g., Cu-Au-Ag, Pb-Zn-Ag), economic cut-off evaluation cannot be conducted on a single
    metal grade. The Net Smelter Return (NSR, $/t) represents the net revenue realized at the mine gate
    after accounting for:
    - Base metal market price.
    - Concentrator / metallurgical recovery.
    - Smelter payable metal percentage.
    - Smelter Treatment Charges (TCs) and Refining Charges (RCs).
    - Penalty deductions for deleterious elements.
    - Concentrate freight, moisture, and offsite realization charges.
    - Gross and net smelter royalties (NSR royalties).

    Formula per Commodity i:
    ------------------------
    Revenue_i ($/t) = Grade_i * Unit_Conversion_i * Recovery_i * Payable_i * (Price_i - RC_i) - TC_i - Penalties_i
    Gross_NSR ($/t) = Sum_i Revenue_i
    NSR ($/t) = max(0, Gross_NSR - offsite_costs) * (1 - Royalty / 100)

    Parameters
    ----------
    block_model : pd.DataFrame
        Block model DataFrame containing estimated grade columns for all commodities.
    commodities_config : Mapping[str, Mapping[str, float]]
        Nested dictionary configuring each commodity:
        {
            'Cu': {
                'grade_col': 'grade_cu', 'price': 3.80, 'unit_conversion': 22.0462,  # $/lb from %
                'recovery': 0.88, 'payability': 0.95, 'refining_charge': 0.08, 'treatment_charge_per_metal_unit': 0.05
            },
            'Au': {
                'grade_col': 'grade_au', 'price': 1900.0, 'unit_conversion': 0.0321507,  # $/oz from g/t
                'recovery': 0.75, 'payability': 0.90, 'refining_charge': 5.0, 'treatment_charge_per_metal_unit': 0.0
            }
        }
    royalty_pct : float, default 0.0
        Overall Net Smelter Return royalty percentage (e.g., 2.0 for 2% NSR royalty).
    offsite_costs_per_tonne : float, default 0.0
        General off-site transportation, insurance, and representation charges per tonne ore.
    currency : str, default "$"
        Currency symbol for output column labeling.

    Returns
    -------
    pd.DataFrame
        Copy of block_model with added columns:
        - 'nsr_gross' : Gross realized revenue before royalties ($/t).
        - 'nsr' : Final Net Smelter Return ($/t) used for economic cut-off application.
        - Individual revenue contribution columns per commodity ('nsr_cu', 'nsr_au', etc.).
    """
    if not isinstance(block_model, pd.DataFrame):
        raise TypeError(f"block_model must be a pandas DataFrame, got {type(block_model)}")
    if len(block_model) == 0:
        raise ValueError("block_model cannot be empty")
    if not isinstance(commodities_config, (dict, Mapping)) or len(commodities_config) == 0:
        raise ValueError("commodities_config must be a non-empty mapping of commodity configurations")

    if not (0.0 <= royalty_pct < 100.0):
        raise ValueError(f"royalty_pct must be between 0.0 and 100.0, got {royalty_pct}")
    if offsite_costs_per_tonne < 0.0:
        raise ValueError(f"offsite_costs_per_tonne must be non-negative (>= 0), got {offsite_costs_per_tonne}")

    out_df = block_model.copy()
    gross_nsr = np.zeros(len(out_df), dtype=float)
    contributions = {}

    for comm_name, cfg in commodities_config.items():
        if not isinstance(cfg, (dict, Mapping)):
            raise TypeError(f"Configuration for commodity '{comm_name}' must be a dictionary or mapping")

        grade_col = cfg.get("grade_col")
        if not grade_col or grade_col not in out_df.columns:
            raise ValueError(f"grade_col '{grade_col}' for commodity '{comm_name}' not found in block_model")

        price = float(cfg.get("price", 0.0))
        if price <= 0.0:
            raise ValueError(f"price for commodity '{comm_name}' must be strictly positive (> 0), got {price}")

        recovery = float(cfg.get("recovery", 1.0))
        if recovery <= 0.0:
            raise ValueError(f"recovery for commodity '{comm_name}' must be strictly positive (> 0), got {recovery}")
        if recovery > 1.0:
            recovery = recovery / 100.0

        payability = float(cfg.get("payability", 1.0))
        if payability <= 0.0:
            raise ValueError(f"payability for commodity '{comm_name}' must be strictly positive (> 0), got {payability}")
        if payability > 1.0:
            payability = payability / 100.0

        unit_conv = float(cfg.get("unit_conversion", 1.0))
        if unit_conv <= 0.0:
            raise ValueError(f"unit_conversion for commodity '{comm_name}' must be strictly positive (> 0), got {unit_conv}")

        rc = float(cfg.get("refining_charge", 0.0))
        if rc < 0.0:
            raise ValueError(f"refining_charge for commodity '{comm_name}' must be non-negative (>= 0), got {rc}")

        tc = float(cfg.get("treatment_charge_per_metal_unit", cfg.get("treatment_charge", 0.0)))
        if tc < 0.0:
            raise ValueError(f"treatment_charge for commodity '{comm_name}' must be non-negative (>= 0), got {tc}")

        penalties = float(cfg.get("penalties_per_tonne", 0.0))
        if penalties < 0.0:
            raise ValueError(f"penalties_per_tonne for commodity '{comm_name}' must be non-negative (>= 0), got {penalties}")

        # Vectorized revenue per tonne ore for this commodity
        grades = out_df[grade_col].values.astype(float)
        net_price = max(0.0, price - rc)
        payable_factor = recovery * payability
        metal_units_per_t = grades * unit_conv
        payable_metal = metal_units_per_t * payable_factor

        comm_gross = payable_metal * net_price - tc * payable_metal - penalties
        comm_gross = np.maximum(0.0, comm_gross)

        comm_col = f"nsr_{comm_name.lower()}"
        out_df[comm_col] = comm_gross
        gross_nsr += comm_gross
        contributions[comm_name] = float(np.mean(comm_gross))

    out_df["nsr_gross"] = gross_nsr
    royalty_factor = 1.0 - (royalty_pct / 100.0)
    net_nsr = np.maximum(0.0, (gross_nsr - offsite_costs_per_tonne) * royalty_factor)
    out_df["nsr"] = net_nsr

    mean_gross = float(np.mean(gross_nsr))
    mean_net = float(np.mean(net_nsr))

    denom_gross = max(1e-6, mean_gross)
    shares = {c: float((val / denom_gross) * 100.0) for c, val in contributions.items()}

    out_df.attrs["commodities"] = list(commodities_config.keys())
    out_df.attrs["royalty_pct"] = royalty_pct
    out_df.attrs["offsite_costs_per_tonne"] = offsite_costs_per_tonne
    out_df.attrs["currency"] = currency
    out_df.attrs["mean_gross_nsr"] = mean_gross
    out_df.attrs["mean_nsr"] = mean_net
    out_df.attrs["commodity_contributions"] = contributions
    out_df.attrs["commodity_percentage_shares"] = shares
    out_df.attrs["footnotes"] = [
        "Net Smelter Return (NSR) calculated under CIM MRMR Best Practice Guidelines (§7.2) for polymetallic deposits (TODO: Manually Verify).",
        "Smelter deductions include Treatment Charges (TC), Refining Charges (RC), and payable metal factors (SME Handbook Ch. 6.1) (TODO: Manually Verify).",
        "Royalty deductions applied on mine-gate realized net revenue (Rossi & Deutsch 2014) (TODO: Manually Verify).",
    ]

    return out_df


def calculate_equivalent_grade(
    block_model: pd.DataFrame,
    primary_commodity: str,
    primary_grade_col: str,
    primary_price: float,
    primary_recovery: float,
    secondary_commodities: Mapping[str, Mapping[str, float]],
    output_col: Optional[str] = None,
    primary_unit_conversion: float = 1.0,
) -> pd.DataFrame:
    """Calculates multi-metal equivalent grade in compliance with JORC Clause 50 and CIM MRMR §7.2.

    Mandatory Regulatory Compliance (JORC Code 2012 Clause 50 & CIM MRMR Guidelines §7.2):
    -------------------------------------------------------------------------------------
    Under international reporting codes, reporting metal equivalents (e.g., CuEq or AuEq)
    requires strict disclosure of:
    1. Assumed commodity prices and metallurgical recoveries for ALL included metals.
    2. Demonstration of reasonable prospect for eventual economic extraction (RPEEE) for each element.
    3. Formula explicitly stated in report:
       EqGrade = Grade_primary + Sum_j [ Grade_j * (Price_j * Rec_j * Conv_j) / (Price_prim * Rec_prim * Conv_prim) ]

    Parameters
    ----------
    block_model : pd.DataFrame
        Block model DataFrame containing primary and secondary grade columns.
    primary_commodity : str
        Primary metal symbol (e.g., 'Cu' for CuEq or 'Au' for AuEq).
    primary_grade_col : str
        Column name of the primary commodity grade in block_model.
    primary_price : float
        Market price per unit of the primary commodity (> 0).
    primary_recovery : float
        Metallurgical recovery fraction of the primary commodity (0.0 to 1.0).
    secondary_commodities : Mapping[str, Mapping[str, float]]
        Dictionary of secondary commodities with grade column, price, recovery, and conversion factor:
        {
            'Au': {'grade_col': 'grade_au', 'price': 1900.0, 'recovery': 0.75, 'conversion_factor': 0.03215 / 22.0462}
        }
    output_col : str, optional
        Name of the resulting equivalent grade column (defaults to f"{primary_commodity.lower()}_eq").
    primary_unit_conversion : float, default 1.0
        Conversion factor for primary metal grade to price unit per tonne ore.

    Returns
    -------
    pd.DataFrame
        Copy of block_model with equivalent grade column populated.
        Attached in `.attrs`:
        - 'formula_string': Textual disclosure formula required by JORC Clause 50.
        - 'primary_commodity', 'secondary_commodities', 'prices', 'recoveries'.
    """
    if not isinstance(block_model, pd.DataFrame):
        raise TypeError(f"block_model must be a pandas DataFrame, got {type(block_model)}")
    if len(block_model) == 0:
        raise ValueError("block_model cannot be empty")
    if primary_grade_col not in block_model.columns:
        raise ValueError(f"primary_grade_col '{primary_grade_col}' not found in block_model")

    if primary_price <= 0.0:
        raise ValueError(f"primary_price must be strictly positive (> 0), got {primary_price}")
    if primary_recovery <= 0.0:
        raise ValueError(f"primary_recovery must be strictly positive (> 0), got {primary_recovery}")
    if primary_recovery > 1.0:
        primary_recovery = primary_recovery / 100.0

    if primary_unit_conversion <= 0.0:
        raise ValueError(f"primary_unit_conversion must be strictly positive (> 0), got {primary_unit_conversion}")

    if not isinstance(secondary_commodities, (dict, Mapping)) or len(secondary_commodities) == 0:
        raise ValueError("secondary_commodities must be a non-empty mapping of secondary commodity configurations")

    out_df = block_model.copy()
    prim_grades = out_df[primary_grade_col].values.astype(float)
    eq_grade = np.copy(prim_grades)

    denom = primary_price * primary_recovery * primary_unit_conversion
    eq_factors = {}
    formula_terms = [primary_commodity]

    for sec_name, cfg in secondary_commodities.items():
        if not isinstance(cfg, (dict, Mapping)):
            raise TypeError(f"Configuration for secondary commodity '{sec_name}' must be a dictionary or mapping")

        g_col = cfg.get("grade_col")
        if not g_col or g_col not in out_df.columns:
            raise ValueError(f"grade_col '{g_col}' for secondary commodity '{sec_name}' not found in block_model")

        sec_p = float(cfg.get("price", 0.0))
        if sec_p <= 0.0:
            raise ValueError(f"price for secondary commodity '{sec_name}' must be strictly positive (> 0), got {sec_p}")

        sec_rec = float(cfg.get("recovery", 1.0))
        if sec_rec <= 0.0:
            raise ValueError(f"recovery for secondary commodity '{sec_name}' must be strictly positive (> 0), got {sec_rec}")
        if sec_rec > 1.0:
            sec_rec = sec_rec / 100.0

        sec_conv = float(cfg.get("conversion_factor", cfg.get("unit_conversion", 1.0)))
        if sec_conv <= 0.0:
            raise ValueError(f"conversion_factor for secondary commodity '{sec_name}' must be strictly positive (> 0), got {sec_conv}")

        num = sec_p * sec_rec * sec_conv
        factor = float(num / denom)
        eq_factors[sec_name] = factor

        sec_grades = out_df[g_col].values.astype(float)
        eq_grade += sec_grades * factor

        formula_terms.append(f"[{sec_name} * ({sec_p:.2f} * {sec_rec:.4f} * {sec_conv:.4g}) / ({primary_price:.2f} * {primary_recovery:.4f} * {primary_unit_conversion:.4g})]")

    final_out_col = output_col if output_col is not None else f"{primary_commodity.lower()}_eq"
    out_df[final_out_col] = eq_grade

    formula_str = f"{primary_commodity}Eq = " + " + ".join(formula_terms)

    out_df.attrs["primary_commodity"] = primary_commodity
    out_df.attrs["primary_price"] = primary_price
    out_df.attrs["primary_recovery"] = primary_recovery
    out_df.attrs["primary_unit_conversion"] = primary_unit_conversion
    out_df.attrs["secondary_commodities"] = list(secondary_commodities.keys())
    out_df.attrs["equivalency_factors"] = eq_factors
    out_df.attrs["formula_string"] = formula_str
    out_df.attrs["output_col"] = final_out_col
    out_df.attrs["mean_primary_grade"] = float(np.mean(prim_grades))
    out_df.attrs["mean_equivalent_grade"] = float(np.mean(eq_grade))
    out_df.attrs["footnotes"] = [
        "Metal equivalent reporting complies with JORC Code (2012) Clause 50 disclosure requirements (TODO: Manually Verify).",
        "Equivalency factors incorporate individual metallurgical recoveries and metal market prices under CIM MRMR §7.2 (TODO: Manually Verify).",
        "Assumes reasonable prospects for eventual economic extraction (RPEEE) for all constituent metals (TODO: Manually Verify).",
    ]

    return out_df


def plot_net_smelter_return_breakdown(
    nsr_results: pd.DataFrame,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (13, 5),
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """Renders dual-panel NSR breakdown: revenue contributions and block model distribution.

    Parameters
    ----------
    nsr_results : pd.DataFrame
        DataFrame produced by calculate_net_smelter_return.
    title : str, optional
        Figure title.
    figsize : Tuple[float, float], default (13, 5)
        Figure size in inches.

    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]
        Matplotlib figure and axes pair (waterfall/bar ax, distribution ax).
    """
    if not isinstance(nsr_results, pd.DataFrame):
        raise TypeError(f"nsr_results must be a pandas DataFrame, got {type(nsr_results)}")
    if "nsr" not in nsr_results.columns or "nsr_gross" not in nsr_results.columns:
        raise ValueError("nsr_results must contain 'nsr' and 'nsr_gross' columns produced by calculate_net_smelter_return")

    commodities = nsr_results.attrs.get("commodities", [])
    currency = nsr_results.attrs.get("currency", "$")
    mean_nsr = float(nsr_results["nsr"].mean())
    p50_nsr = float(nsr_results["nsr"].median())
    mean_gross = float(nsr_results["nsr_gross"].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Revenue contributions by commodity and net NSR
    comm_cols = [f"nsr_{c.lower()}" for c in commodities if f"nsr_{c.lower()}" in nsr_results.columns]
    bar_labels = [c.upper() for c in commodities if f"nsr_{c.lower()}" in nsr_results.columns]
    bar_vals = [float(nsr_results[col].mean()) for col in comm_cols]

    # Add Gross NSR and Net NSR bars
    bar_labels.extend(["Gross NSR", "Net NSR"])
    bar_vals.extend([mean_gross, mean_nsr])

    colors = ["#1976d2", "#ffb300", "#7cb342", "#8e24aa"][:len(comm_cols)] + ["#546e7a", "#2e7d32"]

    bars = ax1.bar(bar_labels, bar_vals, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(
            f"{currency}{h:.2f}/t",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_ylabel(f"Average Value ({currency}/t)", fontsize=11, fontweight="bold")
    ax1.set_title("Average Revenue Breakdown by Metal & Net NSR", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Panel 2: Block Model NSR Distribution
    nsr_vals = nsr_results["nsr"].values
    ax2.hist(nsr_vals, bins=25, color="#1976d2", edgecolor="black", alpha=0.7, density=True, label="Block Frequency")
    ax2.axvline(mean_nsr, color="#d32f2f", linestyle="--", linewidth=2.0, label=f"Mean NSR: {currency}{mean_nsr:.2f}/t")
    ax2.axvline(p50_nsr, color="#2e7d32", linestyle="-.", linewidth=2.0, label=f"Median NSR: {currency}{p50_nsr:.2f}/t")

    ax2.set_xlabel(f"Net Smelter Return ({currency}/t)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax2.set_title(f"Block Model NSR Distribution ({currency}/t)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9)

    sup_title = title or "Polymetallic Net Smelter Return (NSR) Valuation (CIM §7.2 / SME Ch. 6.1)"
    fig.suptitle(sup_title, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig, (ax1, ax2)

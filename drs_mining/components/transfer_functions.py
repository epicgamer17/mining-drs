"""Non-linear transfer functions and volumetric uncertainty for conditional simulations.

Provides vectorized transfer functions that evaluate point-support conditional simulation
ensembles over mining cutoffs and recoverable resource volumes
(Journel 1983; Rossi & Deutsch 2014; Ortiz 2019) (TODO: Manually Verify).

Adheres to CONTRIBUTING.md standards:
- Functional design with standard NumPy arrays and pandas DataFrames.
- Strict input validation and fast failure with informative error messages.
- Vectorized execution across realizations without Python loop overhead.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def simulation_grade_tonnage_uncertainty(
    realizations: np.ndarray,
    cutoffs: Sequence[float],
    tonnes: np.ndarray,
    percentiles: Sequence[float] = (10.0, 50.0, 90.0),
    grade_unit: str = "% Cu",
) -> pd.DataFrame:
    """Evaluates Grade-Tonnage curves across all conditional realizations simultaneously.

    Applies the non-linear cutoff recovery transfer function directly to each realization
    to evaluate the distribution of recoverable ore tonnage, average grade, and contained metal
    (Journel 1983; Rossi & Deutsch 2014) (TODO: Manually Verify).

    Parameters
    ----------
    realizations : np.ndarray
        Array of shape (M, L) representing L simulated realizations across M spatial points/blocks.
    cutoffs : Sequence[float]
        List of cut-off grades to evaluate.
    tonnes : np.ndarray
        Array of block/point tonnages of shape (M,).
    percentiles : Sequence[float], default (10.0, 50.0, 90.0)
        Percentiles to evaluate across realizations.
    grade_unit : str, default "% Cu"
        Grade unit label.

    Returns
    -------
    pd.DataFrame
        Table indexed by cutoff with expected values, standard deviations, and percentiles
        for ore tonnage, average ore grade, and metal tonnes.
        Attributes (.attrs) contain a list of individual `grade_tonnage_table` DataFrames
        for every realization.
    """
    real_arr = np.asarray(realizations, dtype=float)
    if real_arr.ndim != 2:
        raise ValueError(f"realizations must be a 2D array, got shape {real_arr.shape}")
    n_points, n_real = real_arr.shape
    if n_points == 0 or n_real == 0:
        raise ValueError("realizations array cannot be empty.")

    t_arr = np.asarray(tonnes, dtype=float)
    if t_arr.shape != (n_points,):
        raise ValueError(f"tonnes must have shape ({n_points},), got {t_arr.shape}")
    if np.any(t_arr < 0):
        raise ValueError("tonnes cannot contain negative values.")

    cutoff_list = sorted([float(c) for c in cutoffs])
    if len(cutoff_list) == 0:
        raise ValueError("cutoffs list cannot be empty.")
    if cutoff_list[0] < 0:
        raise ValueError("cutoffs cannot be negative.")

    pct_list = [float(p) for p in percentiles]
    for p in pct_list:
        if not (0.0 < p < 100.0):
            raise ValueError(f"Percentile must be strictly between 0 and 100, got {p}")

    records = []
    realization_tables: list[pd.DataFrame] = [
        pd.DataFrame(index=cutoff_list, columns=["ore_tonnes", "ore_grade", "metal_tonnes"], dtype=float)
        for _ in range(n_real)
    ]

    for c in cutoff_list:
        # Mask shape: (M, L)
        mask = real_arr >= c
        ore_tonnes_real = np.sum(t_arr[:, None] * mask, axis=0)  # Shape: (L,)
        metal_tonnes_real = np.sum(t_arr[:, None] * (mask * real_arr) * 0.01, axis=0)  # Shape: (L,)

        # Average grade above cutoff
        ore_grades_real = np.where(
            ore_tonnes_real > 0.0,
            (metal_tonnes_real / (ore_tonnes_real * 0.01)),
            0.0,
        )

        for l_idx in range(n_real):
            realization_tables[l_idx].loc[c, "ore_tonnes"] = float(ore_tonnes_real[l_idx])
            realization_tables[l_idx].loc[c, "ore_grade"] = float(ore_grades_real[l_idx])
            realization_tables[l_idx].loc[c, "metal_tonnes"] = float(metal_tonnes_real[l_idx])

        rec: dict[str, Any] = {
            "cutoff": c,
            "expected_ore_tonnes": float(np.mean(ore_tonnes_real)),
            "ore_tonnes_std": float(np.std(ore_tonnes_real, ddof=1)) if n_real > 1 else 0.0,
        }
        for p in pct_list:
            t_label = f"ore_tonnes_p{int(p) if p.is_integer() else p}"
            rec[t_label] = float(np.percentile(ore_tonnes_real, p))

        rec["expected_grade"] = float(np.mean(ore_grades_real))
        rec["grade_std"] = float(np.std(ore_grades_real, ddof=1)) if n_real > 1 else 0.0
        for p in pct_list:
            g_label = f"grade_p{int(p) if p.is_integer() else p}"
            rec[g_label] = float(np.percentile(ore_grades_real, p))

        rec["expected_metal"] = float(np.mean(metal_tonnes_real))
        for p in pct_list:
            m_label = f"metal_p{int(p) if p.is_integer() else p}"
            rec[m_label] = float(np.percentile(metal_tonnes_real, p))

        records.append(rec)

    df_out = pd.DataFrame(records).set_index("cutoff")
    df_out.attrs["n_realizations"] = n_real
    df_out.attrs["grade_unit"] = grade_unit
    df_out.attrs["realization_tables"] = realization_tables
    return df_out

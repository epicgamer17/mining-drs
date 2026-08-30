"""Analytical Operational Face-Allocation Equations (Appendix A & B, Slides 29-30).

Provides mathematical blending solutions to allocate mass extraction rates
between High-Ore1 Face (Face 1) and Balanced Face (Face 2) based on plant target rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FaceAllocationResult:
    """Analytical face allocation weights and mass rates."""

    face1_rate: float  # t/day from Face 1 (High-Ore1)
    face2_rate: float  # t/day from Face 2 (Balanced)
    face1_weight: float  # Fraction of fleet / extraction allocated to Face 1
    face2_weight: float  # Fraction of fleet / extraction allocated to Face 2
    is_feasible: bool  # True if exact target blend is mathematically achievable


def solve_face_allocation_rates(
    target_ore1_rate: float,
    target_ore2_rate: float,
    face1_ore1_fraction: float = 0.70,
    face2_ore1_fraction: float = 0.65,
) -> FaceAllocationResult:
    """Analytically solves for Face 1 and Face 2 extraction rates and weights.

    Given:
      f1 = face1_ore1_fraction (e.g. 0.70 Ore 1, 0.30 Ore 2)
      f2 = face2_ore1_fraction (e.g. 0.65 Ore 1, 0.35 Ore 2)
      R1 = target_ore1_rate
      R2 = target_ore2_rate
      R_tot = R1 + R2

    The system of linear equations:
      r1 + r2 = R_tot
      r1 * f1 + r2 * f2 = R1

    Yields the closed-form analytical solution (Slide 29):
      r1 = (R1 - R_tot * f2) / (f1 - f2)
      r2 = R_tot - r1

    Returns:
      FaceAllocationResult containing (r1, r2, w1, w2, is_feasible).
    """
    r_tot = max(0.0, float(target_ore1_rate) + float(target_ore2_rate))
    if r_tot <= 1e-12:
        return FaceAllocationResult(
            face1_rate=0.0,
            face2_rate=0.0,
            face1_weight=0.50,
            face2_weight=0.50,
            is_feasible=True,
        )

    f1 = max(0.0, min(1.0, float(face1_ore1_fraction)))
    f2 = max(0.0, min(1.0, float(face2_ore1_fraction)))
    denom = f1 - f2

    if abs(denom) <= 1e-12:
        # Singular case (identical face grades)
        half = r_tot / 2.0
        return FaceAllocationResult(
            face1_rate=half,
            face2_rate=half,
            face1_weight=0.50,
            face2_weight=0.50,
            is_feasible=abs(r_tot * f1 - float(target_ore1_rate)) < 1e-6,
        )

    # Analytical solution
    r1_raw = (float(target_ore1_rate) - r_tot * f2) / denom
    r2_raw = r_tot - r1_raw

    # Check feasibility within physical bounds [0, r_tot]
    is_feasible = (-1e-6 <= r1_raw <= r_tot + 1e-6) and (-1e-6 <= r2_raw <= r_tot + 1e-6)

    # Clamp to valid physical operating bounds
    r1 = max(0.0, min(r_tot, r1_raw))
    r2 = max(0.0, min(r_tot, r_tot - r1))

    w1 = r1 / r_tot if r_tot > 0 else 0.50
    w2 = r2 / r_tot if r_tot > 0 else 0.50

    return FaceAllocationResult(
        face1_rate=r1,
        face2_rate=r2,
        face1_weight=w1,
        face2_weight=w2,
        is_feasible=is_feasible,
    )

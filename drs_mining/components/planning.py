"""Strategic and tactical planning components for mining operations."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Sequence


class MiningPriority(Enum):
    """Monthly mine-level priority selected by the tactical controller."""

    PRODUCTION = auto()
    BALANCED = auto()
    DEVELOPMENT = auto()


@dataclass(frozen=True)
class AreaReadinessTarget:
    """Long-horizon development requirement for a future mining area.

    required_development is measured from the start of strategic planning.
    ready_by_day is also measured from the start of strategic planning. A value
    of None means readiness is tracked but has no schedule deadline.
    """

    required_development: float = 0.0
    ready_by_day: Optional[float] = None


@dataclass(frozen=True)
class StrategicYearTarget:
    """Long-horizon commitments passed from strategic planning to tactical control.

    Defaults are zero so the simulation behaviour remains baseline until
    explicit strategic targets are supplied.
    """

    min_development: float = 0.0
    min_ore1_production: float = 0.0
    min_ore2_production: float = 0.0


def strategic_target_for_year(
    targets: Sequence[StrategicYearTarget], year_index: int
) -> StrategicYearTarget:
    """Return the target for a strategic year.

    If the simulation runs longer than the supplied target list, reuse the last
    target rather than silently dropping the strategic constraints.
    """
    if not targets:
        return StrategicYearTarget()
    idx = max(0, min(int(year_index), len(targets) - 1))
    return targets[idx]


def trajectory_progress_ratio(
    actual: float,
    annual_target: float,
    elapsed_fraction: float,
    eps: float = 1e-12,
) -> float:
    """Actual progress divided by the progress expected by this point in the year.

    A target <= 0 is treated as inactive and returns 1.0.
    """
    if annual_target <= eps:
        return 1.0
    expected = annual_target * max(0.0, min(1.0, elapsed_fraction))
    if expected <= eps:
        return 1.0
    return max(0.0, actual) / expected


def select_mining_priority(
    development_ratio: float,
    ore1_ratio: float,
    ore2_ratio: float,
    tolerance: float = 0.90,
    area2_readiness_trajectory_ratio: float = 1.0,
) -> MiningPriority:
    """Rule-based monthly tactical review; no optimization is performed.

    - If all strategic trajectories are within tolerance, remain BALANCED.
    - Otherwise prioritize the most delayed commitment.
    - Ore 1 and Ore 2 deficits both map to PRODUCTION priority.
    - Annual development or Area 2 readiness schedule deficits map to DEVELOPMENT.

    The selected priority can be used by tactical controllers to adjust
    development allocation or reserve production resources.
    """
    tolerance = max(0.0, min(1.0, tolerance))
    development_schedule_ratio = min(
        development_ratio,
        area2_readiness_trajectory_ratio,
    )
    deficits = {
        MiningPriority.DEVELOPMENT: max(0.0, 1.0 - development_schedule_ratio),
        MiningPriority.PRODUCTION: max(
            0.0, 1.0 - min(ore1_ratio, ore2_ratio)
        ),
    }

    largest_priority, largest_deficit = max(deficits.items(), key=lambda item: item[1])
    if largest_deficit <= (1.0 - tolerance):
        return MiningPriority.BALANCED
    return largest_priority

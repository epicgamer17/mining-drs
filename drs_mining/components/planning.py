from dataclasses import dataclass
from typing import Optional, Sequence
import drs
from drs_mining.components.modes import OperatingMode
from drs_mining.config import FLEET_MODES


@dataclass(frozen=True)
class AreaReadinessTarget:
    """Development requirement for a future mining area.

    required_development is measured from the start of planning.
    ready_by_day is also measured from the start of planning. A value
    of None means readiness is tracked but has no schedule deadline.
    """

    required_development: float = 0.0
    ready_by_day: Optional[float] = None


def select_fleet_mode(
    development_ratio: float,
    ore1_ratio: float,
    ore2_ratio: float,
    tolerance: float = 0.90,
    area2_readiness_trajectory_ratio: float = 1.0,
) -> OperatingMode:
    """Rule-based tactical review selecting the active Fleet OperatingMode.

    - If all trajectories are within tolerance, remain BALANCED.
    - Otherwise prioritize the most delayed commitment.
    - Ore 1 and Ore 2 deficits both map to PRODUCTION mode.
    - Development or Area 2 readiness deficits map to DEVELOPMENT mode.
    """
    tolerance = max(0.0, min(1.0, tolerance))
    development_schedule_ratio = min(
        development_ratio,
        area2_readiness_trajectory_ratio,
    )
    deficits = {
        FLEET_MODES["DEVELOPMENT"]: max(0.0, 1.0 - development_schedule_ratio),
        FLEET_MODES["PRODUCTION"]: max(
            0.0, 1.0 - min(ore1_ratio, ore2_ratio)
        ),
    }

    largest_mode, largest_deficit = max(deficits.items(), key=lambda item: item[1])
    if largest_deficit <= (1.0 - tolerance):
        return FLEET_MODES["PRODUCTION"]
    return largest_mode


class TacticalReviewController(drs.Module):
    """Periodic tactical review that selects fleet operating mode.

    Manages a review timer and calls select_fleet_mode at each interval.
    """

    def __init__(
        self,
        tactical_review_period_days: float = 30.0,
        tactical_progress_tolerance: float = 0.90,
        initial_fleet_mode: Optional[OperatingMode] = None,
    ):
        super().__init__()
        self.tactical_review_period_days = tactical_review_period_days
        self.tactical_progress_tolerance = tactical_progress_tolerance

        self.planning_started = False
        self.tactical_review_timer = drs.Timer(
            "tactical_review_timer", initial_value=0.0, rate=1.0
        )
        self.tactical_review_count = drs.Level(
            "tactical_review_count", initial_value=0.0
        )

        self.fleet_mode: OperatingMode = (
            initial_fleet_mode or FLEET_MODES["PRODUCTION"]
        )

    def start_planning(self):
        """Activates tactical review cycle."""
        if not self.planning_started:
            self.planning_started = True
            self.tactical_review_timer.reset()
            self.tactical_review_count.value = 0.0

    def step_timers(self, dt_days: float):
        """Advances internal timers."""
        if self.planning_started:
            self.tactical_review_timer.step(dt_days)

    def update_mode(
        self,
        area2_readiness_trajectory_ratio: float = 1.0,
        force_mode: Optional[OperatingMode] = None,
    ) -> OperatingMode:
        """Check if a tactical review is due and update fleet mode."""
        if not self.planning_started:
            return self.fleet_mode

        if (
            self.tactical_review_timer.value
            >= self.tactical_review_period_days - 1e-6
            or self.tactical_review_count.value == 0.0
        ):
            self.tactical_review_timer.reset()
            self.tactical_review_count.value += 1.0

            if force_mode is not None:
                self.fleet_mode = force_mode
            else:
                self.fleet_mode = select_fleet_mode(
                    development_ratio=1.0,
                    ore1_ratio=1.0,
                    ore2_ratio=1.0,
                    tolerance=self.tactical_progress_tolerance,
                    area2_readiness_trajectory_ratio=area2_readiness_trajectory_ratio,
                )

        return self.fleet_mode

    @property
    def active_fleet_mode(self) -> OperatingMode:
        return self.fleet_mode

    @active_fleet_mode.setter
    def active_fleet_mode(self, value: OperatingMode) -> None:
        self.fleet_mode = value

    def levels(self) -> Sequence[drs.Level]:
        return (self.tactical_review_timer, self.tactical_review_count)

    def time_to_event(self) -> float:
        min_dt = float("inf")
        for lvl in self.levels():
            dt = lvl.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt

    def step(self, dt: float) -> None:
        for lvl in self.levels():
            lvl.step(dt)

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Any, Union, Dict
import drs
from drs_mining.components.modes import OperatingMode
from drs_mining.config import FLEET_MODES


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


def select_fleet_mode(
    development_ratio: float,
    ore1_ratio: float,
    ore2_ratio: float,
    tolerance: float = 0.90,
    area2_readiness_trajectory_ratio: float = 1.0,
) -> OperatingMode:
    """Rule-based monthly tactical review selecting the active Fleet OperatingMode.

    - If all strategic trajectories are within tolerance, remain BALANCED.
    - Otherwise prioritize the most delayed commitment.
    - Ore 1 and Ore 2 deficits both map to PRODUCTION mode.
    - Annual development or Area 2 readiness schedule deficits map to DEVELOPMENT mode.
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
    """Supervisory controller managing annual strategic cycles, progress tracking,
    and monthly tactical reviews to dynamically set fleet operating modes.
    """

    def __init__(
        self,
        strategic_targets: Optional[Sequence[StrategicYearTarget]] = None,
        strategic_period_days: float = 365.0,
        tactical_review_period_days: float = 30.0,
        tactical_progress_tolerance: float = 0.90,
        development_priority_truck_reservation_fraction: float = 0.20,
        initial_fleet_mode: Optional[OperatingMode] = None,
    ):
        super().__init__()
        self.strategic_targets = list(strategic_targets or [])
        self.strategic_period_days = strategic_period_days
        self.tactical_review_period_days = tactical_review_period_days
        self.tactical_progress_tolerance = tactical_progress_tolerance
        self.development_priority_truck_reservation_fraction = (
            development_priority_truck_reservation_fraction
        )

        self.planning_started = False
        self.strategic_year_index = drs.Level(
            "strategic_year_index", initial_value=0.0
        )
        self.strategic_year_timer = drs.Timer(
            "strategic_year_timer", initial_value=0.0, rate=1.0
        )
        self.tactical_review_timer = drs.Timer(
            "tactical_review_timer", initial_value=0.0, rate=1.0
        )
        self.tactical_review_count = drs.Level(
            "tactical_review_count", initial_value=0.0
        )

        self.development_trajectory_ratio = drs.Level(
            "development_trajectory_ratio", initial_value=1.0
        )
        self.ore1_trajectory_ratio = drs.Level(
            "ore1_trajectory_ratio", initial_value=1.0
        )
        self.ore2_trajectory_ratio = drs.Level(
            "ore2_trajectory_ratio", initial_value=1.0
        )

        self.fleet_mode: OperatingMode = (
            initial_fleet_mode or FLEET_MODES["PRODUCTION"]
        )
        self.development_priority_reserved_trucks = drs.Level(
            "development_priority_reserved_trucks", initial_value=0.0
        )

        # Annual accumulator baselines
        self.annual_ore1_extracted = 0.0
        self.annual_ore2_extracted = 0.0
        self.annual_development_start = 0.0

    def start_planning(self, current_cumulative_dev: float = 0.0):
        """Activates strategic cycle and aligns annual baseline trackers."""
        if not self.planning_started:
            self.planning_started = True
            self.strategic_year_index.value = 0.0
            self.strategic_year_timer.reset()
            self.tactical_review_timer.reset()
            self.tactical_review_count.value = 0.0
            self.annual_ore1_extracted = 0.0
            self.annual_ore2_extracted = 0.0
            self.annual_development_start = float(current_cumulative_dev)

    def step_timers(self, dt_days: float):
        """Advances internal strategic and tactical timers."""
        if self.planning_started:
            self.strategic_year_timer.step(dt_days)
            self.tactical_review_timer.step(dt_days)

    def record_production(self, ore1_mass: float, ore2_mass: float):
        """Accumulate ore production into current strategic year."""
        self.annual_ore1_extracted += ore1_mass
        self.annual_ore2_extracted += ore2_mass

    def update_review(
        self,
        current_cumulative_dev: float,
        area2_readiness_trajectory_ratio: float = 1.0,
        total_trucks: int = 0,
        force_mode: Optional[OperatingMode] = None,
    ) -> OperatingMode:
        """Executes annual rollover and periodic monthly tactical progress reviews."""
        if not self.planning_started:
            return self.fleet_mode

        # 1. Annual Rollover Check
        if self.strategic_year_timer.value >= self.strategic_period_days - 1e-6:
            self.strategic_year_index.value += 1.0
            self.strategic_year_timer.reset()
            self.annual_ore1_extracted = 0.0
            self.annual_ore2_extracted = 0.0
            self.annual_development_start = float(current_cumulative_dev)

        # 2. Compute Trajectory Ratios
        elapsed_year_fraction = max(
            1e-4,
            min(
                1.0,
                self.strategic_year_timer.value / self.strategic_period_days,
            ),
        )
        current_target = strategic_target_for_year(
            self.strategic_targets, int(self.strategic_year_index.value)
        )

        annual_dev = (
            float(current_cumulative_dev) - self.annual_development_start
        )
        self.development_trajectory_ratio.value = trajectory_progress_ratio(
            actual=annual_dev,
            annual_target=current_target.min_development,
            elapsed_fraction=elapsed_year_fraction,
        )
        self.ore1_trajectory_ratio.value = trajectory_progress_ratio(
            actual=self.annual_ore1_extracted,
            annual_target=current_target.min_ore1_production,
            elapsed_fraction=elapsed_year_fraction,
        )
        self.ore2_trajectory_ratio.value = trajectory_progress_ratio(
            actual=self.annual_ore2_extracted,
            annual_target=current_target.min_ore2_production,
            elapsed_fraction=elapsed_year_fraction,
        )

        # 3. Monthly Tactical Review
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
                    development_ratio=float(
                        self.development_trajectory_ratio.value
                    ),
                    ore1_ratio=float(self.ore1_trajectory_ratio.value),
                    ore2_ratio=float(self.ore2_trajectory_ratio.value),
                    tolerance=self.tactical_progress_tolerance,
                    area2_readiness_trajectory_ratio=area2_readiness_trajectory_ratio,
                )

            if (
                self.fleet_mode == FLEET_MODES["DEVELOPMENT"]
                and total_trucks > 0
            ):
                reserved = math.ceil(
                    total_trucks
                    * self.development_priority_truck_reservation_fraction
                )
                self.development_priority_reserved_trucks.value = float(
                    reserved
                )
            else:
                self.development_priority_reserved_trucks.value = 0.0

        return self.fleet_mode

    @property
    def active_fleet_mode(self) -> OperatingMode:
        """The currently active supervisory fleet operating mode."""
        return self.fleet_mode

    @active_fleet_mode.setter
    def active_fleet_mode(self, value: OperatingMode) -> None:
        self.fleet_mode = value


    def step_daily_tactical_review(
        self,
        current_day: float,
        cum_development: float,
        cum_ore1: float,
        cum_ore2: float,
        area2_readiness_tracker: Optional[Any] = None,
        total_trucks: int = 0,
        force_mode: Optional[OperatingMode] = None,
    ) -> OperatingMode:
        """Daily step helper that advances timers and triggers monthly review."""
        if not self.planning_started:
            self.start_planning(cum_development)
        self.strategic_year_timer.step(1.0)
        self.tactical_review_timer.step(1.0)
        a2_ratio = (
            float(area2_readiness_tracker.readiness_trajectory_ratio.value)
            if area2_readiness_tracker is not None
            else 1.0
        )
        return self.update_review(
            current_cumulative_dev=cum_development,
            area2_readiness_trajectory_ratio=a2_ratio,
            total_trucks=total_trucks,
            force_mode=force_mode,
        )

    def levels(self) -> Sequence[drs.Level]:
        return (
            self.strategic_year_index,
            self.strategic_year_timer,
            self.tactical_review_timer,
            self.tactical_review_count,
            self.development_trajectory_ratio,
            self.ore1_trajectory_ratio,
            self.ore2_trajectory_ratio,
            self.development_priority_reserved_trucks,
        )

    def time_to_event(self) -> float:
        min_dt = math.inf
        for lvl in self.levels():
            dt = lvl.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt

    def step(self, dt: float) -> None:
        for lvl in self.levels():
            lvl.step(dt)






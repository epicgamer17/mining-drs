import math
import logging
import random
from dataclasses import dataclass
import time
from typing import Tuple, Optional, Any
import pandas as pd
from .variables import Variable, Level
from .module import Module
from ._execution_context import ExecutionContext
from .exceptions import DeadlockError, ThresholdConfigurationError
from .config import EngineConfig
from .callbacks import Callback, ProgressBarCallback

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Encapsulates the final results of a simulation run."""

    model: Module
    config: Any
    duration: float  # wall time
    steps: int  # number of engine ticks
    sim_time: float  # simulation time reached
    history: Optional["pd.DataFrame"]  # telemetry data
    terminated_reason: str  # "max_time", "condition_met", "deadlock", etc.
    events: Optional[list] = None  # events log

    def print_event_timeline(self):
        """Prints the formatted event timeline if events exist."""
        if not self.events:
            print("No events logged.")
            return

        print("\n--- Event Audit Trail ---")
        for e in self.events:
            details_str = ", ".join(f"{k}={v}" for k, v in e.details.items())
            print(f"t={e.time:<6.2f} | {e.event_type:<15} | [{e.source}] {details_str}")
        print("-------------------------\n")

    def plot(self, *args, **kwargs):
        """Helper to plot telemetry data using pandas."""
        if self.history is None or self.history.empty:
            logger.warning("No telemetry data to plot.")
            return

        try:
            import matplotlib.pyplot as plt

            ax = self.history.plot(*args, **kwargs)
            plt.show()
            return ax
        except ImportError:
            logger.error("matplotlib is required for plotting.")

    def summary(self) -> str:
        """Returns a string summary of the simulation run."""
        lines = [
            f"--- Simulation Summary ---",
            f"Termination Reason : {self.terminated_reason}",
            f"Simulated Time     : {self.sim_time:.2f}",
            f"Wall Clock Time    : {self.duration:.4f} seconds",
            f"Engine Steps       : {self.steps:,}",
        ]
        if self.history is not None:
            lines.append(f"Telemetry Records  : {len(self.history):,}")
        return "\n".join(lines)

    def save(self, path: str):
        """Saves telemetry history to a CSV file."""
        if self.history is not None:
            self.history.to_csv(path, index=False)
            logger.info(f"Saved telemetry to {path}")
        else:
            logger.warning("No telemetry data to save.")


class DRSEngine:
    """The runner that manages the external simulation loop.

    The DRSEngine drives the simulation forward. It evaluates the model to
    determine rates and thresholds, calculates the time until the next event,
    and advances the system state to that precise moment in time.

    Attributes:
        model (Module): The root module of the simulation.
        current_time (float): The current simulation time.
        max_step_size (float): The maximum allowed time step (dt).
        max_deadlock_steps (int): The maximum consecutive zero-time steps allowed.
    """

    def __init__(
        self,
        model: Module,
        config: Optional[EngineConfig] = None,
        progress_bar: bool = False,
        log_level: Optional[str] = None,
        callbacks: Optional[list[Callback]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the DRS Engine.

        Args:
            model (Module): The root Module of your simulation.
            config (Optional[EngineConfig]): Configuration for the engine.
            progress_bar (bool): If True, attaches a Rich progress bar callback.
            log_level (Optional[str]): If provided, configures structured logging at this level.
            callbacks (Optional[list[Callback]]): Custom callbacks to attach.
            seed (Optional[int]): If provided, seeds random and numpy.random for determinism.
            **kwargs: Overrides for configuration parameters.
        """
        self.model = model
        self._seed = seed

        if log_level:
            logging.basicConfig(level=log_level.upper())

        self.callbacks = callbacks or []
        if progress_bar:
            self.callbacks.append(ProgressBarCallback())

        if config is None:
            config = EngineConfig()

        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        self.config = config
        self.current_time = 0.0
        self.max_step_size = (
            self.config.max_step_size
        )  # TODO: why do we have this? why is it not inf by default?
        self.max_deadlock_steps = self.config.max_deadlock_steps
        self.strict_mode = self.config.strict_mode
        self._orphaned_warned_ids = set()
        self.telemetry = None
        self.step_count = 0
        self._resuming = False

    def attach_telemetry(self, telemetry: Any) -> None:
        """
        Attach a Telemetry object to the engine.

        The engine will automatically trigger snapshots at the end of every time step.
        """
        self.telemetry = telemetry

    def save_checkpoint(self, filepath: str) -> None:
        """Save the full engine and model state to a JSON file."""
        from .serialize import save_checkpoint

        save_checkpoint(self, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load the full engine and model state from a JSON file."""
        from .serialize import load_checkpoint

        load_checkpoint(self, filepath)

    def run(self, max_time: float) -> SimulationResult:
        """
        Execute the main simulation loop.

        The loop repeatedly zeros rates, calls the model's `forward()` pass to
        evaluate states, calculates the time until the next threshold is hit
        (`dt`), and integrates all variables forward by `dt`.

        Args:
            max_time (float): The maximum simulation time to run until.

        Raises:
            RuntimeError: If the engine encounters a deadlock (too many consecutive
                zero-time steps).
            ValueError: If the calculated time delta (`dt`) is negative.
        """

        if self._seed is not None:
            random.seed(self._seed)
            try:
                import numpy as np

                np.random.seed(self._seed)
            except ImportError:
                pass

        ExecutionContext.push(self.model)
        ExecutionContext.set_engine(self)
        if not getattr(self, "_resuming", False):
            self.step_count = 0
            self.model.initialize_state()
        else:
            self._resuming = False
        ExecutionContext.pop()

        try:
            self._current_max_time = max_time
            for cb in self.callbacks:
                cb.on_simulation_start(self)

            self._consecutive_zero_dt_count = 0
            termination_reason = "unknown"
            steps = 0
            start_time = time.time()

            while True:
                if self.model.is_terminating_condition_met():
                    termination_reason = "condition_met"
                    break

                for cb in self.callbacks:
                    cb.on_step_start(self)

                if self.current_time >= max_time:
                    termination_reason = "max_time_reached"
                    break

                self._step(max_time)
                steps += 1

            if self.telemetry:
                self.telemetry.snapshot(self.current_time)
            self.model._run_post_step_hooks(self.current_time)
        finally:
            ExecutionContext.set_engine(None)

        end_time = time.time()
        df = self.telemetry.to_dataframe() if self.telemetry else None

        result = SimulationResult(
            model=self.model,
            config=self.config,
            duration=end_time - start_time,
            steps=steps,
            sim_time=self.current_time,
            history=df,
            events=self.telemetry.events if self.telemetry else None,
            terminated_reason=termination_reason,
        )

        for cb in self.callbacks:
            cb.on_complete(self, result)

        return result

    def _step(self, max_time: float) -> None:
        """
        [INTERNAL] Perform a single tick of the engine.

        Evaluates the model, calculates the time until the next event,
        and integrates variables forward.
        """
        self.model._zero_rates()
        self.model()

        current_variables = list(self.model.variables())
        self._check_orphaned_thresholds(current_variables)

        if self.telemetry:
            self.telemetry.snapshot(self.current_time)

        self.model._run_post_step_hooks(self.current_time)

        dt, trigger_var, is_upper = self._calculate_min_dt(current_variables)

        if trigger_var is not None:
            if self.telemetry is not None:
                threshold_hit = (
                    trigger_var.upper_threshold
                    if is_upper
                    else trigger_var.lower_threshold
                )
                self.telemetry.log_event(
                    time=self.current_time + dt,
                    event_type="THRESHOLD",
                    source="DRSEngine",
                    details={
                        "variable": trigger_var.name,
                        "threshold": threshold_hit,
                        "rate": trigger_var.rate,
                        "direction": "upper" if is_upper else "lower",
                    },
                )
            for cb in self.callbacks:
                cb.on_threshold(self, trigger_var, is_upper)

        dt = min(dt, self.max_step_size)
        dt = min(dt, max_time - self.current_time)

        if dt == 0.0:
            self._consecutive_zero_dt_count += 1
            if self._consecutive_zero_dt_count > self.max_deadlock_steps:
                self._handle_deadlock(current_variables, trigger_var)
        else:
            self._consecutive_zero_dt_count = 0

        if dt < 0:
            raise ValueError("Time delta (dt) cannot be negative.")

        logger.debug(
            f"Advancing time by {dt:.4f} to {self.current_time + dt:.4f} (Trigger: {trigger_var.name if trigger_var else 'None'})"
        )

        self.current_time += dt
        self.step_count += 1
        for var in current_variables:
            if hasattr(var, "_update"):
                var._update(dt)

    def _handle_deadlock(
        self, current_variables: list[Variable], trigger_var: Optional[Variable]
    ) -> None:
        """
        [INTERNAL] Handle the case where the engine ping-pongs between states without advancing time.
        """
        state_dump = "\n--- Engine State at Deadlock ---\n"
        for v in current_variables:
            rate_val = getattr(v, "rate", "N/A")
            lower_val = getattr(v, "lower_threshold", "N/A")
            upper_val = getattr(v, "upper_threshold", "N/A")
            state_dump += f"{v.name}: value={v.value}, rate={rate_val}, bounds=[{lower_val}, {upper_val}]\n"

        for cb in self.callbacks:
            cb.on_deadlock(self)

        if self.telemetry is not None:
            self.telemetry.log_event(
                time=self.current_time,
                event_type="DEADLOCK",
                source="DRSEngine",
                details={
                    "trigger_var": trigger_var.name if trigger_var else "None",
                    "trigger_val": trigger_var.value if trigger_var else "None",
                    "trigger_rate": getattr(trigger_var, "rate", "N/A")
                    if trigger_var
                    else "None",
                },
            )

        raise DeadlockError(
            f"Maximum consecutive zero-time steps ({self.max_deadlock_steps}) reached. "
            f"The simulation is ping-ponging between states without advancing time. "
            f"Last trigger: '{trigger_var.name if trigger_var else 'None'}' "
            f"(value={trigger_var.value if trigger_var else 'None'}, "
            f"rate={getattr(trigger_var, 'rate', 'N/A') if trigger_var else 'None'}).\n{state_dump}",
            state_dump=state_dump,
        )

    def _check_orphaned_thresholds(self, variables: list[Variable]) -> None:
        """
        [INTERNAL] Warn once per variable about thresholds set but rate=0.

        Power User Note: This helps catch logic bugs where a state transition
        threshold is set but the state is not actually changing, meaning the
        event will never fire.
        """
        for var in variables:
            if not isinstance(var, Level):
                continue
            if id(var) in self._orphaned_warned_ids:
                continue
            rate = var._rate
            has_threshold = (
                var.lower_threshold != -math.inf or var.upper_threshold != math.inf
            )
            if has_threshold and rate == 0.0:
                self._orphaned_warned_ids.add(id(var))
                owner_name = type(var._owner).__name__ if var._owner else "unknown"
                msg = (
                    f"Orphaned threshold: '{var.name}' (owned by {owner_name}) "
                    f"has lower_threshold={var.lower_threshold}, "
                    f"upper_threshold={var.upper_threshold} "
                    f"but rate=0.0. This threshold will never trigger."
                )
                if self.strict_mode:
                    raise ThresholdConfigurationError(msg)
                logger.warning(msg)

    def _calculate_min_dt(
        self, variables: list[Variable]
    ) -> Tuple[float, Optional[Variable], bool]:
        """
        [INTERNAL] Determine the time step (dt) to the next event/threshold.

        Power User Note: Evaluates all variables in the system to find the
        closest future threshold hit based on current rates.

        Args:
            variables (list[Variable]): A list of all variables in the system.

        Returns:
            Tuple[float, Optional[Variable], bool]:
                - min_dt: The time until the next event.
                - trigger_var: The variable that will hit its threshold.
                - is_upper: True if hitting upper_threshold, False if lower_threshold.
        """
        min_dt = math.inf
        trigger_var = None
        is_upper = True

        for var in variables:
            dt_for_var = math.inf
            var_is_upper = True

            if hasattr(var, "rate"):
                rate = var.rate
                if rate > 0:
                    dt_for_var = (var.upper_threshold - var.value) / rate
                elif rate < 0:
                    dt_for_var = (var.value - var.lower_threshold) / abs(rate)
                    var_is_upper = False

            if -1e-12 <= dt_for_var < min_dt:
                min_dt = max(0.0, dt_for_var)
                trigger_var = var
                is_upper = var_is_upper

        if min_dt == math.inf:
            orphaned = []
            for var in variables:
                if not isinstance(var, Level):
                    continue
                rate = var._rate
                has_threshold = (
                    var.lower_threshold != -math.inf or var.upper_threshold != math.inf
                )
                if has_threshold and rate == 0.0:
                    owner_name = type(var._owner).__name__ if var._owner else "unknown"
                    orphaned.append(f"'{var.name}' ({owner_name})")
            if orphaned and id(None) not in self._orphaned_warned_ids:
                self._orphaned_warned_ids.add(id(None))
                msg = (
                    f"No threshold events pending. "
                    f"Variables with thresholds but rate=0: "
                    f"{', '.join(orphaned)}. "
                    f"Simulation will advance at max_step_size={self.max_step_size}."
                )
                if self.strict_mode:
                    raise ThresholdConfigurationError(msg)
                logger.warning(msg)
            return 1.0, None, True

        return min_dt, trigger_var, is_upper

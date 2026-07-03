import math
import logging
from typing import Tuple, Optional
from .variables import Variable, Level
from .module import Module
from ._execution_context import ExecutionContext
from .exceptions import DeadlockError

logger = logging.getLogger(__name__)


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
        max_step_size: float = 0.5,
        max_deadlock_steps: int = 20,
    ) -> None:
        """
        Initialize the DRS Engine.

        Args:
            model (Module): The root Module of your simulation.
            max_step_size (float): The maximum time delta (dt) the engine is allowed 
                to take in a single step (default: 0.5).
            max_deadlock_steps (int): The maximum consecutive zero-time steps allowed 
                before raising a deadlock error (default: 20).
        """
        self.model = model
        self.current_time = 0.0
        self.max_step_size = max_step_size
        self.max_deadlock_steps = max_deadlock_steps
        self._orphaned_warned_ids = set()

    def run(self, max_time: Optional[float] = None) -> None:
        """
        Execute the main simulation loop.

        The loop repeatedly zeros rates, calls the model's `forward()` pass to
        evaluate states, calculates the time until the next threshold is hit
        (`dt`), and integrates all variables forward by `dt`.

        Args:
            max_time (Optional[float]): The maximum simulation time to run until. 
                If None, the engine runs until a custom terminating condition is met.

        Raises:
            RuntimeError: If the engine encounters a deadlock (too many consecutive
                zero-time steps).
            ValueError: If the calculated time delta (`dt`) is negative.
        """

        ExecutionContext.push(self.model)
        self.model.initialize_state()
        ExecutionContext.pop()

        consecutive_zero_dt_count = 0

        while True:
            if self.model.is_terminating_condition_met():
                break

            if max_time is not None and self.current_time >= max_time:
                break

            self.model._zero_rates()
            self.model()

            current_variables = list(self.model.variables())
            self._check_orphaned_thresholds(current_variables)

            self.model._run_post_step_hooks(self.current_time)

            dt, trigger_var, is_upper = self._calculate_min_dt(current_variables)

            dt = min(dt, self.max_step_size)

            if max_time is not None:
                dt = min(dt, max_time - self.current_time)

            if dt == 0.0:
                consecutive_zero_dt_count += 1
                if consecutive_zero_dt_count > self.max_deadlock_steps:
                    state_dump = "\n--- Engine State at Deadlock ---\n"
                    for v in current_variables:
                        rate_val = getattr(v, "rate", "N/A")
                        lower_val = getattr(v, "lower_threshold", "N/A")
                        upper_val = getattr(v, "upper_threshold", "N/A")
                        state_dump += f"{v.name}: value={v.value}, rate={rate_val}, bounds=[{lower_val}, {upper_val}]\n"

                    raise DeadlockError(
                        f"Maximum consecutive zero-time steps ({self.max_deadlock_steps}) reached. "
                        f"The simulation is ping-ponging between states without advancing time. "
                        f"Last trigger: '{trigger_var.name if trigger_var else 'None'}' "
                        f"(value={trigger_var.value if trigger_var else 'None'}, "
                        f"rate={getattr(trigger_var, 'rate', 'N/A') if trigger_var else 'None'}).\n{state_dump}",
                        state_dump=state_dump
                    )
            else:
                consecutive_zero_dt_count = 0

            if dt < 0:
                raise ValueError("Time delta (dt) cannot be negative.")

            logger.debug(f"Advancing time by {dt:.4f} to {self.current_time + dt:.4f} (Trigger: {trigger_var.name if trigger_var else 'None'})")

            self.current_time += dt
            for var in current_variables:
                if hasattr(var, "_update"):
                    var._update(dt)

        self.model._run_post_step_hooks(self.current_time)

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
                var.lower_threshold != -math.inf or
                var.upper_threshold != math.inf
            )
            if has_threshold and rate == 0.0:
                self._orphaned_warned_ids.add(id(var))
                owner_name = type(var._owner).__name__ if var._owner else "unknown"
                logger.warning(
                    f"Orphaned threshold: '{var.name}' (owned by {owner_name}) "
                    f"has lower_threshold={var.lower_threshold}, "
                    f"upper_threshold={var.upper_threshold} "
                    f"but rate=0.0. This threshold will never trigger."
                )

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
                    var.lower_threshold != -math.inf or
                    var.upper_threshold != math.inf
                )
                if has_threshold and rate == 0.0:
                    owner_name = type(var._owner).__name__ if var._owner else "unknown"
                    orphaned.append(f"'{var.name}' ({owner_name})")
            if orphaned and id(None) not in self._orphaned_warned_ids:
                self._orphaned_warned_ids.add(id(None))
                logger.warning(
                    f"No threshold events pending. "
                    f"Variables with thresholds but rate=0: "
                    f"{', '.join(orphaned)}. "
                    f"Simulation will advance at max_step_size={self.max_step_size}."
                )
            return 1.0, None, True

        return min_dt, trigger_var, is_upper

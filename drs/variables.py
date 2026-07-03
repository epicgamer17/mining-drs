"""
Note on Expression AST System:

The `Expression` class and operator overloading (`__add__`, `__sub__`, etc.) within `Variable`
have been removed for maximum cleanup and performance, making `Variable` incredibly lightweight.
Since the `DRSEngine` recalculates rates dynamically every tick via `self.model()` and never
actually toggles `ExecutionContext.set_tracing(True)`, the framework currently relies entirely
on Eager Evaluation.

If you ever decide to implement the Arena-like drag-and-drop GUI or JSON exporter:
You will need to resurrect the `Expression` AST system to perform symbolic "dry runs" and capture
the structural relationships (the AST) between variables without executing the raw floats. You can
recover the `Expression` class and the magic method overloads (`_op`, `_rop`, etc.) from earlier
Git commits or from the codebase prior to the "Maximum Cleanup" refactor.
"""

import math
from typing import Any, Union
from ._execution_context import ExecutionContext
from .exceptions import StateMutationError


class Variable:
    """Base class for all domain variables.

    Variables hold named state and belong to a specific `Module` owner. They ensure
    that state is tracked properly through the execution context and prevent
    cross-module mutation.

    Attributes:
        name (str): The unique name of the variable.
    """

    def __init__(self, name: str, initial_value: Any = 0.0) -> None:
        """
        Initialize a new Variable.

        Args:
            name: The unique name of the variable.
            initial_value: The starting value (default: 0.0).
        """
        self.name = name
        self._value = initial_value
        self._owner = None

    def _record_read_dependency(self) -> None:
        """
        [INTERNAL] Record that the current executing module has read this variable.

        Power User Note: This is called automatically by the `value` getter. It
        interfaces with the ExecutionContext to build the dependency graph.
        """
        current = ExecutionContext.get_current()
        if current is not None and current is not self._owner:
            current._record_incoming_edge(self)

    @property
    def value(self) -> Any:
        """
        Get the current value of the variable.

        Reading this automatically records a dependency edge in the execution context,
        linking the module that read it to the module that owns it.

        Returns:
            Any: The underlying value of the variable.
        """
        self._record_read_dependency()
        return self._value

    @value.setter
    def value(self, val: Any) -> None:
        """
        Set the value of the variable.

        Args:
            val (Any): The new value to set.

        Raises:
            RuntimeError: If a module attempts to mutate a variable it does not own.
        """
        current = ExecutionContext.get_current()
        if current is not None and current is not self._owner:
            raise StateMutationError(
                f"Illegal Mutation: {type(current).__name__} tried to mutate "
                f"'{self.name}' owned by {type(self._owner).__name__}. "
                f"Modules must communicate by passing Flows. Do not mutate state directly!"
            )
        self._value = val

    @property
    def rate(self) -> float:
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute 'rate'. "
            f"Only drs.Level supports .rate. Use drs.Level() for quantities that flow."
        )

    @rate.setter
    def rate(self, val: Union[float, tuple[float, float, float]]) -> None:
        raise AttributeError(
            f"Cannot set .rate on '{type(self).__name__}'. "
            f"Only drs.Level supports .rate."
        )

    def __hash__(self) -> int:
        return id(self)


class Level(Variable):
    """A variable that accumulates over time based on a rate.

    Levels are the primary way to model physical quantities that flow or change
    continuously over time (e.g., mass in a stockpile, energy in a battery).

    Attributes:
        upper_threshold (float): The maximum limit for the level. The engine will
            stop exactly at this boundary. Defaults to math.inf.
        lower_threshold (float): The minimum limit for the level. Defaults to -math.inf.
    """

    # TODO: Add a floating-point comparison utility to Level (e.g. is_above(threshold, eps)).
    # The proper long-term solution would use an AST-based approach (similar to the Expression
    # system that was removed, see note at top of file) to track variable relationships
    # symbolically. This would avoid raw float comparisons entirely by deferring to the
    # engine's event-driven threshold detection instead of comparing .value directly in
    # user code. Until then, guardrail #1 (orphaned-threshold check in DRSEngine) catches
    # the most common manifestation: thresholds set without corresponding rates.

    def __init__(
        self, name: str, initial_value: float = 0.0, rate: float = 0.0
    ) -> None:
        """
        Initialize a new Level.

        Args:
            name: The unique name of the level.
            initial_value: The starting value (default: 0.0).
            rate: The initial rate of change (default: 0.0).
        """
        super().__init__(name, initial_value)
        self._rate = rate
        self.upper_threshold = math.inf
        self.lower_threshold = -math.inf
        self._rate_set_by = None

    @property
    def rate(self) -> float:
        """
        Get the current rate of change.

        Returns:
            float: The rate at which the level is currently accumulating per time unit.
        """
        self._record_read_dependency()
        return self._rate

    @rate.setter
    def rate(self, val: Union[float, tuple[float, float, float]]) -> None:
        """
        Set the rate of change.

        Args:
            val (Any): Can be a single float representing the new rate, or a tuple
                of `(rate, lower_threshold, upper_threshold)`.

        Raises:
            ValueError: If a tuple is provided but it does not have exactly 3 elements.
        """
        current_actor = ExecutionContext.get_current()
        if current_actor is not None and current_actor is not self._owner:
            if hasattr(current_actor, "_record_incoming_edge"):
                current_actor._record_incoming_edge(self)

        # Rate override guardrail
        if current_actor is not None and self._rate_set_by is not None and self._rate_set_by is not current_actor:
            raise StateMutationError(
                f"Rate Conflict: '{type(current_actor).__name__}' attempted to set the rate of "
                f"'{self.name}', but it was already set by '{type(self._rate_set_by).__name__}' "
                f"during this time step. Multiple modules cannot control the rate of the same Level."
            )
        self._rate_set_by = current_actor

        if isinstance(val, tuple):
            if len(val) == 3:
                self._rate, self.lower_threshold, self.upper_threshold = val
            else:
                raise ValueError(f"Rate tuple must be (rate, lower, upper), got {val}")
        else:
            self._rate = val

    def _update(self, dt: float) -> None:
        """
        [INTERNAL] Step the level forward in time based on its current rate.

        Power User Note: This is called automatically by the DRSEngine. Do not call this
        manually unless you are implementing a custom time-stepping loop.

        Args:
            dt (float): The amount of time to simulate.
        """
        self.value += self.rate * dt


class Timer(Level):
    """A specialized level used to track time.

    Timers are simply Levels that accumulate at a default rate of 1.0 (or -1.0 for countdowns).
    """

    def __init__(
        self, name: str, initial_value: float = 0.0, rate: float = 1.0
    ) -> None:
        """
        Initialize a Timer.

        Args:
            name: The unique name of the timer.
            initial_value: The starting time value (default: 0.0).
            rate: The speed of time (default: 1.0).
        """
        super().__init__(name, initial_value, rate)

    def reset(self) -> None:
        """
        Reset the timer value back to 0.0.

        This sets the absolute value of the timer to 0, but does not modify the rate.
        """
        self.value = 0.0

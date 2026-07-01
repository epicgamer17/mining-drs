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
from .execution_context import ExecutionContext



class Variable:
    """Base class for all domain variables."""

    def __init__(self, name: str, initial_value: Any = 0.0):
        self.name = name
        self._value = initial_value
        self._owner = None


    def _record_read_dependency(self):
        current = ExecutionContext.get_current()
        if current is not None and current is not self._owner:
            current._record_incoming_edge(self)

    @property
    def value(self):
        self._record_read_dependency()
        return self._value

    @value.setter
    def value(self, val):
        current = ExecutionContext.get_current()
        if current is not None and current is not self._owner:
            raise RuntimeError(
                f"Illegal Mutation: {type(current).__name__} tried to mutate "
                f"'{self.name}' owned by {type(self._owner).__name__}. "
                f"Modules must communicate by passing Signals/Flows. Do not mutate state directly!"
            )
        self._value = val

    @property
    def rate(self):
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute 'rate'. "
            f"Only drs.Level supports .rate. Use drs.Level() for quantities that flow."
        )

    @rate.setter
    def rate(self, val):
        raise AttributeError(
            f"Cannot set .rate on '{type(self).__name__}'. "
            f"Only drs.Level supports .rate."
        )

    def get_sources(self) -> list:
        return [self]

    def __hash__(self):
        return id(self)




class Level(Variable):
    """A variable that accumulates over time based on a rate."""

    # TODO: Add a floating-point comparison utility to Level (e.g. is_above(threshold, eps)).
    # The proper long-term solution would use an AST-based approach (similar to the Expression
    # system that was removed, see note at top of file) to track variable relationships
    # symbolically. This would avoid raw float comparisons entirely by deferring to the
    # engine's event-driven threshold detection instead of comparing .value directly in
    # user code. Until then, guardrail #1 (orphaned-threshold check in DRSEngine) catches
    # the most common manifestation: thresholds set without corresponding rates.

    def __init__(self, name: str, initial_value: float = 0.0, rate: float = 0.0):
        super().__init__(name, initial_value)
        self._rate = rate
        self.upper_threshold = math.inf
        self.lower_threshold = -math.inf

    @property
    def rate(self) -> float:
        self._record_read_dependency()
        return self._rate

    @rate.setter
    def rate(self, val):
        current_actor = ExecutionContext.get_current()
        if current_actor is not None and current_actor is not self._owner:
            if hasattr(current_actor, "_record_incoming_edge"):
                current_actor._record_incoming_edge(self)

        if isinstance(val, tuple):
            if len(val) == 3:
                self._rate, self.lower_threshold, self.upper_threshold = val
            else:
                raise ValueError(f"Rate tuple must be (rate, lower, upper), got {val}")
        else:
            self._rate = val

    def update(self, dt: float):
        self.value += self.rate * dt


class Timer(Level):
    """A specialized level used to track time, typically with a rate of 1.0 or -1.0."""

    def __init__(self, name: str, initial_value: float = 0.0, rate: float = 1.0):
        super().__init__(name, initial_value, rate)

    def reset(self):
        self.value = 0.0

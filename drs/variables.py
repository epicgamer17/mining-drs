import math
import enum
from typing import Any, Union
from ._execution_context import ExecutionContext
from .exceptions import StateMutationError


def serialize_val(val: Any) -> Any:
    if isinstance(val, enum.Enum):
        return val.name
    if (
        hasattr(val, "name")
        and hasattr(val, "id")
        and not isinstance(val, (int, float, str, bool))
    ):
        return {"__type__": type(val).__name__, "name": val.name}
    if isinstance(val, float):
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        elif math.isnan(val):
            return "NaN"
    return val


def deserialize_val(val: Any) -> Any:
    if val == "Infinity":
        return math.inf
    elif val == "-Infinity":
        return -math.inf
    elif val == "NaN":
        return math.nan
    return val


class Expression:
    """AST node for tracking mathematical dependencies between Variables."""

    def __init__(self, op: str, left: Any, right: Any):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self) -> float:
        def get_val(node):
            if isinstance(node, Expression):
                return node.evaluate()
            if hasattr(node, "_sim_value"):
                return node._sim_value()
            return node

        l_val = get_val(self.left)
        r_val = get_val(self.right)

        if self.op == "neg":
            return -l_val
        if self.op == "pos":
            return +l_val
        if self.op == "abs":
            return abs(l_val)
        if self.op == "add":
            return l_val + r_val
        if self.op == "sub":
            return l_val - r_val
        if self.op == "mul":
            return l_val * r_val
        if self.op == "div":
            return l_val / r_val if r_val != 0 else 0.0
        if self.op == "gt":
            return l_val > r_val
        if self.op == "lt":
            return l_val < r_val
        if self.op == "ge":
            return l_val >= r_val
        if self.op == "le":
            return l_val <= r_val
        if self.op == "eq":
            return l_val == r_val
        if self.op == "ne":
            return l_val != r_val
        if self.op == "pow":
            return l_val**r_val
        return 0.0

    def get_sources(self) -> list:
        sources = set()
        for side in (self.left, self.right):
            if hasattr(side, "get_sources"):
                sources.update(side.get_sources())
            elif isinstance(side, Variable):
                sources.add(side)
        return list(sources)

    def get_equation(self) -> str:
        op_chars = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "gt": ">",
            "lt": "<",
            "ge": ">=",
            "le": "<=",
            "eq": "==",
            "ne": "!=",
            "pow": "**",
        }
        unary_chars = {"neg": "-", "pos": "+", "abs": "|"}

        def format_node(node):
            if isinstance(node, Expression):
                return node.get_equation()
            if hasattr(node, "name"):
                mod = getattr(node, "_owner", None)
                if mod and hasattr(mod, "name"):
                    return f"{mod.name}.{node.name}"
                elif mod:
                    return f"{type(mod).__name__}.{node.name}"
                return node.name
            return str(node)

        if self.op in unary_chars:
            l = unary_chars[self.op]
            r = unary_chars[self.op] if self.op == "abs" else ""
            return f"({l}{format_node(self.left)}{r})"
        return f"({format_node(self.left)} {op_chars.get(self.op, '?')} {format_node(self.right)})"

    def __bool__(self):
        raise TypeError(
            f"Cannot use Expression ('{self.get_equation()}') as a boolean. "
            f"Use `.value` for immediate evaluation or `drs.Where()` for symbolic branching."
        )

    def __neg__(self):
        return Expression("neg", self, None)

    def __pos__(self):
        return Expression("pos", self, None)

    def __abs__(self):
        return Expression("abs", self, None)

    def __add__(self, other):
        return Expression("add", self, other)

    def __sub__(self, other):
        return Expression("sub", self, other)

    def __mul__(self, other):
        return Expression("mul", self, other)

    def __truediv__(self, other):
        return Expression("div", self, other)

    def __radd__(self, other):
        return Expression("add", other, self)

    def __rsub__(self, other):
        return Expression("sub", other, self)

    def __rmul__(self, other):
        return Expression("mul", other, self)

    def __rtruediv__(self, other):
        return Expression("div", other, self)

    def __gt__(self, other):
        return Expression("gt", self, other)

    def __lt__(self, other):
        return Expression("lt", self, other)

    def __ge__(self, other):
        return Expression("ge", self, other)

    def __le__(self, other):
        return Expression("le", self, other)

    def __eq__(self, other):
        return Expression("eq", self, other)

    def __ne__(self, other):
        return Expression("ne", self, other)

    def __pow__(self, other):
        return Expression("pow", self, other)

    def __rpow__(self, other):
        return Expression("pow", other, self)


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

    def _sim_value(self) -> Any:
        if isinstance(self._value, Expression):
            return self._value.evaluate()
        return self._value

    def get_sources(self) -> list:
        return [self]

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
        if ExecutionContext.is_tracing():
            return self
        return self._sim_value()

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

        if self._value != val:
            engine = ExecutionContext.get_engine()
            if engine and getattr(engine, "telemetry", None):
                engine.telemetry.log_event(
                    time=engine.current_time,
                    event_type="STATE_CHANGE",
                    source=type(current).__name__ if current else "External",
                    details={
                        "variable": self.name,
                        "old_value": self._value,
                        "new_value": val,
                    },
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

    def _unary(self, op: str):
        self._record_read_dependency()
        if ExecutionContext.is_tracing():
            return Expression(op, self, None)
        l_val = self._sim_value()
        if op == "neg":
            return -l_val
        if op == "pos":
            return +l_val
        if op == "abs":
            return abs(l_val)
        return NotImplemented

    def _op(self, op: str, other):
        self._record_read_dependency()
        if isinstance(other, Variable):
            other._record_read_dependency()
        if ExecutionContext.is_tracing():
            return Expression(op, self, other)
        r_val = other._sim_value() if isinstance(other, Variable) else other
        if isinstance(r_val, Expression):
            r_val = r_val.evaluate()
        l_val = self._sim_value()
        if op == "add":
            return l_val + r_val
        if op == "sub":
            return l_val - r_val
        if op == "mul":
            return l_val * r_val
        if op == "div":
            return l_val / r_val if r_val != 0 else 0.0
        if op == "gt":
            return l_val > r_val
        if op == "lt":
            return l_val < r_val
        if op == "ge":
            return l_val >= r_val
        if op == "le":
            return l_val <= r_val
        if op == "eq":
            return l_val == r_val
        if op == "ne":
            return l_val != r_val
        if op == "pow":
            return l_val**r_val
        return NotImplemented

    def _rop(self, op: str, other):
        self._record_read_dependency()
        if isinstance(other, Variable):
            other._record_read_dependency()
        if ExecutionContext.is_tracing():
            return Expression(op, other, self)
        l_val = other._sim_value() if isinstance(other, Variable) else other
        if isinstance(l_val, Expression):
            l_val = l_val.evaluate()
        r_val = self._sim_value()
        if op == "add":
            return l_val + r_val
        if op == "sub":
            return l_val - r_val
        if op == "mul":
            return l_val * r_val
        if op == "div":
            return l_val / r_val if r_val != 0 else 0.0
        if op == "pow":
            return l_val**r_val
        return NotImplemented

    def __neg__(self):
        return self._unary("neg")

    def __pos__(self):
        return self._unary("pos")

    def __abs__(self):
        return self._unary("abs")

    def __add__(self, other):
        return self._op("add", other)

    def __sub__(self, other):
        return self._op("sub", other)

    def __mul__(self, other):
        return self._op("mul", other)

    def __truediv__(self, other):
        return self._op("div", other)

    def __radd__(self, other):
        return self._rop("add", other)

    def __rsub__(self, other):
        return self._rop("sub", other)

    def __rmul__(self, other):
        return self._rop("mul", other)

    def __rtruediv__(self, other):
        return self._rop("div", other)

    def __gt__(self, other):
        return self._op("gt", other)

    def __lt__(self, other):
        return self._op("lt", other)

    def __ge__(self, other):
        return self._op("ge", other)

    def __le__(self, other):
        return self._op("le", other)

    def __eq__(self, other):
        return self._op("eq", other)

    def __ne__(self, other):
        return self._op("ne", other)

    def __pow__(self, other):
        return self._op("pow", other)

    def __rpow__(self, other):
        return self._rop("pow", other)

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
        if ExecutionContext.is_tracing():
            return self._rate
        if isinstance(self._rate, Expression):
            return self._rate.evaluate()
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
        if (
            current_actor is not None
            and self._rate_set_by is not None
            and self._rate_set_by is not current_actor
        ):
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

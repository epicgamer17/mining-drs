import math
from typing import Iterator, Any, Tuple
from .variables import Variable, Level, Timer
from ._execution_context import ExecutionContext
from .data_source import DataPoint
from .flow import Flow


class Module:
    """
    Base class for all DRS models and sub-components.
    Automatically registers variables and sub-modules upon assignment.
    """

    def __init__(self) -> None:
        self._variables = {}
        self._modules = {}
        self.parent = None
        self._post_step_hooks = []
        self._dependencies = []
        self._dep_seen = set()
        self._flow_dependencies = []
        self._flow_dep_seen = set()
        self._data_dependencies = []
        self._data_dep_seen = set()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        caller = ExecutionContext.get_current()
        ExecutionContext.push(self)
        try:
            if caller is not None and caller is not self:

                def validate_drs_type(arg, arg_name):
                    if arg is None:
                        return
                    if isinstance(arg, (tuple, list)):
                        for item in arg:
                            validate_drs_type(item, arg_name)
                        return

                    if not isinstance(arg, (Flow, Variable, DataPoint)):
                        raise RuntimeError(
                            f"Hidden Dependency Error: '{type(caller).__name__}' passed an untracked type "
                            f"'{type(arg).__name__}' to '{type(self).__name__}' for {arg_name}. "
                            f"Inter-module arguments MUST be drs.Flow (physics) or drs.Variable (control)."
                        )

                for i, arg in enumerate(args):
                    validate_drs_type(arg, f"positional arg {i}")
                for key, val in kwargs.items():
                    validate_drs_type(val, f"keyword arg '{key}'")

            for arg in args:
                if isinstance(arg, Flow) and arg._source is not None:
                    ExecutionContext.record_flow_edge(arg._source, self)
                    self._record_flow_edge(arg._source)
                elif isinstance(arg, DataPoint) and arg._source is not None:
                    self._record_data_edge(arg._source)
            for v in kwargs.values():
                if isinstance(v, Flow) and v._source is not None:
                    ExecutionContext.record_flow_edge(v._source, self)
                    self._record_flow_edge(v._source)
                elif isinstance(v, DataPoint) and v._source is not None:
                    self._record_data_edge(v._source)

            result = self.forward(*args, **kwargs)

            if isinstance(result, tuple):
                for res in result:
                    if not isinstance(res, (Flow, DataPoint)):
                        raise RuntimeError(
                            f"Tuple returned from '{type(self).__name__}.forward()' "
                            f"must contain only drs.Flow or drs.DataPoint objects."
                        )
                    res._source = self
                return result

            if isinstance(self, DataSource):
                if result is None:
                    raise RuntimeError(
                        f"'{type(self).__name__}.forward()' must return drs.DataPoint or drs.Flow. "
                        f"DataSource subclasses cannot return None."
                    )
                if not isinstance(result, (Flow, DataPoint)):
                    raise RuntimeError(
                        f"'{type(self).__name__}.forward()' returned "
                        f"'{type(result).__name__}', not drs.Flow or drs.DataPoint."
                    )

            if result is not None and not isinstance(result, (Flow, DataPoint)):
                raise RuntimeError(
                    f"'{type(self).__name__}.forward()' returned "
                    f"'{type(result).__name__}', not a drs.Flow or drs.DataPoint. "
                    f"Inter-module communication must use drs.Flow or drs.DataPoint."
                )

            if isinstance(result, (Flow, DataPoint)):
                result._source = self

            return result
        finally:
            ExecutionContext.pop()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Module subclasses must implement forward()")

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        if hasattr(self, "_variables"):
            self._variables.pop(name, None)
        if hasattr(self, "_modules") and name != "parent":
            self._modules.pop(name, None)

        if isinstance(value, Variable):
            if not hasattr(self, "_variables"):
                raise AttributeError(
                    "Cannot assign variable before Module.__init__() call"
                )
            self._variables[name] = value
            value._owner = self
        elif isinstance(value, Module) and name != "parent":
            if not hasattr(self, "_modules"):
                raise AttributeError(
                    "Cannot assign module before Module.__init__() call"
                )
            self._modules[name] = value
            if not getattr(value, "parent", None):
                value.parent = self

        super().__setattr__(name, value)

    def _record_incoming_edge(self, variable: Variable) -> None:
        """Record that this module reads 'variable' (owned by another module)."""
        if variable._owner is not None and variable._owner is not self:
            key = (id(variable._owner), id(variable))
            if key not in self._dep_seen:
                self._dep_seen.add(key)
                self._dependencies.append((variable._owner, variable))

    def _record_flow_edge(self, source_module: "Module") -> None:
        """Record that this module received a Flow from source_module."""
        if source_module is not None and source_module is not self:
            key = id(source_module)
            if key not in self._flow_dep_seen:
                self._flow_dep_seen.add(key)
                self._flow_dependencies.append(source_module)

    def _record_data_edge(self, source_module: "Module") -> None:
        """Record that this module received a DataPoint from source_module."""
        if source_module is not None and source_module is not self:
            key = id(source_module)
            if key not in self._data_dep_seen:
                self._data_dep_seen.add(key)
                self._data_dependencies.append(source_module)

    def variables(self) -> Iterator[Variable]:
        """Recursively yield all variables without duplicates."""
        seen = set()

        def _get_vars(module):
            for var in module._variables.values():
                var_id = id(var)
                if var_id not in seen:
                    seen.add(var_id)
                    yield var
            for mod in module._modules.values():
                yield from _get_vars(mod)

        yield from _get_vars(self)

    def modules(self) -> Iterator["Module"]:
        """Recursively yield this module and all nested modules."""
        for _, module in self.named_modules():
            yield module

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        """Recursively yield ``(path, module)`` pairs using PyTorch-style names."""
        seen = set()

        def _get_modules(module, module_prefix):
            module_id = id(module)
            if module_id in seen:
                return
            seen.add(module_id)
            yield module_prefix, module

            for name, sub_mod in module._modules.items():
                sub_prefix = name if not module_prefix else f"{module_prefix}.{name}"
                yield from _get_modules(sub_mod, sub_prefix)

        yield from _get_modules(self, prefix)

    def _zero_rates(self) -> None:
        """Zero out rates and remove thresholds for all Levels before the next rate update."""
        for var in self.variables():
            if isinstance(var, Level):
                var._rate = 0.0
                var.upper_threshold = math.inf
                var.lower_threshold = -math.inf

    def initialize_state(self) -> None:
        """Override this to set up initial state before the simulation starts."""
        pass

    def is_terminating_condition_met(self) -> bool:
        """Override this to define custom stopping conditions."""
        return False

    def register_post_step_hook(self, hook_fn: Any) -> None:
        """Registers a callback to be run after every engine step."""
        self._post_step_hooks.append(hook_fn)

    def _run_post_step_hooks(self, current_time: float) -> None:
        """Called by the engine after dt is integrated."""
        for hook in self._post_step_hooks:
            hook(current_time)

    def get_dependency_graph(self) -> list:
        """Return all recorded read dependencies from this module and all sub-modules
        as (source_module, variable) pairs."""
        result = []
        for mod in self.modules():
            result.extend(mod._dependencies)
        return result


# NOTE: could remove and use just modules and Flow instead of DataSource and DataPoint. May be better.
class DataSource(Module):
    """Yields ``DataPoint`` batches one at a time.

    Subclass and implement ``__next__`` to define the data stream.
    Raise ``StopIteration`` when the stream is exhausted::

        class MySource(DataSource):
            def __init__(self):
                super().__init__()
                self._data = [DataPoint(x=1), DataPoint(x=2)]
                self._index = 0

            def __next__(self) -> DataPoint:
                if self._index >= len(self._data):
                    raise StopIteration
                point = self._data[self._index]
                self._index += 1
                return point
    """

    def __init__(self) -> None:
        super().__init__()

    def __iter__(self) -> Iterator[DataPoint]:
        return self

    def __next__(self) -> DataPoint:
        raise StopIteration

    def next(self) -> DataPoint:
        return next(self)


class drs:
    """A namespace to mimic PyTorch's structure"""

    Module = Module
    Variable = Variable
    Level = Level
    Timer = Timer
    DataPoint = DataPoint
    DataSource = DataSource
    Flow = Flow

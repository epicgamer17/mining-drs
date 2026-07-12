import math
from typing import Iterator, Any, Optional
from .variables import Variable, Level, Timer
from ._execution_context import ExecutionContext
from .data_source import DataPoint
from .flow import Flow


class Module:
    """Base class for all DRS models and sub-components.

    Modules are the fundamental building blocks of a simulation. They automatically
    register any `Variable` or `Module` assigned as an attribute, mimicking the
    behavior of PyTorch's `nn.Module`.

    Attributes:
        parent (Optional[Module]): The parent module that owns this module.
    """

    def __init__(self) -> None:
        """Initialize the module and its internal registries."""
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
        """
        Execute the forward pass while managing the ExecutionContext.

        This method acts as a wrapper around `forward()`. It pushes this module
        onto the execution stack, validates that inter-module communication uses
        only `drs.Flow` or `drs.DataPoint`, records dependency edges, and then
        pops the execution stack.

        Args:
            *args: Positional arguments passed to `forward()`.
            **kwargs: Keyword arguments passed to `forward()`.

        Returns:
            Any: The result of the `forward()` pass.

        Raises:
            RuntimeError: If invalid types are passed or returned.
        """
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
        """
        Define the physics and logic of the module.

        This method must be implemented by all subclasses. It is called on every
        time step to update rates and determine flows.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Any: Typically a `drs.Flow`, `drs.DataPoint`, or a tuple of them.
        """
        raise NotImplementedError("Module subclasses must implement forward()")

    # TODO: add @property def current_time(self) -> float
    # This would implicitly grab ExecutionContext.get_engine().current_time.
    # Benefit: Allows modules/controllers to read simulation time natively
    # (e.g. `if self.current_time > 10.0:`) without needing to explicitly instantiate and track a Timer variable.

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
        """
        [INTERNAL] Record that this module reads 'variable' (owned by another module).

        Power User Note: Automatically called by the ExecutionContext to build graphs.
        """
        if variable._owner is not None and variable._owner is not self:
            key = (id(variable._owner), id(variable))
            if key not in self._dep_seen:
                self._dep_seen.add(key)
                self._dependencies.append((variable._owner, variable))

    def _record_flow_edge(self, source_module: "Module") -> None:
        """
        [INTERNAL] Record that this module received a Flow from source_module.

        Power User Note: Automatically called by the ExecutionContext to build flow graphs.
        """
        if source_module is not None and source_module is not self:
            key = id(source_module)
            if key not in self._flow_dep_seen:
                self._flow_dep_seen.add(key)
                self._flow_dependencies.append(source_module)

    def _record_data_edge(self, source_module: "Module") -> None:
        """
        [INTERNAL] Record that this module received a DataPoint from source_module.

        Power User Note: Automatically called by the ExecutionContext to build data graphs.
        """
        if source_module is not None and source_module is not self:
            key = id(source_module)
            if key not in self._data_dep_seen:
                self._data_dep_seen.add(key)
                self._data_dependencies.append(source_module)

    def variables(self) -> Iterator[Variable]:
        """
        Recursively yield all variables owned by this module and its sub-modules.

        Returns:
            Iterator[Variable]: An iterator over all unique variables in the hierarchy.
        """
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
        """
        Recursively yield this module and all nested sub-modules.

        Returns:
            Iterator[Module]: An iterator over the module hierarchy.
        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        """
        Recursively yield `(path, module)` pairs using PyTorch-style names.

        Args:
            prefix (str): The prefix to prepend to the paths.

        Returns:
            Iterator[tuple[str, Module]]: An iterator of `(path, module)` tuples.
        """
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
        """
        [INTERNAL] Zero out rates and remove thresholds for all Levels before the next rate update.
        """
        for var in self.variables():
            if isinstance(var, Level):
                var._rate = 0.0
                var.upper_threshold = math.inf
                var.lower_threshold = -math.inf

    def initialize_state(self) -> None:
        """
        Override this to set up initial state before the simulation starts.

        This is called once by the engine before the first time step.
        """
        pass

    def is_terminating_condition_met(self) -> bool:
        """
        Override this to define custom stopping conditions.

        Returns:
            bool: True if the simulation should stop, False otherwise.
        """
        return False

    def state_dict(self, prefix: str = "") -> dict[str, Any]:
        """
        Returns a dictionary containing the entire state of the module.
        Keys are dotted paths to variable values (e.g., 'submodule.variable.value').
        """
        state = {}
        for name, var in self._variables.items():
            key_base = f"{prefix}.{name}" if prefix else name
            state[f"{key_base}.value"] = var.value

        for name, module in self._modules.items():
            module_prefix = f"{prefix}.{name}" if prefix else name
            state.update(module.state_dict(prefix=module_prefix))

        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Copies state from state_dict into this module and its descendants.
        """
        for key, value in state_dict.items():
            if not key.endswith(".value"):
                continue
            key_base = key[:-6]  # remove .value
            parts = key_base.split(".")
            var_name = parts[-1]
            mod_path = parts[:-1]

            # Navigate to the correct module
            current_mod = self
            try:
                for part in mod_path:
                    current_mod = current_mod._modules[part]
                current_mod._variables[var_name].value = value
            except KeyError:
                import logging

                logging.getLogger(__name__).warning(
                    f"Key '{key}' found in state_dict but not in module hierarchy."
                )

    def to_dict(self, root: Optional["Module"] = None) -> dict[str, Any]:
        """
        Returns a structural JSON-serializable representation of the module architecture.
        """
        from .variables import Level, Expression, serialize_val

        if root is None:
            root = self

        # TODO: Why is expresison not included in serialize_val?
        def _to_dict_serialize_val(val: Any) -> Any:
            if isinstance(val, Expression):
                return {"equation": val.get_equation()}
            return serialize_val(val)

        def get_module_path(rt: Module, target: Module) -> str:
            if target is rt:
                return ""
            for path, mod in rt.named_modules():
                if mod is target:
                    return path
            return getattr(target, "name", type(target).__name__)

        children = {}
        for name, mod in self._modules.items():
            children[name] = mod.to_dict(root)

        variables = {}
        for name, var in self._variables.items():
            var_info = {
                "class": type(var).__name__,
                "value": _to_dict_serialize_val(var._value),
            }
            if isinstance(var, Level):
                var_info["rate"] = _to_dict_serialize_val(var._rate)
                var_info["lower_threshold"] = _to_dict_serialize_val(
                    var.lower_threshold
                )
                var_info["upper_threshold"] = _to_dict_serialize_val(
                    var.upper_threshold
                )
            variables[name] = var_info

        layout = getattr(self, "layout", getattr(self, "metadata", {}))
        if not isinstance(layout, dict):
            layout = {"value": str(layout)}

        attributes = {}
        import json
        from .variables import Variable

        # Exclude reserved framework attributes
        RESERVED_KEYS = {
            "parent",
            "config",
            "telemetry",
            "layout",
            "global_time",
        }

        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k in RESERVED_KEYS:
                continue
            if isinstance(v, (Module, Variable)):
                continue
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                try:
                    json.dumps(v)
                    attributes[k] = v
                except Exception:
                    pass

        flow_inputs = []
        for src in self._flow_dependencies:
            flow_inputs.append(get_module_path(root, src))

        data_inputs = []
        for src in self._data_dependencies:
            data_inputs.append(get_module_path(root, src))

        variable_reads = []
        for src_mod, var in self._dependencies:
            variable_reads.append(
                {"module": get_module_path(root, src_mod), "variable": var.name}
            )

        connections = {
            "flow_inputs": flow_inputs,
            "data_inputs": data_inputs,
            "variable_reads": variable_reads,
        }

        return {
            "class": type(self).__name__,
            "layout": layout,
            "variables": variables,
            "attributes": attributes,
            "children": children,
            "connections": connections,
        }

    def register_post_step_hook(self, hook_fn: Any) -> None:
        """Registers a callback to be run after every engine step."""
        self._post_step_hooks.append(hook_fn)

    def _run_post_step_hooks(self, current_time: float) -> None:
        """
        [INTERNAL] Execute all registered post-step callback functions.

        Power User Note: Called automatically by the DRSEngine after time integration.
        """
        for hook in self._post_step_hooks:
            hook(current_time)

    def get_dependency_graph(self) -> list:
        """
        Get all recorded read dependencies from this module and all sub-modules.

        Returns:
            list[tuple[Module, Variable]]: A list of `(source_module, variable)` pairs
                representing cross-module reads.
        """
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
        """Initialize the DataSource."""
        super().__init__()

    def __iter__(self) -> Iterator[DataPoint]:
        """Return the iterator object itself."""
        return self

    def __next__(self) -> DataPoint:
        """Yield the next DataPoint in the sequence."""
        raise StopIteration

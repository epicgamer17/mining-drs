import json
import math
import random
from typing import Any, Optional, Union
from .module import Module


def save_state(model: Module, filepath: str) -> None:
    """Save a module's state to a JSON file."""
    state = model.state_dict()
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2)


def load_state(model: Module, filepath: str) -> None:
    """Load a module's state from a JSON file."""
    with open(filepath, "r") as f:
        state = json.load(f)
    model.load_state_dict(state)


def export_architecture(model: Module, filepath: str) -> None:
    """Export the module's architecture to a JSON file."""
    arch = model.to_dict()
    with open(filepath, "w") as f:
        json.dump(arch, f, indent=2)


def validate_canvas_json(canvas_data: Union[dict, list], model: Module) -> None:
    """Validate that incoming canvas JSON arrays or dictionaries perfectly match
    the properties, variable boundaries, and hierarchies expected by the backend modules.

    Raises:
        ValueError: If there is a structural/hierarchy mismatch or boundary violation.
    """
    if isinstance(canvas_data, list):
        # Convert flat canvas array of nodes to a hierarchical dictionary
        json_tree = _flat_canvas_to_tree(canvas_data)
    elif isinstance(canvas_data, dict):
        json_tree = canvas_data
    else:
        raise ValueError("canvas_data must be a dictionary or a list of dictionaries.")

    _validate_tree_structure(json_tree, model)


def _flat_canvas_to_tree(flat_list: list[dict[str, Any]]) -> dict[str, Any]:
    # Group by path/id
    nodes_by_path = {}
    for node in flat_list:
        path = node.get("path", node.get("id"))
        if path is None:
            raise ValueError(
                "Canvas JSON array elements must have a 'path' or 'id' property."
            )
        # If ID is "root" or similar, map to empty string for consistency
        if path == "root":
            path = ""
        nodes_by_path[path] = node

    # Reconstruct tree starting from sorted paths by segment depth
    sorted_paths = sorted(
        nodes_by_path.keys(), key=lambda p: len(p.split(".")) if p else 0
    )

    tree = {}
    if "" not in nodes_by_path:
        raise ValueError(
            "Flat canvas list must include a root node with path/id equal to '' or 'root'."
        )

    for path in sorted_paths:
        node = nodes_by_path[path]
        node_copy = {
            "class": node.get("class"),
            "layout": node.get("layout", {}),
            "variables": node.get("variables", {}),
            "children": {},
            "connections": node.get("connections", {}),
        }

        if not path:
            tree = node_copy
        else:
            parts = path.split(".")
            parent_parts = parts[:-1]
            child_name = parts[-1]

            parent_node = tree
            for part in parent_parts:
                if "children" not in parent_node:
                    parent_node["children"] = {}
                if part not in parent_node["children"]:
                    parent_node["children"][part] = {}
                parent_node = parent_node["children"][part]

            if "children" not in parent_node:
                parent_node["children"] = {}
            parent_node["children"][child_name] = node_copy

    return tree


def _validate_tree_structure(
    json_node: dict, current_mod: Module, path: str = ""
) -> None:
    # 1. Validate class
    expected_class = type(current_mod).__name__
    actual_class = json_node.get("class")
    if not actual_class:
        raise ValueError(f"Missing 'class' field at path '{path or 'root'}'.")
    if actual_class != expected_class:
        raise ValueError(
            f"Structural mismatch at '{path or 'root'}': class names do not match. "
            f"Expected: {expected_class}, Got: {actual_class}"
        )

    # 2. Validate variables
    json_vars = json_node.get("variables", {})
    for var_name, var in current_mod._variables.items():
        if var_name not in json_vars:
            raise ValueError(
                f"Structural mismatch at '{path or 'root'}': expected variable '{var_name}' is missing."
            )
        var_data = json_vars[var_name]
        if isinstance(var_data, str):
            # Simple type check matching standard _validate_structure
            if var_data != type(var).__name__:
                raise ValueError(
                    f"Variable type mismatch at '{path or 'root'}.{var_name}': "
                    f"Expected: {type(var).__name__}, Got: {var_data}"
                )
        elif isinstance(var_data, dict):
            # Detailed verification
            var_class = var_data.get("class", var_data.get("type"))
            if var_class and var_class != type(var).__name__:
                raise ValueError(
                    f"Variable type mismatch at '{path or 'root'}.{var_name}': "
                    f"Expected: {type(var).__name__}, Got: {var_class}"
                )

            # Validate boundary/thresholds
            val = var_data.get("value", var_data.get("initial_value"))
            if val is not None and not isinstance(val, dict):  # skip equation dicts
                lower = var_data.get(
                    "lower_threshold", getattr(var, "lower_threshold", -math.inf)
                )
                upper = var_data.get(
                    "upper_threshold", getattr(var, "upper_threshold", math.inf)
                )

                if isinstance(lower, str):
                    lower = _deserialize_val(lower)
                if isinstance(upper, str):
                    upper = _deserialize_val(upper)
                if isinstance(val, str):
                    val = _deserialize_val(val)

                if (
                    isinstance(val, (int, float))
                    and isinstance(lower, (int, float))
                    and isinstance(upper, (int, float))
                ):
                    if val < lower or val > upper:
                        raise ValueError(
                            f"Boundary violation for variable '{var_name}' at path '{path or 'root'}': "
                            f"value {val} is outside boundaries [{lower}, {upper}]."
                        )

    # 3. Validate children
    curr_children = set(current_mod._modules.keys())
    json_children = json_node.get("children", {})
    if not isinstance(json_children, dict):
        raise ValueError(
            f"Field 'children' must be a dictionary at path '{path or 'root'}'."
        )

    # Check that all expected child modules exist in JSON
    for child_name in curr_children:
        if child_name not in json_children:
            raise ValueError(
                f"Structural mismatch at '{path or 'root'}': expected submodule '{child_name}' is missing."
            )
        sub_path = f"{path}.{child_name}" if path else child_name
        _validate_tree_structure(
            json_children[child_name], current_mod._modules[child_name], sub_path
        )


def _serialize_val(val: Any) -> Any:
    """
    [INTERNAL] Convert floats that are infinity or NaN to string representations.

    Power User Note: Used to ensure valid JSON representation of float limits in checkpoints.
    """
    if isinstance(val, float):
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        elif math.isnan(val):
            return "NaN"
    return val


def _deserialize_val(val: Any) -> Any:
    """
    [INTERNAL] Convert string-represented infinities and NaNs back to float.

    Power User Note: Used to restore float limits when loading checkpoints.
    """
    if val == "Infinity":
        return math.inf
    elif val == "-Infinity":
        return -math.inf
    elif val == "NaN":
        return math.nan
    return val


def _serialize_module_structure(model: Module) -> dict[str, Any]:
    """
    [INTERNAL] Recursively build a structural representation of modules and hooks.

    Power User Note: Used during checkpointing to validate schema structure on resume.
    """

    def _build_struct(mod):
        serialized_hooks = []
        for hook in mod._post_step_hooks:
            if hasattr(hook, "__self__") and hasattr(hook, "__name__"):
                obj = hook.__self__
                serialized_hooks.append(f"{type(obj).__name__}.{hook.__name__}")
            elif hasattr(hook, "__name__"):
                serialized_hooks.append(hook.__name__)
            else:
                serialized_hooks.append(str(hook))

        children = {}
        for name, sub_mod in mod._modules.items():
            children[name] = _build_struct(sub_mod)

        variables = {}
        for name, var in mod._variables.items():
            variables[name] = type(var).__name__

        attributes = {}
        for k, v in mod.__dict__.items():
            if k.startswith("_") or k in (
                "parent",
                "config",
                "telemetry",
                "_variables",
                "_modules",
            ):
                continue
            if isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                try:
                    # check JSON serializability
                    json_str = json.dumps(v)
                    attributes[k] = json.loads(json_str)
                except Exception:
                    pass

        return {
            "class": type(mod).__name__,
            "variables": variables,
            "hooks": serialized_hooks,
            "attributes": attributes,
            "children": children,
        }

    return _build_struct(model)


def _validate_structure(
    current: dict[str, Any], saved: dict[str, Any], path: str = ""
) -> None:
    """
    [INTERNAL] Ensure the module class and variable structure matches the checkpoint.

    Power User Note: Validates compatibility before restoring a saved state/checkpoint.
    """
    if current.get("class") != saved.get("class"):
        raise ValueError(
            f"Structural mismatch at '{path or 'root'}': class names do not match. "
            f"Current: {current.get('class')}, Saved: {saved.get('class')}"
        )
    curr_vars = current.get("variables", {})
    saved_vars = saved.get("variables", {})
    if curr_vars != saved_vars:
        raise ValueError(
            f"Structural mismatch at '{path or 'root'}': variables do not match. "
            f"Current: {curr_vars}, Saved: {saved_vars}"
        )
    curr_children = current.get("children", {})
    saved_children = saved.get("children", {})
    if set(curr_children.keys()) != set(saved_children.keys()):
        raise ValueError(
            f"Structural mismatch at '{path or 'root'}': children submodules do not match. "
            f"Current keys: {list(curr_children.keys())}, Saved keys: {list(saved_children.keys())}"
        )
    for name in curr_children:
        sub_path = f"{path}.{name}" if path else name
        _validate_structure(curr_children[name], saved_children[name], sub_path)


def _restore_module_attributes(mod: Module, saved_struct: dict[str, Any]) -> None:
    """
    [INTERNAL] Recursively restore saved custom module primitive attributes.

    Power User Note: Restores non-Variable instance fields from checkpoint structure.
    """
    saved_attrs = saved_struct.get("attributes", {})
    for k, v in saved_attrs.items():
        setattr(mod, k, v)

    curr_children = mod._modules
    saved_children = saved_struct.get("children", {})
    for name, child_mod in curr_children.items():
        if name in saved_children:
            _restore_module_attributes(child_mod, saved_children[name])


def save_checkpoint(engine: Any, filepath: str) -> None:
    """Save the full engine and model state to a JSON file."""
    model = engine.model
    id_to_name = {id(mod): name for name, mod in model.named_modules()}

    # Save variables state
    variables_state = {}
    for name, mod in model.named_modules():
        for var_name, var in mod._variables.items():
            var_path = f"{name}.{var_name}" if name else var_name

            rate_set_by = None
            if getattr(var, "_rate_set_by", None) is not None:
                rate_set_by = id_to_name.get(id(var._rate_set_by))

            var_state = {
                "value": _serialize_val(var._value),
                "rate": _serialize_val(var.rate) if hasattr(var, "rate") else None,
                "upper_threshold": (
                    _serialize_val(var.upper_threshold)
                    if hasattr(var, "upper_threshold")
                    else None
                ),
                "lower_threshold": (
                    _serialize_val(var.lower_threshold)
                    if hasattr(var, "lower_threshold")
                    else None
                ),
                "_rate_set_by": rate_set_by,
            }
            variables_state[var_path] = var_state

    # Python RNG state
    python_rng = list(random.getstate())
    # Convert state tuple element to list
    python_rng[1] = list(python_rng[1])

    # Numpy RNG state
    numpy_rng = None
    try:
        import numpy as np

        np_state = np.random.get_state()
        numpy_rng = [
            np_state[0],
            np_state[1].tolist(),
            np_state[2],
            np_state[3],
            np_state[4],
        ]
    except ImportError:
        pass

    # Telemetry data
    telemetry_data = None
    if engine.telemetry is not None:
        serialized_history = []
        for entry in engine.telemetry.history:
            new_entry = {}
            for k, v in entry.items():
                new_entry[k] = _serialize_val(v)
            serialized_history.append(new_entry)

        serialized_events = []
        for e in engine.telemetry.events:
            new_details = {}
            for k, v in e.details.items():
                new_details[k] = _serialize_val(v)
            serialized_events.append(
                {
                    "time": e.time,
                    "event_type": e.event_type,
                    "source": e.source,
                    "details": new_details,
                }
            )

        telemetry_data = {
            "history": serialized_history,
            "events": serialized_events,
            "event_log_cursor": len(engine.telemetry.events),
            "history_cursor": len(engine.telemetry.history),
        }

    checkpoint = {
        "drs_version": "1.0",
        "engine": {
            "current_time": engine.current_time,
            "step_count": getattr(engine, "step_count", 0),
            "_consecutive_zero_dt_count": getattr(
                engine, "_consecutive_zero_dt_count", 0
            ),
            "rng": {"python": python_rng, "numpy": numpy_rng},
            "telemetry": telemetry_data,
        },
        "model_structure": _serialize_module_structure(model),
        "variables_state": variables_state,
    }

    with open(filepath, "w") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(engine: Any, filepath: str) -> None:
    """Load the full engine and model state from a JSON file."""
    with open(filepath, "r") as f:
        checkpoint = json.load(f)

    model = engine.model

    # 1. Validate structure
    current_structure = _serialize_module_structure(model)
    _validate_structure(current_structure, checkpoint["model_structure"])

    # 2. Restore module attributes
    _restore_module_attributes(model, checkpoint["model_structure"])

    # 3. Restore variables
    name_to_mod = {name: mod for name, mod in model.named_modules()}
    variables_state = checkpoint["variables_state"]
    for var_path, var_state in variables_state.items():
        parts = var_path.split(".")
        var_name = parts[-1]
        mod_path = ".".join(parts[:-1])

        mod = name_to_mod.get(mod_path)
        if mod is None:
            continue

        var = mod._variables.get(var_name)
        if var is None:
            continue

        var._value = _deserialize_val(var_state["value"])
        if hasattr(var, "rate") and var_state["rate"] is not None:
            var._rate = _deserialize_val(var_state["rate"])
        if hasattr(var, "upper_threshold") and var_state["upper_threshold"] is not None:
            var.upper_threshold = _deserialize_val(var_state["upper_threshold"])
        if hasattr(var, "lower_threshold") and var_state["lower_threshold"] is not None:
            var.lower_threshold = _deserialize_val(var_state["lower_threshold"])

        # Restore _rate_set_by reference
        rate_set_by_name = var_state.get("_rate_set_by")
        if rate_set_by_name is not None:
            var._rate_set_by = name_to_mod.get(rate_set_by_name)
        else:
            if hasattr(var, "_rate_set_by"):
                var._rate_set_by = None

    # 4. Restore engine state
    engine_data = checkpoint["engine"]
    engine.current_time = engine_data["current_time"]
    engine.step_count = engine_data["step_count"]
    engine._consecutive_zero_dt_count = engine_data["_consecutive_zero_dt_count"]
    engine._resuming = True

    # 5. Restore RNG states
    rng_data = engine_data["rng"]
    if rng_data.get("python") is not None:
        p_rng = rng_data["python"]
        p_rng_state = (p_rng[0], tuple(p_rng[1]), p_rng[2])
        random.setstate(p_rng_state)

    if rng_data.get("numpy") is not None:
        try:
            import numpy as np

            np_list = rng_data["numpy"]
            np_state = (
                np_list[0],
                np.array(np_list[1], dtype=np.uint32),
                np_list[2],
                np_list[3],
                np_list[4],
            )
            np.random.set_state(np_state)
        except ImportError:
            pass

    # 6. Restore Telemetry
    telemetry_data = engine_data.get("telemetry")
    if telemetry_data is not None and engine.telemetry is not None:
        from .telemetry import Event

        # Restore history
        deserialized_history = []
        for entry in telemetry_data["history"]:
            new_entry = {}
            for k, v in entry.items():
                new_entry[k] = _deserialize_val(v)
            deserialized_history.append(new_entry)
        engine.telemetry.history = deserialized_history

        # Restore events
        deserialized_events = []
        for e in telemetry_data["events"]:
            new_details = {}
            for k, v in e["details"].items():
                new_details[k] = _deserialize_val(v)
            deserialized_events.append(
                Event(
                    time=e["time"],
                    event_type=e["event_type"],
                    source=e["source"],
                    details=new_details,
                )
            )
        engine.telemetry.events = deserialized_events


def compile_canvas_json(
    canvas_data: Union[dict, list],
    class_registry: Optional[dict[str, type]] = None,
    config: Optional[Any] = None,
) -> Module:
    """Compile a canvas JSON configuration into a programmatic Module tree structure with a dynamically-compiled forward pass.

    Args:
        canvas_data: A flat list of canvas node configurations or a hierarchical nested tree dictionary.
        class_registry: A dictionary mapping class names to class types. If not provided, falls back to namespaces in drs and examples.mining.
        config: Simulation config object to propagate to the compiled modules.

    Returns:
        Module: The compiled root module containing all compiled submodules, variables, and links.
    """
    import math
    import drs
    from .module import Module
    from .variables import Variable, Level, Timer
    from .flow import Flow
    from .data_source import DataPoint

    if isinstance(canvas_data, list):
        json_tree = _flat_canvas_to_tree(canvas_data)
    elif isinstance(canvas_data, dict):
        json_tree = canvas_data
    else:
        raise ValueError("canvas_data must be a dictionary or a list of dictionaries.")

    def _is_dummy_forward(func) -> bool:
        if func is None:
            return True
        try:
            import inspect
            import dis

            if hasattr(func, "__func__"):
                func = func.__func__
            insts = list(dis.get_instructions(func))
            opnames = [inst.opname for inst in insts]
            has_calls = any(name.startswith("CALL") for name in opnames)
            has_stores = any(name.startswith("STORE") for name in opnames)
            if not has_calls and not has_stores:
                return True
            if "RAISE_VARARGS" in opnames:
                return True
        except Exception:
            pass
        return False

    # Helper to resolve class type
    def _resolve_class(class_name: str) -> type:
        if class_registry and class_name in class_registry:
            return class_registry[class_name]
        # Check drs namespace
        if hasattr(drs, class_name):
            return getattr(drs, class_name)
        # Check examples.mining.components namespaces
        try:
            import examples.mining.components as mc

            if hasattr(mc, class_name):
                return getattr(mc, class_name)
            import examples.mining.components.models as mm

            if hasattr(mm, class_name):
                return getattr(mm, class_name)
            import examples.mining.components.stockpiles as ms

            if hasattr(ms, class_name):
                return getattr(ms, class_name)
            import examples.mining.components.mine_face as mf

            if hasattr(mf, class_name):
                return getattr(mf, class_name)
            import examples.mining.components.fleet as mfl

            if hasattr(mfl, class_name):
                return getattr(mfl, class_name)
            import examples.mining.components.plant as mp

            if hasattr(mp, class_name):
                return getattr(mp, class_name)
            import examples.mining.components.controllers as mc_ctrl

            if hasattr(mc_ctrl, class_name):
                return getattr(mc_ctrl, class_name)
        except ImportError:
            pass

        import sys

        for module in list(sys.modules.values()):
            if module and hasattr(module, class_name):
                return getattr(module, class_name)

        # Default fallback to Module if class is not found
        return Module

    # Recursive builder
    # Stores equations to resolve after compiling the entire tree
    equations_to_resolve = []

    # We will also keep track of node JSON configs to construct forward passes later
    nodes_info = {}  # path -> json node copy

    def build_module(
        name: str,
        node_data: dict,
        path: str = "",
        existing_obj: Optional[Module] = None,
    ) -> Module:
        cls_name = node_data.get("class", "Module")
        cls = _resolve_class(cls_name)

        if existing_obj is not None:
            obj = existing_obj
        else:
            # Instantiate object using smart signature inspection
            import inspect

            try:
                # Check the signature of the class constructor
                sig = inspect.signature(cls.__init__)
                params = list(sig.parameters.values())
                # Exclude self
                params_no_self = [p for p in params if p.name != "self"]

                # Check if second parameter is config/cfg or if parameter type name contains config
                has_config_param = False
                if params_no_self:
                    first_param = params_no_self[0]
                    if first_param.name in ("config", "cfg") or "Config" in str(
                        first_param.annotation
                    ):
                        has_config_param = True

                if has_config_param and config is not None:
                    obj = cls(config)
                else:
                    # Try to map constructor parameters to attributes from JSON
                    kwargs = {}
                    has_all_required = True
                    attrs = node_data.get("attributes", {})

                    for param in params_no_self:
                        if param.name in attrs:
                            kwargs[param.name] = attrs[param.name]
                        elif param.default == inspect.Parameter.empty:
                            # Required parameter is missing from attributes
                            has_all_required = False
                            break

                    if has_all_required:
                        obj = cls(**kwargs)
                    else:
                        # Fallback to default/empty constructor
                        obj = cls()
            except Exception:
                # Fallback to allocating object via __new__ if constructor signature is custom/incompatible
                obj = cls.__new__(cls)
                Module.__init__(obj)
                if config is not None:
                    obj.config = config

        # Restore layout/metadata
        if "layout" in node_data:
            obj.layout = node_data["layout"]

        # Restore attributes
        attrs = node_data.get("attributes", {})
        for k, v in attrs.items():
            setattr(obj, k, v)

        current_path = f"{path}.{name}" if path else name
        nodes_info[current_path] = node_data

        # Restore variables
        variables_data = node_data.get("variables", {})
        for var_name, var_info in variables_data.items():
            if isinstance(var_info, str):
                var_info = {"class": var_info, "value": 0.0}

            var_class_name = var_info.get("class", "Variable")
            var_value = var_info.get("value", 0.0)

            # De-serialize NaN or Infinity string values
            if isinstance(var_value, str):
                var_value = _deserialize_val(var_value)

            var_cls = getattr(drs, var_class_name, Variable)

            # If variable already exists on obj, reuse it and update value
            existing_var = getattr(obj, var_name, None)
            if (
                isinstance(existing_var, Variable)
                and type(existing_var).__name__ == var_class_name
            ):
                var = existing_var
                import enum

                if isinstance(var._value, enum.Enum) and isinstance(var_value, str):
                    try:
                        enum_cls = type(var._value)
                        var._value = enum_cls[var_value]
                    except KeyError:
                        var._value = var_value
                elif (
                    hasattr(var._value, "name")
                    and hasattr(var._value, "id")
                    and isinstance(var_value, str)
                ):
                    try:
                        enum_cls = type(var._value)
                        var._value = enum_cls(var_value)
                    except Exception:
                        var._value = var_value
                else:
                    var._value = var_value if not isinstance(var_value, dict) else 0.0
            else:
                # Resolve level/timer values vs rate
                if var_cls in (Level, Timer):
                    var = var_cls(
                        var_name,
                        initial_value=(
                            var_value if not isinstance(var_value, dict) else 0.0
                        ),
                    )
                else:
                    var = var_cls(var_name, var_value)
                setattr(obj, var_name, var)

            if var_cls in (Level, Timer):
                # Bounds
                if "lower_threshold" in var_info:
                    var.lower_threshold = _deserialize_val(var_info["lower_threshold"])
                if "upper_threshold" in var_info:
                    var.upper_threshold = _deserialize_val(var_info["upper_threshold"])
                # Rate
                rate_val = var_info.get("rate", 0.0)
                if isinstance(rate_val, dict) and "equation" in rate_val:
                    # Save equation to resolve later
                    equations_to_resolve.append(
                        (var, rate_val["equation"], current_path)
                    )
                elif isinstance(rate_val, str) and rate_val not in (
                    "Infinity",
                    "-Infinity",
                    "NaN",
                ):
                    equations_to_resolve.append((var, rate_val, current_path))
                else:
                    var.rate = _deserialize_val(rate_val)

        # Restore children submodules
        children_data = node_data.get("children", {})
        for child_name, child_data in children_data.items():
            existing_child = getattr(obj, child_name, None)
            if isinstance(existing_child, Module):
                build_module(
                    child_name, child_data, current_path, existing_obj=existing_child
                )
            else:
                child_obj = build_module(child_name, child_data, current_path)
                setattr(obj, child_name, child_obj)

        return obj

    # Compile the tree
    root_name = ""
    root = build_module(root_name, json_tree, "")

    # Set root parent to None
    root.parent = None

    # Resolve equation strings
    # We build a local namespace context for each module containing:
    # - self (resolves to the module)
    # - sibling submodules (resolves to siblings)
    # - parent submodules if available
    name_to_mod = {path: mod for path, mod in root.named_modules()}
    name_to_mod[""] = root

    from ._execution_context import ExecutionContext

    orig_tracing = ExecutionContext.is_tracing()
    ExecutionContext.set_tracing(True)
    try:
        for var, eq_str, mod_path in equations_to_resolve:
            mod = name_to_mod[mod_path]

            # Build local namespace
            local_ns = {"self": mod}

            # Add siblings
            for sib_name, sib_mod in mod._modules.items():
                local_ns[sib_name] = sib_mod

            # Add parent and parent's siblings recursively to allow relative reads
            p = mod.parent
            while p is not None:
                # Add parent direct siblings
                for sib_name, sib_mod in p._modules.items():
                    local_ns[sib_name] = sib_mod
                p = p.parent

            # Strip any leading/trailing parentheses from Equation representation
            cleaned_eq = eq_str.strip()
            if cleaned_eq.startswith("(") and cleaned_eq.endswith(")"):
                cleaned_eq = cleaned_eq[1:-1]

            try:
                # Evaluate using overloaded operators on Variable instances
                # to build Expression AST tree
                expr = eval(cleaned_eq, {}, local_ns)
                var.rate = expr
            except Exception as e:
                # Fallback to float rate 0.0 or raise warning
                import logging

                logging.getLogger(__name__).warning(
                    f"Could not resolve rate equation '{eq_str}' for variable '{var.name}' at path '{mod_path}': {e}"
                )
                var.rate = 0.0
    finally:
        ExecutionContext.set_tracing(orig_tracing)

    # Build connection dependencies & dynamic forward pass for all modules
    for mod_path, mod in name_to_mod.items():
        node_info = nodes_info.get(mod_path, {})
        conns = node_info.get("connections", {})

        # Reconstruct connection references
        flow_inputs = conns.get("flow_inputs", [])
        data_inputs = conns.get("data_inputs", [])
        variable_reads = conns.get("variable_reads", [])

        # Restore dependencies
        for src_path in flow_inputs:
            src_mod = name_to_mod.get(src_path)
            if src_mod:
                mod._record_flow_edge(src_mod)

        for src_path in data_inputs:
            src_mod = name_to_mod.get(src_path)
            if src_mod:
                mod._record_data_edge(src_mod)

        for r_info in variable_reads:
            src_path = r_info.get("module")
            var_name = r_info.get("variable")
            src_mod = name_to_mod.get(src_path)
            if src_mod and var_name in src_mod._variables:
                mod._record_incoming_edge(src_mod._variables[var_name])

        # If this module has children submodules, and has a dummy forward pass, build its dynamic forward()
        if mod._modules and _is_dummy_forward(type(mod).forward):
            # Topological sort of children submodules
            children_names = list(mod._modules.keys())

            # Map child name -> full child path
            child_to_info = {}
            for name in children_names:
                full_path = f"{mod_path}.{name}" if mod_path else name
                child_to_info[name] = nodes_info.get(full_path, {})

            # Helper to run topological sort on children of the current module
            adj = {name: set() for name in children_names}
            in_degree = {name: 0 for name in children_names}

            for name in children_names:
                info = child_to_info[name]
                child_conns = info.get("connections", {})
                sources = child_conns.get("flow_inputs", []) + child_conns.get(
                    "data_inputs", []
                )

                for src in sources:
                    # Find if source is sibling (by comparing relative suffix from parent path)
                    src_suffix = src
                    if mod_path:
                        if src.startswith(mod_path + "."):
                            src_suffix = src[len(mod_path) + 1 :]

                    if src_suffix in adj:
                        if name not in adj[src_suffix]:
                            adj[src_suffix].add(name)
                            in_degree[name] += 1

            queue = [name for name, deg in in_degree.items() if deg == 0]
            queue.sort()

            execution_order = []
            while queue:
                curr = queue.pop(0)
                execution_order.append(curr)
                for neighbor in adj[curr]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            if len(execution_order) < len(children_names):
                # Cycle fallback
                remaining = [
                    name for name in children_names if name not in execution_order
                ]
                execution_order.extend(remaining)

            # Build and bind dynamic forward pass
            def make_forward(order, child_to_info_map, m_path):
                def forward(self, *args, **kwargs):
                    outputs = {}

                    for child_name in order:
                        child_mod = getattr(self, child_name)
                        info = child_to_info_map[child_name]
                        c_conns = info.get("connections", {})

                        flow_ins = c_conns.get("flow_inputs", [])
                        data_ins = c_conns.get("data_inputs", [])

                        inputs = []
                        for src in flow_ins:
                            # Convert absolute source path to relative sibling name
                            src_rel = src
                            if m_path:
                                if src.startswith(m_path + "."):
                                    src_rel = src[len(m_path) + 1 :]
                            inputs.append(outputs.get(src_rel))

                        for src in data_ins:
                            src_rel = src
                            if m_path:
                                if src.startswith(m_path + "."):
                                    src_rel = src[len(m_path) + 1 :]
                            inputs.append(outputs.get(src_rel))

                        # Execute and record
                        res = child_mod(*inputs)
                        outputs[child_name] = res

                    # Return output of the last executed node
                    if order:
                        return outputs.get(order[-1])
                    return None

                return forward

            # Bind the function to this instance
            import types

            mod.forward = types.MethodType(
                make_forward(execution_order, child_to_info, mod_path), mod
            )

    # Re-apply rate expressions in forward pass to prevent _zero_rates from erasing them
    from .variables import Expression

    for mod_path, mod in name_to_mod.items():
        expr_vars = []
        for var_name, var in mod._variables.items():
            if isinstance(var, Level) and isinstance(var._rate, Expression):
                expr_vars.append((var, var._rate))

        if expr_vars:
            # Wrap forward
            orig_forward = mod.forward

            def make_wrapped_forward(orig_f, evars):
                def wrapped_forward(self, *args, **kwargs):
                    res = orig_f(*args, **kwargs)
                    for var, expr in evars:
                        # Re-apply rate expression
                        var.rate = expr
                    return res

                return wrapped_forward

            import types

            mod.forward = types.MethodType(
                make_wrapped_forward(orig_forward, expr_vars), mod
            )

    return root

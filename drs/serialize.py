import json
import math
import random
from typing import Any, Optional
from .module import Module

def save_state(model: Module, filepath: str) -> None:
    """Save a module's state to a JSON file."""
    state = model.state_dict()
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)

def load_state(model: Module, filepath: str) -> None:
    """Load a module's state from a JSON file."""
    with open(filepath, 'r') as f:
        state = json.load(f)
    model.load_state_dict(state)

def export_architecture(model: Module, filepath: str) -> None:
    """Export the module's architecture to a JSON file."""
    arch = model.to_dict()
    with open(filepath, 'w') as f:
        json.dump(arch, f, indent=2)


def _serialize_val(val: Any) -> Any:
    """Helper to convert floats that are infinity or NaN to string representations."""
    if isinstance(val, float):
        if math.isinf(val):
            return "Infinity" if val > 0 else "-Infinity"
        elif math.isnan(val):
            return "NaN"
    return val


def _deserialize_val(val: Any) -> Any:
    """Helper to convert string-represented infinities and NaNs back to float."""
    if val == "Infinity":
        return math.inf
    elif val == "-Infinity":
        return -math.inf
    elif val == "NaN":
        return math.nan
    return val


def _serialize_module_structure(model: Module) -> dict[str, Any]:
    """Helper to recursively build a structural representation of modules and hooks."""
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
            if k.startswith('_') or k in ('parent', 'config', 'telemetry', '_variables', '_modules'):
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
            "children": children
        }
        
    return _build_struct(model)


def _validate_structure(current: dict[str, Any], saved: dict[str, Any], path: str = "") -> None:
    """Helper to ensure the module class and variable structure matches the checkpoint."""
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
    """Helper to recursively restore saved custom module primitive attributes."""
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
                "upper_threshold": _serialize_val(var.upper_threshold) if hasattr(var, "upper_threshold") else None,
                "lower_threshold": _serialize_val(var.lower_threshold) if hasattr(var, "lower_threshold") else None,
                "_rate_set_by": rate_set_by
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
            np_state[4]
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
            serialized_events.append({
                "time": e.time,
                "event_type": e.event_type,
                "source": e.source,
                "details": new_details
            })
            
        telemetry_data = {
            "history": serialized_history,
            "events": serialized_events,
            "event_log_cursor": len(engine.telemetry.events),
            "history_cursor": len(engine.telemetry.history)
        }
        
    checkpoint = {
        "drs_version": "1.0",
        "engine": {
            "current_time": engine.current_time,
            "step_count": getattr(engine, "step_count", 0),
            "_consecutive_zero_dt_count": getattr(engine, "_consecutive_zero_dt_count", 0),
            "rng": {
                "python": python_rng,
                "numpy": numpy_rng
            },
            "telemetry": telemetry_data
        },
        "model_structure": _serialize_module_structure(model),
        "variables_state": variables_state
    }
    
    with open(filepath, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(engine: Any, filepath: str) -> None:
    """Load the full engine and model state from a JSON file."""
    with open(filepath, 'r') as f:
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
        parts = var_path.split('.')
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
                np_list[4]
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
                    details=new_details
                )
            )
        engine.telemetry.events = deserialized_events

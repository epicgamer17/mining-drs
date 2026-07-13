import math
import types
from typing import Any, Optional, Union

import drs
from .module import Module
from .variables import Variable, Level, Timer, Expression, deserialize_val
from ._execution_context import ExecutionContext

CLASS_REGISTRY: dict[str, type] = {}


def register_class(cls: type) -> None:
    CLASS_REGISTRY[cls.__name__] = cls


def _flat_canvas_to_tree(flat_list: list[dict[str, Any]]) -> dict[str, Any]:
    nodes_by_path = {}
    for node in flat_list:
        path = node.get("path", node.get("id"))
        if path is None:
            raise ValueError(
                "Canvas JSON array elements must have a 'path' or 'id' property."
            )
        if path == "root":
            path = ""
        nodes_by_path[path] = node

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
            "attributes": node.get("attributes", {}),
            "connections": node.get("connections", {}),
            "rate_hooks": node.get("rate_hooks", {}),
            "termination_condition": node.get("termination_condition"),
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


def _resolve_class(
    class_name: str, class_registry: Optional[dict[str, type]] = None
) -> type:
    if class_registry and class_name in class_registry:
        return class_registry[class_name]
    if class_name in CLASS_REGISTRY:
        return CLASS_REGISTRY[class_name]
    if hasattr(drs, class_name):
        return getattr(drs, class_name)
    try:
        import drs_mining.components as mc

        if hasattr(mc, class_name):
            return getattr(mc, class_name)
    except ImportError:
        pass
    if class_name == "DRSModel":
        return Module
    raise ValueError(
        f"Class '{class_name}' not found in class_registry, CLASS_REGISTRY, "
        "or drs namespace. Register it with canvas_compiler.register_class() "
        "or pass it via class_registry."
    )


def validate_canvas_json(canvas_data: Union[dict, list], model: Module) -> None:
    if isinstance(canvas_data, list):
        json_tree = _flat_canvas_to_tree(canvas_data)
    elif isinstance(canvas_data, dict):
        json_tree = canvas_data
    else:
        raise ValueError("canvas_data must be a dictionary or a list of dictionaries.")

    def _validate(json_node: dict, mod: Module, path: str = "") -> None:
        _p = path or "root"
        expected_class = type(mod).__name__
        actual_class = json_node.get("class")
        if not actual_class:
            raise ValueError(f"Missing 'class' field at path '{_p}'.")
        if actual_class != expected_class:
            raise ValueError(
                f"Structural mismatch at '{_p}': class names do not match. "
                f"Expected: {expected_class}, Got: {actual_class}"
            )

        json_vars = json_node.get("variables", {})
        for var_name, var in mod._variables.items():
            if var_name not in json_vars:
                raise ValueError(
                    f"Structural mismatch at '{_p}': expected variable '{var_name}' is missing."
                )
            var_data = json_vars[var_name]
            if isinstance(var_data, str):
                if var_data != type(var).__name__:
                    raise ValueError(
                        f"Variable type mismatch at '{_p}.{var_name}': "
                        f"Expected: {type(var).__name__}, Got: {var_data}"
                    )
            elif isinstance(var_data, dict):
                var_class = var_data.get("class", var_data.get("type"))
                if var_class and var_class != type(var).__name__:
                    raise ValueError(
                        f"Variable type mismatch at '{_p}.{var_name}': "
                        f"Expected: {type(var).__name__}, Got: {var_class}"
                    )
                val = var_data.get("value", var_data.get("initial_value"))
                if val is not None and not isinstance(val, dict):
                    lower = var_data.get(
                        "lower_threshold", getattr(var, "lower_threshold", -math.inf)
                    )
                    upper = var_data.get(
                        "upper_threshold", getattr(var, "upper_threshold", math.inf)
                    )
                    if isinstance(lower, str):
                        lower = deserialize_val(lower)
                    if isinstance(upper, str):
                        upper = deserialize_val(upper)
                    if isinstance(val, str):
                        val = deserialize_val(val)
                    if (
                        isinstance(val, (int, float))
                        and isinstance(lower, (int, float))
                        and isinstance(upper, (int, float))
                    ):
                        if val < lower or val > upper:
                            raise ValueError(
                                f"Boundary violation for variable '{var_name}' at path '{_p}': "
                                f"value {val} is outside boundaries [{lower}, {upper}]."
                            )

        curr_children = set(mod._modules.keys())
        json_children = json_node.get("children", {})
        if not isinstance(json_children, dict):
            raise ValueError(f"Field 'children' must be a dictionary at path '{_p}'.")
        for child_name in curr_children:
            if child_name not in json_children:
                raise ValueError(
                    f"Structural mismatch at '{_p}': expected submodule '{child_name}' is missing."
                )
            sub_path = f"{_p}.{child_name}" if path else child_name
            _validate(json_children[child_name], mod._modules[child_name], sub_path)

    _validate(json_tree, model)


def _instantiate_class(
    cls: type,
    name: str,
    attrs: dict,
    config: Optional[Any],
    parent_obj: Optional[Module],
) -> Module:
    import inspect

    try:
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        kwargs = {}
        for param in params:
            if param.name in ("config", "cfg") and config is not None:
                kwargs[param.name] = config
            elif "Config" in str(param.annotation) and config is not None:
                kwargs[param.name] = config
            elif param.name in attrs:
                kwargs[param.name] = attrs[param.name]
            elif parent_obj is not None and param.name in getattr(
                parent_obj, "_modules", {}
            ):
                kwargs[param.name] = parent_obj._modules[param.name]
            elif parent_obj is not None and hasattr(parent_obj, param.name):
                candidate = getattr(parent_obj, param.name)
                if isinstance(candidate, Module):
                    kwargs[param.name] = candidate
                elif param.default == inspect.Parameter.empty:
                    return cls()
            elif param.default == inspect.Parameter.empty:
                return cls()

        obj = cls(**kwargs)
    except Exception:
        obj = cls.__new__(cls)
        Module.__init__(obj)
        if config is not None:
            obj.config = config
    return obj


def _restore_variable(
    obj: Module,
    var_name: str,
    var_info: Any,
    current_path: str,
    class_registry: Optional[dict[str, type]],
    equations: list,
) -> None:
    import enum

    if isinstance(var_info, str):
        var_info = {"class": var_info, "value": 0.0}

    var_class_name = var_info.get("class", "Variable")
    var_value = var_info.get("value", 0.0)

    if isinstance(var_value, dict) and "__type__" in var_value:
        type_name = var_value["__type__"]
        obj_name = var_value.get("name")
        cls = _resolve_class(type_name, class_registry)
        if cls is not None and obj_name is not None:
            try:
                var_value = cls(obj_name)
            except Exception:
                raise ValueError(
                    f"Could not resolve typed object: __type__='{type_name}', "
                    f"name='{obj_name}' for variable '{var_name}' at '{current_path}'."
                )

    if isinstance(var_value, str):
        var_value = deserialize_val(var_value)

    var_cls = getattr(drs, var_class_name, None)
    if var_cls is None:
        raise ValueError(
            f"Variable class '{var_class_name}' not found for variable '{var_name}' at '{current_path}'."
        )

    existing_var = getattr(obj, var_name, None)
    if (
        isinstance(existing_var, Variable)
        and type(existing_var).__name__ == var_class_name
    ):
        var = existing_var
        if isinstance(var._value, enum.Enum) and isinstance(var_value, str):
            if var_value not in type(var._value).__members__:
                raise ValueError(
                    f"Invalid value '{var_value}' for enum {type(var._value).__name__} "
                    f"on variable '{var_name}' at '{current_path}'."
                )
            var._value = type(var._value)[var_value]
        elif (
            hasattr(var._value, "name")
            and hasattr(var._value, "id")
            and isinstance(var_value, str)
        ):
            try:
                var._value = type(var._value)(var_value)
            except Exception:
                raise ValueError(
                    f"Could not construct {type(var._value).__name__} from value '{var_value}' "
                    f"for variable '{var_name}' at '{current_path}'."
                )
        else:
            var._value = var_value if not isinstance(var_value, dict) else 0.0
    else:
        if var_cls in (Level, Timer):
            var = var_cls(
                var_name,
                initial_value=var_value if not isinstance(var_value, dict) else 0.0,
            )
        else:
            var = var_cls(var_name, var_value)
        setattr(obj, var_name, var)

    if var_cls in (Level, Timer):
        if "lower_threshold" in var_info:
            var.lower_threshold = deserialize_val(var_info["lower_threshold"])
        if "upper_threshold" in var_info:
            var.upper_threshold = deserialize_val(var_info["upper_threshold"])
        rate_val = var_info.get("rate", 0.0)
        if isinstance(rate_val, dict) and "equation" in rate_val:
            equations.append((var, rate_val["equation"], current_path))
        elif isinstance(rate_val, str) and rate_val not in (
            "Infinity",
            "-Infinity",
            "NaN",
        ):
            equations.append((var, rate_val, current_path))
        else:
            var.rate = deserialize_val(rate_val)


def _resolve_rate_equations(equations: list, name_to_mod: dict[str, Module]) -> None:
    orig_tracing = ExecutionContext.is_tracing()
    ExecutionContext.set_tracing(True)
    try:
        for var, eq_str, mod_path in equations:
            mod = name_to_mod[mod_path]
            local_ns = {"self": mod}
            for sib_name, sib_mod in mod._modules.items():
                local_ns[sib_name] = sib_mod
            p = mod.parent
            while p is not None:
                for sib_name, sib_mod in p._modules.items():
                    local_ns[sib_name] = sib_mod
                p = p.parent

            cleaned = eq_str.strip()
            if cleaned.startswith("(") and cleaned.endswith(")"):
                cleaned = cleaned[1:-1]

            try:
                var.rate = eval(cleaned, {}, local_ns)
            except Exception as e:
                raise ValueError(
                    f"Could not resolve rate equation '{eq_str}' for variable "
                    f"'{var.name}' at path '{mod_path}': {e}"
                )
    finally:
        ExecutionContext.set_tracing(orig_tracing)


def _record_dependency_edges(name_to_mod: dict[str, Module], nodes_info: dict) -> None:
    for mod_path, mod in name_to_mod.items():
        conns = nodes_info.get(mod_path, {}).get("connections", {})
        for entry in conns.get("flow_inputs", []):
            src = name_to_mod.get(entry["module"])
            if src:
                mod._record_flow_edge(src)
        for entry in conns.get("data_inputs", []):
            src = name_to_mod.get(entry["module"])
            if src:
                mod._record_data_edge(src)
        for r_info in conns.get("variable_reads", []):
            src = name_to_mod.get(r_info["module"])
            if src and r_info["variable"] in src._variables:
                mod._record_incoming_edge(src._variables[r_info["variable"]])


def _is_dummy_forward(func) -> bool:
    try:
        if hasattr(func, "__func__"):
            func = func.__func__
        import dis

        instructions = list(dis.get_instructions(func))
        opnames = [inst.opname for inst in instructions]

        # Detect `raise NotImplementedError(...)` — this is the Module base
        # class's forward() which should be treated as dummy/replaceable.
        for i, inst in enumerate(instructions):
            if inst.opname == "RAISE_VARARGS":
                # Check if the exception being raised is NotImplementedError
                for prev in instructions[:i]:
                    if prev.opname in ("LOAD_GLOBAL", "PUSH_EXC_INFO") and (
                        prev.argval == "NotImplementedError"
                    ):
                        return True
                # Other raise patterns: not dummy
                return False

        has_calls = any(n.startswith("CALL") for n in opnames)
        has_stores = any(n.startswith("STORE") for n in opnames)
        return not has_calls and not has_stores
    except Exception:
        return False


def _topological_sort(children_names: list[str], child_to_info: dict) -> list[str]:
    adj = {name: set() for name in children_names}
    in_degree = {name: 0 for name in children_names}

    for name in children_names:
        for src in child_to_info[name].get("connections", {}).get(
            "flow_inputs", []
        ) + child_to_info[name].get("connections", {}).get("data_inputs", []):
            src_path = src.get("module")
            if src_path and src_path in adj and name not in adj[src_path]:
                adj[src_path].add(name)
                in_degree[name] += 1

    queue = sorted([n for n, d in in_degree.items() if d == 0])
    order = []
    while queue:
        curr = queue.pop(0)
        order.append(curr)
        for neighbor in adj[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    remaining = [n for n in children_names if n not in order]
    order.extend(remaining)
    return order


def _build_dynamic_forward(
    mod: Module,
    mod_path: str,
    children_names: list[str],
    child_to_info: dict,
    restore_rates: list,
) -> types.MethodType:
    order = _topological_sort(children_names, child_to_info)

    def forward(self, *args, **kwargs):
        outputs = {}
        for child_name in order:
            child_mod = getattr(self, child_name)
            c_conns = child_to_info[child_name].get("connections", {})

            child_args = []
            child_kwargs = {}
            for conn in c_conns.get("flow_inputs", []) + c_conns.get("data_inputs", []):
                src_rel = conn["module"]
                src_output = outputs.get(src_rel)
                variable = conn.get("variable")
                output_index = conn.get("output_index")

                if variable:
                    src = getattr(self, src_rel, None) if src_rel else None
                    val = (
                        getattr(src, variable)
                        if src and hasattr(src, variable)
                        else src_output
                    )
                elif output_index is not None and isinstance(src_output, tuple):
                    val = (
                        src_output[output_index]
                        if output_index < len(src_output)
                        else None
                    )
                else:
                    val = src_output

                if conn.get("param"):
                    child_kwargs[conn["param"]] = val
                else:
                    child_args.append(val)

            res = child_mod(*child_args, **child_kwargs)
            if isinstance(res, tuple):
                outputs[child_name] = res
                for idx, element in enumerate(res):
                    outputs[f"{child_name}.{idx}"] = element
            else:
                outputs[child_name] = res

        # Apply rate_hooks from node data
        rate_hooks = child_to_info.get("__rate_hooks__", {})
        if rate_hooks:
            ns = {"self": self}
            for name in order:
                ns[name] = getattr(self, name)
            for var_path, expr_str in rate_hooks.items():
                target = self
                for part in var_path.split(".")[:-1]:
                    target = getattr(target, part)
                setattr(target, var_path.split(".")[-1], eval(expr_str, {}, ns))

        # Restore Expression-based and non-zero rates that _zero_rates would erase
        for var, val in restore_rates:
            var.rate = val

        return outputs.get(order[-1]) if order else None

    return types.MethodType(forward, mod)


def _apply_termination_condition(mod: Module, expr: str) -> None:
    def check(self):
        local_ns = {"self": mod}
        config_val = getattr(mod, "config", None)
        if config_val is None:
            for child_mod in mod._modules.values():
                if hasattr(child_mod, "config"):
                    config_val = child_mod.config
                    break
        local_ns["config"] = config_val
        for child_name, child_mod in mod._modules.items():
            local_ns[child_name] = child_mod
        try:
            return bool(eval(expr, {}, local_ns))
        except Exception:
            return False

    mod.is_terminating_condition_met = types.MethodType(check, mod)


def compile_canvas_json(
    canvas_data: Union[dict, list],
    class_registry: Optional[dict[str, type]] = None,
    config: Optional[Any] = None,
) -> Module:
    if isinstance(canvas_data, list):
        json_tree = _flat_canvas_to_tree(canvas_data)
    elif isinstance(canvas_data, dict):
        json_tree = canvas_data
    else:
        raise ValueError("canvas_data must be a dictionary or a list of dictionaries.")

    equations = []
    nodes_info = {}

    def build_module(name, node_data, path="", existing_obj=None, parent_obj=None):
        cls = _resolve_class(node_data.get("class", "Module"), class_registry)
        attrs = node_data.get("attributes", {})

        obj = existing_obj or _instantiate_class(cls, name, attrs, config, parent_obj)

        if "layout" in node_data:
            obj.layout = node_data["layout"]
        for k, v in attrs.items():
            setattr(obj, k, v)

        current_path = f"{path}.{name}" if path else name
        nodes_info[current_path] = node_data

        for var_name, var_info in node_data.get("variables", {}).items():
            _restore_variable(
                obj, var_name, var_info, current_path, class_registry, equations
            )

        for child_name, child_data in node_data.get("children", {}).items():
            existing = getattr(obj, child_name, None)
            if isinstance(existing, Module):
                build_module(
                    child_name,
                    child_data,
                    current_path,
                    existing_obj=existing,
                    parent_obj=obj,
                )
            else:
                setattr(
                    obj,
                    child_name,
                    build_module(child_name, child_data, current_path, parent_obj=obj),
                )

        return obj

    root = build_module("", json_tree, "")
    root.parent = None

    name_to_mod = {path: mod for path, mod in root.named_modules()}
    name_to_mod[""] = root

    _resolve_rate_equations(equations, name_to_mod)
    _record_dependency_edges(name_to_mod, nodes_info)

    needs_dynamic_forward = {
        mp
        for mp, m in name_to_mod.items()
        if m._modules and _is_dummy_forward(type(m).forward)
    }

    for mod_path in needs_dynamic_forward:
        mod = name_to_mod[mod_path]
        children_names = list(mod._modules.keys())
        child_to_info = {}
        for name in children_names:
            full_path = f"{mod_path}.{name}" if mod_path else name
            child_to_info[name] = nodes_info.get(full_path, {})

        restore_rates = []
        for var in mod._variables.values():
            if isinstance(var, Level):
                if isinstance(var._rate, Expression) or (
                    isinstance(var._rate, (int, float)) and var._rate != 0
                ):
                    restore_rates.append((var, var._rate))

        mod.forward = _build_dynamic_forward(
            mod, mod_path, children_names, child_to_info, restore_rates
        )

    # Restore Expression rates for non-dynamic-forward modules
    # (dynamic-forward modules handle this inside _build_dynamic_forward)
    for mod_path, mod in name_to_mod.items():
        if mod_path in needs_dynamic_forward:
            continue
        restore = [
            (v, v._rate)
            for v in mod._variables.values()
            if isinstance(v, Level) and isinstance(v._rate, Expression)
        ]
        if not restore:
            continue
        orig_f = mod.forward

        def _wrap(f, r):
            def _wrapped(self, *args, **kw):
                res = f(*args, **kw)
                for var, val in r:
                    var.rate = val
                return res

            return _wrapped

        mod.forward = types.MethodType(_wrap(orig_f, restore), mod)

    for mod_path, mod in name_to_mod.items():
        tc = nodes_info.get(mod_path, {}).get("termination_condition")
        if tc:
            _apply_termination_condition(mod, tc)

    return root

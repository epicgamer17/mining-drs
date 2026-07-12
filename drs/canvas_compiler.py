import math
import types
from typing import Any, Optional, Union

import drs
from .module import Module
from .variables import Variable, Level, Timer, Expression
from .flow import Flow
from .data_source import DataPoint
from ._execution_context import ExecutionContext
from .variables import deserialize_val

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


def compile_canvas_json(
    canvas_data: Union[dict, list],
    class_registry: Optional[dict[str, type]] = None,
    config: Optional[Any] = None,
) -> Module:
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

    equations_to_resolve = []
    nodes_info = {}

    def build_module(
        name: str,
        node_data: dict,
        path: str = "",
        existing_obj: Optional[Module] = None,
        parent_obj: Optional[Module] = None,
    ) -> Module:
        cls_name = node_data.get("class", "Module")
        cls = _resolve_class(cls_name, class_registry)
        attrs = node_data.get("attributes", {})

        if existing_obj is not None:
            obj = existing_obj
        else:
            import inspect

            try:
                sig = inspect.signature(cls.__init__)
                params = list(sig.parameters.values())
                params_no_self = [p for p in params if p.name != "self"]

                kwargs = {}
                has_all_required = True
                for param in params_no_self:
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
                            has_all_required = False
                            break
                    elif param.default == inspect.Parameter.empty:
                        has_all_required = False
                        break

                if has_all_required:
                    obj = cls(**kwargs)
                else:
                    obj = cls()
            except Exception:
                obj = cls.__new__(cls)
                Module.__init__(obj)
                if config is not None:
                    obj.config = config
                if not hasattr(obj, "expected_attributes"):
                    obj.expected_attributes = []
                if not hasattr(obj, "name"):
                    obj.name = name

        if "layout" in node_data:
            obj.layout = node_data["layout"]

        for k, v in attrs.items():
            setattr(obj, k, v)

        current_path = f"{path}.{name}" if path else name
        nodes_info[current_path] = node_data

        variables_data = node_data.get("variables", {})
        for var_name, var_info in variables_data.items():
            if isinstance(var_info, str):
                var_info = {"class": var_info, "value": 0.0}

            var_class_name = var_info.get("class", "Variable")
            var_value = var_info.get("value", 0.0)

            if isinstance(var_value, dict) and "__type__" in var_value:
                type_name = var_value["__type__"]
                obj_name = var_value.get("name")
                resolved_cls = _resolve_class(type_name, class_registry)
                if resolved_cls is not None and obj_name is not None:
                    try:
                        var_value = resolved_cls(obj_name)
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
                import enum

                if isinstance(var._value, enum.Enum) and isinstance(var_value, str):
                    enum_cls = type(var._value)
                    if var_value not in enum_cls.__members__:
                        raise ValueError(
                            f"Invalid value '{var_value}' for enum {enum_cls.__name__} "
                            f"on variable '{var_name}' at '{current_path}'."
                        )
                    var._value = enum_cls[var_value]
                elif (
                    hasattr(var._value, "name")
                    and hasattr(var._value, "id")
                    and isinstance(var_value, str)
                ):
                    cls = type(var._value)
                    try:
                        var._value = cls(var_value)
                    except Exception:
                        raise ValueError(
                            f"Could not construct {cls.__name__} from value '{var_value}' "
                            f"for variable '{var_name}' at '{current_path}'."
                        )
                else:
                    var._value = var_value if not isinstance(var_value, dict) else 0.0
            else:
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
                if "lower_threshold" in var_info:
                    var.lower_threshold = deserialize_val(var_info["lower_threshold"])
                if "upper_threshold" in var_info:
                    var.upper_threshold = deserialize_val(var_info["upper_threshold"])
                rate_val = var_info.get("rate", 0.0)
                if isinstance(rate_val, dict) and "equation" in rate_val:
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
                    var.rate = deserialize_val(rate_val)

        children_data = node_data.get("children", {})
        for child_name, child_data in children_data.items():
            existing_child = getattr(obj, child_name, None)
            if isinstance(existing_child, Module):
                build_module(
                    child_name,
                    child_data,
                    current_path,
                    existing_obj=existing_child,
                    parent_obj=obj,
                )
            else:
                child_obj = build_module(
                    child_name, child_data, current_path, parent_obj=obj
                )
                setattr(obj, child_name, child_obj)

        return obj

    root_name = ""
    root = build_module(root_name, json_tree, "")
    root.parent = None

    name_to_mod = {path: mod for path, mod in root.named_modules()}
    name_to_mod[""] = root

    orig_tracing = ExecutionContext.is_tracing()
    ExecutionContext.set_tracing(True)
    try:
        for var, eq_str, mod_path in equations_to_resolve:
            mod = name_to_mod[mod_path]
            local_ns = {"self": mod}
            for sib_name, sib_mod in mod._modules.items():
                local_ns[sib_name] = sib_mod
            p = mod.parent
            while p is not None:
                for sib_name, sib_mod in p._modules.items():
                    local_ns[sib_name] = sib_mod
                p = p.parent

            cleaned_eq = eq_str.strip()
            if cleaned_eq.startswith("(") and cleaned_eq.endswith(")"):
                cleaned_eq = cleaned_eq[1:-1]

            try:
                expr = eval(cleaned_eq, {}, local_ns)
                var.rate = expr
            except Exception as e:
                raise ValueError(
                    f"Could not resolve rate equation '{eq_str}' for variable "
                    f"'{var.name}' at path '{mod_path}': {e}"
                )
    finally:
        ExecutionContext.set_tracing(orig_tracing)

    def _is_dummy_forward(func) -> bool:
        if func is None:
            return True
        try:
            if hasattr(func, "__func__"):
                func = func.__func__
            import dis

            instructions = list(dis.get_instructions(func))
            opnames = [inst.opname for inst in instructions]
            if "RAISE_VARARGS" in opnames:
                return False
            has_calls = any(name.startswith("CALL") for name in opnames)
            has_stores = any(name.startswith("STORE") for name in opnames)
            return not has_calls and not has_stores
        except Exception:
            pass
        return False

    needs_dynamic_forward = {
        mod_path
        for mod_path, mod in name_to_mod.items()
        if mod._modules and _is_dummy_forward(type(mod).forward)
    }

    for mod_path, mod in name_to_mod.items():
        node_info = nodes_info.get(mod_path, {})
        conns = node_info.get("connections", {})

        flow_inputs = conns.get("flow_inputs", [])
        data_inputs = conns.get("data_inputs", [])
        variable_reads = conns.get("variable_reads", [])

        for entry in flow_inputs:
            src_path = entry["module"]
            src_mod = name_to_mod.get(src_path)
            if src_mod:
                mod._record_flow_edge(src_mod)

        for entry in data_inputs:
            src_path = entry["module"]
            src_mod = name_to_mod.get(src_path)
            if src_mod:
                mod._record_data_edge(src_mod)

        for r_info in variable_reads:
            src_path = r_info["module"]
            var_name = r_info["variable"]
            src_mod = name_to_mod.get(src_path)
            if src_mod and var_name in src_mod._variables:
                mod._record_incoming_edge(src_mod._variables[var_name])

    for mod_path in needs_dynamic_forward:
        mod = name_to_mod[mod_path]
        children_names = list(mod._modules.keys())

        child_to_info = {}
        for name in children_names:
            full_path = f"{mod_path}.{name}" if mod_path else name
            child_to_info[name] = nodes_info.get(full_path, {})

        adj = {name: set() for name in children_names}
        in_degree = {name: 0 for name in children_names}

        for name in children_names:
            info = child_to_info[name]
            child_conns = info.get("connections", {})
            sources = child_conns.get("flow_inputs", []) + child_conns.get(
                "data_inputs", []
            )
            for src in sources:
                src_path = src.get("module")
                if src_path is None:
                    continue
                src_suffix = src_path
                if mod_path:
                    if src_path.startswith(mod_path + "."):
                        src_suffix = src_path[len(mod_path) + 1 :]
                if src_suffix in adj:
                    if name not in adj[src_suffix]:
                        adj[src_suffix].add(name)
                        in_degree[name] += 1

        queue = sorted([name for name, deg in in_degree.items() if deg == 0])
        execution_order = []
        while queue:
            curr = queue.pop(0)
            execution_order.append(curr)
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(execution_order) < len(children_names):
            remaining = [name for name in children_names if name not in execution_order]
            execution_order.extend(remaining)

        mod_rate_hooks = nodes_info.get(mod_path, {}).get("rate_hooks", {})
        mod_tc = nodes_info.get(mod_path, {}).get("termination_condition")

        def make_forward(order, child_to_info_map, m_path, rate_hooks, tc_str):
            def _rel(src_path):
                if m_path and src_path.startswith(m_path + "."):
                    return src_path[len(m_path) + 1 :]
                return src_path

            def forward(self, *args, **kwargs):
                outputs = {}
                for child_name in order:
                    child_mod = getattr(self, child_name)
                    info = child_to_info_map[child_name]
                    c_conns = info.get("connections", {})

                    flow_ins = c_conns.get("flow_inputs", [])
                    data_ins = c_conns.get("data_inputs", [])

                    child_args = []
                    child_kwargs = {}
                    for conn in flow_ins + data_ins:
                        src_rel = _rel(conn["module"])
                        src_output = outputs.get(src_rel)
                        variable = conn.get("variable")
                        output_index = conn.get("output_index")

                        if variable:
                            src_mod = getattr(self, src_rel, None) if src_rel else None
                            if src_mod and hasattr(src_mod, variable):
                                val = getattr(src_mod, variable)
                            else:
                                val = src_output
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

                if rate_hooks:
                    ns = {"self": self}
                    for child_name in order:
                        ns[child_name] = getattr(self, child_name)
                    for var_path, expr_str in rate_hooks.items():
                        parts = var_path.split(".")
                        target = self
                        for part in parts[:-1]:
                            target = getattr(target, part)
                        var = getattr(target, parts[-1])
                        var.rate = eval(expr_str, {}, ns)

                if order:
                    return outputs.get(order[-1])
                return None

            return forward

        mod.forward = types.MethodType(
            make_forward(
                execution_order, child_to_info, mod_path, mod_rate_hooks, mod_tc
            ),
            mod,
        )

    # Re-apply rate expressions after forward generation to prevent _zero_rates from erasing them
    for mod_path, mod in name_to_mod.items():
        restore_vars = []
        has_dynamic = mod_path in needs_dynamic_forward
        for var_name, var in mod._variables.items():
            if isinstance(var, Level):
                if isinstance(var._rate, Expression):
                    restore_vars.append((var, var._rate))
                elif (
                    has_dynamic
                    and isinstance(var._rate, (int, float))
                    and var._rate != 0
                ):
                    restore_vars.append((var, var._rate))

        if restore_vars:
            orig_forward = mod.forward

            def make_wrapped_forward(orig_f, rvars):
                def wrapped_forward(self, *args, **kwargs):
                    res = orig_f(*args, **kwargs)
                    for var, val in rvars:
                        var.rate = val
                    return res

                return wrapped_forward

            mod.forward = types.MethodType(
                make_wrapped_forward(orig_forward, restore_vars), mod
            )

    # Apply termination conditions
    for mod_path, mod in name_to_mod.items():
        tc = nodes_info.get(mod_path, {}).get("termination_condition")
        if tc:

            def make_termination_check(expr, target_mod):
                def check(self):
                    local_ns = {"self": target_mod}
                    config_val = getattr(target_mod, "config", None)
                    if config_val is None:
                        for child_mod in target_mod._modules.values():
                            if hasattr(child_mod, "config"):
                                config_val = child_mod.config
                                break
                    local_ns["config"] = config_val
                    for child_name, child_mod in target_mod._modules.items():
                        local_ns[child_name] = child_mod
                    try:
                        return bool(eval(expr, {}, local_ns))
                    except Exception:
                        return False

                return check

            mod.is_terminating_condition_met = types.MethodType(
                make_termination_check(tc, mod), mod
            )

    return root

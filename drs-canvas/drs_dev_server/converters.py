from collections import defaultdict

from drs.variables import Variable


def _remap_telemetry_keys(history, model):
    key_map = {}
    for path, module in model.named_modules():
        for attr_name, var in module._variables.items():
            if isinstance(var, Variable):
                telemetry_key = var.name
                frontend_key = f"{path}.{attr_name}" if path else attr_name
                key_map[telemetry_key] = frontend_key

    remapped = []
    for entry in history:
        new_entry = {}
        for k, v in entry.items():
            new_entry[key_map.get(k, k)] = v
        remapped.append(new_entry)
    return remapped


def react_flow_to_drs_flat(nodes, edges, with_root_defaults=True):
    drs_nodes = []
    parent_exists = any(n.get("id") == "" or n.get("id") == "root" for n in nodes)
    if not parent_exists and with_root_defaults:
        drs_nodes.append(
            {
                "id": "",
                "class": "DRSModel",
                "layout": {"x": 0, "y": 0},
                "variables": {
                    "GlobalTime": {
                        "class": "Timer",
                        "value": 0.0,
                        "lower_threshold": "-Infinity",
                        "upper_threshold": "Infinity",
                        "rate": 1.0,
                    }
                },
                "rate_hooks": {
                    "controller.total_system_ore_mass": "ore1_stock.current_mass.rate + ore2_stock.current_mass.rate"
                },
                "termination_condition": "mine.cumulative_extracted_mass.value >= config.total_ore_to_extract",
            }
        )

    source_flow_consumer_count = defaultdict(int)

    for node in nodes:
        node_id = node.get("id")
        if node_id == "root":
            node_id = ""
        node_data = node.get("data", {})
        node_class = node_data.get("class", "Module")
        position = node.get("position", {"x": 0, "y": 0})
        variables = node_data.get("variables", {})

        flow_inputs = []
        data_inputs = []
        variable_reads = []

        for edge in edges:
            if edge.get("target") == node.get("id"):
                target_handle = edge.get("targetHandle")
                source = edge.get("source")
                if source == "root":
                    source = ""
                if target_handle == "flow-in":
                    edge_data = edge.get("data", {}) or {}
                    output_index = source_flow_consumer_count[source]
                    source_flow_consumer_count[source] += 1
                    flow_inputs.append(
                        {
                            "module": source,
                            "param": edge_data.get("param"),
                            "output_index": output_index,
                        }
                    )
                elif target_handle == "data-in":
                    edge_data = edge.get("data", {}) or {}
                    data_inputs.append(
                        {
                            "module": source,
                            "param": edge_data.get("param"),
                            "variable": edge_data.get("variable"),
                        }
                    )
                elif target_handle == "read-in":
                    edge_data = edge.get("data", {}) or {}
                    read_var = (
                        edge_data.get("variable")
                        or edge_data.get("readVariable")
                        or edge.get("variable")
                    )
                    src_node = next(
                        (n for n in nodes if n.get("id") == edge.get("source")), None
                    )
                    src_vars = (
                        src_node.get("data", {}).get("variables", {})
                        if src_node
                        else {}
                    )
                    if read_var is None:
                        var_names = list(src_vars.keys())
                        if "current_mass" in src_vars:
                            read_var = "current_mass"
                        elif len(var_names) == 1:
                            read_var = var_names[0]
                        else:
                            raise ValueError(
                                f"Read edge '{edge.get('id', '<unknown>')}' from "
                                f"'{source}' to '{node.get('id')}' must specify data.variable."
                            )
                    if read_var not in src_vars:
                        raise ValueError(
                            f"Read edge '{edge.get('id', '<unknown>')}' references missing "
                            f"variable '{read_var}' on source module '{source}'."
                        )
                    variable_reads.append({"module": source, "variable": read_var})

        drs_nodes.append(
            {
                "id": node_id,
                "class": node_class,
                "layout": position,
                "variables": variables,
                "attributes": node_data.get("attributes", {}),
                "connections": {
                    "flow_inputs": flow_inputs,
                    "data_inputs": data_inputs,
                    "variable_reads": variable_reads,
                },
            }
        )
    return drs_nodes

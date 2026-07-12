import http.server
import json
import math
import os
import sys
import traceback

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(ROOT_DIR)
sys.path.append(WORKSPACE_ROOT)


def _serialize_for_response(obj):
    import enum

    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, enum.Enum):
        return obj.name
    if (
        hasattr(obj, "name")
        and hasattr(obj, "id")
        and not isinstance(obj, (int, float, str, bool))
    ):
        return {"__type__": type(obj).__name__, "name": obj.name}
    if isinstance(obj, dict):
        return {k: _serialize_for_response(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_response(v) for v in obj]
    return obj


DEFAULT_NODES = [
    {
        "id": "mine",
        "type": "extractionNode",
        "position": {"x": 80, "y": 220},
        "data": {
            "label": "Concentrator Mine Face",
            "class": "ConcentratorMineFace",
            "variables": {
                "cumulative_extracted_mass": {
                    "class": "Level",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "parcel_extracted_mass": {
                    "class": "Level",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
            },
        },
    },
    {
        "id": "fleet",
        "type": "factoryNode",
        "position": {"x": 340, "y": 200},
        "data": {
            "label": "Continuous Fleet Logistics",
            "class": "ContinuousFleetLogistics",
            "variables": {
                "stockpile2_routing_fraction": {"class": "Variable", "value": 0.0}
            },
        },
    },
    {
        "id": "ore1_stock",
        "type": "bufferNode",
        "position": {"x": 620, "y": 100},
        "data": {
            "label": "Ore 1 Stockpile",
            "class": "Stockpile",
            "attributes": {
                "name": "Ore1Stock",
                "expected_attributes": ["contained_ore_fraction_mass"],
            },
            "variables": {
                "current_mass": {
                    "class": "Level",
                    "value": 42000.0,
                    "lower_threshold": 0.0,
                    "upper_threshold": 60000.0,
                    "rate": 0.0,
                },
                "contained_ore_fraction_mass": {
                    "class": "Level",
                    "value": 12600.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "actual_outflow_rate": {"class": "Variable", "value": 0.0},
            },
        },
    },
    {
        "id": "ore2_stock",
        "type": "bufferNode",
        "position": {"x": 620, "y": 320},
        "data": {
            "label": "Ore 2 Stockpile",
            "class": "Stockpile",
            "attributes": {
                "name": "Ore2Stock",
                "expected_attributes": ["contained_ore_fraction_mass"],
            },
            "variables": {
                "current_mass": {
                    "class": "Level",
                    "value": 18000.0,
                    "lower_threshold": 0.0,
                    "upper_threshold": 60000.0,
                    "rate": 0.0,
                },
                "contained_ore_fraction_mass": {
                    "class": "Level",
                    "value": 5400.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "actual_outflow_rate": {"class": "Variable", "value": 0.0},
            },
        },
    },
    {
        "id": "plant",
        "type": "factoryNode",
        "position": {"x": 920, "y": 220},
        "data": {
            "label": "Concentrator Plant",
            "class": "ConcentratorPlant",
            "variables": {
                "cumulative_milled_mass": {
                    "class": "Level",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                }
            },
        },
    },
    {
        "id": "controller",
        "type": "factoryNode",
        "position": {"x": 340, "y": 440},
        "data": {
            "label": "Concentrator Controller",
            "class": "ConcentratorController",
            "variables": {
                "active_operating_mode": {
                    "class": "Variable",
                    "value": {"__type__": "OperatingMode", "name": "MODE_A"},
                },
                "total_system_ore_mass": {
                    "class": "Level",
                    "value": 60000.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "current_campaign_duration": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "current_contingency_duration": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "cumulative_time_mode_a": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "cumulative_time_mode_a_contingency": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "cumulative_time_mode_a_surging": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "cumulative_time_mode_b": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "cumulative_time_mode_b_contingency": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "cumulative_time_mode_b_surging": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "cumulative_time_shutdown": {
                    "class": "Timer",
                    "value": 0.0,
                    "lower_threshold": "-Infinity",
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                },
                "target_mine_mass_rate": {"class": "Variable", "value": 0.0},
                "target_stock1_outflow_rate": {"class": "Variable", "value": 0.0},
                "target_stock2_outflow_rate": {"class": "Variable", "value": 0.0},
            },
        },
    },
]

DEFAULT_EDGES = [
    {
        "id": "e-flow-mine-fleet",
        "source": "mine",
        "sourceHandle": "flow-out",
        "target": "fleet",
        "targetHandle": "flow-in",
        "style": {"stroke": "#3b82f6", "strokeWidth": 4},
    },
    {
        "id": "e-flow-fleet-ore1",
        "source": "fleet",
        "sourceHandle": "flow-out",
        "target": "ore1_stock",
        "targetHandle": "flow-in",
        "data": {"param": "inflow"},
        "style": {"stroke": "#3b82f6", "strokeWidth": 4},
    },
    {
        "id": "e-flow-fleet-ore2",
        "source": "fleet",
        "sourceHandle": "flow-out",
        "target": "ore2_stock",
        "targetHandle": "flow-in",
        "data": {"param": "inflow"},
        "style": {"stroke": "#3b82f6", "strokeWidth": 4},
    },
    {
        "id": "e-flow-ore1-plant",
        "source": "ore1_stock",
        "sourceHandle": "flow-out",
        "target": "plant",
        "targetHandle": "flow-in",
        "data": {"param": "ore1_outflow"},
        "style": {"stroke": "#3b82f6", "strokeWidth": 4},
    },
    {
        "id": "e-flow-ore2-plant",
        "source": "ore2_stock",
        "sourceHandle": "flow-out",
        "target": "plant",
        "targetHandle": "flow-in",
        "data": {"param": "ore2_outflow"},
        "style": {"stroke": "#3b82f6", "strokeWidth": 4},
    },
    {
        "id": "e-read-ore1-controller",
        "source": "ore1_stock",
        "sourceHandle": "read-out",
        "target": "controller",
        "targetHandle": "read-in",
        "data": {"variable": "current_mass"},
        "style": {"stroke": "#f59e0b", "strokeWidth": 1.5},
    },
    {
        "id": "e-read-ore2-controller",
        "source": "ore2_stock",
        "sourceHandle": "read-out",
        "target": "controller",
        "targetHandle": "read-in",
        "data": {"variable": "current_mass"},
        "style": {"stroke": "#f59e0b", "strokeWidth": 1.5},
    },
    {
        "id": "e-data-controller-ore1",
        "source": "controller",
        "sourceHandle": "data-out",
        "target": "ore1_stock",
        "targetHandle": "data-in",
        "data": {
            "param": "requested_outflow_rate",
            "variable": "target_stock1_outflow_rate",
        },
        "style": {"stroke": "#10b981", "strokeWidth": 2, "strokeDasharray": "5,5"},
    },
    {
        "id": "e-data-controller-ore2",
        "source": "controller",
        "sourceHandle": "data-out",
        "target": "ore2_stock",
        "targetHandle": "data-in",
        "data": {
            "param": "requested_outflow_rate",
            "variable": "target_stock2_outflow_rate",
        },
        "style": {"stroke": "#10b981", "strokeWidth": 2, "strokeDasharray": "5,5"},
    },
]


def _validate_variable_value(node_id, var_name, var_value):
    if isinstance(var_value, str) and var_name == "active_operating_mode":
        raise ValueError(
            f"Variable '{var_name}' on node '{node_id}' has a plain string value "
            f"('{var_value}') but expects a typed object dict. "
            f"Use the format: {{'__type__': 'OperatingMode', 'name': '{var_value.split('.')[-1] if '.' in var_value else var_value}'}}"
        )
    if isinstance(var_value, dict) and "__type__" in var_value:
        import drs_mining.components as mc

        type_name = var_value["__type__"]
        if not hasattr(mc, type_name):
            import drs_mining.components.controllers as mc_ctrl

            if not hasattr(mc_ctrl, type_name):
                import drs_mining.components.models as mm

                if not hasattr(mm, type_name):
                    raise ValueError(
                        f"Unknown __type__ '{type_name}' in value for variable "
                        f"'{var_name}' on node '{node_id}'. "
                        f"Ensure the class is importable."
                    )


def _generate_dashboard_plot(df):
    import io
    import base64
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as _pd

    df = _pd.DataFrame(df)

    df["active_operating_mode_name"] = df["active_operating_mode"].apply(
        lambda x: x.name if hasattr(x, "name") else str(x)
    )

    df["Mode A"] = df["active_operating_mode_name"].apply(
        lambda m: (
            3 if m in ("MODE_A", "MODE_A_CONTINGENCY", "MODE_A_MINE_SURGING") else 0
        )
    )
    df["Mode B"] = df["active_operating_mode_name"].apply(
        lambda m: (
            2 if m in ("MODE_B", "MODE_B_CONTINGENCY", "MODE_B_MINE_SURGING") else 0
        )
    )
    df["Shutdown"] = df["active_operating_mode_name"].apply(
        lambda m: 1 if m == "SHUTDOWN" else 0
    )

    df["Total Ore Stockpile Level"] = df["total_system_ore_mass"] / 1000.0
    df["Ore 1 Stockpile Level"] = df["Ore1Stock_mass"] / 1000.0
    df["Ore 2 Stockpile Level"] = df["Ore2Stock_mass"] / 1000.0

    from drs.plot import (
        plot_time_series,
        plot_dual_axis_step,
        plot_safety_margin,
        build_dashboard,
    )
    from drs_mining.components.plot import (
        plot_ore_with_modes,
        plot_mode_distribution,
        plot_mode_dwell_times,
        plot_normalized_deviation_violin,
        plot_attributed_deficit,
        plot_deficit_disparity,
        plot_deficit_breakdown_bar,
        plot_structural_vs_operational_deficit,
        plot_normalized_cumulative_deficit,
        plot_structural_vs_operational_by_mode,
    )

    palette = {
        "MODE_A": "#1f77b4",
        "MODE_A_CONTINGENCY": "#2ca02c",
        "MODE_A_MINE_SURGING": "#9467bd",
        "MODE_B": "#d62728",
        "MODE_B_CONTINGENCY": "#ff7f0e",
        "MODE_B_MINE_SURGING": "#8c564b",
        "SHUTDOWN": "#FFD700",
    }
    structural_modes = ["SHUTDOWN", "MODE_A"]

    configs = [
        {
            "func": plot_time_series,
            "kwargs": {
                "y_columns": ["Mode A", "Mode B", "Shutdown"],
                "title": "Modes (Step)",
                "is_step": True,
            },
        },
        {
            "func": plot_ore_with_modes,
            "kwargs": {
                "time_col": "time",
                "ore_cols": [
                    "total_system_ore_mass",
                    "Ore1Stock_mass",
                    "Ore2Stock_mass",
                ],
                "mode_col": "active_operating_mode_name",
                "campaign_split_mode": "SHUTDOWN",
                "title": "Ore Stockpiles & Campaigns",
                "palette": palette,
                "hlines": [
                    {
                        "y": 60000,
                        "color": "black",
                        "linestyle": "--",
                        "linewidth": 1.5,
                        "alpha": 0.7,
                        "label": "Target Total (60k)",
                    },
                    {
                        "y": 20400,
                        "color": "red",
                        "linestyle": ":",
                        "linewidth": 2,
                        "alpha": 0.8,
                        "label": "Critical Ore 2 (20.4k)",
                    },
                ],
            },
        },
        {
            "func": plot_dual_axis_step,
            "kwargs": {
                "y1_col": "MassOfCurrentParcel",
                "y2_col": "CurrentParcelRoutingFraction",
                "y1_label": "Parcel Mass (tons)",
                "y2_label": "Grade (% Ore 2)",
                "title": "Current Parcel Properties",
            },
        },
        {
            "func": plot_safety_margin,
            "kwargs": {
                "level_col": "Ore1Stock_mass",
                "constraint_value": 0.0,
                "constraint_type": "lower",
                "title": "Safety Margin: Ore 1 Distance to Floor",
                "danger_threshold": 1000.0,
            },
        },
        {
            "func": plot_safety_margin,
            "kwargs": {
                "level_col": "Ore2Stock_mass",
                "constraint_value": 0.0,
                "constraint_type": "lower",
                "title": "Safety Margin: Ore 2 Distance to Floor",
                "danger_threshold": 1000.0,
            },
        },
        {
            "func": plot_mode_distribution,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "time_col": "time",
                "title": "Mode Distribution (% of Time Spent)",
                "palette": palette,
            },
        },
        {
            "func": plot_mode_dwell_times,
            "kwargs": {
                "time_col": "time",
                "mode_col": "active_operating_mode_name",
                "title": "Mode Stability (Dwell Times)",
            },
        },
        {
            "func": plot_normalized_deviation_violin,
            "kwargs": {
                "title": "Stockpile Deviation Variance (Violin)",
                "target_total": 60000.0,
                "target_ore1": 42000.0,
                "target_ore2": 18000.0,
            },
        },
        {
            "func": plot_attributed_deficit,
            "kwargs": {
                "time_col": "time",
                "mode_col": "active_operating_mode_name",
                "extraction_col": "cumulative_extracted_mass",
                "ideal_rate_per_day": 6000.0,
                "title": "Cumulative Production Deficit by Mode",
                "palette": palette,
            },
        },
        {
            "func": plot_deficit_disparity,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "title": "Mode Efficiency (Time Spent vs. Deficit Caused)",
                "ideal_rate": 6000.0,
            },
        },
        {
            "func": plot_deficit_breakdown_bar,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate_per_day": 6000.0,
                "palette": palette,
            },
        },
        {
            "func": plot_structural_vs_operational_deficit,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate": 6000.0,
                "structural_modes": structural_modes,
            },
        },
        {
            "func": plot_normalized_cumulative_deficit,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate_per_day": 6000.0,
                "palette": palette,
            },
        },
        {
            "func": plot_structural_vs_operational_by_mode,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate": 6000.0,
                "structural_modes": structural_modes,
            },
        },
    ]

    fig_comp = build_dashboard(
        df, configs, title="Comprehensive Mine Diagnostics", figsize=(18, 69)
    )

    buf = io.BytesIO()
    fig_comp.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig_comp)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _remap_telemetry_keys(history, model):
    from drs.variables import Variable

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
            new_key = key_map.get(k, k)
            new_entry[new_key] = v
        remapped.append(new_entry)
    return remapped


def react_flow_to_drs_flat(nodes, edges):
    drs_nodes = []
    parent_exists = any(n.get("id") == "" or n.get("id") == "root" for n in nodes)
    if not parent_exists:
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

    from collections import defaultdict

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

        for var_name, var_info in (
            variables if isinstance(variables, dict) else {}
        ).items():
            if isinstance(var_info, dict):
                var_value = var_info.get("value")
                _validate_variable_value(node_id, var_name, var_value)

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


class DevServerHandler(http.server.BaseHTTPRequestHandler):
    def _respond(self, status=200, body=None, content_type="application/json"):
        self.send_response(status)
        body_bytes = body.encode("utf-8") if isinstance(body, str) else body
        self.send_header("Content-Type", content_type)
        if body_bytes is not None:
            self.send_header("Content-Length", str(len(body_bytes)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if body_bytes is not None:
            self.wfile.write(body_bytes)

    def do_OPTIONS(self):
        self._respond(200)

    def do_GET(self):
        if self.path == "/api/topology":
            file_path = os.path.join(ROOT_DIR, "src", "drs_canvas_state.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    self._respond(
                        200,
                        body=json.dumps(_serialize_for_response(data)).encode("utf-8"),
                    )
                    return
                except Exception as e:
                    self._respond(
                        500,
                        body=json.dumps(
                            _serialize_for_response(
                                {"error": f"Failed to load file: {e}"}
                            )
                        ).encode("utf-8"),
                    )
                    return
            self._respond(
                200,
                body=json.dumps(
                    _serialize_for_response(
                        {"nodes": DEFAULT_NODES, "edges": DEFAULT_EDGES}
                    )
                ).encode("utf-8"),
            )
        else:
            self._respond(
                404,
                body=json.dumps(
                    _serialize_for_response({"error": f"Path '{self.path}' not found"})
                ).encode("utf-8"),
            )

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8"))
        except Exception as e:
            self._respond(
                400,
                body=json.dumps(
                    _serialize_for_response({"error": f"Invalid JSON body: {e}"})
                ).encode("utf-8"),
            )
            return

        if self.path == "/api/topology":
            canvas_file = os.path.join(ROOT_DIR, "src", "drs_canvas_state.json")
            try:
                os.makedirs(os.path.dirname(canvas_file), exist_ok=True)
                with open(canvas_file, "w") as f:
                    json.dump(payload, f, indent=2)
                nodes = payload.get("nodes", [])
                edges = payload.get("edges", [])
                drs_flat = react_flow_to_drs_flat(nodes, edges)
                with open(
                    os.path.join(WORKSPACE_ROOT, "drs_topology_flat.json"), "w"
                ) as f:
                    json.dump(drs_flat, f, indent=2)
                try:
                    from drs.canvas_compiler import _flat_canvas_to_tree

                    drs_tree = _flat_canvas_to_tree(drs_flat)
                    with open(
                        os.path.join(WORKSPACE_ROOT, "drs_topology_tree.json"), "w"
                    ) as f:
                        json.dump(drs_tree, f, indent=2)
                except Exception as tree_ex:
                    print(f"Warning: Hierarchical tree export failed: {tree_ex}")
                self._respond(
                    200,
                    body=json.dumps(
                        _serialize_for_response(
                            {
                                "status": "ok",
                                "message": "Canvas saved and translated successfully",
                            }
                        )
                    ).encode("utf-8"),
                )
            except Exception as e:
                self._respond(
                    500,
                    body=json.dumps(
                        _serialize_for_response(
                            {"error": f"Failed to save topology state: {e}"}
                        )
                    ).encode("utf-8"),
                )

        elif self.path == "/api/compile":
            try:
                nodes = payload.get("nodes", [])
                edges = payload.get("edges", [])
                drs_flat = react_flow_to_drs_flat(nodes, edges)
                from drs.canvas_compiler import compile_canvas_json
                from drs_mining.components.config import ConcentratorConfig

                config = ConcentratorConfig()
                model = compile_canvas_json(drs_flat, config=config)
                self._respond(
                    200,
                    body=json.dumps(
                        _serialize_for_response(
                            {
                                "status": "ok",
                                "message": f"Compilation verification successful! Root Class: {type(model).__name__}",
                            }
                        )
                    ).encode("utf-8"),
                )
            except Exception as e:
                err_details = traceback.format_exc()
                self._respond(
                    400,
                    body=json.dumps(
                        _serialize_for_response(
                            {
                                "status": "error",
                                "message": str(e),
                                "details": err_details,
                            }
                        )
                    ).encode("utf-8"),
                )

        elif self.path == "/api/simulate":
            try:
                nodes = payload.get("nodes", [])
                edges = payload.get("edges", [])
                max_time = float(payload.get("max_time", 100.0))
                seed_value = int(payload.get("seed", 42))

                import random as _random
                import numpy as _numpy

                _random.seed(seed_value)
                _numpy.random.seed(seed_value)

                drs_flat = react_flow_to_drs_flat(nodes, edges)

                from drs.canvas_compiler import compile_canvas_json
                from drs.engine import DRSEngine
                from drs.telemetry import Telemetry
                from drs_mining.components.config import ConcentratorConfig

                config = ConcentratorConfig()
                model = compile_canvas_json(drs_flat, config=config)
                engine = DRSEngine(model)
                telemetry = Telemetry(model)

                telemetry.register_metric(
                    "MassOfCurrentParcel",
                    lambda t, m, s, _: m.mine.active_parcel_initial_mass.value,
                )
                telemetry.register_metric(
                    "CurrentParcelRoutingFraction",
                    lambda t, m, s, _: m.fleet.stockpile2_routing_fraction.value,
                )
                telemetry.register_metric(
                    "Campaign_Shutdown",
                    lambda t, m, s, _: m.controller.current_campaign_duration.value,
                )
                telemetry.register_metric(
                    "Contingency",
                    lambda t, m, s, _: m.controller.current_contingency_duration.value,
                )

                engine.attach_telemetry(telemetry)
                result = engine.run(max_time)

                remapped_history = _remap_telemetry_keys(telemetry.history, model)

                import pandas as _pd

                _df = _pd.DataFrame(telemetry.history)
                _dashboard_png = _generate_dashboard_plot(_df)

                events_out = []
                for e in telemetry.events:
                    events_out.append(
                        {
                            "time": e.time,
                            "event_type": e.event_type,
                            "source": e.source,
                            "details": e.details,
                        }
                    )

                self._respond(
                    200,
                    body=json.dumps(
                        _serialize_for_response(
                            {
                                "status": "ok",
                                "history": remapped_history,
                                "events": events_out,
                                "plots": {"dashboard_png": _dashboard_png},
                            }
                        )
                    ).encode("utf-8"),
                )
            except Exception as e:
                err_details = traceback.format_exc()
                self._respond(
                    400,
                    body=json.dumps(
                        _serialize_for_response(
                            {
                                "status": "error",
                                "message": str(e),
                                "details": err_details,
                            }
                        )
                    ).encode("utf-8"),
                )
        else:
            self._respond(
                404,
                body=json.dumps(
                    _serialize_for_response({"error": f"Path '{self.path}' not found"})
                ).encode("utf-8"),
            )


def run(port=8000):
    server_address = ("", port)
    httpd = http.server.HTTPServer(server_address, DevServerHandler)
    print(f"DRS Workspace Local Dev Server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Dev Server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run(port=port)

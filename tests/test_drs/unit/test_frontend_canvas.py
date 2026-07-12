import sys
from pathlib import Path

import pytest
from drs import Module, Variable, Level
from drs.canvas_compiler import validate_canvas_json

sys.path.insert(0, str(Path(__file__).parents[3] / "drs-canvas"))
from drs_dev_server.converters import react_flow_to_drs_flat


class MineFace(Module):
    def __init__(self):
        super().__init__()
        self.rate = Variable("rate", 10.0)


class Stockpile(Module):
    def __init__(self):
        super().__init__()
        self.current_mass = Level("current_mass", 150.0)
        self.current_mass.lower_threshold = 0.0
        self.current_mass.upper_threshold = 1000.0


class Concentrator(Module):
    def __init__(self):
        super().__init__()
        self.throughput = Variable("throughput", 80.0)


class ModelTopology(Module):
    def __init__(self):
        super().__init__()
        self.mine = MineFace()
        self.stock = Stockpile()
        self.plant = Concentrator()


def test_canvas_flat_json_validation():
    model = ModelTopology()

    # Structure matching React Flow Export Flat format
    canvas_flat_data = [
        {
            "id": "",
            "class": "ModelTopology",
            "layout": {"x": 0, "y": 0},
            "variables": {},
            "connections": {},
        },
        {
            "id": "mine",
            "class": "MineFace",
            "layout": {"x": 100, "y": 150},
            "variables": {"rate": {"class": "Variable", "value": 10.0}},
            "connections": {"flow_inputs": [], "data_inputs": [], "variable_reads": []},
        },
        {
            "id": "stock",
            "class": "Stockpile",
            "layout": {"x": 400, "y": 120},
            "variables": {
                "current_mass": {
                    "class": "Level",
                    "value": 150.0,
                    "lower_threshold": 0.0,
                    "upper_threshold": 1000.0,
                    "rate": 0.0,
                }
            },
            "connections": {
                "flow_inputs": [
                    {"module": "mine", "param": "inflow", "output_index": 0}
                ],
                "data_inputs": [],
                "variable_reads": [],
            },
        },
        {
            "id": "plant",
            "class": "Concentrator",
            "layout": {"x": 720, "y": 180},
            "variables": {"throughput": {"class": "Variable", "value": 80.0}},
            "connections": {
                "flow_inputs": [
                    {"module": "stock", "param": "inflow", "output_index": 0}
                ],
                "data_inputs": [],
                "variable_reads": [],
            },
        },
    ]

    # Should validate perfectly
    validate_canvas_json(canvas_flat_data, model)


def test_canvas_hierarchical_json_validation():
    model = ModelTopology()

    # Structure matching React Flow Export Tree format
    canvas_tree_data = {
        "class": "ModelTopology",
        "layout": {"x": 0, "y": 0},
        "variables": {},
        "children": {
            "mine": {
                "class": "MineFace",
                "layout": {"x": 100, "y": 150},
                "variables": {"rate": {"class": "Variable", "value": 12.5}},
                "children": {},
                "connections": {},
            },
            "stock": {
                "class": "Stockpile",
                "layout": {"x": 400, "y": 120},
                "variables": {
                    "current_mass": {
                        "class": "Level",
                        "value": 200.0,
                        "lower_threshold": 0.0,
                        "upper_threshold": 1000.0,
                        "rate": 0.0,
                    }
                },
                "children": {},
                "connections": {
                    "flow_inputs": [
                        {"module": "mine", "param": "inflow", "output_index": 0}
                    ]
                },
            },
            "plant": {
                "class": "Concentrator",
                "layout": {"x": 720, "y": 180},
                "variables": {"throughput": {"class": "Variable", "value": 90.0}},
                "children": {},
                "connections": {
                    "flow_inputs": [
                        {"module": "stock", "param": "inflow", "output_index": 0}
                    ]
                },
            },
        },
    }

    # Should validate perfectly
    validate_canvas_json(canvas_tree_data, model)


def test_canvas_boundary_violation():
    model = ModelTopology()

    # Variable current_mass value (1200.0) exceeds upper_threshold (1000.0)
    invalid_canvas_data = [
        {"id": "", "class": "ModelTopology", "variables": {}},
        {
            "id": "mine",
            "class": "MineFace",
            "variables": {"rate": {"class": "Variable", "value": 10.0}},
        },
        {
            "id": "stock",
            "class": "Stockpile",
            "variables": {
                "current_mass": {
                    "class": "Level",
                    "value": 1200.0,  # Boundary violation!
                    "lower_threshold": 0.0,
                    "upper_threshold": 1000.0,
                }
            },
        },
        {
            "id": "plant",
            "class": "Concentrator",
            "variables": {"throughput": {"class": "Variable", "value": 80.0}},
        },
    ]

    with pytest.raises(
        ValueError, match="Boundary violation for variable 'current_mass'"
    ):
        validate_canvas_json(invalid_canvas_data, model)


def test_react_flow_read_edges_use_explicit_variable_metadata():
    nodes = [
        {
            "id": "stock",
            "data": {
                "class": "Stockpile",
                "variables": {
                    "wrong_first": {"class": "Variable", "value": 1.0},
                    "current_mass": {"class": "Level", "value": 100.0},
                },
            },
        },
        {
            "id": "controller",
            "data": {"class": "Module", "variables": {}},
        },
    ]
    edges = [
        {
            "id": "read-stock-controller",
            "source": "stock",
            "target": "controller",
            "targetHandle": "read-in",
            "data": {"variable": "current_mass"},
        }
    ]

    flat = react_flow_to_drs_flat(nodes, edges)
    controller = next(node for node in flat if node["id"] == "controller")

    assert controller["connections"]["variable_reads"] == [
        {"module": "stock", "variable": "current_mass"}
    ]


def test_react_flow_read_edges_reject_ambiguous_missing_variable_metadata():
    nodes = [
        {
            "id": "stock",
            "data": {
                "class": "Stockpile",
                "variables": {
                    "first": {"class": "Variable", "value": 1.0},
                    "second": {"class": "Variable", "value": 2.0},
                },
            },
        },
        {
            "id": "controller",
            "data": {"class": "Module", "variables": {}},
        },
    ]
    edges = [
        {
            "id": "read-stock-controller",
            "source": "stock",
            "target": "controller",
            "targetHandle": "read-in",
        }
    ]

    with pytest.raises(ValueError, match="must specify data.variable"):
        react_flow_to_drs_flat(nodes, edges)


def test_react_flow_legacy_read_edges_infer_current_mass():
    nodes = [
        {
            "id": "stock",
            "data": {
                "class": "Stockpile",
                "variables": {
                    "wrong_first": {"class": "Variable", "value": 1.0},
                    "current_mass": {"class": "Level", "value": 100.0},
                },
            },
        },
        {
            "id": "controller",
            "data": {"class": "Module", "variables": {}},
        },
    ]
    edges = [
        {
            "id": "legacy-read-stock-controller",
            "source": "stock",
            "target": "controller",
            "targetHandle": "read-in",
        }
    ]

    flat = react_flow_to_drs_flat(nodes, edges)
    controller = next(node for node in flat if node["id"] == "controller")

    assert controller["connections"]["variable_reads"] == [
        {"module": "stock", "variable": "current_mass"}
    ]

import pytest
from drs import Module, Variable, Level, validate_canvas_json

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
            "connections": {}
        },
        {
            "id": "mine",
            "class": "MineFace",
            "layout": {"x": 100, "y": 150},
            "variables": {
                "rate": {"class": "Variable", "value": 10.0}
            },
            "connections": {
                "flow_inputs": [],
                "data_inputs": [],
                "variable_reads": []
            }
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
                    "rate": 0.0
                }
            },
            "connections": {
                "flow_inputs": ["mine"],
                "data_inputs": [],
                "variable_reads": []
            }
        },
        {
            "id": "plant",
            "class": "Concentrator",
            "layout": {"x": 720, "y": 180},
            "variables": {
                "throughput": {"class": "Variable", "value": 80.0}
            },
            "connections": {
                "flow_inputs": ["stock"],
                "data_inputs": [],
                "variable_reads": []
            }
        }
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
                "variables": {
                    "rate": {"class": "Variable", "value": 12.5}
                },
                "children": {},
                "connections": {}
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
                        "rate": 0.0
                    }
                },
                "children": {},
                "connections": {
                    "flow_inputs": ["mine"]
                }
            },
            "plant": {
                "class": "Concentrator",
                "layout": {"x": 720, "y": 180},
                "variables": {
                    "throughput": {"class": "Variable", "value": 90.0}
                },
                "children": {},
                "connections": {
                    "flow_inputs": ["stock"]
                }
            }
        }
    }
    
    # Should validate perfectly
    validate_canvas_json(canvas_tree_data, model)


def test_canvas_boundary_violation():
    model = ModelTopology()
    
    # Variable current_mass value (1200.0) exceeds upper_threshold (1000.0)
    invalid_canvas_data = [
        {
            "id": "",
            "class": "ModelTopology",
            "variables": {}
        },
        {
            "id": "mine",
            "class": "MineFace",
            "variables": {
                "rate": {"class": "Variable", "value": 10.0}
            }
        },
        {
            "id": "stock",
            "class": "Stockpile",
            "variables": {
                "current_mass": {
                    "class": "Level",
                    "value": 1200.0,  # Boundary violation!
                    "lower_threshold": 0.0,
                    "upper_threshold": 1000.0
                }
            }
        },
        {
            "id": "plant",
            "class": "Concentrator",
            "variables": {
                "throughput": {"class": "Variable", "value": 80.0}
            }
        }
    ]
    
    with pytest.raises(ValueError, match="Boundary violation for variable 'current_mass'"):
        validate_canvas_json(invalid_canvas_data, model)

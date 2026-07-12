import pytest
import math
from drs import (
    Module,
    Variable,
    Level,
    Timer,
    Flow,
    DRSEngine,
    DRSConfig,
)
from drs.canvas_compiler import compile_canvas_json, validate_canvas_json
from drs.variables import Expression


# Mock component classes to compile against
class MockSource(Module):
    def __init__(self):
        super().__init__()
        self.output_rate = Variable("output_rate", 20.0)

    def forward(self) -> Flow:
        # Returns physical flow
        return Flow(self.output_rate.value)


class MockBuffer(Module):
    def __init__(self):
        super().__init__()
        self.mass = Level("mass", 50.0)
        self.mass.lower_threshold = 0.0
        self.mass.upper_threshold = 500.0

    def forward(self, inflow: Flow = None) -> Flow:
        if inflow is not None:
            self.mass.rate = inflow.value - 5.0
        else:
            self.mass.rate = -5.0
        # Returns flow out
        return Flow(5.0)


class MockConsumer(Module):
    def __init__(self):
        super().__init__()
        self.processed = Level("processed", 0.0)

    def forward(self, inflow: Flow = None):
        if inflow is not None:
            self.processed.rate = inflow.value
        else:
            self.processed.rate = 0.0


class MockPipeline(Module):
    def forward(self, *args, **kwargs):
        pass


def test_compile_flat_json():
    canvas_flat = [
        {
            "id": "",
            "class": "MockPipeline",
            "layout": {"x": 0, "y": 0},
            "variables": {},
        },
        {
            "id": "src",
            "class": "MockSource",
            "layout": {"x": 100, "y": 150},
            "variables": {"output_rate": {"class": "Variable", "value": 15.0}},
            "connections": {"flow_inputs": [], "data_inputs": [], "variable_reads": []},
        },
        {
            "id": "buf",
            "class": "MockBuffer",
            "layout": {"x": 400, "y": 150},
            "variables": {
                "mass": {
                    "class": "Level",
                    "value": 100.0,
                    "lower_threshold": 0.0,
                    "upper_threshold": 300.0,
                    "rate": 0.0,
                }
            },
            "connections": {
                "flow_inputs": [
                    {"module": "src", "param": "inflow", "output_index": 0}
                ],
                "data_inputs": [],
                "variable_reads": [],
            },
        },
        {
            "id": "dest",
            "class": "MockConsumer",
            "layout": {"x": 700, "y": 150},
            "variables": {
                "processed": {
                    "class": "Level",
                    "value": 10.0,
                    "lower_threshold": 0.0,
                    "upper_threshold": "Infinity",
                    "rate": 0.0,
                }
            },
            "connections": {
                "flow_inputs": [
                    {"module": "buf", "param": "inflow", "output_index": 0}
                ],
                "data_inputs": [],
                "variable_reads": [],
            },
        },
    ]

    registry = {
        "MockPipeline": MockPipeline,
        "MockSource": MockSource,
        "MockBuffer": MockBuffer,
        "MockConsumer": MockConsumer,
    }

    # Compile the model
    model = compile_canvas_json(canvas_flat, class_registry=registry)

    # 1. Structural Checks
    assert isinstance(model, MockPipeline)
    assert isinstance(model.src, MockSource)
    assert isinstance(model.buf, MockBuffer)
    assert isinstance(model.dest, MockConsumer)

    assert model.src.output_rate.value == 15.0
    assert model.buf.mass.value == 100.0
    assert model.buf.mass.lower_threshold == 0.0
    assert model.buf.mass.upper_threshold == 300.0
    assert model.dest.processed.value == 10.0

    # 2. Layout checks
    assert model.layout == {"x": 0, "y": 0}
    assert model.src.layout == {"x": 100, "y": 150}

    # 3. Connection linkages verification
    assert len(model.buf._flow_dependencies) == 1
    assert model.buf._flow_dependencies[0] is model.src

    # 4. Engine Run verification (checks that dynamically built forward works correctly)
    engine = DRSEngine(model)

    # Step 1: src produces Flow(15.0).
    # buf receives Flow(15.0), sets mass.rate = 15.0 - 5.0 = 10.0, outputs Flow(5.0).
    # dest receives Flow(5.0), sets processed.rate = 5.0.
    engine.run(max_time=1.0)

    # Variables should step according to their compiled rates
    assert model.buf.mass.value == 110.0  # 100.0 + 10.0 * 1.0
    assert model.dest.processed.value == 15.0  # 10.0 + 5.0 * 1.0


def test_compile_tree_json():
    canvas_tree = {
        "class": "MockPipeline",
        "layout": {"x": 0, "y": 0},
        "variables": {},
        "children": {
            "src": {
                "class": "MockSource",
                "layout": {"x": 100, "y": 150},
                "variables": {"output_rate": {"class": "Variable", "value": 25.0}},
                "children": {},
                "connections": {},
            },
            "buf": {
                "class": "MockBuffer",
                "layout": {"x": 400, "y": 150},
                "variables": {
                    "mass": {
                        "class": "Level",
                        "value": 50.0,
                        "lower_threshold": 0.0,
                        "upper_threshold": 200.0,
                        "rate": 0.0,
                    }
                },
                "children": {},
                "connections": {
                    "flow_inputs": [
                        {"module": "src", "param": "inflow", "output_index": 0}
                    ]
                },
            },
        },
    }

    registry = {
        "MockPipeline": MockPipeline,
        "MockSource": MockSource,
        "MockBuffer": MockBuffer,
    }

    model = compile_canvas_json(canvas_tree, class_registry=registry)

    assert isinstance(model, MockPipeline)
    assert model.src.output_rate.value == 25.0
    assert model.buf.mass.value == 50.0
    assert model.buf.mass.upper_threshold == 200.0


def test_compile_rate_equation_ast_resolution():
    canvas_flat = [
        {
            "id": "",
            "class": "MockPipeline",
            "variables": {"control_val": {"class": "Variable", "value": 3.0}},
        },
        {
            "id": "buf",
            "class": "MockPipeline",
            "variables": {
                "mass": {
                    "class": "Level",
                    "value": 50.0,
                    # Reference control_val from parent module
                    "rate": {"equation": "(self.parent.control_val * 4.0)"},
                }
            },
        },
    ]

    registry = {"MockPipeline": MockPipeline, "MockBuffer": MockBuffer}

    model = compile_canvas_json(canvas_flat, class_registry=registry)

    # Verify rate is compiled to Expression
    mass_rate = model.buf.mass._rate
    assert isinstance(mass_rate, Expression)
    assert mass_rate.get_equation() == "(MockPipeline.control_val * 4.0)"

    # Run engine to verify evaluation updates
    engine = DRSEngine(model)
    engine.run(max_time=1.0)

    # rate = 3.0 * 4.0 = 12.0
    # mass = 50.0 + 12.0 = 62.0
    assert model.buf.mass.value == 62.0

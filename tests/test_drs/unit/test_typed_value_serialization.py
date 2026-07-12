"""Tests for typed value deserialization in compile_canvas_json.

Ensures that typed object dicts (e.g. {"__type__": "OperatingMode", "name": "MODE_A"})
are correctly resolved and that invalid/incorrect values fail fast with clear errors.
"""

import pytest
from drs import (
    Module,
    Variable,
    Level,
    DRSEngine,
    DRSConfig,
)
from drs.canvas_compiler import compile_canvas_json
from drs_mining.components import OperatingMode


class MockPipeline(Module):
    def forward(self, *args, **kwargs):
        pass


class TestTypedValueCompilation:
    def test_typed_dict_resolves_successfully(self):
        canvas = {
            "class": "MockPipeline",
            "variables": {
                "mode": {
                    "class": "Variable",
                    "value": {"__type__": "OperatingMode", "name": "MODE_A"},
                }
            },
        }
        registry = {"MockPipeline": MockPipeline}
        model = compile_canvas_json(canvas, class_registry=registry)
        assert isinstance(model.mode.value, OperatingMode)
        assert model.mode.value.name == "MODE_A"

    def test_typed_dict_with_unknown_type_raises_value_error(self):
        canvas = {
            "class": "MockPipeline",
            "variables": {
                "mode": {
                    "class": "Variable",
                    "value": {"__type__": "NonExistentClass", "name": "foo"},
                }
            },
        }
        registry = {"MockPipeline": MockPipeline}
        with pytest.raises(ValueError, match="Class 'NonExistentClass' not found"):
            compile_canvas_json(canvas, class_registry=registry)

    def test_typed_dict_with_invalid_name_raises_value_error(self):
        canvas = {
            "class": "MockPipeline",
            "variables": {
                "mode": {
                    "class": "Variable",
                    "value": {"__type__": "OperatingMode", "name": "MODE_Z"},
                }
            },
        }
        registry = {"MockPipeline": MockPipeline}
        with pytest.raises(ValueError, match="Could not resolve typed object"):
            compile_canvas_json(canvas, class_registry=registry)

    def test_engine_runs_with_typed_value(self):
        canvas = {
            "class": "MockPipeline",
            "variables": {
                "mode": {
                    "class": "Variable",
                    "value": {"__type__": "OperatingMode", "name": "MODE_A"},
                }
            },
        }
        registry = {"MockPipeline": MockPipeline}
        model = compile_canvas_json(canvas, class_registry=registry)
        engine = DRSEngine(model)
        engine.run(max_time=1.0)
        assert isinstance(model.mode.value, OperatingMode)
        assert model.mode.value.name == "MODE_A"


class TestClassResolutionErrors:
    def test_unknown_class_raises_value_error(self):
        canvas = {
            "class": "TotallyBogusModuleName",
            "variables": {},
        }
        with pytest.raises(ValueError, match="not found"):
            compile_canvas_json(canvas)

    def test_unknown_variable_class_raises_value_error(self):
        canvas = {
            "class": "MockPipeline",
            "variables": {
                "x": {"class": "NonExistentVarClass", "value": 1.0},
            },
        }
        registry = {"MockPipeline": MockPipeline}
        with pytest.raises(ValueError, match="Variable class.*not found"):
            compile_canvas_json(canvas, class_registry=registry)

    def test_unknown_variable_class_in_existing_var_uses_new_path(self):
        canvas = {
            "class": "MockPipeline",
            "variables": {
                "x": {"class": "NonExistentVarClass", "value": 1.0},
            },
        }
        registry = {"MockPipeline": MockPipeline}
        with pytest.raises(ValueError, match="Variable class"):
            compile_canvas_json(canvas, class_registry=registry)

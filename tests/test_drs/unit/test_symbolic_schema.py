import pytest
import math
from drs import Module, Variable, Level, Timer, Flow, validate_canvas_json, DRSEngine, DRSConfig
from drs._execution_context import ExecutionContext

class SimpleSubmodule(Module):
    def __init__(self):
        super().__init__()
        self.val_var = Variable("val_var", 10.0)
        self.level_var = Level("level_var", 5.0, rate=0.0)
        self.level_var.lower_threshold = 0.0
        self.level_var.upper_threshold = 100.0
        self.layout = {"x": 50, "y": 100}

    def forward(self, flow_in: Flow = None) -> Flow:
        parent_val = self.parent.control_var.value
        if flow_in is not None:
            self.level_var.rate = flow_in.value * parent_val
        return Flow(5.0)

class SimpleRootModule(Module):
    def __init__(self):
        super().__init__()
        self.sub = SimpleSubmodule()
        self.control_var = Variable("control_var", 2.0)
        self.layout = {"x": 0, "y": 0}

    def forward(self):
        flow = Flow(10.0)
        flow._source = self
        f = self.sub(flow)
        return f


def test_expression_ast_tracing_and_evaluation():
    root = SimpleRootModule()
    
    # Enable tracing
    ExecutionContext.set_tracing(True)
    try:
        # Run forward to trace and build expressions
        root.forward()
        
        # Verify rates are set to Expression instances
        rate_expr = root.sub.level_var._rate
        from drs.variables import Expression
        assert isinstance(rate_expr, Expression)
        
        # Verify equation generation
        eq = rate_expr.get_equation()
        assert eq == "(10.0 * SimpleRootModule.control_var)"
        
        # Verify sources extraction
        sources = rate_expr.get_sources()
        assert len(sources) == 1
        assert any(x is root.control_var for x in sources)
        
        # Verify evaluation
        val = rate_expr.evaluate()
        assert val == 20.0
    finally:
        ExecutionContext.set_tracing(False)


def test_expression_boolean_error():
    x = Variable("x", 5.0)
    y = Variable("y", 10.0)
    
    ExecutionContext.set_tracing(True)
    try:
        expr = x.value > y.value
        with pytest.raises(TypeError, match="Cannot use Expression .* as a boolean"):
            if expr:
                pass
    finally:
        ExecutionContext.set_tracing(False)


def test_expanded_architecture_serialization():
    root = SimpleRootModule()
    
    # Run the model once under tracing to populate dependencies
    ExecutionContext.set_tracing(True)
    try:
        root.forward()
    finally:
        ExecutionContext.set_tracing(False)
        
    serialized = root.to_dict()
    
    # 1. Verify layout
    assert serialized["layout"] == {"x": 0, "y": 0}
    assert serialized["children"]["sub"]["layout"] == {"x": 50, "y": 100}
    
    # 2. Verify variables and expressions
    sub_vars = serialized["children"]["sub"]["variables"]
    assert sub_vars["val_var"]["value"] == 10.0
    assert sub_vars["level_var"]["lower_threshold"] == 0.0
    assert sub_vars["level_var"]["upper_threshold"] == 100.0
    assert sub_vars["level_var"]["rate"] == {"equation": "(10.0 * SimpleRootModule.control_var)"}
    
    # 3. Verify connections and logic flows
    sub_connections = serialized["children"]["sub"]["connections"]
    # flow_inputs relative path to root: root is "", so SimpleRootModule is ""
    assert "" in sub_connections["flow_inputs"]
    assert {"module": "", "variable": "control_var"} in sub_connections["variable_reads"]


def test_json_schema_validator_hierarchical():
    root = SimpleRootModule()
    
    valid_tree = {
        "class": "SimpleRootModule",
        "layout": {"x": 0, "y": 0},
        "variables": {
            "control_var": {"class": "Variable", "value": 2.0}
        },
        "children": {
            "sub": {
                "class": "SimpleSubmodule",
                "layout": {"x": 50, "y": 100},
                "variables": {
                    "val_var": {"class": "Variable", "value": 10.0},
                    "level_var": {"class": "Level", "value": 5.0, "lower_threshold": 0.0, "upper_threshold": 100.0}
                },
                "children": {}
            }
        }
    }
    
    # Should validate successfully
    validate_canvas_json(valid_tree, root)
    
    # Class mismatch check
    invalid_tree_class = dict(valid_tree)
    invalid_tree_class["class"] = "WrongClass"
    with pytest.raises(ValueError, match="class names do not match"):
        validate_canvas_json(invalid_tree_class, root)
        
    # Missing variable check
    invalid_tree_var = {
        "class": "SimpleRootModule",
        "variables": {}, # missing control_var
        "children": valid_tree["children"]
    }
    with pytest.raises(ValueError, match="expected variable 'control_var' is missing"):
        validate_canvas_json(invalid_tree_var, root)

    # Boundary violation check (value < lower_threshold)
    invalid_tree_bound = {
        "class": "SimpleRootModule",
        "variables": {
            "control_var": {"class": "Variable", "value": 2.0}
        },
        "children": {
            "sub": {
                "class": "SimpleSubmodule",
                "variables": {
                    "val_var": {"class": "Variable", "value": 10.0},
                    "level_var": {"class": "Level", "value": -1.0, "lower_threshold": 0.0, "upper_threshold": 100.0}
                },
                "children": {}
            }
        }
    }
    with pytest.raises(ValueError, match="Boundary violation for variable 'level_var'"):
        validate_canvas_json(invalid_tree_bound, root)


def test_json_schema_validator_flat_canvas():
    root = SimpleRootModule()
    
    valid_flat = [
        {
            "id": "root",
            "class": "SimpleRootModule",
            "layout": {"x": 0, "y": 0},
            "variables": {
                "control_var": {"class": "Variable", "value": 2.0}
            }
        },
        {
            "id": "sub",
            "class": "SimpleSubmodule",
            "layout": {"x": 50, "y": 100},
            "variables": {
                "val_var": {"class": "Variable", "value": 10.0},
                "level_var": {"class": "Level", "value": 5.0, "lower_threshold": 0.0, "upper_threshold": 100.0}
            }
        }
    ]
    
    # Should validate successfully
    validate_canvas_json(valid_flat, root)
    
    # Boundary violation in flat array
    invalid_flat_bound = [
        {
            "id": "root",
            "class": "SimpleRootModule",
            "variables": {
                "control_var": {"class": "Variable", "value": 2.0}
            }
        },
        {
            "id": "sub",
            "class": "SimpleSubmodule",
            "variables": {
                "val_var": {"class": "Variable", "value": 10.0},
                "level_var": {"class": "Level", "value": 150.0, "lower_threshold": 0.0, "upper_threshold": 100.0}
            }
        }
    ]
    with pytest.raises(ValueError, match="Boundary violation for variable 'level_var'"):
        validate_canvas_json(invalid_flat_bound, root)

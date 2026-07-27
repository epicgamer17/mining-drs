import pytest
import math
import random
import json
from pathlib import Path
from drs.module import Module
from drs.variables import Level, Timer, Variable
from drs.engine import DRSEngine
from drs.telemetry import Telemetry
from drs.flow import Flow
from drs.serialize import save_checkpoint, load_checkpoint


class CheckpointTestModule(Module):
    def __init__(self):
        super().__init__()
        self.tick_count = 0
        self.custom_state = "initial"
        self.timer = Timer("timer_var", 0.0)
        self.level = Level("level_var", 10.0)

    def initialize_state(self):
        self.tick_count = 0
        self.custom_state = "initialized"

    def forward(self):
        self.tick_count += 1
        self.timer.rate = 1.0
        if self.level.value < 100.0:
            self.level.rate = (2.0, 0.0, 100.0)
        else:
            self.level.rate = 0.0


class DifferentTestModule(Module):
    def __init__(self):
        super().__init__()
        self.other_var = Level("other_var", 0.0)

    def forward(self):
        pass


class CheckpointSource(Module):
    def __init__(self):
        super().__init__()
        self.rate = Variable("rate", 3.0)

    def forward(self):
        return Flow(self.rate.value)


class CheckpointSink(Module):
    def __init__(self):
        super().__init__()
        self.level = Level("level", 0.0)

    def forward(self, inflow=None, control=None):
        flow_value = inflow.value if inflow is not None else 0.0
        control_value = control.value if control is not None else 0.0
        self.level.rate = flow_value + control_value


class CheckpointTopologyModule(Module):
    def __init__(self):
        super().__init__()
        self.source = CheckpointSource()
        self.sink = CheckpointSink()

    def forward(self):
        flow = self.source()
        self.sink(flow, self.source.rate)


def test_checkpoint_save_and_load(tmp_path):
    model = CheckpointTestModule()
    engine = DRSEngine(model, max_step_size=50.0)

    # Run to t=45.0 (which is the first threshold hit for level_var reaching 100)
    res1 = engine.run(max_time=45.0)
    assert engine.current_time == 45.0
    assert engine.step_count == 1
    assert model.level.value == 100.0
    assert model.timer.value == 45.0

    checkpoint_file = tmp_path / "checkpoint.json"
    engine.save_checkpoint(str(checkpoint_file))

    # Check that file exists and contains valid JSON
    assert checkpoint_file.exists()
    with open(checkpoint_file, "r") as f:
        data = json.load(f)
    assert data["drs_version"] == "1.0"
    assert data["engine"]["current_time"] == 45.0
    assert data["engine"]["step_count"] == 1

    # Continue run to max_time=100.0
    engine.run(max_time=100.0)
    assert engine.current_time == 100.0
    assert (
        engine.step_count == 55
    )  # Starts at 45.0 (from previous run) with step_count reset to 0, takes 55 steps of 1.0
    assert model.level.value == 100.0

    # Reload and restore to t=45.0
    engine.load_checkpoint(str(checkpoint_file))
    assert engine.current_time == 45.0
    assert engine.step_count == 1
    assert model.level.value == 100.0
    assert model.timer.value == 45.0
    assert model.custom_state == "initialized"  # Verified custom attributes restore

    # Run again from restored state to max_time=100.0
    res2 = engine.run(max_time=100.0)
    assert engine.current_time == 100.0
    assert engine.step_count == 56  # Starts at step_count=1, takes 55 steps
    assert res2.sim_time == 100.0


def test_checkpoint_branching(tmp_path):
    model = CheckpointTestModule()
    engine = DRSEngine(model, max_step_size=10.0)

    # Run to t=10.0
    engine.run(max_time=10.0)
    checkpoint_file = tmp_path / "branch_pt.json"
    engine.save_checkpoint(str(checkpoint_file))

    # Branch A: continue normal simulation
    engine.run(max_time=30.0)
    val_branch_a = model.level.value

    # Rewind
    engine.load_checkpoint(str(checkpoint_file))
    assert engine.current_time == 10.0
    assert model.level.value == 30.0  # level_var initial=10, rate=2.0, t=10 => value=30

    # Branch B: modify value, then run
    model.level.value = 50.0
    engine.run(max_time=30.0)
    val_branch_b = model.level.value

    # Check that branches are distinct
    assert val_branch_a != val_branch_b
    assert (
        val_branch_b == 90.0
    )  # level_var branch start=50, rate=2.0, dt=20 => value=90


def test_checkpoint_structural_validation(tmp_path):
    model_a = CheckpointTestModule()
    engine_a = DRSEngine(model_a)
    engine_a.run(max_time=10.0)

    checkpoint_file = tmp_path / "struct_test.json"
    engine_a.save_checkpoint(str(checkpoint_file))

    model_b = DifferentTestModule()
    engine_b = DRSEngine(model_b)

    with pytest.raises(
        ValueError, match="Structural mismatch at 'root': class names do not match"
    ):
        engine_b.load_checkpoint(str(checkpoint_file))


def test_checkpoint_rng_restoration(tmp_path):
    model = CheckpointTestModule()
    engine = DRSEngine(model)
    engine.run(max_time=10.0)

    # Set seed and draw random numbers
    random.seed(12345)
    draw_python_1 = [random.random() for _ in range(5)]

    import numpy as np

    np.random.seed(54321)
    draw_numpy_1 = [np.random.random() for _ in range(5)]

    checkpoint_file = tmp_path / "rng_test.json"
    engine.save_checkpoint(str(checkpoint_file))

    # Draw more random numbers
    draw_python_post = [random.random() for _ in range(5)]
    draw_numpy_post = [np.random.random() for _ in range(5)]

    # Restore checkpoint
    engine.load_checkpoint(str(checkpoint_file))

    # Draw again post-restore
    draw_python_restore = [random.random() for _ in range(5)]
    draw_numpy_restore = [np.random.random() for _ in range(5)]

    # Verify exact match
    assert draw_python_post == draw_python_restore
    assert draw_numpy_post == draw_numpy_restore


def test_checkpoint_telemetry(tmp_path):
    model = CheckpointTestModule()
    engine = DRSEngine(model, max_step_size=10.0)
    telemetry = Telemetry(model)
    engine.attach_telemetry(telemetry)

    # Run to t=20.0
    engine.run(max_time=20.0)
    assert len(telemetry.history) > 0

    checkpoint_file = tmp_path / "telemetry_test.json"
    engine.save_checkpoint(str(checkpoint_file))

    len_history_save = len(telemetry.history)
    len_events_save = len(telemetry.events)

    # Run to t=40.0
    engine.run(max_time=40.0)
    assert len(telemetry.history) > len_history_save

    # Restore
    engine.load_checkpoint(str(checkpoint_file))
    assert len(telemetry.history) == len_history_save
    assert len(telemetry.events) == len_events_save


def test_checkpoint_restores_dependency_topology(tmp_path):
    model = CheckpointTopologyModule()
    engine = DRSEngine(model, max_step_size=1.0)
    engine.run(max_time=1.0)

    assert model.sink._flow_dependencies == [model.source]
    assert model.sink._dependencies == [(model.source, model.source.rate)]

    checkpoint_file = tmp_path / "topology_checkpoint.json"
    engine.save_checkpoint(str(checkpoint_file))

    with open(checkpoint_file, "r") as f:
        checkpoint = json.load(f)
    assert checkpoint["topology"]["schema_version"] == 1
    assert {
        "kind": "flow",
        "source": "source",
        "target": "sink",
    } in checkpoint["topology"]["edges"]
    assert {
        "kind": "read",
        "source": "source",
        "target": "sink",
        "variable": "rate",
    } in checkpoint["topology"]["edges"]

    restored_model = CheckpointTopologyModule()
    restored_engine = DRSEngine(restored_model, max_step_size=1.0)
    restored_engine.load_checkpoint(str(checkpoint_file))

    assert restored_model.sink._flow_dependencies == [restored_model.source]
    assert restored_model.sink._dependencies == [
        (restored_model.source, restored_model.source.rate)
    ]

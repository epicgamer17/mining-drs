import pytest
import math
from drs.engine import DRSEngine
from drs.module import Module
from drs.variables import Level


class CountingModule(Module):
    """A minimal module that ticks N times then terminates."""

    def __init__(self, target_ticks: int):
        super().__init__()
        self.target_ticks = target_ticks
        self.tick_count = 0
        self.log = []
        self.var = Level("dummy", 0.0)

    def initialize_state(self):
        self.log.append("init")

    def is_terminating_condition_met(self) -> bool:
        return self.tick_count >= self.target_ticks

    def forward(self):
        self.log.append("forward")
        self.tick_count += 1
        self.var.rate = 1.0


def test_engine_execution_order():
    model = CountingModule(target_ticks=1)
    engine = DRSEngine(model, max_step_size=10.0)
    engine.run()

    assert model.log == ["init", "forward"]
    assert engine.current_time == 1.0


def test_engine_multiple_ticks():
    model = CountingModule(target_ticks=3)
    engine = DRSEngine(model, max_step_size=10.0)
    engine.run()

    assert engine.current_time == 3.0
    assert model.tick_count == 3
    assert model.log.count("init") == 1
    assert model.log.count("forward") == 3


def test_engine_negative_dt_raises_error():
    model = CountingModule(target_ticks=1)
    engine = DRSEngine(model)

    engine._calculate_min_dt = lambda variables: (-1.0, None, True)

    with pytest.raises(ValueError, match="Time delta \\(dt\\) cannot be negative."):
        engine.run()

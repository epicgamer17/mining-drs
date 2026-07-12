import pytest
import pandas as pd
from drs.telemetry import Telemetry
from drs.variables import Level, Timer, Variable


class MockEngine:
    def __init__(self):
        self._variables = {
            "ore_stock": Level("ore_stock", 1000.0),
            "runtime": Timer("runtime", 0.0),
            "trucks_dispatched": Variable("trucks_dispatched", 5),
        }

    def variables(self):
        return iter(self._variables.values())

    def named_modules(self, memo=None, prefix=""):
        yield prefix, self

    @property
    def _modules(self):
        return {}


def test_telemetry_snapshot():
    engine = MockEngine()
    telemetry = Telemetry(engine)

    # Take a snapshot at t=1.0
    telemetry.snapshot(current_time=1.0)

    # Engine updates
    engine._variables["ore_stock"]._update(
        2.0
    )  # Ore stock changes to 1000 + (0 * 2) = 1000.0 (rate is 0)
    engine._variables["ore_stock"].rate = 50.0
    engine._variables["ore_stock"]._update(1.0)  # Ore stock changes to 1050.0
    engine._variables["runtime"]._update(1.0)  # Runtime changes to 1.0
    engine._variables["trucks_dispatched"].value = 6  # Trucks to 6

    # Take a snapshot at t=2.0
    telemetry.snapshot(current_time=2.0)

    history = telemetry.history

    assert len(history) == 2

    # Verify snapshot 1
    assert history[0]["time"] == 1.0
    assert history[0]["ore_stock"] == 1000.0
    assert history[0]["runtime"] == 0.0
    assert history[0]["trucks_dispatched"] == 5

    # Verify snapshot 2
    assert history[1]["time"] == 2.0
    assert history[1]["ore_stock"] == 1050.0
    assert history[1]["runtime"] == 1.0
    assert history[1]["trucks_dispatched"] == 6


def test_telemetry_to_dataframe():
    engine = MockEngine()
    telemetry = Telemetry(engine)
    telemetry.snapshot(current_time=0.0)

    df = telemetry.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert list(df.columns) == ["time", "ore_stock", "runtime", "trucks_dispatched"]

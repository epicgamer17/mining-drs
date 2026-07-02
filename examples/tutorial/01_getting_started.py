"""
01 Getting Started with DRS
===========================
A step-by-step introduction to the Discrete Rate Simulation framework.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Part 1: Core Types ──────────────────────────────────────────────────

from drs.module import drs
from drs.execution_context import ExecutionContext

# Variables hold named state. They are owned by whatever Module creates them.
rate = drs.Variable("extraction_rate", 5000.0)
print(f"1a. Variable value: {rate.value}")

# Levels accumulate over time using a rate (like an integral in dt).
# Only drs.Level has .rate — setting rate on a plain Variable fails fast.
stockpile = drs.Level("ore_stockpile", initial_value=0.0)
stockpile.rate = 5000.0
print(f"1b. Level rate: {stockpile.rate} t/day")

# Timers are Levels that tick at rate=1.0 by default.
clock = drs.Timer("elapsed_days", initial_value=0.0)
clock.rate = 1.0
print(f"1c. Timer initial value: {clock.value}")



class Stockpile(drs.Module):
    """A stockpile that receives ore and feeds a mill."""

    def __init__(self, name: str, initial_mass: float = 0.0):
        super().__init__()
        self.mass = drs.Level(f"{name}_mass", initial_value=initial_mass)
        self.outflow = drs.Variable(f"{name}_outflow", 0.0)

    def forward(self, requested_outflow: float):
        outflow = min(requested_outflow, self.current_mass.value) if self.current_mass.value > 0 else 0.0
        self.mass.rate = self.mass.rate - outflow
        self.outflow.value = outflow
        return outflow


class Mill(drs.Module):
    """A mill that consumes ore from a stockpile."""

    def __init__(self, name: str, max_rate: float):
        super().__init__()
        self.name = name
        self.max_rate = max_rate
        self.total_milled = drs.Level(f"{name}_total_milled", initial_value=0.0)
        self.feed_rate = drs.Variable(f"{name}_feed_rate", 0.0)

    def forward(self, available: float):
        actual = min(available, self.max_rate)
        self.total_milled.rate = actual
        self.feed_rate.value = actual


# ── Part 4: Wiring It Together ─────────────────────────────────────────

class Monitor(drs.Module):
    """Reads the stockpile level — creates a cross-module dependency edge."""

    def __init__(self):
        super().__init__()
        self.observed_level = drs.Variable("observed_level", 0.0)

    def forward(self, stockpile):
        self.observed_level.value = stockpile.current_mass.value


class SimpleMine(drs.Module):
    """A complete mini simulation: mine → stockpile → mill."""

    def __init__(self):
        super().__init__()
        self.stockpile = Stockpile("ore", initial_mass=0.0)
        self.mill = Mill("concentrator", max_rate=6000.0)
        self.monitor = Monitor()
        self.extraction_rate = drs.Variable("mine_rate", 8000.0)

    def forward(self):
        self.stockpile.mass.rate = self.extraction_rate.value
        available = self.stockpile(self.mill.max_rate)
        self.mill(available)
        self.monitor(self.stockpile)


# ── Part 5: Running the Engine ─────────────────────────────────────────

from drs.engine import DRSEngine

model = SimpleMine()
engine = DRSEngine(model, max_step_size=0.5)

print("\n5a. Module tree:")
for path, mod in model.named_modules():
    indent = "  " * path.count(".")
    print(f"{indent}{path or 'root'} ({type(mod).__name__})")

engine.run(max_time=10.0)

print(f"\n5b. After 10 days:")
print(f"    Stockpile mass:   {model.stockpile.current_mass.value:.1f} t")
print(f"    Mill total milled: {model.mill.total_milled.value:.1f} t")
print(f"    Mill feed rate:    {model.mill.feed_rate.value:.1f} t/day")
# Mass balance check: inflow * time = stockpile + milled
total_in = model.extraction_rate.value * engine.current_time
print(f"    Mass balance:      {total_in:.1f} t in = {model.stockpile.current_mass.value + model.mill.total_milled.value:.1f} t out")


# ── Part 6: Threshold-Driven Events ────────────────────────────────────

# Levels have upper_threshold and lower_threshold.
# When a Level's value crosses a threshold, the engine stops dt exactly at the
# crossing point, giving the model a chance to change rates.
#
# This is how DRS models event-driven behavior: set a threshold, and the engine
# guarantees it will stop *at* the boundary.

class BatchProcessor(drs.Module):
    """A simple tank that fills then empties, controlled by thresholds."""

    def __init__(self):
        super().__init__()
        self.tank = drs.Level("tank_level", initial_value=0.0)
        self.cycle_count = drs.Variable("cycles", 0)
        self._filling = True

    def forward(self):
        if self._filling:
            self.tank.rate = 10.0
            self.tank.upper_threshold = 100.0
            if self.tank.value >= 100.0 - 1e-6:
                self._filling = False
                self.cycle_count.value += 1
        else:
            self.tank.rate = -5.0
            self.tank.lower_threshold = 0.0
            if self.tank.value <= 1e-6:
                self._filling = True

    def is_terminating_condition_met(self):
        return self.cycle_count.value >= 3

tank = BatchProcessor()
eng = DRSEngine(tank)
eng.run()
print(f"\n6a. Batch tank after 3 cycles:")
print(f"    Final level: {tank.tank.value:.1f}")
print(f"    Cycles:      {tank.cycle_count.value}")

# The engine automatically calculates dt to hit each threshold exactly.
# Without thresholds, a Level with rate=10 and no upper_threshold would grow unbounded.
# With upper_threshold=100, the engine stops at dt = (100 - current) / 10 each step.

print(f"\n6b. How it works:")
print(f"    dt = (upper_threshold - value) / rate  when rate > 0")
print(f"    dt = (value - lower_threshold) / |rate| when rate < 0")
print(f"    The engine picks the smallest dt across ALL Levels in the model.")


# ── Part 7: Mode Dispatch Convention ───────────────────────────────────

# DRS models often switch between discrete operating modes.
# The convention is: an OperatingMode is a lightweight object with a name.
# A controller decides when to switch, and components read the current mode
# to determine their behavior.

MODE_NAMES = {"HIGH_GEAR": 0, "LOW_GEAR": 1, "OFF": 2}


class OperatingMode:
    __slots__ = ("_name", "_id")
    def __init__(self, name: str):
        self._name = name
        self._id = MODE_NAMES[name]
    @property
    def name(self): return self._name
    def __repr__(self): return f"OperatingMode({self._name})"


MODES = {name: OperatingMode(name) for name in MODE_NAMES}


class Gearbox(drs.Module):
    """A motor that runs at different speeds depending on the operating mode."""

    def __init__(self):
        super().__init__()
        self.output = drs.Level("output", initial_value=0.0)
        self.total_runtime = drs.Timer("runtime")

    def forward(self):
        mode = self.parent.current_mode.value.name
        if mode == "HIGH_GEAR":
            self.output.rate = 100.0
            self.total_runtime.rate = 1.0
        elif mode == "LOW_GEAR":
            self.output.rate = 40.0
            self.total_runtime.rate = 1.0
        else:
            self.output.rate = 0.0
            self.total_runtime.rate = 0.0


class ModeController(drs.Module):
    """Switches between modes based on accumulated output."""

    def __init__(self):
        super().__init__()
        # current_mode is OWNED by the controller — the convention
        self.current_mode = drs.Variable("mode", MODES["HIGH_GEAR"])
        self.gearbox = Gearbox()

    def forward(self):
        self.gearbox()
        g = self.gearbox
        mode_name = self.current_mode.value.name
        # Read gearbox output (cross-module read → dependency edge)
        if mode_name == "HIGH_GEAR" and g.output.value >= 200.0:
            self.current_mode.value = MODES["LOW_GEAR"]
        elif mode_name == "LOW_GEAR" and g.output.value >= 500.0:
            self.current_mode.value = MODES["OFF"]


ctrl = ModeController()
eng2 = DRSEngine(ctrl, max_step_size=1.0)
eng2.run(max_time=20.0)
print(f"\n7a. Mode dispatch after 20 days:")
print(f"    Final output:   {ctrl.gearbox.output.value:.1f}")
print(f"    Total runtime:  {ctrl.gearbox.total_runtime.value:.1f}")
print(f"    Mode sequence:  HIGH_GEAR → LOW_GEAR → OFF")
print(f"    Insight: Controller owns current_mode; Gearbox reads via parent")


# ── Part 8: Dependency Graph ───────────────────────────────────────────

# After running, each module records which cross-module Variables it read.
# These edges reveal the implicit communication structure.

print(f"\n8a. SimpleMine dependency graph ({len(model.get_dependency_graph())} edges):")
for owner, var in model.get_dependency_graph():
    print(f"    {type(owner).__name__}.{var.name}")

print(f"\n8b. ModeController dependency graph ({len(ctrl.get_dependency_graph())} edges):")
for owner, var in ctrl.get_dependency_graph():
    print(f"    {type(owner).__name__}.{var.name}")


# ── Part 9: Fail-Fast Guardrails ──────────────────────────────────────

print("\n9. Fail-fast guardrails:")

# Only Level has .rate:
try:
    rate.value = 100
    _ = rate.rate
except AttributeError as e:
    print(f"   Variable.rate getter: {e}")

try:
    rate.rate = 200
except AttributeError as e:
    print(f"   Variable.rate setter: {e}")

# Cross-module mutations are blocked during forward():
class BadActor(drs.Module):
    def forward(self):
        model.stockpile.current_mass.value = 0.0

bad = BadActor()
try:
    bad()
except RuntimeError as e:
    print(f"   Cross-module mutation blocked: OK")


# ── Part 10: Quick Reference ───────────────────────────────────────────

# Variable     → named state, no rate, no accumulation
# Level        → accumulates over dt, has .rate + thresholds
# Timer        → Level that ticks at rate=1.0 by default
# Expression   → AST from operator overloading (tracing mode)
# Module       → container with __setattr__ auto-registration
# DRSEngine    → runs loop: forward → calc dt → update → repeat
# ExecutionContext → tracks which module is currently executing
# Upper/lower thresholds → tell engine WHERE to stop next dt
# OperatingMode → lightweight dispatch object, stored in Variable
# Dependency graph → (owner, var) pairs recorded from cross-module reads

print("\nDone! The framework is working as expected.")

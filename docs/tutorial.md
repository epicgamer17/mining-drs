# Getting Started with DRS

Welcome to the **Discrete Rate Simulation (DRS)** framework! This step-by-step guide will walk you through the core concepts, from defining variables to running a full simulation with threshold-driven events.

---

## 1. Core Types

At the heart of the framework are three primary data types that hold your simulation's state: `Variable`, `Level`, and `Timer`.

```python
from drs.module import drs

# Variables hold named state. They are owned by whatever Module creates them.
rate = drs.Variable("extraction_rate", 5000.0)
print(f"Variable value: {rate.value}")

# Levels accumulate over time using a rate (like an integral in dt).
# Only drs.Level has .rate — setting rate on a plain Variable fails fast.
stockpile = drs.Level("ore_stockpile", initial_value=0.0)
stockpile.rate = 5000.0
print(f"Level rate: {stockpile.rate} t/day")

# Timers are Levels that tick at rate=1.0 by default.
clock = drs.Timer("elapsed_days", initial_value=0.0)
clock.rate = 1.0
print(f"Timer initial value: {clock.value}")
```

---

## 2. Defining Modules

A `Module` is the fundamental building block of your simulation, heavily inspired by `nn.Module` in PyTorch. A module owns its state (`Variables` and `Levels`) and defines its physics in a `forward()` pass.

Here, we define a physical `Stockpile` and a `Mill`:

```python
class Stockpile(drs.Module):
    """A stockpile that receives ore and feeds a mill."""

    def __init__(self, name: str, initial_mass: float = 0.0):
        super().__init__()
        self.mass = drs.Level(f"{name}_mass", initial_value=initial_mass)
        self.outflow = drs.Variable(f"{name}_outflow", 0.0)

    def forward(self, requested_outflow: float):
        outflow = min(requested_outflow, self.mass.value) if self.mass.value > 0 else 0.0
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
```

---

## 3. Wiring It Together

You can compose modules into larger systems. When a module reads a variable owned by another module, the framework automatically records this as a dependency edge, implicitly building your physical graph!

```python
class Monitor(drs.Module):
    """Reads the stockpile level — creates a cross-module dependency edge."""

    def __init__(self):
        super().__init__()
        self.observed_level = drs.Variable("observed_level", 0.0)

    def forward(self, stockpile):
        self.observed_level.value = stockpile.mass.value


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
```

---

## 4. Running the Engine

With your model built, pass it to the `DRSEngine` to run the simulation loop.

```python
from drs.engine import DRSEngine

model = SimpleMine()
engine = DRSEngine(model, max_step_size=0.5)

# Run for 10 simulated days
engine.run(max_time=10.0)

print(f"After 10 days:")
print(f"Stockpile mass: {model.stockpile.mass.value:.1f} t")
print(f"Mill total milled: {model.mill.total_milled.value:.1f} t")
print(f"Mill feed rate: {model.mill.feed_rate.value:.1f} t/day")
```

---

## 5. Threshold-Driven Events

The true power of DRS is that it **jumps in time** to the exact moment an event occurs. You set thresholds on a `Level`, and the engine guarantees it will stop exactly at the boundary so your model can react and change rates.

```python
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
            
            # Stop filling when full
            if self.tank.value >= 100.0 - 1e-6:
                self._filling = False
                self.cycle_count.value += 1
        else:
            self.tank.rate = -5.0
            self.tank.lower_threshold = 0.0
            
            # Start filling when empty
            if self.tank.value <= 1e-6:
                self._filling = True

    def is_terminating_condition_met(self):
        return self.cycle_count.value >= 3

tank = BatchProcessor()
eng = DRSEngine(tank)
eng.run()
```

### How Time Jumping Works:
The engine calculates the next time step (`dt`) by looking at all active thresholds across the entire model:
- `dt = (upper_threshold - value) / rate` (when rate > 0)
- `dt = (value - lower_threshold) / |rate|` (when rate < 0)

The engine picks the **smallest dt** across all Levels, safely advancing time to the very next event.

---

## 6. Mode Dispatch Convention

Complex simulations often switch between predefined strategies. The convention in Mining-DRS is to use lightweight `OperatingMode` objects. A Controller sets the mode, and sub-components read the mode to determine their behavior.

```python
class OperatingMode:
    def __init__(self, name: str):
        self._name = name
    @property
    def name(self): return self._name

MODES = {
    "HIGH_GEAR": OperatingMode("HIGH_GEAR"), 
    "LOW_GEAR": OperatingMode("LOW_GEAR"), 
    "OFF": OperatingMode("OFF")
}

class ModeController(drs.Module):
    """Switches between modes based on accumulated output."""

    def __init__(self):
        super().__init__()
        # The Controller owns the mode state
        self.current_mode = drs.Variable("mode", MODES["HIGH_GEAR"])
        self.gearbox = Gearbox()

    def forward(self):
        self.gearbox()
        mode_name = self.current_mode.value.name
        
        # Switch mode based on gearbox output
        if mode_name == "HIGH_GEAR" and self.gearbox.output.value >= 200.0:
            self.current_mode.value = MODES["LOW_GEAR"]
        elif mode_name == "LOW_GEAR" and self.gearbox.output.value >= 500.0:
            self.current_mode.value = MODES["OFF"]
```

---

## 7. Fail-Fast Guardrails

To prevent silent physics violations, the engine will aggressively stop you if you try to take shortcuts. 

For instance, trying to manually mutate a variable owned by a different module will result in a `RuntimeError`:

```python
class BadActor(drs.Module):
    def forward(self):
        # Cross-module mutation is strictly blocked during forward()
        model.stockpile.mass.value = 0.0 
```

Enjoy building robust, blazingly fast simulations!

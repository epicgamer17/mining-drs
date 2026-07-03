"""
Tutorial 5: Telemetry & Custom Callbacks
=========================================
Illustrating metric registration, custom callbacks, and progress indicators.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import drs
from drs import Module, Level, Variable
from drs.telemetry import Telemetry
from drs.engine import DRSEngine
from drs.callbacks import Callback

class GrindingMill(Module):
    def __init__(self):
        super().__init__()
        self.ore = Level("ore_milled", initial_value=0.0)
        self.power_draw = Variable("power_draw_kw", 450.0)

    def forward(self):
        # Stop milling if we have reached our target of 500 tons
        if self.ore.value >= 500.0 - 1e-6:
            self.ore.rate = 0.0
        else:
            self.ore.rate = 100.0
            self.ore.upper_threshold = 500.0


# --- 1. Custom Telemetry Metrics ---
print("--- 1. Testing Telemetry & Custom Derived Metrics ---")
model = GrindingMill()
engine = DRSEngine(model)
telemetry = Telemetry(model)
engine.attach_telemetry(telemetry)

# Register custom metric: total power usage efficiency (kWh / ton milled)
# Signature: calc_fn(current_time, model, state, history) -> float
def calc_energy_efficiency(t, mod, state, history):
    milled_tons = mod.ore.value
    power = mod.power_draw.value
    total_energy_kwh = power * t
    return total_energy_kwh / milled_tons if milled_tons > 0 else 0.0

telemetry.register_metric("kwh_per_ton", calc_energy_efficiency)

engine.run(max_time=8.0)

df = telemetry.to_dataframe()
print("Telemetry Output with Derived Metric:")
print(df[["time", "ore_milled", "kwh_per_ton"]])


# --- 2. Custom Lifecycle Callbacks ---
print("\n--- 2. Testing Custom Lifecycle Callbacks ---")

class ThresholdLoggerCallback(Callback):
    def on_simulation_start(self, engine: "DRSEngine") -> None:
        print("[CALLBACK] Simulation is starting!")

    def on_threshold(self, engine: "DRSEngine", trigger_var: "Variable", is_upper: bool) -> None:
        direction = "upper" if is_upper else "lower"
        threshold = trigger_var.upper_threshold if is_upper else trigger_var.lower_threshold
        print(f"[CALLBACK] t={engine.current_time:.2f}: Variable '{trigger_var.name}' hit {direction} threshold of {threshold}!")

    def on_complete(self, engine: "DRSEngine", result) -> None:
        print(f"[CALLBACK] Simulation complete! Total steps: {result.steps}")

model_callback = GrindingMill()
engine_callback = DRSEngine(model_callback, callbacks=[ThresholdLoggerCallback()])
engine_callback.run(max_time=10.0)


# --- 3. Built-in Progress Bar Callback ---
print("\n--- 3. Testing Built-in Progress Bar Callback ---")
model_progress = GrindingMill()
# Enables visual Rich CLI progress bar
engine_progress = DRSEngine(model_progress, progress_bar=True)
engine_progress.run(max_time=20.0)

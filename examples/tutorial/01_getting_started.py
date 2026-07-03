"""
Tutorial 1: Introduction to DRS
================================
A step-by-step introduction to the Discrete Rate Simulation framework.
"""

import sys
import os

# Allow importing drs from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import drs
from drs import Flow, Variable
from drs.engine import DRSEngine

# --- 1. Core Types ---
print("--- 1. Core Types ---")

# Variables hold named state. They are owned by whatever Module creates them.
rate = drs.Variable("extraction_rate", 5000.0)
print(f"Variable value: {rate.value} t/day")

# Levels accumulate over time using a rate (like an integral in dt).
stockpile = drs.Level("ore_stockpile", initial_value=0.0)
stockpile.rate = 5000.0
print(f"Level rate: {stockpile.rate} t/day")

# Timers are Levels that tick at rate=1.0 by default.
clock = drs.Timer("elapsed_days", initial_value=0.0)
print(f"Timer initial value: {clock.value}")


# --- 2. Defining Modules ---
print("\n--- 2. Defining Modules ---")

class Stockpile(drs.Module):
    """A stockpile that receives ore and feeds a mill."""

    def __init__(self, name: str, initial_mass: float = 0.0):
        super().__init__()
        self.mass = drs.Level(f"{name}_mass", initial_value=initial_mass)
        self.outflow = drs.Variable(f"{name}_outflow", 0.0)

    def forward(self, inflow_rate: Flow, requested_outflow: Variable):
        # Determine actual outflow based on what is physically available in the stockpile
        outflow = min(requested_outflow.value, self.mass.value) if self.mass.value > 0 else 0.0
        
        # Physics update: rate of accumulation is inflow minus outflow
        self.mass.rate = inflow_rate.value - outflow
        self.outflow.value = outflow
        return Flow(outflow)


class Mill(drs.Module):
    """A mill that consumes ore from a stockpile."""

    def __init__(self, name: str, max_rate: float):
        super().__init__()
        self.name = name
        self.max_rate = drs.Variable(f"{name}_max_rate", max_rate)
        self.total_milled = drs.Level(f"{name}_total_milled", initial_value=0.0)
        self.feed_rate = drs.Variable(f"{name}_feed_rate", 0.0)

    def forward(self, available: Flow):
        # Mill operates at either the maximum capacity or the available feed rate
        actual = min(available.value, self.max_rate.value)
        self.total_milled.rate = actual
        self.feed_rate.value = actual


# --- 3. Wiring It Together ---
print("\n--- 3. Wiring It Together ---")

class SimpleMine(drs.Module):
    """A complete mini simulation: mine → stockpile → mill."""

    def __init__(self):
        super().__init__()
        self.stockpile = Stockpile("ore", initial_mass=100.0)
        self.mill = Mill("concentrator", max_rate=6000.0)
        self.extraction_rate = drs.Variable("mine_rate", 8000.0)

    def forward(self):
        # 1. Wrap extraction rate in a Flow
        inflow = Flow(self.extraction_rate.value)
        
        # 2. Call stockpile with inflow and the mill's max_rate Variable
        available_outflow = self.stockpile(inflow, self.mill.max_rate)
        
        # 3. Feed the mill with the available outflow Flow
        self.mill(available_outflow)


# --- 4. Running the Engine ---
print("\n--- 4. Running the Engine ---")

# 1. Instantiate the model
model = SimpleMine()

# 2. Give it to the engine
engine = DRSEngine(model, max_step_size=0.5)

# 3. Run for 10 simulated days
result = engine.run(max_time=10.0)

# 4. Check the results
print(result.summary())
print(f"Stockpile mass: {model.stockpile.mass.value:.1f} t")
print(f"Mill total milled: {model.mill.total_milled.value:.1f} t")
print(f"Engine steps: {result.steps}")

"""
Tutorial 4: Checkpointing & State Serialization
================================================
Illustrating how to save and load state, export architecture, and checkpoint.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import drs
from drs import Module, Variable, Level
from drs.engine import DRSEngine
from drs.serialize import (
    save_state,
    load_state,
    export_architecture,
    save_checkpoint,
    load_checkpoint,
)

# Setup temporary file paths in workspace
STATE_FILE = "stockpile_state_temp.json"
ARCH_FILE = "architecture_temp.json"
CHECKPOINT_FILE = "day_50_checkpoint_temp.json"

# --- 1. Variable StateDict and Architecture ---
print("--- 1. Variable StateDict and Architecture ---")

class Stockpile(drs.Module):
    def __init__(self):
        super().__init__()
        self.capacity = drs.Variable("capacity", 500.0)
        self.mass = drs.Level("mass", initial_value=120.0)

    def forward(self):
        pass

model = Stockpile()
print("Original State Dict:", model.state_dict())

# Save variable values
save_state(model, STATE_FILE)
print(f"Saved state to {STATE_FILE}")

# Export structure
export_architecture(model, ARCH_FILE)
print(f"Exported architecture to {ARCH_FILE}")

# Modify variable value
model.mass.value = 450.0
print("Modified Mass:", model.mass.value)

# Restore state
load_state(model, STATE_FILE)
print("Restored Mass (Should be 120.0):", model.mass.value)


# --- 2. Full Engine Checkpointing & Branching ---
print("\n--- 2. Full Engine Checkpointing & Branching ---")

class MineSystem(Module):
    def __init__(self):
        super().__init__()
        self.stockpile = Level("ore", initial_value=0.0)
        self.extraction_rate = Variable("mine_rate", 100.0)

    def forward(self):
        self.stockpile.rate = self.extraction_rate.value

# Setup model and engine
mine_model = MineSystem()
engine = DRSEngine(mine_model)

# Run to Day 50
engine.run(max_time=50.0)
print(f"Day 50 Stockpile Level: {mine_model.stockpile.value}")

# Save the checkpoint
engine.save_checkpoint(CHECKPOINT_FILE)
print(f"Saved engine checkpoint to {CHECKPOINT_FILE}")

# --- Branch A: Mining at 200 t/day from Day 50 to Day 100 ---
print("\n--- Executing Branch A ---")
mine_model.extraction_rate.value = 200.0
result_a = engine.run(max_time=100.0)
print(f"Branch A Final Level (Day 100): {mine_model.stockpile.value}")

# --- Branch B: Loading Checkpoint and mining at 50 t/day ---
print("\n--- Executing Branch B ---")
mine_model_b = MineSystem()
engine_b = DRSEngine(mine_model_b)
# Load checkpoint into the new engine and model
engine_b.load_checkpoint(CHECKPOINT_FILE)

print(f"Restored Engine Time: {engine_b.current_time}")
print(f"Restored Stockpile Level: {engine_b.model.stockpile.value}")

# Modify variable in the restored model
engine_b.model.extraction_rate.value = 50.0
result_b = engine_b.run(max_time=100.0)
print(f"Branch B Final Level (Day 100): {engine_b.model.stockpile.value}")


# Clean up temporary files
for f in [STATE_FILE, ARCH_FILE, CHECKPOINT_FILE]:
    if os.path.exists(f):
        os.remove(f)
print("\nCleaned up temporary checkpoint files.")

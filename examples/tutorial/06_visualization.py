"""
Tutorial 6: Visualizing Module Graphs
======================================
Illustrating how to generate structural and dataflow reports from models.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import drs
from drs import Module, Level, Variable, Flow
from drs.engine import DRSEngine
from drs.vis.module_graph import save_module_graph_report

class MineFace(Module):
    def __init__(self):
        super().__init__()
        self.capacity = Variable("daily_capacity", 5000.0)

    def forward(self):
        # Returns a Flow rate representing material extracted
        return Flow(self.capacity.value)

class Crusher(Module):
    def __init__(self):
        super().__init__()
        self.pile = Level("crusher_pile", initial_value=0.0)

    def forward(self, inflow_flow):
        # Read the inflow rate
        self.pile.rate = inflow_flow.value
        
        # Outflow is rate of feeding the mill (e.g. 4000.0 if stockpile has enough)
        feed = min(4000.0, self.pile.value) if self.pile.value > 0 else 0.0
        self.pile.rate -= feed
        return Flow(feed)

class ConcentratorMill(Module):
    def __init__(self):
        super().__init__()
        self.milled = Level("milled_ore", initial_value=0.0)

    def forward(self, feed_flow):
        # Accumulate milled tons
        self.milled.rate = feed_flow.value

class IntegratedOperation(Module):
    def __init__(self):
        super().__init__()
        self.face = MineFace()
        self.crusher = Crusher()
        self.mill = ConcentratorMill()

    def forward(self):
        extracted = self.face()
        crushed_feed = self.crusher(extracted)
        self.mill(crushed_feed)

print("--- 1. Running Model to Record Dependency Edges ---")
model = IntegratedOperation()
engine = DRSEngine(model)
engine.run(max_time=1.0)

print("\n--- 2. Generating and Saving Module Graph Report ---")
# This will write "module_graph_report.md" and "module_graph_report.png"
report_path = save_module_graph_report(model, path_prefix="module_graph_report", show_vars=True)
print(f"Generated report at: {report_path}")

# Check files were created
for f in ["module_graph_report.md", "module_graph_report.png"]:
    if os.path.exists(f):
        print(f"  [FOUND] {f}")
        # Clean them up to keep the directory clean, or keep them if they are useful
        os.remove(f)
print("Cleaned up generated graph files from test execution.")

"""
Tutorial 7: Operating Modes & Controllers
==========================================
Illustrating clean architectural separation and state-dependent mode controllers.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import drs
from drs import Module, Level, Variable
from drs.engine import DRSEngine
from drs.telemetry import Telemetry

# --- Step 1: The Physical Components ---
class PhysicalStockpile(Module):
    def __init__(self):
        super().__init__()
        self.mass = Level("mass", initial_value=250.0)

    def forward(self, inflow: Variable, outflow: Variable):
        # Read the values from the controller variables
        self.mass.rate = inflow.value - outflow.value


class PhysicalMill(Module):
    def __init__(self):
        super().__init__()
        self.total_milled = Level("total_milled", initial_value=0.0)

    def forward(self, feed_rate: Variable):
        self.total_milled.rate = feed_rate.value


# --- Step 2: The Mode Controller ---
class BlendingController(Module):
    def __init__(self, stockpile: PhysicalStockpile, mill: PhysicalMill):
        super().__init__()
        self.stockpile = stockpile
        self.mill = mill
        
        # State variable to track the active decision mode
        self.active_mode = Variable("mode", "NORMAL")
        
        # Output target variables
        self.mine_target = Variable("mine_target", 80.0)
        self.mill_target = Variable("mill_target", 100.0)

    def forward(self):
        # 1. Read physical state
        pile_mass = self.stockpile.mass.value
        current_mode = self.active_mode.value

        # 2. Evaluate state transitions and change active mode (using epsilons for threshold detection)
        if current_mode == "NORMAL":
            if pile_mass <= 100.0 + 1e-6:
                self.active_mode.value = "CONTINGENCY"
                print(f"[CONTROLLER] t={self.current_time():.2f}: Stockpile low ({pile_mass:.1f}t). Switching to CONTINGENCY.")
        elif current_mode == "CONTINGENCY":
            if pile_mass >= 200.0 - 1e-6:
                self.active_mode.value = "NORMAL"
                print(f"[CONTROLLER] t={self.current_time():.2f}: Stockpile recovered ({pile_mass:.1f}t). Returning to NORMAL.")
            elif pile_mass <= 10.0 + 1e-6:
                self.active_mode.value = "SHUTDOWN"
                print(f"[CONTROLLER] t={self.current_time():.2f}: Stockpile critically empty ({pile_mass:.1f}t). Switching to SHUTDOWN.")
        elif current_mode == "SHUTDOWN":
            if pile_mass >= 150.0 - 1e-6:
                self.active_mode.value = "CONTINGENCY"
                print(f"[CONTROLLER] t={self.current_time():.2f}: Stockpile partially recovered ({pile_mass:.1f}t). Switching to CONTINGENCY.")

        # 3. Apply target dynamics based on active mode
        if self.active_mode.value == "NORMAL":
            self.mine_target.value = 80.0
            self.mill_target.value = 100.0
        elif self.active_mode.value == "CONTINGENCY":
            self.mine_target.value = 100.0 # surge mining
            self.mill_target.value = 40.0  # throttle milling
        elif self.active_mode.value == "SHUTDOWN":
            self.mine_target.value = 100.0
            self.mill_target.value = 0.0   # shutdown milling

        # 4. Bind thresholds to stockpiles to trigger re-evaluations
        if self.active_mode.value == "NORMAL":
            self.stockpile.mass.lower_threshold = 100.0
            self.stockpile.mass.upper_threshold = 500.0
        elif self.active_mode.value == "CONTINGENCY":
            self.stockpile.mass.lower_threshold = 10.0
            self.stockpile.mass.upper_threshold = 200.0
        elif self.active_mode.value == "SHUTDOWN":
            self.stockpile.mass.lower_threshold = -drs.math.inf
            self.stockpile.mass.upper_threshold = 150.0

    def current_time(self) -> float:
        from drs._execution_context import ExecutionContext
        engine = ExecutionContext.get_engine()
        return engine.current_time if engine else 0.0


# --- Step 3: Integrating the Top-Level System ---
class MiningSystem(Module):
    def __init__(self):
        super().__init__()
        self.stockpile = PhysicalStockpile()
        self.mill = PhysicalMill()
        self.controller = BlendingController(self.stockpile, self.mill)

    def forward(self):
        # 1. Run the controller first to update mode and routing decisions
        self.controller()
        
        # 2. Propagate targets to physical components (pass the Variable objects themselves!)
        self.stockpile(self.controller.mine_target, self.controller.mill_target)
        self.mill(self.controller.mill_target)


# --- Running the Simulation ---
print("--- Starting Operating Mode Simulation ---")
model = MiningSystem()
engine = DRSEngine(model)
telemetry = Telemetry(model)
engine.attach_telemetry(telemetry)

# Run for 20 simulated hours
result = engine.run(max_time=20.0)
print(result.summary())
result.print_event_timeline()

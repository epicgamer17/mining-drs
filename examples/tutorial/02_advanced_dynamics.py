"""
Tutorial 2: Advanced Core Dynamics & Guardrails
================================================
Illustrating threshold-driven events and execution guardrails.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import drs
import math
from drs.engine import DRSEngine
from drs.exceptions import StateMutationError, ThresholdConfigurationError
from drs.config import EngineConfig
from drs.telemetry import Telemetry

# --- 1. Threshold-Driven Event Loop ---
print("--- 1. Running a BatchTank with Thresholds ---")

class BatchTank(drs.Module):
    def __init__(self):
        super().__init__()
        self.tank = drs.Level("tank_level", initial_value=0.0)
        self.cycle_count = drs.Variable("cycles", 0)
        self._filling = True

    def forward(self):
        if self._filling:
            # Set rate and upper threshold. Format: (rate, lower_threshold, upper_threshold)
            self.tank.rate = (10.0, -math.inf, 100.0)
            
            # Switch to emptying when full
            if self.tank.value >= 100.0 - 1e-6:
                self._filling = False
                self.cycle_count.value += 1
                print(f"[MODEL] Tank full at t={self.current_time():.2f}! Switching to EMPTYING.")
        else:
            # Set rate and lower threshold
            self.tank.rate = (-20.0, 0.0, math.inf)
            
            # Switch to filling when empty
            if self.tank.value <= 1e-6:
                self._filling = True
                print(f"[MODEL] Tank empty at t={self.current_time():.2f}! Switching to FILLING.")

    def current_time(self) -> float:
        from drs._execution_context import ExecutionContext
        engine = ExecutionContext.get_engine()
        return engine.current_time if engine else 0.0

    def is_terminating_condition_met(self) -> bool:
        return self.cycle_count.value >= 2


tank_model = BatchTank()
# Attach telemetry to capture threshold-hit events in engine log
telemetry = Telemetry(tank_model)
engine = DRSEngine(tank_model)
engine.attach_telemetry(telemetry)

result = engine.run(max_time=50.0)
print(result.summary())
result.print_event_timeline()


# --- 2. Guardrail 1: The Rule of Ownership ---
print("\n--- 2. Guardrail 1: Testing Cross-Module Mutation Block ---")

class TargetModule(drs.Module):
    def __init__(self):
        super().__init__()
        self.val = drs.Variable("target_val", 0.0)
        
    def forward(self):
        pass

class BadActor(drs.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def forward(self):
        print("[BadActor] Attempting to mutate variable owned by TargetModule directly...")
        # This will fail because BadActor doesn't own self.target.val
        self.target.val.value = 100.0

target = TargetModule()
bad_actor = BadActor(target)

class System(drs.Module):
    def __init__(self, target, bad):
        super().__init__()
        self.target = target
        self.bad = bad
    def forward(self):
        self.target()
        self.bad()

sys_model = System(target, bad_actor)
engine = DRSEngine(sys_model)

try:
    engine.run(max_time=1.0)
except StateMutationError as e:
    print(f"Caught expected StateMutationError:\n{e}")


# --- 3. Guardrail 2: Rate Conflict Protection ---
print("\n--- 3. Guardrail 2: Testing Rate Conflict Block ---")

class SharedResource(drs.Module):
    def __init__(self):
        super().__init__()
        self.level = drs.Level("shared_level", initial_value=0.0)
    def forward(self):
        pass

class WriterA(drs.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
    def forward(self):
        self.shared.level.rate = 10.0

class WriterB(drs.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
    def forward(self):
        self.shared.level.rate = -5.0

class ConflictingSystem(drs.Module):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared
        self.writer_a = WriterA(shared)
        self.writer_b = WriterB(shared)
    def forward(self):
        self.shared()
        self.writer_a()
        self.writer_b()

conflict_model = ConflictingSystem(SharedResource())
engine = DRSEngine(conflict_model)

try:
    engine.run(max_time=1.0)
except StateMutationError as e:
    print(f"Caught expected StateMutationError (Rate Conflict):\n{e}")


# --- 4. Guardrail 3: Orphaned Threshold Check ---
print("\n--- 4. Guardrail 3: Testing Orphaned Threshold Check (Strict Mode) ---")

class OrphanedTank(drs.Module):
    def __init__(self):
        super().__init__()
        self.tank = drs.Level("tank_level", initial_value=10.0)

    def forward(self):
        # Setting threshold, but rate is 0.0!
        self.tank.rate = 0.0
        self.tank.lower_threshold = 0.0

orphan_model = OrphanedTank()
strict_config = EngineConfig(strict_mode=True)
engine = DRSEngine(orphan_model, config=strict_config)

try:
    engine.run(max_time=5.0)
except ThresholdConfigurationError as e:
    print(f"Caught expected ThresholdConfigurationError in strict mode:\n{e}")

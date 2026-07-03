"""
Tutorial 3: Streaming Inputs & Data Sources
============================================
Illustrating how to use DataSource and DataPoint for non-homogeneous flows.
"""

import sys
import os
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import drs
from drs import DataSource, DataPoint, Flow
from drs.engine import DRSEngine
from drs.telemetry import Telemetry

# --- 1. Subclassing DataSource ---
class MineTruckSource(DataSource):
    """Generates continuous truck arrivals with varying mass and metal grade."""

    def __init__(self, seed: int = 42):
        super().__init__()
        self.rng = random.Random(seed)
        self.total_trucks = 5
        self.current_truck = 0

    def __next__(self) -> DataPoint:
        if self.current_truck >= self.total_trucks:
            raise StopIteration
        
        self.current_truck += 1
        
        # Generate random characteristics for the truck's load
        mass = self.rng.uniform(40.0, 60.0)    # tons of rock
        grade = self.rng.uniform(0.5, 1.8)     # % metal grade
        
        return DataPoint(mass=mass, grade=grade)


# --- 2. Processing Streams in a Module ---
class OreConveyor(drs.Module):
    def __init__(self, source: MineTruckSource):
        super().__init__()
        self.source = source
        
        # Stockpile state
        self.ore_mass = drs.Level("ore_mass", initial_value=0.0)
        self.metal_mass = drs.Level("metal_mass", initial_value=0.0)
        
        # Current batch properties
        self.active_grade = drs.Variable("active_grade", 0.0)

    def forward(self):
        # Check if the stockpile is empty; if so, load the next batch
        if self.ore_mass.value <= 1e-6:
            try:
                # Retrieve next truck from source
                batch = next(self.source)
                
                # Set stockpile contents instantly
                self.ore_mass.value = batch.mass
                self.metal_mass.value = batch.mass * (batch.grade / 100.0)
                self.active_grade.value = batch.grade
                
                print(f"[CONVEYOR] Loaded truck {self.source.current_truck}: mass={batch.mass:.1f}t, grade={batch.grade:.2f}%")
            except StopIteration:
                # No more data left in stream
                self.ore_mass.rate = 0.0
                self.metal_mass.rate = 0.0
                return None

        # Conveyor discharge rate (10 tons per hour)
        discharge_rate = min(10.0, self.ore_mass.value)
        
        # Calculate grade of outflow (uniform blending assumption)
        grade_fraction = self.metal_mass.value / self.ore_mass.value if self.ore_mass.value > 0 else 0.0
        metal_discharge = discharge_rate * grade_fraction
        
        self.ore_mass.rate = -discharge_rate
        self.metal_mass.rate = -metal_discharge
        
        # Bound limits so we stop exactly when empty
        self.ore_mass.lower_threshold = 0.0
        
        # Return outflow as a Flow object containing mass and average grade
        return Flow((discharge_rate, grade_fraction * 100.0))


# --- 3. Running the Simulation ---
print("--- Running Data Stream Simulation ---")
source = MineTruckSource()
conveyor = OreConveyor(source)

engine = DRSEngine(conveyor)
telemetry = Telemetry(conveyor)
engine.attach_telemetry(telemetry)

result = engine.run(max_time=100.0)
print(result.summary())

# Display the history DataFrame
df = telemetry.to_dataframe()
print("\nSimulation Telemetry Log:")
print(df[["time", "ore_mass", "active_grade"]].head(20))

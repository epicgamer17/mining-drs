# Mining-DRS: Discrete Rate Simulation Framework

Welcome to **Mining-DRS**! Whether you are a mining engineer who has never written a line of Python, or a Python developer who thinks "milling" has something to do with flour, this guide is for you.

This is an object-oriented framework for simulating complex supply chains where material flows continuously over time. We specifically focus on mining operations, but the underlying engine can model any system with continuous flows (like water pipes, electrical grids, or traffic). This library is a Python-based, using concepts similar to PyTorch. 

---

## 1. The Basics: Demystifying the Jargon

### For the Non-Programmers: What is a "DRS"?
Most simulations use a **Fixed-Step** approach. Imagine watching a bathtub fill up, and you check it exactly once every minute. 
- *The Problem:* What if it overflows at 1 minute and 30 seconds? You won't notice until minute 2, and by then, water is on the floor. 
- *The Solution:* Make your checks shorter (every second). But checking every second takes forever to run on a computer!

**Discrete Rate Simulation (DRS)** is different. Instead of checking every minute, the computer looks at the flow rate of the faucet and calculates *exactly* when the tub will be full. It then sets an alarm for that exact moment and lets the simulation "jump" forward in time.
- **The Result:** It runs incredibly fast and never misses a threshold. 

### For the Non-Miners: What are we simulating?
In this simulation, "mining" is just a continuous supply chain puzzle:
1. **Extraction (Faces):** Digging up "parcels" of rock from the earth.
2. **Transport (Fleets):** Trucks or conveyors that move the rock. Think of them as pipes.
3. **Stockpiles:** Giant piles of rock. These act like buffer tanks (or bathtubs) so that if a truck breaks down, the factory doesn't immediately stop.
4. **Processing (Mills/Plants):** The factory that grinds the rock to get the valuable metal out.

The goal of the simulation is usually to test different **Operating Modes** (strategies). For example, "If Stockpile A gets too full, tell the trucks to drive to Stockpile B instead."

---

## 2. Features: Pros and Cons

Why use Mining-DRS instead of commercial software like Arena, or standard Python loops?

### Pros:
- **Blazing Fast:** Because time "jumps" from event to event, it can simulate years of operations in a fraction of a second.
- **Fail-Fast Safety:** The framework is designed to stop and warn you with clear error messages if you break the laws of physics (like trying to drain an empty stockpile).
- **PyTorch-Inspired Architecture:** For programmers, the system feels exactly like building a neural network in PyTorch. Everything is a "Module" that automatically keeps track of its own data.
- **Built-in Telemetry:** Every single change in the system is automatically tracked. You can plot beautiful graphs of your stockpiles over time without writing custom tracking code.

### Cons:
- **No Visual UI (Yet):** Currently, you have to write Python code to build a model. We are actively planning an Arena-style drag-and-drop visual interface.
- **Mindset Shift:** Thinking in continuous "rates" (tons per hour) instead of discrete "events" (a truck arrived) takes some getting used to.

---

## 3. The PyTorch Parallels Dictionary

Mining-DRS was fundamentally designed to be the "PyTorch of DRS." If you are familiar with deep learning, the architecture of this simulation engine will feel like home. Here is how the concepts map 1:1:

| Concept | PyTorch Equivalent | Mining-DRS Equivalent | Why it matters |
|---------|-------------------|-----------------------|----------------|
| **Components** | `nn.Module` | `drs.Module` | Base class for your physical and logical components (e.g., a Plant, Fleet, or Controller). |
| **Hierarchies** | `nn.Sequential` | Nested `drs.Module`s | You compose simulations as trees. Modules automatically register any sub-modules assigned to them via `__setattr__`. No manual `register_module()` calls required. |
| **State/Data** | `nn.Parameter` / `Tensor` | `drs.Variable` / `drs.Level` | The underlying objects holding your state. Just like tensors, assigning `self.my_stock = drs.Level()` automatically registers it to the module. |
| **Logic Step** | `model.forward()` | `model.forward()` | The unified pass where you define all physical routing, state assignments, and mode transitions. |
| **Logging** | `TensorBoard` | `Telemetry` | Automatically tracks and logs the values of all variables at every time step for plotting. |
| **Optimization**| `torch.optim.Optimizer`| `OperatingMode` | Applies specific rules, targets, or constraints to the state (e.g., target milling rates, fleet routing splits). |

---

## 4. How the Engine Works (The Core Logic)

Every simulation in Mining-DRS follows a simple, repeating loop:

1. **Evaluate `forward()` Passes:** 
   - The engine calls `forward()` on all modules to get the instantaneous rates.
   - Controllers look at the system and set targets (e.g., "Send 100 tons/hour to Stockpile A").
   - Physical components update their internal flow rates based on those targets.
2. **Find the Next Event ($dt_{min}$):** The engine calculates exactly how long it will take for any physical variable (Level) to hit a limit (e.g., Stockpile A reaches maximum capacity).
3. **Jump Time:** The simulation clock jumps forward by that exact amount of time ($dt_{min}$). All stockpiles and variables are mathematically updated instantly.
4. **Trigger Transitions:** The system realizes a limit was hit and changes its Operating Mode (e.g., "Stockpile A is full! Switch trucks to Stockpile B!").
5. **Repeat.**

---

## 5. Building Your First Simulator

Here is a simple example of how to build and run a simulation. We compose physical **Modules** together, just like building blocks.

### Step 1: Define the Physical Modules
Every component is a `drs.Module`. It owns its physical state, usually represented as a `drs.Level` (something that can fill up or empty out).

```python
from drs import drs

class SimpleStockpile(drs.Module):
    def __init__(self, max_capacity):
        super().__init__()
        # A Level represents a quantity that changes continuously over time
        self.ore_level = drs.Level("Ore_Level")
        self.max_capacity = max_capacity

    def forward(self, inflow_rate, outflow_rate):
        # Physics: The rate of change is simply what comes in minus what goes out
        self.ore_level.rate = inflow_rate - outflow_rate
```

### Step 2: Define Operating Modes (The Strategy)
Modes tell the simulation what the current goal is. They are grouped under a Controller.

```python
from drs.modes import OperatingMode, RequireDecision

class NormalMode(OperatingMode):
    @property
    def name(self): return "NORMAL"
    
    def apply_dynamics(self, model):
        # Tell the model to flow at 100 tons per hour
        model.inflow_rate = 100.0 

    def check_end_conditions(self, model):
        # If the stockpile hits its capacity, we need to make a decision!
        if model.stockpile.ore_level.value >= model.stockpile.max_capacity:
            return RequireDecision() 
```

### Step 3: Handling Data Streams (For Realism)
In real operations, rock isn't generic; it arrives in batches with different grades of metal. The `drs.DataSource` and `drs.DataPoint` classes let you stream these parcels into your simulation.

```python
import random
from drs import drs

class TruckDataSource(drs.DataSource):
    def __next__(self) -> drs.DataPoint:
        # Generate a random truckload of rock
        mass = random.uniform(40.0, 60.0) 
        grade = random.uniform(0.5, 1.5)
        return drs.DataPoint(mass=mass, grade=grade)
```

### Step 4: Run the Simulation and Plot
We put it all together inside a `DRSEngine`, run it, and plot the results!

```python
from drs import DRSEngine
from drs.plot import build_dashboard, plot_time_series

# 1. Initialize the physical components
my_stockpile = SimpleStockpile(max_capacity=5000)

# 2. Give it to the simulation engine
engine = DRSEngine(my_stockpile)

# 3. Run for 365 simulated days
engine.run(max_time=365.0)

# 4. Plot the results! Telemetry happens automatically.
df = my_stockpile.telemetry.to_dataframe()
dashboard = build_dashboard(df, configs=[
    {"func": plot_time_series, "kwargs": {"y_columns": ["Ore_Level"]}}
])
dashboard.savefig("results.png")
```

---

## 6. Advanced Design Choices & Philosophy

If you are a developer looking under the hood, here is why we built Mining-DRS this way, along with the pros and cons of these decisions:

### Implicit Graph Emergence vs. Explicit Ports
Instead of explicitly wiring "Output Port A" to "Input Port B", connections emerge implicitly from behavior. `ExecutionContext` tracks which module is currently executing. When `Module A` sets a rate that affects `Module B`, an edge is recorded automatically.
- **Pros:** Maximum flexibility, drastically cleaner code, extremely "PyTorchy". No need to write boilerplate port definitions.
- **Cons:** Harder to determine the strict direction of flow, graph tracing is implicit and can be fragile if users circumvent the standard API.

### Fail-Fast Guardrails vs. Silent Bugs
We strictly enforce a "Rule of Ownership." If a Controller tries to magically modify a Stockpile it doesn't own (e.g. `self.stockpile.ore_level.value = 5`), the `ExecutionContext` detects the cross-module mutation and throws a `RuntimeError`. 
- **Pros:** Prevents users from silently destroying the physics of the simulation (e.g. creating matter out of nowhere). Forces communication through explicit rates and flows.
- **Cons:** Requires discipline. Users cannot take hacky shortcuts.

### AST Expression Tracing vs. Python Truthiness
Under the hood, operations like `level > 200` return Abstract Syntax Trees (ASTs) rather than booleans. Attempting to use this in an `if` statement raises a `TypeError`.
- **Pros:** Prevents the classic PyTorch/JAX trap of silent truthiness bugs. The engine can mathematically track conditions perfectly without the user writing complex math formulas. It also acts as the foundation for exporting the system to a JSON UI blueprint.
- **Cons:** Users must explicitly use `.value` (e.g., `if level.value > 200:`) for dynamic Python control flow, which can sometimes trip up beginners.

### Unification of Logic (`forward()`)
In previous designs, calculating rates and stepping state were separate. Now, all physical routing, state assignments, and mode transitions are unified under the `forward()` pass.
- **Pros:** Eliminates boilerplate and closely mirrors PyTorch's architecture. 
- **Cons:** Sometimes a module returns something, sometimes it doesn't. Can be mildly confusing to distinguish pure physical modules from controllers without looking at the code.

### Deterministic Monte Carlo
The system is built to support parallel execution for Monte Carlo simulations. 
- **Pros:** By passing localized Random Number Generators (RNGs) instead of using global random states, every simulation run is perfectly reproducible, enabling robust parallelization.
- **Cons:** Requires explicit tracking and passing of RNG seeds, preventing the use of simple `import random` calls globally.

---

## 7. What's Next? (Roadmap)

We have big plans for the future of Mining-DRS:
- **Visual Interface:** A web-based, Arena-style drag-and-drop UI where non-programmers can build physical flow networks visually, which automatically translates into Python code (and vice versa).
- **Compiler Optimization (`drs.compile`):** Compiling the AST math operations down to raw bytecode to make large-scale Monte Carlo simulations even faster.
- **Parallel Monte Carlo:** Out-of-the-box support for spinning up thousands of scenarios across multiple CPU cores.
- **Automated Mass Balance:** Automatically calculating the flow balancing between multiple child stockpiles in complex supply chains without manual arithmetic.

---

*Enjoy building your simulations! If you hit an error, remember: it's a feature, not a bug—the system is protecting the laws of physics.*

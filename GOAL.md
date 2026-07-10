# DRS Framework — Goals & Design Notes

## Vision

- General DRS, not just mining
- Components in `drs/` are not mining-specific
- Efficient — run FAST
- Python first: simple, fast, easy to read, feels like PyTorch
- Secondarily: drag-and-drop Visual approach like Arena, with Python ↔ Visual round-trip
- The PyTorch of DRS
- Bridge between programmers and non-programmers (mining engineers, etc.)
- Improve on Arena by supporting DRS natively — semantic modules (Plant, Fleet, Mine, Stockpile) instead of pointer entities + assigns
- Hierarchical modules: zoom into a Fleet to see individual trucks as sub-modules
- Fail fast! Errors should be caught early with clear messages
- Based on Navarra's work — operating modes are first-class

---

## Status Overview

### ✅ Completed (Phase 1 & 2)

| Area | What |
|------|------|
| **`__setattr__` auto-registration** | `Module.__setattr__` auto-registers Variables, Levels, Timers, sub-Modules (like PyTorch `nn.Module`). No manual `register_buffer`/`add_module` calls needed. |
| **Expression dual-mode system** | Operator overloading (`__add__`, `__sub__`, `__gt__`, etc.) builds Expression ASTs during tracing; evaluates numerically during simulation. See `drs/variables.py`. |
| **ExecutionContext** | Thread-local stack tracking which module is currently executing via `Module.__call__` push/pop. Enables implicit read/write dependency tracing. |
| **Fail-fast guards** | Illegal cross-module mutations raise `RuntimeError`. `Expression.__bool__` raises `TypeError` (prevents silent truthiness bugs). |
| **Level.rate tuple setter** | `level.rate = (value, lower, upper)` convenience shorthand. |
| **Configurable DRSEngine** | `max_step_size`, `max_deadlock_steps` parameters. |
| **Engine exception safety** | `try/finally` around context push/pop ensures stack integrity even on exception. |
| **Dead code removed** | `Signal` removed from `drs/data.py`. `CoreDRSConfig` deleted. Old import-style `ModeA`/`ModeB`/etc. classes replaced. |
| **Operating modes** | Single `OperatingMode` class + `MODES` dict singleton registry (13 old classes → 1 class + 1 dict). `RequireDecision` sentinel/exception for controller-engine coordination. |
| **Controller timer map** | 120-line `_update_timers` if-elif chain → 8-line dict-driven lookup on `_TIMER_MAP`. Subclasses extend via `_TIMER_MAP = {**Base._TIMER_MAP, ...}`. |
| **Clean controller_decision** | Extracted `_choose_next_campaign_mode()` — CyanidationController overrides just this method for Stage 2 (C/D) logic instead of duplicating the entire decision method. |
| **RL environment** | `RL_MineController` uses `RequireDecision` exception to pause engine for RL agent input. Verified working. |
| **Blending example** | Verified end-to-end: all modes (A, B, contingencies, surging, shutdown) transition correctly through full campaign cycles. |
| **Tests updated** | Integration tests use `controller.forward()` directly and correct attribute paths. |

### 🔄 In Progress / Partially Done

| Item | Notes |
|------|-------|
| **Mass balance automation** (`supply_chain.py`) | Plant still manually sums child stockpile rates. Not yet automated. |
| **Generalized sensor system** | Currently mining-specific (`BaseSensorNetwork`, `ConcentratorSensorNetwork`). Not abstracted for general DRS. |

### ❌ Not Yet Started

| Item | Priority |
|------|----------|
| Visual drag-and-drop system (Arena-like) | **HIGH** |
| `drs.compile()` — AST optimization for Monte Carlo | MEDIUM |
| Parallel Monte Carlo execution | MEDIUM |
| Stochastic drive times / fleet delays | LOW |
| Generalized DataGenerator / DataLoader | LOW |
| Topological order warning | LOW |

---

## Resolved Design Decisions

These are goals from the original vision that are now implemented and considered stable.

### 1. `__setattr__` Auto-Registration (like PyTorch)

Assigning a `drs.Variable`, `drs.Level`, `drs.Timer`, or sub-`drs.Module` via `self.x = ...` in `__init__` automatically registers it as a child. The `_owner` reference is set so the framework knows which module owns each variable. No manual `register_buffer` or `add_module` calls required.

**Status:** ✅ Implemented in `drs/module.py`.

### 2. Implicit Graph Emergence

Graph nodes and connections emerge from behavior, not from explicit registration. Something has to happen (read, write, or compute) for a connection to be recorded. Explicit connections are avoided because they may not match logic, add boilerplate, and are more work.

**Status:** ✅ Implemented via `_record_incoming_edge()` in `Module` + ExecutionContext tracking in `Variable`.

### 3. Expression Dual-Mode (AST + Evaluation)

Operator overloading returns `Expression` AST nodes during tracing, which are evaluated recursively during simulation. The `Variable.value` property checks if it holds an `Expression` and evaluates it on access.

**Status:** ✅ Implemented in `drs/variables.py`.

### 4. Fail-Fast Philosophy

- Illegal cross-module mutations → `RuntimeError` with descriptive message
- `Expression.__bool__` → `TypeError` (no silent truthiness)
- Only `Level` has `.rate` — `Variable` raises `AttributeError` on `.rate = x`

**Status:** ✅ Implemented.

### 5. Operating Modes as First-Class Citizens

Based on Navarra's work, operating modes are central. Single `OperatingMode` class with `name`, `id`, `is_valid_start()`, `get_target_rates()`, `check_end_conditions()`. Singleton registry via `MODES` dict.

**Status:** ✅ Implemented in `examples/mining/components/modes.py`.

### 6. DRSEngine Resets Rates Each Tick

Before calling `model.forward()`, the engine zeros out all rate ASTs. This is the standard DRS approach and is currently working.

**Status:** ✅ Implemented.

### 7. Unification of Logic into `forward()` (update_rates deprecated)

Because the engine uses an implicit graph and traces ASTs dynamically, the traditional separation of "calculating rates" and "stepping state" is obsolete. All physical routing, state assignments, and mode transitions are unified under the `forward()` pass. `update_rates()` is removed to eliminate boilerplate and enforce PyTorch-like encapsulation.

**Status:** ✅ Implemented. `drs.Module` has `forward()`, not `update_rates()`. The engine calls `forward()` once per tick to compute all rates, then integrates levels.

### 8. Controller Uses `RequireDecision` Pattern

`OperatingMode.check_end_conditions()` returns `RequireDecision()` (sentinel) when the engine needs external input. The RL controller raises `RequireDecision` as an actual exception to pause the engine. The engine handles both cases safely.

**Status:** ✅ Implemented.

---

## Core Implementation Architecture

### 1. The "Rule of Ownership" (In-Place Mutation Guardrail)

To prevent users from silently destroying the visual graph by doing `controller.mill.rate = 5`, the Variable setters must check the `ExecutionContext`. If the currently executing module is not the owner of the variable, it raises a `RuntimeError` forcing them to pass a `drs.Signal` instead.

```python
class Variable:
    @value.setter
    def value(self, val):
        current_actor = ExecutionContext.get_current()
        if current_actor is not None and current_actor is not self._owner:
            raise RuntimeError(
                f"'{current_actor.__class__.__name__}' attempted to mutate "
                f"'{self.name}' owned by '{self._owner.__class__.__name__}'. "
                f"Modules must communicate by passing Signals/Flows."
            )
        self._value = val
```

**Status:** ✅ Implemented in `drs/variables.py`.

### 2. The Engine Loop ($dt_{min}$ Integration)

The engine follows these exact steps each tick:

1. **Evaluate `forward()` passes** to get instantaneous rates from all modules.
2. **Iterate over all `drs.Level`s** to find the smallest time to a threshold ($dt_{min}$) — i.e., the earliest moment any level hits its upper or lower bound.
3. **Apply `level.update(dt_min)` globally** — advance all levels by the same $dt_{min}$.
4. **Trigger state/mode transitions** — check if any threshold was crossed and activate new operating modes for the next tick.

**Status:** ✅ Implemented in `drs/engine.py`. `DRSEngine.run()` follows this exact loop.

### 3. Reproducible Stochasticity (Monte Carlo Goal)

Global `random` or `np.random` is banned. For parallel Monte Carlo execution to work correctly, the top-level module must accept an RNG seed and pass localized `RandomState` instances down to stochastic components (like `MineFace`).

```python
class BaseBlendingModel(drs.Module):
    def __init__(self, config, seed=None):
        self.rng = np.random.RandomState(seed)
        self.face1 = MineFace(rng=self.rng)
        self.face2 = MineFace(rng=self.rng)
```

This ensures each parallel replica produces an identical sequence given the same seed, enabling deterministic Monte Carlo.

**Status:** 🔄 Partially implemented. `StochasticFaciesGenerator` accepts a seed, but the pattern is not enforced across all stochastic components (fleet drive times, etc.).

### 4. The UI Blueprint Extraction (Symbolic "Dry Run")

Because the graph is built dynamically at Run-Time (which allows native Python `if`/`else` control flow), the framework needs a **Symbolic Trace** feature. This passes fake "Symbolic" signals through the system before the engine starts, exploring all `if`/`else` branches to emit a complete JSON Simulation IR. This JSON is what powers the Arena-like Drag-and-Drop UI.

Mechanics:
- A "tracing mode" in `ExecutionContext` where all `Variable` reads return symbolic `Expression` objects instead of raw values
- Operator overloads (`__gt__`, `__add__`, etc.) build ASTs AND log edges
- Branch exploration: when the trace hits `if stock_level > 200`, it forks — following both the True and False paths to capture the full decision tree
- Output: a JSON IR encoding all modules, variables, edges, and control-flow branches — the blueprint for the visual canvas

**Status:** ❌ Not implemented. The `Expression` AST system and `ExecutionContext` provide the foundation, but the symbolic trace pass and JSON IR emission do not exist yet.

---

## Unresolved Design Questions

### Group 1: Inter-Module Communication

This is the biggest unresolved design question. There are several competing approaches:

#### Approach A: Port-based (explicit InputPort/OutputPort)

```python
class Stockpile(drs.Module):
    def __init__(self):
        self.inflow = drs.InputPort()
        self.outflow = drs.OutputPort()
        self.level = drs.Level()
```

**Pros:** Explicit, type-safe, maps 1:1 to visual connections
**Cons:** Boilerplate, feels "un-PyTorchy", every class must define ports

#### Approach B: Functional `forward()` with Signal/Flow Passing

```python
def forward(self, inflow_signal, requested_outflow):
    actual_outflow = min(self.level.value / dt, requested_outflow.value)
    self.level.rate = inflow_signal.value - actual_outflow
    return drs.Signal(value=actual_outflow, source_module=self)
```

**Pros:** Clean data flow, edges are obvious from call graph, natural for visualization
**Cons:** Sometimes a module returns something, sometimes it doesn't (confusing). What does `Module.__call__` return for a controller? A sensor? What happens to modules that only set rates internally?

#### Approach C: Bind Method

```python
self.plant.inflow.bind(self.fleet.outflow)
```

**Pros:** Explicit, no heuristics needed
**Cons:** Extra manual step, dynamic binding may be complex

#### Approach D: Implicit via ExecutionContext + `__setattr__` (Current System)

The current approach: modules hold references to each other and set rates directly. ExecutionContext tracks who's doing what.

```python
class ModeController(drs.Module):
    def forward(self):
        if self.stockpile.level > 200:
            self.mill.capacity_target = 6000
```

**Pros:** Maximum flexibility, most PyTorch-like
**Cons:** Hard to determine direction of flow, hard to enforce discipline, graph tracing is implicit and fragile

#### Sub-questions:

- **Signal vs Flow vs dict vs RateVector?** Generic `drs.Signal` with `attributes` dict? A `drs.Flow` dataclass? Dictionary of rates? `VectorLevel`/`VectorRate` for tracking individual component rates explicitly?
- **Everything as a node vs two systems?** Should controllers and sensors be graph nodes (everything is a node) or kept separate from the physical flow network? A separate network for physical stuff creates two systems to track.
- **Is a Signal the same as a Flow?** A `drs.Flow` is "just an ephemeral dataclass that carries the rates between modules during a single execution tick." Is this different from our existing `Variable` class?

**Key tension:** The explicit connector approach (Ports) is robust and good for visualization but adds boilerplate. The implicit approach (current) is flexible and PyTorch-like but makes visualization and correctness harder.

---

### Group 2: Tracing Reads of External Variables

When a module reads a variable owned by another module (e.g., controller reading stockpile level), how do we capture this as a graph edge?

#### Approach A: Pass variables as `forward()` arguments

```python
class ModeController(drs.Module):
    def forward(self, stock_level: drs.Variable):
        if stock_level > 200:
            return drs.Flow(command="MODE_A")
```

**Problem:** Forces users to pass every dependency as an argument — breaks OO encapsulation, not PyTorch-like.

#### Approach B: ExecutionContext read hooks (recommended)

The same `ExecutionContext` used for mutation guards can also log reads:

```python
class Variable:
    @property
    def value(self):
        self._record_read_dependency()  # <-- logs: owner --> current_actor
        return self._value
```

When `ModeController` evaluates `self.stockpile.level > 200`, the `__gt__` overload calls `_record_read_dependency()`. The framework silently logs "Stockpile → ModeController".

**Current status:** Partially implemented. `_record_incoming_edge` exists but the automatic read-hook in every `__gt__`/`value` getter is not yet complete for all operator overloads.

#### Approach C: `__set__` descriptor override

Intercepting Python's descriptor protocol to track assignments. Less explored.

---

### Group 3: Dynamic Control Flow (if/else with Variables)

The classic PyTorch/JAX trap: Python `if stock_level > 200` evaluates `>` which returns an `Expression` AST, and Python can't cast `Expression` to `bool`.

#### Current solution: `Expression.__bool__` raises `TypeError` (fail-fast)

Users must explicitly use `.value`: `if stock_level.value > 200`.

#### Competing approaches:

**A: PyTorch Way (Dynamic)**
Let `stock_level.value > 200` evaluate to a raw boolean. Re-trace the graph on every tick. Easier to code, slightly slower.

**B: JAX Way (Symbolic)**
Implement `drs.Where` or `drs.Switch` that builds conditions into the AST:
```python
self.mill_rate = drs.Where(stock_level > 200, 6000, 3900)
```
Faster, easier to visualize, but less intuitive for Python users.

**Lean:** Currently partial toward JAX approach but not sold.

---

### Group 4: Variable / Data Type System

#### Current types:
- `Variable` — general-purpose state
- `Level` — has `.rate`, integrates over `dt` (continuous flow)
- `Timer` — like Level, monotonically increasing
- `Expression` — AST node returned by operator overloads
- `State` — Categorical (not yet implemented)

#### Competing formulations:

**A: Multiple types (current)** — Variable, Level, Timer, Auxiliary, State
- Problem: Variable and Auxiliary are semantically too similar

**B: Minimal — just Variable and Level**
- Rates, constants, states all as Variable
- Only Level has `.rate` (fail-fast: `Variable.rate = x` raises `AttributeError`)
- Add `drs.Flow` for inter-module messages
- `Variable.value` evaluates Expression AST on access

**C: Two families — Continuous (Variable, Level, Timer) and Categorical (State)**
- Streamline existing implementation
- Remove `@property` interceptions where not needed
- Only Level has rates
- `Variable.value` auto-evaluates AST

---

### Group 5: Container Modules (Top-Level vs Visual Nodes)

The top-level model (e.g., `ConcentratorModel`) is a container, not a graph node. Other modules may also be containers.

**Problem:** How to distinguish containers from leaf nodes without an `is_container` flag? A flag adds boilerplate and is "un-PyTorchy." Easy to forget.

**Options:**
- Heuristic: modules containing sub-modules but no variables are containers
- Explicit: `is_container` flag (undesirable)
- Implicit: container detection via graph topology

---

## Performance & Future Work

### AST Compilation (`drs.compile()`)

Currently each `Expression.evaluate()` recursively walks the AST tree. For Monte Carlo (10,000+ runs), this is slow.

**Planned fix:** `engine.compile()` walks ASTs and uses Python's `compile()` or NumPy ops to produce optimized bytecode.

### Parallel Monte Carlo Execution

Goal: efficient parallel execution for Monte Carlo simulation results.

### Topological Order Warning

Add a warning when execution order doesn't match topological order, which can lead to 1-tick delays.

---

## Visual System (Not Started)

This is the largest remaining goal. Requirements from the vision:

- Drag-and-drop interface like Arena
- Python ↔ Visual round-trip (build in visual, edit in Python, and back)
- Visual system represents how the system actually runs (visual debugging)
- See flow and different operating modes visually
- Visual levels of abstraction — zoom into a module to see its internals
- Create custom components visually (Stockpile class, ConcentratorPlant, etc.)
- Visual system should be usable by non-programmers (mining engineers)
- A non-programmer builds a simple version visually, hands off to programmer for RL/LP

---

## Mining-Specific / Example Improvements

### Mass Balance Automation
Currently the plant manually sums child stockpile rates. This should be automated — when you set the concentrator rate, the fleet rate should implicitly balance.

### Fleet Management Scenario (Navarra 2019, Fig 6)
Goals:
1. Increase time in Mode A, reduce contingencies
2. Keep ~60/40 Ore1/Ore2 stockpile ratio
3. Add stochastic drive times
4. Add conveyors, gradual parcel additions
5. Multiple muck sites with different distances and grades
6. Limited truck capacity

### Stochasticity
- Ore generation (done)
- Fleet drive times (not done)
- Other stochastic elements (not done)

### Sensors & Uncertainty
- True vs belief values
- Are sensors edges or nodes?
- Is this mining-specific or general DRS?

### Other Mining Goals
- Allow passing more than just mass (e.g., cyanide usage)
- Dynamic mass balance enforcement
- Bridge between Navarra and Ruossos
- SGS (Sublevel Stoping) support
- Custom metrics (NPV, etc.)
- Delays for operating changes, travel time
- OpenAI Gym for Mining Optimization — reusable benchmark scenarios

---

## Open Questions (Miscellaneous)

- Should we prevent incorrect edges/connections (e.g., mass flowing into grade)? Or does this add too much boilerplate?
- Can we generalize OreParcel as a `DataGenerator` / `DataLoader` (like PyTorch Dataset)?
- Do I want `rate = inflow - outflow` syntax? Is this nicer than the current approach?
- Is there a way to track data paths without modules returning data? (Currently rates update levels, which feels natural for DRS)
- What about JAX-like functional approach vs SystemC vs Modelica vs Arena? Pros and cons for DRS?

---

## Original "Current Plan" (from earlier goals)

1. ~~Data Types like Resource, Observation, Control~~ — replaced by Expression AST system
2. ~~Register children with `__setattr__`~~ — ✅ Done
3. ~~Graph nodes emerge from behaviour~~ — ✅ Done (ExecutionContext tracing)
4. ~~A `drs.Module` that represents a visual block~~ — ❌ Not started

---

*Last updated: June 2026*

# DRS Architecture & Design

## Overall Goal
Make an "Arena but for Mining". Arena is good for queueing networks, but Geo Statistics are not integrated, and DRS is not native. We want a version of Arena where these things are native.

## Core Design Principles
- **Time Separation:** Time is a level, but it's separated to make it more easy to understand. Keep this.
- **Output Statistics:** Some things are just output statistics, make that clear in the code. Track them by default for any mode (users shouldn't need to define them).
- **Mathematical Correctness:** Use `infinity` in code, not `9999` or `99999`. Maintain semantic shapes (don't represent Nx1 vectors as just N). Avoid hacky arithmetic masks (`min`, `* (X<Y)`); use clean loops, clamps, or size-aware operations.
- **Fail Fast & Explicit:** Provide clear error checking when sequences don't exist. Implement a good way to say "do nothing".
- **Default Parameters:** Allow defaults so everything doesn't need to be set explicitly. Distinguish clearly between universally required DRS parameters and model-specific ones. Use a structured initialization (e.g., config objects, init functions) instead of string functions.
- **Visualization:** Plots should be beautiful.

## Structure Components
1. **Levels**
2. **Timers**
3. **Discretely Dynamic Numerical Variables**
4. **Categorical Variables**

Different models may have a different number of each. Levels and timers could technically be combined into continuous numerical variables, but it's nice to separate them because timers have a particular interpretation. Provide easy ways to reset specific timers or all timers.

## "nn.Module" Idea for DRS
- Pitch idea: Similar to `nn.Module` in PyTorch.
- We have different layers which might be things like the `Rate`, the `Level`, or the `Timer`.
- Allow for math and interaction between these things.
- Allow for something like a `Sequential` for sequences.
- Needs discussion: What are the pros/cons? What are all possible design solutions?

## Eliminating Arena Quirks
- It would be nice to run with the same spreadsheet but without Arena.
- **Modes:** Modes show up a lot. Separate them out so there is a nice, easy, error-checked way of defining them.
- **Sequences:** Need a good, easy way of defining sequences. Avoid Arena-specific sequence logic (e.g., specific assignment for std = 0 on normal dist). These are for a sequence of operations. In the case of the example we had it was mostly around switching modes.
- **Auto-wiring:** In Arena, setting `drs_Level(1)` is manual. In our code, this should ideally be automatic. Things like incrementing/decrementing timers should also be automatic.
- **Hold Blocks & Loops:** Arena uses "hold" blocks to wait for conditions and decides/loops for while loops. Our code shouldn't use these hacks. We can check conditions proactively.
- **Islands:** We use 5 islands in our DRS. Are some of these the same for all DRS models? Can they be provided by default or at least required every time to "fail fast"?
- **Thresholds:** Scanning thresholds should be easy to implement.
- **Initialization:** Make initialization more efficient. Don't use a single for-loop and reassign the last element. Instead, make 4 different for-loops for assignments.
- **Variable Types:** Decide whether to use simple Strings (like the Arena method) or make new custom types.
- **Matrices/Vectors:** Are they the best way, or just an artifact of Arena?

## Open Architecture Questions & Thoughts
- **State Transitions & Gym Integration:** Is it better to define transitions as start/end conditions? This would more easily allow for action masking when using DRS in a Gym Env, training an RL agent to select the operational mode on a day-by-day or campaign basis. 
- **Handling End Conditions:** Is it good practice to have the environment handle the end conditions, or should that be done in the Controller DRS? Intuition says the controller, but from a DRS perspective, there are no fixed timesteps.
- **Modes as Classes:** Are Modes important enough to be a new class rather than an Enum? If we are adding start conditions, end conditions, and actual operation logic for each mode, it makes sense to encapsulate them. (e.g., "Mode A starts if these conditions are met, ends when this condition is met, and does this"). It should still be easy to preempt a mode (e.g., switch from Mode A to Mode A Contingency).
- **PyTorch-like Design:** Is a Mode Class PyTorch-like in nature? Is it okay if it isn't, even though the overall goal is a PyTorch-like system for DRS?
- **Code-defined Equations:** Make it possible to define a set of equations and variable values directly from code.
- **Hidden Overrides:** PREVENT HIDDEN OVERRIDES. What if one module sets a rate and the next one resets it? Ensure the architecture prevents these hidden conflicts.

--- 

Overall goal. Make Arena but for Mining. Arena is good for queueing networks and stuff. but Geo Statistics are not integrated, and DRS is not native. Make a version of arena where these things are Native. 

time is a level, but its seperated to make it more easy to understand. keep this 

notice some things are just output statistics, make that clear in the code somehow 

99999 or 9999 are often used in place of infinity. use infinity in the code instead. 

vectors are often used, and sometimes a N x 1 is represented as an N. Don’t do this, use the semantic meaning and shapes from the document. 

Allow for defaults? so everything doesnt need to be set explicity? 

Also notice, there are some parameters that are required (ie for all DRS models) and some that exist only for this specific model. have a clear destinction in how these are treated, messages that are sent etc. 

so some should be required and have a nice way of defining those (maybe when making the DRS object if we are using objects, or the init fn for a function approach, or init config for a string approach etc) and some should be done when defining the specific model. im not sure if that makes sense. and need a way to add a parameter. 

Try not to use string fns like milene did, but maybe make a way of converting Arena to python or whatever i am using. 

There is also this idea of Modes that seem to show up a lot. maybe we can seperate that out so there is a nice easy way of defining modes. and nice fail fast error checking for that. 

Good error checking on when we do run sequence X or sequence Y if it doesnt exist, and a good way to say do nothing

Also a good easy way of defining sequences

Easy ways to reset timers or specific timers or all timers.

General structure includes: 

1. levels 
2. timers 
3. discretely dynamic numerical variables 
4. categorical variables 

could have a different number of each of these depending on the model. 

levels and timers could be combined to continuos numerical variables. but its nice to seperate because timers have a particular interpretation. 

would be nice to run with the same spreadsheet but without arena. 

some of the sequence logic used in the example spreadsheet and txt may be specific to arena quirks, that could ideally be removed (ie no need for a specific assignment for std = 0 on normal dist or something like that) 

in arena setting the drs_Level(1) and stuff is done manually. ideally this would be done automatically in the code version. there are likely other things that can be done automatically or more easily. like incrementing vs decrementing timer or things like that. 

the way arena does it is we define our functionality/expressions, and then the rates and all these other things in the labels just call those to update correctly and stuff. is there a cleaner way? good question. 

from what i can see there are a lot of similar eval statements and other things, those can probably be abstracted out to be more semantically understandable? doest his have the downside of making the system less abstract or like less capable of doing ANY kind of DRS with ANY ammount of detail. 

again for many of the output statistics it feels like they should be tracked by default for any mode. and not need to be defined by the user if that makes sense. 

plots should be beautiful. 

we use 5 islands in our DRS. are some of these islands the same for all DRS models? Can these be done by default to decrease load on the user? Can these at least be required every time to “fail fast”?

there are things like scanning thresholds that should be able to be done easily.

also we seem to use decide blocks and loops to do like a while loop. this should not be how its done in our code. it should be a clean implementation with best coding practices that dont use the hacky implementation details of the arena method.

make initialization more efficient than the arena version in the example system. dont use 1 for loop and reassign last, instead make 4 different for loops for the assign part (and other similar parts). 

There is a lot of MN(etc etc) and * by stuff that will be 1 or 0 to do the expressions. this should probably not be the case in our code. it should run safely and not need the min, and if it does it should be a clean clamp. but i think it should be possible without clamping. and i think it should be possible without the addition or multiplcation or whatever we are doing (in update Rate Configuration number for example) 

another hacky thing is the “hold” block which is just there so that arena checks the terminating condintion, but in code would not need to “hold” we could just check the condition proactively unlike arena. 

i notice a lot of masking a lot of if length less than max this, or *X<Y and stuff. i feel these are unecessary in a code version, as looping can just be done knowing the size of the item instead of masking. 

We have our different variable types. should we do what he did and use a simple String to determine that or make new types to represent these? 

the arena method uses matrices and vectors, are these the best way? or is this an artifact of arena?

--- 


Allow for the different development phases. This is somewhat mining specific and may be better in the mining-drs goals but i think in general many projects have phases of development and varying levels of abstraction. 

Strategic Planning & Feasibility (High Abstraction)
Pre-Feasibility & Detailed Engineering Design (Intermediate Abstraction)
Execution & Operational Control (Low Abstraction)
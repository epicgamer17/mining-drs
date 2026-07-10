# Global Master TODO

This file consolidates all action items, open questions, and tasks across the mining-drs project.

## 1. Immediate Tasks (Before Monday)
* [ ] Get Visualizations Working
* [x] Get Fleet Simulation Example Working
* [ ] Get SGS (Sequential Gaussian Simulation) working
* [ ] RENAME PAPERS LIKE I DID WITH RL!

## 2. Core DRS Architecture & Engine
* [ ] **Hidden Overrides:** PREVENT HIDDEN OVERRIDES AND STUFF. LIKE WHAT IF ONE MODULE SETS A RATE AND THE NEXT ONE RESETS IT. OR OTHER THINGS LIKE THAT. DO THIS BUY ADDING A GUARDRAIL IF POSSIBLE
* [ ] **Code Generation:** Make it possible to make a set of equations and variable values from code.
* [ ] **Modes as Classes:** Do you think that Modes are important enough that instead of using an Enum we could make a new class? (Start conditions, end conditions, actual operation logic for each mode). It should still be easy to preempt modes (e.g. switch from Mode A to Mode A Contingency). Is a Mode Class PyTorch like in nature? Is it okay that it isn't even though my goal was a PyTorch like system for DRS?
* [ ] **Transitions:** I'm curious if it could be better to define my transitions as start conditions and end conditions, or somehow have that ability to more easily allow for action masking when using my DRS in a Gym Env.
* [ ] **End Conditions in Env vs Controller:** Is it good practice to have the environment handle the end conditions? Or should that be done in my Controller DRS? Intuition says it should be the controller, but from a DRS perspective there are not fixed timesteps.
* [ ] **Optional Feature - Standard Components:** Investigate whether we want a standard library of components (`Source`, `Sink`, `Tank`). (Not sure if we want this yet, keeping DRS as a build-it-yourself framework may be better).
* [ ] **Optional Feature - Parameter Type:** Consider introducing a `Parameter` type alongside `Variable` to add a strict read-only lock to prevent accidental mutation of configuration constants during `forward()` calls.
* [ ] **Optional Feature - Real-World Time:** Consider supporting a `datetime` mapping (e.g. `start_datetime` in `EngineConfig`) for native integration with real-world timestamps. (Not sure if we need this yet).

## 3. Standard Simulation & GeoStatistics
* [ ] **Kriging:** Need to make Kriging stuff (or find online or in a library), SGS and GSGS stuff (or find in a library), and SIS stuff.
* [ ] **Custom GeoStatistics System:** Make own GeoStatistics System (Further future).
* [ ] **Dynamic Mass Balance:** Get dynamic mass balance working with multiple muck sites to a single concentrator/crusher.
* [ ] **Different Feed Rates:** Consider different feed rates from muck sites to our stockpiles.
* [ ] **Maximize Throughput:** maximize the daily tonnage time average (maximize daily throughput).
* [ ] **Stochasticity:** add stochastic drive times.
* [ ] **Conveyors:** add a conveyor and make parcel additions to stockpile gradual.
* [ ] **Visualization Updates:** The route policy in our visualization doesn't seem true to our current system. The controller tooltip is hard to read. Sensor logic inputs vs outputs could be more clear. The concept of "grade" could be improved.
* [ ] **Financials:** Look at citation 11 to add NPV and IRR to existing models and plots (APE1455294.pdf).

## 4. Base Scenarios & Fleet Management Examples
### Scenario Goals
* [ ] Goal 1: increase time in mode A and reduce contingencies
* [ ] Goal 2: keep ratio of stockpile roughly 60% Ore 1 and 40% Ore 2
* [ ] Goal 3: Remake Fig 6 with Fleet Management added to Mode A and Mode B and have a higher throughput (especially on lower target stockpile size). [!2019_NavarraRojas paper]

### Baseline Arena Recreation (Fleet Management)
* [ ] **Step 1:** Set up base variables, holding costs, shortages, etc., and replication limit (200 days).
* [ ] **Step 2:** Setup expressions (`POIS(2.5)`, `TRIA(10, 15, 20)`).
* [ ] **Step 3:** Implement Client Demand Process.
* [ ] **Step 4:** Implement Resupply Process.
* [ ] **Step 5:** Setup Inventory Level Plotting (Stairs mode).
* [ ] **Step 6:** Refactor to Continuous (levels, rates, demand rate parameters).
* [ ] **Step 7:** Final Logic & Visual Basic setup.

## 5. RL Controller Roadmap & Open Questions
* [ ] **Literature & Benchmarks:** 
  * Look at citations 7 and 10 which use ML on DRS of APE1455294.pdf.
  * Reimplement Navarra's experimental setups as baselines.
  * Look into EMPC (Economic Model Predictive Control) literature for mineral processing.
* [ ] **Network Architecture:**
  * Need a GNN.
  * Possibly Hierarchical RL or Multi-Agent RL.
  * Semantic Sorting of Actions or Continuous Action Spaces with similarity matching.
  * GNN + Attention/Pointer Networks.
* [ ] **Reward Function Design:**
  * Formulate multi-objective reward functions (NPV, throughput, profit, uptime, target ore level).
  * Determine how to penalize chattering realistically without arbitrary reward shaping.
  * Handle non-episodic continual learning (Average Reward).
* [ ] **State Representation & Observation:**
  * Fully Observable Baseline: Inventories, Active mode, Rates, Blend composition.
  * POMDP Formulation: Hide future ore composition/latent degradation. Expose delayed measurements, noisy assays, conveyor readings. Include previous actions ($a_{t-1}, a_{t-2}$).
* [ ] **Dynamics & Constraints:**
  * Add price/cost, mechanical constraints, time delays, non-stationarity, and physical limits (stockpile limits).
  * Use Constrained MDP (CMDP) framework for safety limits.
* [ ] **Experiments:**
  * Traditional RL (DQN/PPO) vs Stream RL vs Non-stationary RL (IDBD/CBP).
  * POMDP versions with LSTMs.
  * Discrete vs. Continuous Control (high vs. low level).


---
TO BE FORMATTED
OPTIONAL/UNSURE: Constraint / Invariant System

Natural extension of existing guardrails (mutation protection, deadlock detection):

model.register_invariant(
    "mass_conservation",
    check=lambda m: abs(m.total_mass_in.value - m.total_mass_out.value) < 1e-6,
    on_violation="warn"  # or "raise" in strict mode
)
model.register_invariant(
    "stockpile_bounds",
    check=lambda m: 0 <= m.stockpile.mass.value <= m.stockpile.capacity.value,
    on_violation="raise"
)
Check all invariants every step. Produces an invariant report in SimulationResult. Catches model bugs early.

OPTIONAL/UNSURE: Scenario Runner / Experiment Manager

The Monte Carlo pattern exists ad-hoc in examples. Formalize it into a high-level API:

from drs import Experiment

exp = Experiment(
    model_class=NavarraConcentrator,
    base_config={"plant.max_rate": 6000, "simulation_days": 365},
)
exp.add_scenario("baseline", {})
exp.add_scenario("aggressive_fleet", {"fleet.n_trucks": 12, "fleet.match_factor": 1.2})
exp.add_scenario("conservative", {"controller.bias": "ore2_safety"})

results = exp.run_all(n_replications=30, parallel=True, seed=42)
results.summary()       # Table of means and stds per scenario
results.compare()       # Auto-generated comparison dashboard
results.report_md()     # Markdown report for docs
This would make parameter sweeps, sensitivity analysis, and Monte Carlo studies trivial instead of requiring custom scripts.

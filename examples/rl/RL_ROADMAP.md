# RL Controller TODOs & Open Questions

> [!TODO]
> For Fleet Management: 
> Need a GNN 
> Possibly Hierarchical Reinforcement Learning 
> Possibly Multi-Agent Reinforcement Learning (one agent per truck/loader working together)
> Solutions for Non-Stationary Action Semantics or Entity Matching
>   Possible solutions: Semantic Sorting of Actions (ie closest first, or highest grade)
>   Continuous Action Space: Agent outputs features, the action selected is muck site with those most similar features (ie grade, distance, level) or something like that. 
>   Score based method: $$Score = (w_{grade} \times Grade) - (w_{distance} \times Distance) - (w_{queue} \times QueueSize)$$ highest score is chosen. When a truck needs a destination, your DRS engine looks at all currently active muck sites and calculates a score for each and picks the highest.
>   GNN + Attention/Pointer Networks? (Possibly SOTA). GNN as features, pointer network/attention on Nodes. And that determines what we select

## Overview
1. What is a good reward function?
2. What is a realistic observation/state?
3. What are the actions? 
4. What are the constraints? 
5. What are the dynamics? What needs to be added to our model? 

It seems like there are 3 places to optimize (from my understanding). Initial Values or Control Parameters (not related to mode). This might be like the target amount of each ore or other (modes and dynamics all handled by DRS). Optimize the thresholds for each mode (im not 100% this is valid). Optimize the rates etc of each mode. Each of these may require different objective functions and metrics. Additionally, theres another area of work in learning the geo statistics (or a world model) which then can be passed to a DRS for optimization. How do you decide which you are going to do in a given situation? 

> [!NOTE]
> AI Says: 
> - Learn the Geostats if inputs to the DRS are uncertain or the cause of failure
> - Learn High Level Targets (not goal related params) if the failure is due to poor logistics, and not poor metallurgy or plant performance (and also for longer time horizons)
> - Learn thresholds when human error is the primary cause of instability. As in the equipment works fine but the operators are "messing up" switching modes too late or too early. (benefit of interpretability, a threshold value is very easy to understand)
>     - How often should thresholds be updated?
> - Learn mode parameters (control in specific modes) when human operators follow the plan correctly but the plant still fails to meet production targets or the failure is the plans themselves. (also for minute to minute adjustments).
> **Standard Development Process**
> 1. Fix the Inputs first (World Model).
> 2. High Level Planning (Non Mode).
> 3. Stabilize the Macro (Thresholds).
> 4. Squeeze the Micro (Rates).


> [!TODO]
> - Look into Dynamic Mass Balance and where it fits in here.
> - It may belong in another document but Navarra has also done work using ML to predict geostatistics for a DES to "optimize". 
>    - The framework utilizes a Multilayer Perceptron (MLP) neural network trained on multi-element geochemical assays to predict gold recovery rates and reagent consumption. These predictions are fed in real-time into a DES model that optimizes truck logistics, stockpile blending, and feed campaigns
> - Navarra has also used Random Forests to find the "optimal" trigger points for changing a plants operational modes, which could easily be done with RL.
> - Look at what I have so far here and in my modular RL library. Figure out what can readily be done with minimal extra work, then build from there. A possible direction, take relevant Navarra papers and reimplement their experimental setup, use my RL framework and compare results/metrics.
>   - This is likely the most realistic approach to make meaningful progress in a reasonable timeframe. Instead of designing a new experimental setup from scratch, leverage existing research and frameworks to validate the approach and compare performance. This allows for a clear baseline and incremental improvements. It also means I use the mining ideas from Navarra (ie metrics, scenario, target) and my RL knowledge and codebase. 

> [!NOTE]
> Im curious if for the continuous case we even need the concept of modes when using RL. If we have modes but every campaign the agent chooses the rates etc for the mode (selected by the system?) what is the point of a mode? Also does the timeframe in this case shrink compared to the high level discrete case?
> For the discrete case it makes more sense, we are simply defining high level actions or almost even options for the agent. 
> Additionally there is the question of the discrete case if the time interval should be each campaign or each day (the length of the shortest mode). And the question of how to deal with variable time intervals (options? just not learned ones?).
> I don't believe generally speaking its a good idea to change your reward function necessarily (at least not for RL). I think the better formulation there is probably options, and the goal of the option is to "minimize explosive use", kind of like options of feature attainment or something like that. Changing the reward function would make the problem non-stationary and there would be a lag in adjusting. 
> The predefined options seems better because that allows us to keep these operational modes and the different goals of the modes, and later these modes can be learned. I'm not sure that predefined options exist for the continuous case unless we are learning thresholds somehow? Each campaign the agent could output n continuous actions, one for each threshold, and those would be the thresholds used for the campaign. But the actual rates are predefined. 
---

## 1. What is a good reward function?
Options: 
- Net Present Value or instantaneous Profit
- Throughput
- Money/Profit
- Uptime
- Target Ore Level

May need to make reward a configurable multi-objective function to allow for ablations on the reward function (unfortunately, this means reward shaping).

Generally speaking mining is a continual learning problem. It is also a non-episodic problem. And the environment also tends to be non-stationary due to geo statistics. 

If making an episodic approximation (ie some form of sprints), time to completion (ie time to collect X ore) is generally not of interest (I think). Instead the goal should be to maximize the rate of production, using throughput this will already inherintely minimize time, and more easily transfer to the non episodic case. 

Our reward may need to differ depending on if we are using the low level continuous control or the high level discrete control (as in supervisory control). We may need to penalize chattering (how?). 


> [!IMPORTANT]
> **Open Questions:**
> - What does our reward function need to include? 
> - Is there a reward function that already exists that we can use? Maybe from a mining engineering textbook?
> - Is there a reward function that will not require reward shaping, while giving good AND robust/safe behaviour?
> - How can we penalize chattering while preventing arbitrary penalties and weights (i.e., reward shaping)? How can we make it so our reward reflects the real cost of chattering instead of an arbitrary penalty.

> [!TODO]
> - Look into Economic Model Predictive Control (EMPC) literature for mineral processing. EMPC objective functions translate almost perfectly into RL reward functions. They usually focus on maximizing economic profit while maintaining product specifications (like P80 size in comminution or concentrate grade in flotation). 
> - Use the Constrained MDP (CMDP) framework. The reward function is purely your economic objective (profit). Safety and physical constraints (e.g., tank overflow, equipment over-torque) are handled via separate cost functions that the agent is constrained to keep below a certain threshold.
> - Look into continual non episodic learning (Average Reward)
> - Look into non-stationary RL (IDBD and CBP)
> - Look into stream RL (Stream AC, Stream DQN, etc)


## 2. Important Metrics

RL Metrics:
1. Immediate/Initial Performance
2. Time to Good Performance
3. Can they learn offline (e.g., while they get good, can they train off the normal operations of the mine)?
4. Performance with the stationary and non-stationary environment.
5. Fully observable vs partially observable performance.

Mining Metrics:
1. Grade-Recovery Curve
2. Throughput (tph)
3. Specific Energy (kWh/t)
4. Net Present Value (NPV)

Action Distributions: Is the agent constantly slamming the feed rate between 0% and 100% (Bang-Bang control)? Engineers hate this. They want smooth transitions.

Constraint Adherence Rate: How many times did the agent's proposed action threaten a safety limit or stockpile overflow?

Grade-Recovery Tradeoff: Plotting the agent's performance on a grade-recovery curve compared to historical human performance.

State-Value Heatmaps: Visualizing the agent's value function based on key variables (e.g., Mill Power vs. Throughput) to ensure it aligns with metallurgical physics.

> [!IMPORTANT]
> **Open Questions:**
> What metrics do mining engineers use to optimize/evaluate, and what metrics do they use to understand the policy?
> Are we missing any important metrics?
> How do we compute these metrics? 

## 3. What are the actions? 

- **"Goal" for the plant**
  - Think something like focus on ore 1 or 2, etc. 
  - Discrete (High Level)
- **Operating Mode of the plant**
  - Mode A, Mode A Contingency, etc.
  - Need to likely constrain legal modes? As in, a contingency mode can't always be chosen, right?
  - Discrete (More granular) 
- **Control Variables**
  - Any control variables used by our simulator.
  - Continuous 

Discrete actions are likely not the way to go. They require the human engineer to design all the modes/goals removing the benefit of the RL agent. Instead if we do continuous control we can maintain a high level supervisory version by picking the right variables to control. It allows the agent to (hopefully) pick the optimal feed value for the current ore (instead of having a human guess).

> [!NOTE]
> Navarra has previously (not with RL) done optimization chossing fixed values, and used the time in Mode A (the ideal mode) as the metric. This is different than designing the modes themselves or picking the parameters of the modes. Instead you are picking the other parameters to maximize time in good modes.


> [!IMPORTANT]
> **Open Question:** 
> Should control variables be chosen and roughly fixed, or constantly changing? As in, are they chosen at the start of a day or week and then used for a fixed period of time, or is the agent continually adjusting these values?
> Is it viable for a mines control variables or operational modes etc to be constantly updated? Is this something a mine can keep up with? Does it depend on the parameters chosen?

## 4. What is a realistic observation/state?

> [!IMPORTANT]
> **Open Questions:**
> - What in a real mine (not a simulator) would be available to the agent? 
> - How much should we compute for the agent? How much should it learn?
> - Is it partially observable and if so, what parts and how?

*Note: Will likely need to make this partially observable and include LSTMs.*

### Fully Observable Baseline
- Inventories
- Active mode
- Rates
- Blend composition

> [!IMPORTANT]
> **Open Question:** 
> What other things could we include in the fully observable observation? What's realistic?
> What does our current DRS "provide" that our RL agent could use? 

### POMDP Formulation

**Hide (Latent State):**
- Future ore composition
- Latent degradation
- Exact process rates
- Future parcel arrivals

**Expose (Observations):**
- Delayed measurements
- Noisy assays
- Conveyor readings
- Stockpile estimates
- Previous actions ($a_{t-1}, a_{t-2}$) 
  - useful for the agent to understand the momentum of the plant.
- Shift schedules
  - Operator shift changes often introduce behavioral non-stationarity.

## 5. What are the dynamics? What are the constraints?  What needs to be added to our model? 

> [!WARNING]
> Don't get lost in the weeds here. Remember Navarra mentioned the goal is detail where its needed not everywhere.

We likely shouldnt use an event driven semi-MDP. If we want to use a semi-MDP we should use options. The event driven semi-MDP means that the environment decides when the agent can take an action. It can lead to problems as the agent may want to do one thing but our thresholds/events may lead to very short or inconsistent timesteps. The existing thresholds are also tuned/inherit to the existing handcrafted policy, so they don't necessarily reflect real operational constraints or the learned policy. It may be possible to do the event driven for continuous control, but for the higher level discrete control, it may not be the best approach. It's both easier and likely better to do fixed timesteps. 

Things we may need to add to our model
1. Price/cost (energy, resources, chemicals, value of ore)
2. Mechanical constraints
3. Delays/time delays between actions and sensory data/rewards
4. Non-stationarity (ore composition, prices, equipment degredation, etc)
5. Constraints (stockpile limits, etc.)

> [!IMPORTANT]
> **Open Questions:**
> 1. What parts should be non-stationary? How should this be modeled? 
> 2. How should constraints be enforced? 
>   - Negative reward?
>   - Illegal moves?
>   - Environment dynamics?
> 3. Will we need average reward for non-terminating episodes?

> [!TODO]
> **Options**
> - Use options to make the agent learn operational modes. Something almost exactly like the Option-Critic paper. This env would allow us to apply option critic to a "real world" problem, and extend it possibly to the non episodic (average reward setting) continual, non-stationary, constrained POMDP. 

## 6. Agent Possibilities

1. **Traditional RL**: Traditional RL with batches (DQN and PPO).
2. **Continual learning model**: Stream RL (Stream AC and Stream DQN).
3. **Non-stationary mine**: Add a non-stationary model using CBP and IDBD methods.
4. **POMDP versions**: POMDP versions of each of the above (add an LSTM).

## 7. Possible Experiments

- [ ] Create an example with an RL based controller using traditional RL. Fully observable, stationary. Episodic? Average reward? Pretrain only? Or train "online"
- [ ] Create an example with an RL based controller using stream RL.
- [ ] Add delays to the system. (implement LSTM based)
- [ ] Add non-stationarity to the system. (implement non-stationary stream RL with IDBD and CBP)
- [ ] Add constraints to the system (e.g., stockpile limits, you can't store infinite ore), no chattering.
- [ ] Discrete vs. continuous control (high vs. low level).
- [ ] Different reward functions.

## 8. Future Work

- **GVFs and World Models/Search**: How well can the model learn a simulator? How well can it transfer to the real world or another simulator? What can it learn well?
- **Options**: Can the agent learn options? Can it learn its own thresholds for the Semi-MDP instead of using a hardcoded Semi-MDP formulation? Are these better?
- **Search**: MuZero
- **GVFs**
- **Sim to Real**
- **Other**

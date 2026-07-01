# RL Opportunities in DRS

## RL vs OptQuest
- **OptQuest:** Requires many Monte Carlo trajectories to optimize. If the geostats change, new trajectories must be computed. It provides an approximate best value based on MC samples (which are also approximate).
- **Stream RL:** Can learn online, so it would not require rerunning OptQuest. It uses historical data. True online stream RL can work without generating "parallel" MC simulations (i.e., one single stream of experience).
- **Advantages of RL over OptQuest:**
  1. Constantly adapting (non-stationary).
  2. Less black-box.
  3. Single stream, no simulators or MC samples required in some contexts.
  4. Offline learning from current system data is possible.
  5. More room for improvement and can utilize more techniques.
  6. Possibly requires fewer inputs.
  7. Possibly a better algorithm.
- **Perspective:** Another simpler way of seeing it is maybe that RL can be a function approximation of the formula in the 2019 paper.

## Potential Thesis / Improvements
- RL may be able to account for things like fixed parcel rates and understand how to adapt to the current parcel. (Do parcels even last long enough for that to matter? Given enough granularity, it could possibly do better).
- Could RL handle stochasticity and risk aversion better?
- What human info can we remove as inputs from the system?
- RL may require less info on geo stats and the entire system while maintaining whole-system benefit via the reward function.

## Integration Thoughts
- Geological uncertainty is non-stationary, so TD stream learning might be applicable.
- Geo statistics is separate. Learning or getting the ore distribution is not the simulation problem.
- Base Case: Design a simulation, run it, test solutions manually, MAYBE optimize control variables.
- The process could be: learn on simulator, deploy on simulator.
- General note on control variables: parameters and control variables shouldn't really be frozen.
- We could take the flowchart part of Arena, add geo statistics on top, and then have the ability to use simple distributions for the first stage.
- Can we infer context and optimize without context, or learn the context?

## Relevant Literature & Citations
- **ML on DRS:** Look at citations 7 and 10 which use ML on DRS of APE1455294.pdf.
- **Financials:** Look at citation 11 to add NPV and IRR to existing models and plots (APE1455294.pdf).

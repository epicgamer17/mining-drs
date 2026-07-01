# Challenges and Best Practices for the Numerical Modelling of Tailings and Waste Rocks

> [!NOTE]
> These are notes on a presentation by Vincent Martin. A lot of what he is saying applies to other numerical models.

## Properties of Mine Waste

### Typical Properties of Tailings
* Clay to sand size
* Uniform low $C_u$, low shear strength (typically)
* Low hydraulic conductivity
* Disposed of in tailings storage facilities: slurry, thickened, paste or filtered

### Typical Properties of Waste Rock
* Clay to boulder size
* Well graded (High $C_u$)
* High shear strength (typically)
* High hydraulic conductivity
* Disposed of in waste rock piles: end dumping, push dumping, etc.

## Testing and Data Collection

### Physical Properties
**In the lab:**
* Index properties
* Shear strength
* Hydraulic conductivity
* Triaxial tests
* Water retention curve

**In the field:**
* Sand cone
* Field vane
* CPT, SCPT, etc.
* Pumping tests, slug tests, etc.

> [!TIP]
> Often simple tests (like the sand cone) which have been there for a long time are there for a reason. You can get very good results with these tests. Often simple stuff is just as useful and can give a lot of information.

### Geochemical Properties
**In the lab:**
* Mineralogy
* Static tests
* Kinetic tests
* Columns

*Note: Geochemical properties were not necessarily part of the talk, but are important to the whole picture and the data we need to collect.*

Once we have this information, it serves as the basis for building the numerical model. We use the model to gain more information on our design and management strategy. 

> [!WARNING]
> **TODO:** Clarify exactly what "gain more information on our design and management strategy" means in this context.

---

## Numerical Modelling in Mine Wastes Management

### Why Do We Model?
Basically, we want to either:
* Refine our understanding
* Extrapolate/predict results
* Perform sensitivity analysis
* Support our design decisions

A numerical model remains a tool. It is something we use to support our decision process—basically a giant calculator. You are the one still making the decisions. 
*(Self-Note: WHY IS THAT? I WANT TO CHALLENGE THAT IDEA)*

### Types of Models

* **Block Model:** Typical for mining engineers. They define and refine block models. 
  > [!WARNING]
  > **TODO:** Add this to our system.
* **Rock Stability Models:** Also common (Rock mechanics). 
  > [!TIP]
  > Could be nice to add to our system one day.
* **Waste Rock Pile Model:** Designing shape, capacity, etc.
* **Water Balance Model:** Uses software like GoldSim. Has some stochastic modelling and Monte Carlo simulation. Involves using flowcharts to transfer to a GoldSim model to help understand where our water is coming from and where it is going. Can be used for operations, environmental prediction, and closure design.
* **Stability Analysis:** Limit equilibrium, failure surface. GeoStudio SLOPE/W for stability analysis.
* **Geochemical Models:** Coupled or uncoupled. Reaction kinetics, chemical equilibrium. Predicting long-term pH based on geochemical properties.
* **Hydrogeological Models:** Regional hydrogeological models. Mudflow models.
* **Tailings Facility Modeling:** Want to see where your water table will place itself, and see the flow at the end. Can be a 1D lab model of a column, ranging from simple to complex.

### Unsaturated Hydrogeology
The talk is specifically about hydrogeological models in unsaturated conditions.
* How does it apply to geomodelling?
* Why do we want to do these?
* What is the information we can gain?

**What is unsaturated Hydrogeology?**
Everything that is above our groundwater table is considered unsaturated. Waste rock piles often end up saturated or partially saturated. It changes the behaviour of the material. 

Tailings are fine, waste rocks are coarse. This affects saturation. Tailings may remain saturated, but waste rock may be desaturated (typically).

Always come back to the basic theory. That is the fundamentals of numerical modelling. Typically, numerical modelling will always work better with equations than data points. 

*(A whole bunch of mining specifics, specific scenarios, saturation, unsaturation, not my interest)*

--- 

## Best Practices for Numerical Modelling
*(Note: These best practices were discussed for waste rock but apply to all numerical modelling)*

### Clarify Objectives
The most important step before starting:
* What are your objectives? 
* What is your scope? What is the endgame? 
* Is it useful? 
* What are we doing, and why are we doing it?

You need a clear vision of your modelling goals to move forward.

### Understand Your Software
* What are your governing equations? 
* What are the boundaries and limits? 
* When can they be applied? 
* How does it work? 

It's not a black box; you can get the information out of the model. You have to understand these things.

> [!TIP]
> **Self-Note:** I think there is a lot I can do here for DRS and the DRS system. Providing these governing equations automatically, or displaying these boundaries/limits nicely. Making it not a black box, so you can get the info out of the system. 

### Always Start Simple
* **Conceptual model:** Use pen and paper, a whiteboard, etc. Before building a 3D numerical model with 350,000 elements, do it with pencil and paper. Determine what is coming in and what is coming out. 
* Better a simple model that you understand than a complex model you have no idea what is going on.
* **Limit the number of elements:** Use the minimum number of elements necessary. 
  * For example, in SEEP (his rule of thumb): $> 5000$ in occasional cases. $> 10000$ is usually too much ("REALLY??").
* **Limit the number of materials:** Do you need 5 different types of sand? Which materials actually control the behaviour? 
* **Do I need a 2D or 3D model?** 
  * Better have a 1D model that you understand than a 2D or 3D you don't understand. 1D is possible (he did it for his thesis). Have a good reason to go up in dimensionality.
* **Define your boundary conditions:** Do not overconstrain the model. 
* **Convergence parameters:** Don't go overboard with your convergence parameters, but understand them. If you make them too precise, the model won't converge and you won't be able to explain why. 

The simplest model you can use to get your results will always be your best model. Keep your models simple. Build one that converges, get the results, then make it more complex if needed. 

### Confirm Your Convergence
This is the very first step after a model is run. 
* *Example:* What is the water balance error? How far off is this? 

Before you check your objective, confirm your convergence. 

> [!NOTE]
> **Self-Note:** Does this relate to mass balance? Is this relevant to DRS systems? Can we add some notion of error in some way to inform developers on their decision process?

### Confirm Your Results
You must have a way to confirm your results.
* What are the expected results? 
* Check your water balance... again.
* In unsaturated conditions: watch out for hysteresis. 
* How do you account for parameter variability? 

**Compare your model to:** 
* Empirical equations
* Lab or field data (observations, wells, monitoring probes, etc.)
* Field observations
* etc.

### Documentation and Justification
* Justify your simplifications, educated guesses, and hypotheses.
* Take notes.
* Use references (it is always easier to use the work of others to justify your choices).
* Detail your hypotheses in your report, paper, thesis, etc. Someone else should be able to do the work as well. Include:
  * Software used and version
  * Number of elements, mesh size, etc.
  * Material properties
  * Geometry
  * Boundary conditions
  * etc.

### Modeling Strategies
* Use realistic boundaries when doing a sensitivity analysis.
  > [!TIP]
  > **Self-Note:** This is something I'd really like to add to our system. Some kind of import and export.
* With limited information, use predictive models.
  > [!TIP]
  > **Self-Note:** There is a lot I think we could do with predictive models.

---

## Closing Remarks
* A model is a tool to support YOUR decision process.
* Why are you building a model?
  * Know your scope of work.
  * Know your objectives.
* Understand your models and understand their limits.
* Start simple.
* Use empirical solutions, equations, etc. to validate your results.
* You will make mistakes: how can they be captured? How can they be mitigated?

---

## Q&A

**How do you justify the scopes and ranges of sensitivity analysis?**
Use multiple representative cross-sections to understand variability in material. Another way is if you build a big 3D model, maybe you can make a 1D model of the cover to get a reasonable estimate of the infiltration models of the cover, and then apply that to a 2D cross-section or 3D model. This is called combining models.

In other words, the results of the simpler models are used either as inputs or to justify the sensitivity analysis of the more complex models.


**As regulations require more and more complex models, how should we balance the suggestion you made of using simple models and the simplest model?**
Be as simple as you need to be, and not more so. Some people want to jump to the most complex solution. Even when the most complex models are ultimately needed, you will likely need to start with the simple model anyway and build up to the complex one. Often people start with the complex model, find that it doesn't work, and backtrack to the simple one until they get it working. It is much better to start the other way around.
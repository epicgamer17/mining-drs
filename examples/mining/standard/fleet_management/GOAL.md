# Fleet Management DRS Goals

## Overview
Optimize fleet size and fleet allocation with DRS.
- Create a realistic "scenario" (doesn't have to be a real mine initially).
- Figure out what modes look like in a fleet management system.
- Determine if modes are shared across the mine.
- Connect with current mill stuff and stockpile stuff.

## Fleet Logistics & Operations
- **SME Handbook:** Consult for fleet management cycle time or equipment design.
- **Infrastructure Requirements:** Need to see the roads and levels for distances and cycle time, and for placement of stockpiles, loading sites, etc.
- **Constraints:**
  - Trucks cannot pass each other on a road except in specific spots (which add a delay/wait times; drive uncertainty is important).
  - Sometimes there is a limit on trucks per level because of ventilation (number of CFM used).
  - One shovel can do multiple "green dots" (ore pickup spots), but can be modeled as a large muck. 
  - Trucks are usually assigned to a level, not a specific green dot.
  - The number of trucks and shovels per level, and fleet size, are constrained.
  - Muck sites and crusher sites are predetermined.

## Optimization Opportunities
- **Primary Goal:** Minimize queuing time. Trucks should not wait; shovels waiting is acceptable.
- **Load Cycle:**
  - Shovels/scoops/loaders pick up ore and put it in the truck.
  - Worker efficiency impacts scoops (less efficient workers = less ore per scoop = more scoops needed = more time = less throughput). Worker efficiency also impacts driving to the loading site (they have to drive backwards).
  - A cycle is from the point the truck enters the mine to the point it gets filled up and goes back.
  - Return trips can take longer depending on the ore being carried.
- **Resource Allocation:**
  - You can have many trucks waiting at one loading site, but you don't need 100 trucks for 1 site. Find the minimum amount of trucks waiting to optimize cycle time.
  - Load sites have varying distances to the crusher (this is fixed but important, as further ones take longer).
  - Shovels can move material ~200 to 250 meters to prep stuff for the trucks.
  - **Prioritization:** Only some levels are highly productive (due to ore grade and ore type). Prioritize better grade with better trucks.
  - Loading the truck is A LOT faster than driving to the crusher.

## Key Objectives
- Maximize throughput.
- Minimize fleet size.
- Minimize cycle time.
- Any other considerations: Probably assume that part is not the bottleneck or is "good enough".

## Design Thoughts
- We probably don't need a full map. We can just use distances to the crusher or drive times, and then ore grade and type at each site.
- Rough idea: Put shovels at muck sites that are closest to the crusher, and trucks to maximize throughput from there (probably closer sites need fewer trucks).

## GeoStatistics & Advanced Methods
- **Kriging:** Need to make Kriging stuff (or find online or in a library), SGS and GSGS stuff (or find in a library), and SIS stuff.
- **Custom GeoStatistics System:** Long-term goal to make our own GeoStatistics System.


--- 

rename face_capacity_rate to face_real_extraction_rate 
 
real extraction rate = intercept + beta_lhd * LHD_allocation + beta_truck * truck_allocation + beta_availability * availability - beta_distance * haul_distance - beta_delay * delay_factor
 
delay is for traffic since we have only 2 faces 
 
distance is for distance from stopes to loading area
 
merge distance into delay, distance is just a type of delay like traffic and all the other kinds of delay we have in our delay factor
 
remove intercept as it is always 0.0 
 
productivity of trucks depends on number of scoops and vice versa, that is not modelled 
 
need the match factor
 
it makes distance make much more sense 
 
solve for the match factor under constraints of fleet size to maximize throughput 
 
important to get cycle times of trucks or like efficiency of trucks and scoops, what is there throughput
 
there will be many ways to extract 6000 tonnes 
 

cycle time, efficiency , then match factor, then development



but if we can get the efficiency we are looking for the most efficient way of extracting 6000 so we can maximize mine development
 
traffic delays should be a function of number of trucks and give a delay time that increases cycle time and leads to lower efficiency
 
 

add faces
 
make faces have distance from face to surface 
 
Ore 1 face is further than the balanced face 
 
for now development always fixed portion of fleet developing
 
whenever generating a parcel we should check that enough development has happened 
 
with 2 faces we like it because the problem is not about efficiency but getting the right blend 
 
with match factor we can get efficiency and then we can get mine development 
 
because we can say the extra trucks we get from being more efficient allow for more mine development 
 


(GET EQUATIONS) add match factor and cycle time and distances to faces. possibly need some rough idea of traffic delays too for "efficiency". in other words, more trucks to one face leads to higher cycle times. so match factor changes. will need to add more detail to fleet, scoops and trucks for match factor at each face. 
then add mine development using the extra trucks from efficiency
then add more production faces 
then possibly add equipment maintenance (at the end because everything will run and work without this, but maintenance makes it more realistic) 
solve with more faces for most optimal/efficient solution getting desired throughputs, and then blends and efficiency. there is a balance of each and a pareto bound. may use RL. should agent decide number of scoops AND trucks per face or just number of scoops and the match factor is used for trucks. what for the case of 50 trucks and 5 scoops each scoop a match factor of 11. 


--- 


optimize fleet size and fleet allocation with DRS

make a good “scenario” (probably not real mine)

figure out what modes look like in a fleet management system

are modes shared across the mine 

connect with current mill stuff and stockpile stuff? 

we talked my DRS simulator and what it can do 

we talked about jiangs current work, and the system his company uses now for fleet optimization and management. he showed me the levels and the dashboard.

Look at SME handbook for fleet management cycle time or equipment design (ask chat the right name) 

need to see the roads and levels for distances and cycle time, and for placement of stockpiles and loading sites and stuff

milene thinks im deciding how many trucks and what trucks. The ore is already mined. and its just about getting it to the stockpile

ore is on the green dots. less trucks than levels and that green dots (ore pick up spots). 

trucks cant pass each other on a road. 

sometimes there is a limit on trucks per level because of ventilation (number of CFM used). 

Where is the optimization opportunity here? 

minimize queuing time. dont want to have trucks waiting. shovels waiting is okay. 

shovels or scoops or loaders pick it up and put it in the truck 

shovers or scoops or loaders have worker efficiency (as in workers can make scoops less efficient, they can make each scoop have less ore, so more scoops needed so more time, so less throughput). also worker efficiency in driving to loading site as they have to drive backwards 

a cycle is from the point the truck enters the mine to the point the truck gets filled up and goes back.  

going back can take longer depending on the ore they are carrying. 

can have many trucks waiting at one loading site, but dont need 100 trucks for 1 site so you need a good amount (minimum ammount of trucks waiting for cycle time).

there is a distance from the load site to the crusher. this is not something we can control. but its important because trucks at the far ones are less efficient (they take longer). 

milene says shovels can wait trucks should not. shovels can move stuff like 200 to 250 meters prep stuff for the trucks. 

only some levels are highly productive, because of ore grade and ore type. better grade and better trucks, priotize those levels/sites. 

truck brings ore to the crusher which goes to what ive already done (eventually, through intermediate steps) 

loading the truck is A LOT faster than driving to the crusher

one shovel can do multiple green dots. truck is usually assigned to a level not a green dot. 

deciding mainly shovel location (muck site) and trucks per shovel

optionally one shovel multiple mucks (but can be modled as a large muck) 

constraint on number of trucks and shovels per level and fleet size. also constraint on roads 1 (no passing except specific spots which add a delay and possibly wait times drive uncertainty is important) also muck sites and crusher site decided for you

rough idea put shovels at muck sites that are closest to the crush and trucks to maximize throughput from there (probably close sites less trucks) 

maximize throughput

minimize fleet size 

minimize cycle time

any other considerations probably assume that part is not the bottleneck or good enough

design thought, probably dont need a map and we can just have distances to crusher or drive times and then ore grade and type at each site

--- 

Old but important for notes from prof like not to model individual trucks or scoops: 

Fleet Policy Model Update Brief 20-June 

 

1. Research Direction 

Research question: Can a state-dependent fleet policy sustain Mode A performance by protecting Ore 2 availability and managing development catch-up across normal, contingency, and mine-side surging states? 

For now, Ore 2 warning logic/stockpile levels monitoring and development catch-up are not implemented yet. 

2. Current Program Structure 

The active simulation path is still the many-faces model. It uses two continuous mine faces, continuous fleet logistics, two stockpiles, and the multi-face controller. 

Active model: examples/mining/standard/many_faces_simulation.py 

Main class: ActiveFleetConcentratorModel 

Controller: MultiFaceConcentratorController 

Mine faces: ContinuousMineFace with geological parcels 

Fleet: ContinuousFleetLogistics, not discrete truck dispatching 

Disabled branch: The underground material-handling sandbox remains in the repository but is disabled for now. It represented LHD/truck/drill and parcel-haulage logic, which is too detailed for the current research direction. 

 

3. What Changed 

3.1 Shift-Based Fleet Reallocation 

A 12-hour shift timer was added to the controller. Every shift, or whenever the operating mode changes, the controller refreshes the current face allocation and capacity factors. 

fleet_shift_duration = 0.5 days 

fleet_shift_timer tracks shift progress 

fleet_shift_count records how many shift reallocations occurred 

3.2 Required, Capacity, and Actual Rates 

The previous model effectively treated the policy target rate as the achieved excavation rate. The controller now stores three separate values for each face: 

Rate 

Meaning 

Current Use 

face_required_rate 

What the fleet policy requests 

Policy output 

face_capacity_rate 

What aggregate fleet/resources can support 

Capacity constraint 

face_actual_rate 

min(required, capacity) 

Input to ContinuousMineFace 

3.3 Regression-Style Capacity Function 

Face capacity is now represented as an algebraic multiple-regression-style equation. This follows the professor's guidance: keep computation fast and avoid individual equipment entities for now. 

Capacity equation: capacity = intercept + beta_lhd * LHD_allocation + beta_truck * truck_allocation + beta_availability * availability - beta_distance * haul_distance - beta_delay * delay_factor 

 

The moderate comparison case currently uses LHD allocation = 0.50, truck allocation = 0.50, face availability = 0.93 / 0.91, and delay factor = 0.025 / 0.04. 

3.4 Physical Bounds on Mine-Side Surging 

Mine-side surging previously could request unrealistic extraction rates when the desired ore fraction was near zero. A physical cap and a minimum effective fraction were added. 


4. Latest Comparison Results 

Scenario 

Days 

Mean Actual 

Capacity Lost 

Max Gap 

Utilization 

Min Ore2 

Policy 1 baseline 

1,139 

5,808.6 

0.0 

0.0 

100.00% 

18,000.0 

Policy 1 + fleet capacity limit 

1,139 

5,791.3 

59,135.5 

1,268.3 

98.61% 

18,000.0 

Interpretation: the moderate capacity-constrained case does not collapse Ore 2, but it creates measurable lost production and changes the mode/surging trajectory.  

5. Figures Generated 

Figure 1. Capacity policy comparison across the full 1139-day horizon. 

 

Figure 2. Comprehensive-style diagnostic view for the capacity-limited scenario. 

6. What This Means for Monday 

The model now has a mechanism to compare ideal policy allocation against realized production under fleet constraints. 

The current capacity case is moderate: it creates measurable lost production but does not immediately deplete Ore 2. 

The 12-hour shift layer is in place, but allocation is still fixed by operating mode; it is not yet a true state-dependent policy. 

Current policy changes are mostly triggered by mode changes, not by smooth within-mode state feedback. 

7. Recommended Next Steps 

Refactor fleet allocation into explicit policy families: Mode A general policy with normal, contingency, and mine-side surging sub-policies. 

Keep current mode-based allocation as Policy 1 baseline. 

Add a state-dependent Mode A sub-policy that can adjust allocation every 12-hour shift. 

Later add Ore 2 warning logic, development catch-up state, and shift-level capacity uncertainty. 

Continue using face-level aggregate equations until Navarra gives the go for individual-equipment simulation 

8. Key File Map 

File 

Role 

components/config.py 

Shift duration, physical surging caps, and capacity coefficients. 

components/controllers.py 

12h reallocation, required/capacity/actual rates, capacity equation. 

components/modes.py 

Operating-mode target rates and capped mine-side surging logic. 

components/models.py 

Telemetry metrics and actual-rate input to ContinuousMineFace. 

standard/many_faces_simulation.py 

Comparison runner, CSV summaries, and generated plots. 

 
The route policy in our visualization doesnt seem true to our current system at the moment. At the moment the trucks do or at least should know what the parcel is and use that to always route ore 1 to stockpile 1 and ore 2 to stockpile 2. 
Then the way i think about it the controller looks at the stockpiles and decides which stockpiles to pull from and how much. 
Also sensor logic the inputs vs outputs could maybe be more clear? like i feel like the true ones are all inputs and the outputs are all the belief ones. And maybe theres a way to show that mapping. 
The concept of grade could also be improved as that word kind of means something else. 
also the controller tooltip is so hard to read 

1. Get dynamic mass balance working with multiple muck sites to a single concentrator/crusher
2. Consider different feed rates from muck sites to our stockpiles (which go to our concentrators)
3. maximize the daily tonnage time average (maximize daily throughput)
4. add stochastic drive times
5. add a conveyor and make parcel additions to stockpile gradual

Start with unlimited quantities of ore at each of the faces
the faces have different distances and different grades of ore (one has more ore 1 and one has more ore 2)

parcels always the same size 
parcel grade stays same for each muck site
different tonnage per muck site
trucks/parcels have limited capacity (smaller than the tonnes of the muck site)
lower level faces are always slower also to send ore to the same stockpile. trucks have to drive all the way to the surface or there is a skip. assume no elevator or skip for now.

MAYBE START WITH SCENARIO 2 
goal 1: increase time in mode A and reduce contingencies 
goal 2: keep ratio of stockpile roughly 60% Ore 1 and 40% Ore 2

Goal Remake Fig 6 with Fleet Management added to Mode A and Mode B and have a higher throughput (especially on lower target stockpile size). [!2019_NavarraRojas paper]

BASE SCENARIO. CONCENTRATOR WORKS FOR EXPECTED GEO STATS.
TIMELINE 1: 

No matter what Shutdown 1 day long every 34 Days.
Ore 2 is Valuable Ore
Parcel is random 60% Ore 1 40% Ore 2

We don't want any Mode A Contingency
Geostats will cause uncertainty leading to stockouts if no fleet management (by random chance). For example, if you suddenly get many high Ore 1 parcels you will run out of Ore 2 and enter Mode A Contingency.  

With fleet management, when Ore 2 is low we utilize the fleet on more High Ore 2 faces leading to more Ore 2 in the parcels and so Ore 2 stockpile does not decrease despite unfavorable geo statistics. 

Mode A (Normal Operating Mode):
    Concentrator 6000 t/d
    Input to Concentrator 60% Ore 1 40% Ore 2
    Moderate or high usage of fleet
    Balanced Faces for Ore 1 and Ore 2

Mode A Contingency (Process as much Ore 1 as possible):
    3900 t/d 
    Input to Concentrator 100% Ore 1
    Low usage of fleet on Ore 1 and high usage of fleet on Ore 2


---- 

TIMELINE 2:

MODE B LATER WHEN GEO STATS CHANGE. 

Mode B (Recuperating Operating Mode):
    Concentrator 5400 t/d
    Input to Concentrator 85% Ore 1 15% Ore 2
    Fleet Mine X% High Ore 1 Grade Y% Low Ore 1 Grade
    Moderate or high usage of fleet


Mode B Contingency (Process as much Ore 2 as possible):
    2500 t/d 
    Input to Concentrator 100% Ore 2
    Low useage of fleet
    Low usage of fleet on Ore 2 and high usage of fleet on Ore 1


                                    Ore 1 Stockpile
                                   /               \
             Parcels ===Conveyor===                 ===Conveyor=== Concentrator
                |    Ore 2         \               /
                |--- Face 2         Ore 2 Stockpile
    Ore 1       |    15/85
    Face 1 -----|    1 hour
    45/55
    2 hours


---
QUESTIONS: 
should we try to always have 60/40 Ratio in our stockpiles? 


HOW WOULD RUOUSSOS DO THIS? HE HAD HIS FLEET MANAGEMENT OPTIMIZATION PAPER.
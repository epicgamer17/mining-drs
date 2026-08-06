"""
LHT Switch Optimizer Notes:
Purpose: Optimize operator-truck assignments and mid-road switches so that all operators finish their 12-hour shift at their assigned mine site, while maximizing ore haul productivity.

Context: Two mine sites, 80km haul road apart. Diesel trucks carry ore continuously between sites. The mine operates 24/7 with two 12-hour shift cycles for both operators and dispatchers.

Goals in order:
1. Every operator ends their shift at their assigned site: Hard constraint.
2. Maximize total tonnage hauled, completed loads * truck capacity across all trucks over the shift (main objective).
3. Operators end their shifts in the order they started (ie 4:30 started doesn't end after a 5:30 starter).
4. Minimize unnecessary switches (fewer = less operational disruption).

Priority 1 and Priority 2 sometimes conflict. A single manual transmission truck with a non-qualified operator heading toward it, achieving P1 may require pulling a truck off the road early, significantly reducing P2. The system must make the dispatcher aware and not make these decisions silently.

Priority 3 is a nice one to have but may be violated for additional completed loads (it's basically a tie breaker).

A completed load = Load at Amaruq -> haul -> dump at Meadowbank.

LHXX Truck is 140t.
XXXX Truck is 110t.

Partial hauls count for zero.

Recomputation at any time (RL?) or Linear Programming.

System supports dynamic removal of trucks (for breakdowns) but NOT recovery of stranded operators (yet?).

3 mandatory tire stops at fixed locations along the route each adding 5 minutes of time (in both directions). Locations TBD.

The System shall apply an end-of-shift buffer of 30 minutes at Meadowbank and 20 minutes at Amaruq.

The system must respect operator-transmission qualification: only qualified operators can operate manual-transmission trucks. All operators may drive automatic trucks.

The system shall model switches as occurring at any point along the 80 km road (not restricted to fixed locations).
"""


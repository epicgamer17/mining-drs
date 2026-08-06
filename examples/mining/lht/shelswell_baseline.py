"""
Shelswell (2017) Baseline Truck Haulage Simulation & Paper Notes

===============================================================================
1. SUMMARY OF SHELSWELL (2017) PAPER SPECIFICATIONS
===============================================================================

Trucks vary in size, capacity and dump method to accommodate mine designs. Additional options include engine type, or modifications like sideboards.

Key strength to truck haulage is the ability to adapt accordingly in response to the current operations, future mining processes, and expansions or changes in mine plans as the life of the mine progresses.

Especially important in the absence of rail lines, conveyors and shafts.

Key limitations: wear and tear on equipment. Downtime is impacted by: fleet age, muck fragmentation, truck payloads, truck loading practices, roadway conditions, ventilation and cooling, haulage distance, productivity targets, maintenance regimes, and mine design.

It is important to maximize truck availability with respect to the underground conditions.

Operators are also important. Too few and available trucks can't be used, too many is unnecessary cost.

Need operator and equipment availability with randomness.

Upstream boundary used was the production of ore from stopes and the generation of waste from lateral development.
Assumed that mining activities were able to efficiently generate sufficient tonnes to meet the production and development targets for the simulation.

Downstream boundary was the dumping of material at the run of mine pad and waste stockpile site on surface. ROM and stockpile were considered unconstrained.

Exact mine design:
- 2100m long 5% grade decline access to a single spiral ramp system.
- Ramp was 1800m long and a grade between 8 and 13% with seven primary mine levels. Sublevels not included.
- Distance between each mine level on the ramp was 300m.
- Each level had 2 loadouts: one for ore and one for waste along the access drift of the ramp.
- Ore loadouts were 40m off the ramp while waste loadouts were 55m from the ramp.
- Single truck air doors were incorporated 20m down the level access drift coming off the ramp to control ventilation.
- ROM was located 300m from the portal while the waste stockpile dump was located 440m from the portal.
- Maintenance shop and fuel depot were located on the surface at 260m and 270m respectively.

Production & Schedule:
- The production and development schedule represented mine targets for 1 calendar year (365 days) with a 5.5:1 ratio of ore to waste.
- 11 non-production days were included for holidays, maintenance events, and random shutdowns.
- Ore and waste tonnes were scheduled daily based on a triangular distribution. Tonnes scheduled additively to muck bays.

Shift Schedules & Availabilities:
- 2 shifts per day, 10.5 hours each (12h elapsed).
- Workable availabilities: Haulage 54.17%, Underground LHD 58.33%, Surface Maintenance 79.17%.

Equipment Specs:
- Payload - ore: Truck 26.1 t, LHD 14.0 t
- Payload - waste: Truck 24.6 t, LHD 12.5 t
- Load spot duration: Truck 0.82 min, LHD 0.46 min
- Load duration (ore/waste): Truck 6.69 min, LHD 0.88 min
- Dump spot duration: Truck 0.57 min, LHD 0.55 min
- Dump duration: Truck 0.88 min, LHD 0.73 min
- Speed - surface: loaded 13.4 kph, empty 17.4 kph
- Speed - decline: loaded 11.2 kph, empty 15.1 kph
- Speed - ramp: loaded 9.2 kph, empty 12.9 kph
- Speed - level: loaded 6.6 kph, empty 7.6 kph (LHD: loaded 5.89 kph, empty 6.78 kph)
- Acquisition delay max: 3.0 min (avg 1.5 min)
- Remuck stockpile tram distance: 35 m
- Scheduled PM-associated availability: 99.8 - 100%
- Random failure-associated availability: 30.2 - 95%

Dispatch & Operation Rules:
- Payload type determined by 5.5:1 ore vs waste ratio.
- Dispatched to loadout with highest "unclaimed" tonnes remaining.
- 1 LHD active per level.
- Bounded effective fleet: eff_trucks = min(N_trucks * Availability, N_operators).
- Availability formulas:
  PM Availability = (FreqPM / (FreqPM + DurPM)) * (FreqFUEL / (FreqFUEL + DurFUEL))
  Random Failure Availability = (AVGMTBF / (AVGMTBF + AVGMTTR))
  Overall Mechanical Availability = PM Availability * Random Failure Availability

===============================================================================
2. DRS IMPLEMENTATION DIFFERENCES FROM PAPER
===============================================================================

1. Simulation Methodology (Discrete Rate vs. Discrete Event):
   - Paper: A true Discrete Event Simulation (DES) where individual trucks and loader units are simulated as independent entities executing discrete tasks.
   - Implementation: A Discrete Rate Simulation (DRS) where flow capacities are represented as continuous rates (tonnes/day). Capacity limits are solved analytically at each time step rather than simulated step-by-step.

2. Truck & Loader Cycles:
   - Paper: Truck loading times are simulated using a series of stochastic LHD loading bucket cycles, spot times, and uniform acquisition delays.
   - Implementation: The loading cycle is represented analytically based on average load spot, average acquisition delay, and LHD bucket cycle times.

3. Operator Pooling and Fleet Constraints:
   - Paper: Explicitly models operators and trucks as individual resources that must be acquired. Available trucks remain idle if operators are unavailable.
   - Implementation: Represented analytically by capping the effective fleet size: eff_trucks = min(N_trucks * Availability, N_operators).

4. Mechanical Availability:
   - Paper: Simulates scheduled planned maintenance (PM) at utilization intervals and random breakdowns using MTBF/MTTR probability distributions.
   - Implementation: Approximated analytically by scaling down the effective fleet size by the overall mechanical availability bracket.

5. Traffic & Congestion Delays:
   - Paper: Restricts decline/ramp roadway segments to single-direction travel with passing pull-outs. Loaded trucks are prioritized, and congestion creates queuing bottlenecks.
   - Implementation: Approximated using a linear traffic delay penalty applied to truck cycle times, scaled by the number of trucks allocated per level.
"""

from dataclasses import dataclass
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import drs


@dataclass
class ShelswellConfig:
    """Configuration parameters for the Shelswell (2017) mine design and haulage simulation.
    
    All spatial distances, speeds, payloads, cycle times, and operational constraints match 
    the physical mine layout and equipment specs defined in Shelswell (2017).
    """

    # --- Mine Design Parameters (Underground Layout & Surface Infrastructure) ---
    # 2100m long, 5% decline slope connecting surface portal down to top of spiral ramp
    decline_length: float = 2100.0  # Length of main portal access decline (meters)
    decline_grade: float = 0.05     # 5% grade (rise/run incline or decline slope)

    # 1800m long main spiral ramp interconnecting 7 primary mine levels
    ramp_length: float = 1800.0     # Total spiral ramp travel distance (meters)
    ramp_grades: tuple = (0.08, 0.13)  # 8% to 13% variable grade along spiral ramp turns

    # Vertical / in-ramp travel spacing between adjacent mine levels
    level_spacing: float = 300.0    # 300m ramp distance between each of the 7 primary levels

    # Level access drift distances from spiral ramp off-ramps to active muck bays
    ore_loadout_distance: float = 40.0   # 40m access drift from ramp to ore stope loadout bay
    waste_loadout_distance: float = 55.0 # 55m access drift from ramp to waste development face

    # Ventilation control infrastructure on level access drifts
    air_door_distance: float = 20.0      # Ventilation air doors located 20m off main ramp

    # Surface haulage destinations measured from portal entrance
    rom_pad_distance: float = 300.0          # 300m surface distance to Run-of-Mine ore crusher/pad
    waste_stockpile_distance: float = 440.0  # 440m surface distance to waste rock dump site
    maintenance_shop_distance: float = 260.0 # 260m surface distance to equipment maintenance shop
    fuel_depot_distance: float = 270.0       # 270m surface distance to diesel refueling station

    # --- Vehicle Speeds (km/h) across Mine Corridors ---
    # Surface flat haulage speeds (higher empty speed due to lighter vehicle mass)
    speed_surface_loaded: float = 13.4  # km/h loaded truck on surface roads
    speed_surface_empty: float = 17.4   # km/h empty truck on surface roads

    # Main access decline speeds (retarder braking downhill loaded / power-limited uphill empty)
    speed_decline_loaded: float = 11.2  # km/h loaded truck ascending/descending decline
    speed_decline_empty: float = 15.1   # km/h empty truck ascending/descending decline

    # Steep spiral ramp speeds (steep 8-13% grade restricts climbing speed)
    speed_ramp_loaded: float = 9.2      # km/h loaded truck climbing steep spiral ramp
    speed_ramp_empty: float = 12.9      # km/h empty truck descending steep spiral ramp

    # Horizontal level access drift speeds (speed restricted by narrow tunnel cross-section & corners)
    speed_level_loaded: float = 6.6     # km/h loaded truck in level access drifts
    speed_level_empty: float = 7.6      # km/h empty truck in level access drifts

    # Underground Load-Haul-Dump (LHD) loader tramming speeds
    speed_lhd_loaded: float = 5.89      # km/h loaded LHD bucket mucking speed
    speed_lhd_empty: float = 6.78       # km/h empty LHD bucket returning speed

    # --- Shift "Seat Time" Workable Availability Fractions ---
    # Out of a 12-hour total shift (10.5 hrs scheduled work), effective "operating seat time"
    # accounts for shift handovers, travel time to working face, safety pre-ops, breaks, and fueling.
    seat_time_fraction_truck: float = 0.5417        # 54.17% (~6.5 hrs productive haul time / 12h shift)
    seat_time_fraction_lhd: float = 0.5833          # 58.33% (~7.0 hrs productive loading time / 12h shift)
    seat_time_fraction_maintenance: float = 0.7917  # 79.17% surface maintenance shop shift utilization

    # --- Equipment Cycle Component Durations (Minutes) ---
    # LHD loader mucking sub-cycle times at the underground muck pile
    lhd_load_spot_minutes: float = 0.46          # LHD bucket positioning/spotting time at muck pile (min)
    lhd_load_minutes: float = 0.88               # LHD digging/scooping bucket into blasted rock (min)
    lhd_dump_minutes: float = 0.73               # LHD lifting and dumping bucket into truck bed (min)
    lhd_tram_distance: float = 35.0              # 35m tram distance between muck face and truck load bay
    lhd_acquisition_delay_minutes: float = 1.5   # Average wait for truck positioning / LHD maneuvering (min)

    # Haul truck positioning and dumping durations
    truck_load_spot_minutes: float = 0.82  # Truck backing into spot position under LHD bucket (min)
    truck_dump_spot_minutes: float = 0.57  # Truck backing onto surface ROM pad or waste dump grid (min)
    truck_dump_minutes: float = 0.88       # Raising hydraulic bed, discharging rock, & lowering bed (min)

    # --- Equipment Capacities and Payloads (Tonnes) ---
    # Payloads vary between ore and waste due to rock bulk density and swell factors
    truck_payload_ore: float = 26.1    # Ore payload capacity per haul truck trip (tonnes)
    truck_payload_waste: float = 24.6  # Waste rock payload capacity per haul truck trip (tonnes)
    lhd_payload_ore: float = 14.0      # LHD bucket payload for ore (~2 buckets fill 1 truck) (tonnes)
    lhd_payload_waste: float = 12.5    # LHD bucket payload for waste rock (tonnes)

    # --- Fleet Size, Operator Constraints, and Operational Parameters ---
    total_truck_count: float = 10.0  # Nominal haul truck fleet inventory count
    total_operators: float = 10.0    # Total available truck drivers/operators per shift
    # NOTE: Paper uses stochastic MTBF/MTTR breakdown distributions and scheduled PM intervals in DES.
    # DRS approximates mechanical downtime analytically by scaling down effective fleet size by overall availability.
    overall_mechanical_availability: float = 1.0  # Mechanical uptime ratio (PM & breakdown availability)
    # NOTE: Paper explicitly simulates single-lane decline/ramp passing pullouts with vehicle queuing;
    # DRS approximates ramp traffic congestion analytically using a linear time penalty per allocated truck.
    traffic_delay_per_truck_hours: float = 0.005  # Congestion penalty per truck (0.005 h = 18s) on single-lane ramp pullouts


def get_travel_times_hours(
    level: int, muck_type: str, config: ShelswellConfig
) -> tuple[float, float]:
    """Calculates empty and loaded travel times in hours for a given level.

    In real-world underground haulage:
    - Levels are 1-indexed (Level 1 top of ramp to Level 7 deepest mine level).
    - Travel speed varies by segment: surface flat, decline slope, spiral ramp, and horizontal level drift.
    - Muck type dictates surface dump destination (ROM pad for ore vs waste stockpile for waste rock) 
      and level access drift length (40m for ore vs 55m for waste development).

    NOTE: Shelswell (2017) DES models discrete entity movement along road networks with stochastic 
    acceleration/deceleration and queuing at single-lane passing bays. DRS uses static expected segment 
    speeds and deterministic travel times across corridors.

    Args:
        level: Mine level number (1 to 7).
        muck_type: 'ore' or 'waste'.
        config: ShelswellConfig instance containing layout lengths and vehicle speeds.

    Returns:
        tuple[float, float]: (t_empty_hours, t_loaded_hours)
    """
    # Convert speeds from km/h to meters/hour (km/h * 1000 m/km)
    v_surf_empty = config.speed_surface_empty * 1000.0    # Empty surface speed (m/h)
    v_surf_loaded = config.speed_surface_loaded * 1000.0  # Loaded surface speed (m/h)
    v_dec_empty = config.speed_decline_empty * 1000.0      # Empty decline speed (m/h)
    v_dec_loaded = config.speed_decline_loaded * 1000.0    # Loaded decline speed (m/h)
    v_ramp_empty = config.speed_ramp_empty * 1000.0        # Empty spiral ramp speed (m/h)
    v_ramp_loaded = config.speed_ramp_loaded * 1000.0      # Loaded spiral ramp speed (m/h)
    v_lvl_empty = config.speed_level_empty * 1000.0        # Empty level access drift speed (m/h)
    v_lvl_loaded = config.speed_level_loaded * 1000.0      # Loaded level access drift speed (m/h)

    # Calculate segment distances based on mine design geometry
    d_dec = config.decline_length                         # Main decline distance (2100m fixed)
    d_ramp = (level - 1) * config.level_spacing            # Spiral ramp travel distance to target level (0m for L1, 1800m for L7)

    # Determine level access drift distance and surface destination distance based on material type
    if muck_type == "ore":
        d_lvl = config.ore_loadout_distance               # 40m drift from ramp to ore loadout bay
        d_surf = config.rom_pad_distance                  # 300m surface haul from portal to ROM crusher pad
    else:
        d_lvl = config.waste_loadout_distance             # 55m drift from ramp to waste loadout bay
        d_surf = config.waste_stockpile_distance          # 440m surface haul from portal to waste dump

    # --- Empty Path Travel Time Calculation ---
    # Route: Surface Dump Site -> Surface Haul -> Portal -> Decline Descent -> Ramp Descent -> Level Loadout Bay
    t_empty = (
        (d_surf / v_surf_empty)   # Surface travel time empty (hours)
        + (d_dec / v_dec_empty)   # Decline descent travel time empty (hours)
        + (d_ramp / v_ramp_empty) # Ramp descent travel time empty to target level (hours)
        + (d_lvl / v_lvl_empty)   # Level drift travel time empty to loadout bay (hours)
    )

    # --- Loaded Path Travel Time Calculation ---
    # Route: Level Loadout Bay -> Level Drift -> Ramp Ascent -> Decline Ascent -> Portal -> Surface Haul -> Surface Dump
    t_loaded = (
        (d_lvl / v_lvl_loaded)     # Level drift travel time loaded (hours)
        + (d_ramp / v_ramp_loaded) # Ramp ascent travel time loaded from target level (hours)
        + (d_dec / v_dec_loaded)   # Decline ascent travel time loaded (hours)
        + (d_surf / v_surf_loaded) # Surface travel time loaded to dump site (hours)
    )

    return t_empty, t_loaded


def get_truck_loading_time_hours(muck_type: str, config: ShelswellConfig) -> float:
    """Calculates LHD mucking cycle time and total time required to load one truck in hours.

    In real-world underground operations:
    - An LHD loader scoops muck at the face, trams 35m to the truck load bay, dumps into the truck, 
      and trams back empty.
    - Since truck payload capacity (~26.1t ore / 24.6t waste) is roughly twice LHD bucket capacity 
      (~14.0t ore / 12.5t waste), exactly 2 LHD bucket loads are required to fill 1 truck.
    - Total loading time includes truck spotting, acquisition delay, and 2 full LHD bucket passes.

    NOTE: Shelswell (2017) DES draws individual LHD bucket fill times, spot durations, and positioning 
    delays from probability distributions. DRS evaluates loading duration using fixed expected mean values.

    Args:
        muck_type: 'ore' or 'waste'.
        config: ShelswellConfig instance.

    Returns:
        float: Total truck loading duration in hours.
    """
    # Convert LHD speeds to meters/hour (km/h * 1000)
    v_lhd_empty = config.speed_lhd_empty * 1000.0    # LHD empty tram speed (m/h)
    v_lhd_loaded = config.speed_lhd_loaded * 1000.0  # LHD loaded bucket tram speed (m/h)

    # LHD tramming durations for 35m distance between muck face and truck loading spot (converted to minutes)
    t_lhd_tram_empty_min = (config.lhd_tram_distance / v_lhd_empty) * 60.0    # Tram empty to face (min)
    t_lhd_tram_loaded_min = (config.lhd_tram_distance / v_lhd_loaded) * 60.0  # Tram loaded to truck (min)

    # Single LHD bucket sub-cycle duration: spot + load face + tram loaded + dump in truck + tram empty
    t_lhd_bucket_cycle_min = (
        config.lhd_load_spot_minutes       # 0.46 min spot bucket at muck pile
        + config.lhd_load_minutes          # 0.88 min dig & fill bucket
        + t_lhd_tram_loaded_min            # Tram 35m loaded to truck bay
        + config.lhd_dump_minutes          # 0.73 min lift & dump bucket into truck bed
        + t_lhd_tram_empty_min             # Tram 35m empty back to face
    )

    # Total truck loading time: acquisition/positioning delay + truck spot time + 2 LHD bucket loading passes
    t_load_truck_min = (
        config.lhd_acquisition_delay_minutes  # 1.5 min avg wait / positioning delay upon truck arrival
        + config.truck_load_spot_minutes       # 0.82 min truck backing into spot
        + 2 * t_lhd_bucket_cycle_min           # 2 LHD bucket cycles required to fill 1 truck bed
    )
    
    return t_load_truck_min / 60.0  # Convert total load time from minutes to hours


class ShelswellHaulageModel(drs.Module):
    """Discrete Rate Simulation (DRS) model approximating the Shelswell (2017) truck haulage fleet simulation.
    
    Models continuous material flow rates (tonnes/day) of ore and waste haulage subject to:
    - Bounded fleet capacity (truck count, mechanical availability, operator availability).
    - Match Factor (MF) bottlenecks balancing truck haulage capacity vs LHD loader capacity.
    - Stochastic daily muck scheduling from stope production and development faces.
    """

    def __init__(self, config: ShelswellConfig):
        super().__init__()
        self.config = config

        # --- DRS Timers ---
        # Global simulation clock in days (0.0 to 365.0 days)
        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)
        
        # 1-day interval timer triggering daily production scheduling updates
        self.schedule_timer = drs.Timer("ScheduleTimer", initial_value=0.0)
        self.schedule_timer.upper_threshold = 1.0  # Triggers daily muck replenishment at 1.0 day

        # --- DRS Level States (Material Piles & Production Accumulators) ---
        # Muck piles (tonnes) sitting in underground loading bays ready to be hauled to surface
        self.ore_muck = drs.Level("OreMuck", initial_value=100000.0)      # Ore muck inventory (t)
        self.waste_muck = drs.Level("WasteMuck", initial_value=20000.0)   # Waste rock muck inventory (t)

        # Cumulative material hauled and dumped at surface facilities (tonnes)
        self.ore_hauled = drs.Level("OreHauled", initial_value=0.0)       # Ore delivered to surface ROM pad (t)
        self.waste_hauled = drs.Level("WasteHauled", initial_value=0.0)   # Waste delivered to waste dump (t)

        # --- Instantaneous Haulage Rates (Tonnes/Day) ---
        self.ore_haul_rate = drs.Variable("ore_haul_rate", 0.0)     # Continuous ore haul rate (t/d)
        self.waste_haul_rate = drs.Variable("waste_haul_rate", 0.0) # Continuous waste haul rate (t/d)

        # --- Fleet Parametric Inputs ---
        self.truck_count = drs.Variable("truck_count", config.total_truck_count)  # Physical trucks in fleet
        self.operator_count = drs.Variable("operator_count", config.total_operators) # Truck drivers available
        self.availability = drs.Variable(
            "availability", config.overall_mechanical_availability
        ) # Mechanical availability fraction (0.0 to 1.0)

    def forward(self):
        """DRS Rate Evaluation Step executed continuously by the DRSEngine."""
        # Timers advance at real-time rate (1 day simulated per 1 day model time)
        self.global_time.rate = 1.0
        self.schedule_timer.rate = 1.0

        # --- Daily Production Schedule Replenishment ---
        # When schedule timer hits 1.0 day threshold, replenish underground muck piles based on
        # triangular distributions matching Shelswell (2017) paper targets for 1 year of production.
        # NOTE: Shelswell (2017) DES tracks individual stope muck bays and dynamically dispatches trucks 
        # to the loadout with highest unclaimed tonnes; DRS replenishes muck levels daily and applies a 
        # continuous 5.5:1 ore/waste capacity dispatch fraction.
        if self.schedule_timer.value >= 1.0 - 1e-6:
            self.schedule_timer.reset()
            # Daily ore production target: Triangular(min=3900, max=8100, mode=6000) tonnes/day
            ore_scheduled = random.triangular(3900.0, 8100.0, 6000.0)
            # Daily waste development target: Triangular(min=436.4, max=1745.4, mode=1090.9) tonnes/day
            waste_scheduled = random.triangular(436.4, 1745.4, 1090.9)

            # Add newly blasted muck to underground muck bay levels
            self.ore_muck.value += ore_scheduled
            self.waste_muck.value += waste_scheduled

        # Extract current parametric fleet parameters
        num_trucks = self.truck_count.value
        num_operators = self.operator_count.value
        avail = self.availability.value

        # --- Bounded Effective Fleet Constraint ---
        # Real-World Principle: An operational haulage unit requires BOTH a mechanically sound truck
        # AND an available human operator.
        # - Mechanically available trucks = num_trucks * mechanical_availability
        # - Effective active trucks = min(mechanically available trucks, available operators)
        # NOTE: Shelswell (2017) DES tracks individual operator resource acquisition at shift start; 
        # DRS caps effective active fleet size analytically as min(N_trucks * Availability, N_operators).
        eff_trucks = min(num_trucks * avail, num_operators)

        # --- Active Mining Level Allocation ---
        # Shelswell (2017) paper distributes active fleet across active production levels (1 to 7)
        # to model simultaneous stoping operations and prevent single-level loading bottlenecks.
        active_levels_count = max(1, min(7, int(num_trucks) // 2 + 1))  # Fleet size dictates active level count
        start_lvl = max(1, 4 - (active_levels_count - 1) // 2)            # Center levels around mid-mine (Level 4)
        active_levels = list(range(start_lvl, start_lvl + active_levels_count))

        # Evenly allocate effective active trucks per operating level
        trucks_per_level = eff_trucks / len(active_levels)

        total_ore_cap = 0.0
        total_waste_cap = 0.0

        # --- Compute Level-by-Level Haulage & Loading Capacities ---
        # NOTE: Shelswell (2017) DES generates discrete truck queue events at LHD loading bays; 
        # DRS solves level throughput analytically at each step using Match Factor (MF) equations.
        for lvl in active_levels:
            # Paper dispatch rule: Material split follows 5.5:1 ratio of ore to waste rock tonnage
            frac_ore = 5.5 / 6.5    # ~84.6% of haulage fleet capacity assigned to ore
            frac_waste = 1.0 / 6.5  # ~15.4% of haulage fleet capacity assigned to waste

            # =========================================================================
            # 1. Ore Haulage Capacity & Match Factor Analysis
            # =========================================================================
            # Travel times (hours) for ore cycle from current level
            t_empty_ore, t_loaded_ore = get_travel_times_hours(lvl, "ore", self.config)
            t_load_ore = get_truck_loading_time_hours("ore", self.config)  # LHD loading duration (hours)
            t_dump_ore = (
                self.config.truck_dump_spot_minutes + self.config.truck_dump_minutes
            ) / 60.0  # Truck surface dumping duration (hours)
            
            # Traffic congestion delay penalty based on truck density allocated to level
            t_traffic = self.config.traffic_delay_per_truck_hours * trucks_per_level

            # Total round-trip truck cycle time for ore (hours)
            t_cycle_ore = (
                t_empty_ore + t_load_ore + t_loaded_ore + t_dump_ore + t_traffic
            )

            # Match Factor (MF) for Ore = (Truck Arrival Rate) / (LHD Loader Service Rate)
            # MF = (N_trucks * t_load) / t_cycle
            # - MF < 1.0: Fleet is TRUCK-CONSTRAINED (loader waits for trucks; trucks govern throughput).
            # - MF >= 1.0: Fleet is LOADER-CONSTRAINED (loader runs 100% duty cycle; loader governs throughput).
            mf_ore = (trucks_per_level * t_load_ore) / t_cycle_ore

            if mf_ore < 1.0:
                # Truck-constrained capacity: throughput governed by truck fleet cycle count & truck seat time
                ore_cap = (
                    (trucks_per_level * frac_ore)                                      # Allocated ore trucks
                    * (24.0 * self.config.seat_time_fraction_truck / t_cycle_ore)     # Daily round trips per truck
                    * self.config.truck_payload_ore                                   # Tonnes ore per truck load
                )
            else:
                # Loader-constrained capacity: throughput capped by max LHD mucking rate & LHD seat time
                ore_cap = (
                    (1.0 * frac_ore)                                                  # 1 active LHD loader per level
                    * (24.0 * self.config.seat_time_fraction_lhd / t_load_ore)        # Daily LHD loading cycles
                    * self.config.truck_payload_ore                                   # Tonnes ore loaded per cycle
                )

            # =========================================================================
            # 2. Waste Haulage Capacity & Match Factor Analysis
            # =========================================================================
            # Travel times (hours) for waste cycle from current level
            t_empty_waste, t_loaded_waste = get_travel_times_hours(lvl, "waste", self.config)
            t_load_waste = get_truck_loading_time_hours("waste", self.config) # LHD loading duration (hours)
            t_dump_waste = (
                self.config.truck_dump_spot_minutes + self.config.truck_dump_minutes
            ) / 60.0  # Truck surface dumping duration (hours)

            # Total round-trip truck cycle time for waste (hours)
            t_cycle_waste = (
                t_empty_waste + t_load_waste + t_loaded_waste + t_dump_waste + t_traffic
            )

            # Match Factor (MF) for Waste
            mf_waste = (trucks_per_level * t_load_waste) / t_cycle_waste
            if mf_waste < 1.0:
                # Truck-constrained capacity for waste
                waste_cap = (
                    (trucks_per_level * frac_waste)                                    # Allocated waste trucks
                    * (24.0 * self.config.seat_time_fraction_truck / t_cycle_waste)   # Daily round trips per truck
                    * self.config.truck_payload_waste                                 # Tonnes waste per truck load
                )
            else:
                # Loader-constrained capacity for waste
                waste_cap = (
                    (1.0 * frac_waste)                                                # 1 active LHD loader per level
                    * (24.0 * self.config.seat_time_fraction_lhd / t_load_waste)      # Daily LHD loading cycles
                    * self.config.truck_payload_waste                                 # Tonnes waste loaded per cycle
                )

            # Accumulate multi-level haulage capacities (tonnes/day)
            total_ore_cap += ore_cap
            total_waste_cap += waste_cap

        # Set DRS continuous haulage variables (tonnes/day)
        self.ore_haul_rate.value = total_ore_cap
        self.waste_haul_rate.value = total_waste_cap

        # DRS differential state equations:
        # - Underground muck levels drain at haul rates: d(Muck)/dt = -Haul_Rate
        # - Surface hauled levels accumulate at haul rates: d(Hauled)/dt = +Haul_Rate
        self.ore_muck.rate = -self.ore_haul_rate.value
        self.waste_muck.rate = -self.waste_haul_rate.value
        self.ore_hauled.rate = self.ore_haul_rate.value
        self.waste_hauled.rate = self.waste_haul_rate.value

        # Physical boundary condition: Muck bays cannot drain below 0 tonnes remaining
        self.ore_muck.lower_threshold = 0.0
        self.waste_muck.lower_threshold = 0.0

    def is_terminating_condition_met(self) -> bool:
        """Simulation termination check: stops engine when global clock reaches 365 calendar days."""
        return self.global_time.value >= 365.0


def run_simulation(trucks: int, operators: int, availability: float) -> float:
    """Executes a single simulation run for a given fleet configuration over 354 production days.

    NOTE: Shelswell (2017) DES explicitly simulates 365 calendar days with 11 discrete planned 
    shutdown/holiday days. DRS simulates 354 active production days with continuous daily rates.

    Args:
        trucks: Number of haul trucks in fleet (3 to 10).
        operators: Number of truck operators available per shift (1 to 10).
        availability: Mechanical availability fraction (0.5 to 1.0).

    Returns:
        float: Average daily haulage productivity (tonnes/day).
    """
    # Set static random seeds for deterministic reproducibility across runs
    random.seed(42)
    np.random.seed(42)


    config = ShelswellConfig()

    # Instantiate DRS haulage model and assign sweep parameters
    model = ShelswellHaulageModel(config)
    model.truck_count.value = float(trucks)
    model.operator_count.value = float(operators)
    model.availability.value = float(availability)

    # Initialize DRS solver engine and execute for 354 active production days
    engine = drs.DRSEngine(model)
    engine.run(max_time=354.0)

    # Calculate average daily total productivity (ore + waste tonnes delivered to surface / 354 days)
    total_hauled = model.ore_hauled.value + model.waste_hauled.value
    avg_daily_productivity = total_hauled / 354.0
    return avg_daily_productivity


def generate_figure_2():
    """Replicates Figure 2 from Shelswell (2017): Productivity vs Fleet Size without operator constraints.
    
    Sweeps truck fleet sizes (3 to 10 trucks) across mechanical availability levels (50% to 100%), 
    assuming operator count matches truck count (unconstrained labor).
    Productivity is normalized against the baseline case (10 trucks, 10 ops, 100% availability).
    """
    print(
        "Generating Figure 2 (Productivity vs Fleet Size without operator constraints)..."
    )
    availabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Availability curves (50% to 100%)
    truck_sizes = list(range(3, 11))                  # Fleet sizes (3 to 10 trucks)

    # Compute baseline reference productivity for normalization (10 trucks, 10 ops, 100% avail)
    base_prod = run_simulation(10, 10, 1.0)
    print(f"Base productivity (10 trucks, 10 ops, 100% avail): {base_prod:.2f} t/d")

    plt.figure(figsize=(10, 6))

    # Matplotlib line color mapping matching Shelswell (2017) Figure 2 conventions
    colors = {
        0.5: "green",
        0.6: "brown",
        0.7: "orange",
        0.8: "blue",
        0.9: "hotpink",
        1.0: "black",
    }

    # Iterate over availability levels and plot normalized productivity curves
    for avail in availabilities:
        prod_list = []
        for trucks in truck_sizes:
            # Unconstrained operator sweep: operators = trucks
            prod = run_simulation(trucks, trucks, avail)
            prod_list.append(prod / base_prod)  # Normalize against 10-truck 100% avail baseline

        plt.plot(
            truck_sizes,
            prod_list,
            marker="o",
            label=f"{int(avail*100)}% availability",
            color=colors[avail],
        )

    plt.title(
        "Haulage productivity analysis without haulage operator constraints",
        fontsize=14,
    )
    plt.xlabel("Number of trucks in fleet", fontsize=12)
    plt.ylabel("Normalised productivity", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.ylim(0, 1.1)

    # Save output chart image to plots directory
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/shelswell_fig2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/shelswell_fig2.png")


def generate_figures_3_to_8():
    """Replicates Figures 3 through 8 from Shelswell (2017): Productivity vs Operator Count.
    
    Generates 6 separate figure plots (one for each mechanical availability level from 100% down to 50%).
    Each plot sweeps operator counts (1 to 10) for fleet sizes (3 to 10 trucks), demonstrating how
    operator shortages bottleneck fleet productivity regardless of truck inventory.
    """
    availabilities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # Availabilities for Figs 3, 4, 5, 6, 7, 8
    operator_counts = list(range(1, 11))              # Available operator sweep (1 to 10 drivers)
    truck_counts = list(range(3, 11))                 # Fleet size sweep (3 to 10 trucks)

    # Color scheme for truck fleet size series matching paper figures
    colors = {
        3: "green",
        4: "brown",
        5: "orange",
        6: "blue",
        7: "hotpink",
        8: "purple",
        9: "yellow",
        10: "black",
    }

    for avail in availabilities:
        print(
            f"Generating productivity curves for {int(avail*100)}% mechanical availability..."
        )
        # Baseline reference productivity for normalization at current availability (10 trucks, 10 ops)
        base_prod = run_simulation(10, 10, avail)

        plt.figure(figsize=(10, 6))
        for trucks in truck_counts:
            prod_list = []
            for ops in operator_counts:
                # Operator count is constrained by fleet size: min(ops, trucks)
                prod = run_simulation(trucks, min(ops, trucks), avail)
                prod_list.append(prod / base_prod)  # Normalize against 10-truck baseline

            plt.plot(
                operator_counts,
                prod_list,
                marker="o",
                label=f"{trucks} trucks",
                color=colors[trucks],
            )

        plt.title(
            f"Haulage fleet size productivity analysis with operator constraints ({int(avail*100)}% Availability)",
            fontsize=13,
        )
        plt.xlabel("Number of haulage operators", fontsize=12)
        plt.ylabel("Normalised productivity", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="lower right")
        plt.ylim(0, 1.1)

        # Map availability to corresponding Shelswell (2017) paper Figure number
        fig_num = {1.0: 3, 0.9: 4, 0.8: 5, 0.7: 6, 0.6: 7, 0.5: 8}[avail]
        plt.savefig(f"plots/shelswell_fig{fig_num}.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved plots/shelswell_fig{fig_num}.png")


if __name__ == "__main__":
    generate_figure_2()
    generate_figures_3_to_8()
    print("Replication of all figures complete!")


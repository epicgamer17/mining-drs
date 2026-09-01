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

from __future__ import annotations

import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, List

import drs
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paper specification constants (Shelswell & Labrecque 2017)
# ---------------------------------------------------------------------------
DAYS_IN_YEAR = 365.0
NON_PRODUCTION_DAYS = 11

SHIFT_SECONDS = 12.0 * 3600.0  # 12 h calendar slot per shift
SHIFT_WORK_HOURS = 10.5  # 10.5 h shift duration (Section 2.4)
HAULAGE_SEAT_FRACTION = 0.5417  # 54.17 % of the 12 h shift is usable
SEAT_PER_SHIFT_SEC = HAULAGE_SEAT_FRACTION * SHIFT_SECONDS  # ~6.5 h

# Mine geometry (Section 2.2)
DECLINE_M = 2100.0  # 5 % grade decline
LEVEL_SPACING_M = 300.0  # ramp spacing between the 7 levels
ORE_LEVEL_DRIFT_M = 60.0  # 40 m loadout + 20 m air door
WASTE_LEVEL_DRIFT_M = 75.0  # 55 m loadout + 20 m air door
ROM_SURFACE_M = 300.0  # ROM pad from portal
WASTE_SURFACE_M = 440.0  # waste stockpile from portal
N_LEVELS = 7

# Haulage truck speeds (Table 1, kph)
SPEEDS = {
    "surface": {"empty": 17.4, "loaded": 13.4},
    "decline": {"empty": 15.1, "loaded": 11.2},
    "ramp": {"empty": 12.9, "loaded": 9.2},
    "level": {"empty": 7.6, "loaded": 6.6},
}

# Trucks / loadout (Table 1)
ORE_PAYLOAD = 26.1
WASTE_PAYLOAD = 24.6
TRUCK_LOAD_SPOT_MIN = 0.82  # ± 25 % triangular (Section 2.5.2)
LHD_ACQUISITION_MAX_MIN = 3.0  # uniform 0..max LHD acquisition delay
TRUCK_LOAD_DUR_MIN = 6.69  # Table 1 truck load ore/waste (± 20 %)
BUCKET_PASSES = 2.0  # two LHD buckets per truck payload
LHD_LOAD_SPOT_MIN = 0.46
LHD_LOAD_MIN = 0.88
LHD_DUMP_MIN = 0.73
LHD_TRAM_M = 35.0
LHD_SPEED_LOADED_KPH = 5.89
LHD_SPEED_EMPTY_KPH = 6.78

# Dumping (Table 1, Section 2.5.3)
DUMP_SPOT_MIN = 0.57  # ± 20 % triangular
DUMP_MIN = 0.88  # ± 10 % triangular
ROM_TIP_SITES = 2
WASTE_TIP_SITES = 1

# Production scheduling (Section 2.3) - comfortably above fleet capacity so
# haulage (not mine production) is the measured bound.
ORE_WASTE_RATIO = 5.5
ORE_DAILY_PER_LEVEL = 2200.0  # symmetric triangular ± 35 %
WASTE_DAILY_PER_LEVEL = 400.0  # triangular with +60 % upper limit
ORE_SCHEDULE_TOL = 0.35
WASTE_SCHEDULE_TOL = 0.60

# Refuelling (Section 2.5.5): staggered, 2 pumps, retains the operator.
FUEL_BURN_PCT_PER_SEC = 100.0 / (7.5 * 3600.0)  # ~ tank per 7.5 operating h
REFUEL_DUR_MIN = 25.0  # refuel event duration (± 10 %)
N_FUEL_PUMPS = 2

# Traffic (Section 2.5.4 / Table 2).  The paper models single-direction
# corridor segments with pass bays, loaded trucks keeping the right-of-pass,
# and reports fleet-total traffic growing with *both* truck and operator
# numbers (0-0.63 % of calendar at 3 trucks up to 0-2.72 % at 10, i.e.
# ~2.16 min/shift per extra truck).  We reproduce that trend directionally by
# charging the empty leg a per-trip allowance that grows with the number of
# trucks currently operating in the shared decline/ramp corridor (~one active
# level per two trucks), so congestion rises with fleet size and with operator
# availability (fewer operators => fewer trucks on the road).  The base keeps
# the verified 3-truck cycle profile and stays within the Table 2 maxima.
BASE_PASS_BAY_DELAY_SEC = 13.0  # empty-leg allowance at the 3-truck base
PER_TRUCK_PASS_BAY_DELAY_SEC = 1.0  # extra delay per additional operating truck

DT_MAX = 900.0  # engine drift step cap (sec); event
# boundaries remain exact


# ---------------------------------------------------------------------------
# Shared samplers / geometry helpers
# ---------------------------------------------------------------------------
def _tri(rng: random.Random, mid: float, tol: float) -> float:
    """Symmetric triangular distribution around ``mid`` with width ``tol``."""
    return rng.triangular(mid * (1.0 - tol), mid * (1.0 + tol), mid)


def _lhd_bucket_cycle_sec(rng: random.Random) -> float:
    """One LHD bucket pass (kept for reference; Table-1 load time is used)."""
    tram_loaded_s = LHD_TRAM_M / (LHD_SPEED_LOADED_KPH / 3.6)
    tram_empty_s = LHD_TRAM_M / (LHD_SPEED_EMPTY_KPH / 3.6)
    base_min = (
        LHD_LOAD_SPOT_MIN
        + LHD_LOAD_MIN
        + tram_loaded_s / 60.0
        + LHD_DUMP_MIN
        + tram_empty_s / 60.0
    )
    return _tri(rng, base_min * 60.0, 0.20)


def _in_shift_window(t: float) -> bool:
    """Two 10.5 h shifts per day separated by two 1.5 h off-shift gaps."""
    hod = t % 86400.0
    return (0.0 <= hod < SHIFT_WORK_HOURS * 3600.0) or (
        12.0 * 3600.0 <= hod < 22.5 * 3600.0
    )


# ---------------------------------------------------------------------------
# Discrete state model
# ---------------------------------------------------------------------------
class TruckPhase(Enum):
    IDLE = "idle"  # parked on surface, no operator
    EMPTY = "empty"  # empty travel surface -> loadout
    WAIT_LOAD = "wait_load"  # queued for the level LHD
    SPOT_LOAD = "spot_load"  # truck spotting at the loadout (±25 %)
    ACQUIRE = "acquire"  # waiting for the LHD to arrive (uniform 0-3)
    LOADING = "loading"  # two LHD bucket passes (±20 %)
    LOADED = "loaded"  # loaded travel loadout -> surface
    WAIT_DUMP = "wait_dump"  # queued at the dump tip sites
    DUMPING = "dumping"  # tip spot (±20 %) + dump (±10 %)
    REFUELING = "refueling"  # fuel depot (retains operator)


OPERATING_PHASES = {
    TruckPhase.EMPTY,
    TruckPhase.WAIT_LOAD,
    TruckPhase.SPOT_LOAD,
    TruckPhase.ACQUIRE,
    TruckPhase.LOADING,
    TruckPhase.LOADED,
    TruckPhase.WAIT_DUMP,
    TruckPhase.DUMPING,
}

SEAT_PHASES = OPERATING_PHASES | {TruckPhase.REFUELING}

DUE_PHASES = {
    TruckPhase.EMPTY,
    TruckPhase.SPOT_LOAD,
    TruckPhase.ACQUIRE,
    TruckPhase.LOADING,
    TruckPhase.LOADED,
    TruckPhase.DUMPING,
    TruckPhase.REFUELING,
}


@dataclass
class Operator:
    idx: int
    free: bool = True
    used_seat: float = 0.0


@dataclass
class Truck:
    truck_id: str
    timer: drs.Timer
    phase: TruckPhase = TruckPhase.IDLE
    payload_type: str = "ORE"  # "ORE" or "WASTE"
    target_level: int = 0
    target_loadout: int = -1  # index into sim.loadouts
    current_payload: float = 0.0
    seat_used: float = 0.0
    fuel: float = 100.0
    refuel_threshold: float = 30.0
    operator: int = -1  # index into sim.operators
    trip_start: float = 0.0
    dump_dur: float = 0.0
    down_start: float = math.inf  # downtime window (absolute sec)
    down_end: float = math.inf


@dataclass
class Loadout:
    idx: int
    level: int
    bay_type: str
    muck_remaining: drs.Level
    active: bool = True
    queue: list = field(default_factory=list)
    last_assigned_seq: int = 0  # round-robin tie breaker


@dataclass
class DumpSite:
    name: str
    capacity: int
    in_use: int = 0
    queue: list = field(default_factory=list)
    dumped_level: drs.Level = None
    _active_rate: float = 0.0


# ---------------------------------------------------------------------------
# Simulation module (registered with the DRSEngine)
# ---------------------------------------------------------------------------
class ShelswellHaulage(drs.Module):
    """Faithful event-driven reuse of Shelswell's haulage specification.

    Each truck carries a DRS countdown timer; an ``on_step`` policy performs
    the discrete transitions and a continuous DRS layer integrates dump
    accumulators, muck levels and seat/fuel bookkeeping.
    """

    def __init__(
        self,
        num_trucks: int = 10,
        num_operators: int = 10,
        availability: float = 1.0,
        seed: int = 42,
        pm_availability: float | None = None,
        random_failure_availability: float | None = None,
    ):
        super().__init__()
        self.num_trucks = num_trucks
        self.num_operators = num_operators
        # Paper Section 3.2 / Equations 1-2: overall mechanical availability is
        # the product of the planned-maintenance-associated availability and the
        # random-failure-associated availability.  Either pass the compounded
        # value via ``availability`` or supply the two factors separately.
        if pm_availability is not None and random_failure_availability is not None:
            availability = pm_availability * random_failure_availability
        self.availability = availability
        self.pm_availability = pm_availability
        self.random_failure_availability = random_failure_availability
        self.rng = random.Random(seed)
        self.horizon_sec = DAYS_IN_YEAR * 86400.0

        # Availability: operating credit per shift = A x seat time.  Each
        # shift also gets one downtime window of (1-A) x seat time at a
        # random offset (frees the operator; Equations 1-2).
        self.truck_seat_credit = availability * SEAT_PER_SHIFT_SEC
        self._down_dur = max(0.0, (1.0 - availability) * SEAT_PER_SHIFT_SEC)

        # Calendar housekeeping
        day_pool = list(range(int(DAYS_IN_YEAR)))
        self.rng.shuffle(day_pool)
        self.holidays = set(day_pool[:NON_PRODUCTION_DAYS])
        self._cur_day = -1
        self._shift_marker = -1
        self._holiday_today = False

        # Continuous DRS layer
        self.gt = drs.Timer("gt", 0.0, rate=1.0)
        self.ore_hauled = drs.Level("ore_hauled", 0.0)
        self.waste_hauled = drs.Level("waste_hauled", 0.0)

        # Fleet
        self.trucks = []
        for i in range(1, num_trucks + 1):
            timer = drs.Timer(f"tr_{i}_act", 0.0, rate=-1.0)
            timer.lower_threshold = 0.0
            tr = Truck(truck_id=f"T{i:02d}", timer=timer)
            tr.refuel_threshold = self.rng.uniform(15.0, 40.0)
            self.trucks.append(tr)

        # Operators (one crew per shift, swung in at the shift boundary)
        self.operators = [Operator(i) for i in range(num_operators)]

        # Loadouts: one ore + one waste per level
        self.loadouts = []
        lo_idx = 0
        for level in range(1, N_LEVELS + 1):
            for bay_type in ("ORE", "WASTE"):
                self.loadouts.append(
                    Loadout(
                        idx=lo_idx,
                        level=level,
                        bay_type=bay_type,
                        muck_remaining=drs.Level(f"muck_L{level}_{bay_type}", 0.0),
                    )
                )
                lo_idx += 1

        # Section 3.1: one active level per two haulage trucks, centred in the
        # seven ramp levels, so the tonnage-weighted average haul stays at the
        # paper's ~900 m ramp midpoint for every fleet ("similar average haul
        # distance for all simulation scenarios").  The active set is chosen
        # symmetrically about the middle level (L4 = 900 m).
        k = max(1, int(math.ceil(num_trucks / 2.0)))
        lvls = []
        if k % 2 == 1:
            lvls.append(4)
        for d in range(1, k // 2 + 1):
            lvls += [4 - d, 4 + d]
        self.active_levels = set(lvls)
        for lo in self.loadouts:
            lo.active = lo.level in self.active_levels

        # Surface dump tips (Section 2.5.3): ROM 2 sites, waste 1 site
        self.dump_sites = {
            "ORE": DumpSite("ROM_PAD", ROM_TIP_SITES),
            "WASTE": DumpSite("WASTE_STOCKPILE", WASTE_TIP_SITES),
        }
        self.dump_sites["ORE"].dumped_level = self.ore_hauled
        self.dump_sites["WASTE"].dumped_level = self.waste_hauled

        # Shared level LHD resource (one per level, both loadouts)
        self._lhd_busy = {level: False for level in range(1, N_LEVELS + 1)}
        self._pumps_free = N_FUEL_PUMPS
        self._dispatch_counter = 0
        self.trips = 0
        self._cycle_sum = 0.0
        self.traffic_delay_sum = 0.0

    def levels(self) -> Sequence[drs.Level]:
        base_levels: List[drs.Level] = [
            self.gt,
            self.ore_hauled,
            self.waste_hauled,
        ]
        if hasattr(self, "dump_sites"):
            for dump in self.dump_sites.values():
                if getattr(dump, "dumped_level", None) is not None:
                    base_levels.append(dump.dumped_level)
        if hasattr(self, "trucks"):
            for tr in self.trucks:
                if getattr(tr, "timer", None) is not None:
                    base_levels.append(tr.timer)
        if hasattr(self, "loadouts"):
            for lo in self.loadouts:
                if getattr(lo, "muck_remaining", None) is not None:
                    base_levels.append(lo.muck_remaining)
        return tuple(base_levels)

    # -- DRS (Layer 3) hooks ------------------------------------------------
    def time_to_event(self) -> float:
        """Next truck activity boundary or calendar boundary (relative sec)."""
        best = DT_MAX
        for tr in self.trucks:
            v = tr.timer.value
            if v > 1e-9:
                best = min(best, v)
        t = self.gt.value
        next_day = (math.floor(t / 86400.0) + 1.0) * 86400.0
        next_shift = (math.floor(t / SHIFT_SECONDS) + 1.0) * SHIFT_SECONDS
        best = min(best, next_day - t, next_shift - t)
        return max(best, 1e-6)

    def step(self, dt: float):
        """Continuous DRS integration between discrete events."""
        self.gt.step(dt)
        for tr in self.trucks:
            if tr.timer.value > 0.0:
                tr.timer.step(dt)
            if tr.phase in SEAT_PHASES:
                tr.seat_used = min(self.truck_seat_credit, tr.seat_used + dt)
                if tr.phase in OPERATING_PHASES:
                    tr.fuel = max(0.0, tr.fuel - dt * FUEL_BURN_PCT_PER_SEC)
                if tr.operator >= 0:
                    op = self.operators[tr.operator]
                    op.used_seat = min(SEAT_PER_SHIFT_SEC, op.used_seat + dt)
        for site in self.dump_sites.values():
            site.dumped_level.step(dt)

    def is_terminating_condition_met(self) -> bool:
        return self.gt.value >= self.horizon_sec - 1e-6

    # -- Discrete event policy (Layer 1) -------------------------------------
    def on_event(self, t: float):
        """Engine step policy: calendar + truck transitions + dispatching."""
        self._calendar_update()
        guard = 0
        changed = True
        while changed and guard < 200:
            changed = False
            guard += 1
            for tr in self.trucks:
                if tr.phase == TruckPhase.IDLE:
                    if self._try_dispatch(tr):
                        changed = True
                elif tr.phase in DUE_PHASES and tr.timer.value <= 1e-6:
                    if self._advance(tr):
                        changed = True

    def _calendar_update(self):
        t = self.gt.value
        day = int(t // 86400.0)
        if self._cur_day != day:
            self._cur_day = day
            self._holiday_today = day in self.holidays
            if not self._holiday_today:
                self._schedule_daily()
        marker = int(t // SHIFT_SECONDS)
        if self._shift_marker != marker:
            self._shift_marker = marker
            # Operators are bound while their truck retains them (including a
            # truck waiting to refuel or freshly back on surface, per Section
            # 2.5.1), not only while the truck is mid-haul.
            bound = {tr.operator for tr in self.trucks if tr.operator >= 0}
            for op in self.operators:
                op.used_seat = 0.0
                op.free = op.idx not in bound
            for tr in self.trucks:
                tr.seat_used = 0.0
                self._schedule_down_window(tr, t)

    def _schedule_daily(self):
        """Additive daily ore/waste calls to the active level loadouts."""
        for lo in self.loadouts:
            if not lo.active or lo.muck_remaining.value < 0:
                continue
            if lo.bay_type == "ORE":
                lo.muck_remaining.value += self.rng.triangular(
                    ORE_DAILY_PER_LEVEL * (1 - ORE_SCHEDULE_TOL),
                    ORE_DAILY_PER_LEVEL * (1 + ORE_SCHEDULE_TOL),
                    ORE_DAILY_PER_LEVEL,
                )
            else:
                # Waste has an upper limit of +60 % (Section 2.3): triangular
                # from the nominal value up to nominal x (1 + 0.60), peak at the
                # nominal, so variation is only upward.
                lo.muck_remaining.value += self.rng.triangular(
                    WASTE_DAILY_PER_LEVEL,
                    WASTE_DAILY_PER_LEVEL * (1.0 + WASTE_SCHEDULE_TOL),
                    WASTE_DAILY_PER_LEVEL,
                )

    def _schedule_down_window(self, tr: Truck, t: float):
        """Place this shift's downtime window at a random seat-time offset."""
        if self._down_dur <= 1e-6:
            tr.down_start = math.inf
            tr.down_end = math.inf
            return
        shift_start = math.floor(t / SHIFT_SECONDS) * SHIFT_SECONDS
        offset = self.rng.uniform(0.0, max(1.0, SEAT_PER_SHIFT_SEC - self._down_dur))
        tr.down_start = shift_start + offset
        tr.down_end = tr.down_start + self._down_dur

    def _in_down_window(self, tr: Truck, t: float) -> bool:
        return tr.down_start <= t < tr.down_end

    # -- Transitions ----------------------------------------------------------
    def _advance(self, tr: Truck) -> bool:
        ph = tr.phase
        if ph == TruckPhase.EMPTY:
            self._enter_loadout(tr)
            return True
        if ph == TruckPhase.SPOT_LOAD:
            tr.phase = TruckPhase.ACQUIRE
            tr.timer.value = self.rng.uniform(0.0, LHD_ACQUISITION_MAX_MIN) * 60.0
            return True
        if ph == TruckPhase.ACQUIRE:
            tr.phase = TruckPhase.LOADING
            tr.timer.value = _tri(self.rng, TRUCK_LOAD_DUR_MIN * 60.0, 0.20)
            return True
        if ph == TruckPhase.LOADING:
            self._finish_loading(tr)
            return True
        if ph == TruckPhase.LOADED:
            self._enter_dump(tr)
            return True
        if ph == TruckPhase.DUMPING:
            self._finish_dumping(tr)
            return True
        if ph == TruckPhase.REFUELING:
            self._finish_refuel(tr)
            return True
        return False

    def _enter_loadout(self, tr: Truck):
        lo = self.loadouts[tr.target_loadout]
        if self._lhd_busy[lo.level]:
            lo.queue.append(tr)
            tr.phase = TruckPhase.WAIT_LOAD
            tr.timer.value = 0.0
        else:
            self._lhd_busy[lo.level] = True
            tr.phase = TruckPhase.SPOT_LOAD
            tr.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)

    def _finish_loading(self, tr: Truck):
        lo = self.loadouts[tr.target_loadout]
        cap = ORE_PAYLOAD if tr.payload_type == "ORE" else WASTE_PAYLOAD
        tr.current_payload = _tri(self.rng, cap, 0.10)
        self._lhd_busy[lo.level] = False
        self._serve_next_loadout(lo.level)
        tr.phase = TruckPhase.LOADED
        tr.timer.value = self._travel_time(tr, loaded=True)

    def _serve_next_loadout(self, level: int):
        for lo in self.loadouts:
            if lo.level != level:
                continue
            if lo.queue:
                nxt = lo.queue.pop(0)
                nxt.phase = TruckPhase.SPOT_LOAD
                nxt.timer.value = _tri(self.rng, TRUCK_LOAD_SPOT_MIN * 60.0, 0.25)
                return
        self._lhd_busy[level] = False

    def _enter_dump(self, tr: Truck):
        site = self.dump_sites[tr.payload_type]
        if site.in_use < site.capacity:
            self._start_dump(site, tr)
        else:
            site.queue.append(tr)
            tr.phase = TruckPhase.WAIT_DUMP
            tr.timer.value = 0.0

    def _start_dump(self, site: DumpSite, tr: Truck):
        dur = _tri(self.rng, DUMP_SPOT_MIN * 60.0, 0.20) + _tri(
            self.rng, DUMP_MIN * 60.0, 0.10
        )
        site.in_use += 1
        tr.phase = TruckPhase.DUMPING
        tr.timer.value = dur
        tr.dump_dur = dur
        site._active_rate += tr.current_payload / dur
        site.dumped_level.rate = site._active_rate

    def _finish_dumping(self, tr: Truck):
        site = self.dump_sites[tr.payload_type]
        site._active_rate = max(
            0.0, site._active_rate - tr.current_payload / tr.dump_dur
        )
        site.dumped_level.rate = site._active_rate
        site.in_use -= 1
        if site.queue:
            nxt = site.queue.pop(0)
            self._start_dump(site, nxt)
        self.trips += 1
        self._cycle_sum += self.gt.value - tr.trip_start
        tr.current_payload = 0.0
        tr.target_loadout = -1
        tr.target_level = 0
        # Operator is retained across trips (paper Section 2.5.1: trucks keep
        # their operator through refuelling and re-dispatch; the operator is
        # only released when the truck goes idle because it cannot be
        # dispatched, which happens in _try_dispatch).
        tr.phase = TruckPhase.IDLE
        tr.timer.value = 0.0

    def _finish_refuel(self, tr: Truck):
        self._pumps_free += 1
        tr.fuel = 100.0
        # The operator was retained through refuelling (Section 2.5.1) and
        # stays with the truck for re-dispatch.
        tr.phase = TruckPhase.IDLE
        tr.timer.value = 0.0

    # -- Dispatch ------------------------------------------------------------
    def _try_dispatch(self, tr: Truck) -> bool:
        t = self.gt.value
        if (
            self._holiday_today
            or not _in_shift_window(t)
            or tr.seat_used >= self.truck_seat_credit
            or self._in_down_window(tr, t)
        ):
            # Paper Section 2.5.1: a truck that is not dispatched and goes
            # idle releases its operator.
            self._release_operator(tr)
            return False

        if tr.fuel <= tr.refuel_threshold:
            # Trucks retain their operator when refuelling (Section 2.5.1), so
            # grab one before routing to the depot; only route if a pump is free.
            if self._pumps_free > 0:
                if tr.operator < 0 and not self._acquire_operator(tr):
                    return False
                self._pumps_free -= 1
                tr.phase = TruckPhase.REFUELING
                tr.timer.value = _tri(self.rng, REFUEL_DUR_MIN * 60.0, 0.10)
                return True
            return False  # wait for a free pump (keeps any retained operator)

        # Payload type by the 5.5:1 ore: waste schedule ratio
        if self.rng.random() < ORE_WASTE_RATIO / (ORE_WASTE_RATIO + 1.0):
            ptype = "ORE"
        else:
            ptype = "WASTE"
        tr.payload_type = ptype

        cands = [
            lo
            for lo in self.loadouts
            if lo.active and lo.bay_type == ptype and lo.muck_remaining.value > 5.0
        ]
        if not cands:
            self._release_operator(tr)
            return False

        if not self._acquire_operator(tr):
            self._release_operator(tr)
            return False

        # Dispatch to the loadout with the highest unclaimed tonnes (Section
        # 2.5.1 / notes); unclaimed = inventory minus trucks already assigned,
        # so several trucks may legitimately target the same level and queue
        # at the shared LHD.  Tie -> least-recently assigned (round robin).
        target = max(
            cands, key=lambda lo: (lo.muck_remaining.value, -lo.last_assigned_seq)
        )
        self._dispatch_counter += 1
        target.last_assigned_seq = self._dispatch_counter
        claim = ORE_PAYLOAD if ptype == "ORE" else WASTE_PAYLOAD
        target.muck_remaining.value = max(0.0, target.muck_remaining.value - claim)
        tr.target_loadout = target.idx
        tr.target_level = target.level
        tr.trip_start = self.gt.value
        tr.phase = TruckPhase.EMPTY
        tr.timer.value = self._travel_time(tr, loaded=False)
        return True

    def _acquire_operator(self, tr: Truck) -> bool:
        if tr.operator >= 0 and not self.operators[tr.operator].free:
            return self.operators[tr.operator].used_seat < SEAT_PER_SHIFT_SEC
        for op in self.operators:
            if op.free and op.used_seat < SEAT_PER_SHIFT_SEC:
                op.free = False
                tr.operator = op.idx
                return True
        return False

    def _release_operator(self, tr: Truck):
        if tr.operator >= 0:
            self.operators[tr.operator].free = True
            tr.operator = -1

    # -- Travel --------------------------------------------------------------
    def _travel_time(self, tr: Truck, loaded: bool) -> float:
        """One leg surface<->loadout at the empty/loaded speed profile."""
        load_key = "loaded" if loaded else "empty"
        ore = tr.payload_type == "ORE"
        d_surf = ROM_SURFACE_M if ore else WASTE_SURFACE_M
        d_lvl = ORE_LEVEL_DRIFT_M if ore else WASTE_LEVEL_DRIFT_M
        d_ramp = max(0.0, (tr.target_level - 1) * LEVEL_SPACING_M)

        def seg(dist: float, kind: str) -> float:
            return dist / (SPEEDS[kind][load_key] / 3.6)

        t = (
            seg(d_surf, "surface")
            + seg(DECLINE_M, "decline")
            + seg(d_ramp, "ramp")
            + seg(d_lvl, "level")
        )

        # Pass-bay / congestion surrogate (Section 2.5.4 / Table 2).
        # The paper's fleet-total traffic grows ~0.3 % of calendar per extra
        # truck (2.16 min per shift per truck), shared over the decline/ramp
        # corridor; loaded trucks keep the right-of-pass, so the allowance is
        # charged to the empty leg only.  Delay scales with the current number
        # of operating trucks, so it responds to both fleet size and operator
        # availability.  A small magnitude keeps the verified cycle profile
        # (Table 2, ~44.27 min) unchanged while capturing the trend.
        if not loaded:
            cong = sum(1 for t in self.trucks if t.phase in OPERATING_PHASES)
            delay = BASE_PASS_BAY_DELAY_SEC + PER_TRUCK_PASS_BAY_DELAY_SEC * max(
                0, cong - 3
            )
            t += delay
            self.traffic_delay_sum += delay
        return t

    # -- Runner ---------------------------------------------------------------
    def run(self, total_days: float = DAYS_IN_YEAR) -> float:
        """Runs the simulation and returns average daily tonnes hauled."""
        self.horizon_sec = total_days * 86400.0
        engine = drs.DRSEngine(max_step_size=DT_MAX)
        engine.register(self)
        engine.on_step(self.on_event)
        engine.run(until=self.horizon_sec)
        total = self.ore_hauled.value + self.waste_hauled.value
        self.result_tonnes_per_day = total / total_days
        return self.result_tonnes_per_day


# ---------------------------------------------------------------------------
# High level wrappers (mirror shelswell_baseline.py API)
# ---------------------------------------------------------------------------
def build_shelswell_simulation(
    num_trucks=10,
    num_operators=10,
    mechanical_availability=1.0,
    topology_dict=None,
    seed=42,
    pm_availability=None,
    random_failure_availability=None,
):
    """Constructs a :class:`ShelswellHaulage` instance.

    Overall availability is ``mechanical_availability``, or the product of
    ``pm_availability`` and ``random_failure_availability`` if both are given
    (paper Equations 1-2).
    """
    return ShelswellHaulage(
        num_trucks=num_trucks,
        num_operators=num_operators,
        availability=mechanical_availability,
        seed=seed,
        pm_availability=pm_availability,
        random_failure_availability=random_failure_availability,
    )


def run_haulage_simulation(state, total_days=365.0):
    """Runs the event-driven DRS integration for ``total_days`` (365 baseline)."""
    return state.run(total_days=total_days)


def run_simulation(trucks: int, operators: int, availability: float) -> float:
    """Executes a single run; returns average daily haulage productivity (t/d)."""
    seed = trucks * 10000 + operators * 100 + int(round(availability * 10.0))
    state = build_shelswell_simulation(
        num_trucks=trucks,
        num_operators=operators,
        mechanical_availability=availability,
        seed=seed,
    )
    return run_haulage_simulation(state, total_days=DAYS_IN_YEAR)


def _run_task(args):
    trucks, operators, avail = args
    return (trucks, operators, avail, run_simulation(trucks, operators, avail))


# ---------------------------------------------------------------------------
# Figure replication (Section 3.2/3.3 of the paper)
# ---------------------------------------------------------------------------
# Paper Table 2 (Shelswell & Labrecque 2017): (trucks, avg cycle min,
# loadout-queue % of calendar, traffic delay % of calendar).
PAPER_TABLE_2 = [
    (3, 44.89, 0.03, 0.63),
    (4, 44.56, 0.02, 0.94),
    (5, 44.20, 0.02, 1.24),
    (6, 44.45, 0.01, 1.56),
    (7, 44.57, 0.02, 1.85),
    (8, 44.07, 0.02, 2.11),
    (9, 43.61, 0.02, 2.42),
    (10, 43.79, 0.01, 2.72),
]


def verify_table_2(total_days=365.0) -> bool:
    """Checks the 100 % availability fleet profile against the paper's Table 2.

    Runs fleets of 3..10 trucks (one operator per truck) and reports cycle
    time, loadout-queue delay and traffic delay against the paper.  The paper
    reports "an average cycle time for all fleet compositions of 44.27
    minutes with a maximum variation of -1.5 % and +1.4 %"; the check is that
    every fleet's simulated cycle falls inside that 44.27 +/- 1.5 % band
    (43.61-44.94 min), with loadout queue ~0.02 % of calendar and traffic
    within the Table 2 maxima (0.63 % at 3 trucks to 2.72 % at 10 trucks).
    """
    avg = 44.27
    band = 0.015 * avg
    print(
        "Verifying fleet cycle profile against Shelswell & Labrecque (2017) Table 2 ..."
    )
    print(f"  (paper acceptance band: {avg - band:.2f} - {avg + band:.2f} min)")
    print(
        f"{'N':>3} | {'cycle':>6} {'in-band':>8} {'LQ %':>6} {'paper':>6}"
        f" | {'traffic s/cyc':>12} {'paper max %':>11}"
    )
    ok = True
    for n, cyc_p, lq_p, tr_p in PAPER_TABLE_2:
        sim = ShelswellHaulage(
            num_trucks=n, num_operators=n, availability=1.0, seed=10 * n
        )
        waits = [0.0]

        def step(self, dt):
            for tr in self.trucks:
                if tr.phase == TruckPhase.WAIT_LOAD:
                    waits[0] += dt
            return _orig_step(self, dt)

        _orig_step = ShelswellHaulage.step
        ShelswellHaulage.step = step
        try:
            sim.run(total_days=total_days)
        finally:
            ShelswellHaulage.step = _orig_step

        cycle = sim._cycle_sum / sim.trips / 60.0
        queue_pct = 100.0 * waits[0] / (sim.gt.value * n)
        tr_sec = sim.traffic_delay_sum / sim.trips
        in_band = avg - band <= cycle <= avg + band
        ok = ok and in_band
        shift_trips = (SEAT_PER_SHIFT_SEC / 60.0) / cycle
        tr_pct = tr_sec * shift_trips / (12.0 * 3600.0) * 100.0
        if tr_pct > tr_p + 1e-6:
            ok = False
        print(
            f"{n:3d} | {cycle:6.2f} {str(in_band):>8} {queue_pct:6.3f} {lq_p:6.2f}"
            f" | {tr_sec:12.1f} {tr_pct:11.2f} (paper max {tr_p:.2f})"
        )
    return ok


def generate_figure_2():
    """Figure 2: productivity vs fleet size without operator constraints."""
    print(
        "Generating Figure 2 (productivity vs fleet size, no operator constraints)..."
    )
    availabilities = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    truck_sizes = list(range(3, 11))

    base_prod = run_simulation(10, 10, 1.0)
    print(
        f"Base productivity (10 trucks, 10 ops, 100% availability): {base_prod:.2f} t/d"
    )

    tasks = [(t, t, a) for a in availabilities for t in truck_sizes]
    results_map = {}
    with ProcessPoolExecutor() as executor:
        for t, o, a, prod in tqdm(
            executor.map(_run_task, tasks), total=len(tasks), desc="Figure 2 sweep"
        ):
            results_map[(t, o, a)] = prod

    plt.figure(figsize=(10, 6))
    colors = {
        0.5: "green",
        0.6: "brown",
        0.7: "orange",
        0.8: "blue",
        0.9: "hotpink",
        1.0: "black",
    }
    for avail in availabilities:
        prod_list = [results_map[(t, t, avail)] / base_prod for t in truck_sizes]
        plt.plot(
            truck_sizes,
            prod_list,
            marker="o",
            label=f"{int(avail * 100)}% availability",
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
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/shelswell_fig2.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved plots/shelswell_fig2.png")


def generate_figures_3_to_8():
    """Figures 3-8: productivity vs operator count for each availability."""
    availabilities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    operator_counts = list(range(1, 11))
    truck_counts = list(range(3, 11))

    tasks = [
        (t, min(o, t), a)
        for a in availabilities
        for t in truck_counts
        for o in operator_counts
    ]
    results_map = {}
    with ProcessPoolExecutor() as executor:
        for t, o, a, prod in tqdm(
            executor.map(_run_task, tasks), total=len(tasks), desc="Figures 3-8 sweep"
        ):
            results_map[(t, o, a)] = prod

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
    markers = {3: "o", 4: "s", 5: "^", 6: "v", 7: "D", 8: "P", 9: "X", 10: "*"}

    fig_num = {1.0: 3, 0.9: 4, 0.8: 5, 0.7: 6, 0.6: 7, 0.5: 8}
    os.makedirs("plots", exist_ok=True)

    for avail in availabilities:
        base_prod = results_map[(10, 10, avail)]
        plt.figure(figsize=(10, 6))
        for trucks in reversed(truck_counts):
            prod_list = [
                results_map[(trucks, min(ops, trucks), avail)] / base_prod
                for ops in operator_counts
            ]
            plt.plot(
                operator_counts,
                prod_list,
                marker=markers[trucks],
                markersize=6,
                label=f"{trucks} trucks",
                color=colors[trucks],
                zorder=10 - trucks,
            )
        plt.title(
            f"Haulage fleet size productivity analysis with operator constraints "
            f"({int(avail * 100)}% Availability)",
            fontsize=13,
        )
        plt.xlabel("Number of haulage operators", fontsize=12)
        plt.ylabel("Normalised productivity", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(loc="lower right")
        plt.ylim(0, 1.1)
        plt.savefig(
            f"plots/shelswell_fig{fig_num[avail]}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    print("Saved plots/shelswell_fig3.png through plots/shelswell_fig8.png")


if __name__ == "__main__":
    verify_table_2()
    generate_figure_2()
    generate_figures_3_to_8()
    print("Replication complete (Shelswell 2017 figures 2-8).")

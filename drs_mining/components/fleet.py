"""Consolidated Fleet, Trucks, Operators, and Dump Stations.

Provides unified discrete-event and continuous haulage entities for mine simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import drs


class TruckPhase(Enum):
    """Discrete execution phases of a haul truck in the DES state machine."""

    IDLE = "idle"
    PARKED = "parked"
    EMPTY = "empty"
    WAIT_LOAD = "wait_load"
    SPOT_LOAD = "spot_load"
    ACQUIRE = "acquire"
    LOADING = "loading"
    LOADED = "loaded"
    WAIT_DUMP = "wait_dump"
    SPOT_DUMP = "spot_dump"
    DUMPING = "dumping"
    REFUELING = "refueling"
    MAINTENANCE = "maintenance"


OPERATING_PHASES: Set[TruckPhase] = {
    TruckPhase.EMPTY,
    TruckPhase.WAIT_LOAD,
    TruckPhase.SPOT_LOAD,
    TruckPhase.ACQUIRE,
    TruckPhase.LOADING,
    TruckPhase.LOADED,
    TruckPhase.WAIT_DUMP,
    TruckPhase.SPOT_DUMP,
    TruckPhase.DUMPING,
}

SEAT_PHASES: Set[TruckPhase] = OPERATING_PHASES | {TruckPhase.REFUELING}

DUE_PHASES: Set[TruckPhase] = {
    TruckPhase.EMPTY,
    TruckPhase.SPOT_LOAD,
    TruckPhase.ACQUIRE,
    TruckPhase.LOADING,
    TruckPhase.LOADED,
    TruckPhase.SPOT_DUMP,
    TruckPhase.DUMPING,
    TruckPhase.REFUELING,
}


@dataclass
class Operator:
    """Human operator assigned to a haul truck during a shift."""

    idx: int
    free: bool = True
    used_seat: float = 0.0


@dataclass
class Truck:
    """Unified haul truck entity tracked by both discrete and continuous simulation engines."""

    truck_id: str
    timer: Optional[drs.Timer] = None
    phase: TruckPhase = TruckPhase.PARKED
    target_face_id: int = 1
    target_loadout: int = -1
    target_level: int = 4
    current_payload: float = 0.0
    payload_ore_fraction: float = 0.30
    seat_used: float = 0.0
    fuel: float = 100.0
    refuel_threshold: float = 30.0
    operator: int = -1
    trip_start: float = 0.0
    dump_dur: float = 0.0
    down_start: float = math.inf
    down_end: float = math.inf

    # Truck configuration & attributes
    truck_type: str = "AD30"
    ore_payload_cap: float = 26.1
    waste_payload_cap: float = 24.6
    speeds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    payload_type: str = "ORE"
    current_location: str = "SURFACE_PARKING"
    target_bay_id: Optional[str] = None
    fuel_level_pct: float = 100.0
    fuel_burn_rate_pct_per_sec: float = 0.005
    next_failure_time: float = 1e9

    def __post_init__(self):
        if self.timer is None:
            self.timer = drs.Timer(f"tr_tmr_{self.truck_id}", 0.0, rate=0.0)

    @property
    def state(self) -> TruckPhase:
        return self.phase

    @state.setter
    def state(self, value: TruckPhase):
        self.phase = value

    def get_speed_mps(self, segment_type: str) -> float:
        """Returns speed in meters/second for current load state."""
        load_key = "loaded" if self.current_payload > 0 else "empty"
        kph = self.speeds.get(segment_type, {}).get(load_key, 10.0)
        return kph / 3.6


@dataclass
class SurfaceDumpStation:
    """Surface crusher tip / stockpile dump facility with finite truck bays."""

    name: str = "SURFACE_CRUSHER_HOPPER"
    capacity: int = 2
    in_use: int = 0
    queue: List[Truck] = field(default_factory=list)
    _active_ore1_rate: float = 0.0
    _active_ore2_rate: float = 0.0


@dataclass
class LHD:
    """Load-Haul-Dump (LHD) underground mucking loader entity."""

    lhd_id: str
    level_index: int
    bucket_ore_cap: float = 14.0
    bucket_waste_cap: float = 14.0
    load_spot_min: float = 0.50
    load_min: float = 0.50
    dump_min: float = 0.30
    tram_dist_m: float = 30.0
    speed_loaded_kph: float = 9.0
    speed_empty_kph: float = 12.0

    def get_bucket_cycle_sec(self) -> float:
        """Calculates duration in seconds of 1 LHD digging & tramming bucket pass."""
        speed_loaded_mpm = (
            (self.speed_loaded_kph * 1000.0 / 60.0)
            if self.speed_loaded_kph > 0
            else 1.0
        )
        speed_empty_mpm = (
            (self.speed_empty_kph * 1000.0 / 60.0) if self.speed_empty_kph > 0 else 1.0
        )
        t_tram_loaded_min = self.tram_dist_m / speed_loaded_mpm
        t_tram_empty_min = self.tram_dist_m / speed_empty_mpm
        total_min = (
            self.load_spot_min
            + self.load_min
            + t_tram_loaded_min
            + self.dump_min
            + t_tram_empty_min
        )
        return total_min * 60.0


class ContinuousFleetLogistics(drs.Module):
    """Continuous fleet routing and flow split component."""

    def __init__(self, num_stockpiles: int = 2):
        super().__init__()
        self.num_stockpiles = num_stockpiles
        self.stockpile2_routing_fraction = drs.Variable(
            "stockpile2_routing_fraction", 0.0
        )
        self.routing_fractions = []
        for i in range(num_stockpiles):
            var = drs.Variable(f"stockpile_{i+1}_routing_fraction", 0.0)
            self.routing_fractions.append(var)
            setattr(self, f"stockpile_{i+1}_routing_fraction", var)

    def route(
        self,
        rate: float = 0.0,
        ore_grade: float = 0.0,
        *,
        sources: Sequence = (),
    ) -> Tuple[float, float]:
        """Split mined material into the two stockpile inflows."""
        ore1 = ore2 = total = 0.0
        if sources:
            for src in sources:
                r = src.actual_rate
                g = src.current_ore_grade
                ore2 += r * g
                ore1 += r * (1.0 - g)
                total += r
        else:
            ore2 = rate * ore_grade
            ore1 = rate - ore2
            total = rate
        if total > 1e-6:
            self.stockpile2_routing_fraction.value = ore2 / total
            if len(self.routing_fractions) >= 2:
                self.routing_fractions[0].value = ore1 / total
                self.routing_fractions[1].value = ore2 / total
        return ore1, ore2

    def route_multi(
        self,
        sources: Sequence = (),
        split_fn: Optional[Callable[[float, float], Sequence[float]]] = None,
    ) -> List[float]:
        """Split mined material from arbitrary sources into N stockpile inflows."""
        if not sources:
            return [0.0] * self.num_stockpiles

        total_rate = sum(src.actual_rate for src in sources)
        avg_grade = (
            sum(src.actual_rate * src.current_ore_grade for src in sources) / total_rate
            if total_rate > 1e-6
            else 0.0
        )

        if split_fn is not None:
            inflows = list(split_fn(total_rate, avg_grade))
        else:
            ore1, ore2 = self.route(sources=sources)
            inflows = [ore1, ore2] + [0.0] * max(0, self.num_stockpiles - 2)

        if total_rate > 1e-6:
            for i, inflow in enumerate(inflows):
                if i < len(self.routing_fractions):
                    self.routing_fractions[i].value = inflow / total_rate
            if len(inflows) >= 2:
                self.stockpile2_routing_fraction.value = inflows[1] / total_rate

        return inflows

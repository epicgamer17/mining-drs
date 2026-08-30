"""Discrete-event haulage entities, operators, and dump stations."""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional
import drs


class TruckPhase(Enum):
    """Discrete execution phases of a haul truck in the DES state machine."""

    IDLE = "idle"
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


OPERATING_PHASES = {
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

SEAT_PHASES = OPERATING_PHASES | {TruckPhase.REFUELING}

DUE_PHASES = {
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
class DESTruck:
    """Discrete-event haul truck entity tracked by the event-stepping engine."""

    truck_id: str
    timer: drs.Timer
    phase: TruckPhase = TruckPhase.IDLE
    target_face_id: int = 1
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


# Alias for compatibility
Truck = DESTruck


@dataclass
class SurfaceDumpStation:
    """Surface crusher tip / stockpile dump facility with finite truck bays."""

    name: str = "SURFACE_CRUSHER_HOPPER"
    capacity: int = 2
    in_use: int = 0
    queue: List[DESTruck] = field(default_factory=list)
    _active_ore1_rate: float = 0.0
    _active_ore2_rate: float = 0.0

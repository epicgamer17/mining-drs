from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple
import math

import drs


class TruckState(Enum):
    PARKED = auto()
    TRAVEL_EMPTY = auto()
    WAITING_LOAD = auto()
    LOADING = auto()
    TRAVEL_LOADED = auto()
    WAITING_DUMP = auto()
    DUMPING = auto()
    REFUELING = auto()
    MAINTENANCE = auto()


@dataclass
class Truck:
    truck_id: str
    truck_type: str = "CAT AD30"
    ore_payload_cap: float = 26.1   # tonnes
    waste_payload_cap: float = 24.6 # tonnes

    # State tracking
    state: TruckState = TruckState.PARKED
    current_location: str = "SURFACE_PARKING"
    current_payload: float = 0.0
    payload_type: str = "ORE"       # "ORE" or "WASTE"
    target_level: Optional[int] = None
    target_bay_id: Optional[str] = None

    # Fuel & Mechanical State
    fuel_level_pct: float = 100.0   # 0.0 - 100.0%
    fuel_burn_rate_pct_per_sec: float = 0.005 # ~20h tank
    next_failure_time: float = 1e9  # Calculated from MTBF

    # Speed Specs (kph) across Mine Corridors
    speeds: dict = field(default_factory=lambda: {
        "surface": {"empty": 17.4, "loaded": 13.4},
        "decline": {"empty": 15.1, "loaded": 11.2},
        "ramp":    {"empty": 12.9, "loaded": 9.2},
        "level":   {"empty": 7.6,  "loaded": 6.6}
    })

    def get_speed_mps(self, segment_type: str) -> float:
        """Returns speed in meters/second for current load state."""
        load_key = "loaded" if self.current_payload > 0 else "empty"
        kph = self.speeds.get(segment_type, {}).get(load_key, 10.0)
        return kph / 3.6


@dataclass
class LHD:
    lhd_id: str
    level_index: int
    bucket_ore_cap: float = 14.0   # tonnes
    bucket_waste_cap: float = 12.5 # tonnes

    load_spot_min: float = 0.46
    load_min: float = 0.88
    dump_min: float = 0.73
    tram_dist_m: float = 35.0
    speed_loaded_kph: float = 5.89
    speed_empty_kph: float = 6.78

    def get_bucket_cycle_sec(self) -> float:
        """Calculates duration in seconds of 1 LHD digging & tramming bucket pass."""
        t_tram_loaded_min = (self.tram_dist_m / (self.speed_loaded_kph * 1000.0 / 60.0))
        t_tram_empty_min = (self.tram_dist_m / (self.speed_empty_kph * 1000.0 / 60.0))
        total_min = (self.load_spot_min + self.load_min + t_tram_loaded_min + self.dump_min + t_tram_empty_min)
        return total_min * 60.0


class ContinuousFleetLogistics(drs.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.stockpile2_routing_fraction = drs.Variable(
            "stockpile2_routing_fraction", 0.0
        )

    def route(self, rate=None, ore_grade=None, *, sources=None) -> Tuple[float, float]:
        """Split mined material into the two stockpile inflows.

        Either pass a single source's ``(rate, ore_grade)`` or pass
        ``sources=`` a sequence of mine faces to aggregate. Returns the
        ``(ore1_in, ore2_in)`` flow rates arriving at the Ore1 / Ore2
        stockpiles and stamps ``stockpile2_routing_fraction`` for telemetry.
        """
        ore1 = ore2 = total = 0.0
        if sources is not None:
            for src in sources:
                rate = src.actual_rate
                grade = src.current_ore_grade
                ore2 += rate * grade
                ore1 += rate * (1.0 - grade)
                total += rate
        else:
            ore2 = rate * ore_grade
            ore1 = rate - ore2
            total = rate
        if total > 1e-6:
            self.stockpile2_routing_fraction.value = ore2 / total
        return ore1, ore2


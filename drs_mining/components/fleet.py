from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple, List, Dict, Union, Sequence, Callable
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


def create_truck_fleet(
    count: int,
    prefix: str = "T",
    truck_type: str = "CAT AD30",
    ore_payload_cap: float = 26.1,
    waste_payload_cap: float = 24.6,
    speeds: Optional[dict] = None,
    **kwargs,
) -> List[Truck]:
    """Creates a homogeneous or custom fleet of N trucks."""
    trucks = []
    for i in range(1, count + 1):
        t_id = f"{prefix}{i:02d}"
        truck_args = {
            "truck_id": t_id,
            "truck_type": truck_type,
            "ore_payload_cap": ore_payload_cap,
            "waste_payload_cap": waste_payload_cap,
            **kwargs,
        }
        if speeds is not None:
            truck_args["speeds"] = speeds
        trucks.append(Truck(**truck_args))
    return trucks


def create_lhd_fleet(
    levels: Union[int, Sequence[int]],
    count_per_level: int = 1,
    prefix: str = "LHD",
    bucket_ore_cap: float = 14.0,
    bucket_waste_cap: float = 12.5,
    **kwargs,
) -> List[LHD]:
    """Creates a fleet of LHD loaders assigned to specific mine levels."""
    if isinstance(levels, int):
        level_list = list(range(1, levels + 1))
    else:
        level_list = list(levels)

    lhds = []
    for lvl in level_list:
        for idx in range(1, count_per_level + 1):
            suffix = f"_{idx}" if count_per_level > 1 else ""
            lhd_id = f"{prefix}_L{lvl}{suffix}"
            lhds.append(
                LHD(
                    lhd_id=lhd_id,
                    level_index=lvl,
                    bucket_ore_cap=bucket_ore_cap,
                    bucket_waste_cap=bucket_waste_cap,
                    **kwargs,
                )
            )
    return lhds


class ContinuousFleetLogistics(drs.Module):
    def __init__(self, num_stockpiles: int = 2, *args, **kwargs):
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
            if len(self.routing_fractions) >= 2:
                self.routing_fractions[0].value = ore1 / total
                self.routing_fractions[1].value = ore2 / total
        return ore1, ore2

    def route_multi(
        self,
        sources=None,
        split_fn: Optional[Callable[[float, float], Sequence[float]]] = None,
    ) -> List[float]:
        """Split mined material from arbitrary sources into N stockpile inflows.

        ``split_fn(rate, grade)`` returns a sequence of N flow rates for each source.
        If ``split_fn`` is None, defaults to 2-stockpile binary split (ore1=(1-grade)*rate, ore2=grade*rate).
        """
        inflows = [0.0] * self.num_stockpiles
        total = 0.0
        if sources is not None:
            for src in sources:
                rate = src.actual_rate
                grade = src.current_ore_grade
                total += rate
                if split_fn is not None:
                    src_splits = split_fn(rate, grade)
                else:
                    src_splits = [rate * (1.0 - grade), rate * grade]
                for idx, flow in enumerate(src_splits[: self.num_stockpiles]):
                    inflows[idx] += flow
        if total > 1e-6:
            for idx, flow in enumerate(inflows):
                if idx < len(self.routing_fractions):
                    self.routing_fractions[idx].value = flow / total
            if self.num_stockpiles >= 2:
                self.stockpile2_routing_fraction.value = inflows[1] / total
        return inflows



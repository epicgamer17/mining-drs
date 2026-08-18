from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, List, Sequence, Mapping, Callable
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
    truck_type: str
    ore_payload_cap: float
    waste_payload_cap: float
    speeds: dict

    # State tracking
    state: TruckState = TruckState.PARKED
    current_location: str = "SURFACE_PARKING"
    current_payload: float = 0.0
    payload_type: str = "ORE"       # "ORE" or "WASTE"
    target_level: Optional[int] = None
    target_bay_id: Optional[str] = None

    # Fuel & Mechanical State
    fuel_level_pct: float = 100.0   # 0.0 - 100.0%
    fuel_burn_rate_pct_per_sec: float = 0.005 # per-second burn rate
    next_failure_time: float = 1e9  # Calculated from MTBF

    def get_speed_mps(self, segment_type: str) -> float:
        """Returns speed in meters/second for current load state."""
        load_key = "loaded" if self.current_payload > 0 else "empty"
        kph = self.speeds.get(segment_type, {}).get(load_key, 10.0)
        return kph / 3.6


@dataclass
class LHD:
    lhd_id: str
    level_index: int
    bucket_ore_cap: float
    bucket_waste_cap: float
    load_spot_min: float
    load_min: float
    dump_min: float
    tram_dist_m: float
    speed_loaded_kph: float
    speed_empty_kph: float

    def get_bucket_cycle_sec(self) -> float:
        """Calculates duration in seconds of 1 LHD digging & tramming bucket pass."""
        speed_loaded_mpm = (self.speed_loaded_kph * 1000.0 / 60.0) if self.speed_loaded_kph > 0 else 1.0
        speed_empty_mpm = (self.speed_empty_kph * 1000.0 / 60.0) if self.speed_empty_kph > 0 else 1.0
        t_tram_loaded_min = self.tram_dist_m / speed_loaded_mpm
        t_tram_empty_min = self.tram_dist_m / speed_empty_mpm
        total_min = (self.load_spot_min + self.load_min + t_tram_loaded_min + self.dump_min + t_tram_empty_min)
        return total_min * 60.0


def create_truck_fleet(
    count: int,
    truck_type: str,
    ore_payload_cap: float,
    waste_payload_cap: float,
    speeds: Mapping,
    prefix: str = "T",
    fuel_burn_rate_pct_per_sec: float = 0.005,
    **kwargs,
) -> List[Truck]:
    """Creates a fleet of Truck dataclasses with given configuration."""
    fleet = []
    for i in range(1, count + 1):
        truck_id = f"{prefix}{i:02d}" if prefix else str(i)
        t = Truck(
            truck_id=truck_id,
            truck_type=truck_type,
            ore_payload_cap=ore_payload_cap,
            waste_payload_cap=waste_payload_cap,
            speeds=dict(speeds),
            fuel_burn_rate_pct_per_sec=fuel_burn_rate_pct_per_sec,
            **kwargs,
        )
        fleet.append(t)
    return fleet


def create_lhd_fleet(
    levels: Sequence[int],
    bucket_ore_cap: float,
    bucket_waste_cap: float,
    load_spot_min: float,
    load_min: float,
    dump_min: float,
    tram_dist_m: float,
    speed_loaded_kph: float,
    speed_empty_kph: float,
    count_per_level: int = 1,
    prefix: str = "LHD",
    **kwargs,
) -> List[LHD]:
    """Creates a fleet of LHD loaders assigned to specific mine levels."""
    level_list = list(levels) if isinstance(levels, (list, tuple, range)) else list(range(1, int(levels) + 1))
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
                    load_spot_min=load_spot_min,
                    load_min=load_min,
                    dump_min=dump_min,
                    tram_dist_m=tram_dist_m,
                    speed_loaded_kph=speed_loaded_kph,
                    speed_empty_kph=speed_empty_kph,
                    **kwargs,
                )
            )
    return lhds


class ContinuousFleetLogistics(drs.Module):
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

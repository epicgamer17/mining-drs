from typing import Optional, Union, Dict
import json
import random
import drs
from .fleet import Truck

# TODO: this belongsi n the examples
DEFAULT_SPEEDS = {
    "surface": {"empty": 17.4, "loaded": 13.4},
    "decline": {"empty": 15.1, "loaded": 11.2},
    "ramp": {"empty": 12.9, "loaded": 9.2},
    "level": {"empty": 7.6, "loaded": 6.6},
}


class RoadSegment(drs.Module):
    """Represents a discrete single-lane haulage road corridor with timer decay."""

    def __init__(
        self,
        segment_id: str,
        length_m: float,
        segment_type: str,
    ):
        super().__init__()
        self.segment_id = segment_id
        self.length_m = length_m
        self.segment_type = segment_type  # "decline", "ramp", "surface", "level"

        self.time_until_free = drs.Timer(
            f"road_{segment_id}_t_free", initial_value=0.0, rate=-1.0
        )
        self.time_until_free.lower_threshold = 0.0
        self.decay_rate = drs.Variable(f"road_{segment_id}_decay", initial_value=-1.0)

        self.occupying_truck: Optional[Truck] = None

    def is_available(self) -> bool:
        """Returns True if road segment is free for vehicle entry."""
        return self.time_until_free.value <= 0.0

    def occupy_segment(self, truck: Truck) -> float:
        """Calculates travel duration, occupies segment, and sets continuous availability timer."""
        speed_mps = truck.get_speed_mps(self.segment_type)
        travel_time_s = self.length_m / speed_mps
        self.time_until_free.value = travel_time_s
        self.time_until_free.rate = (-1.0, 0.0, float("inf"))
        self.occupying_truck = truck
        return travel_time_s

    def update_continuous_step(self, dt: float):
        """Integrates continuous timer decay using DRS Timer stepping."""
        if self.time_until_free.value > 0.0:
            self.time_until_free.step(dt)
            if self.time_until_free.value <= 0.0:
                self.occupying_truck = None
                self.time_until_free.rate = 0.0


class MineTopology(drs.Module):
    """Physical underground haulage network geometry, travel dynamics, passing bays,
    and single-lane decline/ramp traffic congestion physics.
    """

    def __init__(
        self,
        decline_m: float = 2100.0,
        level_spacing_m: float = 300.0,
        level_drift_m: float = 60.0,
        surface_m: float = 300.0,
        level_depths: Optional[Dict[int, float]] = None,
        speeds: Optional[Dict[str, Dict[str, float]]] = None,
        base_pass_bay_delay_sec: float = 13.0,
        per_truck_pass_bay_delay_sec: float = 1.0,
        traffic_variation_tol: float = 0.20,
    ):
        super().__init__()
        self.decline_m = decline_m
        self.level_spacing_m = level_spacing_m
        self.level_drift_m = level_drift_m
        self.surface_m = surface_m
        self.level_depths = level_depths or {3: 900.0, 6: 1800.0}
        self.speeds = speeds or DEFAULT_SPEEDS
        self.base_pass_bay_delay_sec = base_pass_bay_delay_sec
        self.per_truck_pass_bay_delay_sec = per_truck_pass_bay_delay_sec
        self.traffic_variation_tol = traffic_variation_tol

    def calculate_traffic_delay_sec(
        self,
        active_truck_count: int,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Calculates passing bay dwell and meeting delay on single-lane ramp corridors."""
        if active_truck_count <= 1:
            return 0.0

        other_trucks = active_truck_count - 1
        mean_delay = (
            self.base_pass_bay_delay_sec
            + self.per_truck_pass_bay_delay_sec * other_trucks
        )
        if rng is not None and self.traffic_variation_tol > 0:
            tol = self.traffic_variation_tol
            return rng.triangular(
                mean_delay * (1.0 - tol), mean_delay * (1.0 + tol), mean_delay
            )
        return mean_delay

    def calculate_travel_time_sec(
        self,
        level: int,
        loaded: bool,
        active_truck_count: int = 1,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Calculates total physical travel time between surface and target underground level."""
        state = "loaded" if loaded else "empty"
        v_surf = (self.speeds["surface"][state] * 1000.0) / 3600.0
        v_decl = (self.speeds["decline"][state] * 1000.0) / 3600.0
        v_ramp = (self.speeds["ramp"][state] * 1000.0) / 3600.0
        v_level = (self.speeds["level"][state] * 1000.0) / 3600.0

        t_surf = self.surface_m / v_surf
        t_decl = self.decline_m / v_decl

        ramp_dist = self.level_depths.get(
            level, max(0.0, float(level - 1)) * self.level_spacing_m
        )
        t_ramp = ramp_dist / v_ramp
        t_level = self.level_drift_m / v_level

        base_travel_time = t_surf + t_decl + t_ramp + t_level
        traffic_delay = self.calculate_traffic_delay_sec(active_truck_count, rng)
        return base_travel_time + traffic_delay

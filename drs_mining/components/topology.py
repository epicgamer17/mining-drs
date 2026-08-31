from typing import Optional, Union, Dict, List, Any
import json
import random
import drs
from .fleet import Truck

DEFAULT_SPEEDS = {
    "surface": {"empty": 17.4, "loaded": 13.4},
    "decline": {"empty": 15.1, "loaded": 11.2},
    "ramp": {"empty": 12.9, "loaded": 9.2},
    "level": {"empty": 7.6, "loaded": 6.6},
}


class PassingBay(drs.Module):
    """Underground turnout or passing bay where haul trucks can pull over and yield right-of-way."""

    def __init__(self, bay_id: str, capacity: int = 10):
        super().__init__()
        self.bay_id = bay_id
        self.capacity = capacity
        self.waiting_trucks: List[Truck] = []

    def queue_truck(self, truck: Truck) -> None:
        """Adds a yielding truck to the passing bay queue."""
        if truck not in self.waiting_trucks:
            self.waiting_trucks.append(truck)
            truck.in_passing_bay = self

    def remove_truck(self, truck: Truck) -> None:
        """Removes a truck from the passing bay queue upon dispatch into a cleared segment."""
        if truck in self.waiting_trucks:
            self.waiting_trucks.remove(truck)
        if truck.in_passing_bay == self:
            truck.in_passing_bay = None

    def has_waiting_loaded_truck(self) -> bool:
        """Checks if any loaded truck (heading UP with right-of-way) is waiting in this bay."""
        return any(
            getattr(t, "phase", None) and t.phase.value == "loaded"
            for t in self.waiting_trucks
        )

    def get_waiting_loaded_truck(self) -> Optional[Truck]:
        """Returns the first loaded truck waiting with right-of-way."""
        for t in self.waiting_trucks:
            if getattr(t, "phase", None) and t.phase.value == "loaded":
                return t
        return None

    def get_waiting_empty_truck(self) -> Optional[Truck]:
        """Returns the first empty truck waiting."""
        for t in self.waiting_trucks:
            if getattr(t, "phase", None) and t.phase.value == "empty":
                return t
        return None


class RoadSegment(drs.Module):
    """Discrete single-lane haulage road corridor block with directional single-capacity lock."""

    def __init__(
        self,
        segment_id: str,
        length_m: float,
        segment_type: str,
        speeds: Optional[Dict[str, Dict[str, float]]] = None,
        is_two_way: bool = False,
    ):
        super().__init__()
        self.segment_id = segment_id
        self.length_m = length_m
        self.segment_type = segment_type  # "surface", "decline", "ramp", "level"
        self.speeds = speeds or DEFAULT_SPEEDS
        self.is_two_way = is_two_way

        self.occupant: Optional[Truck] = None
        self.direction: Optional[str] = None  # "DOWN" (empty) or "UP" (loaded)

        self.upstream_bay: Optional[PassingBay] = None
        self.downstream_bay: Optional[PassingBay] = None

        # Continuous availability timer for DRS integrations
        self.time_until_free = drs.Timer(
            f"road_{segment_id}_t_free", initial_value=0.0, rate=0.0
        )
        self.time_until_free.lower_threshold = 0.0

    def is_free(self) -> bool:
        """Returns True if road segment has no current occupant."""
        if self.is_two_way:
            return True
        return self.occupant is None

    def can_enter(self, truck: Truck, direction: str) -> bool:
        """Evaluates entry permission using Shelswell (2017) right-of-way priority.

        Rules:
          1. Two-way routes (surface roadways) are unconstrained.
          2. If segment is currently occupied by any vehicle, entry is blocked.
          3. If empty truck (heading DOWN), check if an oncoming loaded truck (heading UP)
             is waiting at the downstream bay to enter this segment -> loaded truck has right-of-way.
        """
        if self.is_two_way:
            return True

        if self.occupant is not None:
            return False

        # Downhill empty trucks must yield if an uphill loaded truck is waiting at the downstream bay
        if direction == "DOWN" and self.downstream_bay is not None:
            if self.downstream_bay.has_waiting_loaded_truck():
                return False

        return True

    def occupy(self, truck: Truck, direction: str) -> float:
        """Claims the road segment for the active truck and sets directional occupancy."""
        if not self.is_two_way:
            self.occupant = truck
            self.direction = direction

        trav_time = self.calculate_traversal_time_sec(
            loaded=(direction == "UP"), truck=truck
        )
        self.time_until_free.value = trav_time
        self.time_until_free.rate = (-1.0, 0.0, float("inf"))
        return trav_time

    def release(self, truck: Truck) -> None:
        """Releases the segment lock when a truck exits into a passing bay or destination."""
        if not self.is_two_way and self.occupant == truck:
            self.occupant = None
            self.direction = None
            self.time_until_free.value = 0.0
            self.time_until_free.rate = 0.0

    def calculate_traversal_time_sec(
        self,
        loaded: bool,
        truck: Optional[Truck] = None,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Calculates physical travel time across this specific road segment based on load state."""
        if truck is not None and getattr(truck, "speeds", None):
            speed_mps = truck.get_speed_mps(self.segment_type)
            if speed_mps > 0:
                return self.length_m / speed_mps

        load_key = "loaded" if loaded else "empty"
        speed_kph = self.speeds.get(self.segment_type, {}).get(load_key, 10.0)
        speed_mps = (speed_kph * 1000.0) / 3600.0
        trav_time = self.length_m / speed_mps
        return trav_time

    @property
    def occupying_truck(self) -> Optional[Truck]:
        return self.occupant

    @occupying_truck.setter
    def occupying_truck(self, value: Optional[Truck]):
        self.occupant = value

    def is_available(self) -> bool:
        """Returns True if road segment is free for vehicle entry."""
        return self.is_free()

    def occupy_segment(self, truck: Truck) -> float:
        """Legacy helper: claims segment and sets continuous timer."""
        direction = "UP" if getattr(truck, "current_payload", 0) > 0 else "DOWN"
        return self.occupy(truck, direction)

    def update_continuous_step(self, dt: float) -> None:
        """Integrates continuous timer decay using DRS Timer stepping."""
        if self.time_until_free.value > 0.0:
            self.time_until_free.step(dt)
            if self.time_until_free.value <= 0.0:
                self.occupant = None
                self.time_until_free.rate = 0.0


class MineTopology(drs.Module):
    """Physical underground haulage network geometry, discrete road segments,
    passing bays, and single-lane right-of-way traffic simulation matching Shelswell (2017).
    """

    def __init__(
        self,
        decline_m: float = 2100.0,
        level_spacing_m: float = 300.0,
        level_drift_m: float = 60.0,
        surface_m: float = 300.0,
        segment_len_m: float = 300.0,
        area2_decline_m: float = 4000.0,
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
        self.segment_len_m = segment_len_m
        self.area2_decline_m = area2_decline_m
        self.level_depths = level_depths or {1: 300.0, 2: 600.0, 3: 900.0, 4: 1200.0, 5: 1500.0, 6: 1800.0, 7: 2100.0}
        self.speeds = speeds or DEFAULT_SPEEDS
        self.base_pass_bay_delay_sec = base_pass_bay_delay_sec
        self.per_truck_pass_bay_delay_sec = per_truck_pass_bay_delay_sec
        self.traffic_variation_tol = traffic_variation_tol

        self.segments: Dict[str, RoadSegment] = {}
        self.passing_bays: Dict[str, PassingBay] = {}
        self._build_network()

    def _build_network(self) -> None:
        """Constructs the full discrete road segment and passing bay network."""
        # 1. Surface Roadway (Two-way unconstrained)
        self.bay_portal = PassingBay("bay_portal")
        self.passing_bays["bay_portal"] = self.bay_portal

        self.seg_surface = RoadSegment(
            "seg_surface", self.surface_m, "surface", self.speeds, is_two_way=True
        )
        self.segments["seg_surface"] = self.seg_surface

        # 2. Main Access Decline (2,100 m broken into ~300 m single-lane segments with passing bays)
        n_decl = max(1, int(round(self.decline_m / self.segment_len_m)))
        decl_sub_len = self.decline_m / n_decl

        prev_bay = self.bay_portal
        self.decline_segments: List[RoadSegment] = []
        for i in range(n_decl):
            seg_id = f"decl_seg_{i+1}"
            next_bay_id = f"bay_decl_{i+1}"
            next_bay = PassingBay(next_bay_id)
            self.passing_bays[next_bay_id] = next_bay

            seg = RoadSegment(seg_id, decl_sub_len, "decline", self.speeds)
            seg.upstream_bay = prev_bay
            seg.downstream_bay = next_bay
            self.segments[seg_id] = seg
            self.decline_segments.append(seg)
            prev_bay = next_bay

        # Bottom of decline connects to Level 1 junction
        self.bay_bottom_decline = prev_bay
        self.level_bays: Dict[int, PassingBay] = {1: self.bay_bottom_decline}

        # 3. Spiral Ramp between Levels (Levels 1 to 7 at 300 m spacing)
        curr_bay = self.bay_bottom_decline
        for lvl in range(2, 8):
            seg_id = f"ramp_seg_L{lvl-1}_L{lvl}"
            lvl_bay_id = f"bay_L{lvl}"
            lvl_bay = PassingBay(lvl_bay_id)
            self.passing_bays[lvl_bay_id] = lvl_bay
            self.level_bays[lvl] = lvl_bay

            seg = RoadSegment(seg_id, self.level_spacing_m, "ramp", self.speeds)
            seg.upstream_bay = curr_bay
            seg.downstream_bay = lvl_bay
            self.segments[seg_id] = seg
            curr_bay = lvl_bay

        # 4. Level Access Drifts (Level entry to muck loadout)
        self.level_drift_segments: Dict[int, RoadSegment] = {}
        for lvl in range(1, 8):
            seg_id = f"drift_seg_L{lvl}"
            drift_bay_id = f"bay_loadout_L{lvl}"
            drift_bay = PassingBay(drift_bay_id)
            self.passing_bays[drift_bay_id] = drift_bay

            seg = RoadSegment(seg_id, self.level_drift_m, "level", self.speeds)
            seg.upstream_bay = self.level_bays[lvl]
            seg.downstream_bay = drift_bay
            self.segments[seg_id] = seg
            self.level_drift_segments[lvl] = seg

        # 5. Area 2 Capital Development Decline (from Level 6 to deep face)
        n_area2_decl = max(1, int(round(self.area2_decline_m / self.segment_len_m)))
        area2_sub_len = self.area2_decline_m / n_area2_decl
        prev_bay = self.level_bays.get(6, curr_bay)
        self.area2_decline_segments: List[RoadSegment] = []
        for i in range(n_area2_decl):
            seg_id = f"area2_decl_seg_{i+1}"
            next_bay_id = f"bay_area2_decl_{i+1}"
            next_bay = PassingBay(next_bay_id)
            self.passing_bays[next_bay_id] = next_bay

            seg = RoadSegment(seg_id, area2_sub_len, "decline", self.speeds)
            seg.upstream_bay = prev_bay
            seg.downstream_bay = next_bay
            self.segments[seg_id] = seg
            self.area2_decline_segments.append(seg)
            prev_bay = next_bay

        self.bay_area2_face = prev_bay
        self.passing_bays["bay_area2_face"] = self.bay_area2_face

    def reset_network(self) -> None:
        """Resets all segments and passing bay queues to clean initial state."""
        for seg in self.segments.values():
            seg.occupant = None
            seg.direction = None
            seg.time_until_free.value = 0.0
            seg.time_until_free.rate = 0.0
        for bay in self.passing_bays.values():
            bay.waiting_trucks.clear()

    def get_route(
        self, target_level: int, direction: str, is_capital_dev: bool = False
    ) -> List[RoadSegment]:
        """Returns the ordered list of discrete RoadSegments for travel.

        Args:
            target_level: Destination level underground (e.g. 1 to 7).
            direction: "DOWN" (empty descent) or "UP" (loaded haul to surface).
            is_capital_dev: True if heading to the Area 2 capital decline development face.
        """
        route: List[RoadSegment] = []

        # 1. Surface Roadway
        route.append(self.seg_surface)

        # 2. Main Access Decline
        route.extend(self.decline_segments)

        # 3. Spiral Ramp down to target level
        lvl_target = max(1, target_level)
        if lvl_target > 1:
            for lvl in range(2, lvl_target + 1):
                seg_id = f"ramp_seg_L{lvl-1}_L{lvl}"
                if seg_id in self.segments:
                    route.append(self.segments[seg_id])

        # 4. Capital Development Decline vs Level Drift
        if is_capital_dev:
            route.extend(self.area2_decline_segments)
        else:
            if lvl_target in self.level_drift_segments:
                route.append(self.level_drift_segments[lvl_target])

        if direction == "UP":
            return list(reversed(route))
        return route

    def calculate_traffic_delay_sec(
        self,
        active_truck_count: int,
        rng: Optional[random.Random] = None,
    ) -> float:
        """Legacy helper for analytical calculations."""
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
        """Legacy analytical baseline calculation across the route."""
        direction = "UP" if loaded else "DOWN"
        route = self.get_route(level, direction)
        return sum(s.calculate_traversal_time_sec(loaded, rng) for s in route)

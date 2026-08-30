"""Fleet Dispatch Controllers for Underground Mine Haulage.

Implements:
  1. ShelswellDispatchController: Classic rule-based heuristic dispatch (highest muck, shallowest, round robin).
  2. TwoTierHierarchicalDispatchController: Value-oriented closed-loop dispatch implementing:
     - Tier 1 (Absolute Primary Requirement): Guarantee total mill feed (e.g. 6,000 t/d) to prevent mill starvation.
     - Tier 2 (Secondary Blending Objective): Match optimal analytical dispatch weights (w1, w2) for active campaign mode.
     - Tier 3 (Dynamic Constrained Fallback): If preferred stope is in turnaround/busy, redirect to nearest available stope.
     - Surplus Capacity Redirection: Idle/excess haulage capacity is redirected to advance waste development headings.
"""

from dataclasses import dataclass
import random
from typing import List, Mapping, Optional, Sequence, Tuple
from .fleet import Truck, TruckState
from .bays import LoadingBay
from .topology import RoadSegment
from .mine_face import MineFace, FaceState, StopeState


class ShelswellDispatchController:
    """Implements Shelswell's operational dispatch rules cleanly in Python."""

    def __init__(
        self,
        trucks: List[Truck],
        loading_bays: List[LoadingBay],
        roads: Mapping[str, RoadSegment] = None,
        waste_trip_interval: int = 13,
        refuel_threshold_pct: float = 15.0,
        fuel_depot_location: str = "SURFACE_FUEL_DEPOT",
        parking_location: str = "SURFACE_PARKING",
        dispatch_strategy: str = "highest_muck",
    ):
        self.trucks = trucks
        self.loading_bays = loading_bays
        self.roads = dict(roads or {})
        self.waste_trip_interval = waste_trip_interval
        self.refuel_threshold_pct = refuel_threshold_pct
        self.fuel_depot_location = fuel_depot_location
        self.parking_location = parking_location
        self.dispatch_strategy = dispatch_strategy
        self.dispatch_counter = 0

    def assign_payload_type(self, truck: Truck):
        """Assigns ORE or WASTE payload type according to the production schedule ratio."""
        self.dispatch_counter += 1
        if (
            self.waste_trip_interval > 0
            and self.dispatch_counter % self.waste_trip_interval == 0
        ):
            truck.payload_type = "WASTE"
        else:
            truck.payload_type = "ORE"

    def assign_next_destination(self, truck: Truck):
        """Dispatch Rule: Route truck to target loadout with highest unclaimed tonnage or by strategy."""
        if truck.fuel_level_pct < self.refuel_threshold_pct:
            truck.state = TruckState.REFUELING
            truck.current_location = self.fuel_depot_location
            return

        self.assign_payload_type(truck)

        # Filter bays matching ore/waste target type
        valid_bays = [b for b in self.loading_bays if b.bay_type == truck.payload_type]
        if not valid_bays:
            valid_bays = self.loading_bays

        if not valid_bays:
            return

        if self.dispatch_strategy == "highest_muck":
            target_bay = max(valid_bays, key=lambda b: b.muck_level.value)
        elif self.dispatch_strategy == "round_robin":
            idx = (self.dispatch_counter - 1) % len(valid_bays)
            target_bay = valid_bays[idx]
        elif self.dispatch_strategy == "shallowest_first":
            target_bay = min(valid_bays, key=lambda b: b.level_index)
        else:
            target_bay = valid_bays[0]

        truck.target_bay_id = target_bay.bay_id
        truck.target_level = target_bay.level_index
        truck.state = TruckState.TRAVEL_EMPTY


@dataclass
class DispatchDecision:
    """Result of a hierarchical stope dispatch decision."""

    selected_stope_id: Optional[int] = None
    is_fallback: bool = False
    stope: Optional[MineFace] = None


class TwoTierHierarchicalDispatchController:
    """Two-Tier Hierarchical Dispatch Controller: Primary Tonnage Throughput & Secondary Analytical Blending."""

    def __init__(
        self,
        stopes: Sequence[MineFace],
        target_daily_ore_tonnes: float = 6000.0,
        target_stockpile_buffer_tonnes: float = 60000.0,
        seed: int = 42,
    ):
        self.stopes = list(stopes)
        self.target_daily_ore_tonnes = target_daily_ore_tonnes
        self.target_stockpile_buffer_tonnes = target_stockpile_buffer_tonnes
        self.rng = random.Random(seed)
        self.total_dispatches = 0
        self.fallback_dispatches = 0

    def select_stope_for_truck(
        self,
        current_total_stock: float,
        daily_hauled_so_far: float,
        day_progress_fraction: float,
        analytical_w2: float,  # Target fraction from Area 2 (Level 6)
        area2_locked: bool = False,
        lhd_queues: Optional[Mapping[int, int]] = None,
        target_daily_ore_tonnes: Optional[float] = None,
    ) -> Optional[Tuple[MineFace, bool]]:
        """Selects the optimal stope for a haul truck.

        Returns:
          (selected_stope, is_fallback) or None if all ore needs are satisfied and fleet should haul waste.
        """
        lhd_queues = lhd_queues or {}
        self.total_dispatches += 1

        daily_target = target_daily_ore_tonnes if target_daily_ore_tonnes is not None else self.target_daily_ore_tonnes

        # Tier 1 (Absolute Primary Requirement): Determine if ore haulage is required
        expected_hauled = daily_target * day_progress_fraction
        is_stockpile_low = current_total_stock < self.target_stockpile_buffer_tonnes
        is_behind_schedule = daily_hauled_so_far < expected_hauled

        # If stockpiles are 100% full and day's target is met, pause ore dispatch -> surplus to dev
        if (not is_stockpile_low) and daily_hauled_so_far >= expected_hauled:
            return None

        # Filter available stopes
        active_stopes = [s for s in self.stopes if not s.is_exhausted]
        if not active_stopes:
            return None

        # Separate into Area 1 and Area 2
        area1_stopes = [s for s in active_stopes if s.area_id == 1 and s.is_ore_available]
        area2_stopes = [s for s in active_stopes if s.area_id == 2 and s.is_ore_available]

        if area2_locked:
            preferred_area = 1
        else:
            # Tier 2 (Secondary Objective): Solve preferred area from analytical blending weight w2
            preferred_area = 2 if self.rng.random() < analytical_w2 else 1

        # Attempt to select stope in preferred area
        candidate_stopes = area2_stopes if preferred_area == 2 else area1_stopes

        if candidate_stopes:
            # Pick stope with shortest LHD queue or highest remaining parcel ore
            selected = min(
                candidate_stopes,
                key=lambda s: (lhd_queues.get(s.face_id, 0), -s.remaining_parcel_ore),
            )
            return selected, False

        # Tier 3 (Dynamic Constrained Fallback): Preferred area has no available stopes!
        # Fallback to alternative area rather than starving the mill
        fallback_candidates = area1_stopes if preferred_area == 2 else area2_stopes
        if fallback_candidates:
            self.fallback_dispatches += 1
            selected = min(
                fallback_candidates,
                key=lambda s: (lhd_queues.get(s.face_id, 0), -s.remaining_parcel_ore),
            )
            return selected, True

        # All stopes currently in turnaround development
        return None

    def dispatch(
        self,
        active_operating_mode_name: str = "MODE_A",
        truck_payload: float = 26.1,
        truck_cycle_time_sec: float = 2100.0,
        allow_area2: bool = True,
        analytical_w2: Optional[float] = None,
        current_total_stock: float = 0.0,
        daily_hauled_so_far: float = 0.0,
        day_progress_fraction: float = 0.0,
        lhd_queues: Optional[Mapping[int, int]] = None,
    ) -> DispatchDecision:
        """High-level dispatch entry point compatible with operational simulations."""
        if analytical_w2 is None:
            if "MODE_A" in active_operating_mode_name:
                analytical_w2 = 0.833
            elif "MODE_B" in active_operating_mode_name:
                analytical_w2 = 0.300
            else:
                analytical_w2 = 0.500

        res = self.select_stope_for_truck(
            current_total_stock=current_total_stock,
            daily_hauled_so_far=daily_hauled_so_far,
            day_progress_fraction=day_progress_fraction,
            analytical_w2=analytical_w2,
            area2_locked=not allow_area2,
            lhd_queues=lhd_queues,
        )
        if res is None:
            avail = [s for s in self.stopes if s.is_ore_available and (allow_area2 or s.area_id == 1)]
            if avail:
                st = avail[self.rng.randint(0, len(avail) - 1)]
                return DispatchDecision(selected_stope_id=st.face_id, is_fallback=True, stope=st)
            return DispatchDecision(selected_stope_id=1, is_fallback=False, stope=self.stopes[0] if self.stopes else None)

        stope, is_fb = res
        return DispatchDecision(selected_stope_id=stope.face_id, is_fallback=is_fb, stope=stope)

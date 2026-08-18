from typing import List, Mapping
from .fleet import Truck, TruckState
from .bays import LoadingBay
from .topology import RoadSegment


class ShelswellDispatchController:
    """Implements Shelswell's operational dispatch rules cleanly in Python."""

    def __init__(
        self,
        trucks: List[Truck],
        loading_bays: List[LoadingBay],
        roads: Mapping[str, RoadSegment],
        waste_trip_interval: int,
        refuel_threshold_pct: float,
        fuel_depot_location: str,
        parking_location: str,
        dispatch_strategy: str,
    ):
        self.trucks = trucks
        self.loading_bays = loading_bays
        self.roads = dict(roads)
        self.waste_trip_interval = waste_trip_interval
        self.refuel_threshold_pct = refuel_threshold_pct
        self.fuel_depot_location = fuel_depot_location
        self.parking_location = parking_location
        self.dispatch_strategy = dispatch_strategy
        self.dispatch_counter = 0

    def assign_payload_type(self, truck: Truck):
        """Assigns ORE or WASTE payload type according to the production schedule ratio."""
        self.dispatch_counter += 1
        if self.waste_trip_interval > 0 and self.dispatch_counter % self.waste_trip_interval == 0:
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

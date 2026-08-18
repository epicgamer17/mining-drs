from typing import List, Dict, Optional
from .fleet import Truck, TruckState
from .bays import DRSLoadingBay


class ShelswellDispatchController:
    """Implements Shelswell's operational dispatch rules cleanly in Python:
    
    1. Unclaimed Tonnes Rule: Dispatch trucks to the level loadout with highest remaining tonnage.
    2. Refueling & Breakdown Routing: Intercept trucks when fuel_level_pct < 15% to surface fuel depot.
    3. Payload Schedule Ratio: Distribute haulage target types matching ore to waste ratio (5.5 : 1).
    """

    def __init__(
        self,
        trucks: List[Truck],
        loading_bays: List[DRSLoadingBay],
        roads: Optional[Dict] = None,
        waste_trip_interval: int = 13,
        refuel_threshold_pct: float = 15.0,
        fuel_depot_location: str = "SURFACE_FUEL_DEPOT",
        parking_location: str = "SURFACE_PARKING",
        dispatch_strategy: str = "highest_muck",
    ):
        self.trucks = trucks
        self.loading_bays = loading_bays
        self.roads = roads or {}
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
            bays_with_muck = [b for b in valid_bays if b.muck_level.value > 0]
            if bays_with_muck:
                best_bay = max(bays_with_muck, key=lambda b: b.muck_level.value)
            else:
                best_bay = max(valid_bays, key=lambda b: b.muck_level.value)
        elif self.dispatch_strategy == "round_robin":
            best_bay = valid_bays[self.dispatch_counter % len(valid_bays)]
        else:
            best_bay = valid_bays[0]

        truck.target_bay_id = best_bay.bay_id
        truck.target_level = best_bay.level_index
        truck.state = TruckState.TRAVEL_EMPTY
        truck.current_location = self.parking_location


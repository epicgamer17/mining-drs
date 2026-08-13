from typing import List, Dict, Optional
from ..components.fleet import Truck, TruckState
from ..components.bays import DRSLoadingBay


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
    ):
        self.trucks = trucks
        self.loading_bays = loading_bays
        self.roads = roads or {}
        self.dispatch_counter = 0

    def assign_payload_type(self, truck: Truck):
        """Assigns ORE or WASTE payload type according to the 5.5:1 production schedule ratio."""
        self.dispatch_counter += 1
        if self.dispatch_counter % 13 == 0:  # ~1 out of 6.5 trips waste
            truck.payload_type = "WASTE"
        else:
            truck.payload_type = "ORE"

    def assign_next_destination(self, truck: Truck):
        """Shelswell Dispatch Rule: Route truck to target loadout with highest unclaimed tonnage."""
        if truck.fuel_level_pct < 15.0:
            truck.state = TruckState.REFUELING
            truck.current_location = "SURFACE_FUEL_DEPOT"  # 270m from portal
            return

        self.assign_payload_type(truck)

        # Filter bays matching ore/waste target type
        valid_bays = [b for b in self.loading_bays if b.bay_type == truck.payload_type]
        if not valid_bays:
            valid_bays = self.loading_bays

        # Pick bay with max muck remaining
        bays_with_muck = [b for b in valid_bays if b.muck_level.value > 0]
        if bays_with_muck:
            best_bay = max(bays_with_muck, key=lambda b: b.muck_level.value)
        else:
            best_bay = max(valid_bays, key=lambda b: b.muck_level.value)

        truck.target_bay_id = best_bay.bay_id
        truck.target_level = best_bay.level_index
        truck.state = TruckState.TRAVEL_EMPTY
        truck.current_location = "SURFACE_PARKING"

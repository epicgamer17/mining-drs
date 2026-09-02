"""Haulage routes, cycle times, and face distances."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HaulRoute:
    """Represents the haulage connection between a mine face and dump stockpiles.

    Captures haul distance, round-trip cycle time, and congestion scaling
    as more trucks are allocated to the face (addressing Navarra meeting line 8).
    """

    distance_km: float = 1.0
    base_cycle_time_min: float = 20.0
    congestion_factor: float = 0.05
    truck_payload_tonnes: float = 40.0

    def cycle_time(self, num_trucks: int = 1) -> float:
        """Calculates round-trip cycle time in minutes given allocated trucks.

        Cycle time gradually increases as more trucks use the same face
        due to loading queueing and transit congestion.
        """
        if num_trucks <= 1:
            return self.base_cycle_time_min
        congestion = 1.0 + self.congestion_factor * (num_trucks - 1)
        return self.base_cycle_time_min * congestion

    def max_daily_haulage(self, num_trucks: int = 1, operating_hours: float = 24.0) -> float:
        """Maximum daily tonnage movable by the given truck allocation."""
        if num_trucks <= 0:
            return 0.0
        c_time = self.cycle_time(num_trucks)
        trips_per_truck = (operating_hours * 60.0) / max(1.0, c_time)
        return num_trucks * trips_per_truck * self.truck_payload_tonnes

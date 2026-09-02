"""Mining logistics and fleet capacity functions."""

from __future__ import annotations


def truck_haul_capacity(
    distance_km: float = 1.0,
    num_trucks: int = 1,
    truck_payload_tonnes: float = 100.0,
    base_cycle_time_min: float = 20.0,
    congestion_factor: float = 0.05,
    operating_hours_per_day: float = 24.0,
) -> float:
    """Calculates maximum daily haulage capacity (tonnes/day) given distance and fleet count.

    Cycle time increases with congestion as more trucks are allocated to the same route.
    """
    if num_trucks <= 0:
        return 0.0

    congestion = 1.0 + congestion_factor * (num_trucks - 1)
    cycle_time = (base_cycle_time_min * distance_km) * congestion
    trips_per_truck = (operating_hours_per_day * 60.0) / max(1.0, cycle_time)
    return num_trucks * trips_per_truck * truck_payload_tonnes

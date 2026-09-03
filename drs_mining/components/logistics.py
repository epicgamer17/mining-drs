"""Mining logistics and fleet capacity functions."""

from __future__ import annotations


# TODO: where does this function come from?
def truck_haul_capacity(
    distance_km: float,
    num_trucks: int,
    truck_payload_tonnes: float,
    base_cycle_time_min: float,
    mechanical_availability: float,
    operator_efficiency: float,
    congestion_factor: float = 0.05,
    operating_hours_per_day: float = 24.0,
) -> float:
    """Calculates daily haulage capacity (tonnes/day) given distance, fleet count, and availability.

    Parameters
    ----------
    distance_km : float
        One-way haul distance from loading face to dump location (km).
    num_trucks : int
        Number of active haul trucks allocated to the circuit.
    truck_payload_tonnes : float
        Nominal payload capacity per truck (tonnes).
    base_cycle_time_min : float
        Base round-trip cycle time in minutes (spot, load, haul, dump, return) per km.
    mechanical_availability : float
        Fleet mechanical availability fraction (e.g., 0.85 for 85%). Required without default
        to prevent unrealistic nominal assumptions.
    operator_efficiency : float
        Job operational efficiency fraction (e.g., 0.90 for 90%). Required without default.
    congestion_factor : float, default 0.05
        Incremental cycle time expansion per additional truck due to queueing.
    operating_hours_per_day : float, default 24.0
        Total operational calendar hours per day.

    Returns
    -------
    float
        Total effective hauled tonnage per day.
    """
    if num_trucks <= 0 or distance_km <= 0 or truck_payload_tonnes <= 0:
        return 0.0

    congestion = 1.0 + congestion_factor * max(0, num_trucks - 1)
    cycle_time = (base_cycle_time_min * distance_km) * congestion
    trips_per_truck = (operating_hours_per_day * 60.0) / max(1.0, cycle_time)
    effective_utilization = max(0.0, mechanical_availability) * max(
        0.0, operator_efficiency
    )
    return float(
        num_trucks * trips_per_truck * truck_payload_tonnes * effective_utilization
    )

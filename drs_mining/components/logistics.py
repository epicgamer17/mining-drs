"""Mining logistics and fleet capacity functions."""

from __future__ import annotations

from typing import Optional, Mapping, Any


def truck_haul_capacity(
    distance_km: float,
    num_trucks: int,
    truck_payload_tonnes: float,
    mechanical_availability: float,
    operator_efficiency: float,
    fixed_cycle_time_min: float,
    haul_speed_kmh: Optional[float] = None,
    return_speed_kmh: Optional[float] = None,
    variable_time_per_km_min: Optional[float] = None,
    congestion_factor: float = 0.05,
    operating_hours_per_day: float = 24.0,
) -> float:
    """Calculates daily haulage capacity (tonnes/day) given distance, fleet count, and availability.

    In accordance with the SME Mining Engineering Handbook (§9.2) (TODO: Manually Verify) and
    Caterpillar Performance Handbook (TODO: Manually Verify), open-pit haul truck cycle time consists of:
    1. Fixed time (spot at shovel + load + turn/spot at dump + dump + delays).
    2. Variable travel time (loaded haul travel + empty return travel).

    Parameters
    ----------
    distance_km : float
        One-way haul distance from loading face to dump location (km).
    num_trucks : int
        Number of active haul trucks allocated to the circuit.
    truck_payload_tonnes : float
        Nominal payload capacity per truck (tonnes).
    mechanical_availability : float
        Fleet mechanical availability fraction (e.g., 0.85 for 85%). Required without default
        to prevent unrealistic nominal assumptions.
    operator_efficiency : float
        Job operational efficiency fraction (e.g., 0.90 for 90%). Required without default.
    fixed_cycle_time_min : float
        Fixed cycle time in minutes (spot at shovel, loading passes, turn & dump, delays).
        Required without default because shovel bucket-pass match, spotting conditions,
        and crusher dumps are strictly mine/circuit-specific. Typical range is 3.0 to 6.0 minutes.
    haul_speed_kmh : float, optional
        Average loaded haul speed in km/h (typically 20-30 km/h up-ramp/flat).
        Required if variable_time_per_km_min is not provided.
    return_speed_kmh : float, optional
        Average empty return speed in km/h (typically 35-50 km/h down-ramp/flat).
        Required if variable_time_per_km_min is not provided.
    variable_time_per_km_min : float, optional
        Explicit round-trip travel time in minutes per km. If provided, overrides
        haul_speed_kmh and return_speed_kmh with direct rate.
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

    if fixed_cycle_time_min < 0:
        raise ValueError(
            f"fixed_cycle_time_min must be non-negative, got {fixed_cycle_time_min}"
        )
    if variable_time_per_km_min is not None:
        if variable_time_per_km_min <= 0:
            raise ValueError(
                f"variable_time_per_km_min must be positive, got {variable_time_per_km_min}"
            )
        travel_time = distance_km * variable_time_per_km_min
    else:
        if haul_speed_kmh is None or return_speed_kmh is None:
            raise ValueError(
                "Must provide either variable_time_per_km_min or both (haul_speed_kmh, return_speed_kmh)."
            )
        if haul_speed_kmh <= 0 or return_speed_kmh <= 0:
            raise ValueError("Haul and return speeds must be strictly positive.")
        travel_time = distance_km * (60.0 / haul_speed_kmh + 60.0 / return_speed_kmh)

    base_cycle = fixed_cycle_time_min + travel_time
    congestion = 1.0 + congestion_factor * max(0, num_trucks - 1)
    cycle_time = max(1.0, base_cycle * congestion)

    trips_per_truck = (operating_hours_per_day * 60.0) / cycle_time
    effective_utilization = max(0.0, mechanical_availability) * max(
        0.0, operator_efficiency
    )

    return float(
        num_trucks * trips_per_truck * truck_payload_tonnes * effective_utilization
    )


def truck_cycle_time_breakdown(
    distance_km: float,
    num_trucks: int,
    truck_payload_tonnes: float,
    mechanical_availability: float,
    operator_efficiency: float,
    fixed_cycle_time_min: float,
    haul_speed_kmh: Optional[float] = None,
    return_speed_kmh: Optional[float] = None,
    variable_time_per_km_min: Optional[float] = None,
    congestion_factor: float = 0.05,
    operating_hours_per_day: float = 24.0,
) -> dict[str, float]:
    """Provides a detailed engineering breakdown of haul truck cycle time and fleet performance.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - 'fixed_time_min': Fixed spot, load, dump time.
        - 'haul_travel_min': Loaded haul travel time.
        - 'return_travel_min': Empty return travel time.
        - 'travel_time_min': Total variable travel time.
        - 'base_cycle_time_min': Total cycle time before congestion.
        - 'congestion_multiplier': Traffic and queueing multiplier.
        - 'total_cycle_time_min': Final effective cycle time.
        - 'trips_per_truck_day': Number of round trips per truck per day.
        - 'daily_tonnes_per_truck': Tonnage hauled per truck per day.
        - 'fleet_daily_tonnes': Total hauled tonnage for the entire fleet.
    """
    if distance_km <= 0 or num_trucks <= 0 or truck_payload_tonnes <= 0:
        return {
            "fixed_time_min": 0.0,
            "haul_travel_min": 0.0,
            "return_travel_min": 0.0,
            "travel_time_min": 0.0,
            "base_cycle_time_min": 0.0,
            "congestion_multiplier": 1.0,
            "total_cycle_time_min": 0.0,
            "trips_per_truck_day": 0.0,
            "daily_tonnes_per_truck": 0.0,
            "fleet_daily_tonnes": 0.0,
        }

    if fixed_cycle_time_min < 0:
        raise ValueError(
            f"fixed_cycle_time_min must be non-negative, got {fixed_cycle_time_min}"
        )

    if variable_time_per_km_min is not None:
        if variable_time_per_km_min <= 0:
            raise ValueError(
                f"variable_time_per_km_min must be positive, got {variable_time_per_km_min}"
            )
        haul_travel = distance_km * (variable_time_per_km_min / 2.0)
        return_travel = distance_km * (variable_time_per_km_min / 2.0)
        travel_time = distance_km * variable_time_per_km_min
    else:
        if haul_speed_kmh is None or return_speed_kmh is None:
            raise ValueError(
                "Must provide either variable_time_per_km_min or both (haul_speed_kmh, return_speed_kmh)."
            )
        if haul_speed_kmh <= 0 or return_speed_kmh <= 0:
            raise ValueError("Haul and return speeds must be strictly positive.")
        haul_travel = distance_km * (60.0 / haul_speed_kmh)
        return_travel = distance_km * (60.0 / return_speed_kmh)
        travel_time = haul_travel + return_travel

    base_cycle = fixed_cycle_time_min + travel_time
    congestion = 1.0 + congestion_factor * max(0, num_trucks - 1)
    cycle_time = max(1.0, base_cycle * congestion)

    trips_per_truck = (operating_hours_per_day * 60.0) / cycle_time
    effective_util = max(0.0, mechanical_availability) * max(0.0, operator_efficiency)
    daily_tonnes_truck = trips_per_truck * truck_payload_tonnes * effective_util
    fleet_tonnes = num_trucks * daily_tonnes_truck

    return {
        "fixed_time_min": float(fixed_cycle_time_min),
        "haul_travel_min": float(haul_travel),
        "return_travel_min": float(return_travel),
        "travel_time_min": float(travel_time),
        "base_cycle_time_min": float(base_cycle),
        "congestion_multiplier": float(congestion),
        "total_cycle_time_min": float(cycle_time),
        "trips_per_truck_day": float(trips_per_truck),
        "daily_tonnes_per_truck": float(daily_tonnes_truck),
        "fleet_daily_tonnes": float(fleet_tonnes),
    }

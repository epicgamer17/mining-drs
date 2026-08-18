from typing import Optional, Union
import json
import drs
from .fleet import Truck


class DRSRoadSegment:
    """Models a single-lane road segment or passing bay as a continuous availability timer in DRS.

    When a truck enters the segment, time_until_free is initialized to travel_time = L / v.
    The continuous engine integrates a -1.0 s/s decay rate down to 0.0.
    """

    def __init__(
        self,
        segment_id: str,
        length_m: float,
        segment_type: str,
    ):
        self.segment_id = segment_id
        self.length_m = length_m
        self.segment_type = segment_type  # "decline", "ramp", "surface", "level"

        self.time_until_free = drs.Timer(
            f"road_{segment_id}_t_free", initial_value=0.0, rate=-1.0
        )
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
        self.occupying_truck = truck
        return travel_time_s

    def update_continuous_step(self, dt: float):
        """Integrates continuous timer decay."""
        if self.time_until_free.value > 0.0:
            self.time_until_free.value = max(0.0, self.time_until_free.value - dt)
            if self.time_until_free.value <= 0.0:
                self.occupying_truck = None


# TODO: should this be in python-drs?
def load_topology_dict(topology: Union[dict, list, str, bytes]) -> Union[dict, list]:
    """Loads native Python dictionary/list topology representation.

    Args:
        topology: Path to JSON file, raw JSON string/bytes, or an already-loaded dict/list.

    Returns:
        The loaded topology dict/list structure.
    """
    if isinstance(topology, (dict, list)):
        return topology
    elif isinstance(topology, bytes):
        return json.loads(topology.decode("utf-8"))
    elif isinstance(topology, str):
        trimmed = topology.strip()
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (
            trimmed.startswith("[") and trimmed.endswith("]")
        ):
            return json.loads(trimmed)
        else:
            with open(topology, "r", encoding="utf-8") as f:
                return json.load(f)
    else:
        raise TypeError(
            f"Unsupported topology type: {type(topology)}. Expected dict, list, str (path or JSON), or bytes."
        )

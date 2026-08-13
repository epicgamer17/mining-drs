from typing import Optional
import drs
from .fleet import Truck, TruckState, LHD


class DRSLoadingBay:
    """Bridges LHD/Truck load cycles to DRS Stockpile/Muck Bay Levels and Flow Rates."""

    def __init__(
        self,
        engine,
        bay_id: str,
        bay_type: str,
        level_index: int,
        initial_muck: float = 0.0,
        lhd: Optional[LHD] = None,
    ):
        self.engine = engine
        self.bay_id = bay_id
        self.bay_type = bay_type  # "ORE" or "WASTE"
        self.level_index = level_index
        self.lhd = lhd or LHD(f"LHD_L{level_index}", level_index)

        # Continuous Muck Bay Level (Tonnes available)
        if hasattr(engine, "create_variable"):
            self.muck_level = engine.create_variable(
                name=f"muck_level_L{level_index}_{bay_type}",
                initial_value=initial_muck,
            )
            self.load_rate = engine.create_variable(
                name=f"load_rate_L{level_index}_{bay_type}",
                initial_value=0.0,
            )
        else:
            self.muck_level = drs.Level(
                f"muck_level_L{level_index}_{bay_type}",
                initial_value=initial_muck,
            )
            self.load_rate = drs.Variable(
                f"load_rate_L{level_index}_{bay_type}",
                initial_value=0.0,
            )

        self.active_truck: Optional[Truck] = None
        self.load_time_remaining: float = 0.0
        self.total_load_duration_sec: float = 0.0

    def start_loading(self, truck: Truck) -> bool:
        """Starts loading an active truck at this muck bay gateway."""
        if self.muck_level.value <= 0 or self.active_truck is not None:
            return False

        payload_cap = (
            truck.ore_payload_cap if self.bay_type == "ORE" else truck.waste_payload_cap
        )

        # Calculate exact load duration from Shelswell parameters:
        # Truck Spot (0.82m) + Acquisition Delay (~1.5m) + 2 LHD bucket passes
        lhd_cycle_sec = self.lhd.get_bucket_cycle_sec()
        total_load_time_sec = (0.82 + 1.5) * 60.0 + 2.0 * lhd_cycle_sec  # ~540.6s

        self.active_truck = truck
        self.active_truck.state = TruckState.LOADING
        self.load_time_remaining = total_load_time_sec
        self.total_load_duration_sec = total_load_time_sec

        # Set continuous DRS flow rate (Tonnes / Sec)
        self.load_rate.value = payload_cap / total_load_time_sec
        return True

    def update_continuous_step(self, dt: float):
        """Integrates muck removal and truck loading continuously."""
        if self.active_truck is not None:
            # Drain muck level and fill truck
            tonnes_moved = min(self.load_rate.value * dt, self.muck_level.value)
            self.muck_level.value = max(0.0, self.muck_level.value - tonnes_moved)
            self.active_truck.current_payload += tonnes_moved

            self.load_time_remaining -= dt
            if self.load_time_remaining <= 0.0:
                # Loading complete
                self.active_truck.state = TruckState.TRAVEL_LOADED
                self.active_truck = None
                self.load_rate.value = 0.0


class DRSDumpingBay:
    """Bridges surface truck dumping to DRS Continuous Accumulators."""

    def __init__(self, engine, bay_id: str, bay_type: str, location_name: str):
        self.engine = engine
        self.bay_id = bay_id
        self.bay_type = bay_type  # "ORE" or "WASTE"
        self.location_name = location_name

        if hasattr(engine, "create_variable"):
            self.dumped_total = engine.create_variable(
                name=f"dumped_{bay_type}_total", initial_value=0.0
            )
            self.dump_rate = engine.create_variable(
                name=f"dump_rate_{bay_type}", initial_value=0.0
            )
        else:
            self.dumped_total = drs.Level(
                f"dumped_{bay_type}_total", initial_value=0.0
            )
            self.dump_rate = drs.Variable(
                f"dump_rate_{bay_type}", initial_value=0.0
            )

        self.active_truck: Optional[Truck] = None
        self.dump_time_remaining: float = 0.0

    def start_dumping(self, truck: Truck) -> bool:
        """Starts surface dumping for a loaded truck."""
        if self.active_truck is not None or truck.current_payload <= 0.0:
            return False

        # Shelswell dumping timings: Dump spot (0.57 min) + Bed raise/dump (0.88 min) = 1.45 min (87s)
        total_dump_time_sec = (0.57 + 0.88) * 60.0

        self.active_truck = truck
        self.active_truck.state = TruckState.DUMPING
        self.dump_time_remaining = total_dump_time_sec
        self.dump_rate.value = truck.current_payload / total_dump_time_sec
        return True

    def update_continuous_step(self, dt: float):
        """Integrates truck unloading and surface stockpile accumulation continuously."""
        if self.active_truck is not None:
            tonnes_dumped = min(
                self.dump_rate.value * dt, self.active_truck.current_payload
            )
            self.active_truck.current_payload = max(
                0.0, self.active_truck.current_payload - tonnes_dumped
            )
            self.dumped_total.value += tonnes_dumped

            self.dump_time_remaining -= dt
            if self.dump_time_remaining <= 0.0:
                self.active_truck.current_payload = 0.0
                self.active_truck.state = TruckState.PARKED
                self.active_truck = None
                self.dump_rate.value = 0.0

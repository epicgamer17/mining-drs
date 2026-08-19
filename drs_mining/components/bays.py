from typing import Optional
import drs
from .fleet import Truck, TruckState, LHD


class LoadingBay(drs.Module):
    """Bridges LHD/Truck load cycles to DRS Stockpile/Muck Bay Levels and Flow Rates."""

    def __init__(
        self,
        bay_id: str,
        bay_type: str,
        level_index: int,
        initial_muck: float,
        truck_spot_min: float,
        acquisition_delay_min: float,
        bucket_passes: float,
        lhd: LHD,
    ):
        super().__init__()
        self.bay_id = bay_id
        self.bay_type = bay_type  # "ORE" or "WASTE"
        self.level_index = level_index
        self.initial_muck = initial_muck
        self.lhd = lhd
        self.truck_spot_min = truck_spot_min
        self.acquisition_delay_min = acquisition_delay_min
        self.bucket_passes = bucket_passes

        # Continuous Muck Bay Level (Tonnes available)
        self.muck_level = drs.Level(
            f"muck_level_L{level_index}_{bay_type}",
            initial_value=initial_muck,
        )
        self.muck_level.lower_threshold = 0.0
        self.load_rate = drs.Variable(
            f"load_rate_L{level_index}_{bay_type}",
            initial_value=0.0,
        )

        self.active_truck: Optional[Truck] = None
        self.load_time_remaining: float = 0.0
        self.total_load_duration_sec: float = 0.0

    def calculate_load_duration_sec(self, truck: Truck) -> float:
        """Calculates exact load duration in seconds for an active truck."""
        lhd_cycle_sec = self.lhd.get_bucket_cycle_sec()
        return (
            self.truck_spot_min + self.acquisition_delay_min
        ) * 60.0 + self.bucket_passes * lhd_cycle_sec

    def start_loading(self, truck: Truck) -> bool:
        """Starts loading an active truck at this muck bay gateway."""
        if self.muck_level.value <= 0 or self.active_truck is not None:
            return False

        payload_cap = (
            truck.ore_payload_cap if self.bay_type == "ORE" else truck.waste_payload_cap
        )

        total_load_time_sec = self.calculate_load_duration_sec(truck)
        if total_load_time_sec <= 0.0:
            total_load_time_sec = 1.0

        self.active_truck = truck
        self.active_truck.state = TruckState.LOADING
        self.load_time_remaining = total_load_time_sec
        self.total_load_duration_sec = total_load_time_sec

        # Set continuous DRS flow rate (Tonnes / Sec)
        rate_val = payload_cap / total_load_time_sec
        self.load_rate.value = rate_val
        self.muck_level.rate = -rate_val
        return True

    def update_continuous_step(self, dt: float):
        """Integrates muck removal and truck loading continuously using DRS rate dynamics."""
        if self.active_truck is not None:
            prev_muck = self.muck_level.value
            effective_dt = min(dt, self.load_time_remaining) if self.load_time_remaining > 0 else dt
            self.muck_level.step(effective_dt)
            tonnes_moved = max(0.0, prev_muck - self.muck_level.value)
            self.active_truck.current_payload += tonnes_moved

            self.load_time_remaining -= dt
            if self.load_time_remaining <= 0.0 or self.muck_level.value <= 0.0:
                self.active_truck.state = TruckState.TRAVEL_LOADED
                self.active_truck = None
                self.load_rate.value = 0.0
                self.muck_level.rate = 0.0


class DumpingBay(drs.Module):
    """Bridges surface truck dumping to DRS Continuous Accumulators."""

    def __init__(
        self,
        bay_id: str,
        bay_type: str,
        location_name: str,
        dump_spot_min: float,
        bed_raise_dump_min: float,
    ):
        super().__init__()
        self.bay_id = bay_id
        self.bay_type = bay_type  # "ORE" or "WASTE"
        self.location_name = location_name
        self.dump_spot_min = dump_spot_min
        self.bed_raise_dump_min = bed_raise_dump_min

        self.dumped_total = drs.Level(f"dumped_{bay_type}_total", initial_value=0.0)
        self.dump_rate = drs.Variable(f"dump_rate_{bay_type}", initial_value=0.0)

        self.active_truck: Optional[Truck] = None
        self.dump_time_remaining: float = 0.0

    def calculate_dump_duration_sec(self, truck: Truck) -> float:
        """Calculates dump duration in seconds."""
        duration = (self.dump_spot_min + self.bed_raise_dump_min) * 60.0
        return duration if duration > 0.0 else 1.0

    def start_dumping(self, truck: Truck) -> bool:
        """Starts surface dumping for a loaded truck."""
        if self.active_truck is not None or truck.current_payload <= 0.0:
            return False

        total_dump_time_sec = self.calculate_dump_duration_sec(truck)

        self.active_truck = truck
        self.active_truck.state = TruckState.DUMPING
        self.dump_time_remaining = total_dump_time_sec

        # Flow into surface stockpile accumulator (Tonnes / Sec)
        rate_val = truck.current_payload / total_dump_time_sec
        self.dump_rate.value = rate_val
        self.dumped_total.rate = rate_val
        return True

    def update_continuous_step(self, dt: float):
        """Integrates continuous dumping and flow rate resets using DRS rate dynamics."""
        if self.active_truck is not None:
            prev_dumped = self.dumped_total.value
            effective_dt = min(dt, self.dump_time_remaining) if self.dump_time_remaining > 0 else dt
            self.dumped_total.step(effective_dt)
            tonnes_dumped = min(self.dumped_total.value - prev_dumped, self.active_truck.current_payload)
            self.active_truck.current_payload = max(
                0.0, self.active_truck.current_payload - tonnes_dumped
            )

            self.dump_time_remaining -= dt
            if self.dump_time_remaining <= 0.0:
                self.active_truck.current_payload = 0.0
                self.active_truck.state = TruckState.TRAVEL_EMPTY
                self.active_truck = None
                self.dump_rate.value = 0.0
                self.dumped_total.rate = 0.0

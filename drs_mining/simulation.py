from typing import List, Dict, Optional, Union
import random
import numpy as np
import drs
from tqdm import tqdm

from .components.fleet import Truck, TruckState
from .components.topology import DRSRoadSegment, load_topology_dict
from .components.bays import DRSLoadingBay, DRSDumpingBay
from .controllers.dispatch import ShelswellDispatchController


class HybridDRSModule(drs.Module):
    """State container for simulation time and haul accumulators."""

    def __init__(self):
        super().__init__()
        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)
        self.ore_hauled = drs.Level("OreHauled", initial_value=0.0)
        self.waste_hauled = drs.Level("WasteHauled", initial_value=0.0)

    def step_update(self):
        self.global_time.rate = 1.0


class ShelswellHybridSimulation:
    """Master Simulation Orchestrator uniting Discrete Domain Entities, 
    DRS Rate Gateways, NumPy Vectorization, and Native Event-Driven DRS Integration.
    """

    def __init__(
        self,
        num_trucks: int = 10,
        num_operators: int = 10,
        mechanical_availability: float = 1.0,
        step_size_sec: float = 1.0,
        topology_dict: Optional[Union[dict, list]] = None,
    ):
        if topology_dict is not None:
            top_data = load_topology_dict(topology_dict)
            if isinstance(top_data, dict) and "attributes" in top_data:
                attrs = top_data["attributes"]
                num_trucks = attrs.get("num_trucks", num_trucks)
                num_operators = attrs.get("num_operators", num_operators)
                mechanical_availability = attrs.get("mechanical_availability", mechanical_availability)

        self.num_trucks = num_trucks
        self.num_operators = num_operators
        self.mechanical_availability = mechanical_availability

        # 1. Core DRS Module & Engine
        self.module = HybridDRSModule()
        self.engine = drs.DRSEngine()
        self.engine.register(self.module)

        # 2. Mine Topology & Availability Timers
        self.decline = DRSRoadSegment(self.engine, "decline_2100m", 2100.0, "decline")
        self.ramp_levels = [
            DRSRoadSegment(self.engine, f"ramp_L{i}", 300.0, "ramp")
            for i in range(1, 8)
        ]
        self.surface_rom_road = DRSRoadSegment(self.engine, "surf_rom", 300.0, "surface")
        self.surface_waste_road = DRSRoadSegment(self.engine, "surf_waste", 440.0, "surface")

        # 3. Loading & Dumping Bays (Unconstrained upstream muck supply per paper spec)
        self.loading_bays: List[DRSLoadingBay] = []
        for i in range(1, 8):
            self.loading_bays.append(
                DRSLoadingBay(self.engine, f"L{i}_ORE", "ORE", i, initial_muck=10_000_000.0)
            )
            self.loading_bays.append(
                DRSLoadingBay(self.engine, f"L{i}_WASTE", "WASTE", i, initial_muck=2_000_000.0)
            )

        self.rom_dump_bay = DRSDumpingBay(self.engine, "ROM_PAD", "ORE", "SURFACE_ROM")
        self.waste_dump_bay = DRSDumpingBay(self.engine, "WASTE_DUMP", "WASTE", "SURFACE_WASTE_DUMP")
        self.dump_bays = [self.rom_dump_bay, self.waste_dump_bay]

        # 4. Fleet & Vectorized NumPy State Arrays
        eff_trucks_count = int(min(num_trucks * mechanical_availability, num_operators))
        eff_trucks_count = max(1, eff_trucks_count)
        self.n_fleet = eff_trucks_count

        self.trucks = [
            Truck(truck_id=f"T{i:02d}", truck_type="AD30")
            for i in range(1, eff_trucks_count + 1)
        ]
        self.dispatch = ShelswellDispatchController(self.trucks, self.loading_bays, {})

        # Vectorized state trackers (1D NumPy arrays)
        self.timers = np.zeros(self.n_fleet, dtype=np.float64)
        self.fuel_pct = np.full(self.n_fleet, 100.0, dtype=np.float64)

    def _get_travel_time_sec(self, truck: Truck, is_loaded: bool) -> float:
        """Calculates exact baseline travel duration in seconds based on mine layout."""
        level = truck.target_level or 4
        muck_type = truck.payload_type.lower()

        v_surf = truck.get_speed_mps("surface")
        v_dec = truck.get_speed_mps("decline")
        v_ramp = truck.get_speed_mps("ramp")
        v_lvl = truck.get_speed_mps("level")

        d_dec = 2100.0
        d_ramp = (level - 1) * 300.0
        d_lvl = 40.0 if muck_type == "ore" else 55.0
        d_surf = 300.0 if muck_type == "ore" else 440.0

        if not is_loaded:
            t_total_s = (d_surf / v_surf) + (d_dec / v_dec) + (d_ramp / v_ramp) + (d_lvl / v_lvl)
        else:
            t_total_s = (d_lvl / v_lvl) + (d_ramp / v_ramp) + (d_dec / v_dec) + (d_surf / v_surf)

        return t_total_s

    def step(self, dt_step: float):
        """Single Simulation Step Execution."""
        active_mask = self.timers > 0
        self.timers[active_mask] = np.maximum(0.0, self.timers[active_mask] - dt_step)

        for i, truck in enumerate(self.trucks):
            if truck.state in (TruckState.TRAVEL_EMPTY, TruckState.TRAVEL_LOADED):
                self.fuel_pct[i] -= truck.fuel_burn_rate_pct_per_sec * dt_step
                truck.fuel_level_pct = self.fuel_pct[i]

        for i, truck in enumerate(self.trucks):
            if truck.state == TruckState.PARKED:
                self.dispatch.assign_next_destination(truck)
                if truck.state == TruckState.TRAVEL_EMPTY:
                    t_travel = self._get_travel_time_sec(truck, is_loaded=False)
                    self.timers[i] = t_travel

            elif truck.state == TruckState.TRAVEL_EMPTY:
                if self.timers[i] <= 0.0:
                    truck.state = TruckState.WAITING_LOAD

            if truck.state == TruckState.WAITING_LOAD:
                target_bay = next(
                    (b for b in self.loading_bays if b.bay_id == truck.target_bay_id), None
                )
                if target_bay and target_bay.start_loading(truck):
                    self.timers[i] = target_bay.total_load_duration_sec

            elif truck.state == TruckState.TRAVEL_LOADED:
                if self.timers[i] <= 0.0:
                    truck.state = TruckState.WAITING_DUMP

            if truck.state == TruckState.WAITING_DUMP:
                target_dump = (
                    self.rom_dump_bay if truck.payload_type == "ORE" else self.waste_dump_bay
                )
                if target_dump.start_dumping(truck):
                    self.timers[i] = target_dump.dump_time_remaining

            elif truck.state == TruckState.REFUELING:
                self.fuel_pct[i] = 100.0
                truck.fuel_level_pct = 100.0
                truck.state = TruckState.PARKED

        self.decline.update_continuous_step(dt_step)
        for ramp in self.ramp_levels:
            ramp.update_continuous_step(dt_step)
        for bay in self.loading_bays:
            bay.update_continuous_step(dt_step)
            if bay.active_truck is not None and bay.active_truck.state == TruckState.TRAVEL_LOADED:
                idx = self.trucks.index(bay.active_truck)
                if self.timers[idx] <= 0.0:
                    t_travel = self._get_travel_time_sec(bay.active_truck, is_loaded=True)
                    self.timers[idx] = t_travel

        for dump_bay in self.dump_bays:
            dump_bay.update_continuous_step(dt_step)

        self.module.ore_hauled.value = self.rom_dump_bay.dumped_total.value
        self.module.waste_hauled.value = self.waste_dump_bay.dumped_total.value
        self.module.step_update()
        self.module.global_time._update(dt_step / 86400.0)

    def run_simulation(self, total_days: float = 365.0, dt: float = 300.0, show_progress: bool = False) -> float:
        """Runs event-driven DRS integration over specified days (365 calendar days baseline)."""
        total_seconds = float(total_days * 24.0 * 3600.0)
        current_sec = 0.0
        step_dt = float(dt)

        pbar = tqdm(total=int(total_seconds), desc=f"Simulating {total_days:.0f} days", disable=not show_progress)

        while current_sec < total_seconds:
            time_in_day = current_sec % 86400.0
            is_shift_gap = (10.5 * 3600.0 <= time_in_day < 12.0 * 3600.0) or (
                22.5 * 3600.0 <= time_in_day < 24.0 * 3600.0
            )

            if is_shift_gap:
                self.module.global_time._update(step_dt / 86400.0)
                current_sec += step_dt
                pbar.update(int(step_dt))
                continue

            self.step(step_dt)

            current_sec += step_dt
            pbar.update(int(step_dt))

        pbar.close()
        total_hauled = (
            self.rom_dump_bay.dumped_total.value + self.waste_dump_bay.dumped_total.value
        )
        return total_hauled / total_days

from typing import List, Dict, Optional, Union
import random
import drs
from tqdm import tqdm

from .components.fleet import Truck, TruckState
from .components.topology import DRSRoadSegment, load_topology_dict
from .components.bays import DRSLoadingBay, DRSDumpingBay
from .controllers.dispatch import ShelswellDispatchController


class HybridDRSModule(drs.Module):
    """Container DRS Module registering global variables and simulation accumulators."""

    def __init__(self):
        super().__init__()
        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)
        self.ore_hauled = drs.Level("OreHauled", initial_value=0.0)
        self.waste_hauled = drs.Level("WasteHauled", initial_value=0.0)

    def forward(self):
        self.global_time.rate = 1.0


class ShelswellHybridSimulation:
    """Master Simulation Orchestrator uniting Discrete Domain Entities, 
    DRS Rate Gateways, and the DRS Core Integration Engine.
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
        self.step_size_sec = step_size_sec

        # 1. Instantiate Core DRS Module & Engine
        self.module = HybridDRSModule()
        self.engine = drs.DRSEngine(self.module)

        # 2. Build Mine Topology & Road Availability Timers
        self.decline = DRSRoadSegment(self.engine, "decline_2100m", 2100.0, "decline")
        self.ramp_levels = [
            DRSRoadSegment(self.engine, f"ramp_L{i}", 300.0, "ramp")
            for i in range(1, 8)  # 7 levels, 300m apart
        ]
        self.surface_rom_road = DRSRoadSegment(self.engine, "surf_rom", 300.0, "surface")
        self.surface_waste_road = DRSRoadSegment(self.engine, "surf_waste", 440.0, "surface")

        # 3. Instantiate Loading & Dumping Bays
        self.loading_bays: List[DRSLoadingBay] = []
        for i in range(1, 8):
            self.loading_bays.append(
                DRSLoadingBay(self.engine, f"L{i}_ORE", "ORE", i, initial_muck=20000.0)
            )
            self.loading_bays.append(
                DRSLoadingBay(self.engine, f"L{i}_WASTE", "WASTE", i, initial_muck=4000.0)
            )

        self.rom_dump_bay = DRSDumpingBay(self.engine, "ROM_PAD", "ORE", "SURFACE_ROM")
        self.waste_dump_bay = DRSDumpingBay(self.engine, "WASTE_DUMP", "WASTE", "SURFACE_WASTE_DUMP")
        self.dump_bays = [self.rom_dump_bay, self.waste_dump_bay]

        # 4. Instantiate Fleet & Fleet Constraints
        eff_trucks_count = int(min(num_trucks * mechanical_availability, num_operators))
        eff_trucks_count = max(1, eff_trucks_count)

        self.trucks = [
            Truck(truck_id=f"T{i:02d}", truck_type="AD30")
            for i in range(1, eff_trucks_count + 1)
        ]

        self.dispatch = ShelswellDispatchController(self.trucks, self.loading_bays, {})

        # Internal travel state trackers per truck
        self.travel_timer: Dict[str, float] = {t.truck_id: 0.0 for t in self.trucks}

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

    def step(self, dt: float = 1.0):
        """Single Simulation Step Execution."""
        for truck in self.trucks:
            if truck.state == TruckState.PARKED:
                self.dispatch.assign_next_destination(truck)
                if truck.state == TruckState.TRAVEL_EMPTY:
                    self.travel_timer[truck.truck_id] = self._get_travel_time_sec(truck, is_loaded=False)

            elif truck.state == TruckState.TRAVEL_EMPTY:
                self.travel_timer[truck.truck_id] -= dt
                if self.travel_timer[truck.truck_id] <= 0.0:
                    truck.state = TruckState.WAITING_LOAD

            elif truck.state == TruckState.WAITING_LOAD:
                target_bay = next(
                    (b for b in self.loading_bays if b.bay_id == truck.target_bay_id), None
                )
                if target_bay and target_bay.start_loading(truck):
                    pass

            elif truck.state == TruckState.LOADING:
                pass

            elif truck.state == TruckState.TRAVEL_LOADED:
                if self.travel_timer[truck.truck_id] <= 0.0:
                    self.travel_timer[truck.truck_id] = self._get_travel_time_sec(truck, is_loaded=True)

                self.travel_timer[truck.truck_id] -= dt
                if self.travel_timer[truck.truck_id] <= 0.0:
                    truck.state = TruckState.WAITING_DUMP

            elif truck.state == TruckState.WAITING_DUMP:
                target_dump = (
                    self.rom_dump_bay if truck.payload_type == "ORE" else self.waste_dump_bay
                )
                target_dump.start_dumping(truck)

            elif truck.state == TruckState.DUMPING:
                pass

            elif truck.state == TruckState.REFUELING:
                truck.fuel_level_pct = 100.0
                truck.state = TruckState.PARKED

            if truck.state in (TruckState.TRAVEL_EMPTY, TruckState.TRAVEL_LOADED):
                truck.fuel_level_pct -= truck.fuel_burn_rate_pct_per_sec * dt

        self.decline.update_continuous_step(dt)
        for ramp in self.ramp_levels:
            ramp.update_continuous_step(dt)
        for bay in self.loading_bays:
            bay.update_continuous_step(dt)
        for dump_bay in self.dump_bays:
            dump_bay.update_continuous_step(dt)

        self.module.ore_hauled.value = self.rom_dump_bay.dumped_total.value
        self.module.waste_hauled.value = self.waste_dump_bay.dumped_total.value

        self.module.forward()
        self.module.global_time._update(dt / 86400.0)

    def run_simulation(self, total_days: float = 365.0, dt: float = 60.0, show_progress: bool = False) -> float:
        """Runs mine production schedule over specified days (365 calendar days baseline)."""
        total_seconds = float(total_days * 24.0 * 3600.0)
        step_dt = float(dt)
        total_steps = int(total_seconds / step_dt)

        pbar = tqdm(total=total_steps, desc=f"Simulating {total_days:.0f} days", disable=not show_progress)
        current_sec = 0.0

        while current_sec < total_seconds:
            time_in_day = current_sec % 86400.0
            is_shift_change = (10.5 * 3600.0 < time_in_day < 12.0 * 3600.0) or (
                22.5 * 3600.0 < time_in_day < 24.0 * 3600.0
            )

            if not is_shift_change:
                self.step(step_dt)
            else:
                self.module.global_time._update(step_dt / 86400.0)

            current_sec += step_dt
            pbar.update(1)

        pbar.close()
        total_hauled = (
            self.rom_dump_bay.dumped_total.value + self.waste_dump_bay.dumped_total.value
        )
        return total_hauled / total_days

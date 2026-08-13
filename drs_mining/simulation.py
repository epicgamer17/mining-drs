from typing import List, Dict, Optional
import random
import drs

from .components.fleet import Truck, TruckState
from .components.topology import DRSRoadSegment
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
    ):
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
        # Bounded effective fleet: eff_trucks = min(N_trucks * Availability, N_operators)
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

        # Corridors
        v_surf = truck.get_speed_mps("surface")
        v_dec = truck.get_speed_mps("decline")
        v_ramp = truck.get_speed_mps("ramp")
        v_lvl = truck.get_speed_mps("level")

        d_dec = 2100.0
        d_ramp = (level - 1) * 300.0
        d_lvl = 40.0 if muck_type == "ore" else 55.0
        d_surf = 300.0 if muck_type == "ore" else 440.0

        if not is_loaded:
            # Empty trip: Surface -> Decline -> Ramp -> Level Access
            t_total_s = (d_surf / v_surf) + (d_dec / v_dec) + (d_ramp / v_ramp) + (d_lvl / v_lvl)
        else:
            # Loaded trip: Level Access -> Ramp -> Decline -> Surface Dump
            t_total_s = (d_lvl / v_lvl) + (d_ramp / v_ramp) + (d_dec / v_dec) + (d_surf / v_surf)

        return t_total_s

    def step(self, dt: float = 1.0):
        """Single Simulation Step Execution."""
        # A. Update Discrete Entities & Dispatch Logic
        for truck in self.trucks:
            if truck.state == TruckState.PARKED:
                self.dispatch.assign_next_destination(truck)
                if truck.state == TruckState.TRAVEL_EMPTY:
                    self.travel_timer[truck.truck_id] = self._get_travel_time_sec(truck, is_loaded=False)

            elif truck.state == TruckState.TRAVEL_EMPTY:
                # Travel empty towards assigned level loading bay
                self.travel_timer[truck.truck_id] -= dt
                if self.travel_timer[truck.truck_id] <= 0.0:
                    truck.state = TruckState.WAITING_LOAD

            elif truck.state == TruckState.WAITING_LOAD:
                # Arrived at level loadout bay: find bay and initiate loading
                target_bay = next(
                    (b for b in self.loading_bays if b.bay_id == truck.target_bay_id), None
                )
                if target_bay and target_bay.start_loading(truck):
                    pass

            elif truck.state == TruckState.LOADING:
                # Handled continuously by DRSLoadingBay gateway
                pass

            elif truck.state == TruckState.TRAVEL_LOADED:
                if self.travel_timer[truck.truck_id] <= 0.0:
                    # Set loaded travel time
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
                # Handled continuously by DRSDumpingBay gateway
                pass

            elif truck.state == TruckState.REFUELING:
                # Refueling complete: return to PARKED state
                truck.fuel_level_pct = 100.0
                truck.state = TruckState.PARKED

            # Fuel burn integration
            if truck.state in (TruckState.TRAVEL_EMPTY, TruckState.TRAVEL_LOADED):
                truck.fuel_level_pct -= truck.fuel_burn_rate_pct_per_sec * dt

        # B. Update Continuous Gateways & Availability Timers
        self.decline.update_continuous_step(dt)
        for ramp in self.ramp_levels:
            ramp.update_continuous_step(dt)
        for bay in self.loading_bays:
            bay.update_continuous_step(dt)
        for dump_bay in self.dump_bays:
            dump_bay.update_continuous_step(dt)

        # Sync continuous accumulators with module variables
        self.module.ore_hauled.value = self.rom_dump_bay.dumped_total.value
        self.module.waste_hauled.value = self.waste_dump_bay.dumped_total.value

        # C. Advance Continuous DRS Integration Loop
        self.module.forward()
        self.module.global_time._update(dt / 86400.0)  # advance global clock in days

    def run_simulation(self, total_days: float = 365.0, dt: float = 1.0) -> float:
        """Runs mine production schedule over specified days (365 calendar days baseline).
        
        Incorporates shift schedules: 2 shifts/day, 10.5h working time (1.5h shift gap).
        Returns total average daily productivity (tonnes / day).
        """
        total_seconds = int(total_days * 24 * 3600)
        # Fast step advancement for long sweeps
        step_dt = 10.0  # 10s integration resolution for high speed
        current_sec = 0

        while current_sec < total_seconds:
            # Shift Check: 2 shifts/day, 10.5h work time (1.5h shift gap)
            time_in_day = current_sec % 86400
            is_shift_change = (10.5 * 3600 < time_in_day < 12.0 * 3600) or (
                22.5 * 3600 < time_in_day < 24.0 * 3600
            )

            if not is_shift_change:
                self.step(step_dt)
            else:
                # Hold continuous loading/dumping during shift change
                self.module.global_time._update(step_dt / 86400.0)

            current_sec += int(step_dt)

        total_hauled = (
            self.rom_dump_bay.dumped_total.value + self.waste_dump_bay.dumped_total.value
        )
        return total_hauled / total_days

import random
import drs
from drs.flow import Flow
from .shelswell_routing import get_travel_times_hours, get_truck_loading_time_hours

class ShelswellHaulageModel(drs.Module):
    """Discrete Rate Simulation model approximating the Shelswell (2017) truck haulage fleet simulation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)
        self.schedule_timer = drs.Timer("ScheduleTimer", initial_value=0.0)
        self.schedule_timer.upper_threshold = 1.0
        
        # Muck piles representing scheduled ore/waste not yet hauled
        self.ore_muck = drs.Level("OreMuck", initial_value=100000.0)
        self.waste_muck = drs.Level("WasteMuck", initial_value=20000.0)
        
        # Total hauled tonnes
        self.ore_hauled = drs.Level("OreHauled", initial_value=0.0)
        self.waste_hauled = drs.Level("WasteHauled", initial_value=0.0)
        
        # Instantaneous haul rates
        self.ore_haul_rate = drs.Variable("ore_haul_rate", 0.0)
        self.waste_haul_rate = drs.Variable("waste_haul_rate", 0.0)
        
        # Parametric fleet inputs
        self.truck_count = drs.Variable("truck_count", config.total_truck_count)
        self.operator_count = drs.Variable("operator_count", config.total_operators)
        self.availability = drs.Variable("availability", config.overall_mechanical_availability)

    def forward(self):
        self.global_time.rate = 1.0
        self.schedule_timer.rate = 1.0
        
        # Daily scheduling update
        if self.schedule_timer.value >= 1.0 - 1e-6:
            self.schedule_timer.reset()
            # Symmetric triangular distribution ±35% for ore around peak 6000 t/d
            ore_scheduled = random.triangular(3900.0, 8100.0, 6000.0)
            # Symmetric triangular distribution for waste around peak 1090.9 t/d
            waste_scheduled = random.triangular(436.4, 1745.4, 1090.9)
            
            self.ore_muck.value += ore_scheduled
            self.waste_muck.value += waste_scheduled

        num_trucks = self.truck_count.value
        num_operators = self.operator_count.value
        avail = self.availability.value
        
        # Bounded effective fleet size based on mechanical availability and operators
        eff_trucks = min(num_trucks * avail, num_operators)
        
        # Number of active levels based on paper's configuration
        active_levels_count = max(1, min(7, int(num_trucks) // 2 + 1))
        
        # active levels centered around level 4
        start_lvl = max(1, 4 - (active_levels_count - 1) // 2)
        active_levels = list(range(start_lvl, start_lvl + active_levels_count))
        
        # Even truck distribution across active levels
        trucks_per_level = eff_trucks / len(active_levels)
        
        total_ore_cap = 0.0
        total_waste_cap = 0.0
        
        for lvl in active_levels:
            # 1 loader per level, split according to ore/waste scheduled ratio (5.5:1)
            frac_ore = 5.5 / 6.5
            frac_waste = 1.0 / 6.5
            
            # --- Ore Haulage Capacity ---
            t_empty_ore, t_loaded_ore = get_travel_times_hours(lvl, "ore", self.config)
            t_load_ore = get_truck_loading_time_hours("ore", self.config)
            t_dump_ore = (self.config.truck_dump_spot_minutes + self.config.truck_dump_minutes) / 60.0
            t_traffic = self.config.traffic_delay_per_truck_hours * trucks_per_level
            
            t_cycle_ore = t_empty_ore + t_load_ore + t_loaded_ore + t_dump_ore + t_traffic
            
            mf_ore = (trucks_per_level * t_load_ore) / t_cycle_ore
            if mf_ore < 1.0:
                ore_cap = (trucks_per_level * frac_ore) * (24.0 * self.config.seat_time_fraction_truck / t_cycle_ore) * self.config.truck_payload_ore
            else:
                ore_cap = (1.0 * frac_ore) * (24.0 * self.config.seat_time_fraction_lhd / t_load_ore) * self.config.truck_payload_ore
                
            # --- Waste Haulage Capacity ---
            t_empty_waste, t_loaded_waste = get_travel_times_hours(lvl, "waste", self.config)
            t_load_waste = get_truck_loading_time_hours("waste", self.config)
            t_dump_waste = (self.config.truck_dump_spot_minutes + self.config.truck_dump_minutes) / 60.0
            
            t_cycle_waste = t_empty_waste + t_load_waste + t_loaded_waste + t_dump_waste + t_traffic
            
            mf_waste = (trucks_per_level * t_load_waste) / t_cycle_waste
            if mf_waste < 1.0:
                waste_cap = (trucks_per_level * frac_waste) * (24.0 * self.config.seat_time_fraction_truck / t_cycle_waste) * self.config.truck_payload_waste
            else:
                waste_cap = (1.0 * frac_waste) * (24.0 * self.config.seat_time_fraction_lhd / t_load_waste) * self.config.truck_payload_waste
                
            total_ore_cap += ore_cap
            total_waste_cap += waste_cap

        self.ore_haul_rate.value = total_ore_cap
        self.waste_haul_rate.value = total_waste_cap
        
        self.ore_muck.rate = -self.ore_haul_rate.value
        self.waste_muck.rate = -self.waste_haul_rate.value
        self.ore_hauled.rate = self.ore_haul_rate.value
        self.waste_hauled.rate = self.waste_haul_rate.value
        
        self.ore_muck.lower_threshold = 0.0
        self.waste_muck.lower_threshold = 0.0

    def is_terminating_condition_met(self) -> bool:
        # Replicate 1 year (365 days)
        return self.global_time.value >= 365.0

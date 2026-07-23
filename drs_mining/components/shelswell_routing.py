def get_travel_times_hours(level: int, muck_type: str, config) -> tuple[float, float]:
    """Calculates empty and loaded travel times in hours for a given level.
    
    Levels are 1-indexed (1 to 7).
    muck_type is either 'ore' or 'waste'.
    """
    # Convert speeds from kph to m/hr
    v_surf_empty = config.speed_surface_empty * 1000.0
    v_surf_loaded = config.speed_surface_loaded * 1000.0
    v_dec_empty = config.speed_decline_empty * 1000.0
    v_dec_loaded = config.speed_decline_loaded * 1000.0
    v_ramp_empty = config.speed_ramp_empty * 1000.0
    v_ramp_loaded = config.speed_ramp_loaded * 1000.0
    v_lvl_empty = config.speed_level_empty * 1000.0
    v_lvl_loaded = config.speed_level_loaded * 1000.0

    # Distances in meters
    d_dec = config.decline_length
    d_ramp = (level - 1) * config.level_spacing
    
    if muck_type == "ore":
        d_lvl = config.ore_loadout_distance
        d_surf = config.rom_pad_distance
    else:
        d_lvl = config.waste_loadout_distance
        d_surf = config.waste_stockpile_distance

    # Empty path travel time (ROM/Stockpile -> Portal -> Decline -> Ramp -> Level Loadout)
    t_empty = (d_surf / v_surf_empty) + (d_dec / v_dec_empty) + (d_ramp / v_ramp_empty) + (d_lvl / v_lvl_empty)
    
    # Loaded path travel time (Level Loadout -> Ramp -> Decline -> Portal -> ROM/Stockpile)
    t_loaded = (d_lvl / v_lvl_loaded) + (d_ramp / v_ramp_loaded) + (d_dec / v_dec_loaded) + (d_surf / v_surf_loaded)
    
    return t_empty, t_loaded


def get_truck_loading_time_hours(muck_type: str, config) -> float:
    """Calculates LHD cycle time and total time to load a truck in hours."""
    # Convert LHD speeds from kph to m/hr
    v_lhd_empty = config.speed_lhd_empty * 1000.0
    v_lhd_loaded = config.speed_lhd_loaded * 1000.0
    
    t_lhd_tram_empty_min = config.lhd_tram_distance / v_lhd_empty * 60.0
    t_lhd_tram_loaded_min = config.lhd_tram_distance / v_lhd_loaded * 60.0
    
    t_lhd_bucket_cycle_min = (
        config.lhd_load_spot_minutes
        + config.lhd_load_minutes
        + t_lhd_tram_loaded_min
        + config.lhd_dump_minutes
        + t_lhd_tram_empty_min
    )
    
    # 2 buckets per truck
    t_load_truck_min = (
        config.lhd_acquisition_delay_minutes
        + config.truck_load_spot_minutes
        + 2 * t_lhd_bucket_cycle_min
    )
    return t_load_truck_min / 60.0

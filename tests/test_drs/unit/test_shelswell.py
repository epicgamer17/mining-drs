import math
import pytest
from drs import DRSEngine
from drs_mining.components.config import ShelswellConfig
from drs_mining.components.shelswell_routing import get_travel_times_hours, get_truck_loading_time_hours
from drs_mining.components.shelswell_controller import ShelswellHaulageModel

def test_routing_times():
    config = ShelswellConfig()
    
    # Verify Level 1 Empty and Loaded travel times are computed
    t_empty_1, t_loaded_1 = get_travel_times_hours(1, "ore", config)
    assert t_empty_1 > 0
    assert t_loaded_1 > 0
    
    # Level 7 is deeper than Level 1, so empty/loaded travel times should be longer
    t_empty_7, t_loaded_7 = get_travel_times_hours(7, "ore", config)
    assert t_empty_7 > t_empty_1
    assert t_loaded_7 > t_loaded_1

def test_loading_time():
    config = ShelswellConfig()
    t_load = get_truck_loading_time_hours("ore", config)
    # LHD acquisition (1.5 min) + truck load spot (0.82 min) + 2 * LHD bucket cycle (2.73627 min)
    # = 1.5 + 0.82 + 5.47254 = 7.79254 min = 0.1298757 hours
    expected = 7.792542 / 60.0
    assert math.isclose(t_load, expected, rel_tol=1e-5)

def test_shelswell_model_simulation():
    config = ShelswellConfig()
    # Use smaller time for unit test
    model = ShelswellHaulageModel(config)
    
    # Override terminate condition for fast test
    model.is_terminating_condition_met = lambda: model.global_time.value >= 5.0
    
    engine = DRSEngine(model)
    engine.run(max_time=5.0)
    
    assert engine.current_time >= 5.0
    assert model.ore_hauled.value > 0.0
    assert model.waste_hauled.value > 0.0

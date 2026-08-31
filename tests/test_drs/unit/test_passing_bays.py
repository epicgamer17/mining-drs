"""Unit tests for discrete RoadSegment, PassingBay, and right-of-way priority (Shelswell 2017)."""

import pytest
import drs
from drs_mining.components.topology import MineTopology, RoadSegment, PassingBay
from drs_mining.components.fleet import Truck, TruckPhase, MissionType


def test_single_truck_free_corridor_traversal():
    """Verify that a single truck traverses clear road segments with zero traffic delay."""
    topo = MineTopology()
    truck = Truck("T01", phase=TruckPhase.EMPTY)
    
    route_down = topo.get_route(target_level=3, direction="DOWN")
    assert len(route_down) > 0
    assert route_down[0].segment_id == "seg_surface"
    assert route_down[0].is_two_way
    
    # Traverse through first decline segment
    decl_seg1 = route_down[1]
    assert decl_seg1.is_free()
    assert decl_seg1.can_enter(truck, "DOWN")
    
    dur = decl_seg1.occupy(truck, "DOWN")
    assert dur > 0.0
    assert decl_seg1.occupant == truck
    assert decl_seg1.direction == "DOWN"
    
    decl_seg1.release(truck)
    assert decl_seg1.is_free()
    assert decl_seg1.occupant is None


def test_loaded_truck_right_of_way_priority():
    """Verify that loaded trucks (heading UP) have strict right-of-way over empty trucks (heading DOWN)."""
    topo = MineTopology()
    t_empty = Truck("T_EMPTY", phase=TruckPhase.EMPTY)
    t_loaded = Truck("T_LOADED", phase=TruckPhase.LOADED)
    
    seg = topo.decline_segments[0]
    
    # 1. When a loaded truck is queued at downstream bay, empty truck must yield
    seg.downstream_bay.queue_truck(t_loaded)
    assert not seg.can_enter(t_empty, "DOWN"), "Empty truck must yield to oncoming loaded truck waiting at downstream bay"
    
    # 2. Loaded truck enters segment with priority
    assert seg.can_enter(t_loaded, "UP")
    seg.occupy(t_loaded, "UP")
    seg.downstream_bay.remove_truck(t_loaded)
    assert seg.occupant == t_loaded
    
    # 3. Empty truck is blocked while loaded truck is inside segment
    assert not seg.can_enter(t_empty, "DOWN")
    
    # 4. Once loaded truck clears segment, empty truck can enter
    seg.release(t_loaded)
    assert seg.can_enter(t_empty, "DOWN")


def test_passing_bay_queue_management():
    """Verify queueing, removal, and state tracking in PassingBay."""
    bay = PassingBay("bay_test")
    t1 = Truck("T1", phase=TruckPhase.EMPTY)
    t2 = Truck("T2", phase=TruckPhase.LOADED)
    
    bay.queue_truck(t1)
    bay.queue_truck(t2)
    assert len(bay.waiting_trucks) == 2
    assert t1.in_passing_bay == bay
    assert t2.in_passing_bay == bay
    
    assert bay.has_waiting_loaded_truck()
    assert bay.get_waiting_loaded_truck() == t2
    assert bay.get_waiting_empty_truck() == t1
    
    bay.remove_truck(t2)
    assert not bay.has_waiting_loaded_truck()
    assert t2.in_passing_bay is None
    assert len(bay.waiting_trucks) == 1

"""Unit tests for discrete fleet components."""

import pytest
import drs
from drs_mining.components.fleet import (
    TruckPhase,
    Operator,
    DESTruck,
    SurfaceDumpStation,
    OPERATING_PHASES,
    SEAT_PHASES,
    DUE_PHASES,
)


def test_truck_phase_sets():
    assert TruckPhase.LOADING in OPERATING_PHASES
    assert TruckPhase.REFUELING in SEAT_PHASES
    assert TruckPhase.DUMPING in DUE_PHASES
    assert TruckPhase.IDLE not in OPERATING_PHASES


def test_operator_and_truck_instantiation():
    op = Operator(idx=0, free=True)
    assert op.idx == 0
    assert op.free

    timer = drs.Timer("t_truck", initial_value=0.0)
    truck = DESTruck(truck_id="T01", timer=timer)
    assert truck.truck_id == "T01"
    assert truck.phase in (TruckPhase.IDLE, TruckPhase.PARKED)
    assert truck.fuel == 100.0


def test_surface_dump_station():
    station = SurfaceDumpStation(name="TIP_1", capacity=2)
    assert station.capacity == 2
    assert station.in_use == 0
    assert len(station.queue) == 0

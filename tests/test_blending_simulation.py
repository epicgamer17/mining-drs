import pytest
from drs import DRSEngine
from examples.blending_modes.simulation import (
    build_blending_network,
    _register_and_policy,
)


def test_blending_simulation_smoke():
    network = build_blending_network(
        total_ore_to_extract=100_000.0,
        ore_to_be_extracted_during_warming_period=0.0,
        target_ore_stock_level=10_000.0,
        critical_ore2_level=3_000.0,
        duration_of_production_campaigns=5.0,
        duration_of_shutdowns=1.0,
    )
    reserve, mill, mode_ctrl, ore1_stock, ore2_stock = network

    engine = DRSEngine()
    cumulative_milled_mass = _register_and_policy(engine, network)

    result = engine.run(until=10.0)
    assert result.steps > 0
    assert result.duration == pytest.approx(10.0)
    assert cumulative_milled_mass.value > 0.0
    assert reserve.cumulative_extracted_mass.value > 0.0

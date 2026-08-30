import math
import pytest
import pandas as pd

from drs_mining.config import MILL_MODES
from examples.shelswell_blending.simulation import (
    ShelswellBlendingHaulage,
    TruckPhase,
    run_shelswell_blending_simulation,
)


def test_shelswell_blending_initialization():
    """Verify clean instantiation of all hybrid components, stockpiles, and loadouts."""
    sim = ShelswellBlendingHaulage(
        num_trucks=6,
        num_operators=6,
        availability=0.85,
        target_ore_stock_level=50000.0,
        seed=123,
    )

    assert len(sim.trucks) == 6
    assert len(sim.operators) == 6
    assert len(sim.loadouts) == 21  # 7 levels * 3 bay types (ORE_1, ORE_2, WASTE)

    # Check active levels (for 6 trucks: k = max(1, 3) -> 3 active levels: L4, L3, L5)
    assert len(sim.active_levels) == 3
    assert 4 in sim.active_levels

    # Check continuous stockpiles initial values
    assert math.isclose(sim.ore1_stock.level, 0.70 * 50000.0, rel_tol=1e-5)
    assert math.isclose(sim.ore2_stock.level, 0.30 * 50000.0, rel_tol=1e-5)
    assert "ORE_1" in sim.dump_sites
    assert "ORE_2" in sim.dump_sites
    assert "WASTE" in sim.dump_sites


def test_daily_target_setting_by_mode():
    """Verify that operating modes set appropriate daily extraction quotas."""
    sim = ShelswellBlendingHaulage(num_trucks=8, num_operators=8, seed=42)

    # Initial state: Mode A
    sim._update_operating_mode_and_targets()
    assert sim.daily_target_ore1 == 3600.0
    assert sim.daily_target_ore2 == 2400.0
    assert sim.daily_target_waste == 500.0

    # Switch to Mode B
    sim.mode_controller.active_campaign_mode.value = MILL_MODES["MODE_B"]
    sim._update_operating_mode_and_targets()
    assert sim.daily_target_ore1 == 4600.0
    assert sim.daily_target_ore2 == 800.0
    assert sim.daily_target_waste == 500.0

    # Switch to Shutdown
    sim.mode_controller.active_campaign_mode.value = MILL_MODES["SHUTDOWN"]
    sim._update_operating_mode_and_targets()
    assert sim.daily_target_ore1 == 0.0
    assert sim.daily_target_ore2 == 0.0
    assert sim.daily_target_waste > 500.0  # Development boost during shutdown


def test_target_deficit_dispatch_priorities():
    """Verify that dispatch rule responds dynamically to remaining daily target deficits."""
    sim = ShelswellBlendingHaulage(num_trucks=6, num_operators=6, seed=42)
    sim.daily_target_ore1 = 3000.0
    sim.daily_target_ore2 = 1000.0
    sim.daily_target_waste = 500.0

    # If Ore 1 target is completely satisfied today, dispatch should heavily favor Ore 2 and Waste
    sim.daily_hauled_ore1 = 3000.0
    sim.daily_hauled_ore2 = 0.0
    sim.daily_hauled_waste = 0.0

    choices = [sim._select_payload_by_target_deficit() for _ in range(100)]
    assert "ORE_1" not in choices  # Ore 1 has zero deficit
    assert "ORE_2" in choices
    assert "WASTE" in choices


def test_short_simulation_run_and_telemetry():
    """Verify execution of a multi-day simulation run and integrity of history telemetry."""
    sim, df = run_shelswell_blending_simulation(
        total_days=5.0,
        num_trucks=6,
        num_operators=6,
        availability=0.90,
        plot=False,
    )

    assert sim.trips > 0
    assert sim.ore1_hauled.value > 0.0
    assert sim.ore2_hauled.value > 0.0
    assert sim.waste_hauled.value > 0.0
    assert sim.plant.cumulative_milled_mass.value > 0.0

    # Check dataframe columns and structure
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = [
        "time",
        "ore1_stock",
        "ore2_stock",
        "total_system_ore_mass",
        "active_operating_mode",
        "daily_target_ore1",
        "daily_target_ore2",
        "daily_hauled_ore1",
        "daily_hauled_ore2",
        "active_trucks",
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify no negative stockpiles
    assert (df["ore1_stock"] >= 0.0).all()
    assert (df["ore2_stock"] >= 0.0).all()


def test_stockpile_conservation_of_mass():
    """Verify mass conservation across continuous stockpiles, discrete haulage, and plant milling."""
    sim = ShelswellBlendingHaulage(
        num_trucks=8,
        num_operators=8,
        availability=0.95,
        target_ore_stock_level=60000.0,
        seed=101,
    )
    init_total_ore = sim.ore1_stock.level + sim.ore2_stock.level

    sim.run(total_days=10.0)

    final_total_ore = sim.ore1_stock.level + sim.ore2_stock.level
    total_hauled = sim.ore1_hauled.value + sim.ore2_hauled.value
    total_milled = sim.plant.cumulative_milled_mass.value

    # Mass balance: Final Stockpile = Initial Stockpile + Hauled - Milled
    expected_final = init_total_ore + total_hauled - total_milled
    assert math.isclose(final_total_ore, expected_final, rel_tol=1e-2)

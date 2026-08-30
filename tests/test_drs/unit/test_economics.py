"""Unit tests for EconomicParameters and MetallurgicalPlant economics."""

import pytest
from drs_mining.config.economics import EconomicParameters
from drs_mining.components.plant import MetallurgicalPlant
from drs_mining.components.stockpiles import Stockpile


def test_economic_parameters_defaults():
    params = EconomicParameters()
    assert params.copper_price_per_lb == 4.00
    assert params.gold_price_per_oz == 1900.0
    assert params.annual_discount_rate == 0.05
    assert params.ore1_cu_grade == 0.007
    assert params.ore2_cu_grade == 0.015


def test_metallurgical_plant_economic_stepping():
    s1 = Stockpile(
        name="s1",
        expected_attributes=["ore_grade"],
        initial_mass=10000.0,
        initial_attributes={"ore_grade": 0.0},
    )
    s2 = Stockpile(
        name="s2",
        expected_attributes=["ore_grade"],
        initial_mass=10000.0,
        initial_attributes={"ore_grade": 1.0},
    )
    plant = MetallurgicalPlant(
        stockpiles=[s1, s2],
        economic_params=EconomicParameters(),
    )

    # Step 1 day with production and development
    # out1 = 3600 t/day = 3600/86400 t/sec, out2 = 2400 t/day = 2400/86400 t/sec
    out1_t_sec = 3600.0 / 86400.0
    out2_t_sec = 2400.0 / 86400.0
    delta_dev = 15.0  # 15 metres
    dt_days = 1.0
    t_days = 1.0

    plant.step_economics(
        out1_t_sec=out1_t_sec,
        out2_t_sec=out2_t_sec,
        delta_dev_meters=delta_dev,
        dt_days=dt_days,
        t_days=t_days,
    )

    assert pytest.approx(plant.cumulative_processed_ore1.value) == 3600.0
    assert pytest.approx(plant.cumulative_processed_ore2.value) == 2400.0
    assert plant.cumulative_gross_revenue.value > 0.0
    assert plant.cumulative_processing_cost.value == 6000.0 * 14.0
    assert plant.cumulative_net_cash_flow.value > 0.0

    # Discount factor D(1 day) = (1.05)^(-1/365)
    expected_df = (1.05) ** (-1.0 / 365.0)
    assert pytest.approx(plant.cumulative_npv.value, rel=1e-5) == (
        plant.cumulative_net_cash_flow.value * expected_df
    )
    assert plant.cash_flow_rate_per_day.value == plant.cumulative_net_cash_flow.value

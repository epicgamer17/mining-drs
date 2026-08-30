"""Economic parameters and commodity pricing configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EconomicParameters:
    """Commodity prices, metallurgical recoveries, and unit cost structure."""

    copper_price_per_lb: float = 4.00
    gold_price_per_oz: float = 1900.0
    ore1_cu_grade: float = 0.007  # 0.70% Cu
    ore1_au_grade_gpt: float = 0.40  # 0.40 g/t Au
    ore2_cu_grade: float = 0.015  # 1.50% Cu
    ore2_au_grade_gpt: float = 1.20  # 1.20 g/t Au
    copper_recovery_ore1: float = 0.88
    gold_recovery_ore1: float = 0.70
    copper_recovery_ore2: float = 0.92
    gold_recovery_ore2: float = 0.80
    milling_cost_per_tonne: float = 14.0
    haulage_cost_per_tonne: float = 4.50
    development_cost_per_metre: float = 4500.0
    annual_discount_rate: float = 0.05

    # Direct net-value pricing mode (optional)
    ore1_net_value_per_processed_tonne: Optional[float] = None
    ore2_net_value_per_processed_tonne: Optional[float] = None
    production_cost_per_tonne: Optional[float] = None
    development_cost_per_unit: Optional[float] = None
    fixed_cost_per_day: Optional[float] = None

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
import math
import drs
from drs import Processor
from .stockpiles import Stockpile
from .modes import OperatingMode
from drs_mining.config import MILL_MODES, EconomicParameters


# TODO: could this be more arbitrary? It seems it forces 2 ore types and 2 overall mode types (A and B) with Contingency and Surging. What if we had one or type? Or 3? Or 3 Modes? or just 1? Can we make it work more arbitrarily.
@dataclass(frozen=True)
class PlantDrawRates:
    """Target mass draw rates for stockpile 1 and stockpile 2 into the plant."""

    ore1: float
    ore2: float

    @property
    def total(self) -> float:
        return self.ore1 + self.ore2


class MetallurgicalPlant(Processor):
    """Represents a metallurgical plant / concentrator processing mined ore.

    Encapsulates plant milling rate setpoints, stockpile starvation detection,
    contingency milling, and surging extraction requirements.
    """

    _MODE_TIMER_ATTRS = {
        "MODE_A": "cumulative_time_mode_a",
        "MODE_A_CONTINGENCY": "cumulative_time_mode_a_contingency",
        "MODE_A_MINE_SURGING": "cumulative_time_mode_a_surging",
        "MODE_B": "cumulative_time_mode_b",
        "MODE_B_CONTINGENCY": "cumulative_time_mode_b_contingency",
        "MODE_B_MINE_SURGING": "cumulative_time_mode_b_surging",
        "SHUTDOWN": "cumulative_time_shutdown",
    }

    _RATE_MAP = {
        "MODE_A": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
        "MODE_A_CONTINGENCY": ("mode_a_contingency_ore1_milling_rate", None),
        "MODE_A_MINE_SURGING": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
        "MODE_B": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
        "MODE_B_CONTINGENCY": (None, "mode_b_contingency_ore2_milling_rate"),
        "MODE_B_MINE_SURGING": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
        "SHUTDOWN": (None, None),
    }

    _CONTINGENCY_MODES = {"MODE_A_CONTINGENCY", "MODE_B_CONTINGENCY"}

    def __init__(
        self,
        stockpiles: Sequence[Stockpile],
        max_rate: float = math.inf,
        name: str = "metallurgical_plant",
        target_ore_stock_level: float = 60000.0,
        duration_of_contingency_segments: float = 2.0 / 24.0,
        mode_a_ore1_milling_rate: float = 540.0 * 24.0,
        mode_a_ore2_milling_rate: float = 60.0 * 24.0,
        mode_a_contingency_ore1_milling_rate: float = 500.0 * 24.0,
        mode_b_ore1_milling_rate: float = 300.0 * 24.0,
        mode_b_ore2_milling_rate: float = 300.0 * 24.0,
        mode_b_contingency_ore2_milling_rate: float = 650.0 * 24.0,
        economic_params: Optional[EconomicParameters] = None,
        copper_price_per_lb: float = 4.00,
        gold_price_per_oz: float = 1900.0,
        ore1_cu_grade: float = 0.015,
        ore1_au_grade_gpt: float = 0.50,
        ore2_cu_grade: float = 0.025,
        ore2_au_grade_gpt: float = 1.20,
        copper_recovery_ore1: float = 0.88,
        gold_recovery_ore1: float = 0.70,
        copper_recovery_ore2: float = 0.92,
        gold_recovery_ore2: float = 0.80,
        milling_cost_per_tonne: float = 14.00,
        annual_discount_rate: float = 0.05,
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.stockpiles = list(stockpiles)
        if len(self.stockpiles) < 2:
            raise ValueError(
                f"MetallurgicalPlant requires at least 2 stockpiles (Ore1 and Ore2), got {len(self.stockpiles)}"
            )
        self.ore1_stock = self.stockpiles[0]
        self.ore2_stock = self.stockpiles[1]

        self.target_ore_stock_level = target_ore_stock_level
        self.duration_of_contingency_segments = duration_of_contingency_segments

        self.mode_a_ore1_milling_rate = mode_a_ore1_milling_rate
        self.mode_a_ore2_milling_rate = mode_a_ore2_milling_rate
        self.mode_a_contingency_ore1_milling_rate = mode_a_contingency_ore1_milling_rate
        self.mode_b_ore1_milling_rate = mode_b_ore1_milling_rate
        self.mode_b_ore2_milling_rate = mode_b_ore2_milling_rate
        self.mode_b_contingency_ore2_milling_rate = mode_b_contingency_ore2_milling_rate

        # Economics & Financial Accounting Parameters
        self.economic_params = economic_params or EconomicParameters(
            copper_price_per_lb=copper_price_per_lb,
            gold_price_per_oz=gold_price_per_oz,
            ore1_cu_grade=ore1_cu_grade,
            ore1_au_grade_gpt=ore1_au_grade_gpt,
            ore2_cu_grade=ore2_cu_grade,
            ore2_au_grade_gpt=ore2_au_grade_gpt,
            copper_recovery_ore1=copper_recovery_ore1,
            gold_recovery_ore1=gold_recovery_ore1,
            copper_recovery_ore2=copper_recovery_ore2,
            gold_recovery_ore2=gold_recovery_ore2,
            milling_cost_per_tonne=milling_cost_per_tonne,
            annual_discount_rate=annual_discount_rate,
        )

        p = self.economic_params
        self.copper_price_per_lb = p.copper_price_per_lb
        self.gold_price_per_oz = p.gold_price_per_oz
        self.ore1_cu_grade = p.ore1_cu_grade
        self.ore1_au_grade_gpt = p.ore1_au_grade_gpt
        self.ore2_cu_grade = p.ore2_cu_grade
        self.ore2_au_grade_gpt = p.ore2_au_grade_gpt
        self.copper_recovery_ore1 = p.copper_recovery_ore1
        self.gold_recovery_ore1 = p.gold_recovery_ore1
        self.copper_recovery_ore2 = p.copper_recovery_ore2
        self.gold_recovery_ore2 = p.gold_recovery_ore2
        self.milling_cost_per_tonne = p.milling_cost_per_tonne
        self.annual_discount_rate = p.annual_discount_rate

        self.cumulative_processed_ore1 = drs.Level(
            "cumulative_processed_ore1", initial_value=0.0
        )
        self.cumulative_processed_ore2 = drs.Level(
            "cumulative_processed_ore2", initial_value=0.0
        )
        self.cumulative_recovered_cu_lbs = drs.Level(
            "cumulative_recovered_cu_lbs", initial_value=0.0
        )
        self.cumulative_recovered_au_oz = drs.Level(
            "cumulative_recovered_au_oz", initial_value=0.0
        )
        self.cumulative_gross_revenue = drs.Level(
            "cumulative_gross_revenue", initial_value=0.0
        )
        self.cumulative_milling_cost = drs.Level(
            "cumulative_milling_cost", initial_value=0.0
        )
        self.cumulative_processing_cost = self.cumulative_milling_cost
        self.cumulative_operating_cost = drs.Level(
            "cumulative_operating_cost", initial_value=0.0
        )
        self.cumulative_net_cash_flow = drs.Level(
            "cumulative_net_cash_flow", initial_value=0.0
        )
        self.cumulative_npv = drs.Level("cumulative_npv", initial_value=0.0)

        # Instantaneous rates ($/day) & discounting
        self.daily_revenue_rate = drs.Variable("daily_revenue_rate", initial_value=0.0)
        self.daily_cost_rate = drs.Variable("daily_cost_rate", initial_value=0.0)
        self.cash_flow_rate_per_day = drs.Variable(
            "cash_flow_rate_per_day", initial_value=0.0
        )
        self.discounted_cash_flow_rate_per_day = drs.Variable(
            "discounted_cash_flow_rate_per_day", initial_value=0.0
        )
        self._discount_factor = drs.Variable("discount_factor", initial_value=1.0)

        self._prev_ore1_processed = 0.0
        self._prev_ore2_processed = 0.0
        self._prev_ore_mined = 0.0
        self._prev_dev = 0.0

        self.cumulative_milled_mass = drs.Level(
            "cumulative_milled_mass", initial_value=0.0
        )
        self.active_operating_mode = drs.Variable(
            "active_operating_mode", MILL_MODES["MODE_A"]
        )
        self.current_contingency_duration = drs.Timer(
            "current_contingency_duration", initial_value=0.0
        )

        self.cumulative_time_mode_a = drs.Timer(
            "cumulative_time_mode_a", initial_value=0.0
        )
        self.cumulative_time_mode_a_contingency = drs.Timer(
            "cumulative_time_mode_a_contingency", initial_value=0.0
        )
        self.cumulative_time_mode_a_surging = drs.Timer(
            "cumulative_time_mode_a_surging", initial_value=0.0
        )
        self.cumulative_time_mode_b = drs.Timer(
            "cumulative_time_mode_b", initial_value=0.0
        )
        self.cumulative_time_mode_b_contingency = drs.Timer(
            "cumulative_time_mode_b_contingency", initial_value=0.0
        )
        self.cumulative_time_mode_b_surging = drs.Timer(
            "cumulative_time_mode_b_surging", initial_value=0.0
        )
        self.cumulative_time_shutdown = drs.Timer(
            "cumulative_time_shutdown", initial_value=0.0
        )
        self.total_system_ore_mass = drs.Level(
            "total_system_ore_mass", initial_value=self.target_ore_stock_level
        )

        self.target_stock1_outflow_rate = drs.Variable(
            "target_stock1_outflow_rate", 0.0
        )
        self.target_stock2_outflow_rate = drs.Variable(
            "target_stock2_outflow_rate", 0.0
        )
        self.target_mine_mass_rate = drs.Variable("target_mine_mass_rate", 0.0)

    @property
    def total_duration(self) -> float:
        """Returns total accumulated duration across all operating modes."""
        return sum(getattr(self, t).value for t in self._MODE_TIMER_ATTRS.values())

    def active_duration(self, current_time: float = -1.0) -> float:
        """Operational duration excluding shutdown time."""
        if current_time < 0.0:
            current_time = self.total_duration
        return max(0.0, current_time - self.cumulative_time_shutdown.value)

    def reset_mode_timers(self) -> None:
        """Reset all operating mode timers (e.g. at the end of the warmup period)."""
        for timer_name in self._MODE_TIMER_ATTRS.values():
            getattr(self, timer_name).reset()

    def get_target_rates(
        self,
        campaign_mode: OperatingMode,
        ore1_level: float,
        ore2_level: float,
        stockpile2_routing_fraction: float = 0.0,
    ) -> Tuple[PlantDrawRates, float]:
        """Determines active operational state (normal, contingency, surging, shutdown)
        from campaign mode and stockpile levels, then computes draw rates and mine feed target.

        Returns:
            Tuple[PlantDrawRates, float]: (plant draw rates for ore1 and ore2, aggregate mine target)
        """
        resolved_mode = self._resolve_operating_mode(
            campaign_mode, ore1_level, ore2_level
        )
        self.active_operating_mode.value = resolved_mode

        mode_name = resolved_mode.name
        self._update_mode_timers(mode_name)

        ore1_rate, ore2_rate = self._read_milling_rates(mode_name)

        if "_MINE_SURGING" in mode_name:
            self.total_system_ore_mass.lower_threshold = self.target_ore_stock_level
            p = stockpile2_routing_fraction
            if p <= 1e-4:
                p = 0.30
            if mode_name == "MODE_A_MINE_SURGING":
                effective_fraction = max(1.0 - p, 0.01)
                mine_target = ore1_rate / effective_fraction
            else:
                effective_fraction = max(p, 0.01)
                mine_target = ore2_rate / effective_fraction
        else:
            self.total_system_ore_mass.lower_threshold = -math.inf
            mine_target = ore1_rate + ore2_rate

        self.total_system_ore_mass.value = ore1_level + ore2_level
        self.total_system_ore_mass.rate = (
            self.ore1_stock.rate + self.ore2_stock.rate
        )

        self.target_stock1_outflow_rate.value = ore1_rate
        self.target_stock2_outflow_rate.value = ore2_rate
        self.target_mine_mass_rate.value = mine_target

        return PlantDrawRates(ore1=ore1_rate, ore2=ore2_rate), mine_target

    determine_operating_mode = get_target_rates

    def _resolve_operating_mode(
        self, campaign_mode: OperatingMode, ore1: float, ore2: float
    ) -> OperatingMode:
        c_name = campaign_mode.name
        if c_name == "SHUTDOWN":
            return MILL_MODES["SHUTDOWN"]

        current_name = self.active_operating_mode.value.name
        eps = 1e-9
        total_stock = ore1 + ore2

        if not current_name.startswith(c_name):
            current_name = c_name

        if "_CONTINGENCY" in current_name:
            if self._contingency_complete():
                self.current_contingency_duration.reset()
                return MILL_MODES[c_name]
            if c_name == "MODE_A" and ore1 <= eps:
                return MILL_MODES["MODE_A_MINE_SURGING"]
            if c_name == "MODE_B" and ore2 <= eps:
                return MILL_MODES["MODE_B_MINE_SURGING"]
            return MILL_MODES[current_name]

        if "_MINE_SURGING" in current_name:
            if c_name == "MODE_A" and ore1 > 500.0:
                return MILL_MODES["MODE_A"]
            elif c_name == "MODE_B" and ore2 > 500.0:
                return MILL_MODES["MODE_B"]
            return MILL_MODES[current_name]

        if c_name == "MODE_A":
            if ore1 <= eps:
                return MILL_MODES["MODE_A_MINE_SURGING"]
            if ore2 <= eps:
                self.current_contingency_duration.reset()
                return MILL_MODES["MODE_A_CONTINGENCY"]
            return MILL_MODES["MODE_A"]

        if c_name == "MODE_B":
            if ore1 <= eps:
                self.current_contingency_duration.reset()
                return MILL_MODES["MODE_B_CONTINGENCY"]
            if ore2 <= eps:
                return MILL_MODES["MODE_B_MINE_SURGING"]
            return MILL_MODES["MODE_B"]

        return MILL_MODES[c_name]


    def _contingency_complete(self) -> bool:
        threshold = self.duration_of_contingency_segments
        self.current_contingency_duration.upper_threshold = threshold
        return self.current_contingency_duration.value >= (threshold - 1e-6)

    def _read_milling_rates(self, name: str) -> Tuple[float, float]:
        ore1_attr, ore2_attr = self._RATE_MAP.get(name, (None, None))
        ore1 = getattr(self, ore1_attr, 0.0) if ore1_attr else 0.0
        ore2 = getattr(self, ore2_attr, 0.0) if ore2_attr else 0.0
        return ore1, ore2

    def _update_mode_timers(self, name: str):
        for timer_name in self._MODE_TIMER_ATTRS.values():
            getattr(self, timer_name).rate = 0.0
        timer_attr = self._MODE_TIMER_ATTRS.get(name)
        if timer_attr:
            getattr(self, timer_attr).rate = 1.0

        if name in self._CONTINGENCY_MODES:
            self.current_contingency_duration.rate = (
                1.0,
                -math.inf,
                self.duration_of_contingency_segments,
            )
        else:
            self.current_contingency_duration.rate = 0.0

    def process(self, mass_rate: float) -> None:
        """Draw ``mass_rate`` into the plant for one engine step."""
        self.rate = mass_rate
        self.cumulative_milled_mass.rate = self.actual_rate

    def step_metallurgical_accounting(
        self, draw_ore1_t: float, draw_ore2_t: float
    ) -> Tuple[float, float, float]:
        """Calculates metal recovery, gross revenue ($), and milling cost ($) from milled ore.
        Returns: (revenue_usd, milling_cost_usd, total_tonnes_milled)
        """
        self.cumulative_processed_ore1.value += draw_ore1_t
        self.cumulative_processed_ore2.value += draw_ore2_t

        # Metal production
        cu_tonnes = (
            draw_ore1_t * self.ore1_cu_grade * self.copper_recovery_ore1
            + draw_ore2_t * self.ore2_cu_grade * self.copper_recovery_ore2
        )
        cu_lbs = cu_tonnes * 2204.62
        au_grams = (
            draw_ore1_t * self.ore1_au_grade_gpt * self.gold_recovery_ore1
            + draw_ore2_t * self.ore2_au_grade_gpt * self.gold_recovery_ore2
        )
        au_oz = au_grams / 31.1035

        self.cumulative_recovered_cu_lbs.value += cu_lbs
        self.cumulative_recovered_au_oz.value += au_oz

        revenue = cu_lbs * self.copper_price_per_lb + au_oz * self.gold_price_per_oz
        milling_cost = (draw_ore1_t + draw_ore2_t) * self.milling_cost_per_tonne

        self.cumulative_gross_revenue.value += revenue
        self.cumulative_milling_cost.value += milling_cost

        return revenue, milling_cost, (draw_ore1_t + draw_ore2_t)

    @property
    def daily_revenue(self) -> float:
        return self.daily_revenue_rate.value

    @property
    def daily_cost(self) -> float:
        return self.daily_cost_rate.value

    @property
    def daily_net_cash_flow(self) -> float:
        return self.cash_flow_rate_per_day.value

    @property
    def discount_factor(self) -> float:
        return self._discount_factor.value

    @property
    def cumulative_cash_flow(self) -> float:
        return self.cumulative_net_cash_flow.value

    def step_daily_economics(
        self,
        current_day: float,
        ore1_mined_t: float = 0.0,
        ore2_mined_t: float = 0.0,
        development_units: float = 0.0,
    ):
        """Calculates daily revenues, costs, net cash flows, and discounted NPV."""
        p = self.economic_params
        p_o1 = float(self.cumulative_processed_ore1.value)
        p_o2 = float(self.cumulative_processed_ore2.value)

        d_ore1_proc = max(0.0, p_o1 - self._prev_ore1_processed)
        d_ore2_proc = max(0.0, p_o2 - self._prev_ore2_processed)
        tot_mined = ore1_mined_t + ore2_mined_t
        d_mined = max(0.0, tot_mined - self._prev_ore_mined)
        d_dev = max(0.0, development_units - self._prev_dev)

        self._prev_ore1_processed = p_o1
        self._prev_ore2_processed = p_o2
        self._prev_ore_mined = tot_mined
        self._prev_dev = development_units

        if p.ore1_net_value_per_processed_tonne is not None:
            rev = (
                d_ore1_proc * p.ore1_net_value_per_processed_tonne
                + d_ore2_proc * (p.ore2_net_value_per_processed_tonne or 0.0)
            )
            prod_cost = d_mined * (p.production_cost_per_tonne or 0.0)
            dev_cost = d_dev * (p.development_cost_per_unit or 0.0)
            fixed_cost = p.fixed_cost_per_day or 0.0
            tot_cost = prod_cost + dev_cost + fixed_cost
            net_cf = rev - tot_cost
        else:
            cu_lbs = (
                d_ore1_proc * p.ore1_cu_grade * p.copper_recovery_ore1
                + d_ore2_proc * p.ore2_cu_grade * p.copper_recovery_ore2
            ) * 2204.62
            au_oz = (
                d_ore1_proc * p.ore1_au_grade_gpt * p.gold_recovery_ore1
                + d_ore2_proc * p.ore2_au_grade_gpt * p.gold_recovery_ore2
            ) / 31.1035
            rev = cu_lbs * p.copper_price_per_lb + au_oz * p.gold_price_per_oz
            mill_cost = (d_ore1_proc + d_ore2_proc) * p.milling_cost_per_tonne
            haul_cost = (d_ore1_proc + d_ore2_proc) * p.haulage_cost_per_tonne
            dev_cost = d_dev * p.development_cost_per_metre
            tot_cost = mill_cost + haul_cost + dev_cost
            net_cf = rev - tot_cost

        dfactor = (1.0 + p.annual_discount_rate) ** (-max(0.0, current_day / 365.0))
        self._discount_factor.value = dfactor
        discounted_cf = net_cf * dfactor

        self.daily_revenue_rate.value = rev
        self.daily_cost_rate.value = tot_cost
        self.cash_flow_rate_per_day.value = net_cf
        self.discounted_cash_flow_rate_per_day.value = discounted_cf

        self.cumulative_operating_cost.value += tot_cost
        self.cumulative_net_cash_flow.value += net_cf
        self.cumulative_npv.value += discounted_cf

    def step_economics(
        self,
        out1_t_sec: float,
        out2_t_sec: float,
        delta_dev_meters: float,
        dt_days: float,
        t_days: float,
    ):
        """Continuous timestep update for revenue, opex, capex, and NPV."""
        p = self.economic_params
        dt_sec = dt_days * 86400.0

        ore1_t = out1_t_sec * dt_sec
        ore2_t = out2_t_sec * dt_sec
        self.cumulative_processed_ore1.value += ore1_t
        self.cumulative_processed_ore2.value += ore2_t

        cu_recovered_lbs = (
            ore1_t * p.ore1_cu_grade * p.copper_recovery_ore1
            + ore2_t * p.ore2_cu_grade * p.copper_recovery_ore2
        ) * 2204.62

        au_recovered_oz = (
            ore1_t * p.ore1_au_grade_gpt * p.gold_recovery_ore1
            + ore2_t * p.ore2_au_grade_gpt * p.gold_recovery_ore2
        ) / 31.1035

        revenue = (
            cu_recovered_lbs * p.copper_price_per_lb
            + au_recovered_oz * p.gold_price_per_oz
        )
        milling_cost = (ore1_t + ore2_t) * p.milling_cost_per_tonne
        haulage_cost = (ore1_t + ore2_t) * p.haulage_cost_per_tonne
        dev_capex = delta_dev_meters * p.development_cost_per_metre
        operating_cost = milling_cost + haulage_cost
        net_cash_flow = revenue - operating_cost - dev_capex

        t_years = max(0.0, t_days / 365.0)
        discount_factor = (1.0 + p.annual_discount_rate) ** (-t_years)
        discounted_cash_flow = net_cash_flow * discount_factor

        self.cumulative_gross_revenue.value += revenue
        self.cumulative_milling_cost.value += milling_cost
        self.cumulative_operating_cost.value += operating_cost
        self.cumulative_net_cash_flow.value += net_cash_flow
        self.cumulative_npv.value += discounted_cash_flow

        if dt_days > 1e-12:
            self.cash_flow_rate_per_day.value = net_cash_flow / dt_days
            self.discounted_cash_flow_rate_per_day.value = discounted_cash_flow / dt_days
        else:
            self.cash_flow_rate_per_day.value = 0.0
            self.discounted_cash_flow_rate_per_day.value = 0.0



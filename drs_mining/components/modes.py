from typing import Optional, Union
import drs
from .data import TargetRates


class RequireDecision(Exception):
    pass


_MODE_IDS = {
    "MODE_A": 0,
    "MODE_A_CONTINGENCY": 1,
    "MODE_A_MINE_SURGING": 2,
    "MODE_B": 3,
    "MODE_B_CONTINGENCY": 4,
    "MODE_B_MINE_SURGING": 5,
    "SHUTDOWN": 6,
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

DEFAULT_RATES = {
    "mode_a_ore1_milling_rate": 3600.0,
    "mode_a_ore2_milling_rate": 2400.0,
    "mode_a_contingency_ore1_milling_rate": 3900.0,
    "mode_b_ore1_milling_rate": 4600.0,
    "mode_b_ore2_milling_rate": 800.0,
    "mode_b_contingency_ore2_milling_rate": 2500.0,
}


def _read_rates(name, obj):
    ore1_attr, ore2_attr = _RATE_MAP.get(name, (None, None))
    ore1 = (
        getattr(obj, ore1_attr, DEFAULT_RATES.get(ore1_attr, 0.0))
        if ore1_attr
        else 0.0
    )
    ore2 = (
        getattr(obj, ore2_attr, DEFAULT_RATES.get(ore2_attr, 0.0))
        if ore2_attr
        else 0.0
    )
    return ore1, ore2


class OperatingMode:
    __slots__ = ("_name", "_id")

    def __init__(self, name: str):
        self._name = name
        self._id = _MODE_IDS[name]

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, OperatingMode):
            return self._id == other._id
        return NotImplemented

    def __hash__(self):
        return hash(self._id)

    def __repr__(self):
        return f"OperatingMode({self._name})"

    def get_target_rates(self, model) -> TargetRates:
        ctrl = model.controller
        ore1, ore2 = _read_rates(self._name, ctrl)

        if "_MINE_SURGING" in self._name:
            target_stock = getattr(ctrl, "target_ore_stock_level", 60000.0)
            model.controller.total_system_ore_mass.lower_threshold = target_stock
            p = model.stockpile2_routing_fraction
            if p <= 1e-4 and hasattr(model, "mine") and hasattr(model.mine, "_get_current_attr_value"):
                p = model.mine._get_current_attr_value()
            if self._name in ("MODE_A_MINE_SURGING"):
                effective_fraction = max(1.0 - p, 0.01)
                raw_extraction = ore1 / effective_fraction
            else:
                effective_fraction = max(p, 0.01)
                raw_extraction = ore2 / effective_fraction
            extraction = raw_extraction
            return TargetRates(
                extraction_rate=extraction,
                ore1_milling_rate=ore1,
                ore2_milling_rate=ore2,
            )

        extraction = ore1 + ore2
        return TargetRates(
            extraction_rate=extraction, ore1_milling_rate=ore1, ore2_milling_rate=ore2
        )

    def check_end_conditions(
        self, model
    ) -> Union[Optional["OperatingMode"], RequireDecision]:
        ctrl = model.controller
        n = self._name

        if ctrl.is_campaign_complete():
            return RequireDecision()

        if n == "SHUTDOWN":
            return None

        stockout_epsilon = getattr(ctrl, "stockout_epsilon", 1e-9)
        target_stock = getattr(ctrl, "target_ore_stock_level", 60000.0)

        ore1 = model.ore1_mass
        ore2 = model.ore2_mass

        if "_CONTINGENCY" in n:
            if ctrl.is_contingency_complete():
                return RequireDecision()
            base = n.replace("_CONTINGENCY", "")
            if base == "MODE_A" and ore1 <= stockout_epsilon:
                return MODES[base + "_MINE_SURGING"]
            if base == "MODE_B" and ore2 <= stockout_epsilon:
                return MODES[base + "_MINE_SURGING"]
            return None

        if "_MINE_SURGING" in n:
            if (
                model.controller.total_system_ore_mass.value
                <= target_stock + 1e-6
            ):
                return RequireDecision()
            return None

        if n == "MODE_A":
            if ore1 <= stockout_epsilon:
                return MODES[n + "_MINE_SURGING"]
            if ore2 <= stockout_epsilon:
                ctrl.reset_contingency_timer()
                return MODES[n + "_CONTINGENCY"]
            return None

        if n == "MODE_B":
            if ore1 <= stockout_epsilon:
                ctrl.reset_contingency_timer()
                return MODES[n + "_CONTINGENCY"]
            if ore2 <= stockout_epsilon:
                return MODES[n + "_MINE_SURGING"]
            return None

        return None


MODES = {name: OperatingMode(name) for name in _MODE_IDS}

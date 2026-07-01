from typing import Optional, Union
from drs.module import drs
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
    "MODE_C": 7,
    "MODE_C_CONTINGENCY": 8,
    "MODE_C_MINE_SURGING": 9,
    "MODE_D": 10,
    "MODE_D_CONTINGENCY": 11,
    "MODE_D_MINE_SURGING": 12,
}


_RATE_MAP = {
    "MODE_A": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
    "MODE_A_CONTINGENCY": ("mode_a_contingency_ore1_milling_rate", None),
    "MODE_A_MINE_SURGING": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
    "MODE_B": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
    "MODE_B_CONTINGENCY": (None, "mode_b_contingency_ore2_milling_rate"),
    "MODE_B_MINE_SURGING": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
    "MODE_C": ("mode_c_ore1_milling_rate", "mode_c_ore2_milling_rate"),
    "MODE_C_CONTINGENCY": ("mode_c_contingency_ore1_milling_rate", None),
    "MODE_C_MINE_SURGING": ("mode_c_ore1_milling_rate", "mode_c_ore2_milling_rate"),
    "MODE_D": ("mode_d_ore1_milling_rate", "mode_d_ore2_milling_rate"),
    "MODE_D_CONTINGENCY": (None, "mode_d_contingency_ore2_milling_rate"),
    "MODE_D_MINE_SURGING": ("mode_d_ore1_milling_rate", "mode_d_ore2_milling_rate"),
    "SHUTDOWN": (None, None),
}


def _read_rates(name, config):
    ore1_attr, ore2_attr = _RATE_MAP.get(name, (None, None))
    ore1 = getattr(config, ore1_attr, 0.0) if ore1_attr else 0.0
    ore2 = getattr(config, ore2_attr, 0.0) if ore2_attr else 0.0
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

    def is_valid_start(self, model) -> bool:
        n = self._name
        ore2 = model.ore2_stock.current_mass.value
        if n in ("MODE_A", "MODE_C"):
            return ore2 >= model.plant.config.critical_ore2_level
        if n in ("MODE_B", "MODE_D"):
            return ore2 < model.plant.config.critical_ore2_level
        return True

    def get_target_rates(self, model) -> TargetRates:
        config = model.plant.config
        ore1, ore2 = _read_rates(self._name, config)

        if "_MINE_SURGING" in self._name:
            model.controller.total_system_ore_mass.lower_threshold = (
                config.target_ore_stock_level
            )
            p = model.fleet.stockpile2_routing_fraction.value
            if self._name in ("MODE_A_MINE_SURGING", "MODE_C_MINE_SURGING"):
                effective_fraction = max(
                    1.0 - p, config.min_effective_surging_fraction
                )
                raw_extraction = ore1 / effective_fraction
            else:
                effective_fraction = max(p, config.min_effective_surging_fraction)
                raw_extraction = ore2 / effective_fraction
            extraction = min(raw_extraction, config.max_surging_extraction_rate)
            return TargetRates(
                extraction_rate=extraction,
                ore1_milling_rate=ore1,
                ore2_milling_rate=ore2,
            )

        extraction = min(ore1 + ore2, config.max_mine_extraction_rate)
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

        config = ctrl.config
        ore1 = model.ore1_stock.current_mass.value
        ore2 = model.ore2_stock.current_mass.value

        if "_CONTINGENCY" in n:
            if ctrl.is_contingency_complete():
                return RequireDecision()
            base = n.replace("_CONTINGENCY", "")
            if base in ("MODE_A", "MODE_C") and ore1 <= config.stockout_epsilon:
                return MODES[base + "_MINE_SURGING"]
            if base in ("MODE_B", "MODE_D") and ore2 <= config.stockout_epsilon:
                return MODES[base + "_MINE_SURGING"]
            return None

        if "_MINE_SURGING" in n:
            base = n.replace("_MINE_SURGING", "")
            if base == "MODE_C" and ore2 <= config.stockout_epsilon:
                ctrl.reset_contingency_timer()
                return MODES[base + "_CONTINGENCY"]
            if base == "MODE_D" and ore1 <= config.stockout_epsilon:
                ctrl.reset_contingency_timer()
                return MODES[base + "_CONTINGENCY"]

            if (
                model.controller.total_system_ore_mass.value
                <= config.target_ore_stock_level + 1e-6
            ):
                return RequireDecision()
            return None

        if n in ("MODE_A", "MODE_C"):
            if ore1 <= config.stockout_epsilon:
                return MODES[n + "_MINE_SURGING"]
            if ore2 <= config.stockout_epsilon:
                ctrl.reset_contingency_timer()
                return MODES[n + "_CONTINGENCY"]
            return None

        if n in ("MODE_B", "MODE_D"):
            if ore1 <= config.stockout_epsilon:
                ctrl.reset_contingency_timer()
                return MODES[n + "_CONTINGENCY"]
            if ore2 <= config.stockout_epsilon:
                return MODES[n + "_MINE_SURGING"]
            return None

        return None


MODES = {name: OperatingMode(name) for name in _MODE_IDS}

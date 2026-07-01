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


def _read_rates(name, config):
    ore1_attr, ore2_attr = _RATE_MAP.get(name, (None, None))
    ore1 = getattr(config, ore1_attr, 0.0) if ore1_attr else 0.0
    ore2 = getattr(config, ore2_attr, 0.0) if ore2_attr else 0.0
    return ore1, ore2


# TODO: maybe introduce the concept of a Plant or OperatingConfiguration, which consists of many modes, and automatic transitions between those modes, and then the controller decides a configuration. Instead of a Mode directly.


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
        config = model.plant.config
        ore1, ore2 = _read_rates(self._name, config)

        if "_MINE_SURGING" in self._name:
            model.controller.total_system_ore_mass.lower_threshold = (
                config.target_ore_stock_level
            )
            p = model.fleet.stockpile2_routing_fraction.value
            if self._name == "MODE_A_MINE_SURGING":
                extraction = (ore1 / (1.0 - p)) if (1.0 - p) > 0 else 0.0
            else:
                extraction = (ore2 / p) if p > 0 else 0.0
            return TargetRates(
                extraction_rate=extraction,
                ore1_milling_rate=ore1,
                ore2_milling_rate=ore2,
            )

        return TargetRates(
            extraction_rate=ore1 + ore2, ore1_milling_rate=ore1, ore2_milling_rate=ore2
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
                # TODO: should the controller be the one deciding here? I think these are artifacts as a mode being kind of a configuration and mode at the same time, and our controller dealing with the configuration part (ie when to end surging or when to end contingency)
                return RequireDecision()
            base = n.replace("_CONTINGENCY", "")
            if base == "MODE_A" and ore1 <= config.stockout_epsilon:
                return MODES[base + "_MINE_SURGING"]
            if base == "MODE_B" and ore2 <= config.stockout_epsilon:
                return MODES[base + "_MINE_SURGING"]
            return None

        if "_MINE_SURGING" in n:
            if (
                model.controller.total_system_ore_mass.value
                <= config.target_ore_stock_level + 1e-6
            ):
                # TODO: why do we do this? I think these are artifacts as a mode being kind of a configuration and mode at the same time, and our controller dealing with the configuration part (ie when to end surging or when to end contingency)
                return RequireDecision()
            return None

        if n == "MODE_A":
            if ore1 <= config.stockout_epsilon:
                return MODES[n + "_MINE_SURGING"]
            if ore2 <= config.stockout_epsilon:
                ctrl.reset_contingency_timer()
                return MODES[n + "_CONTINGENCY"]
            return None

        if n == "MODE_B":
            if ore1 <= config.stockout_epsilon:
                ctrl.reset_contingency_timer()
                return MODES[n + "_CONTINGENCY"]
            if ore2 <= config.stockout_epsilon:
                return MODES[n + "_MINE_SURGING"]
            return None

        return None


MODES = {name: OperatingMode(name) for name in _MODE_IDS}

from dataclasses import dataclass


@dataclass
class TargetRates:
    """Requested production rates returned by the Controller."""

    extraction_rate: float
    ore1_milling_rate: float
    ore2_milling_rate: float


@dataclass
class MineOutput:
    """Physical ore output from a Mine Face, consumed by Fleet for routing."""

    extraction_rate: float
    attr_value: float




"""Configuration definitions and standard presets for Mill and Fleet Operating Modes."""

from dataclasses import dataclass
from typing import Dict, Optional
from drs_mining.components.modes import OperatingMode


@dataclass(frozen=True)
class MillModeConfig:
    name: str
    id: int
    ore1_draw_rate: Optional[float] = None
    ore2_draw_rate: Optional[float] = None
    description: str = ""


@dataclass(frozen=True)
class FleetModeConfig:
    name: str
    id: int
    dev_reservation_fraction: float = 0.0
    area2_dev_share: float = 0.50
    description: str = ""


# 1. Default Mill Mode Configs (Processing Plant Campaigns)
MILL_MODE_CONFIGS: Dict[str, MillModeConfig] = {
    "MODE_A": MillModeConfig(
        name="MODE_A",
        id=0,
        ore1_draw_rate=3600.0,
        ore2_draw_rate=2400.0,
        description="High Ore 2 draw campaign",
    ),
    "MODE_A_CONTINGENCY": MillModeConfig(
        name="MODE_A_CONTINGENCY",
        id=1,
        ore1_draw_rate=3900.0,
        ore2_draw_rate=0.0,
        description="Ore 1 only contingency draw",
    ),
    "MODE_A_MINE_SURGING": MillModeConfig(
        name="MODE_A_MINE_SURGING",
        id=2,
        description="Mine surging reduced haulage target",
    ),
    "MODE_B": MillModeConfig(
        name="MODE_B",
        id=3,
        ore1_draw_rate=4600.0,
        ore2_draw_rate=800.0,
        description="Low Ore 2 draw campaign",
    ),
    "MODE_B_CONTINGENCY": MillModeConfig(
        name="MODE_B_CONTINGENCY",
        id=4,
        ore1_draw_rate=0.0,
        ore2_draw_rate=2500.0,
        description="Ore 2 only contingency draw",
    ),
    "MODE_B_MINE_SURGING": MillModeConfig(
        name="MODE_B_MINE_SURGING",
        id=5,
        description="Mine surging reduced haulage target",
    ),
    "SHUTDOWN": MillModeConfig(
        name="SHUTDOWN",
        id=6,
        ore1_draw_rate=0.0,
        ore2_draw_rate=0.0,
        description="Plant maintenance shutdown",
    ),
}

# 2. Default Fleet Mode Configs (Mine Tactical & Development Priorities)
FLEET_MODE_CONFIGS: Dict[str, FleetModeConfig] = {
    "PRODUCTION": FleetModeConfig(
        name="PRODUCTION",
        id=0,
        dev_reservation_fraction=0.0,
        area2_dev_share=0.35,
        description="Maximize ore haulage throughput; surplus to stope development",
    ),
    "DEVELOPMENT": FleetModeConfig(
        name="DEVELOPMENT",
        id=1,
        dev_reservation_fraction=0.20,
        area2_dev_share=0.85,
        description="Prioritize capital decline development for surplus trucks",
    ),
}

# 3. Active Mode Instances
MILL_MODES: Dict[str, OperatingMode] = {
    name: OperatingMode(name, id=cfg.id, category="mill", description=cfg.description)
    for name, cfg in MILL_MODE_CONFIGS.items()
}

FLEET_MODES: Dict[str, OperatingMode] = {
    name: OperatingMode(
        name,
        id=cfg.id,
        category="fleet",
        dev_reservation_fraction=cfg.dev_reservation_fraction,
        area2_dev_share=cfg.area2_dev_share,
        description=cfg.description,
    )
    for name, cfg in FLEET_MODE_CONFIGS.items()
}

# 4. Standard Policy 2 Mill-to-Fleet Mode Mapping Table
POLICY_2_FLEET_MODE_MAP: Dict[str, str] = {
    "MODE_A": "PRODUCTION",
    "MODE_A_CONTINGENCY": "PRODUCTION",
    "MODE_A_MINE_SURGING": "PRODUCTION",
    "MODE_B": "DEVELOPMENT",
    "MODE_B_CONTINGENCY": "PRODUCTION",
    "MODE_B_MINE_SURGING": "PRODUCTION",
    "SHUTDOWN": "PRODUCTION",
}


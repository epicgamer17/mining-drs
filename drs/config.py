from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DRSConfig:
    """Base configuration class for DRS Modules.

    Subclass this using @dataclass to define strictly-typed configuration
    blocks for your models, allowing IDE autocomplete and clear parameterization.
    """

    pass


@dataclass
class EngineConfig(DRSConfig):
    """Configuration for the DRS Engine."""

    max_step_size: float = float("inf")
    max_deadlock_steps: int = 20
    max_time: float = None
    strict_mode: bool = False

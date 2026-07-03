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

    max_step_size: float = (
        0.5  # TODO: should change this to float("inf") but for parity dont, it seems to cause behaviour changes, theory is that its because of precision propogation over longer time horizons. Why do we even need a max step size?
    )
    max_deadlock_steps: int = 20
    max_time: float = None
    strict_mode: bool = False

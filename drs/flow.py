from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from .module import Module


@dataclass
class Flow:
    """
    A unified data structure for tracking physical flows between modules.
    
    Flows represent the movement of continuous quantities (e.g. mass, energy)
    between components in the simulation.
    """
    value: Any
    _source: Optional["Module"] = None

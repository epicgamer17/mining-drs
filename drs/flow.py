from __future__ import annotations
from typing import TYPE_CHECKING, Any, Optional
from dataclasses import dataclass

if TYPE_CHECKING:
    from .module import Module


@dataclass
class Flow:
    """A unified data structure for tracking physical flows between modules.
    
    Flows represent the movement of continuous quantities (e.g., volume, data, energy)
    between components in the simulation. When returned from a module's `forward()`
    pass, the engine automatically tracks the edge between the producing and consuming modules.

    Attributes:
        value (Any): The underlying quantity or value of the flow (often a float).
        _source (Optional[Module]): [INTERNAL] The module that generated this flow. 
            Automatically populated by the engine.
    """
    value: Any
    _source: Optional["Module"] = None

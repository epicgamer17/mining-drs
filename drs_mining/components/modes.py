"""Generic domain abstractions for discrete operating modes."""

from __future__ import annotations

from typing import Any, Dict, Iterator, Mapping, Optional, Sequence
import drs


class RequireDecision(Exception):
    """Raised when an operating policy requires an external decision."""

    pass


class OperatingMode(drs.Module):
    """Represents a discrete operating mode for plant, fleet, or whole-mine systems.

    Inherits from drs.Module to automatically integrate and track its own cumulative
    time in use via an internal drs.Timer.
    """

    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        category: str = "general",
        draw_rates: Optional[Mapping[str, float]] = None,
        **metadata: Any,
    ):
        super().__init__()
        self._name = str(name)
        self._category = str(category)
        self._metadata = dict(metadata)
        self._draw_rates = dict(draw_rates or {})
        self._id = (
            id if id is not None else (hash((self._category, self._name)) & 0x7FFFFFFF)
        )
        self.timer = drs.Timer(
            f"cumulative_time_{self._name.lower()}", initial_value=0.0
        )
        self.timer.rate = 0.0

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def draw_rates(self) -> Dict[str, float]:
        return self._draw_rates

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def cumulative_time(self) -> float:
        """Total time accumulated while this mode was active."""
        return self.timer.value

    def activate(self) -> None:
        """Activate mode and begin integrating cumulative time."""
        self.timer.rate = 1.0

    def deactivate(self) -> None:
        """Deactivate mode and pause integrating cumulative time."""
        self.timer.rate = 0.0

    def reset_timer(self) -> None:
        """Reset cumulative timer to 0.0."""
        self.timer.reset()

    def levels(self) -> Sequence[drs.Level]:
        return (self.timer,)

    def variables(self) -> Iterator[drs.Variable]:
        yield self.timer

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OperatingMode):
            return self._name == other._name and self._category == other._category
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._category, self._name))

    def __repr__(self) -> str:
        if self._category != "general":
            return f"OperatingMode({self._name}, category='{self._category}', time={self.timer.value:.1f})"
        return f"OperatingMode({self._name}, time={self.timer.value:.1f})"

    def __str__(self) -> str:
        return self._name

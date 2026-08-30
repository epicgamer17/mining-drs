"""Generic domain abstractions for discrete operating modes."""

from typing import Optional, Dict, Any


class RequireDecision(Exception):
    """Raised when an operating mode controller requires an external decision."""

    pass


class OperatingMode:
    """Represents a discrete operating mode for plant, fleet, or whole-mine systems."""

    __slots__ = ("_name", "_id", "_category", "_metadata")

    def __init__(
        self,
        name: str,
        id: Optional[int] = None,
        category: str = "general",
        **metadata: Any,
    ):
        self._name = str(name)
        self._category = str(category)
        self._metadata = metadata
        self._id = (
            id if id is not None else (hash((self._category, self._name)) & 0x7FFFFFFF)
        )

    @property
    def id(self) -> int:
        return self._id

    @property
    def value(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OperatingMode):
            return self._name == other._name and self._category == other._category
        if hasattr(other, "name") and hasattr(other, "category"):
            return (
                self._name == getattr(other, "name")
                and self._category == getattr(other, "category")
            )
        if isinstance(other, str):
            return self._name == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._category, self._name))

    def __repr__(self) -> str:
        if self._category != "general":
            return f"OperatingMode({self._name}, category='{self._category}')"
        return f"OperatingMode({self._name})"

    def __str__(self) -> str:
        return self._name

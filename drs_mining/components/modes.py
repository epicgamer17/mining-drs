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


MODES = {name: OperatingMode(name) for name in _MODE_IDS}

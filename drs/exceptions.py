class DRSError(Exception):
    """Base class for all DRS framework exceptions."""

    pass


class StateMutationError(DRSError):
    """Raised when a module attempts to illegally mutate state."""

    pass


class DeadlockError(DRSError):
    """Raised when the engine fails to advance time."""

    def __init__(self, message: str, state_dump: str = ""):
        super().__init__(message)
        self.state_dump = state_dump


class ThresholdConfigurationError(DRSError):
    """Raised when a threshold is configured but cannot be reached."""

    pass

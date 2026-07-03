class DRSError(Exception):
    """Base class for all DRS framework exceptions."""

    pass


# TODO: is this mining specific sort of? is there a better name than physics violation that is more general for non mining simulations
class StateMutationError(DRSError):
    """Raised when a module attempts to illegally mutate state."""

    pass


class DeadlockError(DRSError):
    """Raised when the engine fails to advance time."""

    def __init__(self, message: str, state_dump: str = ""):
        super().__init__(message)
        self.state_dump = state_dump

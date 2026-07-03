class DRSError(Exception):
    """Base class for all DRS framework exceptions."""

    pass


# TODO: is this mining specific sort of? is there a better name than physics violation that is more general for non mining simulations
class StateMutationError(DRSError):
    """Raised when a module attempts to illegally mutate state."""

    pass


class DeadlockError(DRSError):
    """Raised when the engine fails to advance time."""

    pass

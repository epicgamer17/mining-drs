class DRSError(Exception):
    """Base class for all DRS framework exceptions."""

    pass


class StateMutationError(DRSError):
    """Raised when a module attempts to illegally mutate state."""

    def __init__(self, message: str):
        # Capture the call stack from ExecutionContext
        from ._execution_context import ExecutionContext
        stack = getattr(ExecutionContext._local, "stack", [])
        if stack:
            stack_str = " -> ".join([type(mod).__name__ for mod in stack])
            self.message = f"{message}\n\nModule Call Stack:\n{stack_str}"
        else:
            self.message = message
        super().__init__(self.message)


class DeadlockError(DRSError):
    """Raised when the engine fails to advance time."""

    def __init__(self, message: str, state_dump: str = ""):
        super().__init__(message)
        self.state_dump = state_dump


class ThresholdConfigurationError(DRSError):
    """Raised when a threshold is configured but cannot be reached."""

    pass

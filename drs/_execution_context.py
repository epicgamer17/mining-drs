import threading


class ExecutionContext:
    """
    [INTERNAL] Thread-local context tracking active modules and engines during evaluation.

    Power User Note: Enforces state ownership and builds the dependency graph implicitly
    as the simulation steps.
    """

    _local = threading.local()

    @classmethod
    def push(cls, module):
        """
        [INTERNAL] Push a module onto the active execution stack.

        Power User Note: Used when evaluating a module's forward pass to record active scope.
        """
        if not hasattr(cls._local, "stack"):
            cls._local.stack = []
            cls._local.flow_edges = []
        cls._local.stack.append(module)

    @classmethod
    def set_tracing(cls, enabled: bool = True):
        """
        [INTERNAL] Toggle symbolic tracing mode.

        Power User Note: Enables AST expression recording instead of eager evaluation.
        """
        cls._local.tracing = enabled

    @classmethod
    def is_tracing(cls) -> bool:
        """
        [INTERNAL] Check if symbolic tracing mode is active.

        Power User Note: Used by Variable to decide whether to return expressions.
        """
        return getattr(cls._local, "tracing", False)

    @classmethod
    def pop(cls):
        """
        [INTERNAL] Pop the top module from the execution stack.

        Power User Note: Restores the caller's context after a module's execution completes.
        """
        cls._local.stack.pop()

    @classmethod
    def get_current(cls):
        """
        [INTERNAL] Retrieve the currently executing module.

        Power User Note: Returns the module at the top of the stack, or None if executing externally.
        """
        stack = getattr(cls._local, "stack", [])
        return stack[-1] if stack else None

    @classmethod
    def set_engine(cls, engine):
        """
        [INTERNAL] Bind the current DRSEngine to the context.

        Power User Note: Used to provide modules access to simulation clock and configs.
        """
        cls._local.engine = engine

    @classmethod
    def get_engine(cls):
        """
        [INTERNAL] Retrieve the currently active DRSEngine.

        Power User Note: Accesses thread-local storage to retrieve the running engine instance.
        """
        return getattr(cls._local, "engine", None)

    @classmethod
    def record_flow_edge(cls, source, target):
        """
        [INTERNAL] Record a transient flow edge between a source and target module.
        """
        if source is not None:
            if not hasattr(cls._local, "flow_edges"):
                cls._local.flow_edges = []
            cls._local.flow_edges.append((source, target))

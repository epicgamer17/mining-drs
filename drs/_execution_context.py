import threading


class ExecutionContext:
    _local = threading.local()

    @classmethod
    def push(cls, module):
        if not hasattr(cls._local, "stack"):
            cls._local.stack = []
            cls._local.flow_edges = []
        cls._local.stack.append(module)

    @classmethod
    def pop(cls):
        cls._local.stack.pop()

    @classmethod
    def get_current(cls):
        stack = getattr(cls._local, "stack", [])
        return stack[-1] if stack else None

    @classmethod
    def set_engine(cls, engine):
        cls._local.engine = engine
    
    @classmethod
    def get_engine(cls):
        return getattr(cls._local, 'engine', None)



    @classmethod
    def record_flow_edge(cls, source, target):
        if source is not None:
            if not hasattr(cls._local, "flow_edges"):
                cls._local.flow_edges = []
            cls._local.flow_edges.append((source, target))

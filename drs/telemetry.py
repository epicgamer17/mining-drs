import pandas as pd
from typing import Callable, Dict, Any, Optional
from .module import Module


class Telemetry:
    """Automates the recording of all simulation variables over time.
    
    Provides methods to export the recorded history into analysis-ready formats.

    NOTE: this is probably what we should be using to "make" our observations. 
    The telemetry data or sensor data in a way. For MDP just track all variables. 
    For POMDP only some of them.
    
    Attributes:
        model (Module): The root module being tracked.
        history (list[dict]): The recorded history of states.
        tracked_vars (Optional[list[str]]): The names of variables being tracked.
        snapshot_condition (Optional[Callable[[float], bool]]): Condition to take a snapshot.
        on_snapshot (Optional[Callable[[dict[str, Any]], None]]): Callback on snapshot.
        group (Optional[str]): Categorize telemetry channels.
        derived_metrics (Dict[str, Callable]): Custom metrics calculated at each step.
    """

    def __init__(
        self, 
        model: Module,
        tracked_vars: Optional[list[str]] = None,
        snapshot_condition: Optional[Callable[[float], bool]] = None,
        on_snapshot: Optional[Callable[[dict[str, Any]], None]] = None,
        group: Optional[str] = None
    ) -> None:
        """
        Initializes the telemetry system attached to a specific model.
        
        Args:
            model (Module): The root Module of your simulation.
            tracked_vars (Optional[list[str]]): Filter which variables to track. If None, track all.
            snapshot_condition (Optional[Callable[[float], bool]]): Lambda returning bool, snapshot only if True.
            on_snapshot (Optional[Callable[[dict[str, Any]], None]]): Streaming hook for live dashboards.
            group (Optional[str]): Categorize telemetry channels (e.g. "sensors").
        """
        self.model = model
        self.history: list[dict[str, Any]] = []
        self.tracked_vars = tracked_vars
        self.snapshot_condition = snapshot_condition
        self.on_snapshot = on_snapshot
        self.group = group
        self.derived_metrics: Dict[str, Callable[[float, Module, dict[str, Any], list[dict[str, Any]]], float]] = {}

    def register_metric(
        self, name: str, calc_fn: Callable[[float, Module, dict, list], float]
    ):
        """Register a custom metric calculated dynamically at each time step.
        
        Useful for tracking derived metrics like NPV, utilization, or efficiency.

        Args:
            name (str): The name of the metric.
            calc_fn (Callable): The metric function. 
                Signature: `calc_fn(current_time, model, state, history) -> float`
        """
        self.derived_metrics[name] = calc_fn

    def snapshot(self, current_time: float):
        """
        Called automatically at the end of every simulation tick to record the state.

        Args:
            current_time (float): The current simulation time.
        """
        if self.snapshot_condition is not None and not self.snapshot_condition(current_time):
            return

        state = {"time": current_time}

        for variable in self.model.variables():
            if self.tracked_vars is None or variable.name in self.tracked_vars:
                state[variable.name] = variable.value

        for name, func in self.derived_metrics.items():
            state[name] = func(current_time, self.model, state, self.history)

        self.history.append(state)

        if self.on_snapshot is not None:
            self.on_snapshot(state)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the entire simulation history into a Pandas DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame where each row is a time step and columns 
                are tracked variables and derived metrics.
        """
        return pd.DataFrame(self.history)

import pandas as pd
from typing import Callable, Dict, Any
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
        tracked_vars (list[str]): The names of variables being tracked.
        derived_metrics (Dict[str, Callable]): Custom metrics calculated at each step.
    """

    def __init__(self, model: Module) -> None:
        """
        Initializes the telemetry system attached to a specific model.
        
        Args:
            model (Module): The root Module of your simulation. The model is 
                expected to provide a `variables()` method yielding Variable objects.
        """
        self.model = model
        self.history: list[dict[str, Any]] = []
        self.tracked_vars: list[str] = [
            var.name for var in self.model.variables()
        ]  # default to all variables
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
        state = {"time": current_time}

        for variable in self.model.variables():
            if variable.name in self.tracked_vars:
                state[variable.name] = variable.value

        for name, func in self.derived_metrics.items():
            state[name] = func(current_time, self.model, state, self.history)

        self.history.append(state)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the entire simulation history into a Pandas DataFrame.
        
        Returns:
            pd.DataFrame: A DataFrame where each row is a time step and columns 
                are tracked variables and derived metrics.
        """
        return pd.DataFrame(self.history)

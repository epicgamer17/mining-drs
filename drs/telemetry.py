import pandas as pd
from typing import Callable, Dict, Any
from .module import Module


class Telemetry:
    """
    Automates the recording of all simulation variables over time.
    Provides methods to export the recorded history into analysis-ready formats.

    NOTE: this is probably what we should be using to "make" our observations. The telemetry data or sensor data in a way. For MDP just track all variables. For POMDP only some of them.
    """

    def __init__(self, model):
        """
        Initializes the telemetry system attached to a specific model.
        The model is expected to provide a `variables()` method (an iterator of Variable objects).
        """
        self.model = model
        self.history = []
        self.tracked_vars = [
            var.name for var in self.model.variables()
        ]  # default to all variables
        self.derived_metrics: Dict[str, Callable] = {}

    def register_metric(
        self, name: str, calc_fn: Callable[[float, Module, dict, list], float]
    ):
        """Register a custom metric. For things like NPV

        Args:
            name (str): The name of the metric.
            calc_fn (Callable): The metric function. Signature: calc_fn(current_time, model, state, history) -> metric_value
        """
        self.derived_metrics[name] = calc_fn

    def snapshot(self, current_time: float):
        """
        Called automatically at the end of every simulation tick to record the state.
        """
        state = {"time": current_time}

        for variable in self.model.variables():
            if variable.name in self.tracked_vars:
                state[variable.name] = variable.value

        for name, func in self.derived_metrics.items():
            state[name] = func(current_time, self.model, state, self.history)

        self.history.append(state)

    def to_dataframe(self):
        """
        Converts the entire simulation history into a Pandas DataFrame for plotting/analysis.
        """
        return pd.DataFrame(self.history)

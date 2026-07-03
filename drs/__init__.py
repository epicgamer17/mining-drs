import logging

__version__ = "0.1.0"

# Configure a NullHandler to prevent "No handler found" warnings
# Users of the library can configure their own logging handlers
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .module import Module, DataSource
from .engine import DRSEngine, SimulationResult
from .variables import Variable, Level, Timer
from .data_source import DataPoint
from .flow import Flow
from .telemetry import Telemetry
from .exceptions import StateMutationError, DeadlockError
from .config import DRSConfig, EngineConfig
from .callbacks import Callback, ProgressBarCallback
from .serialize import save_state, load_state, export_architecture

# Provide friendly aliases for commonly used classes
Engine = DRSEngine

__all__ = [
    "DRSEngine", 
    "SimulationResult",
    "Callback",
    "ProgressBarCallback",
    "Engine", 
    "Variable",
    "Level",
    "Timer",
    "DataPoint", 
    "DataSource",
    "Module",
    "Flow",
    "Telemetry",
    "StateMutationError",
    "DeadlockError",
    "DRSConfig",
    "EngineConfig",
    "save_state",
    "load_state",
    "export_architecture"
]

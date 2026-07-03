import logging

__version__ = "0.1.0"

# Configure a NullHandler to prevent "No handler found" warnings
# Users of the library can configure their own logging handlers
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .module import Module, DataSource
from .engine import DRSEngine
from .variables import Variable, Level, Timer
from .data_source import DataPoint
from .flow import Flow
from .telemetry import Telemetry
from .exceptions import StateMutationError, DeadlockError
from .config import DRSConfig, EngineConfig

# Provide friendly aliases for commonly used classes
Engine = DRSEngine

__all__ = [
    "DRSEngine", 
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
    "EngineConfig"
]

from drs.canvas_compiler import register_class

from .config import ConcentratorConfig
from .modes import MODES, OperatingMode, RequireDecision
from .generators import StochasticFaciesGenerator
from .mine_face import ConcentratorMineFace, ContinuousMineFace
from .plant import ConcentratorPlant
from .controllers import ConcentratorController, MultiFaceConcentratorController
from .models import (
    ActiveFleetConcentratorModel,
    BaseBlendingModel,
    ConcentratorModel,
)
from .stockpiles import Stockpile

for _cls in [
    ConcentratorConfig,
    ConcentratorMineFace,
    ContinuousMineFace,
    ConcentratorPlant,
    ConcentratorController,
    MultiFaceConcentratorController,
    ActiveFleetConcentratorModel,
    BaseBlendingModel,
    ConcentratorModel,
    Stockpile,
    OperatingMode,
]:
    register_class(_cls)

__all__ = [
    "BaseBlendingModel",
    "ConcentratorModel",
    "ActiveFleetConcentratorModel",
    "MultiFaceConcentratorController",
    "ContinuousMineFace",
    "Stockpile",
]

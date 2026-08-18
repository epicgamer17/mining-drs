import math
import random
from typing import Optional, Any
import drs
from drs import Processor
from .generators import StochasticFaciesGenerator


class MineFace(Processor):
    """Represents a mine face source with stochastic parcel geology and mass tracking."""

    def __init__(
        self,
        name: Optional[str] = None,
        face_id: Optional[int] = None,
        generator: Optional[Any] = None,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        max_rate: float = math.inf,
        initial_parcel_mass: Optional[float] = None,
    ):
        face_name = name or (f"mine_face_{face_id}" if face_id is not None else "mine_face")
        super().__init__(name=face_name, max_rate=max_rate)
        self.face_id = face_id
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies
        self.min_ore_mass = min_ore_mass
        self.max_ore_mass = max_ore_mass
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = ore_to_be_extracted_during_warming_period

        if generator is not None:
            self.generator = generator
        else:
            self.generator = StochasticFaciesGenerator(
                mean_fraction=self.mean_ore_fraction,
                std_dev=self.std_dev_ore_fraction,
                prob_new_facies=self.prob_new_facies,
                variation_same_facies=self.variation_same_facies,
            )

        var_name = f"face{face_id}_ore_fraction" if face_id is not None else "active_parcel_ore_fraction"
        self.active_parcel_ore_fraction = drs.Variable(
            var_name, self.mean_ore_fraction
        )

        init_mass = initial_parcel_mass if initial_parcel_mass is not None else 40000.0
        self.active_parcel_initial_mass = drs.Variable(
            "active_parcel_initial_mass", init_mass
        )

        self.cumulative_extracted_mass = drs.Level(
            "cumulative_extracted_mass", initial_value=0.0
        )
        self.parcel_extracted_mass = drs.Level(
            "parcel_extracted_mass", initial_value=0.0
        )

    @property
    def net_extracted_mass(self) -> float:
        """Encapsulate internal math inside the module."""
        return (
            self.cumulative_extracted_mass.value
            - self.ore_to_be_extracted_during_warming_period
        )

    def is_terminating_condition_met(self) -> bool:
        return (
            self.cumulative_extracted_mass.value >= self.total_ore_to_extract
        )

    def _load_next_batch(self):
        try:
            self.active_parcel_initial_mass.value = random.uniform(
                self.min_ore_mass, self.max_ore_mass
            )
            if hasattr(self.generator, "generate_next"):
                parcel = self.generator.generate_next()
            elif callable(self.generator):
                parcel = self.generator()
            else:
                parcel = next(self.generator)

            if hasattr(parcel, "value"):
                parcel = parcel.value
            if isinstance(parcel, dict):
                ore1_frac = parcel["ore1_frac"]
            elif hasattr(parcel, "ore1_frac"):
                ore1_frac = parcel.ore1_frac
            else:
                ore1_frac = float(parcel)

            self.active_parcel_ore_fraction.value = ore1_frac
        except (StopIteration, TypeError):
            pass


    def _get_current_attr_value(self) -> float:
        return self.active_parcel_ore_fraction.value

    @property
    def current_ore_grade(self) -> float:
        """Attribute value (ore fraction) of the currently active parcel."""
        return self._get_current_attr_value()

    def advance_parcel_state(self):
        """Advance parcel mechanics: cross parcel boundaries and set level thresholds."""
        if (
            self.parcel_extracted_mass.value
            >= self.active_parcel_initial_mass.value - 1e-6
        ):
            self._load_next_batch()
            self.parcel_extracted_mass.value = 0.0
            self.parcel_extracted_mass.upper_threshold = (
                self.active_parcel_initial_mass.value
            )

        if (
            self.cumulative_extracted_mass.value
            < self.ore_to_be_extracted_during_warming_period
        ):
            self.cumulative_extracted_mass.upper_threshold = (
                self.ore_to_be_extracted_during_warming_period
            )
        else:
            self.cumulative_extracted_mass.upper_threshold = (
                self.total_ore_to_extract
            )

        self.parcel_extracted_mass.upper_threshold = (
            self.active_parcel_initial_mass.value
        )

    def step(self, dt: float) -> None:
        """Apply the face's local mechanics for one engine step."""
        self.advance_parcel_state()
        actual = self.actual_rate
        self.cumulative_extracted_mass.rate = actual
        self.parcel_extracted_mass.rate = actual
        super().step(dt)

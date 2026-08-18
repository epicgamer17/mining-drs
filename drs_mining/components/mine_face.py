import math
import random
import drs
from drs import Processor
from .generators import StochasticFaciesGenerator


class MineFace(Processor):
    """Represents a mine face source with stochastic parcel geology and mass tracking."""

    def __init__(
        self,
        name: str,
        face_id: int,
        generator: StochasticFaciesGenerator,
        min_ore_mass: float,
        max_ore_mass: float,
        total_ore_to_extract: float,
        ore_to_be_extracted_during_warming_period: float,
        mean_ore_fraction: float,
        std_dev_ore_fraction: float,
        prob_new_facies: float,
        variation_same_facies: float,
        initial_parcel_mass: float,
        max_rate: float = math.inf,
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.face_id = face_id
        self.generator = generator
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies
        self.min_ore_mass = min_ore_mass
        self.max_ore_mass = max_ore_mass
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = ore_to_be_extracted_during_warming_period

        var_name = f"face{face_id}_ore_fraction" if face_id != 1 else "active_parcel_ore_fraction"
        self.active_parcel_ore_fraction = drs.Variable(
            var_name, self.mean_ore_fraction
        )

        self.active_parcel_initial_mass = drs.Variable(
            "active_parcel_initial_mass", initial_parcel_mass
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
        self.active_parcel_initial_mass.value = random.uniform(
            self.min_ore_mass, self.max_ore_mass
        )
        parcel = self.generator.generate_next()
        if isinstance(parcel, dict):
            self.active_parcel_ore_fraction.value = float(parcel["ore1_frac"])
        elif hasattr(parcel, "ore1_frac"):
            self.active_parcel_ore_fraction.value = float(parcel.ore1_frac)
        elif hasattr(parcel, "value"):
            self.active_parcel_ore_fraction.value = float(parcel.value)
        else:
            self.active_parcel_ore_fraction.value = float(parcel)

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

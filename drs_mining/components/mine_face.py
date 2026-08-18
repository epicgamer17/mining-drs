import math
import random
import drs
from drs import Processor
from .generators import StochasticFaciesGenerator


class BaseMineFace(Processor):
    def __init__(
        self,
        name: str = "mine_face",
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        max_rate: float = math.inf,
    ):
        super().__init__(name=name, max_rate=max_rate)
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
        )

        self.active_parcel_initial_mass = drs.Variable(
            "active_parcel_initial_mass", 0.0
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
        raise NotImplementedError("Subclasses must define how to parse the generator output.")

    def _get_current_attr_value(self) -> float:
        raise NotImplementedError("Subclasses must define current ore attribute value.")

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

    @property
    def current_ore_grade(self) -> float:
        """Attribute value (ore fraction) of the currently active parcel."""
        return self._get_current_attr_value()

    def step(self, dt: float) -> None:
        """Apply the face's local mechanics for one engine step.

        Policies drive ``target_rate``; this step advances parcel state (cross
        boundaries and load the next batch), stamps the cumulative and parcel
        extraction rates with the realised rate, then advances the owned levels
        by ``dt``.
        """
        self.advance_parcel_state()
        actual = self.actual_rate
        self.cumulative_extracted_mass.rate = actual
        self.parcel_extracted_mass.rate = actual
        super().step(dt)


class ConcentratorMineFace(BaseMineFace):
    def __init__(
        self,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        max_rate: float = math.inf,
        name: str = "concentrator_mine_face",
    ):
        super().__init__(
            name=name,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            max_rate=max_rate,
        )
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies
        self.min_ore_mass = min_ore_mass
        self.max_ore_mass = max_ore_mass

        self.generator = StochasticFaciesGenerator(
            mean_fraction=self.mean_ore_fraction,
            std_dev=self.std_dev_ore_fraction,
            prob_new_facies=self.prob_new_facies,
            variation_same_facies=self.variation_same_facies,
        )
        self.active_parcel_ore_fraction = drs.Variable(
            "active_parcel_ore_fraction", self.mean_ore_fraction
        )
        self.active_parcel_initial_mass.value = 40000.0

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


class ContinuousMineFace(BaseMineFace):
    def __init__(
        self,
        face_id: int,
        generator,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        max_rate: float = math.inf,
    ):
        super().__init__(
            name=f"mine_face_{face_id}",
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            max_rate=max_rate,
        )
        self.face_id = face_id
        self.generator = generator
        self.min_ore_mass = min_ore_mass
        self.max_ore_mass = max_ore_mass
        self.active_parcel_ore_fraction = drs.Variable(
            f"face{face_id}_ore_fraction", 0.0
        )
        self._load_next_batch()

    def _load_next_batch(self):
        try:
            if hasattr(self.generator, "generate_next"):
                parcel = self.generator.generate_next()
            elif callable(self.generator):
                parcel = self.generator()
            else:
                parcel = next(self.generator)
            if hasattr(parcel, "value"):
                parcel = parcel.value
            self.active_parcel_initial_mass.value = random.uniform(
                self.min_ore_mass, self.max_ore_mass
            )
            if isinstance(parcel, dict):
                ore1_frac = parcel["ore1_frac"]
            elif hasattr(parcel, "ore1_frac"):
                ore1_frac = parcel.ore1_frac
            else:
                ore1_frac = float(parcel)
            self.active_parcel_ore_fraction.value = 1.0 - ore1_frac
        except (StopIteration, TypeError):
            pass

    def _get_current_attr_value(self) -> float:
        return self.active_parcel_ore_fraction.value



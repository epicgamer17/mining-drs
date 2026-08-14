import random
import drs
from drs.flow import Flow
from .data import MineOutput
from .generators import StochasticFaciesGenerator


class BaseMineFace(drs.Module):
    def __init__(
        self,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
    ):
        super().__init__()
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

    def _load_next_batch(self):
        raise NotImplementedError("Subclasses must define how to parse the DataPoint.")

    def _get_current_attr_value(self) -> float:
        raise NotImplementedError("Subclasses must define current ore attribute value.")

    def forward(self, target_rate=None):
        if target_rate is not None:
            target_extraction_rate = (
                target_rate.value if hasattr(target_rate, "value") else target_rate
            )
        elif self.parent is not None and hasattr(self.parent, "target_mine_mass_rate"):
            target_extraction_rate = self.parent.target_mine_mass_rate
        else:
            target_extraction_rate = 0.0

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

        self.cumulative_extracted_mass.rate = target_extraction_rate
        self.parcel_extracted_mass.rate = target_extraction_rate
        return Flow(
            value=MineOutput(
                extraction_rate=target_extraction_rate,
                attr_value=self._get_current_attr_value(),
            )
        )


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
    ):
        super().__init__(
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
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
            "active_parcel_ore_fraction", 0.0
        )
        self._load_next_batch()

    def _load_next_batch(self):
        try:
            parcel_flow = self.generator()
            parcel = parcel_flow.value

            self.active_parcel_initial_mass.value = random.uniform(
                self.min_ore_mass, self.max_ore_mass
            )
            self.active_parcel_ore_fraction.value = 1.0 - parcel.ore1_frac
        except StopIteration:
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
    ):
        super().__init__(
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
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
            parcel_flow = self.generator()
            parcel = parcel_flow.value
            self.active_parcel_initial_mass.value = random.uniform(
                self.min_ore_mass, self.max_ore_mass
            )
            self.active_parcel_ore_fraction.value = 1.0 - parcel.ore1_frac
        except StopIteration:
            pass

    def _get_current_attr_value(self) -> float:
        return self.active_parcel_ore_fraction.value

    def forward(self, target_rate=None):
        if target_rate is not None:
            target_extraction_rate = target_rate.value
        else:
            target_extraction_rate = 0.0

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

        self.cumulative_extracted_mass.rate = target_extraction_rate
        self.parcel_extracted_mass.rate = target_extraction_rate

        return Flow(
            value=MineOutput(
                extraction_rate=target_extraction_rate,
                attr_value=self.active_parcel_ore_fraction.value,
            )
        )

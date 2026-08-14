import random
import drs
from drs.flow import Flow


class StochasticFaciesGenerator(drs.Module):
    """
    Generates autocorrelated ore fractions using a facies model.
    Decoupled from physical mass generation.
    """
    def __init__(
        self, 
        mean_fraction: float, 
        std_dev: float, 
        prob_new_facies: float = 0.3, 
        variation_same_facies: float = 0.05
    ):
        super().__init__()
        self.mean_fraction = mean_fraction
        self.std_dev = std_dev
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies
        
        self.next_is_new_facies = True
        self.current_fraction = mean_fraction

    def forward(self):
        return Flow(value=next(self))

    def __next__(self) -> dict[str, float]:
        if self.next_is_new_facies:
            if self.std_dev != 0:
                fraction = random.gauss(self.mean_fraction, self.std_dev)
            else:
                fraction = self.mean_fraction
        else:
            fraction = self.current_fraction + self.variation_same_facies * random.uniform(-1, 1)

        # Clip fraction cleanly between 0 and 1
        self.current_fraction = max(0.0, min(1.0, fraction))
        self.next_is_new_facies = random.random() <= self.prob_new_facies

        return {"ore1_frac": self.current_fraction}


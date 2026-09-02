"""Stochastic quality and facies generators for geological reserve models."""

from __future__ import annotations

import random
from typing import Dict

import drs


# TODO: should python-drs introduce a Source component? Similar to Storage and Processor? What would be the benefit?
class StochasticFaciesGenerator(drs.Module):
    """Generates autocorrelated quality fractions using a facies model.

    Decoupled from physical mass generation.
    """

    def __init__(
        self,
        mean_fraction: float,
        std_dev: float,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.05,
        attribute_name: str = "ore2_fraction",
    ):
        super().__init__()
        self.mean_fraction = float(mean_fraction)
        self.std_dev = float(std_dev)
        self.prob_new_facies = float(prob_new_facies)
        self.variation_same_facies = float(variation_same_facies)
        self.attribute_name = str(attribute_name)

        self.next_is_new_facies = True
        self.current_fraction = self.mean_fraction

    def generate_next(self) -> Dict[str, float]:
        return next(self)

    def __next__(self) -> Dict[str, float]:
        if self.next_is_new_facies:
            if self.std_dev != 0:
                fraction = random.gauss(self.mean_fraction, self.std_dev)
            else:
                fraction = self.mean_fraction
        else:
            fraction = (
                self.current_fraction
                + self.variation_same_facies * random.uniform(-1, 1)
            )

        self.current_fraction = max(0.0, min(1.0, fraction))
        self.next_is_new_facies = random.random() <= self.prob_new_facies

        return {self.attribute_name: self.current_fraction}

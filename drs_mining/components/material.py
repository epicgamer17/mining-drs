"""Material stream and discrete entity primitives for mining simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Flow:
    """Continuous material flow with a mass flow rate and arbitrary quality attributes.

    Attributes represent quality concentrations or fractions, such as:
    - Element grades: {"Cu": 0.65, "Au": 1.2}
    - Facies or ore fractions: {"ore2_fraction": 0.3}
    - Physical qualities: {"moisture": 0.05, "hardness": 14.2}
    """

    rate: float
    attributes: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Entity:
    """Discrete packet or batch of material (e.g. truckload, blast parcel, mining block).

    Attributes represent quality concentrations or fractions over the discrete mass.
    """

    mass: float
    attributes: Mapping[str, float] = field(default_factory=dict)


def blend_flows(flows: Sequence[Flow]) -> Flow:
    """Blend multiple incoming material flows into a single unified flow.

    Applies exact conservation of mass and mass-weighted attribute averaging.
    """
    total_rate = sum(f.rate for f in flows)
    if total_rate <= 1e-9:
        return Flow(rate=0.0, attributes={})

    # Gather all unique attributes across all flows
    all_attrs = set()
    for f in flows:
        all_attrs.update(f.attributes.keys())

    blended_attrs: dict[str, float] = {}
    for attr in all_attrs:
        attr_mass_rate = sum(f.rate * f.attributes.get(attr, 0.0) for f in flows)
        blended_attrs[attr] = attr_mass_rate / total_rate

    return Flow(rate=total_rate, attributes=blended_attrs)


def split_flow(flow: Flow, fractions: Mapping[str, float]) -> dict[str, Flow]:
    """Split a material flow into multiple named streams based on split fractions.

    The split fractions should sum to 1.0. Quality attributes remain identical
    across all split streams.
    """
    return {
        name: Flow(rate=flow.rate * frac, attributes=dict(flow.attributes))
        for name, frac in fractions.items()
    }

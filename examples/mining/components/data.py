from dataclasses import dataclass

@dataclass
class TargetRates:
    """Requested production rates returned by the Controller."""

    extraction_rate: float
    ore1_milling_rate: float
    ore2_milling_rate: float


@dataclass
class MineOutput:
    """Physical ore output from a Mine Face, consumed by Fleet for routing."""

    extraction_rate: float
    parcel_mass: float
    attr_value: float


@dataclass
class OreParcel:
    """Discrete material parcel moved through underground handling/haulage."""

    source_face: int
    mass: float
    ore1_fraction: float
    dispatch_time: float
    arrival_time: float


@dataclass
class BlastOutput:
    """Discrete blasted material from a geological parcel."""

    mass: float
    ore1_fraction: float


@dataclass
class LHDLoadingRates:
    """Aggregate LHD material movement rates for one underground face."""

    remuck_to_truck_rate: float
    face_to_truck_rate: float
    face_to_remuck_rate: float

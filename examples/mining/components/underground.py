import random

from drs.flow import Flow
from drs.module import drs
from .data import BlastOutput, LHDLoadingRates, OreParcel


def _rate_value(rate_source) -> float:
    if rate_source is None:
        return 0.0
    if isinstance(rate_source, (int, float)):
        return rate_source
    if isinstance(rate_source, Flow):
        return rate_source.value
    return rate_source.value


def _flow_value(value_source):
    if value_source is None:
        return None
    if isinstance(value_source, Flow):
        return value_source.value
    return value_source.value


def _loaded_rate_value(rate_source) -> float:
    value = _flow_value(rate_source)
    if value is None:
        return 0.0
    if isinstance(value, LHDLoadingRates):
        return value.remuck_to_truck_rate + value.face_to_truck_rate
    return value


def _flag_value(flag_source) -> bool:
    value = _flow_value(flag_source)
    return bool(value)


def _module_time(module) -> float:
    current = module
    while current is not None:
        if hasattr(current, "global_time"):
            return current.global_time.value
        current = getattr(current, "parent", None)
    return 0.0


def _sample_triangular_hours_as_days(params) -> float:
    low, mode, high = params
    return random.triangular(low, high, mode) / 24.0


class UndergroundGeologyFace(drs.Module):
    """Geological parcel source consumed by drill/blast events."""

    def __init__(
        self,
        config,
        face_id: int,
        generator,
    ):
        super().__init__()
        self.config = config
        self.face_id = face_id
        self.generator = generator
        self.active_parcel_initial_mass = drs.Variable(
            f"face{face_id}_active_parcel_initial_mass",
            0.0,
        )
        self.active_parcel_remaining_mass = drs.Level(
            f"face{face_id}_active_parcel_remaining_mass",
            0.0,
        )
        self.active_parcel_ore_fraction = drs.Variable(
            f"face{face_id}_active_parcel_ore_fraction",
            0.0,
        )
        self.cumulative_blasted_mass = drs.Level(
            f"face{face_id}_cumulative_blasted_mass",
            0.0,
        )
        self._load_next_parcel()

    def _load_next_parcel(self):
        parcel_flow = self.generator()
        parcel = parcel_flow.value
        parcel_mass = random.uniform(self.config.min_ore_mass, self.config.max_ore_mass)
        self.active_parcel_initial_mass.value = parcel_mass
        self.active_parcel_remaining_mass.value = parcel_mass
        self.active_parcel_ore_fraction.value = 1.0 - parcel.ore1_frac

    def forward(self, requested_blast_mass=None):
        requested = _rate_value(requested_blast_mass)
        if requested <= 0.0:
            return Flow(value=BlastOutput(mass=0.0, ore1_fraction=0.0))

        if self.active_parcel_remaining_mass.value <= 1e-6:
            self._load_next_parcel()

        blast_mass = min(requested, self.active_parcel_remaining_mass.value)
        ore1_fraction = self.active_parcel_ore_fraction.value

        self.active_parcel_remaining_mass.value -= blast_mass
        self.cumulative_blasted_mass.value += blast_mass

        if self.active_parcel_remaining_mass.value <= 1e-6:
            self._load_next_parcel()

        return Flow(value=BlastOutput(mass=blast_mass, ore1_fraction=ore1_fraction))


class UndergroundFaceInventory(drs.Module):
    """Material inventories at an underground face before surface haulage."""

    def __init__(
        self,
        face_id: int,
        initial_blasted_rock_mass: float = 0.0,
        initial_remuck_mass: float = 0.0,
        initial_ore1_fraction: float = 0.0,
    ):
        super().__init__()
        self.face_id = face_id
        self.blasted_rock_mass = drs.Level(
            f"face{face_id}_blasted_rock_mass",
            initial_value=initial_blasted_rock_mass,
        )
        self.remuck_mass = drs.Level(
            f"face{face_id}_remuck_mass",
            initial_value=initial_remuck_mass,
        )
        self.blasted_ore1_mass = drs.Level(
            f"face{face_id}_blasted_ore1_mass",
            initial_value=initial_blasted_rock_mass * initial_ore1_fraction,
        )
        self.remuck_ore1_mass = drs.Level(
            f"face{face_id}_remuck_ore1_mass",
            initial_value=initial_remuck_mass * initial_ore1_fraction,
        )
        self.ore1_fraction = drs.Variable(
            f"face{face_id}_ore1_fraction",
            initial_ore1_fraction,
        )
        self.loaded_ore1_fraction = drs.Variable(
            f"face{face_id}_loaded_ore1_fraction",
            initial_ore1_fraction,
        )

    def _blasted_ore1_fraction(self) -> float:
        return self.blasted_ore1_mass.value / max(1e-6, self.blasted_rock_mass.value)

    def _remuck_ore1_fraction(self) -> float:
        return self.remuck_ore1_mass.value / max(1e-6, self.remuck_mass.value)

    def forward(
        self,
        blasted_inflow_rate=None,
        blast_output=None,
        blast_mass=None,
        lhd_loading_rates=None,
        lhd_to_remuck_rate=None,
        haulage_outflow_rate=None,
    ):
        blasted_inflow = _rate_value(blasted_inflow_rate)
        blast = _flow_value(blast_output)
        if isinstance(blast, BlastOutput):
            blasted_mass_addition = blast.mass
            if blasted_mass_addition > 0.0:
                self.ore1_fraction.value = blast.ore1_fraction
                self.blasted_ore1_mass.value += blast.mass * blast.ore1_fraction
        else:
            blasted_mass_addition = _rate_value(blast_mass)
        if blasted_mass_addition > 0.0:
            self.blasted_rock_mass.value += blasted_mass_addition

        loading_rates = _flow_value(lhd_loading_rates)
        blasted_ore1_fraction = self._blasted_ore1_fraction()
        remuck_ore1_fraction = self._remuck_ore1_fraction()

        if loading_rates is not None:
            face_to_truck_rate = loading_rates.face_to_truck_rate
            face_to_remuck_rate = loading_rates.face_to_remuck_rate
            remuck_to_truck_rate = loading_rates.remuck_to_truck_rate

            if self.blasted_rock_mass.value <= 1e-6:
                face_to_truck_rate = 0.0
                face_to_remuck_rate = min(face_to_remuck_rate, blasted_inflow)
            if self.remuck_mass.value <= 1e-6:
                remuck_to_truck_rate = min(remuck_to_truck_rate, face_to_remuck_rate)

            self.blasted_rock_mass.rate = (
                blasted_inflow - face_to_truck_rate - face_to_remuck_rate
            )
            self.remuck_mass.rate = face_to_remuck_rate - remuck_to_truck_rate
            self.blasted_ore1_mass.rate = (
                blasted_inflow * self.ore1_fraction.value
                - (face_to_truck_rate + face_to_remuck_rate) * blasted_ore1_fraction
            )
            self.remuck_ore1_mass.rate = (
                face_to_remuck_rate * blasted_ore1_fraction
                - remuck_to_truck_rate * remuck_ore1_fraction
            )
            loaded_rate = face_to_truck_rate + remuck_to_truck_rate
            if loaded_rate > 1e-6:
                self.loaded_ore1_fraction.value = (
                    face_to_truck_rate * blasted_ore1_fraction
                    + remuck_to_truck_rate * remuck_ore1_fraction
                ) / loaded_rate
        else:
            requested_lhd_rate = _rate_value(lhd_to_remuck_rate)
            requested_haulage_rate = _rate_value(haulage_outflow_rate)

            lhd_rate = requested_lhd_rate
            if self.blasted_rock_mass.value <= 1e-6:
                lhd_rate = min(lhd_rate, blasted_inflow)

            haulage_rate = requested_haulage_rate
            if self.remuck_mass.value <= 1e-6:
                haulage_rate = min(haulage_rate, lhd_rate)

            self.blasted_rock_mass.rate = blasted_inflow - lhd_rate
            self.remuck_mass.rate = lhd_rate - haulage_rate
            self.blasted_ore1_mass.rate = (
                blasted_inflow * self.ore1_fraction.value
                - lhd_rate * blasted_ore1_fraction
            )
            self.remuck_ore1_mass.rate = (
                lhd_rate * blasted_ore1_fraction
                - haulage_rate * remuck_ore1_fraction
            )
            if haulage_rate > 1e-6:
                self.loaded_ore1_fraction.value = remuck_ore1_fraction

        if self.blasted_rock_mass.rate < 0:
            self.blasted_rock_mass.lower_threshold = 0.0
            self.blasted_ore1_mass.lower_threshold = 0.0
        if self.remuck_mass.rate < 0:
            self.remuck_mass.lower_threshold = 0.0
            self.remuck_ore1_mass.lower_threshold = 0.0


class LHDRemuckModule(drs.Module):
    """Aggregate LHD capacity moving blasted rock into remuck."""

    def __init__(
        self,
        name: str = "LHD",
        available_lhds: float = 1.0,
        capacity_per_lhd: float = 1000.0,
    ):
        super().__init__()
        self.name = name
        self.available_lhds = drs.Variable(f"{name}_available_lhds", available_lhds)
        self.capacity_per_lhd = drs.Variable(
            f"{name}_capacity_per_lhd",
            capacity_per_lhd,
        )
        self.actual_remuck_rate = drs.Variable(f"{name}_actual_remuck_rate", 0.0)

    def forward(self, requested_rate=None, available_lhds=None):
        requested = _rate_value(requested_rate)
        units = _rate_value(available_lhds) if available_lhds is not None else self.available_lhds.value
        capacity = units * self.capacity_per_lhd.value
        actual_rate = min(requested, capacity) if requested > 0.0 else capacity
        self.actual_remuck_rate.value = actual_rate
        return Flow(value=actual_rate)


class DrillBlastModule(drs.Module):
    """Aggregate drilling that periodically creates blasted rock mass."""

    def __init__(
        self,
        name: str = "DrillBlast",
        drill_work_required: float = 1.0,
        drill_rate_per_drill: float = 0.5,
        blast_tonnage: float = 1000.0,
    ):
        super().__init__()
        self.name = name
        self.drill_work_required = drs.Variable(
            f"{name}_drill_work_required",
            drill_work_required,
        )
        self.drill_rate_per_drill = drs.Variable(
            f"{name}_drill_rate_per_drill",
            drill_rate_per_drill,
        )
        self.blast_tonnage = drs.Variable(f"{name}_blast_tonnage", blast_tonnage)
        self.drill_work_completed = drs.Level(
            f"{name}_drill_work_completed",
            0.0,
        )
        self.blast_count = drs.Variable(f"{name}_blast_count", 0)
        self.last_blast_mass = drs.Variable(f"{name}_last_blast_mass", 0.0)

    def forward(self, available_drills=None):
        blast_mass = 0.0
        if self.drill_work_completed.value >= self.drill_work_required.value - 1e-6:
            blast_mass = self.blast_tonnage.value
            self.drill_work_completed.value = 0.0
            self.blast_count.value += 1

        available = _rate_value(available_drills)
        self.drill_work_completed.rate = available * self.drill_rate_per_drill.value
        self.drill_work_completed.upper_threshold = self.drill_work_required.value
        self.last_blast_mass.value = blast_mass

        return Flow(value=blast_mass)


class LHDLoadingModule(drs.Module):
    """Aggregate LHD loading with remuck priority and slower face tramming."""

    def __init__(
        self,
        name: str = "LHDLoading",
        available_lhds: float = 1.0,
        remuck_loading_capacity_per_lhd: float = 1500.0,
        face_tramming_capacity_per_lhd: float = 700.0,
    ):
        super().__init__()
        self.name = name
        self.available_lhds = drs.Variable(f"{name}_available_lhds", available_lhds)
        self.remuck_loading_capacity_per_lhd = drs.Variable(
            f"{name}_remuck_loading_capacity_per_lhd",
            remuck_loading_capacity_per_lhd,
        )
        self.face_tramming_capacity_per_lhd = drs.Variable(
            f"{name}_face_tramming_capacity_per_lhd",
            face_tramming_capacity_per_lhd,
        )
        self.actual_remuck_to_truck_rate = drs.Variable(
            f"{name}_actual_remuck_to_truck_rate",
            0.0,
        )
        self.actual_face_to_truck_rate = drs.Variable(
            f"{name}_actual_face_to_truck_rate",
            0.0,
        )
        self.actual_face_to_remuck_rate = drs.Variable(
            f"{name}_actual_face_to_remuck_rate",
            0.0,
        )

    def forward(
        self,
        blasted_rock_mass,
        remuck_mass,
        truck_capacity_rate,
        available_lhds=None,
    ):
        truck_capacity = _rate_value(truck_capacity_rate)
        units = _rate_value(available_lhds) if available_lhds is not None else self.available_lhds.value
        remuck_capacity = (
            units * self.remuck_loading_capacity_per_lhd.value
        )
        face_capacity = (
            units * self.face_tramming_capacity_per_lhd.value
        )

        remuck_to_truck_rate = 0.0
        if remuck_mass.value > 1e-6 and truck_capacity > 0.0:
            remuck_to_truck_rate = min(truck_capacity, remuck_capacity)

        lhd_fraction_used_for_remuck = (
            remuck_to_truck_rate / remuck_capacity if remuck_capacity > 0.0 else 0.0
        )
        remaining_face_capacity = face_capacity * max(
            0.0,
            1.0 - lhd_fraction_used_for_remuck,
        )
        remaining_truck_capacity = max(0.0, truck_capacity - remuck_to_truck_rate)

        face_to_truck_rate = 0.0
        face_to_remuck_rate = 0.0
        if blasted_rock_mass.value > 1e-6:
            face_to_truck_rate = min(remaining_truck_capacity, remaining_face_capacity)
            face_to_remuck_rate = max(0.0, remaining_face_capacity - face_to_truck_rate)

        self.actual_remuck_to_truck_rate.value = remuck_to_truck_rate
        self.actual_face_to_truck_rate.value = face_to_truck_rate
        self.actual_face_to_remuck_rate.value = face_to_remuck_rate

        return Flow(
            value=LHDLoadingRates(
                remuck_to_truck_rate=remuck_to_truck_rate,
                face_to_truck_rate=face_to_truck_rate,
                face_to_remuck_rate=face_to_remuck_rate,
            )
        )


class EquipmentAvailability(drs.Module):
    """Scheduled aggregate availability for trucks, LHDs, drills, or other equipment."""

    def __init__(
        self,
        name: str,
        total_units: float,
        downtime_start: float = float("inf"),
        downtime_duration: float = 0.0,
        down_units: float = None,
    ):
        super().__init__()
        self.name = name
        self.total_units = drs.Variable(f"{name}_total_units", total_units)
        self.available_units = drs.Variable(f"{name}_available_units", total_units)
        self.down_units = drs.Variable(f"{name}_down_units", 0.0)
        self.is_down = drs.Variable(f"{name}_is_down", False)
        self.just_went_down = drs.Variable(f"{name}_just_went_down", False)
        self.time_to_next_state_change = drs.Timer(
            f"{name}_time_to_next_state_change",
            initial_value=float("inf"),
            rate=-1.0,
        )
        self.time_to_next_state_change.lower_threshold = 0.0
        self.downtime_start = downtime_start
        self.downtime_duration = downtime_duration
        self.down_units_when_down = total_units if down_units is None else down_units
        self._was_down = False

    def _current_time(self) -> float:
        return _module_time(self)

    def forward(self):
        current_time = self._current_time()
        downtime_end = self.downtime_start + self.downtime_duration
        is_down = self.downtime_start <= current_time < downtime_end
        down_units = min(self.total_units.value, self.down_units_when_down) if is_down else 0.0

        self.just_went_down.value = is_down and not self._was_down
        self.is_down.value = is_down
        self.down_units.value = down_units
        self.available_units.value = max(0.0, self.total_units.value - down_units)
        self._was_down = is_down

        if current_time < self.downtime_start:
            next_change = self.downtime_start
        elif current_time < downtime_end:
            next_change = downtime_end
        else:
            next_change = float("inf")

        self.time_to_next_state_change.rate = -1.0
        self.time_to_next_state_change.lower_threshold = 0.0
        self.time_to_next_state_change.value = max(0.0, next_change - current_time)

        return Flow(value=self.available_units.value)


class EquipmentUnitAvailability(drs.Module):
    """Scheduled availability for one equipment unit."""

    def __init__(
        self,
        name: str,
        downtime_start: float = float("inf"),
        downtime_duration: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self.available_units = drs.Variable(f"{name}_available_units", 1.0)
        self.is_down = drs.Variable(f"{name}_is_down", False)
        self.just_went_down = drs.Variable(f"{name}_just_went_down", False)
        self.time_to_next_state_change = drs.Timer(
            f"{name}_time_to_next_state_change",
            initial_value=float("inf"),
            rate=-1.0,
        )
        self.time_to_next_state_change.lower_threshold = 0.0
        self.downtime_start = downtime_start
        self.downtime_duration = downtime_duration
        self._was_down = False

    def forward(self):
        current_time = _module_time(self)
        downtime_end = self.downtime_start + self.downtime_duration
        is_down = self.downtime_start <= current_time < downtime_end

        self.just_went_down.value = is_down and not self._was_down
        self.is_down.value = is_down
        self.available_units.value = 0.0 if is_down else 1.0
        self._was_down = is_down

        if current_time < self.downtime_start:
            next_change = self.downtime_start
        elif current_time < downtime_end:
            next_change = downtime_end
        else:
            next_change = float("inf")

        self.time_to_next_state_change.rate = -1.0
        self.time_to_next_state_change.lower_threshold = 0.0
        self.time_to_next_state_change.value = max(0.0, next_change - current_time)

        return Flow(value=self.available_units.value)


class EquipmentFleetAvailability(drs.Module):
    """Aggregates individual equipment-unit availability into fleet availability."""

    def __init__(self, name: str, unit_downtime_schedules):
        super().__init__()
        self.name = name
        self.total_units = drs.Variable(
            f"{name}_total_units",
            float(len(unit_downtime_schedules)),
        )
        self.available_units = drs.Variable(f"{name}_available_units", 0.0)
        self.down_units = drs.Variable(f"{name}_down_units", 0.0)
        self.is_down = drs.Variable(f"{name}_is_down", False)
        self.just_went_down = drs.Variable(f"{name}_just_went_down", False)

        for index, schedule in enumerate(unit_downtime_schedules):
            if isinstance(schedule, dict):
                downtime_start = schedule.get("downtime_start", float("inf"))
                downtime_duration = schedule.get("downtime_duration", 0.0)
            else:
                downtime_start, downtime_duration = schedule

            setattr(
                self,
                f"unit_{index}",
                EquipmentUnitAvailability(
                    name=f"{name}_unit_{index}",
                    downtime_start=downtime_start,
                    downtime_duration=downtime_duration,
                ),
            )

    def forward(self):
        available_units = 0.0
        just_went_down = False

        for unit in self._modules.values():
            unit_available = unit()
            available_units += unit_available.value
            just_went_down = just_went_down or unit.just_went_down.value

        self.available_units.value = available_units
        self.down_units.value = self.total_units.value - available_units
        self.is_down.value = available_units <= 1e-6
        self.just_went_down.value = just_went_down

        return Flow(value=self.available_units.value)


class StochasticEquipmentUnitAvailability(drs.Module):
    """Stochastic up/down availability for one equipment unit."""

    def __init__(
        self,
        name: str,
        time_between_failures_hours,
        repair_time_hours,
    ):
        super().__init__()
        self.name = name
        self.time_between_failures_hours = time_between_failures_hours
        self.repair_time_hours = repair_time_hours
        self.available_units = drs.Variable(f"{name}_available_units", 1.0)
        self.is_down = drs.Variable(f"{name}_is_down", False)
        self.just_went_down = drs.Variable(f"{name}_just_went_down", False)
        self.just_repaired = drs.Variable(f"{name}_just_repaired", False)
        self.time_to_next_state_change = drs.Timer(
            f"{name}_time_to_next_state_change",
            initial_value=_sample_triangular_hours_as_days(time_between_failures_hours),
            rate=-1.0,
        )
        self.time_to_next_state_change.lower_threshold = 0.0

    def _sample_next_duration(self):
        if self.is_down.value:
            return _sample_triangular_hours_as_days(self.repair_time_hours)
        return _sample_triangular_hours_as_days(self.time_between_failures_hours)

    def forward(self):
        self.just_went_down.value = False
        self.just_repaired.value = False

        if self.time_to_next_state_change.value <= 1e-6:
            if self.is_down.value:
                self.is_down.value = False
                self.available_units.value = 1.0
                self.just_repaired.value = True
            else:
                self.is_down.value = True
                self.available_units.value = 0.0
                self.just_went_down.value = True

            self.time_to_next_state_change.value = self._sample_next_duration()

        self.time_to_next_state_change.rate = -1.0
        self.time_to_next_state_change.lower_threshold = 0.0

        return Flow(value=self.available_units.value)


class StochasticEquipmentFleetAvailability(drs.Module):
    """Aggregates individually stochastic equipment availability."""

    def __init__(
        self,
        name: str,
        total_units: int,
        time_between_failures_hours,
        repair_time_hours,
    ):
        super().__init__()
        self.name = name
        self.total_units = drs.Variable(f"{name}_total_units", float(total_units))
        self.available_units = drs.Variable(f"{name}_available_units", float(total_units))
        self.down_units = drs.Variable(f"{name}_down_units", 0.0)
        self.is_down = drs.Variable(f"{name}_is_down", False)
        self.just_went_down = drs.Variable(f"{name}_just_went_down", False)
        self.just_repaired = drs.Variable(f"{name}_just_repaired", False)

        for index in range(total_units):
            setattr(
                self,
                f"unit_{index}",
                StochasticEquipmentUnitAvailability(
                    name=f"{name}_unit_{index}",
                    time_between_failures_hours=time_between_failures_hours,
                    repair_time_hours=repair_time_hours,
                ),
            )

    def forward(self):
        available_units = 0.0
        just_went_down = False
        just_repaired = False

        for unit in self._modules.values():
            unit_available = unit()
            available_units += unit_available.value
            just_went_down = just_went_down or unit.just_went_down.value
            just_repaired = just_repaired or unit.just_repaired.value

        self.available_units.value = available_units
        self.down_units.value = self.total_units.value - available_units
        self.is_down.value = available_units <= 1e-6
        self.just_went_down.value = just_went_down
        self.just_repaired.value = just_repaired

        return Flow(value=self.available_units.value)


class AggregateTruckHaulageModule(drs.Module):
    """Creates aggregate haulage parcels and releases them after travel delay."""

    def __init__(
        self,
        name: str = "TruckHaulage",
        source_face: int = 0,
        aggregate_parcel_mass: float = 300.0,
        travel_time: float = 0.05,
    ):
        super().__init__()
        self.name = name
        self.source_face = source_face
        self.aggregate_parcel_mass = drs.Variable(
            f"{name}_aggregate_parcel_mass",
            aggregate_parcel_mass,
        )
        self.travel_time = drs.Variable(f"{name}_travel_time", travel_time)
        self.loaded_mass_buffer = drs.Level(f"{name}_loaded_mass_buffer", 0.0)
        self.time_to_next_arrival = drs.Timer(
            f"{name}_time_to_next_arrival",
            initial_value=0.0,
            rate=0.0,
        )
        self.dispatched_parcel_count = drs.Variable(
            f"{name}_dispatched_parcel_count",
            0,
        )
        self.arrived_parcel_count = drs.Variable(f"{name}_arrived_parcel_count", 0)
        self.last_arrived_mass = drs.Variable(f"{name}_last_arrived_mass", 0.0)
        self.transit_queue = []

    def _current_time(self) -> float:
        return _module_time(self)

    def _dispatch_parcel(self, source_face: int, ore1_fraction: float, mass=None):
        current_time = self._current_time()
        parcel_mass = self.aggregate_parcel_mass.value if mass is None else mass
        parcel = OreParcel(
            source_face=source_face,
            mass=parcel_mass,
            ore1_fraction=ore1_fraction,
            dispatch_time=current_time,
            arrival_time=current_time + self.travel_time.value,
        )
        self.transit_queue.append(parcel)
        self.transit_queue.sort(key=lambda p: p.arrival_time)
        self.dispatched_parcel_count.value += 1
        self.loaded_mass_buffer.value = 0.0

    def _pop_arrivals(self):
        current_time = self._current_time()
        arrivals = []
        while self.transit_queue and self.transit_queue[0].arrival_time <= current_time + 1e-6:
            arrivals.append(self.transit_queue.pop(0))
        return arrivals

    def _update_arrival_timer(self):
        current_time = self._current_time()
        self.time_to_next_arrival.rate = -1.0
        self.time_to_next_arrival.lower_threshold = 0.0
        if self.transit_queue:
            self.time_to_next_arrival.value = max(
                0.0,
                self.transit_queue[0].arrival_time - current_time,
            )
        else:
            self.time_to_next_arrival.value = float("inf")

    def forward(self, loaded_rate=None, ore1_fraction=None, flush_buffer=None):
        loaded = _loaded_rate_value(loaded_rate)
        ore1_frac = _rate_value(ore1_fraction)
        should_flush = _flag_value(flush_buffer)

        if self.loaded_mass_buffer.value >= self.aggregate_parcel_mass.value - 1e-6:
            self._dispatch_parcel(self.source_face, ore1_frac)
        elif should_flush and self.loaded_mass_buffer.value > 1e-6:
            self._dispatch_parcel(
                self.source_face,
                ore1_frac,
                mass=self.loaded_mass_buffer.value,
            )

        self.loaded_mass_buffer.rate = loaded
        self.loaded_mass_buffer.upper_threshold = self.aggregate_parcel_mass.value

        arrivals = self._pop_arrivals()
        self.arrived_parcel_count.value += len(arrivals)
        self.last_arrived_mass.value = arrivals[-1].mass if arrivals else 0.0
        self._update_arrival_timer()

        return Flow(value=arrivals)


class ParcelStockpileReceiver(drs.Module):
    """Receives arrived OreParcels, then unloads them into ore inventories by rate."""

    def __init__(
        self,
        name: str = "ParcelStockpileReceiver",
        unload_rate: float = 6000.0,
    ):
        super().__init__()
        self.name = name
        self.unload_rate = drs.Variable(f"{name}_unload_rate", unload_rate)
        self.buffer_ore1_mass = drs.Level(f"{name}_buffer_ore1_mass", 0.0)
        self.buffer_ore2_mass = drs.Level(f"{name}_buffer_ore2_mass", 0.0)
        self.buffer_total_mass = drs.Level(f"{name}_buffer_total_mass", 0.0)
        self.ore1_mass = drs.Level(f"{name}_ore1_mass", 0.0)
        self.ore2_mass = drs.Level(f"{name}_ore2_mass", 0.0)
        self.total_mass = drs.Level(f"{name}_total_mass", 0.0)
        self.actual_unload_rate = drs.Variable(f"{name}_actual_unload_rate", 0.0)
        self.received_parcel_count = drs.Variable(f"{name}_received_parcel_count", 0)
        self.last_received_mass = drs.Variable(f"{name}_last_received_mass", 0.0)

    def forward(self, arrived_parcels=None):
        parcels = _flow_value(arrived_parcels) or []
        received_mass = 0.0

        for parcel in parcels:
            ore1_mass = parcel.mass * parcel.ore1_fraction
            ore2_mass = parcel.mass * (1.0 - parcel.ore1_fraction)
            self.buffer_ore1_mass.value += ore1_mass
            self.buffer_ore2_mass.value += ore2_mass
            self.buffer_total_mass.value += parcel.mass
            received_mass += parcel.mass

        self.received_parcel_count.value += len(parcels)
        self.last_received_mass.value = received_mass

        if self.buffer_total_mass.value > 1e-6:
            ore1_fraction = self.buffer_ore1_mass.value / self.buffer_total_mass.value
            unload_rate = self.unload_rate.value
        else:
            ore1_fraction = 0.0
            unload_rate = 0.0

        self.actual_unload_rate.value = unload_rate
        self.buffer_total_mass.rate = -unload_rate
        self.buffer_ore1_mass.rate = -unload_rate * ore1_fraction
        self.buffer_ore2_mass.rate = -unload_rate * (1.0 - ore1_fraction)
        self.total_mass.rate = unload_rate
        self.ore1_mass.rate = unload_rate * ore1_fraction
        self.ore2_mass.rate = unload_rate * (1.0 - ore1_fraction)

        if self.buffer_total_mass.rate < 0:
            self.buffer_total_mass.lower_threshold = 0.0
            self.buffer_ore1_mass.lower_threshold = 0.0
            self.buffer_ore2_mass.lower_threshold = 0.0

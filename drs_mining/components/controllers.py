import drs
from drs.flow import Flow
from .modes import OperatingMode, RequireDecision
from .mine_face import BaseMineFace, ConcentratorMineFace
from .fleet import ContinuousFleetLogistics
from .plant import BaseMetallurgicalPlant, ConcentratorPlant
from .modes import MODES


class BaseBlendingController(drs.Module):
    _TIMER_MAP = {
        "MODE_A": "cumulative_time_mode_a",
        "MODE_A_CONTINGENCY": "cumulative_time_mode_a_contingency",
        "MODE_A_MINE_SURGING": "cumulative_time_mode_a_surging",
        "MODE_B": "cumulative_time_mode_b",
        "MODE_B_CONTINGENCY": "cumulative_time_mode_b_contingency",
        "MODE_B_MINE_SURGING": "cumulative_time_mode_b_surging",
        "SHUTDOWN": "cumulative_time_shutdown",
    }
    _CONTINGENCY_MODES = {"MODE_A_CONTINGENCY", "MODE_B_CONTINGENCY"}

    def __init__(
        self,
        mine: BaseMineFace,
        fleet: ContinuousFleetLogistics,
        plant: BaseMetallurgicalPlant,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
    ):
        super().__init__()
        self.mine = mine
        self.fleet = fleet
        self.plant = plant
        self.target_ore_stock_level = target_ore_stock_level
        self.critical_ore2_level = critical_ore2_level
        self.duration_of_production_campaigns = duration_of_production_campaigns
        self.duration_of_shutdowns = duration_of_shutdowns
        self.duration_of_contingency_segments = duration_of_contingency_segments
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
        )

        self.active_operating_mode = drs.Variable(
            "active_operating_mode", MODES["MODE_A"]
        )
        self.total_system_ore_mass = drs.Level(
            "total_system_ore_mass", initial_value=self.target_ore_stock_level
        )

        self.current_campaign_duration = drs.Timer(
            "current_campaign_duration", initial_value=0.0
        )
        self.current_contingency_duration = drs.Timer(
            "current_contingency_duration", initial_value=0.0
        )
        self.cumulative_time_mode_a = drs.Timer(
            "cumulative_time_mode_a", initial_value=0.0
        )
        self.cumulative_time_mode_a_contingency = drs.Timer(
            "cumulative_time_mode_a_contingency", initial_value=0.0
        )
        self.cumulative_time_mode_a_surging = drs.Timer(
            "cumulative_time_mode_a_surging", initial_value=0.0
        )
        self.cumulative_time_mode_b = drs.Timer(
            "cumulative_time_mode_b", initial_value=0.0
        )
        self.cumulative_time_mode_b_contingency = drs.Timer(
            "cumulative_time_mode_b_contingency", initial_value=0.0
        )
        self.cumulative_time_mode_b_surging = drs.Timer(
            "cumulative_time_mode_b_surging", initial_value=0.0
        )
        self.cumulative_time_shutdown = drs.Timer(
            "cumulative_time_shutdown", initial_value=0.0
        )

        self.target_mine_mass_rate = drs.Variable("target_mine_mass_rate", 0.0)
        self.target_stock1_outflow_rate = drs.Variable(
            "target_stock1_outflow_rate", 0.0
        )
        self.target_stock2_outflow_rate = drs.Variable(
            "target_stock2_outflow_rate", 0.0
        )

    @property
    def total_duration(self) -> float:
        """Returns total accumulated duration across all modes and shutdown."""
        return (
            self.cumulative_time_mode_a.value
            + self.cumulative_time_mode_a_contingency.value
            + self.cumulative_time_mode_a_surging.value
            + self.cumulative_time_mode_b.value
            + self.cumulative_time_mode_b_contingency.value
            + self.cumulative_time_mode_b_surging.value
            + self.cumulative_time_shutdown.value
        )

    def active_duration(self, current_time: float = None) -> float:
        """Encapsulate state calculations for active operational duration."""
        if current_time is None:
            current_time = self.total_duration
        return max(0.0, current_time - self.cumulative_time_shutdown.value)

    def is_campaign_complete(self) -> bool:
        m = self.active_operating_mode.value.name
        threshold = (
            self.duration_of_shutdowns
            if m == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )

        self.current_campaign_duration.upper_threshold = threshold

        return self.current_campaign_duration.value >= (threshold - 1e-6)

    def is_contingency_complete(self) -> bool:
        threshold = self.duration_of_contingency_segments

        self.current_contingency_duration.upper_threshold = threshold

        return self.current_contingency_duration.value >= (threshold - 1e-6)

    def reset_campaign_timer(self):
        self.current_campaign_duration.reset()

    def reset_contingency_timer(self):
        self.current_contingency_duration.reset()

    def forward(self) -> Flow:
        mine = self.mine

        if mine is not None and abs(mine.net_extracted_mass) < 1e-6:
            self.cumulative_time_mode_a.reset()
            self.cumulative_time_mode_a_contingency.reset()
            self.cumulative_time_mode_a_surging.reset()
            self.cumulative_time_mode_b.reset()
            self.cumulative_time_mode_b_contingency.reset()
            self.cumulative_time_mode_b_surging.reset()
            self.cumulative_time_shutdown.reset()

        self.total_system_ore_mass.value = self.parent.total_stockpile_mass

        next_mode = self.active_operating_mode.value.check_end_conditions(self.parent)

        if isinstance(next_mode, RequireDecision):
            decision = self.controller_decision()
            if decision:
                next_mode = decision

        if next_mode and not isinstance(next_mode, RequireDecision):
            self.active_operating_mode.value = next_mode

        self._update_timers(self.active_operating_mode.value.name)

        targets = self.active_operating_mode.value.get_target_rates(self.parent)
        self.target_mine_mass_rate.value = targets.extraction_rate
        self.target_stock1_outflow_rate.value = targets.ore1_milling_rate
        self.target_stock2_outflow_rate.value = targets.ore2_milling_rate

    def _update_timers(self, m: str):
        timer_attr = self._TIMER_MAP.get(m)
        if timer_attr:
            getattr(self, timer_attr).rate = 1.0
        self.current_campaign_duration.rate = 1.0
        self.current_campaign_duration.upper_threshold = (
            self.duration_of_shutdowns
            if m == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        if m in self._CONTINGENCY_MODES:
            self.current_contingency_duration.rate = 1.0
            self.current_contingency_duration.upper_threshold = (
                self.duration_of_contingency_segments
            )

    def _choose_next_campaign_mode(self):
        ore2 = self.parent.ore2_mass
        total_stock = self.total_system_ore_mass.value
        EPS = 1e-6
        if ore2 > self.critical_ore2_level:
            return (
                MODES["MODE_A"]
                if total_stock <= self.target_ore_stock_level + EPS
                else MODES["MODE_A_MINE_SURGING"]
            )
        else:
            return (
                MODES["MODE_B"]
                if total_stock <= self.target_ore_stock_level + EPS
                else MODES["MODE_B_MINE_SURGING"]
            )

    def controller_decision(self):
        m = self.active_operating_mode.value.name

        if self.is_campaign_complete():
            self.reset_campaign_timer()
            if m == "SHUTDOWN":
                return self._choose_next_campaign_mode()
            return MODES["SHUTDOWN"]

        if m.endswith("_CONTINGENCY"):
            self.reset_contingency_timer()
        base = m.replace("_CONTINGENCY", "").replace("_MINE_SURGING", "")
        return MODES[base]


class ConcentratorController(BaseBlendingController):
    def __init__(
        self,
        mine: ConcentratorMineFace,
        fleet: ContinuousFleetLogistics,
        plant: ConcentratorPlant,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
    ):
        super().__init__(
            mine=mine,
            fleet=fleet,
            plant=plant,
            target_ore_stock_level=target_ore_stock_level,
            critical_ore2_level=critical_ore2_level,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        )


class MultiFaceConcentratorController(BaseBlendingController):
    """Controller for multi-face mine operations."""

    def __init__(
        self,
        faces,
        fleet,
        plant,
        target_ore_stock_level: float = 60000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        mode_a_ore1_milling_rate: float = 3600.0,
        mode_a_ore2_milling_rate: float = 2400.0,
        mode_a_contingency_ore1_milling_rate: float = 3900.0,
        mode_b_ore1_milling_rate: float = 4600.0,
        mode_b_ore2_milling_rate: float = 800.0,
        mode_b_contingency_ore2_milling_rate: float = 2500.0,
        fleet_shift_duration: float = 0.5,
        total_lhd_count: float = 3.0,
        total_truck_count: float = 10.0,
        max_lhds_per_face: float = 2.0,
        max_trucks_per_face: float = 6.0,
        face_haul_distance: tuple = (1.5, 2.2),
        face_accessibility_fraction: tuple = (0.93, 0.91),
        truck_velocity: float = 15.0,
        loader_cycle_time_hours: float = 0.0833,
        truck_dump_time_hours: float = 0.033,
        traffic_delay_per_truck_hours: float = 0.015,
        fleet_mechanical_availability: float = 0.85,
        loader_payload_tonnes: float = 15.0,
        truck_payload_tonnes: float = 30.0,
        development_rate_per_extra_truck: float = 50.0,
    ):
        super().__init__(
            mine=None,
            fleet=fleet,
            plant=plant,
            target_ore_stock_level=target_ore_stock_level,
            critical_ore2_level=critical_ore2_level,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        )
        self.faces = list(faces)
        self.mode_a_ore1_milling_rate = mode_a_ore1_milling_rate
        self.mode_a_ore2_milling_rate = mode_a_ore2_milling_rate
        self.mode_a_contingency_ore1_milling_rate = mode_a_contingency_ore1_milling_rate
        self.mode_b_ore1_milling_rate = mode_b_ore1_milling_rate
        self.mode_b_ore2_milling_rate = mode_b_ore2_milling_rate
        self.mode_b_contingency_ore2_milling_rate = mode_b_contingency_ore2_milling_rate
        self.fleet_shift_duration = fleet_shift_duration
        self.total_lhd_count = total_lhd_count
        self.total_truck_count = total_truck_count
        self.max_lhds_per_face = max_lhds_per_face
        self.max_trucks_per_face = max_trucks_per_face
        self.face_haul_distance = face_haul_distance
        self.face_accessibility_fraction = face_accessibility_fraction
        self.truck_velocity = truck_velocity
        self.loader_cycle_time_hours = loader_cycle_time_hours
        self.truck_dump_time_hours = truck_dump_time_hours
        self.traffic_delay_per_truck_hours = traffic_delay_per_truck_hours
        self.fleet_mechanical_availability = fleet_mechanical_availability
        self.loader_payload_tonnes = loader_payload_tonnes
        self.truck_payload_tonnes = truck_payload_tonnes
        self.development_rate_per_extra_truck = development_rate_per_extra_truck

        self.total_extra_trucks = drs.Variable("total_extra_trucks", 0.0)
        self.cumulative_mine_development = drs.Level(
            "cumulative_mine_development", initial_value=0.0
        )
        self.face_target_extraction_rates = []
        self.face_real_extraction_rates = []
        self.face_achieved_extraction_rates = []
        self.face_operational_downtime_fractions = []
        self.face_shift_allocation_fractions = []
        self.face_lhd_allocations = []
        self.face_truck_allocations = []
        self.face_match_factors = []
        self.face_truck_cycle_times = []
        self.face_target_rates = self.face_achieved_extraction_rates
        for i in range(len(self.faces)):
            target = drs.Variable(f"face{i}_target_extraction_rate", 0.0)
            real_extraction = drs.Variable(f"face{i}_real_extraction_rate", 0.0)
            achieved = drs.Variable(f"face{i}_achieved_extraction_rate", 0.0)
            operational_downtime = drs.Variable(
                f"face{i}_operational_downtime_fraction", 0.0
            )
            allocation_fraction = drs.Variable(
                f"face{i}_shift_allocation_fraction", 1.0
            )
            setattr(self, f"face{i}_target_extraction_rate", target)
            setattr(self, f"face{i}_real_extraction_rate", real_extraction)
            setattr(self, f"face{i}_achieved_extraction_rate", achieved)
            setattr(
                self, f"face{i}_operational_downtime_fraction", operational_downtime
            )
            setattr(self, f"face{i}_shift_allocation_fraction", allocation_fraction)

            lhd_alloc = drs.Variable(f"face{i}_lhd_allocation", 0.0)
            truck_alloc = drs.Variable(f"face{i}_truck_allocation", 0.0)
            setattr(self, f"face{i}_lhd_allocation", lhd_alloc)
            setattr(self, f"face{i}_truck_allocation", truck_alloc)

            match_factor = drs.Variable(f"face{i}_match_factor", 0.0)
            truck_cycle = drs.Variable(f"face{i}_truck_cycle_time", 0.0)
            setattr(self, f"face{i}_match_factor", match_factor)
            setattr(self, f"face{i}_truck_cycle_time", truck_cycle)

            self.face_target_extraction_rates.append(target)
            self.face_real_extraction_rates.append(real_extraction)
            self.face_achieved_extraction_rates.append(achieved)
            self.face_operational_downtime_fractions.append(operational_downtime)
            self.face_shift_allocation_fractions.append(allocation_fraction)
            self.face_lhd_allocations.append(lhd_alloc)
            self.face_truck_allocations.append(truck_alloc)
            self.face_match_factors.append(match_factor)
            self.face_truck_cycle_times.append(truck_cycle)
        self._mode_allocations = self._precompute_allocations()
        self.current_shift_allocations = None
        self.current_shift_mode_name = None
        self.fleet_shift_timer = drs.Timer("fleet_shift_timer", initial_value=0.0)
        self.fleet_shift_count = drs.Variable("fleet_shift_count", 0)

    def _precompute_allocations(self):
        """Pre-compute fixed face extraction fractions per mode using face mean ore fractions."""
        face_ore1_fracs = [1.0 - f.generator.mean_fraction for f in self.faces]
        f1, f2 = face_ore1_fracs[0], face_ore1_fracs[1]

        modes_to_compute = {
            "MODE_A": (
                self.mode_a_ore1_milling_rate,
                self.mode_a_ore2_milling_rate,
            ),
            "MODE_A_CONTINGENCY": (
                self.mode_a_contingency_ore1_milling_rate,
                0.0,
            ),
            "MODE_B": (
                self.mode_b_ore1_milling_rate,
                self.mode_b_ore2_milling_rate,
            ),
            "MODE_B_CONTINGENCY": (
                0.0,
                self.mode_b_contingency_ore2_milling_rate,
            ),
        }
        result = {}
        for mode_name, (ore1, ore2) in modes_to_compute.items():
            total = ore1 + ore2
            if total <= 0 or abs(f1 - f2) < 1e-12:
                fracs = [0.5, 0.5] if total > 0 else [0.0, 0.0]
            else:
                r1 = (ore1 - total * f2) / (f1 - f2)
                r1 = max(0.0, min(total, r1))
                r2 = total - r1
                fracs = [r1 / total, r2 / total]
            result[mode_name] = fracs

        # Surging modes use extreme allocations to correct the imbalance.
        # MODE_A_MINE_SURGING (ore1 stockout): maximize ore1 → all to face1 (85% ore1)
        # MODE_B_MINE_SURGING (ore2 stockout): maximize ore2 → all to face2 (45% ore2)
        # This ensures surging produces a blend different from the base mode target,
        # letting the stockpile drain and surging exit quickly. Without this, surging
        # would produce the same blend as the base mode (e.g. 60/40 for Mode A),
        # making extraction = milling and the system stuck in surging forever.
        result["MODE_A_MINE_SURGING"] = [1.0, 0.0]
        result["MODE_B_MINE_SURGING"] = [0.0, 1.0]

        return result

    def _get_allocations_for_mode(self, mode_name):
        fracs = self._mode_allocations.get(mode_name)
        if fracs is None:
            base_key = mode_name.replace("_MINE_SURGING", "")
            fracs = self._mode_allocations.get(base_key)
        return fracs

    def _face_config_value(self, name, face_index, default):
        values = getattr(self, name, None)
        if values is None:
            return default
        if face_index < len(values):
            return values[face_index]
        return default

    def _clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def _refresh_shift_allocation_fractions(self):
        if not self.current_shift_allocations:
            return
        for i, factor in enumerate(self.face_shift_allocation_fractions):
            factor.value = self.current_shift_allocations[i]

        self._distribute_discrete_fleet(self.current_shift_allocations)

    def _distribute_discrete_fleet(self, fracs):
        import math

        total_lhd = int(getattr(self, "total_lhd_count", 3.0))
        total_truck = int(getattr(self, "total_truck_count", 10.0))
        max_lhds = int(getattr(self, "max_lhds_per_face", float("inf")))
        max_trucks = int(getattr(self, "max_trucks_per_face", float("inf")))

        num_faces = len(self.faces)
        if num_faces == 0:
            return

        unassigned_lhds = total_lhd
        lhd_assignments = [0] * num_faces

        face_priorities = sorted(range(num_faces), key=lambda i: fracs[i], reverse=True)

        for i in face_priorities:
            target = math.floor(total_lhd * fracs[i])
            assigned = min(target, max_lhds, unassigned_lhds)
            lhd_assignments[i] = assigned
            unassigned_lhds -= assigned

        for i in face_priorities:
            if unassigned_lhds <= 0:
                break
            if lhd_assignments[i] < max_lhds:
                lhd_assignments[i] += 1
                unassigned_lhds -= 1

        for i in range(num_faces):
            self.face_lhd_allocations[i].value = float(lhd_assignments[i])

        unassigned_trucks = total_truck
        truck_assignments = [0] * num_faces

        for i in face_priorities:
            target = math.floor(total_truck * fracs[i])
            assigned = min(target, max_trucks, unassigned_trucks)
            truck_assignments[i] = assigned
            unassigned_trucks -= assigned

        for i in face_priorities:
            if unassigned_trucks <= 0:
                break
            if truck_assignments[i] < max_trucks:
                truck_assignments[i] += 1
                unassigned_trucks -= 1

        for i in range(num_faces):
            self.face_truck_allocations[i].value = float(truck_assignments[i])

    def _face_real_extraction_rate(self, face_index, target_extraction_rate):
        lhd_alloc = self.face_lhd_allocations[face_index].value
        truck_alloc = self.face_truck_allocations[face_index].value

        distance = self._face_config_value("face_haul_distance", face_index, 0.0)
        accessibility = self._clamp(
            self._face_config_value("face_accessibility_fraction", face_index, 1.0),
            0.0,
            1.0,
        )
        mechanical_availability = getattr(self, "fleet_mechanical_availability", 0.85)

        traffic_delay = self.traffic_delay_per_truck_hours * truck_alloc

        travel_time = (2 * distance) / self.truck_velocity

        truck_loading_time_hours = self.loader_cycle_time_hours * (
            self.truck_payload_tonnes / self.loader_payload_tonnes
        )

        truck_cycle_time = (
            travel_time
            + truck_loading_time_hours
            + self.truck_dump_time_hours
            + traffic_delay
        )

        if truck_cycle_time <= 0 or lhd_alloc <= 0:
            self.face_match_factors[face_index].value = 0.0
            self.face_truck_cycle_times[face_index].value = truck_cycle_time
            return 0.0

        match_factor = (truck_alloc * truck_loading_time_hours) / (
            lhd_alloc * truck_cycle_time
        )
        self.face_match_factors[face_index].value = match_factor
        self.face_truck_cycle_times[face_index].value = truck_cycle_time

        if match_factor < 1.0:
            max_rate = (
                (truck_alloc / truck_cycle_time) * self.truck_payload_tonnes * 24.0
            )
        else:
            max_rate = (
                (lhd_alloc / self.loader_cycle_time_hours)
                * self.loader_payload_tonnes
                * 24.0
            )

        final_real_extraction_rate = max_rate * accessibility * mechanical_availability

        if (
            final_real_extraction_rate > target_extraction_rate
            and target_extraction_rate > 0
        ):
            utilization = target_extraction_rate / final_real_extraction_rate
            unused_trucks = truck_alloc * (1.0 - utilization)
        else:
            unused_trucks = 0.0

        self.total_extra_trucks.value += unused_trucks

        return max(0.0, final_real_extraction_rate)

    def _reallocate_fleet_for_shift(self):
        mode_name = self.active_operating_mode.value.name
        self.current_shift_allocations = self._get_allocations_for_mode(mode_name)
        self.current_shift_mode_name = mode_name
        self._refresh_shift_allocation_fractions()
        self.fleet_shift_count.value += 1

    def forward(self):
        self.total_extra_trucks.value = 0.0

        total_net_extracted = sum(f.net_extracted_mass for f in self.faces)
        if abs(total_net_extracted) < 1e-6:
            self.cumulative_time_mode_a.reset()
            self.cumulative_time_mode_a_contingency.reset()
            self.cumulative_time_mode_a_surging.reset()
            self.cumulative_time_mode_b.reset()
            self.cumulative_time_mode_b_contingency.reset()
            self.cumulative_time_mode_b_surging.reset()
            self.cumulative_time_shutdown.reset()

        self.total_system_ore_mass.value = self.parent.total_stockpile_mass

        next_mode = self.active_operating_mode.value.check_end_conditions(self.parent)

        if isinstance(next_mode, RequireDecision):
            decision = self.controller_decision()
            if decision:
                next_mode = decision

        if next_mode:
            self.active_operating_mode.value = next_mode

        mode_name = self.active_operating_mode.value.name
        self._update_timers(mode_name)

        self.fleet_shift_timer.rate = 1.0
        self.fleet_shift_timer.upper_threshold = self.fleet_shift_duration

        shift_due = self.fleet_shift_timer.value >= self.fleet_shift_duration - 1e-6
        mode_changed = mode_name != self.current_shift_mode_name
        if self.current_shift_allocations is None or shift_due or mode_changed:
            self.fleet_shift_timer.reset()
            self._reallocate_fleet_for_shift()

        targets = self.active_operating_mode.value.get_target_rates(self.parent)
        self.target_mine_mass_rate.value = targets.extraction_rate
        self.target_stock1_outflow_rate.value = targets.ore1_milling_rate
        self.target_stock2_outflow_rate.value = targets.ore2_milling_rate

        fracs = self.current_shift_allocations
        if fracs:
            for i, _face in enumerate(self.faces):
                target_extraction_rate = targets.extraction_rate * fracs[i]
                real_extraction_rate = self._face_real_extraction_rate(
                    i, target_extraction_rate
                )
                self.face_target_extraction_rates[i].value = target_extraction_rate
                self.face_real_extraction_rates[i].value = real_extraction_rate
                self.face_achieved_extraction_rates[i].value = min(
                    target_extraction_rate, real_extraction_rate
                )
        else:
            for i, _face in enumerate(self.faces):
                self.face_target_extraction_rates[i].value = 0.0
                self.face_real_extraction_rates[i].value = 0.0
                self.face_achieved_extraction_rates[i].value = 0.0
                self.face_operational_downtime_fractions[i].value = 0.0

        self.cumulative_mine_development.rate = (
            self.total_extra_trucks.value * self.development_rate_per_extra_truck
        )

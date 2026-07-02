from drs.module import drs
from drs.flow import Flow
from .modes import OperatingMode, RequireDecision
from .config import ConcentratorConfig
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
        config,
        mine: BaseMineFace,
        fleet: ContinuousFleetLogistics,
        plant: BaseMetallurgicalPlant,
    ):
        super().__init__()
        self.config = config
        self.mine = mine
        self.fleet = fleet
        self.plant = plant

        self.active_operating_mode = drs.Variable(
            "active_operating_mode", MODES["MODE_A"]
        )
        self.total_system_ore_mass = drs.Level(
            "total_system_ore_mass", initial_value=config.target_ore_stock_level
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

    def is_campaign_complete(self) -> bool:
        c = self.config
        m = self.active_operating_mode.value.name
        threshold = (
            c.duration_of_shutdowns
            if m == "SHUTDOWN"
            else c.duration_of_production_campaigns
        )

        self.current_campaign_duration.upper_threshold = threshold

        return self.current_campaign_duration.value >= (threshold - 1e-6)

    def is_contingency_complete(self) -> bool:
        c = self.config
        threshold = c.duration_of_contingency_segments

        self.current_contingency_duration.upper_threshold = threshold

        return self.current_contingency_duration.value >= (threshold - 1e-6)

    def reset_campaign_timer(self):
        self.current_campaign_duration.reset()

    def reset_contingency_timer(self):
        self.current_contingency_duration.reset()

    def forward(self) -> Flow:
        c = self.config
        mine = self.mine

        if (
            abs(
                mine.cumulative_extracted_mass.value
                - c.ore_to_be_extracted_during_warming_period
            )
            < 1e-6
        ):
            self.cumulative_time_mode_a.reset()
            self.cumulative_time_mode_a_contingency.reset()
            self.cumulative_time_mode_a_surging.reset()
            self.cumulative_time_mode_b.reset()
            self.cumulative_time_mode_b_contingency.reset()
            self.cumulative_time_mode_b_surging.reset()
            self.cumulative_time_shutdown.reset()

        self.total_system_ore_mass.value = (
            self.parent.ore1_stock.current_mass.value
            + self.parent.ore2_stock.current_mass.value
        )

        next_mode = self.active_operating_mode.value.check_end_conditions(self.parent)

        if isinstance(next_mode, RequireDecision):
            decision = self.controller_decision()
            if decision:
                next_mode = decision

        if next_mode:
            self.active_operating_mode.value = next_mode

        self._update_timers(self.active_operating_mode.value.name)

        targets = self.active_operating_mode.value.get_target_rates(self.parent)
        self.target_mine_mass_rate.value = targets.extraction_rate
        self.target_stock1_outflow_rate.value = targets.ore1_milling_rate
        self.target_stock2_outflow_rate.value = targets.ore2_milling_rate

    def _update_timers(self, m: str):
        c = self.config
        timer_attr = self._TIMER_MAP.get(m)
        if timer_attr:
            getattr(self, timer_attr).rate = 1.0
        self.current_campaign_duration.rate = 1.0
        self.current_campaign_duration.upper_threshold = (
            c.duration_of_shutdowns
            if m == "SHUTDOWN"
            else c.duration_of_production_campaigns
        )
        if m in self._CONTINGENCY_MODES:
            self.current_contingency_duration.rate = 1.0
            self.current_contingency_duration.upper_threshold = (
                c.duration_of_contingency_segments
            )

    def _choose_next_campaign_mode(self, config):
        ore2 = self.parent.ore2_stock.current_mass.value
        total_stock = self.total_system_ore_mass.value
        EPS = 1e-6
        if ore2 > config.critical_ore2_level:
            return (
                MODES["MODE_A"]
                if total_stock <= config.target_ore_stock_level + EPS
                else MODES["MODE_A_MINE_SURGING"]
            )
        else:
            return (
                MODES["MODE_B"]
                if total_stock <= config.target_ore_stock_level + EPS
                else MODES["MODE_B_MINE_SURGING"]
            )

    def controller_decision(self):
        c = self.config
        m = self.active_operating_mode.value.name

        if self.is_campaign_complete():
            self.reset_campaign_timer()
            if m == "SHUTDOWN":
                return self._choose_next_campaign_mode(c)
            return MODES["SHUTDOWN"]

        if m.endswith("_CONTINGENCY"):
            self.reset_contingency_timer()
        base = m.replace("_CONTINGENCY", "").replace("_MINE_SURGING", "")
        return MODES[base]


class ConcentratorController(BaseBlendingController):
    def __init__(
        self,
        config: ConcentratorConfig,
        mine: ConcentratorMineFace,
        fleet: ContinuousFleetLogistics,
        plant: ConcentratorPlant,
    ):
        super().__init__(config, mine, fleet, plant)


class MultiFaceConcentratorController(BaseBlendingController):
    """Controller for multi-face mine operations.

    Uses pre-computed fixed face allocation fractions per operating mode,
    computed from each face's generator mean ore fraction. This avoids
    per-timestep linear solves and provides stable campaign-long allocations.

    NOTE: Allocation fractions are computed from face generator means (not
    current parcel values) for stability. The effective ore1 fraction for
    each face is 1.0 - generator.mean_fraction (due to the inversion in
    ContinuousMineFace).

    When the target blend is structurally impossible with the given face
    means, negative face rates are clamped to zero. The resulting stockpile
    imbalance naturally triggers surging mode.
    """

    def __init__(self, config, faces, fleet, plant):
        super().__init__(config, mine=None, fleet=fleet, plant=plant)
        self.faces = list(faces)
        self.total_extra_trucks = drs.Variable("total_extra_trucks", 0.0)
        self.cumulative_mine_development = drs.Level(
            "cumulative_mine_development", initial_value=0.0
        )
        self.face_target_extraction_rates = []
        self.face_real_extraction_rates = []
        self.face_achieved_extraction_rates = []
        self.face_operational_downtime_fractions = []
        self.face_shift_allocation_fractions = []
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
            self.face_target_extraction_rates.append(target)
            self.face_real_extraction_rates.append(real_extraction)
            self.face_achieved_extraction_rates.append(achieved)
            self.face_operational_downtime_fractions.append(operational_downtime)
            self.face_shift_allocation_fractions.append(allocation_fraction)
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
                self.config.mode_a_ore1_milling_rate,
                self.config.mode_a_ore2_milling_rate,
            ),
            "MODE_A_CONTINGENCY": (
                self.config.mode_a_contingency_ore1_milling_rate,
                0.0,
            ),
            "MODE_B": (
                self.config.mode_b_ore1_milling_rate,
                self.config.mode_b_ore2_milling_rate,
            ),
            "MODE_B_CONTINGENCY": (
                0.0,
                self.config.mode_b_contingency_ore2_milling_rate,
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
        values = getattr(self.config, name, None)
        if values is None:
            return default
        if face_index < len(values):
            return values[face_index]
        return default

    def _clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def _refresh_shift_allocation_fractions(self):
        for i, factor in enumerate(self.face_shift_allocation_fractions):
            factor.value = self._face_config_value(
                "face_shift_allocation_fraction", i, 1.0
            )

    def _face_real_extraction_rate(self, face_index, target_extraction_rate):
        c = self.config
        if not getattr(c, "enable_face_capacity_limit", False):
            self.face_operational_downtime_fractions[face_index].value = 0.0
            return target_extraction_rate

        lhd_alloc = self._face_config_value("face_lhd_count", face_index, 0.0)
        truck_alloc = self._face_config_value("face_truck_count", face_index, 0.0)
        distance = self._face_config_value("face_haul_distance", face_index, 0.0)
        accessibility = self._clamp(
            self._face_config_value("face_accessibility_fraction", face_index, 1.0),
            0.0,
            1.0,
        )
        mechanical_availability = getattr(c, "fleet_mechanical_availability", 1.0)

        # 1. Calculate Traffic Delay (function of number of trucks)
        traffic_delay = c.traffic_delay_base + (
            c.traffic_delay_multiplier * truck_alloc
        )

        # 2. Calculate Truck Cycle Time (Travel + Load + Dump + Traffic Delay)
        travel_time = (2 * distance) / c.truck_velocity
        truck_cycle_time = (
            travel_time
            + c.loader_cycle_time_hours
            + c.truck_dump_time_hours
            + traffic_delay
        )

        if truck_cycle_time <= 0 or lhd_alloc <= 0:
            return 0.0

        # 3. Calculate Match Factor
        # MF = (Number of Trucks * Loader Cycle Time) / (Number of Loaders * Truck Cycle Time)
        match_factor = (truck_alloc * c.loader_cycle_time_hours) / (
            lhd_alloc * truck_cycle_time
        )

        # 4. Calculate Efficiency & Throughput based on Match Factor
        if match_factor < 1.0:
            # Under-trucked: Trucks dictate production
            efficiency = match_factor
            max_rate = (
                (truck_alloc / truck_cycle_time) * c.truck_payload_tonnes * 24.0
            )  # tonnes per day
        else:
            # Over-trucked: Loaders dictate production (trucks wait)
            efficiency = 1.0
            max_rate = (
                (lhd_alloc / c.loader_cycle_time_hours) * c.loader_payload_tonnes * 24.0
            )  # tonnes per day

        # Apply accessibility, mechanical availability, and allocation fractions
        allocation_fraction = max(
            0.0, self.face_shift_allocation_fractions[face_index].value
        )
        final_real_extraction_rate = (
            max_rate * accessibility * mechanical_availability * allocation_fraction
        )

        # 5. Calculate Extra Trucks for Development
        # If we are over-trucked (MF > 1), or if real_rate > target_extraction_rate, we have spare trucks
        if (
            final_real_extraction_rate > target_extraction_rate
            and target_extraction_rate > 0
        ):
            # Fraction of fleet actually needed
            utilization = target_extraction_rate / final_real_extraction_rate
            unused_trucks = truck_alloc * (1.0 - utilization)
        else:
            unused_trucks = 0.0

        # Accumulate extra trucks to global development pool
        self.total_extra_trucks.value += unused_trucks

        return max(0.0, final_real_extraction_rate)

    def _reallocate_fleet_for_shift(self):
        mode_name = self.active_operating_mode.value.name
        self.current_shift_allocations = self._get_allocations_for_mode(mode_name)
        self.current_shift_mode_name = mode_name
        self._refresh_shift_allocation_fractions()
        self.fleet_shift_count.value += 1

    def forward(self):
        c = self.config

        self.total_extra_trucks.value = 0.0

        total_extracted = sum(f.cumulative_extracted_mass.value for f in self.faces)
        if abs(total_extracted - c.ore_to_be_extracted_during_warming_period) < 1e-6:
            self.cumulative_time_mode_a.reset()
            self.cumulative_time_mode_a_contingency.reset()
            self.cumulative_time_mode_a_surging.reset()
            self.cumulative_time_mode_b.reset()
            self.cumulative_time_mode_b_contingency.reset()
            self.cumulative_time_mode_b_surging.reset()
            self.cumulative_time_shutdown.reset()

        self.total_system_ore_mass.value = (
            self.parent.ore1_stock.current_mass.value
            + self.parent.ore2_stock.current_mass.value
        )

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
        self.fleet_shift_timer.upper_threshold = c.fleet_shift_duration

        shift_due = self.fleet_shift_timer.value >= c.fleet_shift_duration - 1e-6
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
            self.total_extra_trucks.value * c.development_rate_per_extra_truck
        )

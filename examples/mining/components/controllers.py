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
        self.face_required_rates = []
        self.face_max_extraction_rates = []
        self.face_actual_rates = []
        self.face_effective_delay_factors = []
        self.face_shift_capacity_factors = []
        self.face_target_rates = self.face_actual_rates
        for i in range(len(self.faces)):
            required = drs.Variable(f"face{i}_required_rate", 0.0)
            max_extraction = drs.Variable(f"face{i}_max_extraction_rate", 0.0)
            actual = drs.Variable(f"face{i}_rate", 0.0)
            effective_delay = drs.Variable(f"face{i}_effective_delay_factor", 0.0)
            shift_factor = drs.Variable(f"face{i}_shift_capacity_factor", 1.0)
            setattr(self, f"face{i}_required_rate", required)
            setattr(self, f"face{i}_max_extraction_rate", max_extraction)
            setattr(self, f"face{i}_rate", actual)
            setattr(self, f"face{i}_effective_delay_factor", effective_delay)
            setattr(self, f"face{i}_shift_capacity_factor", shift_factor)
            self.face_required_rates.append(required)
            self.face_max_extraction_rates.append(max_extraction)
            self.face_actual_rates.append(actual)
            self.face_effective_delay_factors.append(effective_delay)
            self.face_shift_capacity_factors.append(shift_factor)
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
            "MODE_A": (self.config.mode_a_ore1_milling_rate, self.config.mode_a_ore2_milling_rate),
            "MODE_A_CONTINGENCY": (self.config.mode_a_contingency_ore1_milling_rate, 0.0),
            "MODE_B": (self.config.mode_b_ore1_milling_rate, self.config.mode_b_ore2_milling_rate),
            "MODE_B_CONTINGENCY": (0.0, self.config.mode_b_contingency_ore2_milling_rate),
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

    def _refresh_shift_capacity_factors(self):
        for i, factor in enumerate(self.face_shift_capacity_factors):
            factor.value = self._face_config_value(
                "face_shift_capacity_factor", i, 1.0
            )

    def _face_effective_delay_factor(self, face_index):
        c = self.config
        base_delay = self._clamp(
            self._face_config_value("face_delay_factor", face_index, 0.0), 0.0, 1.0
        )
        gas_delay = self._clamp(
            self._face_config_value("face_gas_delay_factor", face_index, 0.0),
            0.0,
            1.0,
        )
        truck = max(
            0.0, self._face_config_value("face_truck_allocation", face_index, 0.0)
        )
        congestion_threshold = max(
            0.0,
            self._face_config_value(
                "face_truck_congestion_threshold", face_index, truck
            ),
        )
        excess_truck_allocation = max(0.0, truck - congestion_threshold)
        congestion_delay = (
            max(0.0, c.truck_congestion_delay_sensitivity)
            * excess_truck_allocation
        )
        return self._clamp(base_delay + gas_delay + congestion_delay, 0.0, 1.0)

    def _face_max_extraction_rate(self, face_index, required_rate):
        c = self.config
        if not getattr(c, "enable_face_capacity_limit", False):
            self.face_effective_delay_factors[face_index].value = 0.0
            return required_rate

        lhd = self._face_config_value("face_lhd_allocation", face_index, 0.0)
        truck = self._face_config_value("face_truck_allocation", face_index, 0.0)
        availability = self._clamp(
            self._face_config_value("face_availability", face_index, 1.0),
            0.0,
            1.0,
        )
        distance = self._face_config_value("face_haul_distance", face_index, 0.0)
        delay = self._face_effective_delay_factor(face_index)
        self.face_effective_delay_factors[face_index].value = delay
        shift_factor = max(0.0, self.face_shift_capacity_factors[face_index].value)

        lhd_capacity = max(0.0, lhd) * c.lhd_capacity_per_allocation
        truck_capacity = max(0.0, truck) * c.truck_capacity_per_allocation
        resource_capacity = min(lhd_capacity, truck_capacity)
        distance_excess = max(0.0, distance - c.haul_distance_reference)
        distance_factor = 1.0 / (1.0 + c.haul_distance_sensitivity * distance_excess)
        productive_time = 1.0 - delay

        max_extraction_rate = (
            resource_capacity
            * availability
            * productive_time
            * distance_factor
            * shift_factor
        )
        return max(0.0, max_extraction_rate)

    def _reallocate_fleet_for_shift(self):
        mode_name = self.active_operating_mode.value.name
        self.current_shift_allocations = self._get_allocations_for_mode(mode_name)
        self.current_shift_mode_name = mode_name
        self._refresh_shift_capacity_factors()
        self.fleet_shift_count.value += 1

    def forward(self):
        c = self.config

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
                required_rate = targets.extraction_rate * fracs[i]
                max_extraction_rate = self._face_max_extraction_rate(i, required_rate)
                self.face_required_rates[i].value = required_rate
                self.face_max_extraction_rates[i].value = max_extraction_rate
                self.face_actual_rates[i].value = min(
                    required_rate, max_extraction_rate
                )
        else:
            for i, _face in enumerate(self.faces):
                self.face_required_rates[i].value = 0.0
                self.face_max_extraction_rates[i].value = 0.0
                self.face_actual_rates[i].value = 0.0
                self.face_effective_delay_factors[i].value = 0.0

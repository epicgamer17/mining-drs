import math

import drs
from .modes import MODES, RequireDecision
from .mine_face import BaseMineFace, ConcentratorMineFace
from .fleet import ContinuousFleetLogistics
from .plant import BaseMetallurgicalPlant, ConcentratorPlant


class BaseBlendingController(drs.Module):
    """State container and mode bookkeeping for the blending controller.

    The controller owns configuration, state, and the operating-mode /
    duration-timer bookkeeping exposed through ``update_mode``, the target
    rate computation exposed through ``get_target_rates``, and (for the
    multi-face variant) fleet-shift scheduling and face drive allocation.
    Policy-level decisions (target extraction rates, fleet shift allocation,
    recovery of the RL action) live in the top-level control policy
    (drs_mining.control).
    """

    _MODE_TIMER_ATTRS = {
        "MODE_A": "cumulative_time_mode_a",
        "MODE_A_CONTINGENCY": "cumulative_time_mode_a_contingency",
        "MODE_A_MINE_SURGING": "cumulative_time_mode_a_surging",
        "MODE_B": "cumulative_time_mode_b",
        "MODE_B_CONTINGENCY": "cumulative_time_mode_b_contingency",
        "MODE_B_MINE_SURGING": "cumulative_time_mode_b_surging",
        "SHUTDOWN": "cumulative_time_shutdown",
    }

    _RATE_MAP = {
        "MODE_A": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
        "MODE_A_CONTINGENCY": ("mode_a_contingency_ore1_milling_rate", None),
        "MODE_A_MINE_SURGING": ("mode_a_ore1_milling_rate", "mode_a_ore2_milling_rate"),
        "MODE_B": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
        "MODE_B_CONTINGENCY": (None, "mode_b_contingency_ore2_milling_rate"),
        "MODE_B_MINE_SURGING": ("mode_b_ore1_milling_rate", "mode_b_ore2_milling_rate"),
        "SHUTDOWN": (None, None),
    }

    _DEFAULT_RATES = {
        "mode_a_ore1_milling_rate": 3600.0,
        "mode_a_ore2_milling_rate": 2400.0,
        "mode_a_contingency_ore1_milling_rate": 3900.0,
        "mode_b_ore1_milling_rate": 4600.0,
        "mode_b_ore2_milling_rate": 800.0,
        "mode_b_contingency_ore2_milling_rate": 2500.0,
    }

    _CONTINGENCY_MODES = {"MODE_A_CONTINGENCY", "MODE_B_CONTINGENCY"}

    _RL_ACTION_MODES = [
        MODES["MODE_A"],
        MODES["MODE_B"],
        MODES["MODE_A_MINE_SURGING"],
        MODES["MODE_B_MINE_SURGING"],
    ]

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
        total_ore_to_extract: float = 6600000.0,
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
        self.total_ore_to_extract = total_ore_to_extract

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

    def update_mode(self, ore1_stock, ore2_stock) -> str:
        """High-level mode bookkeeping for one engine step.

        * Resets cumulative per-mode timers on a fresh simulation start,
        * refreshes ``total_system_ore_mass`` from the two stockpiles,
        * evaluates and applies the next operating mode,
        * updates the campaign / contingency duration timers, and
        * wires the system ore mass ``rate`` to the stockpile mass rates.

        ``ore1_stock`` and ``ore2_stock`` are the two stockpile components
        feeding the concentrator; their current masses drive the mode
        transitions. Returns the name of the active mode after the
        transition so callers can detect mode changes.
        """
        self._reset_mode_timers_if_fresh()
        self.total_system_ore_mass.value = (
            ore1_stock.current_mass.value + ore2_stock.current_mass.value
        )

        next_mode = self._next_mode(ore1_stock, ore2_stock)
        if next_mode is not None:
            self.active_operating_mode.value = next_mode

        name = self.active_operating_mode.value.name
        self._update_mode_timers(name)

        self.total_system_ore_mass.rate = (
            ore1_stock.current_mass.rate + ore2_stock.current_mass.rate
        )
        return name

    # Backwards-compatible alias for the pre-Step-3 name.
    step_mode = update_mode

    def get_target_rates(self, mode, fleet=None):
        """Compute the target extraction and stockpile outflow rates for a mode.

        ``mode`` is the active operating-mode name returned by
        :meth:`update_mode`. ``fleet`` supplies the current routing fraction so
        that surging can back out the extraction rate required to fill the
        depleted stockpile. Returns ``(mine_target, stock1_target,
        stock2_target)`` and also stores them on the controller (used as
        telemetry channels).
        """
        name = mode.name if hasattr(mode, "name") else str(mode)
        ore1, ore2 = self._read_rates(name)

        if "_MINE_SURGING" in name:
            self.total_system_ore_mass.lower_threshold = self.target_ore_stock_level
            p = (
                fleet.stockpile2_routing_fraction.value
                if fleet is not None
                else 0.0
            )
            if p <= 1e-4 and getattr(self, "mine", None) is not None:
                p = self.mine._get_current_attr_value()
            if name == "MODE_A_MINE_SURGING":
                effective_fraction = max(1.0 - p, 0.01)
                extraction = ore1 / effective_fraction
            else:
                effective_fraction = max(p, 0.01)
                extraction = ore2 / effective_fraction
        else:
            extraction = ore1 + ore2

        self.target_mine_mass_rate.value = extraction
        self.target_stock1_outflow_rate.value = ore1
        self.target_stock2_outflow_rate.value = ore2
        return extraction, ore1, ore2

    def _read_rates(self, name):
        """Read the per-mode milling-rate pair (ore1, ore2) for ``name``."""
        ore1_attr, ore2_attr = self._RATE_MAP.get(name, (None, None))
        ore1 = (
            getattr(self, ore1_attr, self._DEFAULT_RATES.get(ore1_attr, 0.0))
            if ore1_attr
            else 0.0
        )
        ore2 = (
            getattr(self, ore2_attr, self._DEFAULT_RATES.get(ore2_attr, 0.0))
            if ore2_attr
            else 0.0
        )
        return ore1, ore2

    def _reset_mode_timers_if_fresh(self):
        sources = getattr(self, "faces", None) or (
            [self.mine] if self.mine is not None else []
        )
        if sources and abs(sum(s.net_extracted_mass for s in sources)) < 1e-6:
            for timer_name in self._MODE_TIMER_ATTRS.values():
                getattr(self, timer_name).reset()

    def _next_mode(self, ore1_stock, ore2_stock):
        name = self.active_operating_mode.value.name
        eps = getattr(self, "stockout_epsilon", 1e-9)
        target_stock = getattr(self, "target_ore_stock_level", 60000.0)

        if self._campaign_complete():
            if name == "SHUTDOWN":
                rl_action = getattr(self, "pending_rl_action", None)
                if rl_action is not None:
                    self.current_campaign_duration.reset()
                    self.pending_rl_action = None
                    return self._RL_ACTION_MODES[rl_action]
                if hasattr(self, "pending_rl_action"):
                    raise RequireDecision()
            self.current_campaign_duration.reset()
            return (
                self._choose_next_campaign_mode(ore2_stock)
                if name == "SHUTDOWN"
                else MODES["SHUTDOWN"]
            )

        if name == "SHUTDOWN":
            return None

        ore1 = ore1_stock.current_mass.value
        ore2 = ore2_stock.current_mass.value

        if "_CONTINGENCY" in name:
            if self._contingency_complete():
                self.current_contingency_duration.reset()
                return MODES[name.replace("_CONTINGENCY", "")]
            base = name.replace("_CONTINGENCY", "")
            if base == "MODE_A" and ore1 <= eps:
                return MODES[base + "_MINE_SURGING"]
            if base == "MODE_B" and ore2 <= eps:
                return MODES[base + "_MINE_SURGING"]
            return None

        if "_MINE_SURGING" in name:
            if self.total_system_ore_mass.value <= target_stock + 1e-6:
                return MODES[name.replace("_MINE_SURGING", "")]
            return None

        if name == "MODE_A":
            if ore1 <= eps:
                return MODES[name + "_MINE_SURGING"]
            if ore2 <= eps:
                self.current_contingency_duration.reset()
                return MODES[name + "_CONTINGENCY"]
            return None

        if name == "MODE_B":
            if ore1 <= eps:
                self.current_contingency_duration.reset()
                return MODES[name + "_CONTINGENCY"]
            if ore2 <= eps:
                return MODES[name + "_MINE_SURGING"]
            return None

        return None

    def _campaign_complete(self) -> bool:
        threshold = (
            self.duration_of_shutdowns
            if self.active_operating_mode.value.name == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        self.current_campaign_duration.upper_threshold = threshold
        return self.current_campaign_duration.value >= (threshold - 1e-6)

    def _contingency_complete(self) -> bool:
        threshold = self.duration_of_contingency_segments
        self.current_contingency_duration.upper_threshold = threshold
        return self.current_contingency_duration.value >= (threshold - 1e-6)

    def _choose_next_campaign_mode(self, ore2_stock):
        ore2 = ore2_stock.current_mass.value
        total_stock = self.total_system_ore_mass.value
        EPS = 1e-6
        if ore2 > self.critical_ore2_level:
            return (
                MODES["MODE_A"]
                if total_stock <= self.target_ore_stock_level + EPS
                else MODES["MODE_A_MINE_SURGING"]
            )
        return (
            MODES["MODE_B"]
            if total_stock <= self.target_ore_stock_level + EPS
            else MODES["MODE_B_MINE_SURGING"]
        )

    def _update_mode_timers(self, name):
        for timer_name in self._MODE_TIMER_ATTRS.values():
            getattr(self, timer_name).rate = 0.0
        timer_attr = self._MODE_TIMER_ATTRS.get(name)
        if timer_attr:
            getattr(self, timer_attr).rate = 1.0
        self.current_campaign_duration.rate = 1.0
        self.current_campaign_duration.upper_threshold = (
            self.duration_of_shutdowns
            if name == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        if name in self._CONTINGENCY_MODES:
            self.current_contingency_duration.rate = 1.0
            self.current_contingency_duration.upper_threshold = (
                self.duration_of_contingency_segments
            )
        else:
            self.current_contingency_duration.rate = 0.0

    def is_terminating_condition_met(self) -> bool:
        """True once the combined extraction of every mine source reaches the target."""
        sources = getattr(self, "faces", None) or []
        if not sources and self.mine is not None:
            sources = [self.mine]
        return (
            sum(s.cumulative_extracted_mass.value for s in sources)
            >= self.total_ore_to_extract
        )

    @property
    def state_components(self) -> list:
        """Stateful leaf components owned by this controller (mine, faces, levels, timers).

        These are registered with the engine so it can compute time-to-event
        boundaries and advance state without recursively inspecting the model.
        """
        comps = []
        if self.mine is not None:
            comps.append(self.mine)
        for face in getattr(self, "faces", []) or []:
            comps.append(face)
        comps.append(self.total_system_ore_mass)
        comps.extend(
            [
                self.current_campaign_duration,
                self.current_contingency_duration,
                self.cumulative_time_mode_a,
                self.cumulative_time_mode_a_contingency,
                self.cumulative_time_mode_a_surging,
                self.cumulative_time_mode_b,
                self.cumulative_time_mode_b_contingency,
                self.cumulative_time_mode_b_surging,
                self.cumulative_time_shutdown,
            ]
        )
        fleet_shift_timer = getattr(self, "fleet_shift_timer", None)
        if fleet_shift_timer is not None:
            comps.append(fleet_shift_timer)
        return comps

    def time_to_event(self) -> float:
        """Time until this controller's owned levels hit a state boundary.

        Only the controller's own levels (mode timers, durations, system ore
        mass, fleet-shift timer) are considered, so the controller can be
        registered on the engine alongside its mine/faces/fleet/plant without
        double-counting their boundaries.
        """
        min_dt = math.inf
        for level in self._owned_levels():
            dt = level.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt

    def step(self, dt: float) -> None:
        """Advance this controller's owned levels forward by ``dt``."""
        for level in self._owned_levels():
            level.step(dt)


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
        total_ore_to_extract: float = 6600000.0,
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
            total_ore_to_extract=total_ore_to_extract,
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
        total_ore_to_extract: float = 6600000.0,
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
            total_ore_to_extract=total_ore_to_extract,
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

    def schedule_fleet_shifts(self, mode_name):
        """Drive the fleet-shift timer and reallocate the fleet when due.

        ``mode_name`` is the active mode returned by ``update_mode``. The
        fleet is reallocated on the first step, when the shift timer elapses,
        or when the operating mode changes.
        """
        self.fleet_shift_timer.rate = 1.0
        self.fleet_shift_timer.upper_threshold = self.fleet_shift_duration

        shift_due = self.fleet_shift_timer.value >= self.fleet_shift_duration - 1e-6
        mode_changed = mode_name != self.current_shift_mode_name
        if self.current_shift_allocations is None or shift_due or mode_changed:
            self.fleet_shift_timer.reset()
            self._reallocate_fleet_for_shift()

    def drive_faces(self, mine_target):
        """Split ``mine_target`` across the faces and drive each face's rate.

        Resets the spare-truck tally, computes each face's target/real/achieved
        extraction rates (using the precomputed per-mode allocations and the
        physical fleet model), stamps the per-face telemetry variables, and
        applies the achieved rate to each face. Parcel mechanics run inside
        each face's ``step``.
        """
        self.total_extra_trucks.value = 0.0
        fracs = self.current_shift_allocations
        for i, face in enumerate(self.faces):
            if fracs is not None and i < len(fracs):
                target = mine_target * fracs[i]
                real = self._face_real_extraction_rate(i, target)
                self.face_target_extraction_rates[i].value = target
                self.face_real_extraction_rates[i].value = real
                self.face_achieved_extraction_rates[i].value = min(target, real)
                face.target_rate = self.face_achieved_extraction_rates[i].value
            else:
                self.face_target_extraction_rates[i].value = 0.0
                self.face_real_extraction_rates[i].value = 0.0
                self.face_achieved_extraction_rates[i].value = 0.0
                self.face_operational_downtime_fractions[i].value = 0.0
                face.target_rate = 0.0

        self.cumulative_mine_development.rate = (
            self.total_extra_trucks.value * self.development_rate_per_extra_truck
        )


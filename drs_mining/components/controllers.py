import math
from typing import List, Dict, Mapping, Sequence
import drs
from .modes import MODES, RequireDecision
from .mine_face import MineFace
from .fleet import ContinuousFleetLogistics
from .plant import MetallurgicalPlant


class BlendingController(drs.Module):
    r"""Unified state container and mode/allocation bookkeeping for blending control."""

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

    _CONTINGENCY_MODES = {"MODE_A_CONTINGENCY", "MODE_B_CONTINGENCY"}

    _RL_ACTION_MODES = [
        MODES["MODE_A"],
        MODES["MODE_B"],
        MODES["MODE_A_MINE_SURGING"],
        MODES["MODE_B_MINE_SURGING"],
    ]

    def __init__(
        self,
        faces: Sequence[MineFace],
        fleet: ContinuousFleetLogistics,
        plant: MetallurgicalPlant,
        target_ore_stock_level: float,
        critical_ore2_level: float,
        total_ore_to_extract: float,
        duration_of_production_campaigns: float,
        duration_of_shutdowns: float,
        duration_of_contingency_segments: float,
        mode_a_ore1_milling_rate: float,
        mode_a_ore2_milling_rate: float,
        mode_a_contingency_ore1_milling_rate: float,
        mode_b_ore1_milling_rate: float,
        mode_b_ore2_milling_rate: float,
        mode_b_contingency_ore2_milling_rate: float,
        ore_to_be_extracted_during_warming_period: float,
        fleet_shift_duration: float,
        total_lhd_count: float,
        total_truck_count: float,
        max_lhds_per_face: Sequence[float],
        max_trucks_per_face: Sequence[float],
        face_haul_distance: Sequence[float],
        face_accessibility_fraction: Sequence[float],
        truck_velocity: float,
        loader_cycle_time_hours: float,
        truck_dump_time_hours: float,
        traffic_delay_per_truck_hours: float,
        fleet_mechanical_availability: float,
        loader_payload_tonnes: float,
        truck_payload_tonnes: float,
        development_rate_per_extra_truck: float,
        mode_allocations: Mapping[str, Sequence[float]],
    ):
        super().__init__()
        self.faces = list(faces) if isinstance(faces, (list, tuple)) else [faces]
        if not self.faces:
            raise ValueError("BlendingController requires at least one MineFace in faces.")

        self.mine = self.faces[0] if len(self.faces) == 1 else None
        self.fleet = fleet
        self.plant = plant
        self.target_ore_stock_level = target_ore_stock_level
        self.critical_ore2_level = critical_ore2_level
        self.duration_of_production_campaigns = duration_of_production_campaigns
        self.duration_of_shutdowns = duration_of_shutdowns
        self.duration_of_contingency_segments = duration_of_contingency_segments
        self.ore_to_be_extracted_during_warming_period = ore_to_be_extracted_during_warming_period
        self.total_ore_to_extract = total_ore_to_extract

        self.mode_a_ore1_milling_rate = mode_a_ore1_milling_rate
        self.mode_a_ore2_milling_rate = mode_a_ore2_milling_rate
        self.mode_a_contingency_ore1_milling_rate = mode_a_contingency_ore1_milling_rate
        self.mode_b_ore1_milling_rate = mode_b_ore1_milling_rate
        self.mode_b_ore2_milling_rate = mode_b_ore2_milling_rate
        self.mode_b_contingency_ore2_milling_rate = mode_b_contingency_ore2_milling_rate

        self.fleet_shift_duration = fleet_shift_duration
        self.total_lhd_count = total_lhd_count
        self.total_truck_count = total_truck_count

        n_faces = len(self.faces)
        self.max_lhds_per_face = self._normalize_param_list(max_lhds_per_face, n_faces, 2.0)
        self.max_trucks_per_face = self._normalize_param_list(max_trucks_per_face, n_faces, 6.0)
        self.face_haul_distance = self._normalize_param_list(face_haul_distance, n_faces, 0.0)
        self.face_accessibility_fraction = [
            max(0.0, min(1.0, v))
            for v in self._normalize_param_list(face_accessibility_fraction, n_faces, 1.0)
        ]

        self.truck_velocity = truck_velocity
        self.loader_cycle_time_hours = loader_cycle_time_hours
        self.truck_dump_time_hours = truck_dump_time_hours
        self.traffic_delay_per_truck_hours = traffic_delay_per_truck_hours
        self.fleet_mechanical_availability = fleet_mechanical_availability
        self.loader_payload_tonnes = loader_payload_tonnes
        self.truck_payload_tonnes = truck_payload_tonnes
        self.development_rate_per_extra_truck = development_rate_per_extra_truck

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

        self._mode_allocations = dict(mode_allocations) if mode_allocations else self._precompute_allocations()
        self.current_shift_allocations = None
        self.current_shift_mode_name = None
        self.fleet_shift_timer = drs.Timer("fleet_shift_timer", initial_value=0.0)
        self.fleet_shift_count = drs.Variable("fleet_shift_count", 0)

    @staticmethod
    def _normalize_param_list(param, count: int, default_val: float) -> List[float]:
        if isinstance(param, (int, float)):
            return [float(param)] * count
        lst = [float(x) for x in param]
        if len(lst) >= count:
            return lst[:count]
        if len(lst) > 0:
            return lst + [lst[-1]] * (count - len(lst))
        return [default_val] * count

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

    def active_duration(self, current_time: float = -1.0) -> float:
        """Encapsulate state calculations for active operational duration."""
        if current_time < 0.0:
            current_time = self.total_duration
        return max(0.0, current_time - self.cumulative_time_shutdown.value)

    def update_mode(self, ore1_stock, ore2_stock) -> str:
        """High-level mode bookkeeping for one engine step."""
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

    def get_target_rates(self, mode, fleet=None):
        """Compute the target extraction and stockpile outflow rates for a mode."""
        name = mode.name if hasattr(mode, "name") else str(mode)
        ore1, ore2 = self._read_rates(name)
        eff_fleet = fleet or self.fleet

        if "_MINE_SURGING" in name:
            self.total_system_ore_mass.lower_threshold = self.target_ore_stock_level
            p = eff_fleet.stockpile2_routing_fraction.value if eff_fleet else 0.0
            if p <= 1e-4 and len(self.faces) > 0:
                p = self.faces[0]._get_current_attr_value()
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

    def _read_rates(self, name: str) -> tuple:
        """Read the per-mode milling-rate pair (ore1, ore2) for ``name``."""
        ore1_attr, ore2_attr = self._RATE_MAP.get(name, (None, None))
        ore1 = getattr(self, ore1_attr, 0.0) if ore1_attr else 0.0
        ore2 = getattr(self, ore2_attr, 0.0) if ore2_attr else 0.0
        return ore1, ore2

    def _reset_mode_timers_if_fresh(self):
        if self.faces and abs(sum(s.net_extracted_mass for s in self.faces)) < 1e-6:
            for timer_name in self._MODE_TIMER_ATTRS.values():
                getattr(self, timer_name).reset()

    def _next_mode(self, ore1_stock, ore2_stock):
        name = self.active_operating_mode.value.name
        eps = 1e-9
        target_stock = self.target_ore_stock_level

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
        return (
            sum(s.cumulative_extracted_mass.value for s in self.faces)
            >= self.total_ore_to_extract
        )

    @property
    def state_components(self) -> list:
        """Stateful leaf components owned by this controller (mine faces and controller module)."""
        comps = list(self.faces)
        comps.append(self)
        return comps

    def time_to_event(self) -> float:
        min_dt = math.inf
        for level in self._owned_levels():
            dt = level.time_to_event()
            if 0.0 <= dt < min_dt:
                min_dt = dt
        return min_dt

    def step(self, dt: float) -> None:
        for level in self._owned_levels():
            level.step(dt)

    def _precompute_allocations(self):
        """Pre-compute face extraction fractions per mode using face mean ore fractions."""
        num_faces = len(self.faces)
        if num_faces == 0:
            return {}

        face_ore1_fracs = [1.0 - f.mean_ore_fraction for f in self.faces]

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

        if num_faces == 1:
            for mode_name in modes_to_compute:
                result[mode_name] = [1.0]
            result["MODE_A_MINE_SURGING"] = [1.0]
            result["MODE_B_MINE_SURGING"] = [1.0]
            return result

        if num_faces == 2:
            f1, f2 = face_ore1_fracs[0], face_ore1_fracs[1]
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

            if f1 >= f2:
                result["MODE_A_MINE_SURGING"] = [1.0, 0.0]
                result["MODE_B_MINE_SURGING"] = [0.0, 1.0]
            else:
                result["MODE_A_MINE_SURGING"] = [0.0, 1.0]
                result["MODE_B_MINE_SURGING"] = [1.0, 0.0]
            return result

        # General N-face case (N >= 3)
        idx_max_ore1 = max(range(num_faces), key=lambda i: face_ore1_fracs[i])
        idx_min_ore1 = min(range(num_faces), key=lambda i: face_ore1_fracs[i])

        for mode_name, (ore1, ore2) in modes_to_compute.items():
            total = ore1 + ore2
            if total <= 0:
                result[mode_name] = [1.0 / num_faces] * num_faces
                continue
            target_ore1_ratio = ore1 / total

            below = [i for i in range(num_faces) if face_ore1_fracs[i] <= target_ore1_ratio]
            above = [i for i in range(num_faces) if face_ore1_fracs[i] > target_ore1_ratio]

            if not below:
                fracs = [0.0] * num_faces
                fracs[idx_min_ore1] = 1.0
            elif not above:
                fracs = [0.0] * num_faces
                fracs[idx_max_ore1] = 1.0
            else:
                mean_below = sum(face_ore1_fracs[i] for i in below) / len(below)
                mean_above = sum(face_ore1_fracs[i] for i in above) / len(above)

                if abs(mean_above - mean_below) < 1e-12:
                    alpha = 0.5
                else:
                    alpha = (target_ore1_ratio - mean_below) / (mean_above - mean_below)
                    alpha = max(0.0, min(1.0, alpha))

                fracs = [0.0] * num_faces
                for i in above:
                    fracs[i] = alpha / len(above)
                for i in below:
                    fracs[i] = (1.0 - alpha) / len(below)

            result[mode_name] = fracs

        surging_a = [0.0] * num_faces
        surging_a[idx_max_ore1] = 1.0
        result["MODE_A_MINE_SURGING"] = surging_a

        surging_b = [0.0] * num_faces
        surging_b[idx_min_ore1] = 1.0
        result["MODE_B_MINE_SURGING"] = surging_b

        return result

    def _get_allocations_for_mode(self, mode_name: str) -> list:
        fracs = self._mode_allocations.get(mode_name)
        if fracs is None:
            base_key = mode_name.replace("_MINE_SURGING", "")
            fracs = self._mode_allocations.get(base_key)
        return fracs

    def _refresh_shift_allocation_fractions(self):
        if not self.current_shift_allocations:
            return
        for i, factor in enumerate(self.face_shift_allocation_fractions):
            factor.value = self.current_shift_allocations[i]

        self._distribute_discrete_fleet(self.current_shift_allocations)

    def _distribute_discrete_fleet(self, fracs: list):
        total_lhd = int(self.total_lhd_count)
        total_truck = int(self.total_truck_count)

        num_faces = len(self.faces)
        if num_faces == 0:
            return

        unassigned_lhds = total_lhd
        lhd_assignments = [0] * num_faces
        face_priorities = sorted(range(num_faces), key=lambda i: fracs[i], reverse=True)

        for i in face_priorities:
            max_lhds = int(self.max_lhds_per_face[i])
            target = math.floor(total_lhd * fracs[i])
            assigned = min(target, max_lhds, unassigned_lhds)
            lhd_assignments[i] = assigned
            unassigned_lhds -= assigned

        for i in face_priorities:
            max_lhds = int(self.max_lhds_per_face[i])
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
            max_trucks = int(self.max_trucks_per_face[i])
            target = math.floor(total_truck * fracs[i])
            assigned = min(target, max_trucks, unassigned_trucks)
            truck_assignments[i] = assigned
            unassigned_trucks -= assigned

        for i in face_priorities:
            max_trucks = int(self.max_trucks_per_face[i])
            if unassigned_trucks <= 0:
                break
            if truck_assignments[i] < max_trucks:
                truck_assignments[i] += 1
                unassigned_trucks -= 1

        for i in range(num_faces):
            self.face_truck_allocations[i].value = float(truck_assignments[i])

    def _face_real_extraction_rate(self, face_index: int, target_extraction_rate: float) -> float:
        lhd_alloc = self.face_lhd_allocations[face_index].value
        truck_alloc = self.face_truck_allocations[face_index].value

        distance = self.face_haul_distance[face_index]
        accessibility = self.face_accessibility_fraction[face_index]
        mechanical_availability = self.fleet_mechanical_availability

        traffic_delay = self.traffic_delay_per_truck_hours * truck_alloc
        travel_time = (2 * distance) / self.truck_velocity if self.truck_velocity > 0 else 0.0

        truck_loading_time_hours = (
            self.loader_cycle_time_hours * (self.truck_payload_tonnes / self.loader_payload_tonnes)
            if self.loader_payload_tonnes > 0
            else 0.0
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
                if self.loader_cycle_time_hours > 0
                else 0.0
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

    def schedule_fleet_shifts(self, mode_name: str):
        """Drive the fleet-shift timer and reallocate the fleet when due."""
        self.fleet_shift_timer.rate = 1.0
        self.fleet_shift_timer.upper_threshold = self.fleet_shift_duration

        shift_due = self.fleet_shift_timer.value >= self.fleet_shift_duration - 1e-6
        mode_changed = mode_name != self.current_shift_mode_name
        if self.current_shift_allocations is None or shift_due or mode_changed:
            self.fleet_shift_timer.reset()
            self._reallocate_fleet_for_shift()

    def drive_faces(self, mine_target: float):
        """Split ``mine_target`` across the faces and drive each face's rate."""
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

import math
from typing import List, Dict, Mapping, Sequence, Optional, Union, Tuple
import drs
from .modes import OperatingMode, RequireDecision
from .mine_face import MineFace
from drs_mining.config import MILL_MODES


class OperatingModeController(drs.Module):
    """High-level supervisory controller that manages campaign timers and operating mode transitions."""

    def __init__(
        self,
        duration_of_production_campaigns: float,
        duration_of_shutdowns: float,
        critical_ore2_level: float,
        target_ore_stock_level: float = 60000.0,
        total_ore_to_extract: float = 6600000.0,
        initial_mode: OperatingMode = MILL_MODES["MODE_A"],
    ):
        super().__init__()
        self.duration_of_production_campaigns = duration_of_production_campaigns
        self.duration_of_shutdowns = duration_of_shutdowns
        self.critical_ore2_level = critical_ore2_level
        self.target_ore_stock_level = target_ore_stock_level
        self.total_ore_to_extract = total_ore_to_extract

        self.active_campaign_mode = drs.Variable("active_campaign_mode", initial_mode)
        self.active_operating_mode = self.active_campaign_mode

        self.current_campaign_duration = drs.Timer(
            "current_campaign_duration", initial_value=0.0
        )

    def update(
        self,
        ore2_stock_level: float,
        total_stock_level: Optional[float] = None,
    ) -> OperatingMode:
        """Advance campaign timer and determine active campaign mode (MODE_A, MODE_B, SHUTDOWN)."""
        name = self.active_campaign_mode.value.name

        if self._campaign_complete():
            if name == "SHUTDOWN":
                rl_action = getattr(self, "pending_rl_action", None)
                if rl_action is not None:
                    self.current_campaign_duration.reset()
                    self.pending_rl_action = None
                    action_modes = [
                        MILL_MODES["MODE_A"],
                        MILL_MODES["MODE_B"],
                        MILL_MODES["MODE_A_MINE_SURGING"],
                        MILL_MODES["MODE_B_MINE_SURGING"],
                    ]
                    next_mode = action_modes[rl_action]
                    self.active_campaign_mode.value = next_mode
                    self._update_campaign_timers(next_mode.name)
                    return next_mode

                if hasattr(self, "pending_rl_action"):
                    raise RequireDecision()

                self.current_campaign_duration.reset()
                next_mode = self._choose_next_campaign_mode(
                    ore2_stock_level, total_stock_level
                )
                self.active_campaign_mode.value = next_mode
            else:
                self.current_campaign_duration.reset()
                self.active_campaign_mode.value = MILL_MODES["SHUTDOWN"]

        active_name = self.active_campaign_mode.value.name
        self._update_campaign_timers(active_name)
        return self.active_campaign_mode.value

    def _choose_next_campaign_mode(
        self, ore2_stock_level: float, total_stock_level: Optional[float] = None
    ) -> OperatingMode:
        if ore2_stock_level > self.critical_ore2_level:
            return MILL_MODES["MODE_A"]
        return MILL_MODES["MODE_B"]

    def _campaign_complete(self) -> bool:
        threshold = (
            self.duration_of_shutdowns
            if self.active_campaign_mode.value.name == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        self.current_campaign_duration.upper_threshold = threshold
        return self.current_campaign_duration.value >= (threshold - 1e-6)

    def _update_campaign_timers(self, name: str):
        threshold = (
            self.duration_of_shutdowns
            if name == "SHUTDOWN"
            else self.duration_of_production_campaigns
        )
        self.current_campaign_duration.rate = (1.0, -math.inf, threshold)

    def is_terminating_condition_met(self) -> bool:
        return False


class FleetController(drs.Module):
    """Underground haulage & extraction controller managing discrete fleet allocation,
    cycle times, match factors, and per-face extraction rates.
    """

    def __init__(
        self,
        faces: Sequence[MineFace],
        fleet_shift_duration: float = 0.5,
        total_lhd_count: float = 3.0,
        total_truck_count: float = 10.0,
        max_lhds_per_face: Sequence[float] = (2.0,),
        max_trucks_per_face: Sequence[float] = (6.0,),
        face_haul_distance: Sequence[float] = (1.5, 2.2),
        face_accessibility_fraction: Sequence[float] = (0.93, 0.91),
        truck_velocity: float = 15.0,
        loader_cycle_time_hours: float = 0.0833,
        truck_dump_time_hours: float = 0.033,
        traffic_delay_per_truck_hours: float = 0.015,
        fleet_mechanical_availability: float = 0.85,
        loader_payload_tonnes: float = 15.0,
        truck_payload_tonnes: float = 30.0,
        development_rate_per_extra_truck: float = 50.0,
        mode_allocations: Optional[Mapping[str, Sequence[float]]] = None,
        mode_rates: Optional[Mapping[str, Tuple[float, float]]] = None,
    ):
        super().__init__()
        self.faces = list(faces) if isinstance(faces, (list, tuple)) else [faces]
        if not self.faces:
            raise ValueError("FleetController requires at least one MineFace in faces.")

        self.fleet_shift_duration = fleet_shift_duration
        self.total_lhd_count = total_lhd_count
        self.total_truck_count = total_truck_count

        n_faces = len(self.faces)
        self.max_lhds_per_face = self._normalize_param_list(
            max_lhds_per_face, n_faces, 2.0
        )
        self.max_trucks_per_face = self._normalize_param_list(
            max_trucks_per_face, n_faces, 6.0
        )
        self.face_haul_distance = self._normalize_param_list(
            face_haul_distance, n_faces, 0.0
        )
        self.face_accessibility_fraction = [
            max(0.0, min(1.0, v))
            for v in self._normalize_param_list(
                face_accessibility_fraction, n_faces, 1.0
            )
        ]

        self.truck_velocity = truck_velocity
        self.loader_cycle_time_hours = loader_cycle_time_hours
        self.truck_dump_time_hours = truck_dump_time_hours
        self.traffic_delay_per_truck_hours = traffic_delay_per_truck_hours
        self.fleet_mechanical_availability = fleet_mechanical_availability
        self.loader_payload_tonnes = loader_payload_tonnes
        self.truck_payload_tonnes = truck_payload_tonnes
        self.development_rate_per_extra_truck = development_rate_per_extra_truck

        self.fleet_shift_timer = drs.Timer("fleet_shift_timer", initial_value=0.0)
        self.fleet_shift_count = drs.Variable("fleet_shift_count", 0)
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

        self._mode_allocations = (
            dict(mode_allocations)
            if mode_allocations
            else self._precompute_allocations(mode_rates)
        )
        self.current_shift_allocations = None
        self.current_shift_mode_name = None

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

    @staticmethod
    def _allocations_for_ore_fracs(
        face_ore1_fracs: Sequence[float],
        modes_to_compute: Mapping[str, Tuple[float, float]],
    ) -> Dict[str, List[float]]:
        """Solve per-mode face-allocation fractions for a given set of faces.

        For two faces this is the exact Appendix-A blending solution (face with the
        highest Ore-1 fraction carries the richest feed); for N >= 3 a ratio-matched
        alpha blend across faces above/below the target Ore-1 ratio.
        """
        num_faces = len(face_ore1_fracs)
        result = {}

        if num_faces == 0:
            return result

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

            below = [
                i for i in range(num_faces) if face_ore1_fracs[i] <= target_ore1_ratio
            ]
            above = [
                i for i in range(num_faces) if face_ore1_fracs[i] > target_ore1_ratio
            ]

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

    def _precompute_allocations(
        self, mode_rates: Optional[Mapping[str, Tuple[float, float]]] = None
    ) -> Dict[str, List[float]]:
        face_ore1_fracs = [1.0 - f.mean_ore_fraction for f in self.faces]

        default_rates = {
            "MODE_A": (540.0 * 24.0, 60.0 * 24.0),
            "MODE_A_CONTINGENCY": (500.0 * 24.0, 0.0),
            "MODE_B": (300.0 * 24.0, 300.0 * 24.0),
            "MODE_B_CONTINGENCY": (0.0, 450.0 * 24.0),
        }
        self._mode_rates = dict(mode_rates) if mode_rates else default_rates
        return self._allocations_for_ore_fracs(face_ore1_fracs, self._mode_rates)

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
            if fracs[i] <= 1e-12:
                continue
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
            if fracs[i] <= 1e-12:
                continue
            if truck_assignments[i] < max_trucks:
                truck_assignments[i] += 1
                unassigned_trucks -= 1

        for i in range(num_faces):
            self.face_truck_allocations[i].value = float(truck_assignments[i])

    def _face_real_extraction_rate(
        self, face_index: int, target_extraction_rate: float
    ) -> float:
        lhd_alloc = self.face_lhd_allocations[face_index].value
        truck_alloc = self.face_truck_allocations[face_index].value

        distance = self.face_haul_distance[face_index]
        accessibility = self.face_accessibility_fraction[face_index]
        mechanical_availability = self.fleet_mechanical_availability

        traffic_delay = self.traffic_delay_per_truck_hours * truck_alloc
        travel_time = (
            (2 * distance) / self.truck_velocity if self.truck_velocity > 0 else 0.0
        )

        truck_loading_time_hours = (
            self.loader_cycle_time_hours
            * (self.truck_payload_tonnes / self.loader_payload_tonnes)
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

    def _schedule_shifts(self, mode_name: str):
        self.fleet_shift_timer.rate = (1.0, -math.inf, self.fleet_shift_duration)

        shift_due = self.fleet_shift_timer.value >= self.fleet_shift_duration - 1e-6
        mode_changed = mode_name != self.current_shift_mode_name
        if self.current_shift_allocations is None or shift_due or mode_changed:
            self.fleet_shift_timer.reset()
            self.current_shift_allocations = self._get_allocations_for_mode(mode_name)
            self.current_shift_mode_name = mode_name
            self._refresh_shift_allocation_fractions()
            self.fleet_shift_count.value += 1

    def allocate(
        self,
        mine_target: float,
        mode: Union[OperatingMode, str],
        active_faces: Optional[Sequence[MineFace]] = None,
    ) -> List[float]:
        """Schedules shifts, splits ``mine_target`` across faces, enforces fleet haulage limits,
        and returns the achievable extraction rates for each face.

        When ``active_faces`` is provided, only those faces receive targets and
        equipment; the per-mode allocation fractions are renormalized over the active
        subset and locked/removed faces get zero rates.
        """
        mode_name = mode.name if hasattr(mode, "name") else str(mode)
        self._schedule_shifts(mode_name)

        override = None
        if active_faces is not None:
            active_set = set(active_faces)
            active_order = [f for f in self.faces if f in active_set]
            if 0 < len(active_order) < len(self.faces):
                # Solve the mode fractions over the active subset directly so the
                # Ore-1/Ore-2 blend stays exact for the currently available faces.
                subset_fracs = self._allocations_for_ore_fracs(
                    [1.0 - f.mean_ore_fraction for f in active_order],
                    self._mode_rates,
                )
                mode_fracs = subset_fracs.get(mode_name)
                if mode_fracs is None:
                    mode_fracs = subset_fracs.get(
                        mode_name.replace("_MINE_SURGING", "")
                    )
                if mode_fracs is not None:
                    index_of = {id(f): i for i, f in enumerate(self.faces)}
                    override = [0.0] * len(self.faces)
                    for j, face in enumerate(active_order):
                        override[index_of[id(face)]] = mode_fracs[j]

        if override is not None:
            self.current_shift_allocations = override
            self._refresh_shift_allocation_fractions()
        else:
            base_fracs = self._get_allocations_for_mode(mode_name)
            if base_fracs is not None:
                self.current_shift_allocations = list(base_fracs)
                self._refresh_shift_allocation_fractions()

        self.total_extra_trucks.value = 0.0
        fracs = self.current_shift_allocations
        achieved_rates = []

        for i, face in enumerate(self.faces):
            if fracs is not None and i < len(fracs):
                target = mine_target * fracs[i]
                real = self._face_real_extraction_rate(i, target)
                achieved = min(target, real)
                self.face_target_extraction_rates[i].value = target
                self.face_real_extraction_rates[i].value = real
                self.face_achieved_extraction_rates[i].value = achieved
                achieved_rates.append(achieved)
            else:
                self.face_target_extraction_rates[i].value = 0.0
                self.face_real_extraction_rates[i].value = 0.0
                self.face_achieved_extraction_rates[i].value = 0.0
                self.face_operational_downtime_fractions[i].value = 0.0
                achieved_rates.append(0.0)

        self.cumulative_mine_development.rate = (
            self.total_extra_trucks.value * self.development_rate_per_extra_truck
        )
        return achieved_rates

    def is_terminating_condition_met(self) -> bool:
        return False

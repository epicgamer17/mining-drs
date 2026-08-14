import drs
from drs.flow import Flow
from .data import TargetRates
from .modes import OperatingMode, RequireDecision
from .config import ConcentratorConfig
from .mine_face import BaseMineFace, ConcentratorMineFace
from .fleet import ContinuousFleetLogistics
from .plant import BaseMetallurgicalPlant, ConcentratorPlant
from .modes import MODES
from .planning import (
    MiningPriority,
    select_mining_priority,
    strategic_target_for_year,
    trajectory_progress_ratio,
)


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
        self.unassigned_trucks_to_development = drs.Variable(
            "unassigned_trucks_to_development", 0.0
        )
        self.development_priority_reserved_trucks = drs.Variable(
            "development_priority_reserved_trucks", 0.0
        )
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
        self.current_shift_reserved_trucks = 0.0
        self.fleet_shift_timer = drs.Timer("fleet_shift_timer", initial_value=0.0)
        self.fleet_shift_count = drs.Variable("fleet_shift_count", 0)

        self.mode_blend_feasible = drs.Variable("mode_blend_feasible", True)
        self.constrained_mode_active = drs.Variable(
            "constrained_mode_active", False
        )
        self.achievable_ore1_fraction = drs.Variable(
            "achievable_ore1_fraction", 1.0
        )
        self.achievable_ore2_fraction = drs.Variable(
            "achievable_ore2_fraction", 0.0
        )

        # Strategic / tactical monitoring state.
        # These variables track annual commitments and tactical responses.
        self.strategic_year_timer = drs.Timer(
            "strategic_year_timer", initial_value=0.0
        )
        self.tactical_review_timer = drs.Timer(
            "tactical_review_timer", initial_value=0.0
        )
        self.strategic_planning_started = drs.Variable(
            "strategic_planning_started", False
        )
        self.strategic_year_index = drs.Variable("strategic_year_index", 0)
        self.tactical_review_count = drs.Variable("tactical_review_count", 0)
        self.mining_priority = drs.Variable(
            "mining_priority", MiningPriority.BALANCED
        )

        self.year_start_ore1_production = drs.Variable(
            "year_start_ore1_production", 0.0
        )
        self.year_start_ore2_production = drs.Variable(
            "year_start_ore2_production", 0.0
        )
        self.year_start_development = drs.Variable(
            "year_start_development", 0.0
        )

        self.ytd_ore1_production = drs.Variable("ytd_ore1_production", 0.0)
        self.ytd_ore2_production = drs.Variable("ytd_ore2_production", 0.0)
        self.ytd_development = drs.Variable("ytd_development", 0.0)

        self.annual_target_ore1 = drs.Variable("annual_target_ore1", 0.0)
        self.annual_target_ore2 = drs.Variable("annual_target_ore2", 0.0)
        self.annual_target_development = drs.Variable(
            "annual_target_development", 0.0
        )

        self.ore1_trajectory_ratio = drs.Variable("ore1_trajectory_ratio", 1.0)
        self.ore2_trajectory_ratio = drs.Variable("ore2_trajectory_ratio", 1.0)
        self.development_trajectory_ratio = drs.Variable(
            "development_trajectory_ratio", 1.0
        )

        # Area 2 readiness is a strategic/tactical state that can gate future
        # face production when the physical unlock is enabled.
        self.strategic_start_development = drs.Variable(
            "strategic_start_development", 0.0
        )
        self.area2_required_development = drs.Variable(
            "area2_required_development", 0.0
        )
        self.area2_ready_by_day = drs.Variable("area2_ready_by_day", -1.0)
        self.area2_development_progress = drs.Variable(
            "area2_development_progress", 0.0
        )
        self.area2_cumulative_development = drs.Level(
            "area2_cumulative_development", initial_value=0.0
        )
        self.area2_development_allocation_fraction = drs.Variable(
            "area2_development_allocation_fraction", 0.0
        )
        self.area2_readiness_fraction = drs.Variable(
            "area2_readiness_fraction", 1.0
        )
        self.area2_readiness_trajectory_ratio = drs.Variable(
            "area2_readiness_trajectory_ratio", 1.0
        )
        self.area2_ready = drs.Variable("area2_ready", True)
        self.area2_deadline_missed = drs.Variable(
            "area2_deadline_missed", False
        )
        self.area2_currently_late = drs.Variable("area2_currently_late", False)
        self.area2_completed_late = drs.Variable("area2_completed_late", False)
        self.area2_ready_day = drs.Variable("area2_ready_day", -1.0)

    def _current_strategic_target(self):
        targets = getattr(self.config, "strategic_targets", ())
        return strategic_target_for_year(
            targets, int(self.strategic_year_index.value)
        )

    def _start_strategic_planning_if_ready(self):
        '''Start annual/monthly planning clocks after the existing warm-up period.'''

        if self.strategic_planning_started.value:
            return True

        total_extracted = sum(
            face.cumulative_extracted_mass.value for face in self.faces
        )
        warmup = float(self.config.ore_to_be_extracted_during_warming_period)
        if total_extracted < warmup - 1e-6:
            return False

        parent = self.parent
        if not (
            hasattr(parent, "cumulative_ore1_production")
            and hasattr(parent, "cumulative_ore2_production")
        ):
            return False

        self.year_start_ore1_production.value = (
            parent.cumulative_ore1_production.value
        )
        self.year_start_ore2_production.value = (
            parent.cumulative_ore2_production.value
        )
        self.year_start_development.value = self.cumulative_mine_development.value
        self.strategic_start_development.value = self.cumulative_mine_development.value
        self.strategic_year_timer.reset()
        self.tactical_review_timer.reset()
        self.strategic_planning_started.value = True
        return True

    def _update_strategic_progress(self):
        parent = self.parent
        if not (
            hasattr(parent, "cumulative_ore1_production")
            and hasattr(parent, "cumulative_ore2_production")
        ):
            return

        ore1_total = parent.cumulative_ore1_production.value
        ore2_total = parent.cumulative_ore2_production.value
        development_total = self.cumulative_mine_development.value

        self.ytd_ore1_production.value = max(
            0.0, ore1_total - self.year_start_ore1_production.value
        )
        self.ytd_ore2_production.value = max(
            0.0, ore2_total - self.year_start_ore2_production.value
        )
        self.ytd_development.value = max(
            0.0, development_total - self.year_start_development.value
        )

        target = self._current_strategic_target()
        self.annual_target_ore1.value = target.min_ore1_production
        self.annual_target_ore2.value = target.min_ore2_production
        self.annual_target_development.value = target.min_development

        period = max(1e-12, float(self.config.strategic_period_days))
        elapsed_fraction = self.strategic_year_timer.value / period

        self.ore1_trajectory_ratio.value = trajectory_progress_ratio(
            self.ytd_ore1_production.value,
            target.min_ore1_production,
            elapsed_fraction,
        )
        self.ore2_trajectory_ratio.value = trajectory_progress_ratio(
            self.ytd_ore2_production.value,
            target.min_ore2_production,
            elapsed_fraction,
        )
        self.development_trajectory_ratio.value = trajectory_progress_ratio(
            self.ytd_development.value,
            target.min_development,
            elapsed_fraction,
        )

        self._update_area2_readiness()

    def _strategic_elapsed_days(self) -> float:
        period = max(1e-12, float(self.config.strategic_period_days))
        return (
            float(self.strategic_year_index.value) * period
            + float(self.strategic_year_timer.value)
        )

    def _update_area2_readiness(self):
        '''Track cumulative development toward future Area 2 readiness.'''

        target = getattr(self.config, "area2_readiness_target", None)
        required = max(0.0, float(getattr(target, "required_development", 0.0)))
        deadline = getattr(target, "ready_by_day", None)
        elapsed_days = self._strategic_elapsed_days()

        progress = max(0.0, float(self.area2_cumulative_development.value))
        if required > 1e-12:
            progress = min(progress, required)

        self.area2_required_development.value = required
        self.area2_ready_by_day.value = (
            -1.0 if deadline is None else float(deadline)
        )
        self.area2_development_progress.value = progress

        if required <= 1e-12:
            self.area2_readiness_fraction.value = 1.0
            self.area2_readiness_trajectory_ratio.value = 1.0
            self.area2_ready.value = True
            self.area2_deadline_missed.value = False
            self.area2_currently_late.value = False
            self.area2_completed_late.value = False
            return

        fraction = max(0.0, min(1.0, progress / required))
        ready = progress >= required - 1e-9

        self.area2_readiness_fraction.value = fraction
        self.area2_ready.value = ready

        if ready and self.area2_ready_day.value < 0.0:
            self.area2_ready_day.value = elapsed_days

        currently_late = False
        completed_late = False
        if ready:
            trajectory_ratio = 1.0
        elif deadline is None or float(deadline) <= 0.0:
            trajectory_ratio = 1.0
        else:
            deadline_days = float(deadline)
            expected_fraction = max(0.0, min(1.0, elapsed_days / deadline_days))
            expected_progress = required * expected_fraction
            trajectory_ratio = (
                1.0
                if expected_progress <= 1e-12
                else max(0.0, progress) / expected_progress
            )
            currently_late = elapsed_days > deadline_days + 1e-9

        if deadline is not None and float(deadline) > 0.0 and ready:
            completed_late = self.area2_ready_day.value > float(deadline) + 1e-9

        self.area2_readiness_trajectory_ratio.value = trajectory_ratio
        self.area2_currently_late.value = currently_late
        self.area2_completed_late.value = completed_late
        self.area2_deadline_missed.value = currently_late or completed_late

    def _update_area2_development_allocation(self):
        # Allocate part of explicit mine-development metres to the Area 2 project.
        target = getattr(self.config, "area2_readiness_target", None)
        required = max(0.0, float(getattr(target, "required_development", 0.0)))

        if bool(getattr(self.config, "area2_counterfactual_disable", False)):
            self.area2_development_allocation_fraction.value = 0.0
            self.area2_cumulative_development.rate = 0.0
            return

        if (
            not self.strategic_planning_started.value
            or required <= 1e-12
            or self.area2_ready.value
        ):
            self.area2_development_allocation_fraction.value = 0.0
            self.area2_cumulative_development.rate = 0.0
            return

        priority = self.mining_priority.value
        if priority == MiningPriority.DEVELOPMENT:
            fraction = float(self.config.area2_development_fraction_development)
        elif priority == MiningPriority.PRODUCTION:
            fraction = float(self.config.area2_development_fraction_production)
        else:
            fraction = float(self.config.area2_development_fraction_balanced)

        fraction = max(0.0, min(1.0, fraction))
        total_development_rate = max(
            0.0, float(self.cumulative_mine_development.rate)
        )

        self.area2_development_allocation_fraction.value = fraction
        self.area2_cumulative_development.rate = total_development_rate * fraction
        self.area2_cumulative_development.upper_threshold = required

    def _advance_strategic_year(self):
        parent = self.parent
        if not (
            hasattr(parent, "cumulative_ore1_production")
            and hasattr(parent, "cumulative_ore2_production")
        ):
            self.strategic_year_timer.reset()
            return

        self.year_start_ore1_production.value = (
            parent.cumulative_ore1_production.value
        )
        self.year_start_ore2_production.value = (
            parent.cumulative_ore2_production.value
        )
        self.year_start_development.value = self.cumulative_mine_development.value
        self.strategic_year_index.value += 1
        self.strategic_year_timer.reset()
        self._update_strategic_progress()

    def _review_tactical_priority(self):
        self._update_strategic_progress()
        self.mining_priority.value = select_mining_priority(
            self.development_trajectory_ratio.value,
            self.ore1_trajectory_ratio.value,
            self.ore2_trajectory_ratio.value,
            tolerance=float(self.config.tactical_progress_tolerance),
            area2_readiness_trajectory_ratio=(
                self.area2_readiness_trajectory_ratio.value
            ),
        )
        self.tactical_review_count.value += 1
        self.tactical_review_timer.reset()

    def _update_strategic_tactical_monitoring(self):
        """Update annual strategic progress and monthly tactical priority.

        The strategic layer records annual progress; the tactical priority can
        reserve production trucks for development.
        """

        if not self._start_strategic_planning_if_ready():
            self.strategic_year_timer.rate = 0.0
            self.tactical_review_timer.rate = 0.0
            return

        strategic_days = max(1e-12, float(self.config.strategic_period_days))
        tactical_days = max(1e-12, float(self.config.tactical_review_period_days))

        self.strategic_year_timer.rate = 1.0
        self.strategic_year_timer.upper_threshold = strategic_days
        self.tactical_review_timer.rate = 1.0
        self.tactical_review_timer.upper_threshold = tactical_days

        if self.strategic_year_timer.value >= strategic_days - 1e-6:
            self._advance_strategic_year()

        self._update_strategic_progress()

        if self.tactical_review_timer.value >= tactical_days - 1e-6:
            self._review_tactical_priority()

    def _update_development_truck_reservation(self):
        reserve_fraction = 0.0
        if (
            bool(self.strategic_planning_started.value)
            and not bool(getattr(self.config, "area2_counterfactual_disable", False))
            and self.mining_priority.value == MiningPriority.DEVELOPMENT
        ):
            reserve_fraction = float(
                getattr(
                    self.config,
                    "development_priority_truck_reservation_fraction",
                    0.0,
                )
            )

        reserve_fraction = max(0.0, min(1.0, reserve_fraction))
        total_trucks = max(0.0, float(getattr(self.config, "total_truck_count", 0.0)))
        self.development_priority_reserved_trucks.value = float(
            int(round(total_trucks * reserve_fraction))
        )

    def _mean_ore1_fraction_for_face(self, face_index):
        face = self.faces[face_index]
        generator = getattr(face, "generator", None)
        if generator is not None and hasattr(generator, "mean_fraction"):
            return max(0.0, min(1.0, 1.0 - float(generator.mean_fraction)))
        return max(0.0, min(1.0, float(face.active_parcel_ore_fraction.value)))

    def _achievable_ore_fractions(self, fracs):
        if not fracs:
            return 0.0, 0.0

        ore1_fraction = 0.0
        for i, fraction in enumerate(fracs):
            ore1_fraction += max(0.0, float(fraction)) * (
                self._mean_ore1_fraction_for_face(i)
            )
        ore1_fraction = max(0.0, min(1.0, ore1_fraction))
        return ore1_fraction, 1.0 - ore1_fraction

    def _max_extraction_without_component_overbuild(self, targets, ore1_frac, ore2_frac):
        limits = []
        eps = 1e-12
        if ore1_frac > eps:
            limits.append(max(0.0, float(targets.ore1_milling_rate)) / ore1_frac)
        elif float(targets.ore1_milling_rate) > eps:
            return 0.0

        if ore2_frac > eps:
            limits.append(max(0.0, float(targets.ore2_milling_rate)) / ore2_frac)
        elif float(targets.ore2_milling_rate) > eps:
            return 0.0

        return min(limits) if limits else max(0.0, float(targets.extraction_rate))

    def _targets_constrained_for_available_blend(self, targets):
        if not bool(getattr(self.config, "mode_blend_feasibility_enabled", True)):
            self.mode_blend_feasible.value = True
            self.constrained_mode_active.value = False
            return targets

        ore1_frac, ore2_frac = self._achievable_ore_fractions(
            self.current_shift_allocations
        )
        self.achievable_ore1_fraction.value = ore1_frac
        self.achievable_ore2_fraction.value = ore2_frac

        original_extraction = max(0.0, float(targets.extraction_rate))
        if original_extraction <= 1e-12:
            self.mode_blend_feasible.value = True
            self.constrained_mode_active.value = False
            return targets

        mode_name = self.active_operating_mode.value.name
        requested_ore1_overbuild = (
            original_extraction * ore1_frac
            > float(targets.ore1_milling_rate) + 1e-6
        )
        requested_ore2_overbuild = (
            original_extraction * ore2_frac
            > float(targets.ore2_milling_rate) + 1e-6
        )
        requested_blend_feasible = not (
            requested_ore1_overbuild or requested_ore2_overbuild
        )

        if "_MINE_SURGING" not in mode_name:
            self.mode_blend_feasible.value = requested_blend_feasible
            self.constrained_mode_active.value = False
            return targets

        constrained_extraction = original_extraction
        if mode_name == "MODE_A_MINE_SURGING":
            constrained_extraction = (
                float(targets.ore1_milling_rate) / ore1_frac
                if ore1_frac > 1e-12
                else 0.0
            )
        elif mode_name == "MODE_B_MINE_SURGING":
            constrained_extraction = (
                float(targets.ore2_milling_rate) / ore2_frac
                if ore2_frac > 1e-12
                else 0.0
            )

        max_extraction = self._max_extraction_without_component_overbuild(
            targets,
            ore1_frac,
            ore2_frac,
        )
        constrained_extraction = min(constrained_extraction, max_extraction)
        constrained_extraction = max(0.0, constrained_extraction)

        self.constrained_mode_active.value = (
            abs(constrained_extraction - original_extraction) > 1e-6
        )
        self.mode_blend_feasible.value = (
            requested_blend_feasible and not self.constrained_mode_active.value
        )

        if not self.constrained_mode_active.value:
            return targets

        return TargetRates(
            extraction_rate=constrained_extraction,
            ore1_milling_rate=targets.ore1_milling_rate,
            ore2_milling_rate=targets.ore2_milling_rate,
        )

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
        # MODE_A_MINE_SURGING (ore1 stockout): maximize ore1 by using face1.
        # MODE_B_MINE_SURGING (ore2 stockout): maximize ore2 by using face2.
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
        if not self.current_shift_allocations:
            return
        for i, factor in enumerate(self.face_shift_allocation_fractions):
            factor.value = self.current_shift_allocations[i]
            
        self._distribute_discrete_fleet(self.current_shift_allocations)

    def _distribute_discrete_fleet(self, fracs):
        import math
        c = self.config
        total_lhd = int(getattr(c, "total_lhd_count", 5.0))
        reserved_trucks = int(
            max(0.0, float(self.development_priority_reserved_trucks.value))
        )
        total_truck = max(0, int(getattr(c, "total_truck_count", 10.0)) - reserved_trucks)
        max_lhds = int(getattr(c, "max_lhds_per_face", float('inf')))
        max_trucks = int(getattr(c, "max_trucks_per_face", float('inf')))
        
        num_faces = len(self.faces)
        if num_faces == 0:
            return
            
        # Distribute LHDs
        unassigned_lhds = total_lhd
        lhd_assignments = [0] * num_faces
        
        # Sort faces by their target fraction, descending
        # so the face that wants the most gets priority for rounding
        face_priorities = sorted(range(num_faces), key=lambda i: fracs[i], reverse=True)
        
        # Base assignment: integer floor, clamped to physical limits
        for i in face_priorities:
            target = math.floor(total_lhd * fracs[i])
            assigned = min(target, max_lhds, unassigned_lhds)
            lhd_assignments[i] = assigned
            unassigned_lhds -= assigned
            
        # Distribute remaining unassigned LHDs sequentially
        for i in face_priorities:
            if unassigned_lhds <= 0:
                break
            if lhd_assignments[i] < max_lhds:
                lhd_assignments[i] += 1
                unassigned_lhds -= 1
                
        # Update DRS variables
        for i in range(num_faces):
            self.face_lhd_allocations[i].value = float(lhd_assignments[i])
            
        # Distribute Trucks
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

    def _is_area2_face_locked(self, face_index):
        # Return True when the future Area 2 production face is unavailable.
        if not bool(getattr(self.config, "area2_physical_unlock_enabled", False)):
            return False

        area2_index = int(getattr(self.config, "area2_face_index", 1))
        if face_index != area2_index:
            return False

        if bool(getattr(self.config, "area2_counterfactual_disable", False)):
            return True

        target = getattr(self.config, "area2_readiness_target", None)
        required = max(0.0, float(getattr(target, "required_development", 0.0)))
        if required <= 1e-12:
            return False

        return not (
            bool(self.strategic_planning_started.value)
            and bool(self.area2_ready.value)
        )

    def _face_real_extraction_rate(self, face_index, target_extraction_rate):
        c = self.config
        
        # Read the discrete integer assignments
        lhd_alloc = self.face_lhd_allocations[face_index].value
        truck_alloc = self.face_truck_allocations[face_index].value

        # A future Area 2 face cannot produce until development readiness is met.
        # Assigned trucks may be redeployed to the existing development pool.
        if self._is_area2_face_locked(face_index):
            self.face_match_factors[face_index].value = 0.0
            self.face_truck_cycle_times[face_index].value = 0.0
            self.face_operational_downtime_fractions[face_index].value = 1.0
            if bool(
                getattr(
                    self.config,
                    "area2_redeploy_locked_face_trucks_to_development",
                    True,
                )
            ) and not bool(getattr(self.config, "area2_counterfactual_disable", False)):
                self.total_extra_trucks.value += max(0.0, float(truck_alloc))
            return 0.0
        
        distance = self._face_config_value("face_haul_distance", face_index, 0.0)
        accessibility = self._clamp(
            self._face_config_value("face_accessibility_fraction", face_index, 1.0),
            0.0,
            1.0,
        )
        mechanical_availability = getattr(c, "fleet_mechanical_availability", 1.0)

        # 1. Calculate Traffic Delay (function of number of trucks)
        # TODO: Implement non-linear gridlock delay. Currently traffic delay is linear,
        # meaning adding trucks always strictly increases throughput asymptotically.
        # In reality, cramming too many trucks into a face causes gridlock where
        # marginal throughput becomes negative.
        traffic_delay = c.traffic_delay_per_truck_hours * truck_alloc

        # 2. Calculate Truck Cycle Time (Travel + Load + Dump + Traffic Delay)
        travel_time = (2 * distance) / c.truck_velocity
        
        # Calculate how long it takes to completely fill one truck
        truck_loading_time_hours = c.loader_cycle_time_hours * (c.truck_payload_tonnes / c.loader_payload_tonnes)
        
        # TODO: Implement blasted inventory limits (drill & blast cycle).
        # Currently the face is assumed to have infinite blasted muck ready.
        # Surging 100% of production from one face should rapidly deplete the blasted 
        # inventory, forcing downtime for drilling and blasting.
        
        truck_cycle_time = (
            travel_time
            + truck_loading_time_hours
            + c.truck_dump_time_hours
            + traffic_delay
        )

        if truck_cycle_time <= 0 or lhd_alloc <= 0:
            self.face_match_factors[face_index].value = 0.0
            self.face_truck_cycle_times[face_index].value = truck_cycle_time
            return 0.0

        # 3. Calculate Match Factor
        # MF = (Number of Trucks * Time to Load One Truck) / (Number of Loaders * Truck Cycle Time)
        match_factor = (truck_alloc * truck_loading_time_hours) / (
            lhd_alloc * truck_cycle_time
        )
        self.face_match_factors[face_index].value = match_factor
        self.face_truck_cycle_times[face_index].value = truck_cycle_time

        # 4. Calculate Efficiency & Throughput based on Match Factor
        if match_factor < 1.0:
            # Under-trucked: Trucks dictate production
            max_rate = (
                (truck_alloc / truck_cycle_time) * c.truck_payload_tonnes * 24.0
            )  # tonnes per day
        else:
            # Over-trucked: Loaders dictate production (trucks wait)
            max_rate = (
                (lhd_alloc / c.loader_cycle_time_hours) * c.loader_payload_tonnes * 24.0
            )  # tonnes per day

        # Apply accessibility, mechanical availability, and allocation fractions
        final_real_extraction_rate = (
            max_rate * accessibility * mechanical_availability
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

        if not bool(getattr(self.config, "area2_counterfactual_disable", False)):
            self.total_extra_trucks.value += unused_trucks

        return max(0.0, final_real_extraction_rate)

    def _allocations_for_available_faces(self, base_fracs):
        # Renormalize production allocation over physically available faces.
        n = len(self.faces)
        if n == 0:
            return []

        if base_fracs is None:
            base_fracs = [1.0 / n] * n
        else:
            base_fracs = list(base_fracs)

        available = [not self._is_area2_face_locked(i) for i in range(n)]
        masked = [
            max(0.0, float(base_fracs[i])) if available[i] else 0.0
            for i in range(n)
        ]
        total = sum(masked)

        # If a mode requested only the locked face, fall back to the remaining
        # feasible face rather than sending production to an impossible target.
        if total <= 1e-12:
            masked = [1.0 if available[i] else 0.0 for i in range(n)]
            total = sum(masked)

        if total <= 1e-12:
            return [0.0] * n

        return [value / total for value in masked]

    def _reallocate_fleet_for_shift(self):
        mode_name = self.active_operating_mode.value.name
        base_allocations = self._get_allocations_for_mode(mode_name)
        self.current_shift_allocations = self._allocations_for_available_faces(
            base_allocations
        )
        self.current_shift_mode_name = mode_name
        self.current_shift_reserved_trucks = float(
            self.development_priority_reserved_trucks.value
        )
        self._refresh_shift_allocation_fractions()

        total_trucks = float(getattr(self.config, "total_truck_count", 0.0))
        reserved_trucks = max(
            0.0, float(self.development_priority_reserved_trucks.value)
        )
        production_trucks = max(0.0, total_trucks - reserved_trucks)
        assigned_trucks = sum(
            max(0.0, float(v.value)) for v in self.face_truck_allocations
        )
        unassigned = max(0.0, production_trucks - assigned_trucks)

        area2_project_active = (
            not bool(getattr(self.config, "area2_counterfactual_disable", False))
            and bool(getattr(
                self.config,
                "area2_redeploy_locked_face_trucks_to_development",
                True,
            ))
            and self._is_area2_face_locked(
                int(getattr(self.config, "area2_face_index", 1))
            )
        )
        self.unassigned_trucks_to_development.value = (
            unassigned if area2_project_active else 0.0
        )

        self.fleet_shift_count.value += 1

    def forward(self):
        c = self.config

        self.total_extra_trucks.value = 0.0
        self._update_strategic_tactical_monitoring()
        self._update_development_truck_reservation()

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
        reservation_changed = (
            abs(
                float(self.development_priority_reserved_trucks.value)
                - float(self.current_shift_reserved_trucks)
            )
            > 1e-6
        )
        if (
            self.current_shift_allocations is None
            or shift_due
            or mode_changed
            or reservation_changed
        ):
            self.fleet_shift_timer.reset()
            self._reallocate_fleet_for_shift()

        targets = self.active_operating_mode.value.get_target_rates(self.parent)
        targets = self._targets_constrained_for_available_blend(targets)
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

        self.total_extra_trucks.value += max(
            0.0, float(self.unassigned_trucks_to_development.value)
        )
        self.total_extra_trucks.value += max(
            0.0, float(self.development_priority_reserved_trucks.value)
        )

        development_metres_per_extra_truck_day = float(
            getattr(
                c,
                "development_metres_per_extra_truck_per_day",
                5.0,
            )
        )
        self.cumulative_mine_development.rate = (
            self.total_extra_trucks.value
            * max(0.0, development_metres_per_extra_truck_day)
        )
        self._update_area2_development_allocation()

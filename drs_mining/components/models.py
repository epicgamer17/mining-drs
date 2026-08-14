import drs
from drs.telemetry import Telemetry

from .config import BaseDualStockpileConfig, ConcentratorConfig
from .stockpiles import Stockpile
from .mine_face import BaseMineFace, ConcentratorMineFace, ContinuousMineFace
from .fleet import ContinuousFleetLogistics
from .plant import BaseMetallurgicalPlant, ConcentratorPlant
from .controllers import (
    BaseBlendingController,
    ConcentratorController,
    MultiFaceConcentratorController,
)
from .generators import (
    StochasticFaciesGenerator,
)


def _equipment_schedules(total_units, downtime_start, downtime_duration, schedules):
    if schedules is not None:
        return schedules
    return [(downtime_start, downtime_duration) for _ in range(int(total_units))]


class BaseBlendingModel(drs.Module):
    def __init__(self, config: BaseDualStockpileConfig, enable_telemetry: bool = False):
        super().__init__()
        self.config = config
        self.enable_telemetry = enable_telemetry

        self.generator = None
        self.mine: BaseMineFace = None
        self.fleet: ContinuousFleetLogistics = None
        self.plant: BaseMetallurgicalPlant = None
        self.controller: BaseBlendingController = None

        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)

    def setup_telemetry(self):
        if self.enable_telemetry:
            self.telemetry = Telemetry(self)
            
            self.telemetry.register_metric(
                "MassOfCurrentParcel",
                lambda t, m, s, _: m.mine.active_parcel_initial_mass.value,
            )
            self.telemetry.register_metric(
                "CurrentParcelRoutingFraction",
                lambda t, m, s, _: m.fleet.stockpile2_routing_fraction.value,
            )
            self.telemetry.register_metric(
                "Campaign_Shutdown",
                lambda t, m, s, _: m.controller.current_campaign_duration.value,
            )
            self.telemetry.register_metric(
                "Contingency",
                lambda t, m, s, _: m.controller.current_contingency_duration.value,
            )

    def forward(self):
        # TODO: why do we need to do this, shouldnt it be default? also do this in two places.
        self.global_time.rate = 1.0

        self.controller()

        mine_flow = self.mine()
        ore1_flow, ore2_flow = self.fleet(mine_flow)

        out1 = self.ore1_stock(
            self.controller.target_stock1_outflow_rate, inflow=ore1_flow
        )
        out2 = self.ore2_stock(
            self.controller.target_stock2_outflow_rate, inflow=ore2_flow
        )

        self.plant(out1, out2)

        self.controller.total_system_ore_mass.rate = (
            self.ore1_stock.current_mass.rate + self.ore2_stock.current_mass.rate
        )

    def is_terminating_condition_met(self) -> bool:
        return (
            self.mine.cumulative_extracted_mass.value
            >= self.config.total_ore_to_extract
        )

    def print_statistics(self):
        print("\n--- Output Statistics ---")
        total_time = (
            self.controller.cumulative_time_mode_a.value
            + self.controller.cumulative_time_mode_a_contingency.value
            + self.controller.cumulative_time_mode_a_surging.value
            + self.controller.cumulative_time_mode_b.value
            + self.controller.cumulative_time_mode_b_contingency.value
            + self.controller.cumulative_time_mode_b_surging.value
            + self.controller.cumulative_time_shutdown.value
        )

        if total_time > 0:
            print(
                f"PortionOfTimeInModeA: {self.controller.cumulative_time_mode_a.value / total_time:.4f}"
            )
            print(
                f"PortionOfTimeInModeAContingency: {self.controller.cumulative_time_mode_a_contingency.value / total_time:.4f}"
            )
            print(
                f"PortionOfTimeInModeAMineSurging: {self.controller.cumulative_time_mode_a_surging.value / total_time:.4f}"
            )
            print(
                f"PortionOfTimeInModeB: {self.controller.cumulative_time_mode_b.value / total_time:.4f}"
            )
            print(
                f"PortionOfTimeInModeBContingency: {self.controller.cumulative_time_mode_b_contingency.value / total_time:.4f}"
            )
            print(
                f"PortionOfTimeInModeBMineSurging: {self.controller.cumulative_time_mode_b_surging.value / total_time:.4f}"
            )
            print(
                f"PortionOfTimeInShutdown: {self.controller.cumulative_time_shutdown.value / total_time:.4f}"
            )
        else:
            print("Total time is 0. Cannot calculate mode portions.")

        active_time = total_time - self.controller.cumulative_time_shutdown.value
        if active_time > 0:
            if hasattr(self.plant, "cumulative_milled_mass"):
                total_ore_processed = self.plant.cumulative_milled_mass.value
            else:
                total_ore_processed = (
                    self.mine.cumulative_extracted_mass.value
                    - self.config.ore_to_be_extracted_during_warming_period
                )

            throughput = total_ore_processed / active_time
            print(f"Throughput: {throughput:.4f} tons/day")
        else:
            print("Active time is 0. Cannot calculate throughput.")


class ConcentratorModel(BaseBlendingModel):
    def __init__(self, config: ConcentratorConfig, enable_telemetry: bool = False):
        super().__init__(config, enable_telemetry)

        self.mine = ConcentratorMineFace(self.config)
        self.fleet = ContinuousFleetLogistics()

        initial_fraction = self.config.mean_ore_fraction
        initial_mass1 = (1 - initial_fraction) * self.config.target_ore_stock_level
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=initial_mass1,
            initial_attributes={
                "contained_ore_fraction_mass": initial_mass1
                * self.config.mean_ore_fraction
            },
        )
        initial_mass2 = initial_fraction * self.config.target_ore_stock_level
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=initial_mass2,
            initial_attributes={
                "contained_ore_fraction_mass": initial_mass2
                * self.config.mean_ore_fraction
            },
        )

        self.plant = ConcentratorPlant(
            self.config, self.mine, self.fleet, self.ore1_stock, self.ore2_stock
        )
        self.controller = ConcentratorController(
            self.config, self.mine, self.fleet, self.plant
        )

        self.setup_telemetry()


class ActiveFleetConcentratorModel(BaseBlendingModel):
    def __init__(self, config: ConcentratorConfig, enable_telemetry: bool = False):
        super().__init__(config, enable_telemetry)

        gen1 = StochasticFaciesGenerator(
            mean_fraction=0.15,
            std_dev=0.075,
            prob_new_facies=config.prob_new_facies,
            variation_same_facies=config.variation_same_facies,
        )
        gen2 = StochasticFaciesGenerator(
            mean_fraction=0.45,
            std_dev=0.025,
            prob_new_facies=config.prob_new_facies,
            variation_same_facies=config.variation_same_facies,
        )

        self.face1 = ContinuousMineFace(config, face_id=1, generator=gen1)
        self.face2 = ContinuousMineFace(config, face_id=2, generator=gen2)
        self.fleet = ContinuousFleetLogistics()

        # Cumulative produced ore by type for annual strategic progress tracking.
        # These are monitoring variables only and do not affect operational flows.
        self.cumulative_ore1_production = drs.Level(
            "cumulative_ore1_production", initial_value=0.0
        )
        self.cumulative_ore2_production = drs.Level(
            "cumulative_ore2_production", initial_value=0.0
        )

        # Strategic economic accounting.  These states evaluate the plan but do
        # not modify physical flows or operational decisions.
        self.cumulative_processed_ore1 = drs.Level(
            "cumulative_processed_ore1", initial_value=0.0
        )
        self.cumulative_processed_ore2 = drs.Level(
            "cumulative_processed_ore2", initial_value=0.0
        )
        self.cumulative_period_value = drs.Level(
            "cumulative_period_value", initial_value=0.0
        )
        self.cumulative_production_cost = drs.Level(
            "cumulative_production_cost", initial_value=0.0
        )
        self.cumulative_development_cost = drs.Level(
            "cumulative_development_cost", initial_value=0.0
        )
        self.cumulative_fixed_cost = drs.Level(
            "cumulative_fixed_cost", initial_value=0.0
        )
        self.cumulative_cash_flow = drs.Level(
            "cumulative_cash_flow", initial_value=0.0
        )
        self.cumulative_discounted_cash_flow = drs.Level(
            "cumulative_discounted_cash_flow", initial_value=0.0
        )
        self.discount_factor = drs.Variable("discount_factor", 1.0)
        self.current_cash_flow_rate = drs.Variable(
            "current_cash_flow_rate", 0.0
        )
        self.current_discounted_cash_flow_rate = drs.Variable(
            "current_discounted_cash_flow_rate", 0.0
        )
        self.area2_future_access_value_pv = drs.Variable(
            "area2_future_access_value_pv", 0.0
        )
        self.npv_proxy = drs.Variable("npv_proxy", 0.0)
        self.operating_npv_proxy = drs.Variable("operating_npv_proxy", 0.0)

        # Face-2-origin tracers for physical Area 2 incremental value.
        self.ore1_area2_origin_mass = drs.Level(
            "ore1_area2_origin_mass", initial_value=0.0
        )
        self.ore2_area2_origin_mass = drs.Level(
            "ore2_area2_origin_mass", initial_value=0.0
        )
        self.current_area2_mined_rate = drs.Variable(
            "current_area2_mined_rate", 0.0
        )
        self.current_area2_processed_ore1_rate = drs.Variable(
            "current_area2_processed_ore1_rate", 0.0
        )
        self.current_area2_processed_ore2_rate = drs.Variable(
            "current_area2_processed_ore2_rate", 0.0
        )
        self.cumulative_area2_processed_ore1 = drs.Level(
            "cumulative_area2_processed_ore1", initial_value=0.0
        )
        self.cumulative_area2_processed_ore2 = drs.Level(
            "cumulative_area2_processed_ore2", initial_value=0.0
        )
        self.current_area2_incremental_cash_flow_rate = drs.Variable(
            "current_area2_incremental_cash_flow_rate", 0.0
        )
        self.cumulative_area2_incremental_discounted_cash_flow = drs.Level(
            "cumulative_area2_incremental_discounted_cash_flow",
            initial_value=0.0,
        )
        self.area2_incremental_npv = drs.Variable("area2_incremental_npv", 0.0)

        initial_fraction = self.config.mean_ore_fraction
        initial_mass1 = (1 - initial_fraction) * config.target_ore_stock_level
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=initial_mass1,
            initial_attributes={
                "contained_ore_fraction_mass": initial_mass1 * initial_fraction
            },
        )
        initial_mass2 = initial_fraction * config.target_ore_stock_level
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=initial_mass2,
            initial_attributes={
                "contained_ore_fraction_mass": initial_mass2 * initial_fraction
            },
        )

        self.plant = ConcentratorPlant(
            config, None, self.fleet, self.ore1_stock, self.ore2_stock
        )

        self.controller = MultiFaceConcentratorController(
            config, faces=[self.face1, self.face2], fleet=self.fleet, plant=self.plant
        )

        self.setup_telemetry()
        if self.enable_telemetry:
            self.telemetry.register_metric(
                "face1_alloc",
                lambda t, m, s, _: m.controller.face_target_rates[0].value
                / max(1e-12, m.controller.target_mine_mass_rate.value),
            )
            self.telemetry.register_metric(
                "face2_alloc",
                lambda t, m, s, _: m.controller.face_target_rates[1].value
                / max(1e-12, m.controller.target_mine_mass_rate.value),
            )
            self.telemetry.register_metric(
                "face1_target_extraction_rate",
                lambda t, m, s, _: m.controller.face_target_extraction_rates[0].value,
            )
            self.telemetry.register_metric(
                "face1_real_extraction_rate",
                lambda t, m, s, _: m.controller.face_real_extraction_rates[0].value,
            )
            self.telemetry.register_metric(
                "face1_achieved_extraction_rate",
                lambda t, m, s, _: m.controller.face_achieved_extraction_rates[0].value,
            )
            self.telemetry.register_metric(
                "face1_operational_downtime_fraction",
                lambda t, m, s, _: m.controller.face_operational_downtime_fractions[
                    0
                ].value,
            )
            self.telemetry.register_metric(
                "face2_target_extraction_rate",
                lambda t, m, s, _: m.controller.face_target_extraction_rates[1].value,
            )
            self.telemetry.register_metric(
                "face2_real_extraction_rate",
                lambda t, m, s, _: m.controller.face_real_extraction_rates[1].value,
            )
            self.telemetry.register_metric(
                "face2_achieved_extraction_rate",
                lambda t, m, s, _: m.controller.face_achieved_extraction_rates[1].value,
            )
            self.telemetry.register_metric(
                "face2_operational_downtime_fraction",
                lambda t, m, s, _: m.controller.face_operational_downtime_fractions[
                    1
                ].value,
            )
            self.telemetry.register_metric(
                "fleet_shift_count",
                lambda t, m, s, _: m.controller.fleet_shift_count.value,
            )
            self.telemetry.register_metric(
                "fleet_shift_timer",
                lambda t, m, s, _: m.controller.fleet_shift_timer.value,
            )
            self.telemetry.register_metric(
                "face1_real_capacity",
                lambda t, m, s, h: m.controller.face_real_extraction_rates[0].value
            )
            self.telemetry.register_metric(
                "face1_target_rate",
                lambda t, m, s, h: m.controller.face_target_rates[0].value
            )
            self.telemetry.register_metric(
                "face1_match_factor",
                lambda t, m, s, h: m.controller.face_match_factors[0].value 
            )
            self.telemetry.register_metric(
                "face1_truck_cycle_time_hours",
                lambda t, m, s, h: m.controller.face_truck_cycle_times[0].value
            )
            self.telemetry.register_metric(
                "face2_real_capacity",
                lambda t, m, s, h: m.controller.face_real_extraction_rates[1].value
            )
            self.telemetry.register_metric(
                "face2_target_rate",
                lambda t, m, s, h: m.controller.face_target_rates[1].value
            )
            self.telemetry.register_metric(
                "face2_match_factor",
                lambda t, m, s, h: m.controller.face_match_factors[1].value 
            )
            self.telemetry.register_metric(
                "face2_truck_cycle_time_hours",
                lambda t, m, s, h: m.controller.face_truck_cycle_times[1].value
            )
            self.telemetry.register_metric(
                "total_unused_trucks",
                lambda t, m, s, h: m.controller.total_extra_trucks.value
            )
            self.telemetry.register_metric(
                "unassigned_trucks_to_development",
                lambda t, m, s, _: (
                    m.controller.unassigned_trucks_to_development.value
                ),
            )
            self.telemetry.register_metric(
                "development_priority_reserved_trucks",
                lambda t, m, s, _: (
                    m.controller.development_priority_reserved_trucks.value
                ),
            )
            self.telemetry.register_metric(
                "mine_development_rate_m_per_day",
                lambda t, m, s, _: m.controller.cumulative_mine_development.rate,
            )
            self.telemetry.register_metric(
                "mode_blend_feasible",
                lambda t, m, s, _: m.controller.mode_blend_feasible.value,
            )
            self.telemetry.register_metric(
                "constrained_mode_active",
                lambda t, m, s, _: m.controller.constrained_mode_active.value,
            )
            self.telemetry.register_metric(
                "achievable_ore1_fraction",
                lambda t, m, s, _: m.controller.achievable_ore1_fraction.value,
            )
            self.telemetry.register_metric(
                "achievable_ore2_fraction",
                lambda t, m, s, _: m.controller.achievable_ore2_fraction.value,
            )
            self.telemetry.register_metric(
                "strategic_planning_started",
                lambda t, m, s, _: m.controller.strategic_planning_started.value,
            )
            self.telemetry.register_metric(
                "strategic_year_index",
                lambda t, m, s, _: m.controller.strategic_year_index.value,
            )
            self.telemetry.register_metric(
                "tactical_review_count",
                lambda t, m, s, _: m.controller.tactical_review_count.value,
            )
            self.telemetry.register_metric(
                "mining_priority",
                lambda t, m, s, _: m.controller.mining_priority.value.name,
            )
            self.telemetry.register_metric(
                "ytd_ore1_production",
                lambda t, m, s, _: m.controller.ytd_ore1_production.value,
            )
            self.telemetry.register_metric(
                "ytd_ore2_production",
                lambda t, m, s, _: m.controller.ytd_ore2_production.value,
            )
            self.telemetry.register_metric(
                "ytd_development",
                lambda t, m, s, _: m.controller.ytd_development.value,
            )
            self.telemetry.register_metric(
                "ore1_trajectory_ratio",
                lambda t, m, s, _: m.controller.ore1_trajectory_ratio.value,
            )
            self.telemetry.register_metric(
                "ore2_trajectory_ratio",
                lambda t, m, s, _: m.controller.ore2_trajectory_ratio.value,
            )
            self.telemetry.register_metric(
                "development_trajectory_ratio",
                lambda t, m, s, _: m.controller.development_trajectory_ratio.value,
            )
            self.telemetry.register_metric(
                "area2_required_development",
                lambda t, m, s, _: m.controller.area2_required_development.value,
            )
            self.telemetry.register_metric(
                "area2_ready_by_day",
                lambda t, m, s, _: m.controller.area2_ready_by_day.value,
            )
            self.telemetry.register_metric(
                "area2_development_progress",
                lambda t, m, s, _: m.controller.area2_development_progress.value,
            )
            self.telemetry.register_metric(
                "area2_readiness_fraction",
                lambda t, m, s, _: m.controller.area2_readiness_fraction.value,
            )
            self.telemetry.register_metric(
                "area2_readiness_trajectory_ratio",
                lambda t, m, s, _: m.controller.area2_readiness_trajectory_ratio.value,
            )
            self.telemetry.register_metric(
                "area2_development_allocation_fraction",
                lambda t, m, s, _: m.controller.area2_development_allocation_fraction.value,
            )
            self.telemetry.register_metric(
                "area2_cumulative_development",
                lambda t, m, s, _: m.controller.area2_cumulative_development.value,
            )
            self.telemetry.register_metric(
                "area2_ready",
                lambda t, m, s, _: m.controller.area2_ready.value,
            )
            self.telemetry.register_metric(
                "area2_physical_unlocked",
                lambda t, m, s, _: (
                    not m.controller._is_area2_face_locked(
                        int(getattr(m.config, "area2_face_index", 1))
                    )
                ),
            )
            self.telemetry.register_metric(
                "area2_deadline_missed",
                lambda t, m, s, _: m.controller.area2_deadline_missed.value,
            )
            self.telemetry.register_metric(
                "area2_currently_late",
                lambda t, m, s, _: m.controller.area2_currently_late.value,
            )
            self.telemetry.register_metric(
                "area2_completed_late",
                lambda t, m, s, _: m.controller.area2_completed_late.value,
            )
            self.telemetry.register_metric(
                "area2_ready_day",
                lambda t, m, s, _: m.controller.area2_ready_day.value,
            )
            self.telemetry.register_metric(
                "cumulative_processed_ore1",
                lambda t, m, s, _: m.cumulative_processed_ore1.value,
            )
            self.telemetry.register_metric(
                "cumulative_processed_ore2",
                lambda t, m, s, _: m.cumulative_processed_ore2.value,
            )
            self.telemetry.register_metric(
                "cumulative_period_value",
                lambda t, m, s, _: m.cumulative_period_value.value,
            )
            self.telemetry.register_metric(
                "cumulative_production_cost",
                lambda t, m, s, _: m.cumulative_production_cost.value,
            )
            self.telemetry.register_metric(
                "cumulative_development_cost",
                lambda t, m, s, _: m.cumulative_development_cost.value,
            )
            self.telemetry.register_metric(
                "cumulative_fixed_cost",
                lambda t, m, s, _: m.cumulative_fixed_cost.value,
            )
            self.telemetry.register_metric(
                "cumulative_cash_flow",
                lambda t, m, s, _: m.cumulative_cash_flow.value,
            )
            self.telemetry.register_metric(
                "discount_factor",
                lambda t, m, s, _: m.discount_factor.value,
            )
            self.telemetry.register_metric(
                "current_cash_flow_rate",
                lambda t, m, s, _: m.current_cash_flow_rate.value,
            )
            self.telemetry.register_metric(
                "current_discounted_cash_flow_rate",
                lambda t, m, s, _: m.current_discounted_cash_flow_rate.value,
            )
            self.telemetry.register_metric(
                "cumulative_discounted_cash_flow",
                lambda t, m, s, _: m.cumulative_discounted_cash_flow.value,
            )
            self.telemetry.register_metric(
                "area2_future_access_value_pv",
                lambda t, m, s, _: m.area2_future_access_value_pv.value,
            )
            self.telemetry.register_metric(
                "operating_npv_proxy",
                lambda t, m, s, _: m.operating_npv_proxy.value,
            )
            self.telemetry.register_metric(
                "current_area2_mined_rate",
                lambda t, m, s, _: m.current_area2_mined_rate.value,
            )
            self.telemetry.register_metric(
                "current_area2_processed_ore1_rate",
                lambda t, m, s, _: m.current_area2_processed_ore1_rate.value,
            )
            self.telemetry.register_metric(
                "current_area2_processed_ore2_rate",
                lambda t, m, s, _: m.current_area2_processed_ore2_rate.value,
            )
            self.telemetry.register_metric(
                "cumulative_area2_processed_ore1",
                lambda t, m, s, _: m.cumulative_area2_processed_ore1.value,
            )
            self.telemetry.register_metric(
                "cumulative_area2_processed_ore2",
                lambda t, m, s, _: m.cumulative_area2_processed_ore2.value,
            )
            self.telemetry.register_metric(
                "current_area2_incremental_cash_flow_rate",
                lambda t, m, s, _: m.current_area2_incremental_cash_flow_rate.value,
            )
            self.telemetry.register_metric(
                "cumulative_area2_incremental_discounted_cash_flow",
                lambda t, m, s, _: (
                    m.cumulative_area2_incremental_discounted_cash_flow.value
                ),
            )
            self.telemetry.register_metric(
                "area2_incremental_npv",
                lambda t, m, s, _: m.area2_incremental_npv.value,
            )
            self.telemetry.register_metric(
                "area2_attributed_npv",
                lambda t, m, s, _: m.area2_incremental_npv.value,
            )
            self.telemetry.register_metric(
                "npv_proxy",
                lambda t, m, s, _: m.npv_proxy.value,
            )
            self.telemetry.register_metric(
                "ore2_ratio",
                lambda t, m, s, _: m.ore2_stock.current_mass.value
                / max(
                    1e-6,
                    m.ore1_stock.current_mass.value + m.ore2_stock.current_mass.value,
                ),
            )
            self.telemetry.register_metric(
                "face1_extracted_mass",
                lambda t, m, s, _: m.face1.cumulative_extracted_mass.value,
            )
            self.telemetry.register_metric(
                "face2_extracted_mass",
                lambda t, m, s, _: m.face2.cumulative_extracted_mass.value,
            )
            self.telemetry.register_metric(
                "face1_parcel_mass",
                lambda t, m, s, _: m.face1.active_parcel_initial_mass.value,
            )
            self.telemetry.register_metric(
                "face1_parcel_ratio",
                lambda t, m, s, _: m.face1.active_parcel_ore_fraction.value,
            )
            self.telemetry.register_metric(
                "face2_parcel_mass",
                lambda t, m, s, _: m.face2.active_parcel_initial_mass.value,
            )
            self.telemetry.register_metric(
                "face2_parcel_ratio",
                lambda t, m, s, _: m.face2.active_parcel_ore_fraction.value,
            )
            self.telemetry.register_metric(
                "mixed_achieved_extraction_rate",
                lambda t, m, s, _: sum(
                    rate.value for rate in m.controller.face_achieved_extraction_rates
                ),
            )
            self.telemetry.register_metric(
                "mixed_target_extraction_rate",
                lambda t, m, s, _: sum(
                    rate.value for rate in m.controller.face_target_extraction_rates
                ),
            )
            self.telemetry.register_metric(
                "mixed_real_extraction_rate",
                lambda t, m, s, _: sum(
                    rate.value for rate in m.controller.face_real_extraction_rates
                ),
            )
            self.telemetry.register_metric(
                "mixed_ore1_fraction",
                lambda t, m, s, _: 1.0 - m.fleet.stockpile2_routing_fraction.value,
            )

    @staticmethod
    def _bounded_fraction(numerator, denominator):
        if denominator <= 1e-12:
            return 0.0
        return max(0.0, min(1.0, numerator / denominator))

    def _update_area2_origin_accounting(
        self,
        face2_flow,
        ore1_inflow,
        ore2_inflow,
        ore1_outflow,
        ore2_outflow,
    ):
        """Trace Face-2-origin material through the existing well-mixed stockpiles."""
        if not bool(getattr(self.config, "area2_physical_unlock_enabled", False)):
            self.ore1_area2_origin_mass.rate = 0.0
            self.ore2_area2_origin_mass.rate = 0.0
            self.current_area2_mined_rate.value = 0.0
            self.current_area2_processed_ore1_rate.value = 0.0
            self.current_area2_processed_ore2_rate.value = 0.0
            self.cumulative_area2_processed_ore1.rate = 0.0
            self.cumulative_area2_processed_ore2.rate = 0.0
            return

        material = face2_flow.value
        area2_mined_rate = max(0.0, float(material.extraction_rate))
        ore1_fraction = max(0.0, min(1.0, float(material.attr_value)))
        area2_ore1_in = area2_mined_rate * ore1_fraction
        area2_ore2_in = area2_mined_rate * (1.0 - ore1_fraction)
        self.current_area2_mined_rate.value = area2_mined_rate

        total_ore1_in = max(0.0, float(ore1_inflow.value.extraction_rate))
        total_ore2_in = max(0.0, float(ore2_inflow.value.extraction_rate))
        total_ore1_out = max(0.0, float(ore1_outflow.value))
        total_ore2_out = max(0.0, float(ore2_outflow.value))

        ore1_stock_mass = max(0.0, float(self.ore1_stock.current_mass.value))
        ore2_stock_mass = max(0.0, float(self.ore2_stock.current_mass.value))

        if ore1_stock_mass > 1e-12:
            ore1_area2_fraction = self._bounded_fraction(
                float(self.ore1_area2_origin_mass.value), ore1_stock_mass
            )
        else:
            ore1_area2_fraction = self._bounded_fraction(area2_ore1_in, total_ore1_in)

        if ore2_stock_mass > 1e-12:
            ore2_area2_fraction = self._bounded_fraction(
                float(self.ore2_area2_origin_mass.value), ore2_stock_mass
            )
        else:
            ore2_area2_fraction = self._bounded_fraction(area2_ore2_in, total_ore2_in)

        area2_ore1_processed = total_ore1_out * ore1_area2_fraction
        area2_ore2_processed = total_ore2_out * ore2_area2_fraction

        self.ore1_area2_origin_mass.rate = area2_ore1_in - area2_ore1_processed
        self.ore2_area2_origin_mass.rate = area2_ore2_in - area2_ore2_processed
        self.ore1_area2_origin_mass.lower_threshold = 0.0
        self.ore2_area2_origin_mass.lower_threshold = 0.0

        self.current_area2_processed_ore1_rate.value = area2_ore1_processed
        self.current_area2_processed_ore2_rate.value = area2_ore2_processed
        self.cumulative_area2_processed_ore1.rate = area2_ore1_processed
        self.cumulative_area2_processed_ore2.rate = area2_ore2_processed

    def _zero_strategic_economic_rates(self):
        self.cumulative_processed_ore1.rate = 0.0
        self.cumulative_processed_ore2.rate = 0.0
        self.cumulative_period_value.rate = 0.0
        self.cumulative_production_cost.rate = 0.0
        self.cumulative_development_cost.rate = 0.0
        self.cumulative_fixed_cost.rate = 0.0
        self.cumulative_cash_flow.rate = 0.0
        self.cumulative_discounted_cash_flow.rate = 0.0
        self.current_cash_flow_rate.value = 0.0
        self.current_discounted_cash_flow_rate.value = 0.0
        self.current_area2_incremental_cash_flow_rate.value = 0.0
        self.cumulative_area2_incremental_discounted_cash_flow.rate = 0.0

    def _update_strategic_economics(self, ore1_outflow, ore2_outflow):
        """Evaluate whole-mine operating NPV and physical Area 2 incremental NPV."""
        if not self.controller.strategic_planning_started.value:
            self._zero_strategic_economic_rates()
            self.discount_factor.value = 1.0
            self.area2_future_access_value_pv.value = 0.0
            operating_npv = float(self.cumulative_discounted_cash_flow.value)
            self.operating_npv_proxy.value = operating_npv
            self.npv_proxy.value = operating_npv
            self.area2_incremental_npv.value = float(
                self.cumulative_area2_incremental_discounted_cash_flow.value
            )
            return

        ore1_processed_rate = max(0.0, float(ore1_outflow.value))
        ore2_processed_rate = max(0.0, float(ore2_outflow.value))
        self.cumulative_processed_ore1.rate = ore1_processed_rate
        self.cumulative_processed_ore2.rate = ore2_processed_rate

        period_value_rate = (
            ore1_processed_rate * float(self.config.ore1_net_value_per_processed_tonne)
            + ore2_processed_rate * float(self.config.ore2_net_value_per_processed_tonne)
        )
        self.cumulative_period_value.rate = period_value_rate

        mine_production_rate = sum(
            max(0.0, float(rate.value))
            for rate in self.controller.face_achieved_extraction_rates
        )
        production_cost_rate = (
            mine_production_rate * float(self.config.production_cost_per_tonne)
        )
        self.cumulative_production_cost.rate = production_cost_rate

        development_rate = max(
            0.0, float(self.controller.cumulative_mine_development.rate)
        )
        development_cost_rate = (
            development_rate * float(self.config.development_cost_per_unit)
        )
        self.cumulative_development_cost.rate = development_cost_rate

        fixed_cost_rate = max(0.0, float(self.config.fixed_cost_per_day))
        self.cumulative_fixed_cost.rate = fixed_cost_rate

        cash_flow_rate = (
            period_value_rate
            - production_cost_rate
            - development_cost_rate
            - fixed_cost_rate
        )
        self.current_cash_flow_rate.value = cash_flow_rate
        self.cumulative_cash_flow.rate = cash_flow_rate

        strategic_days = (
            float(self.controller.strategic_year_index.value)
            * float(self.config.strategic_period_days)
            + float(self.controller.strategic_year_timer.value)
        )
        annual_discount_rate = float(self.config.annual_discount_rate)
        if annual_discount_rate <= -1.0:
            raise ValueError("annual_discount_rate must be greater than -1.0")
        discount_factor = 1.0 / (
            (1.0 + annual_discount_rate) ** (strategic_days / 365.0)
        )
        self.discount_factor.value = discount_factor

        discounted_cash_flow_rate = cash_flow_rate * discount_factor
        self.current_discounted_cash_flow_rate.value = discounted_cash_flow_rate
        self.cumulative_discounted_cash_flow.rate = discounted_cash_flow_rate

        area2_value_rate = (
            float(self.current_area2_processed_ore1_rate.value)
            * float(self.config.ore1_net_value_per_processed_tonne)
            + float(self.current_area2_processed_ore2_rate.value)
            * float(self.config.ore2_net_value_per_processed_tonne)
        )
        area2_mining_cost_rate = (
            max(0.0, float(self.current_area2_mined_rate.value))
            * float(self.config.production_cost_per_tonne)
        )
        area2_development_cost_rate = (
            max(0.0, float(self.controller.area2_cumulative_development.rate))
            * float(self.config.development_cost_per_unit)
        )
        # This is an attributed within-run value. The counterfactual Area 2
        # NPV is calculated in simulation.py as:
        # NPV(WITH Area2) - NPV(WITHOUT Area2), using identical random seeds.
        area2_incremental_cash_flow_rate = (
            area2_value_rate
            - area2_mining_cost_rate
            - area2_development_cost_rate
        )
        self.current_area2_incremental_cash_flow_rate.value = (
            area2_incremental_cash_flow_rate
        )
        self.cumulative_area2_incremental_discounted_cash_flow.rate = (
            area2_incremental_cash_flow_rate * discount_factor
        )
        self.area2_incremental_npv.value = float(
            self.cumulative_area2_incremental_discounted_cash_flow.value
        )

        # No synthetic Area 2 access value when physical Area 2 production is on.
        self.area2_future_access_value_pv.value = 0.0

        operating_npv = float(self.cumulative_discounted_cash_flow.value)
        self.operating_npv_proxy.value = operating_npv
        self.npv_proxy.value = operating_npv

    def setup_telemetry(self):
        # NOTE: Intentionally NOT calling super().setup_telemetry() because
        # the base class registers metrics referencing m.mine which is None
        # in the multi-face case. Face/parcel metrics are registered below.
        if self.enable_telemetry:
            self.telemetry = Telemetry(self)
            self.register_post_step_hook(self.telemetry.snapshot)
            self.telemetry.register_metric(
                "Campaign_Shutdown",
                lambda t, m, s, _: m.controller.current_campaign_duration.value,
            )
            self.telemetry.register_metric(
                "Contingency",
                lambda t, m, s, _: m.controller.current_contingency_duration.value,
            )

    def forward(self):
        self.global_time.rate = 1.0
        self.controller()
        face1_flow = self.face1(self.controller.face_achieved_extraction_rates[0])
        face2_flow = self.face2(self.controller.face_achieved_extraction_rates[1])
        ore1_flow, ore2_flow = self.fleet(face1_flow, face2_flow)

        self.cumulative_ore1_production.rate = ore1_flow.value.extraction_rate
        self.cumulative_ore2_production.rate = ore2_flow.value.extraction_rate

        out1 = self.ore1_stock(
            self.controller.target_stock1_outflow_rate, inflow=ore1_flow
        )
        out2 = self.ore2_stock(
            self.controller.target_stock2_outflow_rate, inflow=ore2_flow
        )
        self.plant(out1, out2)
        self._update_area2_origin_accounting(
            face2_flow, ore1_flow, ore2_flow, out1, out2
        )
        self._update_strategic_economics(out1, out2)

        self.controller.total_system_ore_mass.rate = (
            self.ore1_stock.current_mass.rate + self.ore2_stock.current_mass.rate
        )

    def is_terminating_condition_met(self):
        total_extracted = (
            self.face1.cumulative_extracted_mass.value
            + self.face2.cumulative_extracted_mass.value
        )
        return total_extracted >= self.config.total_ore_to_extract

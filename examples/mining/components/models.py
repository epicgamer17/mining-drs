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
            self.register_post_step_hook(self.telemetry.snapshot)

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
        ore1_flow, ore2_flow = self.fleet(
            self.face1(self.controller.face_achieved_extraction_rates[0]),
            self.face2(self.controller.face_achieved_extraction_rates[1]),
        )
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

    def is_terminating_condition_met(self):
        total_extracted = (
            self.face1.cumulative_extracted_mass.value
            + self.face2.cumulative_extracted_mass.value
        )
        return total_extracted >= self.config.total_ore_to_extract

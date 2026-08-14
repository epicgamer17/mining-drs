import math
import drs
from drs.telemetry import Telemetry

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
    def __init__(
        self,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        target_ore_stock_level: float = 60000.0,
        enable_telemetry: bool = False,
    ):
        super().__init__()
        self.total_ore_to_extract = total_ore_to_extract
        self.ore_to_be_extracted_during_warming_period = (
            ore_to_be_extracted_during_warming_period
        )
        self.target_ore_stock_level = target_ore_stock_level
        self.enable_telemetry = enable_telemetry

        self.generator = None
        self.mine: BaseMineFace = None
        self.fleet: ContinuousFleetLogistics = None
        self.plant: BaseMetallurgicalPlant = None
        self.controller: BaseBlendingController = None

        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)

    @property
    def ore1_mass(self) -> float:
        return self.ore1_stock.current_mass.value if hasattr(self, "ore1_stock") and self.ore1_stock else 0.0

    @property
    def ore2_mass(self) -> float:
        return self.ore2_stock.current_mass.value if hasattr(self, "ore2_stock") and self.ore2_stock else 0.0

    @property
    def total_stockpile_mass(self) -> float:
        return self.ore1_mass + self.ore2_mass

    @property
    def stockpile2_routing_fraction(self) -> float:
        return self.fleet.stockpile2_routing_fraction.value if hasattr(self, "fleet") and self.fleet else 0.0

    @property
    def target_mine_mass_rate(self) -> float:
        return self.controller.target_mine_mass_rate.value if hasattr(self, "controller") and self.controller else 0.0

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
            >= self.total_ore_to_extract
        )

    def print_statistics(self):
        print("\n--- Output Statistics ---")
        total_time = self.controller.total_duration

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

        active_time = self.controller.active_duration(total_time)
        if active_time > 0:
            if hasattr(self.plant, "cumulative_milled_mass"):
                total_ore_processed = self.plant.cumulative_milled_mass.value
            else:
                total_ore_processed = self.mine.net_extracted_mass

            throughput = total_ore_processed / active_time
            print(f"Throughput: {throughput:.4f} tons/day")
        else:
            print("Active time is 0. Cannot calculate throughput.")


class ConcentratorModel(BaseBlendingModel):
    def __init__(
        self,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        target_ore_stock_level: float = 60000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        min_ore_mass: float = 30000.0,
        max_ore_mass: float = 50000.0,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        replication_length: float = math.inf,
        enable_telemetry: bool = False,
    ):
        super().__init__(
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            target_ore_stock_level=target_ore_stock_level,
            enable_telemetry=enable_telemetry,
        )
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.critical_ore2_level = critical_ore2_level
        self.duration_of_production_campaigns = duration_of_production_campaigns
        self.duration_of_shutdowns = duration_of_shutdowns
        self.duration_of_contingency_segments = duration_of_contingency_segments
        self.min_ore_mass = min_ore_mass
        self.max_ore_mass = max_ore_mass
        self.prob_new_facies = prob_new_facies
        self.variation_same_facies = variation_same_facies
        self.replication_length = replication_length

        self.mine = ConcentratorMineFace(
            mean_ore_fraction=mean_ore_fraction,
            std_dev_ore_fraction=std_dev_ore_fraction,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
            min_ore_mass=min_ore_mass,
            max_ore_mass=max_ore_mass,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        )
        self.fleet = ContinuousFleetLogistics()

        initial_fraction = mean_ore_fraction
        initial_mass1 = (1 - initial_fraction) * target_ore_stock_level
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=initial_mass1,
            initial_attributes={
                "contained_ore_fraction_mass": initial_mass1 * mean_ore_fraction
            },
        )
        initial_mass2 = initial_fraction * target_ore_stock_level
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=initial_mass2,
            initial_attributes={
                "contained_ore_fraction_mass": initial_mass2 * mean_ore_fraction
            },
        )

        self.plant = ConcentratorPlant(
            self.mine, self.fleet, self.ore1_stock, self.ore2_stock
        )
        self.controller = ConcentratorController(
            mine=self.mine,
            fleet=self.fleet,
            plant=self.plant,
            target_ore_stock_level=target_ore_stock_level,
            critical_ore2_level=critical_ore2_level,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        )

        self.setup_telemetry()


class ActiveFleetConcentratorModel(BaseBlendingModel):
    def __init__(
        self,
        mean_ore_fraction: float = 0.30,
        std_dev_ore_fraction: float = 0.05,
        target_ore_stock_level: float = 60000.0,
        total_ore_to_extract: float = 6600000.0,
        ore_to_be_extracted_during_warming_period: float = 600000.0,
        critical_ore2_level: float = 20400.0,
        duration_of_production_campaigns: float = 34.0,
        duration_of_shutdowns: float = 1.0,
        duration_of_contingency_segments: float = 1.0,
        prob_new_facies: float = 0.3,
        variation_same_facies: float = 0.01,
        replication_length: float = math.inf,
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
        enable_telemetry: bool = False,
    ):
        super().__init__(
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            target_ore_stock_level=target_ore_stock_level,
            enable_telemetry=enable_telemetry,
        )
        self.mean_ore_fraction = mean_ore_fraction
        self.std_dev_ore_fraction = std_dev_ore_fraction
        self.replication_length = replication_length

        gen1 = StochasticFaciesGenerator(
            mean_fraction=0.15,
            std_dev=0.075,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
        )
        gen2 = StochasticFaciesGenerator(
            mean_fraction=0.45,
            std_dev=0.025,
            prob_new_facies=prob_new_facies,
            variation_same_facies=variation_same_facies,
        )

        self.face1 = ContinuousMineFace(
            face_id=1,
            generator=gen1,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        )
        self.face2 = ContinuousMineFace(
            face_id=2,
            generator=gen2,
            total_ore_to_extract=total_ore_to_extract,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
        )
        self.fleet = ContinuousFleetLogistics()

        initial_fraction = mean_ore_fraction
        initial_mass1 = (1 - initial_fraction) * target_ore_stock_level
        self.ore1_stock = Stockpile(
            name="Ore1Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=initial_mass1,
            initial_attributes={
                "contained_ore_fraction_mass": initial_mass1 * initial_fraction
            },
        )
        initial_mass2 = initial_fraction * target_ore_stock_level
        self.ore2_stock = Stockpile(
            name="Ore2Stock",
            expected_attributes=["contained_ore_fraction_mass"],
            initial_mass=initial_mass2,
            initial_attributes={
                "contained_ore_fraction_mass": initial_mass2 * initial_fraction
            },
        )

        self.plant = ConcentratorPlant(
            None, self.fleet, self.ore1_stock, self.ore2_stock
        )

        self.controller = MultiFaceConcentratorController(
            faces=[self.face1, self.face2],
            fleet=self.fleet,
            plant=self.plant,
            target_ore_stock_level=target_ore_stock_level,
            critical_ore2_level=critical_ore2_level,
            duration_of_production_campaigns=duration_of_production_campaigns,
            duration_of_shutdowns=duration_of_shutdowns,
            duration_of_contingency_segments=duration_of_contingency_segments,
            ore_to_be_extracted_during_warming_period=ore_to_be_extracted_during_warming_period,
            mode_a_ore1_milling_rate=mode_a_ore1_milling_rate,
            mode_a_ore2_milling_rate=mode_a_ore2_milling_rate,
            mode_a_contingency_ore1_milling_rate=mode_a_contingency_ore1_milling_rate,
            mode_b_ore1_milling_rate=mode_b_ore1_milling_rate,
            mode_b_ore2_milling_rate=mode_b_ore2_milling_rate,
            mode_b_contingency_ore2_milling_rate=mode_b_contingency_ore2_milling_rate,
            fleet_shift_duration=fleet_shift_duration,
            total_lhd_count=total_lhd_count,
            total_truck_count=total_truck_count,
            max_lhds_per_face=max_lhds_per_face,
            max_trucks_per_face=max_trucks_per_face,
            face_haul_distance=face_haul_distance,
            face_accessibility_fraction=face_accessibility_fraction,
            truck_velocity=truck_velocity,
            loader_cycle_time_hours=loader_cycle_time_hours,
            truck_dump_time_hours=truck_dump_time_hours,
            traffic_delay_per_truck_hours=traffic_delay_per_truck_hours,
            fleet_mechanical_availability=fleet_mechanical_availability,
            loader_payload_tonnes=loader_payload_tonnes,
            truck_payload_tonnes=truck_payload_tonnes,
            development_rate_per_extra_truck=development_rate_per_extra_truck,
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
                lambda t, m, s, h: m.controller.face_real_extraction_rates[0].value,
            )
            self.telemetry.register_metric(
                "face1_target_rate",
                lambda t, m, s, h: m.controller.face_target_rates[0].value,
            )
            self.telemetry.register_metric(
                "face1_match_factor",
                lambda t, m, s, h: m.controller.face_match_factors[0].value,
            )
            self.telemetry.register_metric(
                "face1_truck_cycle_time_hours",
                lambda t, m, s, h: m.controller.face_truck_cycle_times[0].value,
            )
            self.telemetry.register_metric(
                "face2_real_capacity",
                lambda t, m, s, h: m.controller.face_real_extraction_rates[1].value,
            )
            self.telemetry.register_metric(
                "face2_target_rate",
                lambda t, m, s, h: m.controller.face_target_rates[1].value,
            )
            self.telemetry.register_metric(
                "face2_match_factor",
                lambda t, m, s, h: m.controller.face_match_factors[1].value,
            )
            self.telemetry.register_metric(
                "face2_truck_cycle_time_hours",
                lambda t, m, s, h: m.controller.face_truck_cycle_times[1].value,
            )
            self.telemetry.register_metric(
                "total_unused_trucks",
                lambda t, m, s, h: m.controller.total_extra_trucks.value,
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
        return total_extracted >= self.total_ore_to_extract

from drs.module import drs
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


def _equipment_availability(
    name,
    total_units,
    downtime_start,
    downtime_duration,
    schedules,
    time_between_failures_hours,
    repair_time_hours,
):
    if time_between_failures_hours is not None and repair_time_hours is not None:
        return StochasticEquipmentFleetAvailability(
            name=name,
            total_units=int(total_units),
            time_between_failures_hours=time_between_failures_hours,
            repair_time_hours=repair_time_hours,
        )

    return EquipmentFleetAvailability(
        name=name,
        unit_downtime_schedules=_equipment_schedules(
            total_units,
            downtime_start,
            downtime_duration,
            schedules,
        ),
    )


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
                lambda t, m, s, h: m.mine.active_parcel_initial_mass.value,
            )
            self.telemetry.register_metric(
                "CurrentParcelRoutingFraction",
                lambda t, m, s, h: m.fleet.stockpile2_routing_fraction.value,
            )
            self.telemetry.register_metric(
                "Campaign_Shutdown",
                lambda t, m, s, h: m.controller.current_campaign_duration.value,
            )
            self.telemetry.register_metric(
                "Contingency",
                lambda t, m, s, h: m.controller.current_contingency_duration.value,
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
                "face1_alloc", lambda t, m, s, h: m.controller.face_target_rates[0].value / max(1e-12, m.controller.target_mine_mass_rate.value)
            )
            self.telemetry.register_metric(
                "face2_alloc", lambda t, m, s, h: m.controller.face_target_rates[1].value / max(1e-12, m.controller.target_mine_mass_rate.value)
            )
            self.telemetry.register_metric(
                "face1_required_rate",
                lambda t, m, s, h: m.controller.face_required_rates[0].value,
            )
            self.telemetry.register_metric(
                "face1_max_extraction_rate",
                lambda t, m, s, h: m.controller.face_max_extraction_rates[0].value,
            )
            self.telemetry.register_metric(
                "face1_actual_rate",
                lambda t, m, s, h: m.controller.face_actual_rates[0].value,
            )
            self.telemetry.register_metric(
                "face1_effective_delay_factor",
                lambda t, m, s, h: m.controller.face_effective_delay_factors[
                    0
                ].value,
            )
            self.telemetry.register_metric(
                "face2_required_rate",
                lambda t, m, s, h: m.controller.face_required_rates[1].value,
            )
            self.telemetry.register_metric(
                "face2_max_extraction_rate",
                lambda t, m, s, h: m.controller.face_max_extraction_rates[1].value,
            )
            self.telemetry.register_metric(
                "face2_actual_rate",
                lambda t, m, s, h: m.controller.face_actual_rates[1].value,
            )
            self.telemetry.register_metric(
                "face2_effective_delay_factor",
                lambda t, m, s, h: m.controller.face_effective_delay_factors[
                    1
                ].value,
            )
            self.telemetry.register_metric(
                "fleet_shift_count",
                lambda t, m, s, h: m.controller.fleet_shift_count.value,
            )
            self.telemetry.register_metric(
                "fleet_shift_timer",
                lambda t, m, s, h: m.controller.fleet_shift_timer.value,
            )
            self.telemetry.register_metric(
                "ore2_ratio",
                lambda t, m, s, h: m.ore2_stock.current_mass.value
                / max(
                    1e-6,
                    m.ore1_stock.current_mass.value + m.ore2_stock.current_mass.value,
                ),
            )
            self.telemetry.register_metric(
                "face1_extracted_mass",
                lambda t, m, s, h: m.face1.cumulative_extracted_mass.value,
            )
            self.telemetry.register_metric(
                "face2_extracted_mass",
                lambda t, m, s, h: m.face2.cumulative_extracted_mass.value,
            )
            self.telemetry.register_metric(
                "face1_parcel_mass",
                lambda t, m, s, h: m.face1.active_parcel_initial_mass.value,
            )
            self.telemetry.register_metric(
                "face1_parcel_ratio",
                lambda t, m, s, h: m.face1.active_parcel_ore_fraction.value,
            )
            self.telemetry.register_metric(
                "face2_parcel_mass",
                lambda t, m, s, h: m.face2.active_parcel_initial_mass.value,
            )
            self.telemetry.register_metric(
                "face2_parcel_ratio",
                lambda t, m, s, h: m.face2.active_parcel_ore_fraction.value,
            )
            self.telemetry.register_metric(
                "mixed_extraction_rate",
                lambda t, m, s, h: sum(
                    rate.value for rate in m.controller.face_actual_rates
                ),
            )
            self.telemetry.register_metric(
                "mixed_required_extraction_rate",
                lambda t, m, s, h: sum(
                    rate.value for rate in m.controller.face_required_rates
                ),
            )
            self.telemetry.register_metric(
                "mixed_max_extraction_rate",
                lambda t, m, s, h: sum(
                    rate.value for rate in m.controller.face_max_extraction_rates
                ),
            )
            self.telemetry.register_metric(
                "mixed_ore1_fraction",
                lambda t, m, s, h: 1.0 - m.fleet.stockpile2_routing_fraction.value,
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
                lambda t, m, s, h: m.controller.current_campaign_duration.value,
            )
            self.telemetry.register_metric(
                "Contingency",
                lambda t, m, s, h: m.controller.current_contingency_duration.value,
            )

    def forward(self):
        self.global_time.rate = 1.0
        self.controller()
        ore1_flow, ore2_flow = self.fleet(
            self.face1(self.controller.face_actual_rates[0]),
            self.face2(self.controller.face_actual_rates[1]),
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


class UndergroundMaterialHandlingModel(drs.Module):
    """Standalone underground material handling sandbox for parcel flow testing."""

    def __init__(
        self,
        initial_blasted_rock_mass: float = 2000.0,
        initial_remuck_mass: float = 0.0,
        ore1_fraction: float = 0.7,
        truck_capacity_rate: float = 600.0,
        drill_units: float = 1.0,
        lhd_units: float = 1.0,
        truck_units: float = 1.0,
        drill_downtime_start: float = float("inf"),
        drill_downtime_duration: float = 0.0,
        drill_downtime_schedules=None,
        drill_time_between_failures_hours=None,
        drill_repair_time_hours=None,
        drill_work_required: float = 1.0,
        drill_rate_per_drill: float = 0.5,
        blast_tonnage: float = 1000.0,
        lhd_downtime_start: float = float("inf"),
        lhd_downtime_duration: float = 0.0,
        lhd_downtime_schedules=None,
        lhd_time_between_failures_hours=None,
        lhd_repair_time_hours=None,
        truck_downtime_start: float = float("inf"),
        truck_downtime_duration: float = 0.0,
        truck_downtime_schedules=None,
        truck_time_between_failures_hours=None,
        truck_repair_time_hours=None,
        aggregate_parcel_mass: float = 300.0,
        travel_time: float = 0.1,
        target_received_mass: float = 1200.0,
    ):
        raise RuntimeError(
            "UndergroundMaterialHandlingModel is disabled for now. "
            "Use ActiveFleetConcentratorModel / many_faces_simulation while the "
            "model follows face-level allocation and continuous fleet flow."
        )
        super().__init__()
        self.global_time = drs.Timer("GlobalTime", initial_value=0.0)
        self.nominal_truck_capacity_rate = drs.Variable(
            "nominal_truck_capacity_rate",
            truck_capacity_rate,
        )
        self.effective_truck_capacity_rate = drs.Variable(
            "effective_truck_capacity_rate",
            truck_capacity_rate,
        )
        self.flush_loaded_buffer = drs.Variable("flush_loaded_buffer", False)
        self.target_received_mass = target_received_mass

        self.geology_generator = StochasticFaciesGenerator(
            mean_fraction=1.0 - ore1_fraction,
            std_dev=0.075,
            prob_new_facies=0.3,
            variation_same_facies=0.01,
        )
        self.geology_face = UndergroundGeologyFace(
            config=ConcentratorConfig(),
            face_id=1,
            generator=self.geology_generator,
        )

        self.face_inventory = UndergroundFaceInventory(
            face_id=1,
            initial_blasted_rock_mass=initial_blasted_rock_mass,
            initial_remuck_mass=initial_remuck_mass,
            initial_ore1_fraction=ore1_fraction,
        )
        self.drill_availability = _equipment_availability(
            name="DrillAvailability",
            total_units=drill_units,
            downtime_start=drill_downtime_start,
            downtime_duration=drill_downtime_duration,
            schedules=drill_downtime_schedules,
            time_between_failures_hours=drill_time_between_failures_hours,
            repair_time_hours=drill_repair_time_hours,
        )
        self.lhd_availability = _equipment_availability(
            name="LHDAvailability",
            total_units=lhd_units,
            downtime_start=lhd_downtime_start,
            downtime_duration=lhd_downtime_duration,
            schedules=lhd_downtime_schedules,
            time_between_failures_hours=lhd_time_between_failures_hours,
            repair_time_hours=lhd_repair_time_hours,
        )
        self.truck_availability = _equipment_availability(
            name="TruckAvailability",
            total_units=truck_units,
            downtime_start=truck_downtime_start,
            downtime_duration=truck_downtime_duration,
            schedules=truck_downtime_schedules,
            time_between_failures_hours=truck_time_between_failures_hours,
            repair_time_hours=truck_repair_time_hours,
        )
        self.drill_blast = DrillBlastModule(
            drill_work_required=drill_work_required,
            drill_rate_per_drill=drill_rate_per_drill,
            blast_tonnage=blast_tonnage,
        )
        self.lhd = LHDLoadingModule(
            available_lhds=lhd_units,
            remuck_loading_capacity_per_lhd=1000.0,
            face_tramming_capacity_per_lhd=700.0,
        )
        self.haulage = AggregateTruckHaulageModule(
            source_face=1,
            aggregate_parcel_mass=aggregate_parcel_mass,
            travel_time=travel_time,
        )
        self.parcel_receiver = ParcelStockpileReceiver()

    def forward(self):
        self.global_time.rate = 1.0

        available_drills = self.drill_availability()
        available_lhds = self.lhd_availability()
        available_trucks = self.truck_availability()
        requested_blast_mass = self.drill_blast(available_drills)
        blast_output = self.geology_face(requested_blast_mass)
        self.effective_truck_capacity_rate.value = (
            self.nominal_truck_capacity_rate.value
            * available_trucks.value
            / max(1e-6, self.truck_availability.total_units.value)
        )
        self.flush_loaded_buffer.value = (
            self.lhd_availability.just_went_down.value
            or self.truck_availability.just_went_down.value
        )

        loading_rates = self.lhd(
            self.face_inventory.blasted_rock_mass,
            self.face_inventory.remuck_mass,
            self.effective_truck_capacity_rate,
            available_lhds,
        )
        self.face_inventory(blast_output=blast_output, lhd_loading_rates=loading_rates)
        arrived_parcels = self.haulage(
            loading_rates,
            self.face_inventory.loaded_ore1_fraction,
            self.flush_loaded_buffer,
        )
        self.parcel_receiver(arrived_parcels)

    def is_terminating_condition_met(self):
        return self.parcel_receiver.total_mass.value >= self.target_received_mass

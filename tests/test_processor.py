import pytest
from drs import Processor, Flow, Entity


def test_processor_pass_through_and_rate_limit():
    proc = Processor(name="mill", max_rate=5000.0)
    inflow = Flow(rate=6000.0, attributes={"Cu": 0.75, "Au": 1.2})

    outflow = proc.process(inflow)
    assert proc.actual_rate == 5000.0
    assert outflow.rate == 5000.0
    assert outflow.attributes["Cu"] == 0.75
    assert outflow.attributes["Au"] == 1.2

    proc.step(1.0)
    assert proc.cumulative_processed.value == 5000.0


def test_processor_metallurgical_separation():
    proc = Processor(name="flotation_bank", max_rate=10000.0)
    # Feed: 1000 t/h @ 1.0% Cu, 0.5 g/t Au
    inflow = Flow(rate=1000.0, attributes={"Cu": 1.0, "Au": 0.5})

    # Separate into concentrate (5% mass pull, 90% Cu recovery, 80% Au recovery) and tailings
    conc, tails = proc.separate(
        inflow,
        recoveries={"Cu": 0.90, "Au": 0.80},
        mass_pull=0.05,
    )

    # 5% mass pull = 50 t/h conc, 950 t/h tails
    assert conc.rate == pytest.approx(50.0)
    assert tails.rate == pytest.approx(950.0)

    # Conc grade: (1.0% * 0.90) / 0.05 = 18.0% Cu
    assert conc.attributes["Cu"] == pytest.approx(18.0)
    # Tail grade: (1.0% * 0.10) / 0.95 = 0.10526% Cu
    assert tails.attributes["Cu"] == pytest.approx(0.10 / 0.95)

    # Mass balance check: 50 * 18.0 + 950 * (0.10/0.95) = 900 + 100 = 1000 total metal
    total_metal = conc.rate * conc.attributes["Cu"] + tails.rate * tails.attributes["Cu"]
    assert total_metal == pytest.approx(inflow.rate * inflow.attributes["Cu"])


def test_processor_discrete_entity():
    proc = Processor(name="crusher")
    entity = Entity(mass=100.0, attributes={"hardness": 5.0})
    out = proc.process_entity(entity)
    assert out.mass == 100.0
    assert proc.cumulative_processed.value == 100.0

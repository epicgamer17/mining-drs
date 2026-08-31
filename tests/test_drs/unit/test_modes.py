"""Unit tests for generic OperatingMode and configuration mode instances."""

import pytest
from drs_mining.components.modes import OperatingMode, RequireDecision
from drs_mining.config import (
    MILL_MODES,
    FLEET_MODES,
    MILL_MODE_CONFIGS,
    FLEET_MODE_CONFIGS,
    MillModeConfig,
    FleetModeConfig,
)


def test_generic_operating_mode_instantiation():
    """Verify generic OperatingMode creation with arbitrary name, category, and metadata."""
    mode = OperatingMode(
        name="CRUSHING_CAMPAIGN_1",
        id=101,
        category="processing",
        target_throughput=4500.0,
        feed_grade=0.035,
    )
    assert mode.name == "CRUSHING_CAMPAIGN_1"
    assert mode.id == 101
    assert mode.value == 101
    assert mode.category == "processing"
    assert mode.metadata["target_throughput"] == 4500.0
    assert mode.metadata["feed_grade"] == 0.035
    assert str(mode) == "CRUSHING_CAMPAIGN_1"
    assert "OperatingMode(CRUSHING_CAMPAIGN_1, category='processing')" in repr(mode)


def test_operating_mode_default_id_generation():
    """Verify automatic ID generation when ID is omitted."""
    mode1 = OperatingMode("AUTO_MODE_1", category="fleet")
    mode2 = OperatingMode("AUTO_MODE_2", category="fleet")
    assert isinstance(mode1.id, int)
    assert isinstance(mode2.id, int)
    assert mode1.id != mode2.id


def test_operating_mode_equality_and_hashing():
    """Verify equality semantics and dictionary key / set compatibility."""
    m1 = OperatingMode("BALANCED", id=1, category="fleet")
    m2 = OperatingMode("BALANCED", id=1, category="fleet")
    m_mill = OperatingMode("BALANCED", id=1, category="mill")

    assert m1 == m2
    assert m1 != m_mill  # Different category!
    assert m1 == "BALANCED"

    # Set and Dict hashing
    mode_set = {m1, m2, m_mill}
    assert len(mode_set) == 2  # m1 and m2 deduplicate; m_mill is distinct

    mode_map = {m1: "fleet_action", m_mill: "mill_action"}
    assert mode_map[m2] == "fleet_action"
    assert mode_map[m_mill] == "mill_action"


def test_config_mill_mode_instances():
    """Verify standard MILL_MODES are valid OperatingMode instances with category='mill'."""
    assert len(MILL_MODES) == len(MILL_MODE_CONFIGS)
    for name, mode in MILL_MODES.items():
        assert isinstance(mode, OperatingMode)
        assert mode.name == name
        assert mode.category == "mill"
        assert name in MILL_MODE_CONFIGS


def test_config_fleet_mode_instances():
    """Verify standard FLEET_MODES are valid OperatingMode instances with category='fleet'."""
    assert len(FLEET_MODES) == len(FLEET_MODE_CONFIGS)
    for name, mode in FLEET_MODES.items():
        assert isinstance(mode, OperatingMode)
        assert mode.name == name
        assert mode.category == "fleet"
        assert name in FLEET_MODE_CONFIGS

    assert FLEET_MODES["DEVELOPMENT"].metadata["dev_reservation_fraction"] == 0.20
    assert FLEET_MODES["PRODUCTION"].metadata["dev_reservation_fraction"] == 0.0


def test_require_decision_exception():
    """Verify RequireDecision exception behavior."""
    with pytest.raises(RequireDecision):
        raise RequireDecision("External operator decision needed")

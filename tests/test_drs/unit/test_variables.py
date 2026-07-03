import pytest
from drs.variables import Variable, Level, Timer


def test_variable_initialization():
    var = Variable("test_var", 10.5)
    assert var.name == "test_var"
    assert var.value == 10.5

    var_default = Variable("default_var")
    assert var_default.value == 0.0


def test_level_initialization_and_update():
    level = Level("water_level", 100.0, rate=-5.0)
    assert level.name == "water_level"
    assert level.value == 100.0
    assert level.rate == -5.0

    level._update(2.0)  # dt = 2.0
    assert level.value == 90.0


def test_timer_initialization_update_and_reset():
    timer = Timer("stopwatch")
    assert timer.name == "stopwatch"
    assert timer.value == 0.0
    assert timer.rate == 1.0

    timer._update(5.5)
    assert timer.value == 5.5

    timer.reset()
    assert timer.value == 0.0


def test_timer_countdown():
    timer = Timer("countdown", initial_value=10.0, rate=-1.0)
    timer._update(3.0)
    assert timer.value == 7.0

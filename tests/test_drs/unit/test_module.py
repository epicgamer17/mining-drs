import pytest
from drs.module import Module
from drs.variables import Level, Timer


def test_module_variable_list_registration():
    class TestModule(Module):
        def __init__(self):
            super().__init__()
            for i in range(5):
                setattr(self, f"truck_{i}", Level(f"truck_{i}", 0.0))
            self.single = Timer("single_timer", 0.0)

        def forward(self):
            pass

    mod = TestModule()

    # 5 trucks + 1 single = 6 variables
    vars_list = list(mod.variables())
    assert len(vars_list) == 6

    # Check that they were registered with the expected names
    assert "truck_0" in mod._variables
    assert "truck_4" in mod._variables
    assert "single" in mod._variables

    assert mod._variables["truck_0"].name == "truck_0"
    assert mod._variables["truck_4"].name == "truck_4"


def test_semantic_modules_can_own_nested_physical_state():
    class Truck(Module):
        def __init__(self, capacity):
            super().__init__()
            self.payload = Level("Payload")
            self.capacity = capacity

        def forward(self):
            pass

    class Fleet(Module):
        def __init__(self, num_trucks):
            super().__init__()
            for i in range(num_trucks):
                setattr(self, f"truck_{i}", Truck(capacity=50))
            self.total_delivered = Level("Total_Delivered")

        def forward(self):
            for truck in self._modules.values():
                if truck.payload.value == truck.capacity:
                    self.total_delivered.rate += truck.payload.value

    fleet = Fleet(num_trucks=2)

    assert list(fleet._modules) == ["truck_0", "truck_1"]
    assert fleet.truck_0.parent is fleet
    assert fleet.truck_1.payload._owner is fleet.truck_1
    assert [name for name, _ in fleet.named_modules()] == ["", "truck_0", "truck_1"]

    names = {var.name for var in fleet.variables()}
    assert names == {"Payload", "Total_Delivered"}
    assert len(list(fleet.variables())) == 3


def test_module_registers_variable_and_module_direct_assignment():
    class Leaf(Module):
        def __init__(self, name):
            super().__init__()
            self.level = Level(name)

        def forward(self):
            pass

    class Plant(Module):
        def __init__(self):
            super().__init__()
            self.ore = Level("Ore")
            self.water = Level("Water")
            self.crusher = Leaf("Crusher")
            self.mill = Leaf("Mill")

        def forward(self):
            pass

    plant = Plant()

    assert set(plant._variables) == {"ore", "water"}
    assert set(plant._modules) == {"crusher", "mill"}
    assert plant.crusher.parent is plant
    assert plant.ore._owner is plant

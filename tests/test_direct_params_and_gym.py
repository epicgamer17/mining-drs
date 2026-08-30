from drs_mining.components import (
    MineFace,
    OperatingModeController,
    FleetController,
    load_topology_dict,
    build_mining_simulation,
)
from drs_mining.rl.environments import MiningRLEnv
import gymnasium as gym
import json
import os


def test_direct_parameter_instantiation():
    faces, fleet, plant, mode_controller, fleet_controller, ore1_stock, ore2_stock = (
        build_mining_simulation(
            num_faces=1,
            target_ore_stock_level=50000.0,
            mean_ore_fraction=0.35,
            total_ore_to_extract=1000000.0,
        )
    )
    mine = faces[0]
    assert mine.mean_ore_fraction == 0.35
    assert mine.total_ore_to_extract == 1000000.0
    assert mode_controller.target_ore_stock_level == 50000.0
    assert plant.target_ore_stock_level == 50000.0

    faces, fleet, plant, mode_controller, fleet_controller, ore1_stock, ore2_stock = (
        build_mining_simulation(
            num_faces=2,
            total_truck_count=12.0,
            total_lhd_count=4.0,
        )
    )
    assert faces and len(faces) == 2
    assert all(isinstance(face, MineFace) for face in faces)
    assert isinstance(mode_controller, OperatingModeController)
    assert isinstance(fleet_controller, FleetController)
    assert fleet_controller.total_truck_count == 12.0
    assert fleet_controller.total_lhd_count == 4.0


def test_topology_dict_loading(tmp_path):
    flat_data = [{"road_id": 1, "length": 100.0}]
    tree_data = {"segments": [{"road_id": 1, "length": 100.0}]}

    # Test direct list and dict
    assert isinstance(load_topology_dict(flat_data), list)
    assert isinstance(load_topology_dict(tree_data), dict)

    # Test JSON string
    assert isinstance(load_topology_dict(json.dumps(flat_data)), list)
    assert isinstance(load_topology_dict(json.dumps(tree_data)), dict)

    # Test file paths
    flat_file = tmp_path / "flat.json"
    flat_file.write_text(json.dumps(flat_data))
    assert isinstance(load_topology_dict(str(flat_file)), list)

    tree_file = tmp_path / "tree.json"
    tree_file.write_text(json.dumps(tree_data))
    assert isinstance(load_topology_dict(str(tree_file)), dict)


def test_gym_make_environment():
    env = gym.make("MiningEnv-v0", max_steps=10, target_ore_stock_level=60000.0)
    assert isinstance(env.unwrapped, MiningRLEnv)

    obs, info = env.reset(seed=42)
    assert obs.shape == (5,)

    truncated = False
    step_count = 0
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        step_count += 1
        if terminated or truncated:
            break

    assert step_count == 10
    assert truncated
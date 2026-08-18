from drs_mining.components import (
    MineFace,
    BlendingController,
    load_topology_dict,
    build_mining_simulation,
)
from drs_mining.rl.environments import MiningRLEnv
import gymnasium as gym
import os


def test_direct_parameter_instantiation():
    faces, fleet, plant, controller, ore1_stock, ore2_stock = (
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
    assert controller.target_ore_stock_level == 50000.0

    faces, fleet, plant, controller, ore1_stock, ore2_stock = (
        build_mining_simulation(
            num_faces=2,
            total_truck_count=12.0,
            total_lhd_count=4.0,
        )
    )
    assert faces and len(faces) == 2
    assert all(isinstance(face, MineFace) for face in faces)
    assert isinstance(controller, BlendingController)
    assert controller.total_truck_count == 12.0
    assert controller.total_lhd_count == 4.0



def test_topology_dict_loading():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    flat_path = os.path.join(root_dir, "drs_topology_flat.json")
    tree_path = os.path.join(root_dir, "drs_topology_tree.json")

    flat_dict = load_topology_dict(flat_path)
    assert isinstance(flat_dict, list)

    tree_dict = load_topology_dict(tree_path)
    assert isinstance(tree_dict, dict)


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
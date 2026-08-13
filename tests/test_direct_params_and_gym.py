import pytest
import os
import json
import gymnasium as gym
import drs_mining.rl.environments
from drs_mining.components import (
    ConcentratorModel,
    ActiveFleetConcentratorModel,
    ConcentratorMineFace,
    ContinuousMineFace,
    ConcentratorPlant,
    ConcentratorController,
    MultiFaceConcentratorController,
    load_topology_dict,
    build_simulation_from_dict,
)
from drs_mining.simulation import ShelswellHybridSimulation
from drs_mining.rl.environments import MiningRLEnv


def test_direct_parameter_instantiation():
    model = ConcentratorModel(
        target_ore_stock_level=50000.0,
        mean_ore_fraction=0.35,
        total_ore_to_extract=1000000.0,
    )
    assert model.target_ore_stock_level == 50000.0
    assert model.mine.mean_ore_fraction == 0.35
    assert model.mine.total_ore_to_extract == 1000000.0
    assert model.controller.target_ore_stock_level == 50000.0

    active_model = ActiveFleetConcentratorModel(
        total_truck_count=12.0,
        total_lhd_count=4.0,
    )
    assert active_model.controller.total_truck_count == 12.0
    assert active_model.controller.total_lhd_count == 4.0


def test_topology_dict_loading():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    flat_path = os.path.join(root_dir, "drs_topology_flat.json")
    tree_path = os.path.join(root_dir, "drs_topology_tree.json")

    flat_dict = load_topology_dict(flat_path)
    assert isinstance(flat_dict, list)

    tree_dict = load_topology_dict(tree_path)
    assert isinstance(tree_dict, dict)

    sim_flat = build_simulation_from_dict(flat_dict)
    assert isinstance(sim_flat, ConcentratorModel)

    sim_tree = build_simulation_from_dict(tree_dict)
    assert isinstance(sim_tree, ConcentratorModel)

    hybrid_sim = ShelswellHybridSimulation(topology_dict=flat_dict)
    assert hybrid_sim.num_trucks == 10


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

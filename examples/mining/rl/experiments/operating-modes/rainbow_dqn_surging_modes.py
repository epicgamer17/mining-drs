"""
Rainbow DQN implementation for the DRS Mining Simulation (Surging Modes).
Adapted from rl-stuff/examples/dqn/rainbow_dqn_cartpole.py.
"""

import sys
import os

# Ensure the root directory is on the path so we can import 'examples.mining'
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
)

from functional.initialization import set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple, Dict, Any, Optional
import numpy as np
import random
import wandb

from functional.replay_buffer import (
    init_per_buffer,
    sample_per,
    update_priorities,
    circular_write_strategy,
    with_per_tracking,
    make_n_step_accumulator,
)
from functional.schedules import get_linear_schedule
from functional.losses import with_per_weights, cross_entropy_loss
from functional.td import compute_categorical_q_td_target
from functional.action_selection import (
    argmax_selector,
    expected_value,
    gather_q_values,
)
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network
from functional.visualization import log_distributional_metrics
from functional.utils import (
    to_tensor,
    to_numpy_action,
)
from networks.noisy_linear import NoisyLinear

# Import the custom environment and config
from drs_mining.rl.environments import MiningRLEnv, RLMineConfig
from drs_mining.components import ConcentratorConfig

# --- Constants ---
BATCH_SIZE = 128
GAMMA = 1.0  # No discount!
LEARNING_RATE = 1e-3
MAX_STEPS = 120_000
UPDATE_FREQ = 4
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 500
TARGET_NET_UPDATE_FREQ = 100
HIDDEN_SIZE = 64
SEED = 42

# PER Constants
ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100_000

# Distributional (C51) Constants
V_MIN = -100.0
V_MAX = 0.0  # Increased for mining-drs which likely has larger scale rewards
ATOM_SIZE = 51
SUPPORT = torch.linspace(V_MIN, V_MAX, ATOM_SIZE)

# Multi-step
N_STEPS = 3

# Seeding for reproducibility
set_seed(SEED)


class RainbowNetwork(nn.Module):
    """
    Rainbow DQN Network: Combines Dueling architecture, Noisy Nets, and Distributional RL.
    """

    def __init__(self, input_shape: Tuple, num_actions: int, atom_size: int = 51):
        super().__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.atom_size = atom_size

        # Shared feature extractor
        self.feature_layer = NoisyLinear(input_shape[0], HIDDEN_SIZE)

        # Dueling Heads: Value and Advantage
        # Both output distributions over atoms
        self.advantage_head = nn.Sequential(
            NoisyLinear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            NoisyLinear(HIDDEN_SIZE, num_actions * atom_size),
        )
        self.value_head = nn.Sequential(
            NoisyLinear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            NoisyLinear(HIDDEN_SIZE, atom_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.feature_layer(x))

        advantage = self.advantage_head(x).view(-1, self.num_actions, self.atom_size)
        value = self.value_head(x).view(-1, 1, self.atom_size)

        # Dueling combination for distributional RL
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_atoms

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=0.0)
    args = parser.parse_args()

    # --- 1. Initialization (Defining the State) ---
    sim_config = ConcentratorConfig(
        replication_length=9999.0,
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
    )
    config = RLMineConfig(sim_config=sim_config)
    # NOTE: Rainbow DQN fails without dense rewards
    env = MiningRLEnv(config, reward_type="dense")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    device = torch.device("cpu")

    model = RainbowNetwork(obs_shape, num_actions, ATOM_SIZE).to(device)
    target_model = RainbowNetwork(obs_shape, num_actions, ATOM_SIZE).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize Prioritized Replay Buffer
    buffer_state = init_per_buffer(
        capacity=BUFFER_CAPACITY,
        shapes={
            "obs": obs_shape,
            "action": (1,),
            "reward": (),
            "terminated": (),
            "truncated": (),
            "next_obs": obs_shape,
            "gamma": (),
        },
        device=device,
    )
    per_add_transition = with_per_tracking(circular_write_strategy)

    # Initialize N-Step Accumulator
    accumulate_n_step, reset_accumulator = make_n_step_accumulator(
        n_steps=N_STEPS, gamma=GAMMA
    )

    obs, info = env.reset(seed=SEED)
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    # Initialize W&B
    wandb.init(
        project="rainbow-dqn-mining-drs",
        name=f"rainbow_dqn_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}",
        config={
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE,
            "buffer_capacity": BUFFER_CAPACITY,
            "n_steps": N_STEPS,
            "atom_size": ATOM_SIZE,
        },
    )

    # --- 2. The Monolithic Loop ---
    for step in range(MAX_STEPS):
        # 1. Act
        with torch.inference_mode():
            obs_tensor = to_tensor(obs[None, ...], device=device)

            model.reset_noise()

            predictions = model(obs_tensor)
            expected_qs = expected_value(predictions, support=SUPPORT.to(device))
            action, _ = argmax_selector(expected_qs)
            action_np = to_numpy_action(action)

        # 2. Step Env
        action_int = int(action_np.item())
        next_obs, reward, terminated, truncated, env_info = env.step(action_int)

        # 3. N-Step Accumulation and Buffer Addition
        n_step_transitions = accumulate_n_step(
            to_tensor(obs).unsqueeze(0),
            action,
            to_tensor([reward]),
            to_tensor(next_obs).unsqueeze(0),
            to_tensor([terminated]),
            to_tensor([truncated]),
        )

        if n_step_transitions.batch_size[0] > 0:
            buffer_state = per_add_transition(buffer_state, n_step_transitions)

        obs = next_obs

        if terminated or truncated:
            if "episode" in env_info:
                wandb.log(
                    {
                        "episode_return": env_info["episode"]["r"][0],
                        "episode_length": env_info["episode"]["l"][0],
                    },
                    step=step,
                )
            obs, _ = env.reset()
            reset_accumulator()

        # --- 3. Update Loop ---
        if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
            beta = get_linear_schedule(step, BETA_START, 1.0, BETA_FRAMES)
            beta_tensor = torch.tensor(beta, dtype=torch.float32, device=device)

            batch, tree_indices, is_weights = sample_per(
                buffer_state, BATCH_SIZE, beta=beta_tensor
            )

            model.reset_noise()
            target_model.reset_noise()

            logits = model(batch["obs"])
            with torch.no_grad():
                next_logits_online = model(batch["next_obs"])
                next_logits_target = target_model(batch["next_obs"])

                next_expected_qs = expected_value(
                    next_logits_online, support=SUPPORT.to(device)
                )
                next_actions, _ = argmax_selector(next_expected_qs)

                td_target = compute_categorical_q_td_target(
                    next_logits_target,
                    next_actions.squeeze(-1),
                    batch["reward"],
                    batch["terminated"],
                    batch["gamma"],
                    support=SUPPORT.to(device),
                    v_min=V_MIN,
                    v_max=V_MAX,
                    atom_size=ATOM_SIZE,
                )

            pred_sa_logits = gather_q_values(logits, batch["action"])

            per_loss_fn = with_per_weights(cross_entropy_loss, is_weights)
            loss, info_dict = per_loss_fn(pred_sa_logits, td_target)

            optimizer = apply_gradients(
                optimizer, loss, model=model, clip_grad_norm=10.0
            )

            buffer_state = update_priorities(
                buffer_state, tree_indices, info_dict["priorities"], alpha=ALPHA
            )

            if step % 100 == 0:
                log_dict = {
                    k: v
                    for k, v in info_dict.items()
                    if k not in ["predictions", "priorities"]
                }
                log_dict.update({"beta": beta})

                log_dict.update(
                    log_distributional_metrics(
                        info_dict, SUPPORT, step, log_chart=(step % 1000 == 0)
                    )
                )

                wandb.log(log_dict, step=step)

        if step % TARGET_NET_UPDATE_FREQ == 0:
            hard_update_target_network(model, target_model)

    # Save the trained model
    import os

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(
        weights_dir,
        f"rainbow_dqn_mining_drs_model_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}.pt",
    )
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to {weights_path}")

    wandb.finish()

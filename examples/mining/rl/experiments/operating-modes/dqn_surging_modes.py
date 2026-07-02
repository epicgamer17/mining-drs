"""
DQN implementation for the DRS Mining Simulation (Surging Modes).
Adapted from rl-stuff/examples/dqn/dqn_cartpole.py.
"""

import sys

sys.path.append("/Users/jonathanlamontange-kratz/Documents/GitHub/rl-stuff")
sys.path.append("/Users/jonathanlamontange-kratz/Documents/GitHub/mining-drs")

from functional.initialization import layer_init, set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from typing import Tuple
import numpy as np
import random
import wandb
from tensordict import TensorDict

from functional.replay_buffer import (
    init_buffer,
    circular_write_strategy,
    uniform_sample,
)
from functional.losses import mse_loss
from functional.td import compute_q_td_target
from functional.action_selection import (
    argmax_selector,
    gather_q_values,
    with_epsilon_greedy,
)
from functional.schedules import get_linear_schedule
from functional.optimizer import apply_gradients
from functional.network import hard_update_target_network
from functional.utils import (
    to_tensor,
    to_numpy_action,
)

# Import the custom environment and config
from examples.mining.rl.environments import MiningRLEnv, RLMineConfig
from examples.mining.components import ConcentratorConfig

# Constants
BATCH_SIZE = 128
GAMMA = 1.0  # No discount!
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY_FRAMES = 50_000
LEARNING_RATE = 1e-3
MAX_STEPS = 120_000
UPDATE_FREQ = 4
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 500
TARGET_NET_UPDATE_FREQ = 100
HIDDEN_SIZE = 64
SEED = 42

# Seeding for reproducibility
set_seed(SEED)


class DQN(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.l1 = layer_init(nn.Linear(input_shape[0], HIDDEN_SIZE))
        self.l2 = layer_init(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE))
        self.l3 = layer_init(nn.Linear(HIDDEN_SIZE, num_actions), std=1.0)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


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
        std_dev_ore_fraction=args.std_dev_ore_fraction
    )
    config = RLMineConfig(sim_config=sim_config)
    # NOTE: DQN fails without dense rewards
    env = MiningRLEnv(config, reward_type="dense")
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Convert shapes for the buffer since Observation is Box(4,)
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    device = torch.device("cpu")

    model = DQN(obs_shape, num_actions).to(device)
    target_model = DQN(obs_shape, num_actions).to(device)
    target_model.load_state_dict(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    buffer_state = init_buffer(
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

    obs, info = env.reset(seed=SEED)
    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    action_selector = with_epsilon_greedy(argmax_selector)

    # Initialize W&B
    wandb.init(
        project="dqn-mining-drs",
        name=f"dqn_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}",
        config={
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "eps_start": EPS_START,
            "eps_end": EPS_END,
            "eps_decay": EPS_DECAY_FRAMES,
            "lr": LEARNING_RATE,
            "total_stockpile_level": args.total_stockpile_level,
            "std_dev_ore_fraction": args.std_dev_ore_fraction,
        },
    )

    for step in range(MAX_STEPS):
        # print(f"Step {step}/{MAX_STEPS}")
        # 1. Calculate Epsilon dynamically for this step
        current_epsilon = get_linear_schedule(
            step, EPS_START, EPS_END, EPS_DECAY_FRAMES
        )

        # 2. Act (Pure function)
        with torch.inference_mode():
            obs_tensor = to_tensor(obs[None, ...], device=device)

            predictions = model(obs_tensor)
            action, info_dict = action_selector(
                predictions=predictions,
                epsilon=current_epsilon,
                num_actions=num_actions,
                generator=rng_key,
            )
            rng_key = info_dict["generator"]
            action_np = to_numpy_action(action)

        # 2. Step Env
        action_int = int(action_np.item())
        next_obs, reward, terminated, truncated, env_info = env.step(action_int)

        # 3. Add to Buffer
        transition = {
            "obs": to_tensor(obs),
            "action": action.squeeze(0).detach().to(torch.long),
            "reward": to_tensor(reward),
            "terminated": to_tensor(terminated),
            "truncated": to_tensor(truncated),
            "next_obs": to_tensor(next_obs),
            "gamma": torch.tensor(GAMMA, dtype=torch.float32),
        }
        buffer_state, _ = circular_write_strategy(
            buffer_state, TensorDict(transition, batch_size=[]).unsqueeze(0)
        )

        # Update state for next tick
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

        # --- 3. The Update Loop ---
        if step > MIN_BUFFER_SIZE and step % UPDATE_FREQ == 0:
            # Sample
            batch = uniform_sample(buffer_state, rng_key, BATCH_SIZE)

            # 1. Forward Passes (Online and Target)
            q_values = model(batch["obs"])
            with torch.no_grad():
                next_q_values = target_model(batch["next_obs"])

                # 2. Next Action Selection (Pure Primitive)
                next_actions, _ = argmax_selector(next_q_values)

                # 3. Target Calculation (Pure Primitive)
                td_target = compute_q_td_target(
                    next_q_values,
                    next_actions.squeeze(-1),
                    batch["reward"],
                    batch["terminated"],
                    batch["gamma"],
                )

            # 4. Prediction Extraction (Current actions)
            pred_sa = gather_q_values(q_values, batch["action"])

            # 5. Loss Calculation (Pure Primitive)
            loss, info_dict_loss = mse_loss(pred_sa, td_target)
            loss = loss.mean()

            # Apply Updates
            optimizer = apply_gradients(
                optimizer, loss, model=model, clip_grad_norm=10.0
            )

            if step % 100 == 0:
                info_dict_loss.update(
                    {
                        "loss": loss.item(),
                        "epsilon": current_epsilon,
                        "q_values/mean": pred_sa.mean().detach(),
                        "q_values/min": pred_sa.min().detach(),
                        "q_values/max": pred_sa.max().detach(),
                        "td_targets/mean": td_target.mean().detach(),
                        "rewards/mean": batch["reward"].mean().detach(),
                    }
                )
                wandb.log(info_dict_loss, step=step)

        # 4. Target Network Update
        if step % TARGET_NET_UPDATE_FREQ == 0:
            hard_update_target_network(model, target_model)

    # Save the trained model
    import os

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"dqn_mining_drs_model_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to {weights_path}")

    wandb.finish()

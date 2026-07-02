"""
PPO implementation for the DRS Mining Simulation (Surging Modes).
Adapted from rl-stuff/examples/ppo/ppo_cartpole.py.
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
from functools import partial

from functional.action_selection import sample_distribution
from functional.optimizer import apply_gradients
from functional.returns import compute_gae
from functional.losses import (
    clipped_surrogate_loss,
    entropy_loss,
    probability_ratio,
    clipped_mse_loss,
    mse_loss,
)
from torch.optim.lr_scheduler import LinearLR
from functional.visualization import compute_explained_variance
from functional.rollout_buffer import (
    init_rollout_buffer,
    store_rollout_step,
    flatten_rollout_buffer,
    record_truncations,
    get_rollout_next_values,
    yield_shuffled_minibatches,
)
from functional.utils import (

    standardize_tensor,
    to_tensor,
    to_numpy_action,
)
from tensordict import TensorDict


# Import the custom environment and config
from examples.mining.rl.environments import MiningRLEnv, RLMineConfig
from examples.mining.components import ConcentratorConfig

# Constants
LEARNING_RATE = 2.5e-4
MAX_ITERATIONS = 976
GAMMA = 1.0  # No discount!
GAE_LAMBDA = 0.95
ENTROPY_COEFF = 0.01
CRITIC_COEFF = 0.5
MAX_GRAD_NORM = 0.5
STEPS_PER_ENV = 128
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 128
CLIP_COEF = 0.2
TARGET_KL = None
NUM_ENVS = 4
SEED = 42

# Seeding for reproducibility
set_seed(SEED)


class ActorCritic(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, x):
        x_actor = self.actor(x)
        x_critic = self.critic(x)
        return x_actor, x_critic


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", type=float, default=5.0)
    args = parser.parse_args()

    # --- 1. Initialization (Defining the State) ---
    def make_env(seed, idx):
        def thunk():
            sim_config = ConcentratorConfig(
                replication_length=99999.0, 
                std_dev_ore_fraction=args.std_dev_ore_fraction, 
                target_ore_stock_level=args.total_stockpile_level
            )
            config = RLMineConfig(sim_config=sim_config)
            env = MiningRLEnv(config)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk

    envs = gym.vector.SyncVectorEnv([make_env(SEED + i, i) for i in range(NUM_ENVS)])

    obs_shape = envs.single_observation_space.shape
    num_actions = envs.single_action_space.n
    device = torch.device("cpu")

    model = ActorCritic(obs_shape, num_actions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)

    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERATIONS
    )

    obs, env_info = envs.reset(seed=SEED)

    # Pre-allocate rollout buffers using the new functional system
    shapes = {
        "observations": obs_shape,
        "actions": (1,),
        "logprobs": (1,),
        "rewards": (),
        "terminated": (),
        "truncated": (),
        "values": (1,),
        "logits": (num_actions,),
    }
    buffer = init_rollout_buffer(
        steps_per_env=STEPS_PER_ENV,
        num_envs=NUM_ENVS,
        shapes=shapes,
        device=device,
    )

    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    # Initialize W&B
    wandb.init(
        project="ppo-mining-drs",
        name=f"ppo_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}",
        config={
            "lr": LEARNING_RATE,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "update_epochs": UPDATE_EPOCHS,
            "minibatch_size": MINIBATCH_SIZE,
            "clip_coef": CLIP_COEF,
            "num_envs": NUM_ENVS,
            "steps_per_env": STEPS_PER_ENV,
        },
    )
    wandb.define_metric("*", step_metric="global_step")
    global_step = 0

    # Using full episodes
    for iteration in range(MAX_ITERATIONS):
        # print("iteration: ", iteration)
        # 1. Data Collection Phase
        with torch.inference_mode():
            for step in range(STEPS_PER_ENV):
                obs_tensor = to_tensor(obs, device=device)
                logits, value = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action, info_dict = sample_distribution(dist, explore=True)
                action_np = to_numpy_action(action)

                # 2. Step Env
                next_obs, reward, terminated, truncated, env_step_info = envs.step(
                    action_np
                )
                global_step += NUM_ENVS

                # 3. Add to "online" buffers
                transition = TensorDict(
                    {
                        "observations": obs_tensor,
                        "actions": action,
                        "logprobs": info_dict["log_prob"].detach(),
                        "rewards": to_tensor(reward, device=device),
                        "terminated": to_tensor(terminated, device=device),
                        "truncated": to_tensor(truncated, device=device),
                        "values": value.detach(),
                        "logits": logits.detach(),
                    },
                    batch_size=[NUM_ENVS],
                )
                store_rollout_step(buffer=buffer, step=step, transition=transition)

                # 4. Handle Truncations (Gymnasium auto-resets)
                if "final_observation" in env_step_info:
                    from functional.utils import extract_vector_env_final_obs

                    env_indices, final_obs = extract_vector_env_final_obs(env_step_info)
                    # Filter to only record environments that were truncated
                    trunc_mask = truncated[env_indices]
                    if trunc_mask.any():
                        record_truncations(
                            buffer,
                            step,
                            torch.as_tensor(
                                env_indices[trunc_mask], dtype=torch.long, device=device
                            ),
                            torch.as_tensor(
                                final_obs[trunc_mask],
                                dtype=torch.float32,
                                device=device,
                            ),
                        )

                if "final_info" in env_step_info:
                    for item in env_step_info["final_info"]:
                        if item is not None and "episode" in item:
                            wandb.log(
                                {
                                    "episode_return": item["episode"]["r"][0],
                                    "episode_length": item["episode"]["l"][0],
                                    "global_step": global_step,
                                }
                            )

                # NOTE: obs = next_obs is moved to here.
                obs = next_obs

            # Compute last values for the re-evaluation pass
            last_obs_tensor = to_tensor(obs, device=device)
            _, last_values = model(last_obs_tensor)

            # Calculate Next Values (handling truncations)
            next_values = get_rollout_next_values(
                buffer,
                last_values,
                get_value_fn=lambda obs: model(obs)[1],
                device=device,
            )

        # 2. Advantage & Target Calculation
        advantages = compute_gae(
            rewards=buffer.data["rewards"],
            terminated=buffer.data["terminated"],
            truncated=buffer.data["truncated"],
            values=buffer.data["values"].squeeze(-1),
            next_values=next_values.squeeze(-1),
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
        )

        returns = advantages.unsqueeze(-1) + buffer.data["values"]

        # Flatten buffer for loss calculations
        flat_data = flatten_rollout_buffer(buffer)
        flat_advantages = advantages.view(-1, 1)
        flat_returns = returns.view(-1, 1)

        flat_data["advantages"] = flat_advantages
        flat_data["returns"] = flat_returns

        # 3. The Update Loop (Multiple Epochs & Minibatches)
        epoch_losses = []
        clip_fractions = []
        approx_kls = []

        for epoch in range(UPDATE_EPOCHS):
            epoch_kls = []
            for mb in yield_shuffled_minibatches(
                flat_data, MINIBATCH_SIZE, generator=rng_key
            ):
                # Re-evaluate the policy and value function on the minibatch
                new_logits, new_values = model(mb["observations"])

                # Re-calculate log probabilities and entropy
                dist = torch.distributions.Categorical(logits=new_logits)
                # dist has batch shape [B]. mb["actions"] is [B, 1]. Squeeze for log_prob, unsqueeze output.
                new_log_probs = dist.log_prob(mb["actions"].squeeze(-1)).unsqueeze(-1)

                # 1. Policy Loss (Clipped Surrogate)
                ratio = probability_ratio(
                    old_log_probs=mb["logprobs"],
                    new_log_probs=new_log_probs,
                )

                # Advantage Standardisation (Minibatch level as per CleanRL)
                mb_advantages = mb["advantages"]
                mb_advantages = standardize_tensor(mb_advantages)

                pg_loss, pg_info = clipped_surrogate_loss(
                    ratio=ratio,
                    advantages=mb_advantages,
                    clip_coef=CLIP_COEF,
                )
                pg_loss = pg_loss.mean()

                # 2. Value Loss
                critic_loss, _ = clipped_mse_loss(
                    predictions=new_values,
                    old_predictions=mb["values"],
                    targets=mb["returns"],
                    clip_coef=CLIP_COEF,
                )
                critic_loss = critic_loss.mean()

                # 3. Entropy Loss
                ent_loss, _ = entropy_loss(dist)
                ent_loss = ent_loss.mean()

                # Total Loss
                loss = pg_loss + CRITIC_COEFF * critic_loss + ENTROPY_COEFF * ent_loss

                # Apply Updates
                optimizer = apply_gradients(
                    optimizer, loss, model=model, clip_grad_norm=MAX_GRAD_NORM
                )

                # Track Metrics
                with torch.no_grad():
                    epoch_kls.append(pg_info["policy/approx_kl"].item())
                    clip_fractions.append(pg_info["policy/clip_fraction"].item())
                    approx_kls.append(pg_info["policy/approx_kl"].item())
                    epoch_losses.append(loss.item())

            if TARGET_KL is not None and np.mean(epoch_kls) > 1.5 * TARGET_KL:
                break

        # Step the learning rate down
        scheduler.step()

        buffer.truncation_records.clear()

        # Logging
        if iteration % 10 == 0:
            explained_var = compute_explained_variance(
                flat_returns.detach().cpu().numpy(),
                flat_data["values"].detach().cpu().numpy(),
            )

            log_dict = {
                "learning_rate": scheduler.get_last_lr()[0],
                "loss/total": np.mean(epoch_losses),
                "loss/critic": critic_loss.item(),
                "loss/policy": pg_loss.item(),
                "loss/entropy": ent_loss.item(),
                "value/mean": flat_data["values"].mean().item(),
                "value/return_mean": flat_returns.mean().item(),
                "value/explained_variance": explained_var,
                "advantages/mean": flat_advantages.mean().item(),
                "advantages/std": flat_advantages.std().item(),
                "ppo/clip_fraction": np.mean(clip_fractions),
                "ppo/approx_kl": np.mean(approx_kls),
                "global_step": global_step,
            }
            wandb.log(log_dict)

    # Save the trained model
    import os

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"ppo_mining_drs_model_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to {weights_path}")

    wandb.finish()

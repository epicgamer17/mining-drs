"""
PPO LSTM implementation for the DRS Mining Simulation (Surging Modes).
Adapted from rl-stuff/examples/ppo/ppo_lstm_cartpole.py.
"""

import sys
import os

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
from einops import rearrange
from tensordict import TensorDict

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
from functional.network import unroll_rnn
from functional.rollout_buffer import (
    init_rollout_buffer,
    store_rollout_step,
    flatten_rollout_buffer,
    record_truncations,
    get_rollout_next_values,
    yield_sequential_minibatches,
)
from functional.utils import (

    standardize_tensor,
    to_tensor,
    to_numpy_action,
    extract_vector_env_final_obs,
)


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


class ActorCriticLSTM(nn.Module):
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        # Separate Actor Path
        self.actor_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor_lstm = nn.LSTM(64, 64)
        for name, param in self.actor_lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor_head = layer_init(nn.Linear(64, num_actions), std=0.01)

        # Separate Critic Path
        self.critic_feature_extractor = nn.Sequential(
            layer_init(nn.Linear(input_shape[0], 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.critic_lstm = nn.LSTM(64, 64)
        for name, param in self.critic_lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.critic_head = layer_init(nn.Linear(64, 1), std=1.0)

    def forward(
        self,
        x: torch.Tensor,
        lstm_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        dones: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        # lstm_state is [actor_h, actor_c, critic_h, critic_c]
        actor_h, actor_c, critic_h, critic_c = lstm_state

        B = actor_h.shape[1]

        if x.shape[0] % B != 0:
            B = x.shape[0]
            actor_h = torch.zeros(
                self.actor_lstm.num_layers,
                B,
                self.actor_lstm.hidden_size,
                device=x.device,
            )
            actor_c = torch.zeros(
                self.actor_lstm.num_layers,
                B,
                self.actor_lstm.hidden_size,
                device=x.device,
            )
            critic_h = torch.zeros(
                self.critic_lstm.num_layers,
                B,
                self.critic_lstm.hidden_size,
                device=x.device,
            )
            critic_c = torch.zeros(
                self.critic_lstm.num_layers,
                B,
                self.critic_lstm.hidden_size,
                device=x.device,
            )
            dones = torch.zeros(B, device=x.device)

        T = x.shape[0] // B

        # Feature Extraction
        actor_hidden = self.actor_feature_extractor(x)
        critic_hidden = self.critic_feature_extractor(x)

        # Prepare for LSTM: [T*B, F] -> [B, T, F]
        actor_hidden = rearrange(actor_hidden, "(t b) f -> b t f", b=B, t=T)
        critic_hidden = rearrange(critic_hidden, "(t b) f -> b t f", b=B, t=T)
        mb_dones = rearrange(dones, "(t b) -> b t", b=B, t=T)

        # Unroll LSTMs
        actor_hidden_seq, (actor_h, actor_c) = unroll_rnn(
            self.actor_lstm, actor_hidden, (actor_h, actor_c), mb_dones
        )
        critic_hidden_seq, (critic_h, critic_c) = unroll_rnn(
            self.critic_lstm, critic_hidden, (critic_h, critic_c), mb_dones
        )

        # Re-flatten: [B, T, F] -> [T*B, F]
        actor_hidden_seq = rearrange(actor_hidden_seq, "b t f -> (t b) f")
        critic_hidden_seq = rearrange(critic_hidden_seq, "b t f -> (t b) f")

        logits = self.actor_head(actor_hidden_seq)
        value = self.critic_head(critic_hidden_seq)

        return logits, value, (actor_h, actor_c, critic_h, critic_c)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument("--std_dev_ore_fraction", "--std_dev_new_facies", dest="std_dev_ore_fraction", type=float, default=2.0)
    args = parser.parse_args()

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

    model = ActorCriticLSTM(obs_shape, num_actions).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-5)
    scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=MAX_ITERATIONS
    )

    obs, env_info = envs.reset(seed=SEED)

    shapes = {
        "observations": obs_shape,
        "actions": (1,),
        "logprobs": (1,),
        "rewards": (),
        "terminated": (),
        "truncated": (),
        "values": (1,),
        "logits": (num_actions,),
        "dones": (),
    }
    buffer = init_rollout_buffer(
        steps_per_env=STEPS_PER_ENV,
        num_envs=NUM_ENVS,
        shapes=shapes,
        device=device,
    )

    rng_key = torch.Generator(device=device)
    rng_key.manual_seed(SEED)

    wandb.init(
        project="ppo-lstm-mining-drs",
        name=f"ppo_lstm_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}",
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

    next_lstm_state = (
        torch.zeros(
            model.actor_lstm.num_layers,
            NUM_ENVS,
            model.actor_lstm.hidden_size,
            device=device,
        ),
        torch.zeros(
            model.actor_lstm.num_layers,
            NUM_ENVS,
            model.actor_lstm.hidden_size,
            device=device,
        ),
        torch.zeros(
            model.critic_lstm.num_layers,
            NUM_ENVS,
            model.critic_lstm.hidden_size,
            device=device,
        ),
        torch.zeros(
            model.critic_lstm.num_layers,
            NUM_ENVS,
            model.critic_lstm.hidden_size,
            device=device,
        ),
    )
    next_done = torch.zeros(NUM_ENVS, device=device)

    for iteration in range(MAX_ITERATIONS):
        initial_lstm_state = tuple(s.clone() for s in next_lstm_state)

        with torch.inference_mode():
            for step in range(STEPS_PER_ENV):
                obs_tensor = to_tensor(obs, device=device)
                logits, value, next_lstm_state = model(
                    obs_tensor, next_lstm_state, next_done
                )

                dist = torch.distributions.Categorical(logits=logits)
                action, info_dict = sample_distribution(dist, explore=True)
                action_np = to_numpy_action(action)

                next_obs, reward, terminated, truncated, env_step_info = envs.step(
                    action_np
                )
                global_step += NUM_ENVS

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
                        "dones": next_done,
                    },
                    batch_size=[NUM_ENVS],
                )
                store_rollout_step(buffer=buffer, step=step, transition=transition)

                if "final_observation" in env_step_info:
                    env_indices, final_obs = extract_vector_env_final_obs(env_step_info)
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

                obs = next_obs
                next_done = torch.as_tensor(
                    terminated | truncated, dtype=torch.float32, device=device
                )

            last_obs_tensor = to_tensor(obs, device=device)
            _, last_values, _ = model(last_obs_tensor, next_lstm_state, next_done)

            def get_value_fn(o):
                N = o.shape[0]
                zero_states = (
                    torch.zeros(
                        model.actor_lstm.num_layers,
                        N,
                        model.actor_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.actor_lstm.num_layers,
                        N,
                        model.actor_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.critic_lstm.num_layers,
                        N,
                        model.critic_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.critic_lstm.num_layers,
                        N,
                        model.critic_lstm.hidden_size,
                        device=device,
                    ),
                )
                zero_dones = torch.zeros(N, device=device)
                return model(o, zero_states, zero_dones)[1]

            next_values = get_rollout_next_values(
                buffer, last_values, get_value_fn=get_value_fn, device=device
            )

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

        buffer.data["advantages"] = advantages.unsqueeze(-1)
        buffer.data["returns"] = returns

        epoch_losses = []
        clip_fractions = []
        approx_kls = []

        for epoch in range(UPDATE_EPOCHS):
            epoch_kls = []
            minibatch_generator = yield_sequential_minibatches(
                buffer.data,
                num_envs=NUM_ENVS,
                num_minibatches=4,
                initial_lstm_states=initial_lstm_state,
                generator=rng_key,
            )

            for mb, mb_initial_lstm_state in minibatch_generator:
                new_logits, new_values, _ = model(
                    mb["observations"], mb_initial_lstm_state, mb["dones"]
                )

                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(mb["actions"].squeeze(-1)).unsqueeze(-1)

                ratio = probability_ratio(
                    old_log_probs=mb["logprobs"], new_log_probs=new_log_probs
                )
                mb_advantages = standardize_tensor(mb["advantages"])

                pg_loss, pg_info = clipped_surrogate_loss(
                    ratio=ratio, advantages=mb_advantages, clip_coef=CLIP_COEF
                )
                pg_loss = pg_loss.mean()

                critic_loss, _ = clipped_mse_loss(
                    predictions=new_values,
                    old_predictions=mb["values"],
                    targets=mb["returns"],
                    clip_coef=CLIP_COEF,
                )
                critic_loss = critic_loss.mean()

                ent_loss, _ = entropy_loss(dist)
                ent_loss = ent_loss.mean()

                loss = pg_loss + CRITIC_COEFF * critic_loss + ENTROPY_COEFF * ent_loss
                optimizer = apply_gradients(
                    optimizer, loss, model=model, clip_grad_norm=MAX_GRAD_NORM
                )

                with torch.no_grad():
                    epoch_kls.append(pg_info["policy/approx_kl"].item())
                    clip_fractions.append(pg_info["policy/clip_fraction"].item())
                    approx_kls.append(pg_info["policy/approx_kl"].item())
                    epoch_losses.append(loss.item())

            if TARGET_KL is not None and np.mean(epoch_kls) > 1.5 * TARGET_KL:
                break

        scheduler.step()
        buffer.truncation_records.clear()

        if iteration % 10 == 0:
            flat_returns = returns.flatten()
            flat_values = buffer.data["values"].flatten()
            explained_var = compute_explained_variance(
                flat_returns.detach().cpu().numpy(),
                flat_values.detach().cpu().numpy(),
            )

            log_dict = {
                "learning_rate": scheduler.get_last_lr()[0],
                "loss/total": np.mean(epoch_losses),
                "loss/critic": critic_loss.item(),
                "loss/policy": pg_loss.item(),
                "loss/entropy": ent_loss.item(),
                "value/mean": buffer.data["values"].mean().item(),
                "value/return_mean": returns.mean().item(),
                "value/explained_variance": explained_var,
                "advantages/mean": advantages.mean().item(),
                "advantages/std": advantages.std().item(),
                "ppo/clip_fraction": np.mean(clip_fractions),
                "ppo/approx_kl": np.mean(approx_kls),
                "global_step": global_step,
            }
            wandb.log(log_dict)

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"ppo_lstm_mining_drs_model_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Saved model weights to {weights_path}")

    wandb.finish()

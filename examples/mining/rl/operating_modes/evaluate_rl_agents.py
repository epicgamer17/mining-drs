import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Ensure the root directory is on the path so we can import 'examples.mining'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from drs_mining.rl.environments import MiningRLEnv
from dqn_surging_modes import DQN
from ppo_surging_modes import ActorCritic
from ppo_lstm_surging_modes import ActorCriticLSTM
from rainbow_dqn_surging_modes import (
    RainbowNetwork,
    ATOM_SIZE,
    SUPPORT,
)
from functional.action_selection import expected_value


def evaluate_rl_throughput(model, env, seed, device):
    """Run a single episode with the given model and return the throughput."""
    obs, _ = env.reset(seed=seed)
    terminated = False

    if isinstance(model, ActorCriticLSTM):
        lstm_state = (
            torch.zeros(
                model.actor_lstm.num_layers,
                1,
                model.actor_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.actor_lstm.num_layers,
                1,
                model.actor_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.critic_lstm.num_layers,
                1,
                model.critic_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.critic_lstm.num_layers,
                1,
                model.critic_lstm.hidden_size,
                device=device,
            ),
        )
        dones = torch.zeros(1, device=device)

    while not terminated:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if isinstance(model, DQN):
                # DQN evaluation (greedy action)
                q_values = model(obs_tensor)
                action = q_values.argmax(dim=-1).item()
            elif isinstance(model, RainbowNetwork):
                # Rainbow DQN evaluation
                q_atoms = model(obs_tensor)
                expected_qs = expected_value(q_atoms, support=SUPPORT.to(device))
                action = expected_qs.argmax(dim=-1).item()
            elif isinstance(model, ActorCriticLSTM):
                # PPO LSTM evaluation
                logits, _, lstm_state = model(obs_tensor, lstm_state, dones)
                action = logits.argmax(dim=-1).item()
            elif isinstance(model, ActorCritic):
                # PPO evaluation (deterministic action using argmax)
                logits, _ = model(obs_tensor)
                action = logits.argmax(dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        if isinstance(model, ActorCriticLSTM):
            dones = torch.tensor(
                [terminated or truncated], dtype=torch.float32, device=device
            )

    # Calculate throughput from the underlying sim
    sim = env.sim
    config = env.config

    active_time = sim.controller.active_duration()
    if active_time > 0:
        return sim.mine.net_extracted_mass / active_time
    return 0.0


def plot_rl_monte_carlo_throughput(
    model, model_name: str, device, N: int = 10, total_stockpile_level: float = 60000.0
):
    sigmas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    means = []
    stds = []

    print(f"\n--- Running Monte Carlo Evaluation for {model_name} (N={N}) ---")
    for sigma in sigmas:
        env = MiningRLEnv(
            replication_length=99999.0,
            std_dev_ore_fraction=sigma / 100.0,
            target_ore_stock_level=total_stockpile_level,
        )

        throughputs = []
        for seed in range(N):
            tp = evaluate_rl_throughput(model, env, seed, device)
            throughputs.append(tp)

        mean_tp = float(np.mean(throughputs))
        std_tp = float(np.std(throughputs))
        means.append(mean_tp)
        stds.append(std_tp)
        print(f"Sigma: {sigma}%, Mean Throughput: {mean_tp:.2f}, Std Dev: {std_tp:.2f}")

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        sigmas,
        means,
        yerr=stds,
        fmt="-o",
        capsize=5,
        capthick=2,
        ecolor="black",
        markerfacecolor="blue",
        markeredgecolor="blue",
        color="gray",
    )

    plt.title(
        f"Expected Simulated Throughput by Geological Uncertainty ({model_name}, N={N})",
        fontsize=14,
    )
    plt.xlabel("Sigma geo (%)", fontsize=12)
    plt.ylabel("Mean Campaign Throughput (t/d)", fontsize=12)
    plt.ylim(5500, 6000)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(
        f"plots/Monte_Carlo_Throughput_Fig5_{model_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved 'plots/Monte_Carlo_Throughput_Fig5_{model_name}.png'.\n")


def plot_policy_decision_heatmap(
    model, model_name: str, device, config=None, times=[250, 500, 750, 1000]
):
    print(f"\n--- Generating Policy Decision Heatmap for {model_name} ---")
    target_stock = getattr(config, "target_ore_stock_level", getattr(getattr(config, "sim_config", None), "target_ore_stock_level", 60000.0)) if config is not None else 60000.0
    time_scale = getattr(config, "time_scaling_factor", 1000.0) if config is not None else 1000.0

    ore1_vals = np.linspace(0, target_stock, 100)
    ore_fraction_vals = np.linspace(0, 100, 100)

    fig, axes = plt.subplots(
        2, len(times), figsize=(5 * len(times), 10), sharex=True, sharey=True
    )
    if len(times) == 1:
        axes = np.array([axes]).T

    from matplotlib.colors import ListedColormap

    colors = ["#1f77b4", "#d62728"]
    cmap_discrete = ListedColormap(colors)

    for idx, t in enumerate(times):
        batch_size = 10000
        X, Y = np.meshgrid(ore1_vals, ore_fraction_vals)
        grid_ore1 = X.flatten()
        grid_ore_fraction = Y.flatten()
        grid_ore2 = target_stock - grid_ore1

        grid_obs = np.zeros((batch_size, 5), dtype=np.float32)
        grid_obs[:, 0] = grid_ore1 / target_stock
        grid_obs[:, 1] = grid_ore2 / target_stock
        grid_obs[:, 2] = 1.0
        grid_obs[:, 3] = grid_ore_fraction / 100.0
        grid_obs[:, 4] = t / time_scale

        grid_obs_tensor = torch.tensor(grid_obs, device=device)

        with torch.no_grad():
            if isinstance(model, DQN):
                q_values = model(grid_obs_tensor)
                value_cont = (q_values[:, 1] - q_values[:, 0]).cpu().numpy()
                value_disc = q_values.argmax(dim=-1).cpu().numpy()
            elif isinstance(model, RainbowNetwork):
                q_atoms = model(grid_obs_tensor)
                expected_qs = expected_value(q_atoms, support=SUPPORT.to(device))
                value_cont = (expected_qs[:, 1] - expected_qs[:, 0]).cpu().numpy()
                value_disc = expected_qs.argmax(dim=-1).cpu().numpy()
            elif isinstance(model, ActorCriticLSTM):
                dummy_state = (
                    torch.zeros(
                        model.actor_lstm.num_layers,
                        batch_size,
                        model.actor_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.actor_lstm.num_layers,
                        batch_size,
                        model.actor_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.critic_lstm.num_layers,
                        batch_size,
                        model.critic_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.critic_lstm.num_layers,
                        batch_size,
                        model.critic_lstm.hidden_size,
                        device=device,
                    ),
                )
                dones_dummy = torch.zeros(batch_size, device=device)
                logits, _, _ = model(grid_obs_tensor, dummy_state, dones_dummy)
                probs = torch.softmax(logits, dim=-1)
                value_cont = probs[:, 1].cpu().numpy()
                value_disc = logits.argmax(dim=-1).cpu().numpy()
            else:
                logits, _ = model(grid_obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                value_cont = probs[:, 1].cpu().numpy()
                value_disc = logits.argmax(dim=-1).cpu().numpy()

        action_map_cont = value_cont.reshape(100, 100)
        action_map_disc = value_disc.reshape(100, 100)

        ax_cont = axes[0, idx] if len(times) > 1 else axes[0, 0]
        ax_disc = axes[1, idx] if len(times) > 1 else axes[1, 0]

        # --- Top Plot (Continuous) ---
        if isinstance(model, (DQN, RainbowNetwork)):
            max_abs = max(abs(np.min(action_map_cont)), abs(np.max(action_map_cont)))
            if max_abs == 0:
                max_abs = 1e-5
            c1 = ax_cont.pcolormesh(
                X,
                Y,
                action_map_cont,
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
                shading="auto",
            )
            if idx == len(times) - 1:
                fig.colorbar(
                    c1, ax=axes[0, :], label="Q(Mode B) - Q(Mode A)", fraction=0.02
                )
        else:
            c1 = ax_cont.pcolormesh(
                X, Y, action_map_cont, cmap="coolwarm", vmin=0, vmax=1, shading="auto"
            )
            if idx == len(times) - 1:
                fig.colorbar(c1, ax=axes[0, :], label="P(Mode B)", fraction=0.02)

        ax_cont.set_title(f"Time = {t} Days\n(Gradient)")
        if idx == 0:
            ax_cont.set_ylabel("Parcel Ore Fraction (%)")

        # --- Bottom Plot (Discrete) ---
        _ = ax_disc.pcolormesh(
            X,
            Y,
            action_map_disc,
            cmap=cmap_discrete,
            vmin=-0.5,
            vmax=1.5,
            shading="auto",
        )
        ax_disc.set_title(f"Time = {t} Days\n(Sharp)")
        ax_disc.set_xlabel("Ore 1 Stockpile")
        if idx == 0:
            ax_disc.set_ylabel("Parcel Ore Fraction (%)")

    fig.suptitle(f"Policy Decision Heatmap ({model_name})", fontsize=16)
    plt.tight_layout()
    out_name = f"plots/Policy_Decision_Heatmap_{model_name}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{out_name}'.")


def plot_non_stationary_time_slice(
    model,
    model_name: str,
    device,
    config=None,
    ore_fractions=[10, 25, 50, 75, 90],
):
    print(f"\n--- Generating Non-Stationary Time Slice for {model_name} ---")
    target_stock = getattr(config, "target_ore_stock_level", getattr(getattr(config, "sim_config", None), "target_ore_stock_level", 60000.0)) if config is not None else 60000.0
    time_scale = getattr(config, "time_scaling_factor", 1000.0) if config is not None else 1000.0

    time_vals = np.linspace(0, 1200, 100)
    ore1_vals = np.linspace(0, target_stock, 100)

    fig, axes = plt.subplots(
        2,
        len(ore_fractions),
        figsize=(5 * len(ore_fractions), 10),
        sharex=True,
        sharey=True,
    )
    if len(ore_fractions) == 1:
        axes = np.array([axes]).T

    from matplotlib.colors import ListedColormap

    colors = ["#1f77b4", "#d62728"]
    cmap_discrete = ListedColormap(colors)

    for idx, ore_fraction in enumerate(ore_fractions):
        batch_size = 10000
        X, Y = np.meshgrid(time_vals, ore1_vals)
        grid_t = X.flatten()
        grid_ore1 = Y.flatten()
        grid_ore2 = target_stock - grid_ore1

        grid_obs = np.zeros((batch_size, 5), dtype=np.float32)
        grid_obs[:, 0] = grid_ore1 / target_stock
        grid_obs[:, 1] = grid_ore2 / target_stock
        grid_obs[:, 2] = 1.0
        grid_obs[:, 3] = ore_fraction / 100.0
        grid_obs[:, 4] = grid_t / time_scale

        grid_obs_tensor = torch.tensor(grid_obs, device=device)

        with torch.no_grad():
            if isinstance(model, DQN):
                q_values = model(grid_obs_tensor)
                value_cont = (q_values[:, 1] - q_values[:, 0]).cpu().numpy()
                value_disc = q_values.argmax(dim=-1).cpu().numpy()
            elif isinstance(model, RainbowNetwork):
                q_atoms = model(grid_obs_tensor)
                expected_qs = expected_value(q_atoms, support=SUPPORT.to(device))
                value_cont = (expected_qs[:, 1] - expected_qs[:, 0]).cpu().numpy()
                value_disc = expected_qs.argmax(dim=-1).cpu().numpy()
            elif isinstance(model, ActorCriticLSTM):
                dummy_state = (
                    torch.zeros(
                        model.actor_lstm.num_layers,
                        batch_size,
                        model.actor_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.actor_lstm.num_layers,
                        batch_size,
                        model.actor_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.critic_lstm.num_layers,
                        batch_size,
                        model.critic_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.critic_lstm.num_layers,
                        batch_size,
                        model.critic_lstm.hidden_size,
                        device=device,
                    ),
                )
                dones_dummy = torch.zeros(batch_size, device=device)
                logits, _, _ = model(grid_obs_tensor, dummy_state, dones_dummy)
                probs = torch.softmax(logits, dim=-1)
                value_cont = probs[:, 1].cpu().numpy()
                value_disc = logits.argmax(dim=-1).cpu().numpy()
            else:
                logits, _ = model(grid_obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                value_cont = probs[:, 1].cpu().numpy()
                value_disc = logits.argmax(dim=-1).cpu().numpy()

        action_map_cont = value_cont.reshape(100, 100)
        action_map_disc = value_disc.reshape(100, 100)

        ax_cont = axes[0, idx] if len(ore_fractions) > 1 else axes[0, 0]
        ax_disc = axes[1, idx] if len(ore_fractions) > 1 else axes[1, 0]

        # --- Top Plot (Continuous) ---
        if isinstance(model, (DQN, RainbowNetwork)):
            max_abs = max(abs(np.min(action_map_cont)), abs(np.max(action_map_cont)))
            if max_abs == 0:
                max_abs = 1e-5
            c1 = ax_cont.pcolormesh(
                X,
                Y,
                action_map_cont,
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
                shading="auto",
            )
            if idx == len(ore_fractions) - 1:
                fig.colorbar(
                    c1, ax=axes[0, :], label="Q(Mode B) - Q(Mode A)", fraction=0.02
                )
        else:
            c1 = ax_cont.pcolormesh(
                X, Y, action_map_cont, cmap="coolwarm", vmin=0, vmax=1, shading="auto"
            )
            if idx == len(ore_fractions) - 1:
                fig.colorbar(c1, ax=axes[0, :], label="P(Mode B)", fraction=0.02)

        ax_cont.set_title(f"Ore Fraction = {ore_fraction}%\n(Gradient)")
        if idx == 0:
            ax_cont.set_ylabel("Ore 1 Stockpile")

        # --- Bottom Plot (Discrete) ---
        _ = ax_disc.pcolormesh(
            X,
            Y,
            action_map_disc,
            cmap=cmap_discrete,
            vmin=-0.5,
            vmax=1.5,
            shading="auto",
        )
        ax_disc.set_title(f"Ore Fraction = {ore_fraction}%\n(Sharp)")
        ax_disc.set_xlabel("Time (Days)")
        if idx == 0:
            ax_disc.set_ylabel("Ore 1 Stockpile")

    fig.suptitle(f"Non-Stationary Time Slice ({model_name})", fontsize=16)
    plt.tight_layout()
    out_name = f"plots/Non_Stationary_Time_Slice_{model_name}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved '{out_name}'.")


def generate_rl_dashboard(
    model,
    model_name: str,
    device,
    seed: int = 42,
    total_stockpile_level: float = 60000.0,
):
    print(
        f"\n--- Generating Comprehensive Dashboard for {model_name} (Seed={seed}) ---"
    )
    # 2. Get Obs Shape
    env = MiningRLEnv(
        replication_length=99999.0,
        target_ore_stock_level=total_stockpile_level,
        enable_telemetry=True,
    )

    # Run the simulation
    obs, _ = env.reset(seed=seed)
    terminated = False

    if isinstance(model, ActorCriticLSTM):
        lstm_state = (
            torch.zeros(
                model.actor_lstm.num_layers,
                1,
                model.actor_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.actor_lstm.num_layers,
                1,
                model.actor_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.critic_lstm.num_layers,
                1,
                model.critic_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.critic_lstm.num_layers,
                1,
                model.critic_lstm.hidden_size,
                device=device,
            ),
        )
        dones = torch.zeros(1, device=device)

    step_count = 0
    while not terminated:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if isinstance(model, DQN):
                q_values = model(obs_tensor)
                action = q_values.argmax(dim=-1).item()
                print(
                    f"[Step {step_count:4d}] DQN Q-values: {np.round(q_values.cpu().numpy()[0], 2)} | Selected Action: {action}"
                )
            elif isinstance(model, RainbowNetwork):
                q_atoms = model(obs_tensor)
                expected_qs = expected_value(q_atoms, support=SUPPORT.to(device))
                action = expected_qs.argmax(dim=-1).item()
                print(
                    f"[Step {step_count:4d}] Rainbow Q-values: {np.round(expected_qs.cpu().numpy()[0], 2)} | Selected Action: {action}"
                )
            elif isinstance(model, ActorCriticLSTM):
                logits, value, lstm_state = model(obs_tensor, lstm_state, dones)
                action = logits.argmax(dim=-1).item()
                probs = torch.softmax(logits, dim=-1)
                print(
                    f"[Step {step_count:4d}] PPO-LSTM Value: {value.item():.2f} | Action Probs: {np.round(probs.cpu().numpy()[0], 3)} | Selected Action: {action}"
                )
            else:
                logits, value = model(obs_tensor)
                action = logits.argmax(dim=-1).item()
                probs = torch.softmax(logits, dim=-1)
                print(
                    f"[Step {step_count:4d}] PPO Value: {value.item():.2f} | Action Probs: {np.round(probs.cpu().numpy()[0], 3)} | Selected Action: {action}"
                )

        obs, reward, terminated, truncated, _ = env.step(action)
        step_count += 1
        if isinstance(model, ActorCriticLSTM):
            dones = torch.tensor(
                [terminated or truncated], dtype=torch.float32, device=device
            )

    result = env.engine.run(until=env.config.replication_length)

    df = result.history

    # --- Mode Transition Log ---
    # --- Mode Transition Log ---
    state_change_events = [
        e
        for e in result.events
        if e.event_type == "STATE_CHANGE"
        and e.details.get("variable") == "active_operating_mode"
    ]
    if state_change_events:
        print("\n--- Mode Transition Log ---")
        for e in state_change_events:
            old = (
                e.details["old_value"].name
                if hasattr(e.details["old_value"], "name")
                else str(e.details["old_value"])
            )
            new = (
                e.details["new_value"].name
                if hasattr(e.details["new_value"], "name")
                else str(e.details["new_value"])
            )
            print(f"Time: {e.time:.2f} | Transition: {old} -> {new}")
        print("---------------------------\n")

    # --- Cumulative Deficit by Mode Log ---
    import pandas as pd

    dt = df["time"].diff().fillna(0)
    actual_extraction_step = df["cumulative_extracted_mass"].diff().fillna(0)
    ideal_extraction_step = dt * 6000.0
    step_deficit = (ideal_extraction_step - actual_extraction_step).clip(lower=0)

    # We still need the active_operating_mode_name column for downstream plotting and analysis
    df["active_operating_mode_name"] = df["active_operating_mode"].apply(
        lambda x: x.name if hasattr(x, "name") else str(x)
    )

    deficit_df = pd.DataFrame(
        {"mode": df["active_operating_mode_name"], "deficit": step_deficit}
    )

    total_deficit_by_mode = (
        deficit_df.groupby("mode")["deficit"].sum().sort_values(ascending=False)
    )

    print("\n--- Cumulative Lost Production (Deficit) by Mode ---")
    total_lost = total_deficit_by_mode.sum()
    for mode, lost in total_deficit_by_mode.items():
        mode_name = str(mode).split(".")[-1]
        pct = (lost / total_lost * 100) if total_lost > 0 else 0
        print(f"{mode_name}: {lost:.1f} tons ({pct:.1f}%)")
    print(f"TOTAL: {total_lost:.1f} tons")
    print("----------------------------------------------------\n")

    print("\n--- Mode Distribution (% of Time Spent) ---")
    df_dist = df.copy()
    df_dist["dt"] = df_dist["time"].diff().shift(-1).fillna(0)
    durations = df_dist.groupby("active_operating_mode_name")["dt"].sum()
    total_t = durations.sum()
    if total_t > 0:
        for mode, duration in durations.items():
            pct = (duration / total_t) * 100
            print(f"{mode}: {pct:.1f}%")
    print("-" * 43 + "\n")

    # Create Modes Series
    df["Mode A"] = df["active_operating_mode_name"].apply(
        lambda m: (
            3
            if m
            in (
                "MODE_A",
                "MODE_A_CONTINGENCY",
                "MODE_A_MINE_SURGING",
            )
            else 0
        )
    )
    df["Mode B"] = df["active_operating_mode_name"].apply(
        lambda m: (
            2
            if m
            in (
                "MODE_B",
                "MODE_B_CONTINGENCY",
                "MODE_B_MINE_SURGING",
            )
            else 0
        )
    )
    df["Shutdown"] = df["active_operating_mode_name"].apply(
        lambda m: 1 if m == "SHUTDOWN" else 0
    )

    # Create Ore Level Series (scaled by 1000)
    df["total_system_ore_mass"] = df["Ore1Stock_mass"] + df["Ore2Stock_mass"]
    df["Total Ore Stockpile Level"] = df["total_system_ore_mass"] / 1000.0
    df["Ore 1 Stockpile Level"] = df["Ore1Stock_mass"] / 1000.0
    df["Ore 2 Stockpile Level"] = df["Ore2Stock_mass"] / 1000.0

    from drs.plot import (
        plot_time_series,
        plot_dual_axis_step,
        plot_safety_margin,
        build_dashboard,
    )
    from drs_mining.components.plot import (
        plot_ore_with_modes,
        plot_mode_distribution,
        plot_mode_dwell_times,
        plot_normalized_deviation_violin,
        plot_attributed_deficit,
        plot_deficit_disparity,
        plot_deficit_breakdown_bar,
        plot_structural_vs_operational_deficit,
        plot_normalized_cumulative_deficit,
        plot_structural_vs_operational_by_mode,
    )

    palette = {
        "MODE_A": "#1f77b4",
        "MODE_A_CONTINGENCY": "#2ca02c",
        "MODE_A_MINE_SURGING": "#9467bd",
        "MODE_B": "#d62728",
        "MODE_B_CONTINGENCY": "#ff7f0e",
        "MODE_B_MINE_SURGING": "#8c564b",
        "SHUTDOWN": "#FFD700",
    }

    structural_modes = ["SHUTDOWN", "MODE_A"]

    configs = [
        {
            "func": plot_time_series,
            "kwargs": {
                "y_columns": ["Mode A", "Mode B", "Shutdown"],
                "title": "Modes (Step)",
                "is_step": True,
            },
        },
        {
            "func": plot_ore_with_modes,
            "kwargs": {
                "time_col": "time",
                "ore_cols": [
                    "total_system_ore_mass",
                    "Ore1Stock_mass",
                    "Ore2Stock_mass",
                ],
                "mode_col": "active_operating_mode_name",
                "campaign_split_mode": "SHUTDOWN",
                "title": "Ore Stockpiles & Campaigns",
                "palette": palette,
                "hlines": [
                    {
                        "y": 60000,
                        "color": "black",
                        "linestyle": "--",
                        "linewidth": 1.5,
                        "alpha": 0.7,
                        "label": "Target Total (60k)",
                    },
                    {
                        "y": 20400,
                        "color": "red",
                        "linestyle": ":",
                        "linewidth": 2,
                        "alpha": 0.8,
                        "label": "Critical Ore 2 (20.4k)",
                    },
                ],
            },
        },
        {
            "func": plot_dual_axis_step,
            "kwargs": {
                "y1_col": "MassOfCurrentParcel",
                "y2_col": "CurrentParcelRoutingFraction",
                "y1_label": "Parcel Mass (tons)",
                "y2_label": "Grade (% Ore 2)",
                "title": "Current Parcel Properties",
            },
        },
        {
            "func": plot_safety_margin,
            "kwargs": {
                "level_col": "Ore1Stock_mass",
                "constraint_value": 0.0,
                "constraint_type": "lower",
                "title": "Safety Margin: Ore 1 Distance to Floor",
                "danger_threshold": 1000.0,
            },
        },
        {
            "func": plot_safety_margin,
            "kwargs": {
                "level_col": "Ore2Stock_mass",
                "constraint_value": 0.0,
                "constraint_type": "lower",
                "title": "Safety Margin: Ore 2 Distance to Floor",
                "danger_threshold": 1000.0,
            },
        },
        {
            "func": plot_mode_distribution,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "time_col": "time",
                "title": "Mode Distribution (% of Time Spent)",
                "palette": palette,
            },
        },
        {
            "func": plot_mode_dwell_times,
            "kwargs": {
                "time_col": "time",
                "mode_col": "active_operating_mode_name",
                "title": "Mode Stability (Dwell Times)",
            },
        },
        {
            "func": plot_normalized_deviation_violin,
            "kwargs": {
                "title": "Stockpile Deviation Variance (Violin)",
                "target_total": 60000.0,
                "target_ore1": 42000.0,
                "target_ore2": 18000.0,
            },
        },
        {
            "func": plot_attributed_deficit,
            "kwargs": {
                "time_col": "time",
                "mode_col": "active_operating_mode_name",
                "extraction_col": "cumulative_extracted_mass",
                "ideal_rate_per_day": 6000.0,
                "title": "Cumulative Production Deficit by Mode",
                "palette": palette,
            },
        },
        {
            "func": plot_deficit_disparity,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "title": "Mode Efficiency (Time Spent vs. Deficit Caused)",
                "ideal_rate": 6000.0,
            },
        },
        {
            "func": plot_deficit_breakdown_bar,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate_per_day": 6000.0,
                "palette": palette,
            },
        },
        {
            "func": plot_structural_vs_operational_deficit,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate": 6000.0,
                "structural_modes": structural_modes,
            },
        },
        {
            "func": plot_normalized_cumulative_deficit,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate_per_day": 6000.0,
                "palette": palette,
            },
        },
        {
            "func": plot_structural_vs_operational_by_mode,
            "kwargs": {
                "mode_col": "active_operating_mode_name",
                "ideal_rate": 6000.0,
                "structural_modes": structural_modes,
            },
        },
    ]

    fig_comp = build_dashboard(
        df, configs, title=f"Comprehensive Diagnostics ({model_name})", figsize=(18, 69)
    )
    out_name = f"plots/Comprehensive_Diagnostics_Plot_{model_name}.png"
    fig_comp.savefig(out_name)
    plt.close(fig_comp)
    print(f"Saved '{out_name}'.\n")


import imageio
import io


def generate_policy_decision_video(
    model, model_name: str, device, config=None, seed: int = 42
):
    print(f"\n--- Generating Policy Decision Video for {model_name} (Seed={seed}) ---")
    target_stock = getattr(config, "target_ore_stock_level", getattr(getattr(config, "sim_config", None), "target_ore_stock_level", 60000.0)) if config is not None else 60000.0
    env = MiningRLEnv(target_ore_stock_level=target_stock, enable_telemetry=True)
    obs, _ = env.reset(seed=seed)
    terminated = False

    if isinstance(model, ActorCriticLSTM):
        lstm_state = (
            torch.zeros(
                model.actor_lstm.num_layers,
                1,
                model.actor_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.actor_lstm.num_layers,
                1,
                model.actor_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.critic_lstm.num_layers,
                1,
                model.critic_lstm.hidden_size,
                device=device,
            ),
            torch.zeros(
                model.critic_lstm.num_layers,
                1,
                model.critic_lstm.hidden_size,
                device=device,
            ),
        )
        dones = torch.zeros(1, device=device)

    trajectory = []

    target_stock = config.sim_config.target_ore_stock_level
    time_scale = config.time_scaling_factor

    while not terminated:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        ore1 = obs[0] * target_stock
        ore2 = obs[1] * target_stock
        ore_fraction = obs[3] * 100.0
        t_day = obs[4] * time_scale

        with torch.no_grad():
            if isinstance(model, DQN):
                q_values = model(obs_tensor)
                action = q_values.argmax(dim=-1).item()
            elif isinstance(model, RainbowNetwork):
                q_atoms = model(obs_tensor)
                expected_qs = expected_value(q_atoms, support=SUPPORT.to(device))
                action = expected_qs.argmax(dim=-1).item()
            elif isinstance(model, ActorCriticLSTM):
                logits, _, lstm_state = model(obs_tensor, lstm_state, dones)
                action = logits.argmax(dim=-1).item()
            else:
                logits, _ = model(obs_tensor)
                action = logits.argmax(dim=-1).item()

        trajectory.append((t_day, ore1, ore_fraction, action))

        obs, reward, terminated, truncated, _ = env.step(action)
        if isinstance(model, ActorCriticLSTM):
            dones = torch.tensor(
                [terminated or truncated], dtype=torch.float32, device=device
            )

    frames = []
    ore1_vals = np.linspace(0, target_stock, 100)
    ore_fraction_vals = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(ore1_vals, ore_fraction_vals)

    grid_ore1 = X.flatten()
    grid_ore_fraction = Y.flatten()
    grid_ore2 = target_stock - grid_ore1

    from matplotlib.colors import ListedColormap

    # We only have 2 actions in RL (0: Mode A, 1: Mode B)
    colors = ["#1f77b4", "#d62728"]
    cmap_discrete = ListedColormap(colors)

    for step_idx, (
        t_day,
        current_ore1,
        current_ore_fraction,
        chosen_action,
    ) in enumerate(trajectory):
        batch_size = 10000
        grid_obs = np.zeros((batch_size, 5), dtype=np.float32)
        grid_obs[:, 0] = grid_ore1 / target_stock
        grid_obs[:, 1] = grid_ore2 / target_stock
        grid_obs[:, 2] = 1.0
        grid_obs[:, 3] = grid_ore_fraction / 100.0
        grid_obs[:, 4] = t_day / time_scale

        grid_obs_tensor = torch.tensor(grid_obs, device=device)

        with torch.no_grad():
            if isinstance(model, DQN):
                q_values = model(grid_obs_tensor)
                value_cont = (q_values[:, 1] - q_values[:, 0]).cpu().numpy()
                value_disc = q_values.argmax(dim=-1).cpu().numpy()
            elif isinstance(model, RainbowNetwork):
                q_atoms = model(grid_obs_tensor)
                expected_qs = expected_value(q_atoms, support=SUPPORT.to(device))
                value_cont = (expected_qs[:, 1] - expected_qs[:, 0]).cpu().numpy()
                value_disc = expected_qs.argmax(dim=-1).cpu().numpy()
            elif isinstance(model, ActorCriticLSTM):
                dummy_state = (
                    torch.zeros(
                        model.actor_lstm.num_layers,
                        batch_size,
                        model.actor_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.actor_lstm.num_layers,
                        batch_size,
                        model.actor_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.critic_lstm.num_layers,
                        batch_size,
                        model.critic_lstm.hidden_size,
                        device=device,
                    ),
                    torch.zeros(
                        model.critic_lstm.num_layers,
                        batch_size,
                        model.critic_lstm.hidden_size,
                        device=device,
                    ),
                )
                dones_dummy = torch.zeros(batch_size, device=device)
                logits, _, _ = model(grid_obs_tensor, dummy_state, dones_dummy)
                probs = torch.softmax(logits, dim=-1)
                value_cont = probs[:, 1].cpu().numpy()
                value_disc = logits.argmax(dim=-1).cpu().numpy()
            else:
                logits, _ = model(grid_obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                value_cont = probs[:, 1].cpu().numpy()
                value_disc = logits.argmax(dim=-1).cpu().numpy()

        action_map_cont = value_cont.reshape(100, 100)
        action_map_disc = value_disc.reshape(100, 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # --- Left Plot (Continuous) ---
        if isinstance(model, (DQN, RainbowNetwork)):
            max_abs = max(abs(np.min(action_map_cont)), abs(np.max(action_map_cont)))
            if max_abs == 0:
                max_abs = 1e-5
            c1 = ax1.pcolormesh(
                X,
                Y,
                action_map_cont,
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
                shading="auto",
            )
            fig.colorbar(c1, ax=ax1, label="Q(Mode B) - Q(Mode A)")
        else:
            c1 = ax1.pcolormesh(
                X, Y, action_map_cont, cmap="coolwarm", vmin=0, vmax=1, shading="auto"
            )
            fig.colorbar(c1, ax=ax1, label="P(Mode B)")

        # --- Right Plot (Discrete) ---
        _ = ax2.pcolormesh(
            X,
            Y,
            action_map_disc,
            cmap=cmap_discrete,
            vmin=-0.5,
            vmax=1.5,
            shading="auto",
        )

        mode_names = ["Mode A", "Mode B"]
        action_name = mode_names[chosen_action]

        dot_color = "white" if chosen_action == 0 else "black"

        for ax in [ax1, ax2]:
            ax.scatter(
                [current_ore1],
                [current_ore_fraction],
                color=dot_color,
                edgecolor="black",
                s=100,
                label=f"State ({action_name})",
                zorder=5,
            )
            ax.set_xlabel("Ore 1 Stockpile")
            ax.set_ylabel("Parcel Grade (%)")
            ax.legend(loc="upper right")

        fig.suptitle(f"Time = {t_day:.1f} Days | Step {step_idx}")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        frames.append(imageio.v2.imread(buf))
        plt.close(fig)

    out_name = f"Policy_Decision_Video_{model_name}.gif"
    imageio.mimsave(out_name, frames, duration=200)
    print(f"Saved '{out_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["dqn", "ppo", "ppo_lstm", "rainbow_dqn"],
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model weights. Defaults to experiments/weights/<model_type>_mining_drs_model.pt",
    )
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--total_stockpile_level", type=float, default=60000.0)
    parser.add_argument(
        "--std_dev_ore_fraction",
        type=float,
        dest="std_dev_ore_fraction",
        default=5.0,
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    # We need to temporarily instantiate an env to get shapes
    temp_env = MiningRLEnv(
        target_ore_stock_level=args.total_stockpile_level,
        std_dev_ore_fraction=args.std_dev_ore_fraction,
    )
    obs_shape = temp_env.observation_space.shape
    num_actions = temp_env.action_space.n

    if args.model_type == "dqn":
        model = DQN(obs_shape, num_actions).to(device)
        default_model_name = f"dqn_mining_drs_model_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}.pt"
    elif args.model_type == "rainbow_dqn":
        from rainbow_dqn_surging_modes import RainbowNetwork

        model = RainbowNetwork(obs_shape, num_actions, ATOM_SIZE).to(device)
        default_model_name = f"rainbow_dqn_mining_drs_model_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}.pt"
    elif args.model_type == "ppo_lstm":
        from ppo_lstm_surging_modes import ActorCriticLSTM

        model = ActorCriticLSTM(obs_shape, num_actions).to(device)
        default_model_name = f"ppo_lstm_mining_drs_model_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}.pt"
    else:
        from ppo_surging_modes import ActorCritic

        model = ActorCritic(obs_shape, num_actions).to(device)
        default_model_name = f"ppo_mining_drs_model_{int(args.total_stockpile_level)}_{int(args.std_dev_ore_fraction)}.pt"

    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights", default_model_name
        )

    if not os.path.exists(model_path):
        print(
            f"Skipping evaluation for {args.model_type.upper()}: Weights not found at {model_path}"
        )
        sys.exit(0)

    print(f"Loading weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Generate heatmaps and video
    plot_policy_decision_heatmap(model, args.model_type.upper(), device, temp_config)
    generate_policy_decision_video(
        model, args.model_type.upper(), device, temp_config, seed=11
    )
    plot_non_stationary_time_slice(model, args.model_type.upper(), device, temp_config)

    # Generate the comprehensive dashboard for a single run
    generate_rl_dashboard(
        model,
        args.model_type.upper(),
        device,
        seed=11,
        total_stockpile_level=args.total_stockpile_level,
    )

    # Run the Monte Carlo throughput evaluation
    plot_rl_monte_carlo_throughput(
        model,
        args.model_type.upper(),
        device,
        N=args.N,
        total_stockpile_level=args.total_stockpile_level,
    )

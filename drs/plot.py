import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns


def plot_time_series(
    df,
    y_columns: list,
    time_col: str = "time",
    title: str = None,
    y_label: str = None,
    is_step: bool = False,
    ax=None,
    add_legend: bool = True,
    colors: list = None,
    **line_kwargs,
):
    if time_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain a '{time_col}' column for time-series plotting."
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        own_ax = True
    else:
        own_ax = False

    for i, col in enumerate(y_columns):
        if col in df.columns:
            color = colors[i % len(colors)] if colors else None

            kwargs = dict(line_kwargs)
            kwargs.setdefault("linewidth", 2)

            if is_step:
                kwargs.setdefault("where", "post")
                ax.step(df[time_col], df[col], label=col, color=color, **kwargs)
            else:
                ax.plot(df[time_col], df[col], label=col, color=color, **kwargs)

    if title:
        ax.set_title(title, fontsize=14, pad=15)
    if y_label:
        ax.set_ylabel(y_label, fontsize=12)

    if add_legend:
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1, 1.1),
            ncol=len(y_columns),
            frameon=True,
        )

    if own_ax:
        fig = ax.figure
        fig.tight_layout()
        return fig
    return ax



def plot_safety_margin(
    df,
    level_col: str,
    constraint_value: float,
    time_col: str = "time",
    constraint_type: str = "upper",
    title: str = "Safety Margin (Distance to Constraint)",
    danger_threshold: float = None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        own_ax = True
    else:
        own_ax = False

    if constraint_type == "upper":
        margin = constraint_value - df[level_col]
    else:
        margin = df[level_col] - constraint_value

    ax.plot(df[time_col], margin, label="Safety Margin", color="steelblue", linewidth=2)
    ax.axhline(
        y=0,
        color="red",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label="Constraint Boundary",
    )

    if danger_threshold is not None:
        ax.axhline(
            y=danger_threshold,
            color="orange",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Danger Threshold ({danger_threshold})",
        )
        ax.fill_between(
            df[time_col],
            margin,
            0,
            where=(margin < danger_threshold),
            color="red",
            alpha=0.15,
            label="Danger Zone",
        )

    ax.fill_between(
        df[time_col],
        margin,
        0,
        where=(margin < 0),
        color="red",
        alpha=0.3,
        label="Constraint Violated",
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Simulation Time", fontsize=12)
    ax.set_ylabel("Margin (distance to constraint)", fontsize=12)
    ax.legend(loc="best", frameon=True)

    if own_ax:
        fig = ax.figure
        fig.tight_layout()
        return fig
    return ax


def plot_dual_axis_step(
    df,
    y1_col: str,
    y2_col: str,
    y1_label: str = "Axis 1",
    y2_label: str = "Axis 2",
    y1_color: str = "saddlebrown",
    y2_color: str = "darkorange",
    time_col: str = "time",
    title: str = "Dual Axis Step Plot",
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        own_ax = True
    else:
        own_ax = False

    if y1_col in df.columns:
        line1 = ax.step(
            df[time_col],
            df[y1_col],
            label=y1_label,
            color=y1_color,
            where="post",
            linewidth=2,
        )
        ax.set_ylabel(y1_label, color=y1_color, fontsize=12)
        ax.tick_params(axis="y", labelcolor=y1_color)
    else:
        line1 = []

    if y2_col in df.columns:
        ax_twin = ax.twinx()
        line2 = ax_twin.step(
            df[time_col],
            df[y2_col],
            label=y2_label,
            color=y2_color,
            where="post",
            linewidth=2,
        )
        ax_twin.set_ylabel(y2_label, color=y2_color, fontsize=12)
        ax_twin.tick_params(axis="y", labelcolor=y2_color)
    else:
        line2 = []

    lines = line1 + line2
    if lines:
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", bbox_to_anchor=(1.12, 1))

    ax.set_title(title, fontsize=14, pad=15)
    ax.grid(True)
    ax.set_xlabel("Simulation Time", fontsize=12)

    if own_ax:
        fig = ax.figure
        fig.tight_layout()
        return fig
    return ax


def build_dashboard(df, plot_configs, title="Simulation Dashboard", figsize=(16, 20)):
    num_plots = len(plot_configs)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_plots, 1, figure=fig)

    axes = []
    time_ax = None
    for i, config in enumerate(plot_configs):
        func = config["func"]
        is_time_series = func.__name__ not in [

            "plot_mode_distribution",
            "plot_mode_dwell_times",
            "plot_normalized_deviation_violin",
            "plot_deficit_disparity",

            "plot_deficit_breakdown_pie",
            "plot_deficit_breakdown_bar",
            "plot_structural_vs_operational_by_mode",
        ]

        if is_time_series:
            ax = fig.add_subplot(gs[i, 0], sharex=time_ax)
            if time_ax is None:
                time_ax = ax
        else:
            ax = fig.add_subplot(gs[i, 0])

        axes.append(ax)

        func = config["func"]
        kwargs = config.get("kwargs", {})

        func(df, ax=ax, **kwargs)

        ax.tick_params(labelbottom=True)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def plot_mode_distribution(
    df,
    mode_col="current_mode",
    time_col="time",
    title="Mode Distribution (% Time)",
    ax=None,
    palette=None,
    verbose=True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        own_ax = True
    else:
        own_ax = False

    if mode_col not in df.columns or time_col not in df.columns:
        if own_ax:
            return fig
        return ax

    df_sorted = df.copy()
    df_sorted["dt"] = df_sorted[time_col].diff().shift(-1).fillna(0)

    df_sorted["mode_str"] = df_sorted[mode_col].apply(
        lambda x: getattr(x, "name", str(x))
    )

    durations = df_sorted.groupby("mode_str")["dt"].sum()
    total_time = durations.sum()

    if total_time > 0:
        percentages = (durations / total_time) * 100
    else:
        percentages = durations * 0

    percentages = percentages.sort_values(ascending=True)

    if verbose:
        print(f"\n--- {title} ---")
        for mode, pct in percentages.items():
            print(f"{mode}: {pct:.1f}%")
        print("-" * (8 + len(title)))

    import matplotlib

    cmap = matplotlib.colormaps["tab10"]
    palette = palette or {}

    colors = []
    for mode in percentages.index:
        mode_name = getattr(mode, "name", str(mode))
        mode_str = str(mode).split(".")[-1].upper()
        if mode_name in palette:
            colors.append(palette[mode_name])
        elif mode_str in palette:
            colors.append(palette[mode_str])
        else:
            idx = sum(ord(c) for c in str(mode)) % 10
            colors.append(cmap(idx))

    bars = ax.barh(
        percentages.index.astype(str), percentages.values, color=colors, alpha=0.8
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("% of Total Simulation Time", fontsize=12)
    ax.set_xlim(0, max(100, percentages.max() + 10))
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    if own_ax:
        fig.tight_layout()
        return fig
    return ax


def plot_mode_dwell_times(
    df,
    time_col="time",
    mode_col="current_mode",
    title="Mode Stability (Dwell Times)",
    ax=None,
    verbose=True,
):
    df = df.copy()
    df[mode_col] = df[mode_col].astype(str)

    blocks = (df[mode_col] != df[mode_col].shift(1)).cumsum().rename("block")

    df["dt"] = df[time_col].diff().shift(-1).fillna(0)

    durations = df.groupby([blocks, mode_col])["dt"].sum().reset_index()
    durations.columns = ["block", "mode", "duration"]

    durations = durations[durations["duration"] > 0.01]

    if verbose:
        print(f"\n--- {title} ---")
        dwell_summary = durations.groupby("mode")["duration"].agg(
            ["count", "mean", "median", "max"]
        )
        print(dwell_summary.round(2).to_string())
        print("-" * (8 + len(title)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        own_ax = True
    else:
        own_ax = False

    sns.boxplot(
        data=durations,
        x="duration",
        y="mode",
        ax=ax,
        palette="Set2",
        hue="mode",
        legend=False,
    )
    sns.stripplot(
        data=durations, x="duration", y="mode", color="black", alpha=0.4, size=4, ax=ax
    )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Duration Before Switch (Days)", fontsize=12)
    ax.set_ylabel("")

    ax.axvline(
        x=2.0,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Chattering Threshold (<2 days)",
    )
    ax.legend(loc="lower right")

    if own_ax:
        fig = ax.figure
        fig.tight_layout()
        return fig
    return ax

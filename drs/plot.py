import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# Visual styling constants
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_STYLE = "seaborn-v0_8-whitegrid"
DEFAULT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def apply_plot_style() -> None:
    """
    Apply the default matplotlib stylesheet and seaborn configuration.
    
    Power User Note: Standardizes visual aesthetics across all plots.
    """
    plt.style.use(DEFAULT_STYLE)


def _setup_axes(ax=None, figsize=DEFAULT_FIGSIZE):
    """
    [INTERNAL] Helper to initialize or reuse a matplotlib axes.
    
    Power User Note: Reduces boilerplate for creating figure/axes objects.
    """
    apply_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


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
    """Plot multiple time-series curves on a single axis."""
    if time_col not in df.columns:
        raise ValueError(
            f"DataFrame must contain a '{time_col}' column for time-series plotting."
        )

    fig, ax, own_ax = _setup_axes(ax, DEFAULT_FIGSIZE)
    plot_colors = colors or DEFAULT_COLORS

    for i, col in enumerate(y_columns):
        if col in df.columns:
            color = plot_colors[i % len(plot_colors)]

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
    """Plot level safety margin relative to upper or lower constraints."""
    fig, ax, own_ax = _setup_axes(ax, (12, 6))

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
    """Plot two steps on dual y-axes for scale-disparate variables."""
    fig, ax, own_ax = _setup_axes(ax, DEFAULT_FIGSIZE)

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
        fig.tight_layout()
        return fig
    return ax


def build_dashboard(df, plot_configs, title="Simulation Dashboard", figsize=(16, 20)):
    """Assemble a multi-plot layout sharing x-axes where appropriate."""
    num_plots = len(plot_configs)
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(num_plots, 1, figure=fig)

    axes = []
    time_ax = None
    for i, config in enumerate(plot_configs):
        func = config["func"]
        is_time_series = config.get("is_time_series", func.__name__ not in [
            "plot_mode_distribution",
            "plot_mode_dwell_times",
            "plot_normalized_deviation_violin",
            "plot_deficit_disparity",
            "plot_deficit_breakdown_pie",
            "plot_deficit_breakdown_bar",
            "plot_structural_vs_operational_by_mode",
        ])

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

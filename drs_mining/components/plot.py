from typing import Optional, Tuple, List, Dict, Any, Union
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from drs.plot import (
    Dashboard,
    _get_ax,
    apply_plot_style,
    plot_dual_axis_step,
    plot_safety_margin,
    plot_time_series,
)


def plot_ore_with_modes(
    df,
    time_col="time",
    ore_cols=None,
    mode_col="active_operating_mode",
    title="Ore Stockpiles with Mode Switch Markers",
    campaign_split_mode=None,
    hlines=None,
    ax=None,
    palette=None,
):
    if ore_cols is None:
        ore_cols = ["total_system_ore_mass"]
    elif isinstance(ore_cols, str):
        ore_cols = [ore_cols]

    ax = _get_ax(ax, figsize=(14, 7))

    unique_modes = df[mode_col].unique()
    import matplotlib

    cmap = matplotlib.colormaps["tab10"]
    palette = palette or {}

    mode_colors = {}
    for i, mode in enumerate(unique_modes):
        mode_name = getattr(mode, "name", str(mode))
        mode_str = str(mode).split('.')[-1].upper()

        if mode_name in palette:
            mode_colors[mode] = palette[mode_name]
        elif mode_str in palette:
            mode_colors[mode] = palette[mode_str]
        else:
            mode_colors[mode] = cmap(i % 10)

    if campaign_split_mode is not None and campaign_split_mode in unique_modes:
        mode_colors[campaign_split_mode] = "#FFD700"

    change_idx = df.index[df[mode_col] != df[mode_col].shift(1)].tolist()

    for i, start_idx in enumerate(change_idx):
        mode = df.loc[start_idx, mode_col]
        t_start = df.loc[start_idx, time_col]

        if i + 1 < len(change_idx):
            t_end = df.loc[change_idx[i + 1], time_col]
        else:
            t_end = df[time_col].iloc[-1]

        alpha_val = (
            0.75
            if (campaign_split_mode is not None and mode == campaign_split_mode)
            else 0.10
        )
        ax.axvspan(t_start, t_end, alpha=alpha_val, color=mode_colors[mode])

        ore_line_colors = ["black", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
    plot_time_series(
        df,
        y_columns=ore_cols,
        time_col=time_col,
        ax=ax,
        add_legend=False,
        colors=ore_line_colors,
        alpha=0.9,
        zorder=3,
    )

    for start_idx in change_idx:
        if start_idx == df.index[0]:
            continue

        mode = df.loc[start_idx, mode_col]
        t = df.loc[start_idx, time_col]
        color = mode_colors[mode]

        ax.axvline(x=t, color=color, linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

    if campaign_split_mode is not None:
        campaign_starts = []
        in_campaign = False

        for start_idx in change_idx:
            mode = df.loc[start_idx, mode_col]
            t_start = df.loc[start_idx, time_col]

            if mode != campaign_split_mode and not in_campaign:
                in_campaign = True
                campaign_starts.append((t_start, start_idx))
            elif mode == campaign_split_mode and in_campaign:
                in_campaign = False

        primary_ore = (
            ore_cols[0] if len(ore_cols) > 0 and ore_cols[0] in df.columns else None
        )

        for i, (t_start, idx) in enumerate(campaign_starts):
            if primary_ore:
                y_val = df.loc[idx, primary_ore]
                ax.plot(
                    t_start, y_val, marker="X", color="black", markersize=9, zorder=5
                )
                ax.text(
                    t_start,
                    y_val + (ax.get_ylim()[1] * 0.03),
                    f"C{i+1}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                    zorder=6,
                )

    if hlines:
        for hline in hlines:
            ax.axhline(**hline)

    from matplotlib.patches import Patch

    mode_patches = [
        Patch(
            facecolor=mode_colors[m],
            alpha=0.75 if m == campaign_split_mode else 0.35,
            label=str(m),
        )
        for m in unique_modes
    ]
    ore_handles = ax.get_legend_handles_labels()[0]
    ore_labels = ax.get_legend_handles_labels()[1]

    all_handles = list(ore_handles) + mode_patches
    all_labels = list(ore_labels) + [str(m) for m in unique_modes]
    ax.legend(
        all_handles,
        all_labels,
        loc="upper right",
        bbox_to_anchor=(1, 1.12),
        ncol=min(len(all_labels), 5),
        frameon=True,
        fontsize=9,
    )
    ax.set_ylabel("Ore Stockpile", fontsize=12)
    ax.set_xlabel("Simulation Time", fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)

    return ax


def plot_normalized_deviation_violin(
    df,
    title="Stockpile Deviation Variance (Violin)",
    target_total=60000.0,
    target_ore1=42000.0,
    target_ore2=18000.0,
    col_total="total_system_ore_mass",
    col_ore1="Ore1Stock_mass",
    col_ore2="Ore2Stock_mass",
    ax=None
):
    ax = _get_ax(ax, figsize=(10, 6))

    if col_total not in df.columns and "OreStock" in df.columns:
        col_total = "OreStock"
    if col_ore1 not in df.columns and "Ore1Stock" in df.columns:
        col_ore1 = "Ore1Stock"
    if col_ore2 not in df.columns and "Ore2Stock" in df.columns:
        col_ore2 = "Ore2Stock"

    dev_total = ((df[col_total] - target_total) / target_total) * 100 if target_total else df[col_total] * 0
    dev_ore1 = ((df[col_ore1] - target_ore1) / target_ore1) * 100 if target_ore1 else df[col_ore1] * 0
    dev_ore2 = ((df[col_ore2] - target_ore2) / target_ore2) * 100 if target_ore2 else df[col_ore2] * 0

    dev_df = pd.DataFrame({
        "Total Stockpile": dev_total,
        "Ore 1": dev_ore1,
        "Ore 2": dev_ore2
    })
    melted_df = dev_df.melt(var_name="Stockpile Type", value_name="Deviation (%)")

    palette = {"Total Stockpile": "gray", "Ore 1": "#1f77b4", "Ore 2": "#d62728"}

    sns.violinplot(
        data=melted_df,
        y="Stockpile Type",
        x="Deviation (%)",
        hue="Stockpile Type",
        legend=False,
        palette=palette,
        inner="quartile",
        cut=0,
        ax=ax
    )

    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label="Perfect Target (0%)", zorder=0)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("Deviation from Target (%)", fontsize=12)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(loc="upper right")

    return ax


def _resolve_extraction_series(df: pd.DataFrame, extraction_col: Any) -> pd.Series:
    """Helper to safely extract step-by-step mined mass series from a dataframe."""
    if isinstance(extraction_col, (list, tuple)):
        valid = [c for c in extraction_col if c in df.columns]
        if valid:
            return df[valid].sum(axis=1).diff().shift(-1).fillna(0)
    elif extraction_col in df.columns:
        return df[extraction_col].diff().shift(-1).fillna(0)

    for candidate in ["total_mined", "total_extracted_ore", "cumulative_extracted_mass", "cumulative_milled_mass"]:
        if candidate in df.columns:
            return df[candidate].diff().shift(-1).fillna(0)

    candidates = [c for c in df.columns if "extracted_mass" in c or "mined" in c]
    if candidates:
        return df[candidates].sum(axis=1).diff().shift(-1).fillna(0)

    return pd.Series(0.0, index=df.index)


def plot_attributed_deficit(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass",
                            ideal_rate_per_day=6000.0, title="Cumulative Production Deficit by Mode", ax=None, palette=None):
    ax = _get_ax(ax, figsize=(12, 6))

    dt = df[time_col].diff().shift(-1).fillna(0) if time_col in df.columns else df.get("day", pd.Series(0, index=df.index)).diff().shift(-1).fillna(0)
    actual_extraction_step = _resolve_extraction_series(df, extraction_col)

    ideal_extraction_step = dt * ideal_rate_per_day
    step_deficit = ideal_extraction_step - actual_extraction_step

    step_deficit = step_deficit.clip(lower=0)

    mode_col_name = mode_col if mode_col in df.columns else ("active_operating_mode_name" if "active_operating_mode_name" in df.columns else "mill_mode")
    time_col_name = time_col if time_col in df.columns else "day"

    deficit_df = pd.DataFrame({
        'time': df[time_col_name],
        'mode': df[mode_col_name].astype(str),
        'deficit': step_deficit
    })

    pivot_df = deficit_df.pivot_table(index='time', columns='mode', values='deficit', aggfunc='sum').fillna(0)

    cumulative_pivot = pivot_df.cumsum()

    cols = list(cumulative_pivot.columns)
    shutdown_mode = next((c for c in cols if "SHUTDOWN" in str(c).upper()), None)
    if shutdown_mode and shutdown_mode in cols:
        cols.remove(shutdown_mode)
        cols = [shutdown_mode] + cols

    import matplotlib
    cmap = matplotlib.colormaps["tab10"]
    palette = palette or {}
    colors = []
    for idx, c in enumerate(cols):
        clean_c = str(c).split('.')[-1].upper()
        if clean_c in palette:
            colors.append(palette[clean_c])
        elif clean_c == "MODE_A_CONTINGENCY":
            colors.append("gold")
        elif clean_c == "MODE_B_CONTINGENCY":
            colors.append("cyan")
        elif clean_c == "SHUTDOWN":
            colors.append("gray")
        else:
            colors.append(cmap(idx % 10))

    cumulative_pivot[cols].plot.area(ax=ax, alpha=0.8, linewidth=0, color=colors)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Simulation Time (Days)", fontsize=12)
    ax.set_ylabel("Cumulative Lost Production (Tons)", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    total_lost = cumulative_pivot.iloc[-1].sum() if not cumulative_pivot.empty else 0
    ax.text(0.02, 0.95, f"Total Lost: {total_lost:,.0f} tons",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8))

    handles, labels = ax.get_legend_handles_labels()
    clean_labels = [str(l).split('.')[-1] for l in labels]
    ax.legend(handles, clean_labels, loc='upper left')

    return ax


def plot_deficit_disparity(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass", ideal_rate=6000.0, title="Mode Efficiency (Time Spent vs. Deficit Caused)", ax=None, verbose=True):
    ax = _get_ax(ax, figsize=(10, 6))

    df = df.copy()
    mode_col_name = mode_col if mode_col in df.columns else ("active_operating_mode_name" if "active_operating_mode_name" in df.columns else "mill_mode")
    time_col_name = time_col if time_col in df.columns else "day"
    df[mode_col_name] = df[mode_col_name].astype(str)
    df['dt'] = df[time_col_name].diff().shift(-1).fillna(0)
    df['dx'] = _resolve_extraction_series(df, extraction_col)

    df['ideal_dx'] = df['dt'] * ideal_rate
    df['deficit'] = (df['ideal_dx'] - df['dx']).clip(lower=0)

    summary = df.groupby(mode_col_name).agg({'dt': 'sum', 'deficit': 'sum'})

    summary['% of Total Time'] = (summary['dt'] / summary['dt'].sum()) * 100
    summary['% of Total Deficit'] = (summary['deficit'] / summary['deficit'].sum()) * 100

    if verbose:
        print(f"\n--- {title} ---")
        print(summary[['% of Total Time', '% of Total Deficit']].round(1).to_string())
        print("-" * (8 + len(title)))

    melted = summary[['% of Total Time', '% of Total Deficit']].reset_index().melt(
        id_vars=mode_col_name, var_name="Metric", value_name="Percentage"
    )

    order = summary.sort_values('% of Total Deficit', ascending=False).index

    sns.barplot(data=melted, y=mode_col_name, x="Percentage", hue="Metric", order=order, palette=["#1f77b4", "#d62728"], ax=ax)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    return ax


def plot_deficit_breakdown_bar(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass", ideal_rate_per_day=6000.0, title="Final Deficit Breakdown by Mode (%)", ax=None, palette=None, verbose=True):
    ax = _get_ax(ax, figsize=(10, 6))

    df = df.copy()
    mode_col_name = mode_col if mode_col in df.columns else ("active_operating_mode_name" if "active_operating_mode_name" in df.columns else "mill_mode")
    time_col_name = time_col if time_col in df.columns else "day"
    df['dt'] = df[time_col_name].diff().shift(-1).fillna(0)
    df['dx'] = _resolve_extraction_series(df, extraction_col)
    df['deficit'] = ((df['dt'] * ideal_rate_per_day) - df['dx']).clip(lower=0)
    df['mode_str'] = df[mode_col_name].astype(str).apply(lambda x: x.split('.')[-1])

    summary = df.groupby('mode_str')['deficit'].sum()
    summary = summary[summary > 0].sort_values(ascending=True)
    total_deficit = summary.sum()

    if total_deficit > 0:
        summary_pct = (summary / total_deficit) * 100
    else:
        summary_pct = summary

    if verbose:
        print(f"\n--- {title} ---")
        for mode in summary.index[::-1]:
            print(f"{mode}: {summary[mode]:,.1f} t ({summary_pct[mode]:.1f}%)")
        print(f"TOTAL LOST: {total_deficit:,.1f} t")
        print("-" * (8 + len(title)))

    palette = palette or {}
    colors = [palette.get(m.upper(), "gray") for m in summary.index]

    bars = ax.barh(summary.index, summary_pct.values, color=colors, edgecolor='black', alpha=0.8)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("% of Total Lost Tonnage", fontsize=12)
    ax.set_xlim(0, max(summary_pct.max() * 1.15, 100))

    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=11, fontweight='bold')

    ax.text(0.95, 0.05, f"Total Lost: {total_deficit:,.0f} t",
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

    return ax


def plot_structural_vs_operational_deficit(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass", ideal_rate=6000.0, structural_modes=None, ax=None, verbose=True):
    ax = _get_ax(ax, figsize=(10, 6))

    df = df.copy()
    mode_col_name = mode_col if mode_col in df.columns else ("active_operating_mode_name" if "active_operating_mode_name" in df.columns else "mill_mode")
    time_col_name = time_col if time_col in df.columns else "day"
    df['dt'] = df[time_col_name].diff().shift(-1).fillna(0)
    df['dx'] = _resolve_extraction_series(df, extraction_col)
    df['deficit'] = ((df['dt'] * ideal_rate) - df['dx']).clip(lower=0)
    df['mode_str'] = df[mode_col_name].astype(str)

    structural_modes = structural_modes or []

    def classify_bucket(mode):
        if any(sm in mode for sm in structural_modes) and "CONTINGENCY" not in mode and "SURGING" not in mode:
            return "Structural (Unavoidable: Geology & Shutdowns)"
        else:
            return "Operational (Avoidable: Control Logic & Contingencies)"

    df['Deficit_Type'] = df['mode_str'].apply(classify_bucket)

    pivot = df.pivot_table(index=time_col_name, columns='Deficit_Type', values='deficit', aggfunc='sum').fillna(0)
    cumsum_pivot = pivot.cumsum()

    if verbose:
        title_str = "Structural vs. Operational Deficit"
        print(f"\n--- {title_str} ---")
        final_totals = cumsum_pivot.iloc[-1] if not cumsum_pivot.empty else {}
        for deficit_type, val in final_totals.items():
            print(f"{deficit_type}: {val:,.1f} t")
        print("-" * (8 + len(title_str)))

    cols = sorted(list(cumsum_pivot.columns), reverse=True)
    cumsum_pivot[cols].plot.area(ax=ax, color=["gray", "firebrick"], alpha=0.7, linewidth=0)

    ax.set_title("Structural vs. Operational Deficit", fontsize=14, pad=15)
    ax.set_xlabel("Simulation Time (Days)", fontsize=12)
    ax.set_ylabel("Cumulative Lost Tonnage", fontsize=12)
    ax.legend(loc='upper left')

    ax.text(0.5, 0.85, "RL Optimization Target:\nSquash the Red Layer to Zero",
            transform=ax.transAxes, fontsize=12, color="firebrick", fontweight="bold",
            ha="center", bbox=dict(facecolor='white', alpha=0.8, edgecolor='firebrick'))

    return ax


def plot_normalized_cumulative_deficit(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass", ideal_rate_per_day=6000.0, title="Deficit Composition Over Time (100% Stacked)", ax=None, palette=None):
    ax = _get_ax(ax, figsize=(12, 6))

    df = df.copy()
    mode_col_name = mode_col if mode_col in df.columns else ("active_operating_mode_name" if "active_operating_mode_name" in df.columns else "mill_mode")
    time_col_name = time_col if time_col in df.columns else "day"
    df['dt'] = df[time_col_name].diff().shift(-1).fillna(0)
    df['dx'] = _resolve_extraction_series(df, extraction_col)
    df['deficit'] = ((df['dt'] * ideal_rate_per_day) - df['dx']).clip(lower=0)
    df['mode_str'] = df[mode_col_name].astype(str).apply(lambda x: x.split('.')[-1])

    pivot_df = df.pivot_table(index=time_col_name, columns='mode_str', values='deficit', aggfunc='sum').fillna(0)
    cumulative_pivot = pivot_df.cumsum()

    row_sums = cumulative_pivot.sum(axis=1)
    normalized_pivot = cumulative_pivot.div(row_sums.replace(0, 1), axis=0) * 100

    cols = list(normalized_pivot.columns)
    if "SHUTDOWN" in cols:
        cols.remove("SHUTDOWN")
        cols = ["SHUTDOWN"] + cols

    palette = palette or {}
    colors = [palette.get(c.upper(), "gray") for c in cols]

    normalized_pivot[cols].plot.area(ax=ax, alpha=0.8, linewidth=0, color=colors)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Simulation Time (Days)", fontsize=12)
    ax.set_ylabel("% of Total Cumulative Deficit", fontsize=12)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

    return ax


def plot_structural_vs_operational_by_mode(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass", ideal_rate=6000.0, title="Structural vs. Operational Deficit by Base Mode", structural_modes=None, base_mode_mapper=None, ax=None, verbose=True):
    ax = _get_ax(ax, figsize=(10, 6))

    df = df.copy()
    mode_col_name = mode_col if mode_col in df.columns else ("active_operating_mode_name" if "active_operating_mode_name" in df.columns else "mill_mode")
    time_col_name = time_col if time_col in df.columns else "day"
    df['dt'] = df[time_col_name].diff().shift(-1).fillna(0)
    df['dx'] = _resolve_extraction_series(df, extraction_col)
    df['deficit'] = ((df['dt'] * ideal_rate) - df['dx']).clip(lower=0)
    df['mode_str'] = df[mode_col_name].astype(str).apply(lambda x: x.split('.')[-1])

    structural_modes = structural_modes or []

    def get_base_mode(m):
        if base_mode_mapper:
            return base_mode_mapper(m)
        return m.split('_CONTINGENCY')[0].split('_MINE')[0]

    def get_deficit_type(m):
        if any(sm in m for sm in structural_modes) and "CONTINGENCY" not in m and "SURGING" not in m:
            return "Structural (Unavoidable)"
        return "Operational (Avoidable)"

    df['Base_Mode'] = df['mode_str'].apply(get_base_mode)
    df['Deficit_Type'] = df['mode_str'].apply(get_deficit_type)

    summary = df.groupby(['Base_Mode', 'Deficit_Type'])['deficit'].sum().unstack(fill_value=0)

    for col in ["Structural (Unavoidable)", "Operational (Avoidable)"]:
        if col not in summary.columns:
            summary[col] = 0

    if verbose:
        print(f"\n--- {title} ---")
        print(summary.round(1).to_string())
        print("-" * (8 + len(title)))

    order = sorted(df['Base_Mode'].unique())
    summary = summary.reindex(order).fillna(0)

    col_order = ["Operational (Avoidable)", "Structural (Unavoidable)"]
    summary = summary[[c for c in col_order if c in summary.columns]]

    summary.plot(kind='bar', stacked=True, color=["firebrick", "gray"], ax=ax, alpha=0.85, edgecolor='black')

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Total Lost Tonnage", fontsize=12)
    ax.tick_params(axis='x', rotation=0, labelsize=11)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    ax.legend(title="Deficit Classification", loc="upper left")

    return ax


def plot_mode_distribution(
    df,
    mode_col="current_mode",
    time_col="time",
    title="Mode Distribution (% Time)",
    ax=None,
    palette=None,
    verbose=True,
):
    ax = _get_ax(ax, figsize=(10, 4))

    if mode_col not in df.columns or time_col not in df.columns:
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

    ax = _get_ax(ax, figsize=(10, 6))

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

    return ax


def plot_truck_idle_and_utilization(
    df,
    time_col: str = "time",
    idle_col: str = "trucks_idle",
    operating_col: str = "trucks_operating",
    area1_col: str = "trucks_area1_operating",
    area2_col: str = "trucks_area2_operating",
    refueling_col: str = "trucks_refueling",
    dev_col: Optional[str] = "trucks_dev_reserved",
    total_trucks: Optional[int] = None,
    title: str = "Haul Fleet Utilization & Idle Time Breakdown",
    ax=None,
    verbose: bool = False,
):
    """Plots stacked fleet status over time (Area 1 Production vs Area 2 Production vs Development vs Refueling vs Idle)."""
    ax = _get_ax(ax, figsize=(14, 5))

    t = df[time_col]
    op = df[operating_col] if operating_col in df.columns else pd.Series(0, index=df.index)
    op1 = df[area1_col] if area1_col in df.columns else op
    op2 = df[area2_col] if area2_col in df.columns else pd.Series(0, index=df.index)
    refuel = df[refueling_col] if refueling_col in df.columns else pd.Series(0, index=df.index)
    dev = df[dev_col] if (dev_col and dev_col in df.columns) else pd.Series(0, index=df.index)
    idle = df[idle_col] if idle_col in df.columns else pd.Series(0, index=df.index)

    if total_trucks is None:
        total_trucks = int((op + dev + refuel + idle).max()) if not df.empty else 18

    # Stacked layers:
    # Layer 1: Area 1 Production Haulage
    ax.fill_between(t, 0, op1, label="Area 1 Production Haulage", color="#1976D2", alpha=0.75, step="post")
    # Layer 2: Area 2 Production Haulage
    ax.fill_between(t, op1, op1 + op2, label="Area 2 Production Haulage", color="#388E3C", alpha=0.75, step="post")
    # Layer 3: Mine Development Priority (Reserved Fleet)
    y_dev = op1 + op2
    ax.fill_between(t, y_dev, y_dev + dev, label="Mine Development Priority Fleet", color="#7B1FA2", alpha=0.65, step="post")
    # Layer 4: Refueling & Service
    y_refuel = y_dev + dev
    ax.fill_between(t, y_refuel, y_refuel + refuel, label="Refueling / Service", color="#F57C00", alpha=0.70, step="post")
    # Layer 5: Idle & Standby (Buffer Pacing)
    y_top = y_refuel + refuel
    ax.fill_between(t, y_top, y_top + idle, label="Idle (Buffer Pacing / Standby)", color="#78909C", alpha=0.45, step="post")

    ax.plot(t, op + dev + refuel + idle, label=f"Total Fleet ({total_trucks} Trucks)", color="#212121", linewidth=1.5, linestyle=":")

    mean_idle = idle.mean()
    mean_op = op.mean()
    util_pct = (mean_op / max(1, total_trucks)) * 100.0

    ax.set_title(f"{title} (Mean Idle: {mean_idle:.1f} trucks, Avg Production Utilization: {util_pct:.1f}%)", fontsize=12, pad=10)
    ax.set_ylabel("Truck Count")
    ax.set_xlabel("Simulation Time (Days)")
    ax.set_ylim(0, total_trucks * 1.15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.90)

    if verbose:
        print(f"[{title}] Mean Operating: {mean_op:.2f} trucks | Mean Idle: {mean_idle:.2f} trucks | Avg Utilization: {util_pct:.1f}%")

    return ax


# ============================================================
# Shared diagnostics palette / assumptions
# ============================================================

MODE_PALETTE = {
    "MODE_A": "#1f77b4",
    "MODE_A_CONTINGENCY": "#2ca02c",
    "MODE_A_MINE_SURGING": "#9467bd",
    "MODE_B": "#d62728",
    "MODE_B_CONTINGENCY": "#ff7f0e",
    "MODE_B_MINE_SURGING": "#8c564b",
    "SHUTDOWN": "#FFD700",
}

STRUCTURAL_MODES = ["SHUTDOWN", "MODE_A"]


# ============================================================
# Post-processing / diagnostics helpers (extracted from the
# single-face and multi-face example scripts)
# ============================================================


def prepare_history(df):
    """Add the derived mode/stockpile columns shared by the diagnostics helpers."""
    df = df.copy()
    if "active_operating_mode" in df.columns:
        df["active_operating_mode_name"] = df["active_operating_mode"].apply(
            lambda x: x.name if hasattr(x, "name") else (str(x) if x else "None")
        )
    elif "active_operating_mode_name" not in df.columns:
        df["active_operating_mode_name"] = "None"

    df["prev_mode_name"] = df["active_operating_mode_name"].shift(1)

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

    if "total_system_ore_mass" not in df.columns:
        if "Ore1Stock_mass" in df.columns and "Ore2Stock_mass" in df.columns:
            df["total_system_ore_mass"] = df["Ore1Stock_mass"] + df["Ore2Stock_mass"]
        else:
            df["total_system_ore_mass"] = 0.0

    df["Total Ore Stockpile Level"] = df["total_system_ore_mass"] / 1000.0
    df["Ore 1 Stockpile Level"] = df["Ore1Stock_mass"] / 1000.0 if "Ore1Stock_mass" in df.columns else 0.0
    df["Ore 2 Stockpile Level"] = df["Ore2Stock_mass"] / 1000.0 if "Ore2Stock_mass" in df.columns else 0.0

    # Campaign-level cash flow and daily variation ranges
    cf_col = (
        "current_discounted_cash_flow_rate"
        if "current_discounted_cash_flow_rate" in df.columns
        else ("current_cash_flow_rate" if "current_cash_flow_rate" in df.columns else None)
    )
    if cf_col is not None and "active_operating_mode_name" in df.columns:
        campaign_id = (
            df["active_operating_mode_name"] != df["active_operating_mode_name"].shift(1)
        ).cumsum()
        df["campaign_id"] = campaign_id
        grouped = df.groupby("campaign_id")[cf_col]
        df["campaign_cash_flow_rate"] = grouped.transform("mean")
        df["daily_cash_flow_min"] = grouped.transform("min")
        df["daily_cash_flow_max"] = grouped.transform("max")

    # Fleet Area 1 / Area 2 breakdown fallback
    if "trucks_area1_operating" not in df.columns or df["trucks_area1_operating"].isna().all():
        if "trucks_operating_face1" in df.columns:
            df["trucks_area1_operating"] = df["trucks_operating_face1"]
            df["trucks_area2_operating"] = df.get("trucks_operating_face2", 0.0)
        elif "trucks_operating" in df.columns:
            if "analytical_face1_weight" in df.columns:
                w1 = df["analytical_face1_weight"].clip(0.0, 1.0)
                df["trucks_area1_operating"] = df["trucks_operating"] * w1
                df["trucks_area2_operating"] = df["trucks_operating"] * (1.0 - w1)
            else:
                a2_ready = df.get("area2_ready", False)
                mode_a = df.get("Mode A", 0.0) == 1.0
                w2 = np.where(a2_ready, np.where(mode_a, 0.65, 0.35), 0.0)
                df["trucks_area2_operating"] = df["trucks_operating"] * w2
                df["trucks_area1_operating"] = df["trucks_operating"] - df["trucks_area2_operating"]

    return df



def print_state_change_transitions(events, variable="active_operating_mode"):
    """Print mode transitions from engine ``STATE_CHANGE`` events."""
    state_change_events = [
        e
        for e in events
        if e.event_type == "STATE_CHANGE" and e.details.get("variable") == variable
    ]
    if not state_change_events:
        return
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


def print_transition_log(
    df,
    critical_ore2_level=20400.0,
    target_ore_stock_level=60000.0,
    label="",
):
    """Print mode transitions derived from the history DataFrame."""
    df = df.copy()
    if "prev_mode_name" not in df.columns:
        df["prev_mode_name"] = df["active_operating_mode_name"].shift(1)
    print(f"\n--- Mode Transition Log ({label}) ---")
    print(df["active_operating_mode_name"].unique()[:5])
    transitions = df[
        (df["active_operating_mode_name"] != df["prev_mode_name"])
        & df["prev_mode_name"].notna()
    ]

    for idx, row in transitions.iterrows():
        print(
            f"Time: {row['time']:.2f} | Transition: {row['prev_mode_name']} -> {row['active_operating_mode_name']}"
        )
        print(
            f"  \u21b3 Ore1 Stock: {row['Ore1Stock_mass']:.1f} | Ore2 Stock: {row['Ore2Stock_mass']:.1f} (Critical: {critical_ore2_level}) | Total Stock: {row['total_system_ore_mass']:.1f} (Target: {target_ore_stock_level})"
        )
        camp_dur = row.get("current_campaign_duration", 0.0)
        cont_dur = row.get("current_contingency_duration", 0.0)
        print(
            f"  \u21b3 Campaign/Shutdown Timer: {camp_dur:.2f} | Contingency Timer: {cont_dur:.2f}"
        )
    print("---------------------------\n")


def print_deficit_by_mode(
    df,
    extraction_cols,
    ideal_rate=6000.0,
    heading="Cumulative Lost Production (Deficit) by Mode",
):
    """Print a per-mode cumulative production deficit table.

    ``extraction_cols`` names the cumulative extraction columns to sum; the
    single-face history uses ``["cumulative_extracted_mass"]`` while the
    multi-face history uses ``["face1_extracted_mass", "face2_extracted_mass"]``.
    """
    dt = df["time"].diff().fillna(0) if "time" in df.columns else df.get("day", pd.Series(0, index=df.index)).diff().fillna(0)
    valid_cols = [c for c in extraction_cols if c in df.columns]
    if not valid_cols:
        for candidate in ["total_mined", "total_extracted_ore", "cumulative_extracted_mass", "cumulative_milled_mass"]:
            if candidate in df.columns:
                valid_cols = [candidate]
                break
        if not valid_cols:
            valid_cols = [c for c in df.columns if "extracted_mass" in c or "mined" in c]

    if valid_cols:
        actual_extraction_step = df[valid_cols].sum(axis=1).diff().fillna(0)
    else:
        actual_extraction_step = pd.Series(0.0, index=df.index)

    ideal_extraction_step = dt * ideal_rate
    step_deficit = (ideal_extraction_step - actual_extraction_step).clip(lower=0)

    mode_col = "active_operating_mode_name" if "active_operating_mode_name" in df.columns else "mill_mode"
    deficit_df = pd.DataFrame(
        {"mode": df[mode_col], "deficit": step_deficit}
    )

    total_deficit_by_mode = (
        deficit_df.groupby("mode")["deficit"].sum().sort_values(ascending=False)
    )

    print(f"\n--- {heading} ---")
    total_lost = total_deficit_by_mode.sum()
    for mode, lost in total_deficit_by_mode.items():
        mode_name = str(mode).split(".")[-1]
        pct = (lost / total_lost * 100) if total_lost > 0 else 0
        print(f"{mode_name}: {lost:.1f} tons ({pct:.1f}%)")
    print(f"TOTAL: {total_lost:.1f} tons")
    print("----------------------------------------------------\n")


def plot_single_face_dashboard(
    df,
    save_path="plots/Comprehensive_Diagnostics_Plot.png",
    figsize=(18, 69),
    title="Comprehensive Mine Diagnostics",
    palette=None,
):
    """Build and save the 14-panel single-face diagnostics dashboard."""
    palette = palette or MODE_PALETTE

    dash = Dashboard(
        nrows=14, ncols=1, figsize=figsize, sharex=False, title=title
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 8, 11, 12])

    plot_time_series(
        df,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Modes (Step)",
        is_step=True,
        ax=dash[0],
    )
    plot_ore_with_modes(
        df,
        time_col="time",
        ore_cols=[
            "total_system_ore_mass",
            "Ore1Stock_mass",
            "Ore2Stock_mass",
        ],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Ore Stockpiles & Campaigns",
        palette=palette,
        hlines=[
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
        ax=dash[1],
    )
    plot_dual_axis_step(
        df,
        y1_col="MassOfCurrentParcel",
        y2_col="CurrentParcelRoutingFraction",
        y1_label="Parcel Mass (tons)",
        y2_label="Grade (% Ore 2)",
        title="Current Parcel Properties",
        ax=dash[2],
    )
    plot_safety_margin(
        df,
        level_col="Ore1Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 1 Distance to Floor",
        danger_threshold=1000.0,
        ax=dash[3],
    )
    plot_safety_margin(
        df,
        level_col="Ore2Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 2 Distance to Floor",
        danger_threshold=1000.0,
        ax=dash[4],
    )
    plot_mode_distribution(
        df,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% of Time Spent)",
        palette=palette,
        ax=dash[5],
    )
    plot_mode_dwell_times(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        title="Mode Stability (Dwell Times)",
        ax=dash[6],
    )
    plot_normalized_deviation_violin(
        df,
        title="Stockpile Deviation Variance (Violin)",
        target_total=60000.0,
        target_ore1=42000.0,
        target_ore2=18000.0,
        ax=dash[7],
    )
    plot_attributed_deficit(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        extraction_col="cumulative_extracted_mass",
        ideal_rate_per_day=6000.0,
        title="Cumulative Production Deficit by Mode",
        palette=palette,
        ax=dash[8],
    )
    plot_deficit_disparity(
        df,
        mode_col="active_operating_mode_name",
        title="Mode Efficiency (Time Spent vs. Deficit Caused)",
        ideal_rate=6000.0,
        ax=dash[9],
    )
    plot_deficit_breakdown_bar(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[10],
    )
    plot_structural_vs_operational_deficit(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate=6000.0,
        structural_modes=STRUCTURAL_MODES,
        ax=dash[11],
    )
    plot_normalized_cumulative_deficit(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[12],
    )
    plot_structural_vs_operational_by_mode(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate=6000.0,
        structural_modes=STRUCTURAL_MODES,
        ax=dash[13],
    )

    dash.save(save_path)
    return dash


def plot_multi_face_dashboard(
    df,
    name="Dynamic Fleet Allocation",
    save_dir="plots",
    figsize=(18, 69),
    palette=None,
):
    """Build and save the 23-panel multi-face diagnostics dashboard."""
    palette = palette or MODE_PALETTE

    dash = Dashboard(
        nrows=23,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title=f"Comprehensive Mine Diagnostics ({name})",
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 20, 21])

    plot_time_series(
        df,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Modes (Step)",
        is_step=True,
        ax=dash[0],
    )
    plot_ore_with_modes(
        df,
        time_col="time",
        ore_cols=[
            "total_system_ore_mass",
            "Ore1Stock_mass",
            "Ore2Stock_mass",
        ],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Ore Stockpiles & Campaigns",
        palette=palette,
        hlines=[
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
        ax=dash[1],
    )
    plot_dual_axis_step(
        df,
        y1_col="face1_parcel_mass",
        y2_col="face1_parcel_ratio",
        y1_label="Face 1 Parcel Mass (tons)",
        y2_label="Face 1 Ore 1 Fraction",
        title="Face 1 Current Parcel Properties",
        y1_color="saddlebrown",
        y2_color="darkorange",
        ax=dash[2],
    )
    plot_dual_axis_step(
        df,
        y1_col="face2_parcel_mass",
        y2_col="face2_parcel_ratio",
        y1_label="Face 2 Parcel Mass (tons)",
        y2_label="Face 2 Ore 1 Fraction",
        title="Face 2 Current Parcel Properties",
        y1_color="saddlebrown",
        y2_color="darkorange",
        ax=dash[3],
    )
    plot_dual_axis_step(
        df,
        y1_col="mixed_achieved_extraction_rate",
        y2_col="mixed_ore1_fraction",
        y1_label="Combined Extraction Rate (t/d)",
        y2_label="Mixed Ore 1 Fraction",
        title="Combined Mine Output Properties",
        y1_color="saddlebrown",
        y2_color="darkorange",
        ax=dash[4],
    )
    plot_time_series(
        df,
        y_columns=[
            "mixed_target_extraction_rate",
            "mixed_real_extraction_rate",
            "mixed_achieved_extraction_rate",
        ],
        title="Fleet-Constrained Extraction Rates",
        is_step=True,
        ax=dash[5],
    )
    plot_time_series(
        df,
        y_columns=["face1_alloc", "face2_alloc", "ore2_ratio"],
        title="Active Fleet Allocation & Stockpile Ratio",
        is_step=True,
        ax=dash[6],
    )
    plot_time_series(
        df,
        y_columns=["face1_real_capacity", "face1_target_rate"],
        title="Face 1 Real Capacity vs Target Rate (Headroom)",
        is_step=True,
        ax=dash[7],
    )
    plot_time_series(
        df,
        y_columns=["face2_real_capacity", "face2_target_rate"],
        title="Face 2 Real Capacity vs Target Rate (Headroom)",
        is_step=True,
        ax=dash[8],
    )
    plot_time_series(
        df,
        y_columns=["face1_match_factor", "face2_match_factor"],
        title="Match Factor per Face (1.0 = balanced)",
        is_step=True,
        ax=dash[9],
    )
    plot_time_series(
        df,
        y_columns=["total_unused_trucks"],
        title="Total Unused Trucks (Spare Fleet Capacity)",
        is_step=True,
        ax=dash[10],
    )
    plot_time_series(
        df,
        y_columns=[
            "face1_truck_cycle_time_hours",
            "face2_truck_cycle_time_hours",
        ],
        title="Truck Cycle Times (Hours) & Traffic Delays",
        is_step=True,
        ax=dash[11],
    )
    plot_safety_margin(
        df,
        level_col="Ore1Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 1 Distance to Floor",
        danger_threshold=1000.0,
        ax=dash[12],
    )
    plot_safety_margin(
        df,
        level_col="Ore2Stock_mass",
        constraint_value=0.0,
        constraint_type="lower",
        title="Safety Margin: Ore 2 Distance to Floor",
        danger_threshold=1000.0,
        ax=dash[13],
    )
    plot_mode_distribution(
        df,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% of Time Spent)",
        palette=palette,
        ax=dash[14],
    )
    plot_mode_dwell_times(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        title="Mode Stability (Dwell Times)",
        ax=dash[15],
    )
    plot_normalized_deviation_violin(
        df,
        title="Stockpile Deviation Variance (Violin)",
        target_total=60000.0,
        target_ore1=42000.0,
        target_ore2=18000.0,
        ax=dash[16],
    )
    plot_attributed_deficit(
        df,
        time_col="time",
        mode_col="active_operating_mode_name",
        extraction_col="cumulative_extracted_mass",
        ideal_rate_per_day=6000.0,
        title="Cumulative Production Deficit by Mode",
        palette=palette,
        ax=dash[17],
    )
    plot_deficit_disparity(
        df,
        mode_col="active_operating_mode_name",
        title="Mode Efficiency (Time Spent vs. Deficit Caused)",
        ideal_rate=6000.0,
        ax=dash[18],
    )
    plot_deficit_breakdown_bar(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[19],
    )
    plot_structural_vs_operational_deficit(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate=6000.0,
        structural_modes=STRUCTURAL_MODES,
        ax=dash[20],
    )
    plot_normalized_cumulative_deficit(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate_per_day=6000.0,
        palette=palette,
        ax=dash[21],
    )
    plot_structural_vs_operational_by_mode(
        df,
        mode_col="active_operating_mode_name",
        ideal_rate=6000.0,
        structural_modes=STRUCTURAL_MODES,
        ax=dash[22],
    )

    prefix = name.lower().replace(" ", "_")
    dash.save(f"{save_dir}/Comprehensive_Diagnostics_Plot_{prefix}.png")
    plt.close(dash.fig)
    return df


def print_strategic_economic_summary(
    with_df: pd.DataFrame, without_df: Optional[pd.DataFrame] = None
):
    """Prints comprehensive whole-mine economic results and incremental NPV."""
    final_with = with_df.iloc[-1]
    npv_with = float(final_with.get("cumulative_npv", final_with.get("operating_npv_proxy", 0.0)))
    cf_with = float(final_with.get("cumulative_cash_flow", 0.0))
    milled_with = float(final_with.get("total_processed", final_with.get("cumulative_milled_mass", 0.0)))
    dev_with = float(final_with.get("cumulative_development", final_with.get("cumulative_mine_development", 0.0)))

    print("\n" + "=" * 80)
    print(" STRATEGIC ECONOMICS & INCREMENTAL NPV SUMMARY")
    print("=" * 80)
    print(f"WITH Area 2 (Base Case):")
    print(f"  Total Ore Milled:            {milled_with:,.1f} t")
    print(f"  Total Mine Development:      {dev_with:,.1f} metres")
    print(f"  Cumulative Undiscounted CF:  ${cf_with:,.2f}")
    print(f"  Operating Net Present Value: ${npv_with:,.2f}")

    if without_df is not None and not without_df.empty:
        final_without = without_df.iloc[-1]
        npv_without = float(final_without.get("cumulative_npv", final_without.get("operating_npv_proxy", 0.0)))
        cf_without = float(final_without.get("cumulative_cash_flow", 0.0))
        milled_without = float(final_without.get("total_processed", final_without.get("cumulative_milled_mass", 0.0)))
        dev_without = float(final_without.get("cumulative_development", final_without.get("cumulative_mine_development", 0.0)))
        incremental_npv = npv_with - npv_without

        print(f"\nWITHOUT Area 2 (Counterfactual):")
        print(f"  Total Ore Milled:            {milled_without:,.1f} t")
        print(f"  Total Mine Development:      {dev_without:,.1f} metres")
        print(f"  Cumulative Undiscounted CF:  ${cf_without:,.2f}")
        print(f"  Operating Net Present Value: ${npv_without:,.2f}")

        print(f"\n--------------------------------------------------------------------------------")
        print(f" >>> TRUE INCREMENTAL NPV OF AREA 2 CAPITAL PROJECT: ${incremental_npv:,.2f} <<<")
        print(f"--------------------------------------------------------------------------------")
    print("=" * 80 + "\n")


def plot_two_area_dashboard(
    df: pd.DataFrame,
    output_path: str = "plots/two_area_dashboard.png",
    title: str = "Two-Area Strategic Planning & Operations Dashboard",
    palette: dict = None,
    figsize: Tuple[int, int] = (16, 40),
):
    """Builds and saves the standardized multi-panel two-area diagnostics dashboard."""
    import os
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(6, 1, figsize=figsize, sharex=True)
    time_col = "day" if "day" in df.columns else "time"

    unlock_rows = df[df.get("area2_ready", False) == True]
    unlock_time = float(unlock_rows[time_col].iloc[0]) if not unlock_rows.empty else None

    deplete_rows = df[df.get("area1_exhausted", False) == True]
    deplete_time = float(deplete_rows[time_col].iloc[0]) if not deplete_rows.empty else None
    if deplete_time is None and "area1_depleted_day" in df.columns:
        valid_dep = df[df["area1_depleted_day"] >= 0.0]
        if not valid_dep.empty:
            deplete_time = float(valid_dep["area1_depleted_day"].iloc[0])

    deplete2_rows = df[df.get("area2_exhausted", False) == True]
    deplete2_time = float(deplete2_rows[time_col].iloc[0]) if not deplete2_rows.empty else None
    if deplete2_time is None and "area2_depleted_day" in df.columns:
        valid_dep2 = df[df["area2_depleted_day"] >= 0.0]
        if not valid_dep2.empty:
            deplete2_time = float(valid_dep2["area2_depleted_day"].iloc[0])

    # 1. Stockpiles
    ax = axes[0]
    if "ore1_stockpile" in df.columns:
        ax.plot(df[time_col], df["ore1_stockpile"], label="Ore 1 Stockpile (t)", color="#1976D2")
        ax.plot(df[time_col], df["ore2_stockpile"], label="Ore 2 Stockpile (t)", color="#D32F2F")
        ax.plot(df[time_col], df["total_stockpile"], label="Total Stockpile (t)", color="#388E3C", linestyle="--")
    elif "Ore1Stock_mass" in df.columns:
        ax.plot(df[time_col], df["Ore1Stock_mass"], label="Ore 1 Stockpile (t)", color="#1976D2")
        ax.plot(df[time_col], df["Ore2Stock_mass"], label="Ore 2 Stockpile (t)", color="#D32F2F")
        ax.plot(df[time_col], df["total_system_ore_mass"], label="Total Stockpile (t)", color="#388E3C", linestyle="--")
    ax.axhline(60000.0, color="black", linestyle=":", label="Buffer Target (60kt)")
    if unlock_time is not None:
        ax.axvline(unlock_time, color="#2E7D32", linestyle="-.", linewidth=1.8, label=f"Area 2 Unlocked (Day {unlock_time:.1f})")
    if deplete_time is not None:
        ax.axvline(deplete_time, color="#D32F2F", linestyle="--", linewidth=1.8, label=f"Area 1 Depleted (Day {deplete_time:.1f})")
    if deplete2_time is not None:
        ax.axvline(deplete2_time, color="#7B1FA2", linestyle="-.", linewidth=1.8, label=f"Area 2 Depleted (Day {deplete2_time:.1f})")
    ax.set_ylabel("Stockpile Mass (t)")
    ax.set_title(f"{title} - Surface Buffers")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 2. Cumulative Production
    ax = axes[1]
    o1_col = "ore1_mined" if "ore1_mined" in df.columns else "Ore1_Mined"
    o2_col = "ore2_mined" if "ore2_mined" in df.columns else "Ore2_Mined"
    if o1_col in df.columns:
        ax.plot(df[time_col], df[o1_col] / 1000.0, label="Ore 1 Mined (kt)", color="#1976D2")
    if o2_col in df.columns:
        ax.plot(df[time_col], df[o2_col] / 1000.0, label="Ore 2 Mined (kt)", color="#D32F2F")
    if unlock_time is not None:
        ax.axvline(unlock_time, color="#2E7D32", linestyle="-.", linewidth=1.8)
    if deplete_time is not None:
        ax.axvline(deplete_time, color="#D32F2F", linestyle="--", linewidth=1.8)
    if deplete2_time is not None:
        ax.axvline(deplete2_time, color="#7B1FA2", linestyle="-.", linewidth=1.8)
    ax.set_ylabel("Cumulative Ore (kt)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 3. Capital Development & Readiness
    ax = axes[2]
    dev_col = "cumulative_development" if "cumulative_development" in df.columns else "cumulative_mine_development"
    if dev_col in df.columns:
        ax.plot(df[time_col], df[dev_col], label="Cumulative Development (m)", color="#7B1FA2")
        ax.axhline(4000.0, color="red", linestyle="--", label="Area 2 Target (4,000 m)")
    if unlock_time is not None:
        ax.axvline(unlock_time, color="#7B1FA2", linestyle="-.", linewidth=1.8, label=f"Area 2 Unlocked (Day {unlock_time:.1f})")
    if deplete_time is not None:
        ax.axvline(deplete_time, color="#D32F2F", linestyle="--", linewidth=1.8, label=f"Area 1 Depleted (Day {deplete_time:.1f})")
    if deplete2_time is not None:
        ax.axvline(deplete2_time, color="#1B5E20", linestyle="-.", linewidth=1.8, label=f"Area 2 Depleted (Day {deplete2_time:.1f})")
    ax.set_ylabel("Development (m)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 4. Cash Flows & Economics
    ax = axes[3]
    npv_col = "cumulative_npv" if "cumulative_npv" in df.columns else "operating_npv_proxy"
    cf_col = "cumulative_cash_flow" if "cumulative_cash_flow" in df.columns else "cumulative_cash_flow"
    if npv_col in df.columns:
        ax.plot(df[time_col], df[npv_col] / 1e6, label="Discounted NPV ($M, r=5%)", color="#2E7D32", linewidth=2)
    if cf_col in df.columns:
        ax.plot(df[time_col], df[cf_col] / 1e6, label="Undiscounted Cash Flow ($M)", color="#0288D1", linestyle="--")
    if unlock_time is not None:
        ax.axvline(unlock_time, color="#2E7D32", linestyle="-.", linewidth=1.8, label=f"Area 2 Unlocked (Day {unlock_time:.1f})")
    if deplete_time is not None:
        ax.axvline(deplete_time, color="#D32F2F", linestyle="--", linewidth=1.8, label=f"Area 1 Depleted (Day {deplete_time:.1f})")
    if deplete2_time is not None:
        ax.axvline(deplete2_time, color="#7B1FA2", linestyle="-.", linewidth=1.8, label=f"Area 2 Depleted (Day {deplete2_time:.1f})")
    ax.set_ylabel("Economics ($M)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # 5. Fleet Operating vs Idle Trucks
    ax = axes[4]
    if "trucks_operating" in df.columns:
        ax.plot(df[time_col], df["trucks_operating"], label="Operating Trucks", color="#F57C00")
        ax.plot(df[time_col], df["trucks_idle"], label="Idle Trucks", color="#757575", linestyle=":")
    ax.set_ylabel("Truck Count")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 6. Mode Distributions
    ax = axes[5]
    mode_col = "mill_mode" if "mill_mode" in df.columns else "active_operating_mode_name"
    if mode_col in df.columns:
        ax.plot(df[time_col], df[mode_col], label="Mill Operating Mode", color="#00796B", drawstyle="steps-post")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Mill Mode")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_full_hierarchy_dashboard(
    df_p1: pd.DataFrame,
    df_p2: pd.DataFrame,
    output_path: str = "plots/two_area_full_hierarchy.png",
    palette: dict = None,
    figsize: Tuple[int, int] = (18, 63),
) -> Dashboard:
    """Renders 14-panel comprehensive comparative visualization dashboard for full three-level hierarchy."""
    import os
    palette = palette or MODE_PALETTE
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    df_p1 = prepare_history(df_p1)
    df_p2 = prepare_history(df_p2)

    unlock_rows_p2 = df_p2[df_p2["area2_ready"] == True]
    unlock_time_p2 = (
        float(unlock_rows_p2["time"].iloc[0])
        if not unlock_rows_p2.empty
        else None
    )

    unlock_rows_p1 = df_p1[df_p1["area2_ready"] == True]
    unlock_time_p1 = (
        float(unlock_rows_p1["time"].iloc[0])
        if not unlock_rows_p1.empty
        else None
    )

    deplete_rows_p2 = df_p2[df_p2.get("area1_exhausted", False) == True]
    deplete_time_p2 = (
        float(deplete_rows_p2["time"].iloc[0])
        if not deplete_rows_p2.empty
        else None
    )

    deplete_rows_p1 = df_p1[df_p1.get("area1_exhausted", False) == True]
    deplete_time_p1 = (
        float(deplete_rows_p1["time"].iloc[0])
        if not deplete_rows_p1.empty
        else None
    )

    if deplete_time_p2 is None and "face1_mined" in df_p2.columns and len(df_p2) > 1:
        f1_max_p2 = df_p2["face1_mined"].max()
        if f1_max_p2 > 100.0:
            sub_p2 = df_p2[df_p2["face1_mined"] >= f1_max_p2 - 1e-3]
            if not sub_p2.empty and sub_p2["time"].iloc[0] < df_p2["time"].iloc[-1] - 1.0:
                deplete_time_p2 = float(sub_p2["time"].iloc[0])

    if deplete_time_p2 is None and "area1_depleted_day" in df_p2.columns:
        valid_d2 = df_p2[df_p2["area1_depleted_day"] >= 0.0]
        if not valid_d2.empty:
            deplete_time_p2 = float(valid_d2["area1_depleted_day"].iloc[0])

    if deplete_time_p1 is None and "face1_mined" in df_p1.columns and len(df_p1) > 1:
        f1_max_p1 = df_p1["face1_mined"].max()
        if f1_max_p1 > 100.0:
            sub_p1 = df_p1[df_p1["face1_mined"] >= f1_max_p1 - 1e-3]
            if not sub_p1.empty and sub_p1["time"].iloc[0] < df_p1["time"].iloc[-1] - 1.0:
                deplete_time_p1 = float(sub_p1["time"].iloc[0])

    if deplete_time_p1 is None and "area1_depleted_day" in df_p1.columns:
        valid_d1 = df_p1[df_p1["area1_depleted_day"] >= 0.0]
        if not valid_d1.empty:
            deplete_time_p1 = float(valid_d1["area1_depleted_day"].iloc[0])

    deplete2_rows_p2 = df_p2[df_p2.get("area2_exhausted", False) == True]
    deplete2_time_p2 = (
        float(deplete2_rows_p2["time"].iloc[0])
        if not deplete2_rows_p2.empty
        else None
    )
    if deplete2_time_p2 is None and "area2_depleted_day" in df_p2.columns:
        valid_d2_2 = df_p2[df_p2["area2_depleted_day"] >= 0.0]
        if not valid_d2_2.empty:
            deplete2_time_p2 = float(valid_d2_2["area2_depleted_day"].iloc[0])
    if deplete2_time_p2 is None and "face2_mined" in df_p2.columns and len(df_p2) > 1:
        f2_max_p2 = df_p2["face2_mined"].max()
        if f2_max_p2 > 100.0:
            sub2_p2 = df_p2[df_p2["face2_mined"] >= f2_max_p2 - 1e-3]
            if not sub2_p2.empty and sub2_p2["time"].iloc[0] < df_p2["time"].iloc[-1] - 1.0:
                deplete2_time_p2 = float(sub2_p2["time"].iloc[0])

    deplete2_rows_p1 = df_p1[df_p1.get("area2_exhausted", False) == True]
    deplete2_time_p1 = (
        float(deplete2_rows_p1["time"].iloc[0])
        if not deplete2_rows_p1.empty
        else None
    )
    if deplete2_time_p1 is None and "area2_depleted_day" in df_p1.columns:
        valid_d2_1 = df_p1[df_p1["area2_depleted_day"] >= 0.0]
        if not valid_d2_1.empty:
            deplete2_time_p1 = float(valid_d2_1["area2_depleted_day"].iloc[0])
    if deplete2_time_p1 is None and "face2_mined" in df_p1.columns and len(df_p1) > 1:
        f2_max_p1 = df_p1["face2_mined"].max()
        if f2_max_p1 > 100.0:
            sub2_p1 = df_p1[df_p1["face2_mined"] >= f2_max_p1 - 1e-3]
            if not sub2_p1.empty and sub2_p1["time"].iloc[0] < df_p1["time"].iloc[-1] - 1.0:
                deplete2_time_p1 = float(sub2_p1["time"].iloc[0])

    dash = Dashboard(
        nrows=14,
        ncols=1,
        figsize=figsize,
        sharex=False,
        title="Three-Level Strategic, Tactical & Analytical Blending Mining Benchmark",
    )
    dash.link_xaxes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    # 0. Cumulative Operating NPV Comparison (Policy 2 vs Policy 1)
    ax0 = dash[0]
    ax0.step(
        df_p2["time"],
        df_p2["operating_npv_proxy"] / 1e6,
        label="Policy 2: Value-Oriented Control + Analytical Blending",
        color="#2e7d32",
        linewidth=2.4,
        where="post",
    )
    ax0.step(
        df_p1["time"],
        df_p1["operating_npv_proxy"] / 1e6,
        label="Policy 1: Local-Objective Myopic Baseline",
        color="#c62828",
        linestyle="-",
        linewidth=2.0,
        where="post",
    )
    if unlock_time_p2 is not None:
        ax0.axvspan(
            df_p2["time"].min(),
            unlock_time_p2,
            color="#ffebee",
            alpha=0.35,
            label="Policy 2: Area 2 Locked (Capital Phase)",
        )
        ax0.axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Policy 2 Area 2 Unlocked (Day {unlock_time_p2:.1f})",
        )
        t_max = max(df_p2["time"].max(), df_p1["time"].max())
        text_x = (
            unlock_time_p2 + (t_max * 0.03)
            if (unlock_time_p2 < t_max * 0.80)
            else unlock_time_p2 - (t_max * 0.18)
        )
        y_pos = float(df_p2["operating_npv_proxy"].max() / 1e6) * 0.55
        ax0.annotate(
            f"* P2 AREA 2 UNLOCKED\nDay {unlock_time_p2:.1f}",
            xy=(unlock_time_p2, y_pos),
            xytext=(text_x, y_pos * 1.15),
            arrowprops=dict(
                facecolor="#2e7d32",
                edgecolor="#2e7d32",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#e8f5e9",
                edgecolor="#2e7d32",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#2e7d32",
            zorder=10,
        )

    if unlock_time_p1 is not None:
        ax0.axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle=":",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Policy 1 Area 2 Unlocked (Day {unlock_time_p1:.1f})",
        )
        t_max = max(df_p2["time"].max(), df_p1["time"].max())
        text_x_p1 = (
            unlock_time_p1 + (t_max * 0.03)
            if (unlock_time_p1 < t_max * 0.80)
            else unlock_time_p1 - (t_max * 0.18)
        )
        y_pos_p1 = float(df_p1["operating_npv_proxy"].max() / 1e6) * 0.45
        ax0.annotate(
            f"* P1 AREA 2 UNLOCKED\nDay {unlock_time_p1:.1f}",
            xy=(unlock_time_p1, y_pos_p1),
            xytext=(text_x_p1, y_pos_p1 * 0.85),
            arrowprops=dict(
                facecolor="#c62828",
                edgecolor="#c62828",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffebee",
                edgecolor="#c62828",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#c62828",
            zorder=10,
        )

    if deplete_time_p2 is not None:
        ax0.axvline(
            deplete_time_p2,
            color="#2e7d32",
            linestyle="--",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Policy 2 Area 1 Depleted (Day {deplete_time_p2:.1f})",
        )
        t_max = max(df_p2["time"].max(), df_p1["time"].max())
        text_x_p2_dep = (
            deplete_time_p2 - (t_max * 0.18)
            if (deplete_time_p2 > t_max * 0.70)
            else deplete_time_p2 + (t_max * 0.03)
        )
        y_pos_p2_dep = float(df_p2["operating_npv_proxy"].max() / 1e6) * 0.85
        ax0.annotate(
            f"* P2 AREA 1 DEPLETED\nDay {deplete_time_p2:.1f}",
            xy=(deplete_time_p2, y_pos_p2_dep),
            xytext=(text_x_p2_dep, y_pos_p2_dep * 0.90),
            arrowprops=dict(
                facecolor="#2e7d32",
                edgecolor="#2e7d32",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#e8f5e9",
                edgecolor="#2e7d32",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#2e7d32",
            zorder=10,
        )

    if deplete_time_p1 is not None:
        ax0.axvline(
            deplete_time_p1,
            color="#c62828",
            linestyle="--",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Policy 1 Area 1 Depleted (Day {deplete_time_p1:.1f})",
        )
        t_max = max(df_p2["time"].max(), df_p1["time"].max())
        text_x_p1_dep = (
            deplete_time_p1 + (t_max * 0.03)
            if (deplete_time_p1 < t_max * 0.80)
            else deplete_time_p1 - (t_max * 0.18)
        )
        y_pos_p1_dep = float(df_p1["operating_npv_proxy"].max() / 1e6) * 0.70
        ax0.annotate(
            f"* P1 AREA 1 DEPLETED\nDay {deplete_time_p1:.1f}",
            xy=(deplete_time_p1, y_pos_p1_dep),
            xytext=(text_x_p1_dep, y_pos_p1_dep * 1.10),
            arrowprops=dict(
                facecolor="#c62828",
                edgecolor="#c62828",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffebee",
                edgecolor="#c62828",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#c62828",
            zorder=10,
        )

    if deplete2_time_p2 is not None:
        ax0.axvline(
            deplete2_time_p2,
            color="#1b5e20",
            linestyle="-.",
            linewidth=2.2,
            alpha=0.95,
            label=f"* Policy 2 Area 2 Depleted (Day {deplete2_time_p2:.1f})",
        )
    if deplete2_time_p1 is not None:
        ax0.axvline(
            deplete2_time_p1,
            color="#b71c1c",
            linestyle="-.",
            linewidth=2.2,
            alpha=0.95,
            label=f"* Policy 1 Area 2 Depleted (Day {deplete2_time_p1:.1f})",
        )

    ax0.set_title(
        "Cumulative Operating NPV (@ 5% Discount Rate): Policy 2 vs Policy 1"
    )
    ax0.set_ylabel("Operating NPV (M$)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower right", framealpha=0.90)

    # 1. Campaign Discounted Cash Flow Rates & Daily Variation Comparison
    ax1 = dash[1]
    if "campaign_cash_flow_rate" in df_p2.columns:
        ax1.step(
            df_p2["time"],
            df_p2["campaign_cash_flow_rate"] / 1e3,
            label="Campaign Discounted CF Rate: Policy 2 ($k/day)",
            color="#2e7d32",
            linewidth=2.2,
            where="post",
        )
        if "daily_cash_flow_min" in df_p2.columns and "daily_cash_flow_max" in df_p2.columns:
            ax1.fill_between(
                df_p2["time"],
                df_p2["daily_cash_flow_min"] / 1e3,
                df_p2["daily_cash_flow_max"] / 1e3,
                color="#2e7d32",
                alpha=0.18,
                step="post",
                label="Daily CF Range (Policy 2)",
            )
    else:
        ax1.plot(
            df_p2["time"],
            df_p2["current_discounted_cash_flow_rate"] / 1e3,
            label="Discounted CF Rate: Policy 2 ($k/day)",
            color="#2e7d32",
            alpha=0.85,
        )

    if "campaign_cash_flow_rate" in df_p1.columns:
        ax1.step(
            df_p1["time"],
            df_p1["campaign_cash_flow_rate"] / 1e3,
            label="Campaign Discounted CF Rate: Policy 1 ($k/day)",
            color="#c62828",
            linestyle="-",
            linewidth=2.0,
            where="post",
        )
        if "daily_cash_flow_min" in df_p1.columns and "daily_cash_flow_max" in df_p1.columns:
            ax1.fill_between(
                df_p1["time"],
                df_p1["daily_cash_flow_min"] / 1e3,
                df_p1["daily_cash_flow_max"] / 1e3,
                color="#c62828",
                alpha=0.12,
                step="post",
                label="Daily CF Range (Policy 1)",
            )
    else:
        ax1.plot(
            df_p1["time"],
            df_p1["current_discounted_cash_flow_rate"] / 1e3,
            label="Discounted CF Rate: Policy 1 ($k/day)",
            color="#c62828",
            linestyle="-",
            linewidth=1.8,
            alpha=0.85,
        )

    ax1.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

    if unlock_time_p2 is not None:
        ax1.axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    if unlock_time_p1 is not None:
        ax1.axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle=":",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete_time_p2 is not None:
        ax1.axvline(
            deplete_time_p2,
            color="#2e7d32",
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
        )
    if deplete_time_p1 is not None:
        ax1.axvline(
            deplete_time_p1,
            color="#c62828",
            linestyle="--",
            linewidth=1.8,
            alpha=0.75,
        )
    if deplete2_time_p2 is not None:
        ax1.axvline(
            deplete2_time_p2,
            color="#1b5e20",
            linestyle="-.",
            linewidth=1.8,
            alpha=0.75,
        )
    if deplete2_time_p1 is not None:
        ax1.axvline(
            deplete2_time_p1,
            color="#b71c1c",
            linestyle="-.",
            linewidth=1.8,
            alpha=0.75,
        )
    ax1.set_title("Campaign Discounted Cash Flow Rate & Daily Variation Range ($k/day)")
    ax1.set_ylabel("Rate ($k/day)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left", framealpha=0.90)

    # 2. Stacked Development & Production: Policy 2 (Capital Dev + Stopes + Area 1/2 Production)
    ax2 = dash[2]
    dev_factor = 0.05  # 50 tonnes per meter of development = 0.05 kt/m
    p2_cap_dev_kt = df_p2["area2_cumulative_development"] * dev_factor
    p2_stope_dev_kt = (df_p2["cumulative_mine_development"] - df_p2["area2_cumulative_development"]).clip(lower=0.0) * dev_factor
    p2_ore1_kt = df_p2["area1_mined"] / 1000.0
    p2_ore2_kt = df_p2["area2_mined"] / 1000.0

    p2_y1 = p2_cap_dev_kt
    p2_y2 = p2_y1 + p2_stope_dev_kt
    p2_y3 = p2_y2 + p2_ore1_kt
    p2_y4 = p2_y3 + p2_ore2_kt

    ax2.fill_between(
        df_p2["time"],
        0,
        p2_y1,
        label="Area 2 Capital Decline (0 → 4,000 m / 200 kt Target)",
        color="#7b1fa2",
        alpha=0.75,
        step="post",
    )
    ax2.fill_between(
        df_p2["time"],
        p2_y1,
        p2_y2,
        label="Stope & Level Sustaining Development",
        color="#ff9800",
        alpha=0.55,
        step="post",
    )
    ax2.fill_between(
        df_p2["time"],
        p2_y2,
        p2_y3,
        label="Area 1 Stope Ore Production",
        color="#1976d2",
        alpha=0.65,
        step="post",
    )
    ax2.fill_between(
        df_p2["time"],
        p2_y3,
        p2_y4,
        label="Area 2 Stope Ore Production",
        color="#388e3c",
        alpha=0.65,
        step="post",
    )
    ax2.step(
        df_p2["time"],
        p2_y4,
        label="Total Material (Development + Production)",
        color="#1b5e20",
        linewidth=2.0,
        where="post",
    )
    if unlock_time_p2 is not None:
        ax2.axvline(
            unlock_time_p2,
            color="#7b1fa2",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete_time_p2 is not None:
        ax2.axvline(
            deplete_time_p2,
            color="#2e7d32",
            linestyle="--",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Policy 2 Area 1 Depleted (Day {deplete_time_p2:.1f})",
        )
        t_max_p2 = df_p2["time"].max()
        text_x_p2_dep = (
            deplete_time_p2 - (t_max_p2 * 0.18)
            if (deplete_time_p2 > t_max_p2 * 0.70)
            else deplete_time_p2 + (t_max_p2 * 0.03)
        )
        ax2.annotate(
            f"* AREA 1 DEPLETED\nDay {deplete_time_p2:.1f}",
            xy=(deplete_time_p2, p2_y4.max() * 0.85),
            xytext=(text_x_p2_dep, p2_y4.max() * 0.70),
            arrowprops=dict(
                facecolor="#2e7d32",
                edgecolor="#2e7d32",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#e8f5e9",
                edgecolor="#2e7d32",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#2e7d32",
            zorder=10,
        )
    if deplete2_time_p2 is not None:
        ax2.axvline(
            deplete2_time_p2,
            color="#1b5e20",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.90,
            label=f"* Policy 2 Area 2 Depleted (Day {deplete2_time_p2:.1f})",
        )
    ax2.set_title(
        "Policy 2: Stacked Underground Development & Stope Production (Decline + Stopes + Area 1/2 Ore)"
    )
    ax2.set_ylabel("Cumulative Material (kt)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", framealpha=0.90)

    # 3. Stacked Development & Production: Policy 1 (Myopic Baseline)
    ax3 = dash[3]
    p1_cap_dev_kt = df_p1["area2_cumulative_development"] * dev_factor
    p1_stope_dev_kt = (df_p1["cumulative_mine_development"] - df_p1["area2_cumulative_development"]).clip(lower=0.0) * dev_factor
    p1_ore1_kt = df_p1["area1_mined"] / 1000.0
    p1_ore2_kt = df_p1["area2_mined"] / 1000.0

    p1_y1 = p1_cap_dev_kt
    p1_y2 = p1_y1 + p1_stope_dev_kt
    p1_y3 = p1_y2 + p1_ore1_kt
    p1_y4 = p1_y3 + p1_ore2_kt

    ax3.fill_between(
        df_p1["time"],
        0,
        p1_y1,
        label="Area 2 Capital Decline (Emergency Finish on Depletion)",
        color="#c2185b",
        alpha=0.75,
        step="post",
    )
    ax3.fill_between(
        df_p1["time"],
        p1_y1,
        p1_y2,
        label="Stope & Level Sustaining Development",
        color="#e65100",
        alpha=0.55,
        step="post",
    )
    ax3.fill_between(
        df_p1["time"],
        p1_y2,
        p1_y3,
        label="Area 1 Stope Ore Production",
        color="#1976d2",
        alpha=0.65,
        step="post",
    )
    ax3.fill_between(
        df_p1["time"],
        p1_y3,
        p1_y4,
        label="Area 2 Stope Ore Production",
        color="#d32f2f",
        alpha=0.65,
        step="post",
    )
    ax3.step(
        df_p1["time"],
        p1_y4,
        label="Total Material (Development + Production)",
        color="#b71c1c",
        linewidth=2.0,
        where="post",
    )
    if unlock_time_p1 is not None:
        ax3.axvline(
            unlock_time_p1,
            color="#c2185b",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Policy 1 Area 2 Unlocked (Day {unlock_time_p1:.1f})",
        )
        t_max_p1 = df_p1["time"].max()
        text_x = (
            unlock_time_p1 + (t_max_p1 * 0.03)
            if (unlock_time_p1 < t_max_p1 * 0.80)
            else unlock_time_p1 - (t_max_p1 * 0.18)
        )
        ax3.annotate(
            f"* AREA 2 UNLOCKED\nDay {unlock_time_p1:.1f}",
            xy=(unlock_time_p1, 200.0),
            xytext=(text_x, 1500.0),
            arrowprops=dict(
                facecolor="#c2185b",
                edgecolor="#c2185b",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffebee",
                edgecolor="#c62828",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#c2185b",
            zorder=10,
        )
    if deplete_time_p1 is not None:
        ax3.axvline(
            deplete_time_p1,
            color="#c62828",
            linestyle="--",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Policy 1 Area 1 Depleted (Day {deplete_time_p1:.1f})",
        )
        t_max_p1 = df_p1["time"].max()
        text_x_p1_dep = (
            deplete_time_p1 + (t_max_p1 * 0.03)
            if (deplete_time_p1 < t_max_p1 * 0.80)
            else deplete_time_p1 - (t_max_p1 * 0.18)
        )
        ax3.annotate(
            f"* AREA 1 DEPLETED\nDay {deplete_time_p1:.1f}",
            xy=(deplete_time_p1, p1_y3.max() * 0.85),
            xytext=(text_x_p1_dep, p1_y3.max() * 0.70),
            arrowprops=dict(
                facecolor="#c62828",
                edgecolor="#c62828",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffebee",
                edgecolor="#c62828",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#c62828",
            zorder=10,
        )
    if deplete2_time_p1 is not None:
        ax3.axvline(
            deplete2_time_p1,
            color="#b71c1c",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.90,
            label=f"* Policy 1 Area 2 Depleted (Day {deplete2_time_p1:.1f})",
        )
    ax3.set_title(
        "Policy 1: Stacked Underground Development & Stope Production (Decline + Stopes + Area 1/2 Ore)"
    )
    ax3.set_ylabel("Cumulative Material (kt)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", framealpha=0.90)

    # 4. Stockpiles: Policy 2
    plot_ore_with_modes(
        df_p2,
        time_col="time",
        ore_cols=["total_system_ore_mass", "Ore1Stock_mass", "Ore2Stock_mass"],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Stockpiles & Campaigns: Policy 2 (Dual-Area Supply with Analytical Blending)",
        palette=palette,
        hlines=[
            {
                "y": 60000.0,
                "color": "black",
                "linestyle": "--",
                "label": "Target Total (60k)",
            },
            {
                "y": 20400.0,
                "color": "red",
                "linestyle": ":",
                "label": "Critical Ore 2 (20.4k)",
            },
        ],
        ax=dash[4],
    )
    if unlock_time_p2 is not None:
        dash[4].axvspan(
            df_p2["time"].min(),
            unlock_time_p2,
            color="#ffebee",
            alpha=0.35,
            label="Mine 2 Locked",
        )
        dash[4].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Mine 2 Unlocked (Day {unlock_time_p2:.1f})",
        )
        t_max = df_p2["time"].max()
        text_x = (
            unlock_time_p2 + (t_max * 0.03)
            if (unlock_time_p2 < t_max * 0.80)
            else unlock_time_p2 - (t_max * 0.18)
        )
        dash[4].annotate(
            f"* MINE 2 UNLOCKED\nDay {unlock_time_p2:.1f}",
            xy=(unlock_time_p2, 48000.0),
            xytext=(text_x, 52000.0),
            arrowprops=dict(
                facecolor="#2e7d32",
                edgecolor="#2e7d32",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#e8f5e9",
                edgecolor="#2e7d32",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#2e7d32",
            zorder=10,
        )
    if deplete_time_p2 is not None:
        dash[4].axvline(
            deplete_time_p2,
            color="#2e7d32",
            linestyle="--",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Area 1 Depleted (Day {deplete_time_p2:.1f})",
        )
        t_max_p2 = df_p2["time"].max()
        text_x_p2_dep = (
            deplete_time_p2 - (t_max_p2 * 0.18)
            if (deplete_time_p2 > t_max_p2 * 0.70)
            else deplete_time_p2 + (t_max_p2 * 0.03)
        )
        dash[4].annotate(
            f"* AREA 1 DEPLETED\nDay {deplete_time_p2:.1f}",
            xy=(deplete_time_p2, 40000.0),
            xytext=(text_x_p2_dep, 45000.0),
            arrowprops=dict(
                facecolor="#2e7d32",
                edgecolor="#2e7d32",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#e8f5e9",
                edgecolor="#2e7d32",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#2e7d32",
            zorder=10,
        )
    if deplete2_time_p2 is not None:
        dash[4].axvline(
            deplete2_time_p2,
            color="#1b5e20",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.90,
            label=f"* Area 2 Depleted (Day {deplete2_time_p2:.1f})",
        )
    dash[4].legend(loc="upper right", framealpha=0.90)

    # 5. Stockpiles: Policy 1
    plot_ore_with_modes(
        df_p1,
        time_col="time",
        ore_cols=["total_system_ore_mass", "Ore1Stock_mass", "Ore2Stock_mass"],
        mode_col="active_operating_mode_name",
        campaign_split_mode="SHUTDOWN",
        title="Stockpiles & Campaigns: Policy 1 (Severe Ore 2 Starvation & Mode B Trapping)",
        palette=palette,
        hlines=[
            {
                "y": 60000.0,
                "color": "black",
                "linestyle": "--",
                "label": "Target Total (60k)",
            },
            {
                "y": 20400.0,
                "color": "red",
                "linestyle": ":",
                "label": "Critical Ore 2 (20.4k)",
            },
        ],
        ax=dash[5],
    )
    if unlock_time_p1 is not None:
        dash[5].axvspan(
            df_p1["time"].min(),
            unlock_time_p1,
            color="#ffebee",
            alpha=0.35,
            label="Mine 2 Locked",
        )
        dash[5].axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle="-.",
            linewidth=2.5,
            alpha=0.95,
            label=f"* Mine 2 Unlocked (Day {unlock_time_p1:.1f})",
        )
        t_max_p1 = df_p1["time"].max()
        text_x_p1 = (
            unlock_time_p1 + (t_max_p1 * 0.03)
            if (unlock_time_p1 < t_max_p1 * 0.80)
            else unlock_time_p1 - (t_max_p1 * 0.18)
        )
        dash[5].annotate(
            f"* MINE 2 UNLOCKED\nDay {unlock_time_p1:.1f}",
            xy=(unlock_time_p1, 48000.0),
            xytext=(text_x_p1, 52000.0),
            arrowprops=dict(
                facecolor="#c62828",
                edgecolor="#c62828",
                shrink=0.08,
                width=2.0,
                headwidth=8,
            ),
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#ffebee",
                edgecolor="#c62828",
                linewidth=1.8,
                alpha=0.95,
            ),
            fontsize=10,
            fontweight="bold",
            color="#c62828",
            zorder=10,
        )
    if deplete_time_p1 is not None:
        dash[5].axvline(
            deplete_time_p1,
            color="#c62828",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
            label=f"* Area 1 Depleted (Day {deplete_time_p1:.1f})",
        )
    if deplete2_time_p1 is not None:
        dash[5].axvline(
            deplete2_time_p1,
            color="#b71c1c",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.90,
            label=f"* Area 2 Depleted (Day {deplete2_time_p1:.1f})",
        )
    dash[5].legend(loc="upper right", framealpha=0.90)

    # 6. Policy 2: Analytical Operational Face Allocation Weights (Appendix A & B)
    plot_time_series(
        df_p2,
        y_columns=["analytical_face1_weight", "analytical_face2_weight"],
        title="Policy 2: Analytical Face Allocation Dispatch Weights w1 (Face 1) & w2 (Face 2) [Slide 29]",
        is_step=True,
        ax=dash[6],
    )
    if unlock_time_p2 is not None:
        dash[6].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete_time_p2 is not None:
        dash[6].axvline(
            deplete_time_p2,
            color="#2e7d32",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete2_time_p2 is not None:
        dash[6].axvline(
            deplete2_time_p2,
            color="#1b5e20",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    dash[6].set_ylabel("Dispatch Weight (Fraction)")
    dash[6].set_ylim(-0.05, 1.05)
    dash[6].legend(loc="upper right")

    # 7. Operating Modes Timeline: Policy 2
    plot_time_series(
        df_p2,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes Timeline: Policy 2 (Balanced High-Grade Campaigns)",
        is_step=True,
        ax=dash[7],
    )
    if unlock_time_p2 is not None:
        dash[7].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
            label=f"* Policy 2 Unlocked (Day {unlock_time_p2:.1f})",
        )
    if deplete_time_p2 is not None:
        dash[7].axvline(
            deplete_time_p2,
            color="#2e7d32",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
            label=f"* Policy 2 Area 1 Depleted (Day {deplete_time_p2:.1f})",
        )
    if deplete2_time_p2 is not None:
        dash[7].axvline(
            deplete2_time_p2,
            color="#1b5e20",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
            label=f"* Policy 2 Area 2 Depleted (Day {deplete2_time_p2:.1f})",
        )
    dash[7].legend(loc="upper right", framealpha=0.90)

    # 8. Operating Modes Timeline: Policy 1
    plot_time_series(
        df_p1,
        y_columns=["Mode A", "Mode B", "Shutdown"],
        title="Operating Modes Timeline: Policy 1 (Permanently Trapped in Low-Throughput Mode B)",
        is_step=True,
        ax=dash[8],
    )
    if unlock_time_p1 is not None:
        dash[8].axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
            label=f"* Policy 1 Unlocked (Day {unlock_time_p1:.1f})",
        )
    if deplete_time_p1 is not None:
        dash[8].axvline(
            deplete_time_p1,
            color="#c62828",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
            label=f"* Policy 1 Area 1 Depleted (Day {deplete_time_p1:.1f})",
        )
    if deplete2_time_p1 is not None:
        dash[8].axvline(
            deplete2_time_p1,
            color="#b71c1c",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
            label=f"* Policy 1 Area 2 Depleted (Day {deplete2_time_p1:.1f})",
        )
    dash[8].legend(loc="upper right", framealpha=0.90)

    # 9. Strategic Trajectory Ratios: Policy 2
    plot_time_series(
        df_p2,
        y_columns=[
            "development_trajectory_ratio",
            "area2_readiness_trajectory_ratio",
            "ore1_trajectory_ratio",
            "ore2_trajectory_ratio",
        ],
        title="Policy 2: Strategic & Area 2 Trajectory Progress Ratios (Level 2 Tactical Reviews)",
        is_step=True,
        ax=dash[9],
    )
    if unlock_time_p2 is not None:
        dash[9].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete_time_p2 is not None:
        dash[9].axvline(
            deplete_time_p2,
            color="#2e7d32",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete2_time_p2 is not None:
        dash[9].axvline(
            deplete2_time_p2,
            color="#1b5e20",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    dash[9].axhline(
        0.90, color="red", linestyle=":", label="Tolerance Threshold (0.90)"
    )
    dash[9].axhline(
        1.00, color="gray", linestyle="--", label="Target Trajectory (1.00)"
    )
    dash[9].set_ylabel("Trajectory Ratio")
    dash[9].legend(loc="upper right")

    # 10. Fleet Utilization & Idle Time: Policy 2
    plot_truck_idle_and_utilization(
        df_p2,
        title="Policy 2: Haul Fleet Utilization & Idle Time Breakdown",
        ax=dash[10],
    )
    if unlock_time_p2 is not None:
        dash[10].axvline(
            unlock_time_p2,
            color="#2e7d32",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete_time_p2 is not None:
        dash[10].axvline(
            deplete_time_p2,
            color="#2e7d32",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete2_time_p2 is not None:
        dash[10].axvline(
            deplete2_time_p2,
            color="#1b5e20",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )

    # 11. Fleet Utilization & Idle Time: Policy 1
    plot_truck_idle_and_utilization(
        df_p1,
        title="Policy 1: Haul Fleet Utilization & Idle Time Breakdown",
        ax=dash[11],
    )
    if unlock_time_p1 is not None:
        dash[11].axvline(
            unlock_time_p1,
            color="#c62828",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete_time_p1 is not None:
        dash[11].axvline(
            deplete_time_p1,
            color="#c62828",
            linestyle="--",
            linewidth=2.0,
            alpha=0.85,
        )
    if deplete2_time_p1 is not None:
        dash[11].axvline(
            deplete2_time_p1,
            color="#b71c1c",
            linestyle="-.",
            linewidth=2.0,
            alpha=0.85,
        )

    # 12. Mode Distribution: Policy 2
    plot_mode_distribution(
        df_p2,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% Time Spent - Policy 2 Hierarchical Value-Oriented Control)",
        palette=palette,
        ax=dash[12],
    )

    # 13. Mode Distribution: Policy 1
    plot_mode_distribution(
        df_p1,
        mode_col="active_operating_mode_name",
        time_col="time",
        title="Mode Distribution (% Time Spent - Policy 1 Local Myopic Baseline)",
        palette=palette,
        ax=dash[13],
    )

    dash.save(output_path)
    if output_path != "plots/full_hierarchy_dashboard.png":
        dash.save("plots/full_hierarchy_dashboard.png")
    print(f"Saved full hierarchy benchmark dashboard to '{output_path}'.")
    return dash



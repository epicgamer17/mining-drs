import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from drs.plot import _get_ax


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

    from drs.plot import plot_time_series

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


def plot_state_space(
    df,
    col_x="Ore1Stock_mass",
    col_y="Ore2Stock_mass",
    title="State Space Trajectory",
    ax=None,
):
    ax = _get_ax(ax, figsize=(8, 8))

    if col_x not in df.columns or col_y not in df.columns:
        return ax

    ax.plot(df[col_x], df[col_y], color="gray", alpha=0.5, linewidth=1, zorder=1)

    time_col = "time" if "time" in df.columns else df.columns[0]
    scatter = ax.scatter(
        df[col_x], df[col_y], c=df[time_col], cmap="viridis", s=10, zorder=2
    )

    ax.figure.colorbar(scatter, ax=ax, label="Simulation Time")

    ax.plot(
        df[col_x].iloc[0],
        df[col_y].iloc[0],
        "go",
        markersize=10,
        label="Start",
        zorder=3,
    )
    ax.plot(
        df[col_x].iloc[-1],
        df[col_y].iloc[-1],
        "ro",
        markersize=10,
        label="End",
        zorder=3,
    )

    ax.set_xlabel(col_x, fontsize=12)
    ax.set_ylabel(col_y, fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(0, color="red", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    ax.grid(True)

    return ax


def plot_cumulative_throughput(
    df,
    extraction_col="cumulative_extracted_mass",
    time_col="time",
    ideal_rate=None,
    title="Cumulative Throughput vs Target",
    ax=None,
):
    ax = _get_ax(ax, figsize=(10, 6))

    if extraction_col not in df.columns or time_col not in df.columns:
        return ax

    ax.plot(
        df[time_col],
        df[extraction_col],
        label="Actual Extraction",
        color="green",
        linewidth=2,
    )

    if ideal_rate is not None:
        min_time = df[time_col].min()
        max_time = df[time_col].max()
        ideal_times = [min_time, max_time]

        start_val = df[extraction_col].min()
        ideal_vals = [start_val, start_val + ideal_rate * (max_time - min_time)]

        ax.plot(
            ideal_times,
            ideal_vals,
            "k--",
            label=f"Ideal Rate ({ideal_rate}/t)",
            alpha=0.7,
            linewidth=2,
        )

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("Cumulative Ore Extracted", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True)

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


def plot_attributed_deficit(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass",
                            ideal_rate_per_day=6000.0, title="Cumulative Production Deficit by Mode", ax=None, palette=None):
    ax = _get_ax(ax, figsize=(12, 6))

    dt = df[time_col].diff().shift(-1).fillna(0)
    actual_extraction_step = df[extraction_col].diff().shift(-1).fillna(0)

    ideal_extraction_step = dt * ideal_rate_per_day
    step_deficit = ideal_extraction_step - actual_extraction_step

    step_deficit = step_deficit.clip(lower=0)

    deficit_df = pd.DataFrame({
        'time': df[time_col],
        'mode': df[mode_col].astype(str),
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
        mode_name = str(c).split('.')[-1].upper()
        if mode_name in palette:
            colors.append(palette[mode_name])
        else:
            colors.append(cmap(idx % 10))

    cumulative_pivot[cols].plot.area(ax=ax, alpha=0.8, linewidth=0, color=colors)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Simulation Time (Days)")
    ax.set_ylabel("Cumulative Lost Tonnage")

    handles, labels = ax.get_legend_handles_labels()
    clean_labels = [str(l).split('.')[-1] for l in labels]
    ax.legend(handles, clean_labels, loc='upper left')

    return ax


def plot_deficit_disparity(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass", ideal_rate=6000.0, title="Mode Efficiency (Time Spent vs. Deficit Caused)", ax=None, verbose=True):
    ax = _get_ax(ax, figsize=(10, 6))

    df = df.copy()
    df[mode_col] = df[mode_col].astype(str)
    df['dt'] = df[time_col].diff().shift(-1).fillna(0)
    df['dx'] = df[extraction_col].diff().shift(-1).fillna(0)

    df['ideal_dx'] = df['dt'] * ideal_rate
    df['deficit'] = (df['ideal_dx'] - df['dx']).clip(lower=0)

    summary = df.groupby(mode_col).agg({'dt': 'sum', 'deficit': 'sum'})

    summary['% of Total Time'] = (summary['dt'] / summary['dt'].sum()) * 100
    summary['% of Total Deficit'] = (summary['deficit'] / summary['deficit'].sum()) * 100

    if verbose:
        print(f"\n--- {title} ---")
        print(summary[['% of Total Time', '% of Total Deficit']].round(1).to_string())
        print("-" * (8 + len(title)))

    melted = summary[['% of Total Time', '% of Total Deficit']].reset_index().melt(
        id_vars=mode_col, var_name="Metric", value_name="Percentage"
    )

    order = summary.sort_values('% of Total Deficit', ascending=False).index

    sns.barplot(data=melted, y=mode_col, x="Percentage", hue="Metric", order=order, palette=["#1f77b4", "#d62728"], ax=ax)

    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    return ax


def plot_deficit_breakdown_bar(df, time_col="time", mode_col="active_operating_mode", extraction_col="cumulative_extracted_mass", ideal_rate_per_day=6000.0, title="Final Deficit Breakdown by Mode (%)", ax=None, palette=None, verbose=True):
    ax = _get_ax(ax, figsize=(10, 6))

    df = df.copy()
    df['dt'] = df[time_col].diff().shift(-1).fillna(0)
    df['dx'] = df[extraction_col].diff().shift(-1).fillna(0)
    df['deficit'] = ((df['dt'] * ideal_rate_per_day) - df['dx']).clip(lower=0)
    df['mode_str'] = df[mode_col].astype(str).apply(lambda x: x.split('.')[-1])

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
    df['dt'] = df[time_col].diff().shift(-1).fillna(0)
    df['dx'] = df[extraction_col].diff().shift(-1).fillna(0)
    df['deficit'] = ((df['dt'] * ideal_rate) - df['dx']).clip(lower=0)
    df['mode_str'] = df[mode_col].astype(str)

    structural_modes = structural_modes or []

    def classify_bucket(mode):
        if any(sm in mode for sm in structural_modes) and "CONTINGENCY" not in mode and "SURGING" not in mode:
            return "Structural (Unavoidable: Geology & Shutdowns)"
        else:
            return "Operational (Avoidable: Control Logic & Contingencies)"

    df['Deficit_Type'] = df['mode_str'].apply(classify_bucket)

    pivot = df.pivot_table(index=time_col, columns='Deficit_Type', values='deficit', aggfunc='sum').fillna(0)
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
    df['dt'] = df[time_col].diff().shift(-1).fillna(0)
    df['dx'] = df[extraction_col].diff().shift(-1).fillna(0)
    df['deficit'] = ((df['dt'] * ideal_rate_per_day) - df['dx']).clip(lower=0)
    df['mode_str'] = df[mode_col].astype(str).apply(lambda x: x.split('.')[-1])

    pivot_df = df.pivot_table(index=time_col, columns='mode_str', values='deficit', aggfunc='sum').fillna(0)
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
    df['dt'] = df[time_col].diff().shift(-1).fillna(0)
    df['dx'] = df[extraction_col].diff().shift(-1).fillna(0)
    df['deficit'] = ((df['dt'] * ideal_rate) - df['dx']).clip(lower=0)
    df['mode_str'] = df[mode_col].astype(str).apply(lambda x: x.split('.')[-1])

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

"""Plot saved BENCH model outputs for a selected run folder with large-scale stochastic aggregation."""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def find_run_folder(output_root: str, run_name: str = None) -> str:
    if run_name:
        candidate = os.path.join(output_root, run_name)
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f"Run folder not found: {candidate}")

    folders = [f for f in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(f)]
    if not folders:
        raise FileNotFoundError(f"No run folders found under: {output_root}")
    return sorted(folders)[-1]


def create_plots_dir(run_folder: str) -> str:
    plots_dir = os.path.join(run_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def safe_plot(df, x_col, y_col, title, filename, plots_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_col], df[y_col], marker='o', linewidth=2)
    plt.title(title)
    plt.xlabel(x_col.capitalize())
    plt.ylabel(y_col.replace('_', ' ').capitalize())
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path)
    plt.close()
    return output_path


def _extract_variable_series(df: pd.DataFrame, y_col: str) -> np.ndarray:
    """Helper to cleanly extract arrays, implementing custom translation/scaling fallbacks."""
    if y_col == 'co2_emitted_tons_per_capita':
        if 'emissions_per_capita_tons' in df.columns:
            return df['emissions_per_capita_tons'].values
        if 'emissions_per_capita_kg_co2' in df.columns:
            return df['emissions_per_capita_kg_co2'].values / 1000.0
    elif y_col == 'emissions_avoided_tons_per_capita':
        if 'emissions_avoided_per_capita' in df.columns:
            return df['emissions_avoided_per_capita'].values / 1000.0
        if 'total_emissions_avoided_kg_co2' in df.columns:
            return df['total_emissions_avoided_kg_co2'].values / 1000.0
            
    if y_col in df.columns:
        return df[y_col].values
        
    return None


def find_run_data_for_config(output_root: str, configs: List[Dict[str, Any]]) -> List[Tuple[str, List[pd.DataFrame]]]:
    """
    Finds and collects data groups across all random seed runs for each configuration.
    Uses relaxed substring filtering to cleanly support custom runtime timestamp folder paths.
    """
    batch_data = []
    
    if not os.path.exists(output_root):
        print(f"Error: Target batch output folder does not exist: {output_root}")
        return batch_data

    all_subfolders = [f for f in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(f)]
    
    for config in configs:
        base_label = config.get('run_label')
        if not base_label:
            continue
            
        seed_dfs = []
        config_matched_folders = []
        for folder in all_subfolders:
            folder_name = os.path.basename(folder)
            if folder_name.endswith(f"_{base_label}") or f"_{base_label}_" in folder_name or folder_name == base_label:
                config_matched_folders.append(folder)
        
        config_matched_folders.sort()
        
        for matched_folder in config_matched_folders:
            csv_path = os.path.join(matched_folder, "annual_aggregates.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    seed_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load data matrix row from {csv_path}: {e}")
            
        if seed_dfs:
            batch_data.append((base_label, seed_dfs))
            print(f"✓ Found {len(seed_dfs)} functional seed datasets for setup tracking: '{base_label}'")
        else:
            print(f"Warning: No historical seed simulation folders found for configuration: '{base_label}'")
            
    return batch_data


def plot_batch_comparison(batch_data: List[Tuple[str, List[pd.DataFrame]]], 
                          x_col: str, y_col: str, title: str, 
                          filename: str, plots_dir: str) -> str:
    """Plots averaged metrics across configurations equipped with shaded 95% Confidence Intervals."""
    plt.figure(figsize=(11, 6.5))
    plotted = False

    print("Plot: ", title)

    for label, dfs in batch_data:
        combined_matrix = []
        x_values = None
        
        for df in dfs:
            y_series = _extract_variable_series(df, y_col)
            if x_col in df.columns and y_series is not None:
                df_sorted = df.sort_values(by=x_col)
                x_values = df_sorted[x_col].values
                y_series_sorted = _extract_variable_series(df_sorted, y_col)
                combined_matrix.append(y_series_sorted)
                
        if not combined_matrix:
            continue
            
        data_matrix = np.array(combined_matrix)
        n_samples = data_matrix.shape[0]
        
        mean_line = np.mean(data_matrix, axis=0)
        line, = plt.plot(x_values, mean_line, label=f"{label} (N={n_samples})", marker='o', linewidth=2, markersize=4)

        #print(mean_line)
        
        color = line.get_color()
        plotted = True
        
        if n_samples > 1:
            std_dev = np.std(data_matrix, axis=0, ddof=1)
            standard_error = std_dev / np.sqrt(n_samples)
            margin_error = 1.96 * standard_error
            
            lower_bound = mean_line - margin_error
            upper_bound = mean_line + margin_error
            
            plt.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.15)

    if not plotted:
        plt.close()
        return ''

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(x_col.capitalize(), fontsize=12)
    
    if y_col == 'co2_emitted_tons_per_capita':
        plt.ylabel('CO2 Emissions per Capita (tons)', fontsize=12)
    elif y_col == 'emissions_avoided_tons_per_capita':
        plt.ylabel('CO2 Emissions Avoided (tons per capita)', fontsize=12)
    else:
        plt.ylabel(y_col.replace('_', ' ').capitalize(), fontsize=12)
        
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e0e0e0')
    plt.tight_layout()
    
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_combined_actions(batch_data: List[Tuple[str, List[pd.DataFrame]]],
                          x_col: str, plots_dir: str) -> str:
    """
    3-row subplot grid: one row per action type, all scenarios as coloured lines.
    This keeps the legend small (scenario names only) and makes cross-scenario
    comparison easy for each action type independently.
    """
    action_configs = [
        ('action_1_count', 'Investment (PV Installation)'),
        ('action_2_count', 'Conservation (Efficiency)'),
        ('action_3_count', 'Switching (Renewable)'),
    ]

    # One colour per scenario/config
    scenario_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    any_plotted = False

    for ax, (action_col, action_title) in zip(axes, action_configs):
        ax.set_title(action_title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Households taking action', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

        for (label, dfs), color in zip(batch_data, scenario_colors):
            combined_matrix = []
            x_values = None

            for df in dfs:
                if x_col in df.columns and action_col in df.columns:
                    df_sorted = df.sort_values(by=x_col)
                    if x_values is None:
                        x_values = df_sorted[x_col].values
                    combined_matrix.append(df_sorted[action_col].values)

            if not combined_matrix or x_values is None:
                continue

            data_matrix = np.array(combined_matrix)
            n_samples = data_matrix.shape[0]
            mean_line = np.mean(data_matrix, axis=0)

            ax.plot(x_values, mean_line, label=f"{label} (N={n_samples})",
                    color=color, linewidth=2, marker='o', markersize=4)

            if n_samples > 1:
                std_dev = np.std(data_matrix, axis=0, ddof=1)
                se = std_dev / np.sqrt(n_samples)
                ax.fill_between(x_values, mean_line - 1.96 * se,
                                mean_line + 1.96 * se, color=color, alpha=0.15)

            any_plotted = True

        ax.legend(loc='upper left', frameon=True, facecolor='white',
                  edgecolor='#e0e0e0', fontsize=9)

    if not any_plotted:
        plt.close()
        return ''

    axes[-1].set_xlabel('Year', fontsize=11)
    fig.suptitle('Household Actions Over Time (mean ± 95% CI)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(plots_dir, 'batch_comparison_all_actions.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_subaction_counts(batch_data: List[Tuple[str, List[pd.DataFrame]]],
                          x_col: str, plots_dir: str) -> str:
    """
    3-row × 2-col subplot grid for the 6 action sub-types.
    Row = action type (Investment / Conservation / Switching).
    Col = agent sub-type (Grey | Brown+Green, or To-Brown | To-Green).
    All scenarios plotted as coloured lines with 95% CI shading on each panel.
    """
    subaction_configs = [
        # (column_name, panel_title, row, col)
        ('inv_grey_count',        'Investment — Grey agents (act12)',          0, 0),
        ('inv_brown_green_count', 'Investment — Brown/Green agents (act11)',   0, 1),
        ('con_grey_count',        'Conservation — Grey agents (act40)',        1, 0),
        ('con_brown_green_count', 'Conservation — Brown/Green agents (act21)', 1, 1),
        ('swi_to_brown_count',    'Switching — Grey → Brown (act32)',          2, 0),
        ('swi_to_green_count',    'Switching — Brown → Green (act31)',         2, 1),
    ]

    scenario_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    any_plotted = False

    for col_name, panel_title, row, col in subaction_configs:
        ax = axes[row, col]
        ax.set_title(panel_title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Households', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

        for (label, dfs), color in zip(batch_data, scenario_colors):
            combined_matrix = []
            x_values = None

            for df in dfs:
                if x_col in df.columns and col_name in df.columns:
                    df_sorted = df.sort_values(by=x_col)
                    if x_values is None:
                        x_values = df_sorted[x_col].values
                    combined_matrix.append(df_sorted[col_name].values)

            if not combined_matrix or x_values is None:
                continue

            data_matrix = np.array(combined_matrix)
            n_samples = data_matrix.shape[0]
            mean_line = np.mean(data_matrix, axis=0)

            ax.plot(x_values, mean_line, label=f"{label} (N={n_samples})",
                    color=color, linewidth=2, marker='o', markersize=4)

            if n_samples > 1:
                se = np.std(data_matrix, axis=0, ddof=1) / np.sqrt(n_samples)
                ax.fill_between(x_values, mean_line - 1.96 * se,
                                mean_line + 1.96 * se, color=color, alpha=0.15)
            any_plotted = True

        ax.legend(loc='upper left', frameon=True, facecolor='white',
                  edgecolor='#e0e0e0', fontsize=8)

    if not any_plotted:
        plt.close()
        return ''

    for col in range(2):
        axes[-1, col].set_xlabel('Year', fontsize=10)

    fig.suptitle('Action Sub-Types Over Time (mean ± 95% CI)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = os.path.join(plots_dir, 'batch_subaction_counts.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_energy_by_type(batch_data: List[Tuple[str, List[pd.DataFrame]]],
                        x_col: str, plots_dir: str) -> str:
    """Stacked area chart of mean consumption split by grey / brown / green."""
    fig, axes = plt.subplots(1, len(batch_data), figsize=(6 * len(batch_data), 5), squeeze=False)

    for ax, (label, dfs) in zip(axes[0], batch_data):
        cols = ['consumption_grey', 'consumption_brown', 'consumption_green']
        colors = ['#888888', '#a0522d', '#3cb371']
        names  = ['Grey (fossil)', 'Brown (mixed)', 'Green (renewable)']

        x_values = None
        matrices = {c: [] for c in cols}
        for df in dfs:
            if x_col not in df.columns:
                continue
            df_s = df.sort_values(by=x_col)
            if x_values is None:
                x_values = df_s[x_col].values
            for c in cols:
                if c in df_s.columns:
                    matrices[c].append(df_s[c].values)

        if x_values is None or not matrices[cols[0]]:
            ax.set_title(f"{label}\n(no data)")
            continue

        means = [np.mean(np.array(matrices[c]), axis=0) if matrices[c] else np.zeros_like(x_values)
                 for c in cols]
        ax.stackplot(x_values, *means, labels=names, colors=colors, alpha=0.75)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('Consumption (kWh)', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Energy Consumption by Type', fontsize=13, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'batch_energy_by_type.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_awareness_motivation(batch_data: List[Tuple[str, List[pd.DataFrame]]],
                              x_col: str, plots_dir: str) -> str:
    """Line plot of avg_awareness and avg_motivation trajectories with 95% CI."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for label, dfs in batch_data:
        for ax, var, var_label in [
            (ax1, 'avg_awareness', 'Average Awareness'),
            (ax2, 'avg_motivation', 'Average Motivation'),
        ]:
            matrix, x_values = [], None
            for df in dfs:
                if x_col not in df.columns or var not in df.columns:
                    continue
                df_s = df.sort_values(by=x_col)
                if x_values is None:
                    x_values = df_s[x_col].values
                matrix.append(df_s[var].values)

            if not matrix or x_values is None:
                continue

            arr = np.array(matrix)
            mean = arr.mean(axis=0)
            line, = ax.plot(x_values, mean, label=label, marker='o', linewidth=2, markersize=4)
            if arr.shape[0] > 1:
                se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ax.fill_between(x_values, mean - 1.96 * se, mean + 1.96 * se,
                                color=line.get_color(), alpha=0.15)

        ax1.set_title('Awareness Trajectory (95% CI)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=10); ax1.set_ylabel('Score (0-7)', fontsize=10)
        ax1.legend(fontsize=9); ax1.grid(True, linestyle='--', alpha=0.4)

        ax2.set_title('Motivation Trajectory (95% CI)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=10); ax2.set_ylabel('Score (0-7)', fontsize=10)
        ax2.legend(fontsize=9); ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'batch_awareness_motivation.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_income_group_actions(output_root: str, configs: List[Dict],
                              plots_dir: str) -> str:
    """
    Bar chart of cumulative action counts (investment / conservation / switching)
    broken down by income group, averaged across seeds.
    Reads actions_by_income_group.csv from each matched run folder.
    """
    all_subfolders = sorted([
        f for f in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(f)
    ])

    action_cols  = ['action_1_count', 'action_2_count', 'action_3_count']
    action_names = ['Investment', 'Conservation', 'Switching']
    bar_colors   = ['#2E86AB', '#A23B72', '#F18F01']
    n_groups     = 7
    x            = np.arange(n_groups)
    width        = 0.8 / max(len(configs), 1) / len(action_cols)

    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = False

    for cfg_idx, config in enumerate(configs):
        label = config.get('run_label', '')
        if not label:
            continue
        matched = [f for f in all_subfolders
                   if os.path.basename(f).endswith(f'_{label}')
                   or f'_{label}_' in os.path.basename(f)
                   or os.path.basename(f) == label]

        dfs = []
        for folder in matched:
            csv_path = os.path.join(folder, 'actions_by_income_group.csv')
            if os.path.exists(csv_path):
                try:
                    dfs.append(pd.read_csv(csv_path))
                except Exception:
                    pass

        if not dfs:
            continue

        combined = pd.concat(dfs, ignore_index=True)
        # Sum across all years, then average across seeds (# seeds = # matched folders)
        group_totals = combined.groupby('income_group')[action_cols].sum() / max(len(dfs), 1)

        for a_idx, (col, name, color) in enumerate(zip(action_cols, action_names, bar_colors)):
            if col not in group_totals.columns:
                continue
            offset = (cfg_idx * len(action_cols) + a_idx) * width
            vals = [group_totals.loc[g, col] if g in group_totals.index else 0 for g in range(1, 8)]
            ax.bar(x + offset, vals, width, label=f"{label} – {name}", color=color,
                   alpha=0.7 if cfg_idx == 0 else 0.4,
                   hatch='' if cfg_idx == 0 else '//')
            plotted = True

    if not plotted:
        plt.close()
        return ''

    ax.set_xticks(x + width * float(len(configs) * len(action_cols) - 1) / 2)
    ax.set_xticklabels([f"Group {g}" for g in range(1, 8)], fontsize=10)
    ax.set_title('Cumulative Actions by Income Group (mean across seeds)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Income Group', fontsize=11)
    ax.set_ylabel('Cumulative Action Count', fontsize=11)
    ax.legend(fontsize=8, loc='upper left', ncol=3)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    output_path = os.path.join(plots_dir, 'batch_income_group_actions.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_awareness_by_income_group(output_root: str, configs: List[Dict],
                                   plots_dir: str) -> str:
    """
    One subfigure per scenario.  Each panel shows avg_awareness over time for
    income groups 1-7 (mean ± 95% CI across seeds), read from
    actions_by_income_group.csv in each run folder.
    """
    all_subfolders = sorted([
        f for f in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(f)
    ])

    group_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                    '#9467bd', '#8c564b', '#e377c2']

    configs_with_data = []
    for config in configs:
        label = config.get('run_label', '')
        if not label:
            continue
        matched = [f for f in all_subfolders
                   if os.path.basename(f).endswith(f'_{label}')
                   or f'_{label}_' in os.path.basename(f)]
        dfs = []
        for folder in matched:
            p = os.path.join(folder, 'actions_by_income_group.csv')
            if os.path.exists(p):
                try:
                    dfs.append(pd.read_csv(p))
                except Exception:
                    pass
        if dfs:
            configs_with_data.append((label, dfs))

    if not configs_with_data:
        return ''

    n = len(configs_with_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (label, dfs) in zip(axes, configs_with_data):
        for g, color in zip(range(1, 8), group_colors):
            matrix, years = [], None
            for df in dfs:
                grp = df[df['income_group'] == g].sort_values('year')
                if grp.empty or 'avg_awareness' not in grp.columns:
                    continue
                if years is None:
                    years = grp['year'].values
                matrix.append(grp['avg_awareness'].values)

            if not matrix or years is None:
                continue

            arr = np.array(matrix)
            mean = arr.mean(axis=0)
            ax.plot(years, mean, label=f'Group {g}', color=color, linewidth=2)
            if arr.shape[0] > 1:
                se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ax.fill_between(years, mean - 1.96 * se, mean + 1.96 * se,
                                color=color, alpha=0.12)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Avg Awareness', fontsize=9)
        ax.legend(fontsize=7, loc='upper left', ncol=2)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Awareness by Income Group per Scenario (95% CI)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'batch_awareness_by_income_group.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_cumulative_actions_per_scenario(batch_data: List[Tuple[str, List[pd.DataFrame]]],
                                         x_col: str, plots_dir: str) -> str:
    """
    One subfigure per scenario.  Each panel shows cumulative action counts for
    Investment, Conservation and Switching (mean ± 95% CI across seeds).
    Cumsum is applied per seed before averaging so uncertainty is preserved.
    """
    if not batch_data:
        return ''

    action_cols  = ['action_1_count', 'action_2_count', 'action_3_count']
    action_names = ['Investment (PV)', 'Conservation', 'Switching']
    action_colors = ['#2E86AB', '#A23B72', '#F18F01']

    n = len(batch_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, dfs) in zip(axes, batch_data):
        for col, name, color in zip(action_cols, action_names, action_colors):
            matrix, x_values = [], None
            for df in dfs:
                if x_col not in df.columns or col not in df.columns:
                    continue
                df_s = df.sort_values(by=x_col)
                if x_values is None:
                    x_values = df_s[x_col].values
                matrix.append(df_s[col].cumsum().values)

            if not matrix or x_values is None:
                continue

            arr = np.array(matrix)
            mean = arr.mean(axis=0)
            ax.plot(x_values, mean, label=name, color=color, linewidth=2, marker='o', markersize=3)
            if arr.shape[0] > 1:
                se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ax.fill_between(x_values, mean - 1.96 * se, mean + 1.96 * se,
                                color=color, alpha=0.15)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Cumulative Households', fontsize=9)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Cumulative Actions by Type per Scenario (95% CI)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'batch_cumulative_actions_per_scenario.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_total_emissions_per_scenario(batch_data: List[Tuple[str, List[pd.DataFrame]]],
                                      x_col: str, plots_dir: str) -> str:
    """
    One subfigure per scenario showing total CO2 emissions (tons) over time,
    mean ± 95% CI across seeds.
    """
    col = 'total_emissions_tons_co2'
    supported = any(col in df.columns for _, dfs in batch_data for df in dfs)
    if not supported:
        return ''

    n = len(batch_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, dfs) in zip(axes, batch_data):
        matrix, x_values = [], None
        for df in dfs:
            if x_col not in df.columns or col not in df.columns:
                continue
            df_s = df.sort_values(by=x_col)
            if x_values is None:
                x_values = df_s[x_col].values
            matrix.append(df_s[col].values)

        if not matrix or x_values is None:
            ax.set_visible(False)
            continue

        arr = np.array(matrix)
        mean = arr.mean(axis=0)
        ax.plot(x_values, mean, color='#C0392B', linewidth=2, marker='o', markersize=3)
        if arr.shape[0] > 1:
            se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ax.fill_between(x_values, mean - 1.96 * se, mean + 1.96 * se,
                            color='#C0392B', alpha=0.15)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Total CO2 Emissions (tons)', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Total CO2 Emissions per Scenario (95% CI)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'batch_total_emissions_per_scenario.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_emissions_avoided_by_source(batch_data: List[Tuple[str, List[pd.DataFrame]]],
                                      x_col: str, plots_dir: str) -> str:
    """
    Three-panel figure: CO2 emissions avoided per capita (tons) broken down by
    action source — Investment, Conservation, Switching.  Each panel shows all
    configurations with 95% CI shading, matching the style of the main
    emissions-avoided trajectory plot.
    """
    source_cols = [
        ('em_avoided_inv_per_capita', 'Investment (PV)'),
        ('em_avoided_con_per_capita', 'Conservation'),
        ('em_avoided_swi_per_capita', 'Switching'),
    ]

    supported = any(
        col in df.columns
        for _, dfs in batch_data
        for df in dfs
        for col, _ in source_cols
    )
    if not supported:
        return ''

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for ax, (col, source_name) in zip(axes, source_cols):
        for label, dfs in batch_data:
            matrix, x_values = [], None
            for df in dfs:
                if x_col not in df.columns or col not in df.columns:
                    continue
                df_s = df.sort_values(by=x_col)
                if x_values is None:
                    x_values = df_s[x_col].values
                matrix.append(df_s[col].values / 1000.0)  # kg → tons

            if not matrix or x_values is None:
                continue

            arr = np.array(matrix)
            mean = arr.mean(axis=0)
            line, = ax.plot(x_values, mean, label=label, marker='o', linewidth=2, markersize=4)
            if arr.shape[0] > 1:
                se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ax.fill_between(x_values, mean - 1.96 * se, mean + 1.96 * se,
                                color=line.get_color(), alpha=0.15)

        ax.set_title(source_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Year', fontsize=10)
        ax.set_ylabel('CO2 Avoided per Capita (tons)', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('CO2 Emissions Avoided per Capita by Action Source (95% CI)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'batch_emissions_avoided_by_source.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_awareness_bins(batch_data: List[Tuple[str, List[pd.DataFrame]]],
                        x_col: str, plots_dir: str) -> str:
    """
    One subfigure per scenario. Each panel shows the count of households in each
    of 7 awareness bins ([0-1], (1-2], ..., (6-7]) over time, matching the NetLogo
    draw_awareness plot (7 pens). Mean ± 95% CI across seeds.
    """
    if not batch_data:
        return ''

    bin_cols  = [f'aware_bin_{i}' for i in range(1, 8)]
    bin_labels = ['[0–1]', '(1–2]', '(2–3]', '(3–4]', '(4–5]', '(5–6]', '(6–7]']
    bin_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2']

    # Drop any config whose CSVs lack the bin columns (old runs pre-dating this stat)
    valid = [(label, dfs) for label, dfs in batch_data
             if dfs and bin_cols[0] in dfs[0].columns]
    if not valid:
        return ''

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (label, dfs) in zip(axes, valid):
        for col, bin_label, color in zip(bin_cols, bin_labels, bin_colors):
            matrix, x_values = [], None
            for df in dfs:
                df_s = df.sort_values(x_col)
                if col not in df_s.columns:
                    continue
                if x_values is None:
                    x_values = df_s[x_col].values
                matrix.append(df_s[col].values)

            if not matrix or x_values is None:
                continue

            arr = np.array(matrix)
            mean = arr.mean(axis=0)
            ax.plot(x_values, mean, label=bin_label, color=color, linewidth=2)
            if arr.shape[0] > 1:
                se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
                ax.fill_between(x_values, mean - 1.96 * se, mean + 1.96 * se,
                                color=color, alpha=0.12)

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Year', fontsize=9)
        ax.set_ylabel('Number of Households', fontsize=9)
        ax.legend(fontsize=7, loc='upper left', ncol=2, title='Awareness bin')
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Awareness Distribution over Time per Scenario (95% CI)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, 'batch_awareness_bins.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path


def plot_batch_for_config(config_file_path: str, output_root: str) -> List[str]:
    from model.config_loader import load_config_file
    print(f"Loading batch configurations from: {config_file_path}")
    configs = load_config_file(config_file_path)
    
    batch_data = find_run_data_for_config(output_root, configs)
    if not batch_data:
        print("No run simulation data folders located for plotting.")
        return []
        
    plots_dir = create_plots_dir(output_root)
    saved_plots = []
    
    # Variables to plot (removed individual action counts)
    variables_to_plot = [
        'green_share_percent',
        'total_energy_saved_kwh',
        'co2_emitted_tons_per_capita',
        'emissions_avoided_tons_per_capita'
    ]
    
    for var in variables_to_plot:
        if var == 'co2_emitted_tons_per_capita':
            supported = any(('emissions_per_capita_kg_co2' in df.columns or 'emissions_per_capita_tons' in df.columns) for _, dfs in batch_data for df in dfs)
            title = 'CO2 Emissions per Capita (tons) (95% CI)'
        elif var == 'emissions_avoided_tons_per_capita':
            supported = any(('emissions_avoided_per_capita' in df.columns or 'total_emissions_avoided_kg_co2' in df.columns) for _, dfs in batch_data for df in dfs)
            title = 'CO2 Emissions Avoided per Capita (tons) (95% CI)'
        else:
            supported = any(var in df.columns for _, dfs in batch_data for df in dfs)
            title = f"Averaged {var.replace('_', ' ').capitalize()} (95% CI)"
            
        if not supported:
            continue
            
        filename = f"batch_comparison_{var}.png"
        plot_path = plot_batch_comparison(batch_data, 'year', var, title, filename, plots_dir)
        if plot_path:
            saved_plots.append(plot_path)
    
    # Add combined actions plot
    combined_plot_path = plot_combined_actions(batch_data, 'year', plots_dir)
    if combined_plot_path:
        saved_plots.append(combined_plot_path)
        print(f"✓ Created combined actions plot: {combined_plot_path}")

    # Sub-action breakdown (6-panel grid)
    subaction_path = plot_subaction_counts(batch_data, 'year', plots_dir)
    if subaction_path:
        saved_plots.append(subaction_path)
        print(f"✓ Created sub-action counts plot: {subaction_path}")

    # Energy by type (stacked area)
    energy_type_path = plot_energy_by_type(batch_data, 'year', plots_dir)
    if energy_type_path:
        saved_plots.append(energy_type_path)
        print(f"✓ Created energy-by-type plot: {energy_type_path}")

    # Awareness and motivation trajectories
    aw_mot_path = plot_awareness_motivation(batch_data, 'year', plots_dir)
    if aw_mot_path:
        saved_plots.append(aw_mot_path)
        print(f"✓ Created awareness/motivation plot: {aw_mot_path}")

    # Emissions avoided decomposed by action source (3-panel)
    em_source_path = plot_emissions_avoided_by_source(batch_data, 'year', plots_dir)
    if em_source_path:
        saved_plots.append(em_source_path)
        print(f"✓ Created emissions-avoided-by-source plot: {em_source_path}")

    # Awareness by income group per scenario
    aw_ig_path = plot_awareness_by_income_group(output_root, configs, plots_dir)
    if aw_ig_path:
        saved_plots.append(aw_ig_path)
        print(f"✓ Created awareness-by-income-group plot: {aw_ig_path}")

    # Awareness bin distribution over time (7 lines matching NetLogo draw_awareness)
    aw_bins_path = plot_awareness_bins(batch_data, 'year', plots_dir)
    if aw_bins_path:
        saved_plots.append(aw_bins_path)
        print(f"✓ Created awareness bins plot: {aw_bins_path}")

    # Cumulative actions per scenario (one panel per scenario)
    cum_act_path = plot_cumulative_actions_per_scenario(batch_data, 'year', plots_dir)
    if cum_act_path:
        saved_plots.append(cum_act_path)
        print(f"✓ Created cumulative actions per scenario plot: {cum_act_path}")

    # Total emissions per scenario (one panel per scenario)
    tot_em_path = plot_total_emissions_per_scenario(batch_data, 'year', plots_dir)
    if tot_em_path:
        saved_plots.append(tot_em_path)
        print(f"✓ Created total emissions per scenario plot: {tot_em_path}")

    # Income-group action bar chart
    ig_path = plot_income_group_actions(output_root, configs, plots_dir)
    if ig_path:
        saved_plots.append(ig_path)
        print(f"✓ Created income-group actions plot: {ig_path}")

    print(f"✓ Stochastic batch plots successfully saved inside directory: {plots_dir}")
    return saved_plots


def build_parser():
    parser = argparse.ArgumentParser(description='Plot BENCH simulation outputs for a run folder or config batch.')
    parser.add_argument(
        '--run', '-r',
        help='The run folder name inside the output directory. If omitted, the latest run is used.'
    )
    parser.add_argument(
        '--config-file', '-c',
        help='Path to a JSON or YAML configuration file for batch plotting multiple runs.'
    )
    parser.add_argument(
        '--output-root', '-o',
        default=os.path.join(os.path.dirname(__file__), 'output'),
        help='Root output directory containing run folders.'
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.config_file:
        plot_batch_for_config(args.config_file, args.output_root)
    else:
        try:
            run_folder = find_run_folder(args.output_root, args.run)
            print(f"Plotting individual summary results for: {run_folder}")
            csv_path = os.path.join(run_folder, "annual_aggregates.csv")
            if not os.path.exists(csv_path):
                print(f"Error: Could not find annual_aggregates.csv in {run_folder}")
                return 1
                
            df = pd.read_csv(csv_path)
            plots_dir = create_plots_dir(run_folder)
            
            # Updated variables - removed individual action counts
            variables = [
                ('green_share_percent', "Green Consumption Share (%)"),
                ('total_emissions_avoided_kg_co2', "Avoided Emissions (kg CO2)"),
                ('total_energy_saved_kwh', "Saved Energy (kWh)")
            ]
            
            for var, title in variables:
                if var in df.columns:
                    safe_plot(df, 'year', var, title, f"summary_{var}.png", plots_dir)
            
            # Create combined actions plot for individual run
            plt.figure(figsize=(12, 7))
            action_cols = ['action_1_count', 'action_2_count', 'action_3_count']
            action_names = ['Investment (PV Installation)', 'Conservation (Efficiency)', 'Switching (Renewable)']
            colors = ['#2E86AB', '#A23B72', '#F18F01']
            markers = ['o', 's', '^']
            
            for col, name, color, marker in zip(action_cols, action_names, colors, markers):
                if col in df.columns:
                    plt.plot(df['year'], df[col], label=name, color=color, marker=marker, 
                            linewidth=2, markersize=6, markevery=2)
            
            plt.title('Household Actions Over Time', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Number of Households Taking Action', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='best', frameon=True, facecolor='white', edgecolor='#e0e0e0')
            plt.tight_layout()
            
            output_path = os.path.join(plots_dir, 'summary_all_actions.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"✓ Individual plots written into: {plots_dir}")
        except Exception as e:
            print(f"Error executing fallback plotting: {e}")
            return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
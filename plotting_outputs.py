"""Plot saved BENCH model outputs for a selected run folder."""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from utils.config_loader import load_config_file


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
    #plt.close()
    return output_path


def plot_multi_series(df, x_col, y_cols, title, filename, plots_dir):
    plt.figure(figsize=(10, 6))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], marker='o', label=col)
    plt.title(title)
    plt.xlabel(x_col.capitalize())
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path)
    #plt.close()
    return output_path


def _sanitize_string(value: str) -> str:
    safe = ''.join(c if c.isalnum() or c in {'_', '-'} else '_' for c in value.strip().replace(' ', '_'))
    return safe.strip('_')[:80]


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def find_run_folder_by_label(output_root: str, run_label: str) -> str:
    run_label_slug = _sanitize_string(run_label)
    folders = [f for f in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(f)]
    matching = [f for f in folders if run_label_slug and run_label_slug in os.path.basename(f)]
    if not matching:
        raise FileNotFoundError(
            f"No run folder containing label '{run_label}' found under {output_root}"
        )
    return sorted(matching)[-1]


def load_run_data(run_folder: str) -> pd.DataFrame:
    annual_path = os.path.join(run_folder, "annual_aggregates.csv")
    if not os.path.exists(annual_path):
        raise FileNotFoundError(f"annual_aggregates.csv not found in {run_folder}")
    return load_csv(annual_path).sort_values('year')


def create_batch_plots_dir(output_root: str) -> str:
    plots_dir = os.path.join(output_root, 'batch_plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _get_series_for_variable(df: pd.DataFrame, x_col: str, y_col: str):
    if y_col == 'co2_emitted_tons_per_capita':
        if 'emissions_per_capita_tons' in df.columns:
            return df[x_col], df['emissions_per_capita_tons']
        if 'emissions_per_capita_kg_co2' in df.columns:
            return df[x_col], df['emissions_per_capita_kg_co2'] / 1000.0
        return None, None
    if y_col == 'emissions_avoided_tons_per_capita':
        if 'emissions_avoided_per_capita' in df.columns:
            return df[x_col], df['emissions_avoided_per_capita'] / 1000.0
        return None, None
    if y_col in df.columns:
        return df[x_col], df[y_col]
    return None, None


def plot_batch_comparison(run_data: List[Tuple[str, pd.DataFrame]],
                          x_col: str, y_col: str,
                          title: str, filename: str,
                          plots_dir: str) -> str:
    plt.figure(figsize=(10, 6))
    plotted = False
    for label, df in run_data:
        x_series, y_series = _get_series_for_variable(df, x_col, y_col)
        if x_series is not None and y_series is not None:
            plt.plot(x_series, y_series, marker='o', linewidth=2, label=label)
            plotted = True
    if not plotted:
        return ''
    plt.title(title)
    plt.xlabel(x_col.capitalize())
    if y_col == 'co2_emitted_tons_per_capita':
        plt.ylabel('CO2 Emissions per Capita (tons)')
    elif y_col == 'emissions_avoided_tons_per_capita':
        plt.ylabel('CO2 Emissions Avoided (tons per capita)')
    else:
        plt.ylabel(y_col.replace('_', ' ').capitalize())
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    output_path = os.path.join(plots_dir, filename)
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_batch_for_config(config_file: str, output_root: str):
    configs = load_config_file(config_file)
    plots_dir = create_batch_plots_dir(output_root)
    run_data = []

    for config in configs:
        run_label = config.get('run_label')
        if not run_label:
            run_label = config.get('policy', 'unknown_run')
            print(f"Warning: config missing run_label, using fallback label '{run_label}'")

        run_folder = find_run_folder_by_label(output_root, run_label)
        #print(f"Found run folder for label '{run_label}': {run_folder}")
        annual_df = load_run_data(run_folder)
        run_data.append((run_label, annual_df))

    saved_plots = []
    # time series to compare across runs
    comparison_vars = [
        'lce_share_percent',
        'action_1_count',
        'action_2_count',
        'action_3_count',
        'total_energy_saved_kwh',
        'total_emissions_avoided_kg_co2',
        'total_investment',
        'co2_emitted_tons_per_capita'
    ]

    for var in comparison_vars:
        if var == 'co2_emitted_tons_per_capita':
            supported = any('emissions_per_capita_kg_co2' in df.columns for _, df in run_data)
            title = 'CO2 Emissions per Capita (tons) Across Runs'
        elif var == 'emissions_avoided_tons_per_capita':
            supported = any('emissions_avoided_per_capita' in df.columns for _, df in run_data)
            title = 'CO2 Emissions Avoided per Capita (tons) Across Runs'
        else:
            supported = any(var in df.columns for _, df in run_data)
            title = f"{var.replace('_', ' ').capitalize()} Across Runs"

        if not supported:
            continue

        filename = f"comparison_{var}.png"
        plot_path = plot_batch_comparison(run_data, 'year', var, title, filename, plots_dir)
        if plot_path:
            saved_plots.append(plot_path)

    plt.show()

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
        print("No config file provided for batch plotting. Please provide a config file with --config-file to generate batch comparison plots.")

if __name__ == '__main__':
    main()

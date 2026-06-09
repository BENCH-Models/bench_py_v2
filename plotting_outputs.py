"""Plot saved BENCH model outputs for a selected run folder."""

import argparse
import glob
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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
    
    return output_path


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def plot_for_run(run_folder: str):
    plots_dir = create_plots_dir(run_folder)
    print(f"Loading run data from: {run_folder}")

    annual_path = os.path.join(run_folder, "annual_aggregates.csv")
    if not os.path.exists(annual_path):
        raise FileNotFoundError(f"annual_aggregates.csv not found in {run_folder}")

    annual_df = load_csv(annual_path).sort_values('year')
    saved_plots = []

    if 'lce_share_percent' in annual_df.columns:
        saved_plots.append(safe_plot(
            annual_df, 'year', 'lce_share_percent',
            'LCE Share Over Time', 'lce_share_percent.png', plots_dir
        ))

    action_columns = [col for col in ['action_1_count', 'action_2_count', 'action_3_count'] if col in annual_df.columns]
    if action_columns:
        saved_plots.append(plot_multi_series(
            annual_df, 'year', action_columns,
            'Action Counts Over Time', 'action_counts.png', plots_dir
        ))

    metric_columns = [col for col in ['total_energy_saved_kwh', 'total_emissions_avoided_kg_co2', 'total_investment'] if col in annual_df.columns]
    if metric_columns:
        saved_plots.append(plot_multi_series(
            annual_df, 'year', metric_columns,
            'Energy, Emissions, and Investment Over Time', 'energy_emissions_investment.png', plots_dir
        ))

    trajectory_files = glob.glob(os.path.join(run_folder, 'trajectory_*.csv'))
    for trajectory_path in trajectory_files:
        variable = Path(trajectory_path).stem.replace('trajectory_', '')
        trajectory_df = load_csv(trajectory_path)
        if 'year' in trajectory_df.columns and variable in trajectory_df.columns:
            saved_plots.append(safe_plot(
                trajectory_df, 'year', variable,
                f'Trajectory: {variable}', f'trajectory_{variable}.png', plots_dir
            ))

    return saved_plots


def build_parser():
    parser = argparse.ArgumentParser(description='Plot BENCH simulation outputs for a run folder.')
    parser.add_argument(
        '--run', '-r',
        help='The run folder name inside the output directory. If omitted, the latest run is used.'
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

    run_folder = find_run_folder(args.output_root, args.run)
    plot_for_run(run_folder)
    plt.show()

if __name__ == '__main__':
    main()

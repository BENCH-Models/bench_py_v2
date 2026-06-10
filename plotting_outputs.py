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
    elif y_col in df.columns:
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

    # Read all subdirectories currently available inside the batch session folder
    all_subfolders = [f for f in glob.glob(os.path.join(output_root, "*")) if os.path.isdir(f)]
    
    for config in configs:
        base_label = config.get('run_label')
        if not base_label:
            continue
            
        seed_dfs = []
        
        # Filter directories that match our structural scenario flag label (e.g. '_FDC' or '_Baseline')
        config_matched_folders = []
        for folder in all_subfolders:
            folder_name = os.path.basename(folder)
            if folder_name.endswith(f"_{base_label}") or f"_{base_label}_" in folder_name or folder_name == base_label:
                config_matched_folders.append(folder)
        
        # Sort folders chronologically by name so seed ordering remains intact
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
    
    for label, dfs in batch_data:
        combined_matrix = []
        x_values = None
        
        for df in dfs:
            y_series = _extract_variable_series(df, y_col)
            if x_col in df.columns and y_series is not None:
                df_sorted = df.sort_values(by=x_col)
                x_values = df_sorted[x_col].values
                # Re-extract sorted arrays securely
                y_series_sorted = _extract_variable_series(df_sorted, y_col)
                combined_matrix.append(y_series_sorted)
                
        if not combined_matrix:
            continue
            
        data_matrix = np.array(combined_matrix)
        n_samples = data_matrix.shape[0]
        
        # Compute mean trajectory
        mean_line = np.mean(data_matrix, axis=0)
        
        line, = plt.plot(x_values, mean_line, label=f"{label} (N={n_samples})", marker='o', linewidth=2, markersize=4)
        color = line.get_color()
        plotted = True
        
        # Apply standard normal distribution error properties for large sample data pools (N >= 100)
        if n_samples > 1:
            std_dev = np.std(data_matrix, axis=0, ddof=1)
            standard_error = std_dev / np.sqrt(n_samples)
            margin_error = 1.96 * standard_error
            
            lower_bound = mean_line - margin_error
            upper_bound = mean_line + margin_error
            
            plt.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.08)

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


def plot_batch_for_config(config_file_path: str, output_root: str) -> List[str]:
    from utils.config_loader import load_config_file
    print(f"Loading batch configurations from: {config_file_path}")
    configs = load_config_file(config_file_path)
    
    batch_data = find_run_data_for_config(output_root, configs)
    if not batch_data:
        print("No run simulation data folders located for plotting.")
        return []
        
    plots_dir = create_plots_dir(output_root)
    saved_plots = []
    
    variables_to_plot = [
        'lce_share_percent',
        'action_1_count',
        'action_2_count',
        'action_3_count',
        'total_emissions_avoided_kg_co2',
        'total_energy_saved_kwh',
        'co2_emitted_tons_per_capita',
        'emissions_avoided_tons_per_capita'
    ]
    
    for var in variables_to_plot:
        # Check if the target variable or its supporting fallback column is present
        if var == 'co2_emitted_tons_per_capita':
            supported = any(('emissions_per_capita_kg_co2' in df.columns or 'emissions_per_capita_tons' in df.columns) for _, dfs in batch_data for df in dfs)
            title = 'CO2 Emissions per Capita (tons) (95% CI)'
        elif var == 'emissions_avoided_tons_per_capita':
            supported = any('emissions_avoided_per_capita' in df.columns for _, dfs in batch_data for df in dfs)
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
            
            variables = [
                ('lce_share_percent', "LCE Consumption Share (%)"),
                ('action_1_count', "Action 1 Counts"),
                ('action_2_count', "Action 2 Counts"),
                ('action_3_count', "Action 3 Counts"),
                ('total_emissions_avoided_kg_co2', "Avoided Emissions (kg CO2)"),
                ('total_energy_saved_kwh', "Saved Energy (kWh)")
            ]
            
            for var, title in variables:
                if var in df.columns:
                    safe_plot(df, 'year', var, title, f"summary_{var}.png", plots_dir)
            print(f"✓ Individual plots written into: {plots_dir}")
        except Exception as e:
            print(f"Error executing fallback plotting: {e}")
            return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
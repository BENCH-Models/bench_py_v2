"""
BENCH Model - Main Entry Point
Behavioral Energy Consumption Household Model in Pure Python

Usage:
    python main.py
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model.bench_model import BENCHModel
from plotting_outputs import plot_batch_for_config
from utils.config_loader import load_config_file, normalize_run_config
from utils.constants import DEFAULT_LEARNING_TYPE, OUTPUT_DIR, NUMBER_SEED_RUNS, VERBOSE


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run BENCH model simulations from a configuration file for sensitivity analysis."
    )
    parser.add_argument(
        "--config-file", "-c",
        help="Path to a JSON or YAML configuration file defining one or more model runs."
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate plots for each completed run."
    )
    parser.add_argument(
        "--base-path", "-b",
        default=str(project_root),
        help="Project root path for data and output directories."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output during simulations."
    )
    return parser


def run_model(config: dict, base_path: str, output_root: str = None, seed: int = None) -> bool:
    # Explicitly ensure that run_label is provided, fallback to configuration metadata if missing
    label = config.get("run_label")
    if not label:
        label = f"{config.get('scenario')}_{config.get('policy')}_{config.get('learning_type')}"

    model = BENCHModel(
        case_study=config.get("case_study", "Netherlands-Overijssel"),
        scenario=config.get("scenario", "Ref_SSP2"),
        policy=config.get("policy", "Ref"),
        learning_type=config.get("learning_type", DEFAULT_LEARNING_TYPE),
        run_label=label,  # Explicitly pass down the configuration key label
        base_path=base_path,
        output_root=output_root,
        seed=seed,  # Pass the random seed to BENCHModel
    )
    model.debug = config.get("debug", False)
    
    if VERBOSE:
        print(f"\n=== RUNNING: {model.run_id if hasattr(model, 'run_id') else config.get('run_label', label)} (Seed: {seed}) ===")
    
    success = model.run()
    if not success:
        print(f"✗ Run failed for seed {seed}")
        return False

    # Get summary safely using the implementation present in BENCHModel or its components
    summary = model.get_summary()
    if VERBOSE and summary:
        print("\nSEED RUN CUMULATIVE RESULTS:")
        print("-" * 60)
        print(f"Total Investment: €{summary.get('total_investment', 0):,.2f}")
        print(f"Total Energy Saved: {summary.get('total_energy_saved', 0):,.0f} kWh")
        print(f"Total Emissions Avoided: {summary.get('total_emissions_avoided', 0):,.0f} kg CO2")
        print(f"Total Actions: {summary.get('actions_cumulative', 0):,.0f}")

        print("\nExporting results...")
        
    files = model.export_results()
    return True


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.config_file:
        configs = load_config_file(args.config_file)
        if not configs:
            print(f"No valid configurations found in {args.config_file}")
            return 1
    else:
        raise ValueError("No configuration file provided. Use --config-file to specify a JSON or YAML config.")

    batch_output_root = None
    if args.config_file:
        config_name = Path(args.config_file).stem
        batch_output_root = os.path.join(
            args.base_path,
            OUTPUT_DIR,
            f"{config_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        print(f"Batch output root: {batch_output_root}")

    all_success = True
    for index, config in enumerate(configs, start=1):
        config_label = config.get('run_label', 'Unnamed')
        print(f"\n=== Configuration {index}/{len(configs)}: {config_label} ===")
        config["debug"] = config.get("debug", args.debug)
        
        # Nested execution loop running NUMBER_SEED_RUNS times for the exact same parameters
        for seed_idx in range(NUMBER_SEED_RUNS):
            print(f" -> Executing Seed Run {seed_idx + 1}/{NUMBER_SEED_RUNS} (Seed: {seed_idx})")
            success = run_model(config, args.base_path, output_root=batch_output_root, seed=seed_idx)
            if not success:
                all_success = False

    if args.plot:
        if not args.config_file:
            print("Plotting requires --config-file when using --plot from main.py")
        else:
            plot_batch_for_config(args.config_file, batch_output_root)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
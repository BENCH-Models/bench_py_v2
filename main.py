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
from utils.constants import DEFAULT_LEARNING_TYPE, OUTPUT_DIR


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


def run_model(config: dict, base_path: str, output_root: str = None) -> bool:
    model = BENCHModel(
        case_study=config.get("case_study", "Netherlands-Overijssel"),
        scenario=config.get("scenario", "Ref_SSP2"),
        policy=config.get("policy", "Ref"),
        learning_type=config.get("learning_type", DEFAULT_LEARNING_TYPE),
        run_label=config.get("run_label"),
        base_path=base_path,
        output_root=output_root,
    )
    model.debug = config.get("debug", False)

    print(f"\n=== RUNNING: {model.run_id} ===")
    success = model.run(verbose=True)
    if not success:
        print(f"✗ Run failed: {model.run_id}")
        return False

    summary = model.get_summary()
    print("\nFINAL CUMULATIVE RESULTS:")
    print("-" * 60)
    print(f"Total Investment: €{summary['total_investment']:,.2f}")
    print(f"Total Energy Saved: {summary['total_energy_saved']:,.0f} kWh")
    print(f"Total Emissions Avoided: {summary['total_emissions_avoided']:,.0f} kg CO2")
    print(f"Total Actions: {summary['actions_cumulative']:,.0f}")

    print("\nExporting results...")
    files = model.export_results()
    #print(f"Results exported to: {model.run_output_dir}")
    #print(f"Summary files: {len(files)}")

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
        configs = [
            normalize_run_config({
                "case_study": "Netherlands-Overijssel",
                "scenario": "Ref_SSP2",
                "policy": "Carbon price pressure-100",
                "learning_type": "Fast adaptation",
                "debug": args.debug,
            })
        ]



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
        print(f"\n=== Configuration {index}/{len(configs)} ===")
        config["debug"] = config.get("debug", args.debug)
        success = run_model(config, args.base_path, output_root=batch_output_root)
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

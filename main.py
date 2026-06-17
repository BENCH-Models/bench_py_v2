"""
BENCH Model - Main Entry Point (Parallelized with Joblib)
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
from joblib import Parallel, delayed

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model.bench_model import BENCHModel
from plotting_outputs import plot_batch_for_config
from utils.config_loader import load_config_file, normalize_run_config
from utils.constants import DEFAULT_LEARNING_TYPE, OUTPUT_DIR, NUMBER_SEED_RUNS, VERBOSE


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run BENCH model simulations from a configuration file in parallel using joblib."
    )
    parser.add_argument(
        "--config-file", "-c",
        help="Path to a JSON or YAML configuration file defining one or more model runs."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation after simulations complete."
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
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=-1,
        help="Number of CPU cores to utilize. -1 utilizes all available cores."
    )
    return parser


def run_single_job(config: dict, base_path: str, batch_output_root: str, seed: int) -> bool:
    """Target function executed across parallel workers."""
    base_label = config.get("run_label")
    if not base_label:
        base_label = f"{config.get('scenario')}_{config.get('policy')}_{config.get('learning_type')}"

    # CRITICAL FIX: Append the seed directly to the runtime label configuration.
    # This guarantees that parallel worker nodes running within the same second
    # generate entirely separate folder paths on the Windows file system.
    unique_parallel_label = f"{base_label}_seed_{seed}"

    try:
        model = BENCHModel(
            case_study=config.get("case_study"),
            scenario=config.get("scenario"),
            policy=config.get("policy"),
            learning_type=config.get("learning_type"),
            run_label=unique_parallel_label,
            base_path=base_path,
            output_root=batch_output_root,
            seed=seed,
            carbon_price_awareness=config.get("carbon_price_awareness"),
            satisfaction_regret=config.get("satisfaction_regret")
        )
        model.debug = config.get("debug", False)
        
        # Run silently to avoid scrambled multi-core print overlapping
        success = model.run()
        if not success:
            print(f"✗ Run failed for config '{base_label}' on seed {seed}")
            return False

        # Get summary safely using the implementation present in BENCHModel or its components
        summary = model.get_summary()
        if VERBOSE and summary:
            print(f"\n=== RESULTS: {base_label} (Seed: {seed}) ===")
            print("-" * 60)
            print(f"Total Investment: €{summary.get('total_investment', 0):,.2f}")
            print(f"Total Energy Saved: {summary.get('total_energy_saved', 0):,.0f} kWh")
            print(f"Total Emissions Avoided: {summary.get('total_emissions_avoided', 0):,.0f} kg CO2")
            print(f"Total Actions: {summary.get('actions_cumulative', 0):,.0f}")
            
        model.export_results()
        return True

    except Exception as e:
        print(f"✗ Exception occurred on Configuration '{base_label}' | Seed {seed}: {e}")
        return False


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

    # Build the linear task parameter combination list.
    # Upgrade 2+3: each scenario can declare its own `runs` count; falls back to
    # the global NUMBER_SEED_RUNS constant when omitted.
    tasks = []
    for config in configs:
        config["debug"] = config.get("debug", args.debug)
        n_runs = int(config.get("runs", NUMBER_SEED_RUNS))
        for seed_idx in range(n_runs):
            tasks.append((config, args.base_path, batch_output_root, seed_idx))

    print(f"\nInitializing joblib Parallel Pool using n_jobs={args.workers}...")
    print(f"Total stochastic runs to process: {len(tasks)}\n")

    # Execute jobs concurrently with automated joblib tracking status output
    #args.workers
    results = Parallel(n_jobs=args.workers, verbose=10)(
        delayed(run_single_job)(cfg, bp, out, sd) for cfg, bp, out, sd in tasks
    )

    all_success = all(results)

    if not args.no_plot and batch_output_root:
        print("\nAll simulations completed. Beginning stochastic batch aggregation plots...")
        plot_batch_for_config(args.config_file, batch_output_root)

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
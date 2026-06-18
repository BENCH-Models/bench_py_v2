"""
Sobol sensitivity analysis for the BENCH model.

Usage:
    python -m sensitivity.run_sa --base-path . --output sa_results
    python -m sensitivity.run_sa --n-samples 64 --n-jobs 4

Outputs:
    sa_results/
        sobol_S1.csv    â€” first-order indices
        sobol_ST.csv    â€” total-order indices
        sobol_S2.csv    â€” second-order indices
        sobol_S1_plot.png
        sobol_ST_plot.png
"""

import argparse
import contextlib
import os
import types
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from SALib.sample import sobol as sobol_sampler
    from SALib.analyze import sobol as sobol_analyzer
except ImportError:
    raise SystemExit(
        "SALib is required for sensitivity analysis. Install it with:\n"
        "  pip install SALib"
    )

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None  # fallback to serial


# â”€â”€ Parameter space definition â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Each entry: (module_path, attribute_name, lower_bound, upper_bound)
# Kept to parameters that are clearly calibration choices (not fixed constants).

SA_PROBLEM = {
    "names": [
        "learn_awareness_fast",
        "learn_awareness_slow",
        "learn_per_nab_pbc",
        "learn_su_nor",
        "regret_per_nab_pbc",
        "regret_su_nor",
        "behavioral_cap",
        "pv_annual_cost",
        "pv_energy_output",
        "conservation_rate",
    ],
    "bounds": [
        [1.02, 1.10],   # learn_awareness_fast
        [1.01, 1.05],   # learn_awareness_slow
        [1.02, 1.10],   # learn_per_nab_pbc
        [1.02, 1.12],   # learn_su_nor
        [0.90, 0.99],   # regret_per_nab_pbc
        [0.93, 0.99],   # regret_su_nor
        [5.50, 6.90],   # behavioral_cap
        [300.0, 700.0], # pv_annual_cost (â‚¬/year)
        [1200, 2200],   # pv_energy_output (kWh/year)
        [0.30, 0.70],   # conservation_rate
    ],
    "num_vars": 10,
}


@contextlib.contextmanager
def _patch_params(param_dict: Dict[str, Any]):
    """
    Context manager that temporarily overrides constants in model.parameters
    for a single model run, then restores original values.
    """
    import model.parameters as const_mod

    _mapping = {
        "learn_awareness_fast": (const_mod, "LEARN_AWARENESS_RATE_FAST"),
        "learn_awareness_slow": (const_mod, "LEARN_AWARENESS_RATE_SLOW"),
        "learn_per_nab_pbc":    (const_mod, "LEARN_PER_NAB_PBC_RATE"),
        "learn_su_nor":         (const_mod, "LEARN_SU_NOR_RATE"),
        "regret_per_nab_pbc":   (const_mod, "REGRET_PER_NAB_PBC_RATE"),
        "regret_su_nor":        (const_mod, "REGRET_SU_NOR_RATE"),
        "behavioral_cap":       (const_mod, "BEHAVIORAL_CAP"),
        "pv_annual_cost":       (const_mod, "INVESTMENT_PV_ANNUAL_COST"),
        "pv_energy_output":     (const_mod, "INVESTMENT_PV_ENERGY_OUTPUT"),
        "conservation_rate":    (const_mod, "CONSERVATION_RATE"),
    }

    saved = {}
    try:
        for name, val in param_dict.items():
            mod, attr = _mapping[name]
            saved[(mod, attr)] = getattr(mod, attr)
            setattr(mod, attr, val)
        yield
    finally:
        for (mod, attr), orig in saved.items():
            setattr(mod, attr, orig)


def _run_single(param_vector: np.ndarray, param_names: List[str],
                base_path: str, seed: int) -> Dict[str, float]:
    """
    Run one BENCH model instance with given parameter vector.
    Returns a dict of scalar outputs used as SA response variables.
    """
    from model.bench_model import BENCHModel

    param_dict = dict(zip(param_names, param_vector))

    with _patch_params(param_dict):
        model = BENCHModel(
            case_study="Netherlands-Overijssel",
            scenario="Ref_SSP2",
            policy="Ref",
            learning_type="Fast adaptation",
            base_path=base_path,
            seed=seed,
            carbon_price_awareness=False,
            satisfaction_regret=True,
        )
        model.run()

    # Extract scalar outputs from final year stats
    final = model.statistics.annual_stats.get(2030, {})
    return {
        "green_share":            final.get("green_share_percent", 0.0),
        "action_1_count":         final.get("action_1_count", 0.0),
        "action_3_count":         final.get("action_3_count", 0.0),
        "avg_awareness":          final.get("avg_awareness", 0.0),
        "co2_per_capita":         final.get("co2_emitted_tons_per_capita", 0.0),
        "total_energy_saved_kwh": final.get("total_energy_saved_kwh", 0.0),
    }


def run_sobol_analysis(base_path: str, n_samples: int = 64,
                       seed: int = 42, n_jobs: int = 1,
                       output_dir: str = "sa_results") -> None:
    """
    Generate Sobol sample, run model for each, compute and save indices.

    Args:
        base_path:  project root (where data/ lives)
        n_samples:  base sample size N â€” total runs = N*(D+2) where D=num_vars
        seed:       RNG seed for sampling
        n_jobs:     parallel workers (-1 = all cores via joblib)
        output_dir: directory to write results
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating Sobol sample (N={n_samples}, D={SA_PROBLEM['num_vars']})â€¦")
    sample = sobol_sampler.sample(SA_PROBLEM, n_samples, calc_second_order=True,
                                  seed=seed)
    n_runs = sample.shape[0]
    print(f"Running {n_runs} model evaluationsâ€¦")

    def _task(i):
        return _run_single(sample[i], SA_PROBLEM["names"], base_path, seed=i)

    if Parallel is not None and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_task)(i) for i in range(n_runs)
        )
    else:
        results = [_task(i) for i in range(n_runs)]

    output_keys = list(results[0].keys())

    # Compute and save Sobol indices for each output variable
    all_S1, all_ST = [], []
    for key in output_keys:
        Y = np.array([r[key] for r in results])
        Si = sobol_analyzer.analyze(SA_PROBLEM, Y, calc_second_order=True,
                                    print_to_console=False)
        s1_row = dict(zip(SA_PROBLEM["names"], Si["S1"]))
        s1_row["output"] = key
        all_S1.append(s1_row)

        st_row = dict(zip(SA_PROBLEM["names"], Si["ST"]))
        st_row["output"] = key
        all_ST.append(st_row)

    pd.DataFrame(all_S1).set_index("output").to_csv(
        os.path.join(output_dir, "sobol_S1.csv"))
    pd.DataFrame(all_ST).set_index("output").to_csv(
        os.path.join(output_dir, "sobol_ST.csv"))

    print(f"Sobol indices written to {output_dir}/")
    _plot_indices(all_S1, all_ST, SA_PROBLEM["names"], output_dir)


def _plot_indices(s1_rows, st_rows, param_names, output_dir):
    """Bar charts of S1 and ST for each output variable."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available â€” skipping SA plots")
        return

    s1_df = pd.DataFrame(s1_rows).set_index("output")[param_names]
    st_df = pd.DataFrame(st_rows).set_index("output")[param_names]

    for df, label, fname in [
        (s1_df, "First-order (S1)", "sobol_S1_plot.png"),
        (st_df, "Total-order (ST)", "sobol_ST_plot.png"),
    ]:
        n_out = len(df)
        fig, axes = plt.subplots(1, n_out, figsize=(4 * n_out, 5), squeeze=False)
        for ax, (output_var, row) in zip(axes[0], df.iterrows()):
            ax.barh(param_names, row.values, color="#2E86AB", alpha=0.8)
            ax.set_title(output_var, fontsize=9, fontweight="bold")
            ax.set_xlabel(label, fontsize=9)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        fig.suptitle(f"Sobol {label} Indices", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=200)
        plt.close()
    print(f"SA plots written to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Sobol SA for BENCH v2 model")
    parser.add_argument("--base-path", default=".", help="Project root directory")
    parser.add_argument("--n-samples", type=int, default=64,
                        help="Sobol base N (total runs = N*(D+2))")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Parallel workers (requires joblib; -1 = all cores)")
    parser.add_argument("--output", default="sa_results",
                        help="Output directory for results")
    args = parser.parse_args()

    run_sobol_analysis(
        base_path=args.base_path,
        n_samples=args.n_samples,
        seed=args.seed,
        n_jobs=args.n_jobs,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

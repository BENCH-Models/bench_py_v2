# BENCH_v.2_py — Behavioral Energy Consumption Household Model

This doesnt quite replicate the results from the Netlogo but it is useful for understand how the model works if you arent familiar with Netlogo.

A Python re-implementation of the [BENCH NetLogo model](netlogo/BENCH_v.02.nlogox). Simulates ~3,000 household agents making annual energy decisions (solar PV investment, conservation, fuel switching) from 2015–2030 in the Netherlands (Overijssel).

Agents are characterised by income group, energy consumption, and a set of behavioral factors — awareness, motivation, perceived behavioral control (PBC), personal and social norms — that determine whether and what actions they take each year. Social learning spreads behavioral change through a spatial neighbourhood grid. Multiple policy scenarios (reference, carbon price) and learning types (none, fast, slow adaptation) can be configured and run in parallel across many random seeds.

---

## Project structure

```
BENCH_py_v.2/
│
├── model/
│   ├── parameters.py      # All constants (scales, thresholds, investment costs, etc.)
│   ├── population.py      # Population class — vectorised numpy arrays holding all agent state
│   ├── vectorized.py      # All behavioural logic: awareness, motivation, utility, decisions,
│   │                      #   social learning, regret — operates on the Population arrays
│   ├── bench_model.py     # Orchestrator: initialises population, steps through 2015–2030,
│   │                      #   calls vectorized functions in order each year
│   ├── loader.py          # Reads price CSVs and CGE economic parameters from data/
│   ├── statistics.py      # Aggregates per-year population statistics into dicts
│   ├── output.py          # Writes annual_aggregates.csv, trajectory CSVs, etc.
│   └── config_loader.py   # Parses YAML scenario configs
│
├── configs/               # YAML files defining which scenarios/policies/seeds to run
│   ├── bench_v2_scenarios_base.yaml
│   ├── bench_v2_scenarios_base_with_cp_awareness.yaml
│   └── ...
│
├── data/                  # Input data (prices, household baseline, CGE parameters)
├── netlogo/               # Original NetLogo model (reference only)
├── sensitivity/           # SALib-based sensitivity analysis scripts
│
├── main.py                # CLI entry point — runs configs in parallel via joblib
├── plotting_outputs.py    # All post-run plotting functions
└── pyproject.toml
```

### Key files in plain English

| File | What it does |
|---|---|
| `model/population.py` | Defines the `Population` dataclass — a collection of numpy arrays, one entry per agent, holding everything from income and consumption to awareness, norms, action flags, and emissions avoided. |
| `model/vectorized.py` | The behavioural engine. Every step these functions run in sequence: update awareness → update motivation → compute delta (PBC→consideration factor) → calculate budgets → compute expected utility → agents decide actions → compute outcomes (energy saved, CO2 avoided) → social learning → regret. All operations are vectorised across the full population at once (no per-agent loop). |
| `model/bench_model.py` | Wires everything together. Loads data, creates the `Population`, then calls the functions in `vectorized.py` for each year 2015–2030. |
| `model/statistics.py` | After each year, computes population-level summaries (green share, action counts, awareness bins, emissions, etc.) and stores them for export. |
| `model/parameters.py` | Single source of truth for all numeric constants — behavioral scale (0–7), guilt threshold (5.21), delta levels, investment cost and output, CO2 factors, etc. Change a parameter here and it propagates everywhere. |
| `configs/*.yaml` | Each YAML file lists one or more scenario blocks. Each block specifies `case_study`, `scenario`, `policy`, `learning_type`, and `runs` (number of random seeds). `main.py` reads this file and runs every block. |
| `plotting_outputs.py` | Post-run plotting. Pass it a config file and output root; it reads the CSVs from every seed folder and produces figures (green share, emissions, awareness bins, cumulative actions, etc.). |

---

## Setup with uv

[uv](https://docs.astral.sh/uv/) is a fast Python package manager. If you don't have it:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then, from the project root:

```bash
# Create virtual environment and install dependencies
uv sync

# Activate (PowerShell)
.venv\Scripts\Activate.ps1
```

---

## Running a scenario

Point `main.py` at a config file:

```bash
python main.py --config-file configs/bench_v2_scenarios_base.yaml
```

This runs every scenario block in the YAML, each with the specified number of seeds, in parallel. Outputs land in a timestamped folder inside `output/`.

**Skip plots** (faster, useful for debugging):
```bash
python main.py --config-file configs/bench_v2_scenarios_base.yaml --no-plot
```


### Config file format

```yaml
- case_study: Netherlands-Overijssel
  scenario: Ref_SSP2
  policy: Ref
  learning_type: No learning
  run_label: Baseline
  carbon_price_awareness: false
  satisfaction_regret: false
  runs: 64
```

Available policies: `Ref`, `Carbon price pressure-10/25/50/100`  
Available learning types: `No learning`, `Slow adaptation`, `Fast adaptation`

### Outputs

Each run folder contains:
- `annual_aggregates.csv` — one row per year, all population-level statistics
- `actions_by_income_group.csv` — per-year breakdown by income group (1–7)
- `trajectory_*.csv` — single-variable time series (green share, emissions, etc.)
- `plots/` — auto-generated figures

---


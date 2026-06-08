# BENCH Python Model - Project Structure

## Overview

This is a complete Python implementation of the BENCH (Behavioral Energy Consumption Household) agent-based model, originally created in NetLogo. It simulates household energy decisions from 2015-2030 in the Netherlands and Spain, tracking adoption of renewable energy, energy conservation measures, and behavioral factors.

**Key Features:**
- Pure Python implementation (no Mesa or other ABM frameworks)
- 3,000-5,000+ household agents with individual attributes
- Behavioral decision-making based on knowledge, motivation, norms, and perceived behavioral control
- Three main action types: Solar PV investment, energy conservation, and switching to renewables
- Social learning through peer influence
- Complete market dynamics with multiple policy scenarios
- Comprehensive output and statistics export

## Project Structure

```
BENCH_py_v.2/
├── agents/                          # Household agent definitions
│   ├── __init__.py
│   └── household.py                 # Household class with all attributes & methods
│
├── model/                           # Core simulation engine
│   ├── __init__.py
│   ├── bench_model.py              # Main BENCHModel orchestrator
│   └── market.py                   # [Future] Market clearing mechanisms
│
├── data_loader/                     # Data input handling
│   ├── __init__.py
│   └── loader.py                   # DataLoader for CSV files
│
├── behavioral/                      # Behavioral decision-making modules
│   ├── __init__.py
│   ├── utility.py                  # Utility calculation engine
│   ├── decision_making.py          # Behavioral decision logic
│   └── learning.py                 # Social learning mechanisms
│
├── utils/                          # Utilities and tools
│   ├── __init__.py
│   ├── constants.py                # Global constants and parameters
│   ├── statistics.py               # Statistics aggregation
│   └── output.py                   # Results export to CSV
│
├── netlogo/                        # Original NetLogo files (reference)
│   ├── BENCH_v.02.nlogox          # Original NetLogo model
│   └── data/                       # Data files (shared with Python)
│       ├── household.csv
│       ├── cge-nl-ssp2-*.csv
│       ├── primes-nl-ref-prices.csv
│       ├── primes-spn-ref-prices.csv
│       └── ESP_*.* (GIS files)
│
├── output/                         # Simulation results (auto-generated)
│   ├── annual_aggregates.csv
│   ├── annual_household_data.csv
│   ├── actions_breakdown.csv
│   ├── summary_report.txt
│   └── trajectory_*.csv
│
├── main.py                         # Entry point - run simulation
├── requirements.txt                # Python dependencies
├── README_STRUCTURE.md             # This file
└── NETLOGO_STRUCTURE_AND_PYTHON_PLAN.md  # Original analysis & plan
```

## Files Description

### Core Modules

#### `agents/household.py`
Defines the `Household` class with:
- **Demographic attributes**: ID, income, age, income group (1-7), dwelling type
- **Energy attributes**: consumption, energy source (FF/LCE/SLCE)
- **Behavioral factors**: awareness, guilt, knowledge, norms, motivation, PBC
- **Actions**: investment, conservation, switching flags
- **Financial tracking**: investments, savings, switching benefits
- **Environmental**: emissions avoided
- **Methods**: awareness updates, motivation calculation, budget constraints, state export

#### `model/bench_model.py`
Main simulation engine with:
- **Initialization**: load data, create ~3,000 household agents
- **Annual step**: execute one year's simulation
- **Loop control**: 2015-2030 annual timesteps
- **Integration**: coordinates all behavioral modules
- **Results tracking**: aggregates statistics

#### `data_loader/loader.py`
Handles all data input:
- Reads household.csv (baseline population)
- Loads price scenarios (11 columns × 16 years)
- Loads CGE economic parameters
- Provides interface for model to access data
- Supports data filtering and statistics

#### `behavioral/utility.py`
Utility calculation engine:
- **normalize_budgets()**: Calibrate budget constraints across population
- **calculate_expected_utility()**: Compute utility using:
  - Budget/consumption combinations (z values)
  - Environmental preferences
  - Behavioral factors (guilt, motivation, PBC)
  - Formula: `UE = ((z_norm * (1-α)) + (e_norm * α)) * (1-δ) + ((K+M+pbc/7)*δ)`
- **calculate_all_utilities()**: Run for all action types
- **should_take_action()**: Utility threshold decision

#### `behavioral/decision_making.py`
Behavioral decision logic:
- **activate_knowledge()**: Update awareness from components
- **update_motivation()**: Calculate motivation from norms
- **consider_action()**: Determine PBC and delta factors
- **decide_action()**: Main decision logic dispatching to source-specific functions
- **calculate_energy_savings()**: Compute savings from actions
- **calculate_financial_outcomes()**: Investment and switching costs/benefits
- **calculate_emissions_avoided()**: CO2 reduction (0.5 kg/kWh)

#### `behavioral/learning.py`
Social learning mechanisms:
- **learn_from_peers()**: Influence from neighboring households
- **recall_memory()**: 2015 historical behavior initialization
- **update_memory()**: Store year experiences
- **update_satisfaction()**: Feedback from action outcomes
- **update_regret()**: Response to price changes
- **get_network_effect()**: Quantify peer influence

#### `utils/constants.py`
All global parameters:
- Model configuration (years, timesteps)
- Income groups and dwelling labels
- Energy flags and action types
- Behavioral scales (0-7)
- PBC thresholds and delta factors
- Investment parameters (€487.59/year, 1700 kWh output)
- File paths

#### `utils/statistics.py`
Population-level aggregation:
- **aggregate_population_stats()**: Overall metrics per year
- **aggregate_by_income_group()**: Breakdown by income (groups 1-7)
- **aggregate_by_dwelling_label()**: Breakdown by efficiency (labels 1-6)
- **store_annual_stats()**: Time-series tracking
- **get_cumulative_stats()**: Period totals
- **get_trajectory()**: Variable over time

#### `utils/output.py`
Results export:
- **export_annual_aggregates()**: Population-level CSV
- **export_household_actions()**: Household-level CSV per year
- **export_summary_report()**: Text summary
- **export_trajectory()**: Time series CSV
- **export_all_results()**: Complete results package

### Configuration Files

#### `requirements.txt`
```
pandas
numpy
```

#### `main.py`
Entry point script with example usage:
- Create BENCHModel instance
- Set case study, scenario, policy
- Run 16-year simulation
- Print summary
- Export results

## Running the Model

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Execution
```bash
python main.py
```

### Custom Configuration
Edit `main.py`:
```python
CASE_STUDY = "Netherlands-Overijssel"  # or "Spain-Navarre"
SCENARIO = "Ref_SSP2"
POLICY = "Ref"  # or "Carbon price pressure-25", etc.
```

### Programmatic Use
```python
from model.bench_model import BENCHModel

model = BENCHModel(
    case_study="Netherlands-Overijssel",
    scenario="Ref_SSP2",
    policy="Ref"
)

model.run()
files = model.export_results()
```

## Data Files

### Inputs (in `netlogo/data/`)
- **household.csv**: 2015 household baseline (columns: id, group id, income, consumption, lce user?, ..., many behavioral attributes)
- **cge-nl-ssp2-*.csv**: Economic parameters from CGE model
- **primes-nl-ref-prices.csv**: 16 rows × 11 columns
  - Row: years 2015-2030
  - Columns: different policy scenarios
- **ESP_*.shp/dbf/prj**: Geographic boundaries (not used in core dynamics)

### Outputs (in `output/`)
- **annual_aggregates.csv**: Population-level statistics per year
- **annual_household_data_YYYY.csv**: Individual household snapshots
- **actions_by_income_group.csv**: Action adoption by income
- **summary_report.txt**: Human-readable summary
- **trajectory_*.csv**: Time series for each key variable

## Key Model Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Start Year | 2015 | Baseline year |
| End Year | 2030 | Final simulation year |
| Timestep | 1 year | Annual steps |
| Households | ~3,000 | Population size |
| Income Groups | 7 | Stratification levels |
| Dwelling Labels | 6 | A-F energy efficiency |
| Energy Sources | 3 | FF, LCE, SLCE |
| Action Types | 3 | Investment, Conservation, Switching |
| PV Payback | 10 years | €487.59/year, 1700 kWh output |
| Conservation Rate | 50% | Max energy reduction |
| CO2 Factor | 0.5 kg/kWh | FF electricity emissions |
| Alpha (utility) | 0.5 | Budget vs environment weight |
| Guilt Threshold | 5.21 | Awareness level for guilt |
| PBC Threshold | 4.0 | Control level for consideration |

## Behavioral Decision Process (Per Year)

1. **Awareness Update**: Calculate average of knowledge, climate awareness, education
2. **Guilt Determination**: High if awareness > 5.21
3. **Motivation**: Average of personal norms + social norms
4. **Responsibility**: Triggered by high guilt + high motivation + high norms
5. **PBC Consideration**: Perceived behavioral control determines delta factor (0.2-0.7)
6. **Budget Constraints**: Calculate discretionary income for scenarios
7. **Normalization**: Scale budgets 0-1 using population max
8. **Utility Calculation**: Combine budget utility, environmental preference, behavioral factors
9. **Action Decision**: Choose action if utility exceeds household's actual utility + threshold
10. **Outcomes**: Calculate energy savings, investments, emissions avoided
11. **Learning**: Update from peer actions and outcomes
12. **Aggregation**: Compute population statistics

## Comparison with Original NetLogo

| Aspect | NetLogo | Python |
|--------|---------|--------|
| **Agent System** | Breed system | Class-based OOP |
| **Agent Iteration** | `ask households` | `for hh in households` |
| **Global Variables** | Model variables | BENCHModel attributes |
| **Random Numbers** | `random-float` | `random.uniform()`, `numpy.random` |
| **Data Loading** | CSV extension | Pandas DataFrames |
| **GIS Support** | GIS extension | Removed (not critical) |
| **Visualization** | Built-in plots | Matplotlib (external) |
| **Output** | File write commands | Pandas CSV export |
| **Language** | NetLogo (functional) | Python (OOP) |

## Extending the Model

### Add a New Behavioral Factor
1. Add attribute to `Household.__init__()` in `agents/household.py`
2. Create update method in behavioral module
3. Call in `BENCHModel.step()`

### Add a New Action Type
1. Define constants in `utils/constants.py`
2. Add utility calculation in `behavioral/utility.py`
3. Implement decision logic in `behavioral/decision_making.py`
4. Update aggregations in `utils/statistics.py`

### Add a New Policy Scenario
1. Update price data in `netlogo/data/primes-nl-ref-prices.csv`
2. Map policy name in `DataLoader.get_price_scenario_index()`
3. Add to `utils/constants.py` POLICIES list
4. Pass to `BENCHModel()` constructor

## Validation & Testing

To validate against original NetLogo model:
1. Run both models with identical case study, scenario, policy
2. Compare outputs:
   - LCE adoption trajectory
   - Action adoption curves
   - Energy savings cumulative
   - CO2 avoided cumulative
3. Results should converge (minor differences due to random number stream differences)

## License & Attribution

Original NetLogo model: Leila Niamir, University of Twente & IIASA
Python implementation: [Your Name/Organization]

## Contact & Support

For issues or questions about the Python implementation, refer to the structure documentation and inline comments in source files.

# BENCH NetLogo Model - Code Structure & Python Replication Plan

## 1. NETLOGO MODEL CODE STRUCTURE MAP

### 1.1 Model Overview
- **Model Name:** BENCH (Behavioral Energy Consumption Household) v0.2
- **Author:** Leila Niamir
- **Time Period:** 2015-2030 (annual timesteps)
- **Geographic Focus:** Netherlands (Overijssel) & Spain (Navarre) case studies
- **Core Mechanism:** Agent-based household energy decision-making model with behavioral components

### 1.2 Main Components

#### **A. AGENTS & BREEDS**
```
Households (breed: households, singular: household)
  - Directed Links: active-links, inactive-links (for social networks)
```

#### **B. DATA INPUTS (CSV files from data/ folder)**
```
household.csv              - Initial household characteristics (loaded at setup)
cge-nl-ssp2-h.csv         - CGE economic inputs for Netherlands 
cge-nl-ssp2-con.csv       - Consumption inputs
cge-nl-ssp2-alpha.csv     - Alpha parameter
primes-nl-ref-prices.csv  - Electricity prices (multiple scenarios)
primes-nl-ref-con.csv     - Consumption reference
primes-nl-ref-noncon.csv  - Non-residential consumption
ESP_adm4.*                - Shapefile data (geographic boundaries)
ESP_cities.*              - Shapefile data (city locations)
ESP_pol.*                 - Shapefile data (political boundaries)
```

#### **C. MAJOR PROCEDURES (Functions/Methods)**

**Initialization:**
1. `load-map` - Load GIS shapefiles (geographic visualization)
2. `show-population` - Display household agents spatially
3. `load-data` - Read CSV files into global variables
4. `setup` - Initialize household agents with attributes from CSV

**Main Loop (`go` procedure - runs annually 2015-2030):**
1. `recallmemory` - Recall historical behavior patterns (2015 only)
2. **Behavioral Decision-Making Chain:**
   - `cpinfo` - Compute cross-price information
   - `knowledge` - Calculate awareness level
   - `motivation` - Compute motivation for actions
   - `consideration_NAT` - Consider constraints (delta parameters 0.2-0.7)
3. **Utility Calculations:**
   - `normalize-1` - Normalize consumption/income for utilities
   - `utility_exp_NAT` - Calculate expected utility for each action
   - `update_utilities_NAT` - Update utility calculations
4. **Actions & Economics:**
   - `action` - Decide and execute household actions
   - `save` - Calculate energy/money savings
   - `emissions` - Calculate CO2 emissions avoided
   - `price` - Set exogenous energy prices
5. **Learning & Updating:**
   - `learn` - Social network learning
   - `update_heq` - Update household equilibrium
   - `update_memory` - Update memory variables
   - `update_data` - Update household-level data
6. **Reporting & Visualization:**
   - `debug` - Output debug info
   - Various `draw_*` procedures for plotting

#### **D. HOUSEHOLD ATTRIBUTES (Agent Variables)**

**Demographic:**
- `h_id` - Household ID
- `h_id_group` - Income group (1-7)
- `h_income` - Annual income
- `h_age` - Age

**Energy Consumption:**
- `h_q` - Electricity consumption (kWh/year)
- `flag?` - Energy source: 0=Grey(FF), 1=Green(LCE), 2=Super-Green(SLCE)
- `h_totalff`, `h_totallce`, `h_totalslce` - Consumption by type

**Behavioral Attributes:**
- `know` - Knowledge level (0-7)
- `cee_aw` - Climate change awareness (0-7)
- `ed_aw` - Education/awareness level
- `h_aware` - Average awareness
- `guilt` - Guilt feeling ("L" or "H")
- `K` - Normalized guilt factor (0-1)

**Norms & Attitudes:**
- `per_nab1`, `per_nab2`, `per_nab3` - Personal norms for 3 actions
- `su_nor1`, `su_nor2`, `su_nor3` - Social/subjective norms
- `h_motiv1`, `h_motiv2`, `h_motiv3` - Motivation levels
- `M1`, `M2`, `M3` - Normalized motivation (0-1)

**Perceived Behavioral Control:**
- `pbc1`, `pbc2`, `pbc3` - PBC for 3 actions (0-7 scale)
- `delta1`, `delta2`, `delta3` - Behavioral adjustment factors (0.2-0.7)

**Utilities:**
- `utility_exp_brown1/2/31`, `utility_exp_grey1/2/32`, `utility_exp_green1/2/3` - Expected utilities
- `utility_brown1/2/31`, `utility_grey1/2/32`, `utility_green1/2/3` - Actual utilities
- `z_brown1/2/3`, `z_grey1/2/3`, `z_green1/2` - Budget/consumption combinations

**Actions Taken:**
- `act1`, `act11`, `act12` - Investment (PV installation)
- `act2`, `act50`, `act21` - Conservation (efficiency)
- `act3`, `act31`, `act32` - Switching (to renewable)
- `hh_actions` - Action vector [6 elements]

**Financial:**
- `h_invest` - Money invested in PV (487.59 annually for 10 years)
- `h_invest_save` - PV energy production (kWh)
- `h_conserv` - Energy saved through conservation (50% of h_q)
- `h_conserv_p` - Money saved through conservation
- `h_switch` - Money saved/spent on switching

**Environmental:**
- `em_total` - CO2 emissions (kg)
- `em_avoided` - Emissions avoided (kg)

**Dwelling Characteristics:**
- `dw_el` - Dwelling energy label (1-6, A-F equivalent)
- `dw_st` - Dwelling type
- `Owner` - Property owner boolean

**Memory & Learning:**
- `influence_*` - Social influence variables
- Various experience tracking variables

#### **E. GLOBAL VARIABLES (OUTPUTS/AGGREGATES)**

**Population Statistics:**
- `n_households` - Total households
- `n_suppliers` - Number of suppliers (typically 1)

**Consumption Aggregates:**
- `h_totalff`, `h_totallce`, `h_totalslce` - Total consumption by type
- `consumption_total` - Total electricity consumption
- `h_lceshare` - Share of renewable consumption (%)

**Market Variables:**
- `m_p_grey` - Market price grey electricity (€/kWh)
- `m_p_brown` - Market price green electricity (€/kWh)
- `m_p_green` - Market price zero-carbon electricity
- `p_star_ff`, `p_star_lce`, `p_star_zero` - Prices after market clearing

**Action Counters (per year):**
- `action_1`, `action_2`, `action_31`, `action_32` - Counts by action type
- `action_total` - Total households taking action
- `action_1_cum`, `action_2_cum`, etc. - Cumulative counts

**Action Statistics (by income group 1-7):**
- `act1_inc1` to `act1_inc7` - Action 1 by income
- `act2_inc1` to `act2_inc7` - Action 2 by income
- `act3_inc1` to `act3_inc7` - Action 3 by income

**Action Statistics (by dwelling label A-F):**
- `act1_a` to `act1_f` - Action 1 by dwelling label
- `act2_a` to `act2_f` - Action 2 by dwelling label
- `act3_a` to `act3_f` - Action 3 by dwelling label

**Energy Savings:**
- `energy_sav` - Energy saved per year (kWh)
- `energy_sav_cum` - Cumulative energy saved (kWh)
- `conserv_total`, `conserv_cum` - Conservation money savings

**Emissions:**
- `em_sav` - Emissions avoided per year (kg CO2)
- `em_sav_cum` - Cumulative emissions avoided
- `co2_total`, `co2_percapita` - Total CO2 produced by households
- Multiple CO2 tracking variables by action type and income group

**Financial Aggregates:**
- `invest_total`, `invest_cum` - Investment amounts
- `switch_total`, `switch_cum` - Switching benefits
- `income_total`, `saving_total`, `consumption_total` - Household aggregates

**Awareness & Motivation (Population Level):**
- `aware_total` - Sum of household awareness
- `utility_exp_brown`, `utility_exp_grey` - Expected utilities

**Network Variables:**
- `ngb_*` - Various neighbor/network influence variables

**Time:**
- `year` - Current year (2015-2030)
- `n` - Timestep counter

---

## 2. PYTHON REPLICATION STRATEGY

### 2.1 Core Architecture

**Without Mesa (Pure Python OOP):**
```
project/
├── agents/
│   └── household.py          # Household agent class
├── model/
│   ├── bench_model.py        # Main simulation engine
│   └── market.py             # Market mechanics
├── data/
│   ├── loader.py             # CSV data loading
│   └── [existing CSV files]
├── behavioral/
│   ├── decision_making.py    # Behavioral decision logic
│   ├── utility.py            # Utility calculations
│   └── learning.py           # Learning mechanisms
├── utils/
│   ├── constants.py          # Global constants
│   ├── statistics.py         # Aggregation functions
│   └── output.py             # Results export
└── main.py                   # Entry point
```

### 2.2 Implementation Approach

**Phase 1: Foundation (Data & Agents)**
1. Create `Household` agent class with all attributes
2. Build `DataLoader` to read all CSV files
3. Implement agent initialization from household.csv
4. Create basic agent update methods

**Phase 2: Decision-Making Logic**
1. Implement behavioral factors (knowledge, motivation, norms)
2. Build utility calculation engine
3. Implement action decision logic

**Phase 3: Market & Economics**
1. Implement price setting logic
2. Add energy consumption tracking
3. Calculate savings/investments
4. Emissions calculations

**Phase 4: Simulation Loop & Output**
1. Build main loop (2015-2030 annual cycles)
2. Implement aggregation statistics
3. Add output/logging functionality
4. Validation against original outputs

### 2.3 Key Design Changes from NetLogo

| Aspect | NetLogo | Python | Change Rationale |
|--------|---------|--------|------------------|
| **Agent Storage** | Built-in breed system | List of Household objects | Standard OOP approach |
| **Global State** | Global variables | Model class attributes | Better encapsulation |
| **Agent Iteration** | `ask households` | `for household in model.households` | Standard iteration |
| **Links/Networks** | Link breeds | Dictionary/NetworkX optional | Simpler for current use |
| **Random Values** | `random-float`, `random` | `numpy.random`, `random.uniform` | Python standard libraries |
| **Time** | `ticks` counter | Explicit year variable | More explicit control |
| **Data Loading** | `csv` extension | `pandas.read_csv` | Standard Python approach |
| **GIS/Maps** | `gis` extension | Removed initially | Not critical for behavior |
| **Output** | File writing | CSV export via pandas | Standard Python |
| **Plotting** | Built-in graphing | matplotlib/seaborn | Can be added later |

### 2.4 Behavioral Simplifications (Optional - Keep Close to Original)

**What to Keep:**
- Complex decision-making with multiple behavioral factors
- Income-based stratification (7 groups)
- Dwelling energy labels (6 categories)
- Three types of actions (Investment, Conservation, Switching)
- Social learning mechanism
- Historical memory recall

**What Can Be Simplified:**
- Spatial visualization (maps) - replaced with abstract positioning
- Some edge-case calculations that appear dormant in NetLogo code
- Complex market clearing that isn't fully implemented in original

### 2.5 Data Structure Example (Household Agent)

```python
class Household:
    def __init__(self, h_id, h_group, h_income, h_q, flag, **kwargs):
        # Demographic
        self.h_id = h_id
        self.h_id_group = h_group  # 1-7 income groups
        self.h_income = h_income
        
        # Energy
        self.h_q = h_q  # Base consumption
        self.flag = flag  # 0=Grey, 1=Green, 2=Super-Green
        
        # Behavioral attributes
        self.knowledge = 0.0  # 0-7
        self.awareness = {'cee': 0, 'ed': 0}
        self.guilt = 'L'  # 'L' or 'H'
        self.norms = {'personal': [0,0,0], 'social': [0,0,0]}
        self.motivation = [0, 0, 0]
        self.pbc = [0, 0, 0]  # Perceived behavioral control
        
        # Utilities
        self.utility_expected = {}
        self.utility_actual = {}
        
        # Actions (boolean flags)
        self.actions = [False, False, False]  # [invest, conserve, switch]
        self.actions_cumulative = [0, 0, 0]
        
        # Financial
        self.investment = 0
        self.savings = 0
        
        # Memory
        self.memory = {}
        self.influences = {}
    
    def update_behavioral_factors(self, global_params):
        """Calculate knowledge, motivation, consideration"""
        
    def calculate_utilities(self, prices, market_state):
        """Calculate expected and actual utilities for each action"""
        
    def make_decision(self):
        """Decide which actions to take based on utilities"""
        
    def update_results(self, prices):
        """Calculate energy/money savings and emissions"""
```

### 2.6 Model Class Structure

```python
class BENCHModel:
    def __init__(self, case_study, scenario, policy, config):
        self.case_study = case_study  # "Netherlands" or "Spain"
        self.scenario = scenario
        self.policy = policy
        self.year = 2015
        self.households = []
        self.prices = {}
        
    def initialize(self):
        """Load data and create agents"""
        
    def step(self):
        """Execute one year's simulation"""
        
    def run(self, end_year=2030):
        """Run full simulation"""
        
    def get_aggregates(self):
        """Calculate all global statistics"""
        
    def export_results(self, filepath):
        """Save results to CSV"""
```

---

## 3. EXPECTED OUTPUTS & REPLICATION VERIFICATION

### 3.1 Output Files Generated
1. **annual_household_data.csv** - Per-year household-level data
2. **annual_aggregates.csv** - Population-level statistics per year
3. **final_results.csv** - Summary statistics
4. **actions_breakdown.csv** - Actions by income/dwelling group

### 3.2 Verification Metrics (Compare Original vs Python)
- Total households: Must match input (typically 3,000-5,000)
- LCE adoption rate trajectory 2015-2030
- Energy savings cumulative
- CO2 avoided cumulative
- Investment distribution by income
- Action distribution: Investment/Conservation/Switching ratios


"""
Main BENCH Model simulation engine
Orchestrates the entire simulation workflow
"""

import os
import re
import math
import random
import datetime
from typing import Dict, List, Optional
from agents.household import Household
from data_loader.loader import DataLoader
from behavioral.utility import UtilityCalculator
from behavioral.decision_making import DecisionMaker
from behavioral.learning import LearningMechanism
from utils.statistics import StatisticsAggregator
from utils.output import ResultsExporter
import pandas as pd
from utils.constants import (
    CARBON_POLICY_TARGETS, EMISSIONS_FACTOR_GRAY, EMISSIONS_FACTOR_BROWN, EMISSIONS_FACTOR_GREEN, KG_TO_TONS,
    M_P_GREEN_BASE, M_P_BROWN_BASE, M_P_GREY_BASE, MODEL_START_YEAR, MODEL_END_YEAR,
    INCOME_GROUPS, DWELLING_LABELS,
    FLAG_NAMES, OUTPUT_DIR,
    DEFAULT_LEARNING_TYPE,
    VERBOSE,
    INVESTMENT_PV_ENERGY_OUTPUT,
    PRIMES_NL_PRICES_FILE
)


class BENCHModel:
    """
    Main BENCH Model implementation in pure Python.
    
    Manages:
    - Agent creation and lifecycle
    - Annual simulation loop (2015-2030)
    - Behavioral decision-making
    - Market dynamics
    - Results tracking and export
    """
    
    def __init__(self, 
                case_study: str, 
                scenario: str, 
                policy: str,
                learning_type: str = DEFAULT_LEARNING_TYPE,
                run_label: str = None,
                base_path: str = ".",
                output_root: str = None,
                seed: Optional[int] = None,
                carbon_price_awareness: bool = True,
                satisfaction_regret: bool = True
                ):
        # ... basic attribute assignments ...
        
        self.case_study = case_study
        self.scenario = scenario
        self.policy = policy
        self.learning_type = learning_type
        self.run_label = run_label
        self.base_path = base_path
        self.output_root = output_root or os.path.join(self.base_path, OUTPUT_DIR)
        self.run_id = self._generate_run_id()
        self.run_output_dir = os.path.join(self.output_root, self.run_id)
        self.carbon_price_awareness = carbon_price_awareness
        self.satisfaction_regret = satisfaction_regret 
        
        # Simulation state
        self.year = MODEL_START_YEAR
        self.households = []
        self.n_households = 0
        
        # Components - CREATE THESE FIRST!
        self.data_loader = DataLoader(base_path)
        self.utility_calculator = UtilityCalculator()
        self.decision_maker = DecisionMaker()
        self.learning_mechanism = LearningMechanism()
        self.statistics = StatisticsAggregator()
        
        # THEN load trajectories (depends on data_loader)
        self.cge_data = self.data_loader.load_cge_trajectories()
        
        # Market state - load price trajectories
        self.price_trajectories = {}
        self._load_price_trajectories()
        self.prices = {
            'm_p_grey': M_P_GREY_BASE,
            'm_p_brown': M_P_BROWN_BASE,
            'm_p_green': M_P_GREEN_BASE,
        }

        # --- Set Stochastic Seed ---
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
        
        # Update output directory
        if run_label and seed is not None:
            self.run_label = f"{run_label}_seed_{seed}"
        elif run_label:
            self.run_label = run_label
        
        # Build output subfolder
        actual_output_dir = output_root if output_root else os.path.join(self.base_path, OUTPUT_DIR)
        if self.run_label:
            actual_output_dir = os.path.join(actual_output_dir, self.run_label)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            actual_output_dir = os.path.join(actual_output_dir, f"run_{timestamp}")

        self.exporter = ResultsExporter(output_dir=self.run_output_dir)
        
        self.run_config = {
            'case_study': self.case_study,
            'scenario': self.scenario,
            'policy': self.policy,
            'learning_type': self.learning_type,
            'run_label': self.run_label,
            'run_id': self.run_id,
            'start_year': MODEL_START_YEAR,
            'end_year': MODEL_END_YEAR,
            'output_dir': self.run_output_dir,
        }

        # Configuration
        self.debug = False
        self.memory_recall = True
        if VERBOSE:
            self._print_agent_summary()


    def initialize(self) -> bool:
        """
        Initialize model: load data and create agents.
        
        Returns:
            True if successful, False otherwise
        """
        if VERBOSE:
            print(f"\n{'='*60}")
            print("BENCH MODEL INITIALIZATION")
            print(f"{'='*60}")
            print(f"Case Study: {self.case_study}")
            print(f"Scenario: {self.scenario}")
            print(f"Policy: {self.policy}")
            print(f"{'='*60}\n")
        
        # Load all data
        if not self.data_loader.load_all_data():
            return False
        
        # Create household agents
        if not self._create_agents():
            return False
        
        return True
    
    def _sanitize_string(self, value: str) -> str:
        safe = re.sub(r'[^A-Za-z0-9_-]+', '_', value.strip().replace(' ', '_'))
        return safe.strip('_')[:80]
    
    def _generate_run_id(self) -> str:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        case_slug = self._sanitize_string(self.case_study)
        scenario_slug = self._sanitize_string(self.scenario)
        policy_slug = self._sanitize_string(self.policy)
        label_slug = self._sanitize_string(self.run_label) if self.run_label else None
        parts = [timestamp, case_slug, scenario_slug, policy_slug]
        if label_slug:
            parts.append(label_slug)
        return "_".join(parts)
    
    def _create_agents(self) -> bool:
        """
        Create unique household agents using baseline data from 2015 and
        attach their multi-year income time series.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Fetch raw household dataset containing all years from the data loader
            raw_household_data = self.data_loader.get_all_households_data()

            income_time_series_map = {}
            if 'id' in raw_household_data.columns and 'year' in raw_household_data.columns and 'income' in raw_household_data.columns:
                # Group by 'id' to isolate each unique agent's trajectory rows
                for agent_id, group in raw_household_data.groupby('id'):
                    # Create a dictionary mapping { year: income } for this specific individual agent
                    income_time_series_map[int(agent_id)] = dict(
                        zip(group['year'].astype(int), group['income'].astype(float))
                    )
            else:
                if VERBOSE:
                    print("Warning: Missing 'id', 'year', or 'income' columns. Cannot build individual time series.")

            # 2. Filter down to the baseline year 2015 to isolate initialization data
            if 'year' in raw_household_data.columns:
                household_data = raw_household_data[raw_household_data['year'] == 2015]
            else:
                household_data = raw_household_data
                if VERBOSE:
                    print("Warning: 'year' column not found in household data. Proceeding with raw data rows.")

            # 3. Ensure absolute uniqueness by dropping duplicate entries for the same agent ID in 2015
            if 'id' in household_data.columns:
                initial_count = len(household_data)
                household_data = household_data.drop_duplicates(subset=['id'], keep='first')
                dropped_count = initial_count - len(household_data)
                if dropped_count > 0 and VERBOSE:
                    print(f"-> Removed {dropped_count} duplicate agent row entries for year 2015.")

            # 4. Iterate over unique baseline household rows to instantiate agents
            for _, row in household_data.iterrows():
                # Extract attributes from data
                h_id = int(row.get('id'))
                income_group = int(row.get('group id'))
                income = float(row.get('income'))
                consumption = float(row.get('consumption'))
                
                # Energy flag: interpret from data
                energy_flag = int(row.get('lce user?'))
                if energy_flag > 1:
                    energy_flag = 2
                
                dwelling_label = int(row.get('dw_el')) if 'dw_el' in row else 3
                owner = bool(row.get('Owner'))
                
                # Convert row to dictionary and pull out what you already handled explicitly
                row_dict = row.to_dict()
                for key in ['id', 'group id', 'income', 'consumption', 'lce user?', 'dw_el', 'Owner', 'year']:
                    row_dict.pop(key, None)
                    
                # Create household agent
                household = Household(
                    h_id=h_id,
                    income_group=income_group,
                    income=income,
                    consumption_q=consumption,
                    energy_flag=energy_flag,
                    dwelling_label=dwelling_label,
                    owner=owner,
                    **row_dict
                )
                
                # This attaches a .income_trajectory dictionary attribute to the agent: e.g., {2015: 40810.0, 2016: 42000.0, ...}
                household.income_trajectory = income_time_series_map.get(h_id, {2015: income})
                
                # Set initial behavioral attributes if available
                if 'knowledge' in row or 'know' in row:
                    household.know = float(row.get('knowledge'))
                if 'cee_aw' in row:
                    household.cee_aw = float(row.get('cee_aw'))
                if 'ed_aw' in row:
                    household.ed_aw = float(row.get('ed_aw'))
                
                # Set norms if available - USE DICTIONARY ACCESS
                if 'personal1' in row:
                    household.per_nab['investment'] = float(row.get('personal1'))
                    household.per_nab['conservation'] = float(row.get('personal2')) if 'personal2' in row else 0
                    household.per_nab['switching'] = float(row.get('personal3')) if 'personal3' in row else 0

                if 'social1' in row:
                    household.su_nor['investment'] = float(row.get('social1'))
                    household.su_nor['conservation'] = float(row.get('social2')) if 'social2' in row else 0
                    household.su_nor['switching'] = float(row.get('social3')) if 'social3' in row else 0

                if 'pbc1' in row:
                    household.pbc['investment'] = float(row.get('pbc1'))
                    household.pbc['conservation'] = float(row.get('pbc2')) if 'pbc2' in row else 0
                    household.pbc['switching'] = float(row.get('pbc3')) if 'pbc3' in row else 0

                # Also add ep if available
                if 'ep' in row:
                    household.ep = float(row.get('ep'))
                
                # Initial awareness and motivation calculations
                household.update_awareness()
                household.update_motivation(self.case_study)
                
                self.households.append(household)
            
            self.n_households = len(self.households)
            self._place_households_on_grid()
            if VERBOSE:
                print(f"✓ Created {self.n_households} unique household agents from Year 2015 baseline with complete historical income trajectories.")
            return True
            
        except Exception as e:
            print(f"✗ Error creating agents: {e}")
            return False
        
    def _place_households_on_grid(self) -> None:
        """Place households onto a rectangular grid and precompute neighbor lists."""
        if self.n_households == 0:
            return

        grid_side = math.ceil(math.sqrt(self.n_households))
        self.grid_width = grid_side
        self.grid_height = math.ceil(self.n_households / self.grid_width)

        grid_cells = [
            (x, y)
            for y in range(self.grid_height)
            for x in range(self.grid_width)
        ][: self.n_households]
        random.shuffle(grid_cells)

        for household, (x, y) in zip(self.households, grid_cells):
            household.grid_x = x
            household.grid_y = y

        self.grid_index = {
            (household.grid_x, household.grid_y): household
            for household in self.households
        }

        # Upgrade 1: precompute neighbor lists once at init (O(1) per lookup thereafter)
        self._nbr_cache: dict = {
            hh.h_id: self._get_grid_neighbors_raw(hh)
            for hh in self.households
        }

    def _get_grid_neighbors_raw(self, household) -> List[Household]:
        """Raw 8-cell lookup used only during grid setup to populate the cache."""
        if not hasattr(household, 'grid_x'):
            return []
        return [
            self.grid_index[pos]
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
            for pos in [(household.grid_x + dx, household.grid_y + dy)]
            if pos in self.grid_index
        ]

    def _get_grid_neighbors(self, household) -> List[Household]:
        """Return precomputed 8-cell neighbors for a household (O(1) lookup)."""
        return self._nbr_cache.get(household.h_id, [])

    def _apply_satisfaction_and_regret(self) -> None:
        """Apply satisfaction and regret for households that took actions in previous years."""
        if self.year < 2017:
            return
        
        for household in self.households:
            satisfaction = self.decision_maker.calculate_satisfaction(household)
            if satisfaction != "none":
                # Store satisfaction for learning/regret
                if not hasattr(household, 'satisfaction_history'):
                    household.satisfaction_history = {}
                household.satisfaction_history[self.year] = satisfaction
                
                # Apply regret effects
                self.decision_maker.apply_regret(household, satisfaction, self.learning_type)

    def _print_agent_summary(self) -> None:
        """Print summary of created agents."""
        if self.n_households == 0:
            return
        
        ff_count = sum(1 for hh in self.households if hh.flag == 0)
        brown_count = sum(1 for hh in self.households if hh.flag == 1)
        green_count = sum(1 for hh in self.households if hh.flag == 2)
        
        avg_income = sum(hh.h_income for hh in self.households) / self.n_households
        avg_consumption = sum(hh.h_q for hh in self.households) / self.n_households
        avg_awareness = sum(hh.h_aware for hh in self.households) / self.n_households
        
        print(f"\nAgent Summary (Baseline 2015):")
        print(f"  Total Households: {self.n_households}")
        print(f"  Energy Source Distribution:")
        print(f"    - Gray electricity: {ff_count} ({ff_count/self.n_households*100:.1f}%)")
        print(f"    - Brown electricity: {brown_count} ({brown_count/self.n_households*100:.1f}%)")
        print(f"    - Green electricity: {green_count} ({green_count/self.n_households*100:.1f}%)")
        print(f"  Average Income: €{avg_income:,.0f}")
        print(f"  Average Consumption: {avg_consumption:,.0f} kWh/year")
        print(f"  Average Awareness: {avg_awareness:.2f}")
    
    def step(self) -> bool:
        """Execute one year of simulation matching NetLogo's go procedure order."""

        if self.year > MODEL_END_YEAR:
            return False
        
        # === MEMORY RECALL (only 2015) ===
        if self.year == MODEL_START_YEAR and self.memory_recall:
            self._recall_memory()
        
        # === UPDATE PRICES ===
        self._update_prices()
        
        # === CARBON PRICE AWARENESS (cpinfo) ===
        if self.carbon_price_awareness and self.year >= 2016:
            for household in self.households:
                self.decision_maker.apply_carbon_price_awareness(household, self.policy, self.year)
        
        # === KNOWLEDGE ACTIVATION ===
        for household in self.households:
            self.decision_maker.activate_knowledge(household, self.case_study)
        
        # === MOTIVATION ===
        for household in self.households:
            self.decision_maker.update_motivation(household, self.case_study)
        
        # === CONSIDERATION (with energy patterns for conservation) ===
        for household in self.households:
            for action_name in ['investment', 'conservation', 'switching']:
                ep = getattr(household, 'ep', None) if action_name == 'conservation' else None
                household.consider_constraints(action_name, ep)
        
        # === UPDATE HOUSEHOLD ATTRIBUTES AND BUDGETS ===
        for household in self.households:
            household.set_income_for_year(self.year)
            household.calculate_budgets(self.prices)
        
        # === GLOBAL POPULATION NORMALIZATION (normalize-1) ===
        self.utility_calculator.normalize_budgets(self.households)
        
        # === NORMALIZE BUDGET VALUES PER HOUSEHOLD ===
        for household in self.households:
            self.utility_calculator.normalize_budget_values(household)
        
        # === ACTUAL UTILITIES (update_utilities_NAT) - PASS ALL HOUSEHOLDS AND PRICES ===
        self.utility_calculator.calculate_actual_utility(self.households, self.prices)
        
        # === SATISFACTION AND REGRET (only from 2017 onwards) ===
        if self.satisfaction_regret and self.year >= 2017:
            self._apply_satisfaction_and_regret()
        
        # === EXPECTED UTILITIES (utility_exp_NAT) ===
        for household in self.households:
            self.utility_calculator.calculate_all_expected_utilities(household)
        
        # === ACTIONS (only from 2016 onwards) ===
        if self.year >= 2016:
            for household in self.households:
                self.decision_maker.decide_action(household, {}, self.utility_calculator)
        
        # === SAVE ENERGY AND EMISSIONS ===
        for household in self.households:
            self.decision_maker.calculate_energy_savings(household, self.prices)
            self.decision_maker.calculate_financial_outcomes(household, self.prices)
            self.decision_maker.calculate_emissions_avoided(household, self.prices)
        
        # === UPDATE ECONOMIC DATA (income, consumption, alpha from CGE) ===
        self._update_economic_data()
        
        # === SOCIAL LEARNING ===
        self._apply_social_learning()
        
        # === UPDATE ENERGY CONSUMPTION (update_heq) ===
        self._update_energy_consumption()
        
        # === UPDATE MEMORY (reset annual flags and apply cooldowns) ===
        self._update_memory()
        
        # === RECORD STATISTICS ===
        stats = self.statistics.aggregate_population_stats(self.households, self.year)
        self.statistics.store_annual_stats(self.year, stats)
        ig_stats = self.statistics.aggregate_by_income_group(self.households, self.year)
        self.statistics.store_income_group_stats(self.year, ig_stats)
        
        self.year += 1
        return True
    
    def _update_economic_data(self) -> None:
        """
        Update household economic data from CGE trajectories.
        Matches NetLogo's update_data procedure.
        
        This updates:
        - Income (h_income) using growth multipliers from CGE data
        - Electricity consumption (h_q) using growth multipliers from CGE data
        - Alpha parameter (share of electricity consumption)
        """
        # First update happens in 2016 (NetLogo uses n=1 for 2016)
        if self.year < 2016:
            return
        
        # Check if CGE data is available
        if not self.cge_data:
            if VERBOSE and self.year == 2016:
                print("Warning: No CGE data loaded - skipping economic updates")
            return
        
        # Calculate year index for CGE data
        # NetLogo uses 'n' which starts at 0 for 2015, so:
        # 2015: n=0, 2016: n=1, 2017: n=2, etc.
        year_index = self.year - 2015
        
        # Get the raw data lists
        income_data = self.cge_data.get('income', {}).get('raw', [])
        consumption_data = self.cge_data.get('consumption', {}).get('raw', [])
        alpha_data = self.cge_data.get('alpha', {}).get('raw', [])
        
        # Check if we have data for this year
        if year_index >= len(income_data):
            if VERBOSE and self.year == 2016:
                print(f"Warning: No CGE data for year index {year_index}")
            return
        
        for household in self.households:
            income_group = household.h_income_group
            
            # Map income group to column index (matching NetLogo)
            # NetLogo uses different 't' offsets per income group:
            # Income group 1: uses n (0,1,2,3,4,5...)
            # Income group 2: uses n+1 (1,2,3,4,5,6...)
            # Income group 3: uses n+2 (2,3,4,5,6,7...)
            # Income group 4: uses n+3 (3,4,5,6,7,8...)
            # Income groups 5+: uses n+4 (4,5,6,7,8,9...)
            
            if income_group == 1:
                col_idx = 0
                cons_idx = 0
                alpha_idx = 0
            elif income_group == 2:
                col_idx = 0
                cons_idx = 1  # NetLogo uses t = n+1
                alpha_idx = 1
            elif income_group == 3:
                col_idx = 0
                cons_idx = 2  # NetLogo uses t = n+2
                alpha_idx = 2
            elif income_group == 4:
                col_idx = 0
                cons_idx = 3  # NetLogo uses t = n+3
                alpha_idx = 3
            else:  # Income groups 5, 6, 7
                col_idx = 0
                cons_idx = 4  # NetLogo uses t = n+4
                alpha_idx = 4
            
            # Update income using growth multiplier
            if income_data and year_index < len(income_data):
                try:
                    # Get the multiplier for this year and column
                    if isinstance(income_data[year_index], (list, tuple)):
                        income_multiplier = income_data[year_index][col_idx]
                    else:
                        income_multiplier = income_data[year_index]
                    
                    # Apply multiplier (NetLogo multiplies current income by growth factor)
                    household.h_income = household.h_income * income_multiplier
                except (IndexError, TypeError) as e:
                    if VERBOSE and self.year == 2016:
                        print(f"Warning: Could not update income for group {income_group}: {e}")
            
            # Update consumption (h_q) using growth multiplier
            if consumption_data and year_index < len(consumption_data):
                try:
                    # Get the multiplier for this year and consumption index
                    if isinstance(consumption_data[year_index], (list, tuple)):
                        consumption_multiplier = consumption_data[year_index][cons_idx]
                    else:
                        consumption_multiplier = consumption_data[year_index]
                    
                    # Apply multiplier
                    household.h_q = household.h_q * consumption_multiplier
                except (IndexError, TypeError) as e:
                    if VERBOSE and self.year == 2016:
                        print(f"Warning: Could not update consumption for group {income_group}: {e}")
            
            # Update alpha (share parameter)
            if alpha_data and year_index < len(alpha_data):
                try:
                    # Get alpha value for this year and alpha index
                    if isinstance(alpha_data[year_index], (list, tuple)):
                        new_alpha = alpha_data[year_index][alpha_idx]
                    else:
                        new_alpha = alpha_data[year_index]
                    
                    # Set alpha directly (not multiplied)
                    household.alpha = new_alpha
                except (IndexError, TypeError) as e:
                    if VERBOSE and self.year == 2016:
                        print(f"Warning: Could not update alpha for group {income_group}: {e}")

    def _recall_memory(self) -> None:
        """Apply 2015 memory recall for historical behavior."""
        for household in self.households:
            self.learning_mechanism.recall_memory(
                household, {}, household.h_income_group,
                household.flag, self.case_study
            )

    def _update_memory(self) -> None:
        """
        Update household memory and apply action cooldowns.
        Matches NetLogo's update_memory procedure.
        
        This method:
        1. Increments cooldown counters for actions taken in previous years
        2. Re-enables actions after cooldown periods expire
        3. Resets annual action flags for the next year
        
        Cooldown periods:
        - Investment (act11/act12): 10 years before can invest again
        - Conservation (act21/act40): 5 years before can conserve again
        - Switching (act32): 2 years before can switch again
        """
        # Initialize cooldown counters if they don't exist (first run)
        for household in self.households:
            if not hasattr(household, 'act11_year'):
                household.act11_year = 0
                household.act12_year = 0
                household.act21_year = 0
                household.act40_year = 0
                household.act31_year = 0
                household.act32_year = 0
        
        # First: Increment counters for actions that are currently active
        for household in self.households:
            # Investment counters (act11 for brown/green, act12 for grey)
            if household.act11:
                household.act11_year += 1
            if household.act12:
                household.act12_year += 1
            
            # Conservation counters (act21 for brown/green, act40 for grey)
            if household.act21:
                household.act21_year += 1
            if household.act40:
                household.act40_year += 1
            
            # Switching counters (act31 for brown->green, act32 for grey->brown)
            if household.act31:
                household.act31_year += 1
            if household.act32:
                household.act32_year += 1
        
        # Second: Apply cooldown expirations (re-enable actions after cooldown)
        for household in self.households:
            # === INVESTMENT COOLDOWN (10 years) ===
            # After 10 years, investment can be taken again
            if household.act11_year >= 10:
                household.hh_actions[0] = 0  # Reset action vector
                household.act11 = False      # Clear the action flag
                household.act11_year = 0     # Reset counter
            
            if household.act12_year >= 10:
                household.hh_actions[1] = 0
                household.act12 = False
                household.act12_year = 0
            
            # === CONSERVATION COOLDOWN (5 years) ===
            # After 5 years, conservation can be taken again
            if household.act21_year >= 5:
                household.hh_actions[2] = 0
                household.act21 = False
                household.act21_year = 0
            
            if household.act40_year >= 5:
                household.hh_actions[3] = 0
                household.act40 = False
                household.act40_year = 0
            
            # === SWITCHING COOLDOWN (2 years for act32 grey->brown) ===
            # After 2 years, switching can be taken again
            if household.act32_year >= 2:
                household.hh_actions[5] = 0
                household.act32 = False
                household.act32_year = 0
            
            # Note: act31 (brown->green) does NOT have a cooldown in NetLogo
            # Once switched to green, they stay green (no switching back)
        
        # Third: Reset annual action flags for the next year
        # These are the flags that count for yearly statistics (act1, act2, act3)
        self._reset_annual_actions()

    def _load_price_trajectories(self) -> None:
        """Load price trajectories from data files."""
        file_path = os.path.join(self.base_path, PRIMES_NL_PRICES_FILE)
        
        if not os.path.exists(file_path):
            print(f"Warning: Price file not found: {file_path}")
            return
        
        df = pd.read_csv(file_path, header=None)
        
        # Store prices for each year and policy
        self.price_trajectories = {}
        
        for year_idx, year in enumerate(range(2015, 2031)):
            if year_idx >= len(df):
                break
                
            self.price_trajectories[year] = {}
            
            if self.policy == "Ref":
                self.price_trajectories[year]['m_p_grey'] = df.iloc[year_idx, 0]
                self.price_trajectories[year]['m_p_brown'] = df.iloc[year_idx, 0]
                self.price_trajectories[year]['m_p_green'] = df.iloc[year_idx, 0]
                
            elif self.policy == "Carbon price pressure-10":
                self.price_trajectories[year]['m_p_brown'] = df.iloc[year_idx, 1]
                self.price_trajectories[year]['m_p_grey'] = df.iloc[year_idx, 2]
                self.price_trajectories[year]['m_p_green'] = M_P_GREEN_BASE
                
            elif self.policy == "Carbon price pressure-25":
                self.price_trajectories[year]['m_p_brown'] = df.iloc[year_idx, 3]
                self.price_trajectories[year]['m_p_grey'] = df.iloc[year_idx, 4]
                self.price_trajectories[year]['m_p_green'] = M_P_GREEN_BASE
                
            elif self.policy == "Carbon price pressure-50":
                self.price_trajectories[year]['m_p_brown'] = df.iloc[year_idx, 5]
                self.price_trajectories[year]['m_p_grey'] = df.iloc[year_idx, 6]
                self.price_trajectories[year]['m_p_green'] = M_P_GREEN_BASE
                
            elif self.policy == "Carbon price pressure-100":
                self.price_trajectories[year]['m_p_brown'] = df.iloc[year_idx, 7]
                self.price_trajectories[year]['m_p_grey'] = df.iloc[year_idx, 8]
                self.price_trajectories[year]['m_p_green'] = M_P_GREEN_BASE
                
            elif self.policy == "Carbon price pressure-2020":
                self.price_trajectories[year]['m_p_brown'] = df.iloc[year_idx, 9]
                self.price_trajectories[year]['m_p_grey'] = df.iloc[year_idx, 10]
                self.price_trajectories[year]['m_p_green'] = M_P_GREEN_BASE

    def _update_prices(self) -> None:
        """Update market electricity prices from PRIMES trajectories."""
        if self.year in self.price_trajectories:
            self.prices = self.price_trajectories[self.year].copy()

    def _apply_social_learning(self) -> None:
        """
        Apply the selected learning algorithm to households after 2015.
        Matches NetLogo's learn procedure placement.
        """
        
        if self.year < 2016 or self.learning_type == "No learning":
            return

        for household in self.households:
            if not (household.act1 or household.act50 or household.act3):
                continue

            neighbors = self._get_grid_neighbors(household)
            if not neighbors:
                continue

            # Bug 2 fix: call one learning block per action taken, each updating only
            # the norms for that action type (matches NetLogo's three separate learn blocks)
            if household.act1:
                self.learning_mechanism.apply_learning(
                    household, neighbors, self.year, self.learning_type,
                    'investment', get_neighbors_fn=self._get_grid_neighbors
                )
            if household.act50:
                self.learning_mechanism.apply_learning(
                    household, neighbors, self.year, self.learning_type,
                    'conservation', get_neighbors_fn=self._get_grid_neighbors
                )
            if household.act3:
                self.learning_mechanism.apply_learning(
                    household, neighbors, self.year, self.learning_type,
                    'switching', get_neighbors_fn=self._get_grid_neighbors
                )

    def _update_energy_consumption(self) -> None:
        """
        Update household energy consumption based on actions taken.
        Matches NetLogo's update_heq procedure.
        
        Order of operations (matches NetLogo):
        1. Investment reduces consumption by fixed amount (1700 kWh)
        2. Conservation reduces consumption by saved amount (50% of original)
        3. Minimum consumption floor of 1000 kWh after conservation
        
        Note: h_conserv is calculated in calculate_energy_savings BEFORE this method
            using the original h_q value (before any reductions)
        """
        for household in self.households:
            # === INVESTMENT (act1) ===
            # NetLogo: if (act1 = True and h_q > 0) [set h_q (h_q - 1700)]
            if household.act1 and household.h_q > 0:
                household.h_q = household.h_q - INVESTMENT_PV_ENERGY_OUTPUT
                
                # Check if household became a self-producer
                if household.h_q <= 0:
                    household.hh_sta = "self-producer"
                    household.h_q = 0
            
            # === CONSERVATION (act50) ===
            # NetLogo: if ((act50 = True) and (h_q > 1000)) [set h_q (h_q - h_conserv)]
            #          if ((act50 = True) and (h_q <= 1000)) [set hh_sta "efficient" set h_q 1000]
            if household.act2 or household.act50:
                if household.h_q > 1000:
                    # Reduce consumption by saved amount
                    household.h_q = household.h_q - household.h_conserv
                    # Ensure minimum of 1000 kWh
                    if household.h_q < 1000:
                        household.h_q = 1000
                else:
                    # Already at or below minimum
                    household.hh_sta = "efficient"
                    household.h_q = 1000

    def _reset_annual_actions(self) -> None:
        """
        Reset annual action flags.
        These are the flags that count for yearly statistics.
        
        Note: act11, act12, act21, act40, act31, act32 are PERMANENT records
        that track if an action was EVER taken and are used for cooldowns.
        They are NOT reset here.
        """
        for household in self.households:
            household.act1 = False   # General investment flag for current year
            household.act2 = False   # General conservation flag for current year
            household.act3 = False   # General switching flag for current year
            # Do NOT reset act11, act12, act21, act40, act31, act32

    def run(self) -> bool:
        """
        Run complete simulation from 2015 to 2030.
        
        Args:
            verbose: Print progress messages
            
        Returns:
            True if successful
        """
        if not self.initialize():
            print("✗ Model initialization failed")
            return False
        
        if VERBOSE:
            print(f"{'='*60}")
            print("SIMULATION RUNNING")
            print(f"{'='*60}\n")
        
        while self.year <= MODEL_END_YEAR:
            if not self.step():
                break
            
            if VERBOSE and self.year % 5 == 0:
                print(f"✓ Year {self.year} completed")

        if VERBOSE:
            print(f"\n{'='*60}")
            print("SIMULATION COMPLETE")
            print(f"{'='*60}\n")
        
        return True
    
    def export_results(self) -> List[str]:
        """Export all simulation results, including run configuration."""
        files = self.exporter.export_all_results(
            self, MODEL_START_YEAR, MODEL_END_YEAR
        )
        files.extend(self.exporter.export_run_config(self.run_config))
        return files
    
    def get_summary(self) -> Dict:
        """Get summary of simulation results."""
        return self.statistics.get_cumulative_stats(MODEL_START_YEAR, MODEL_END_YEAR)

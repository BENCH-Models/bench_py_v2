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
from utils.constants import (
    CARBON_POLICY_TARGETS, EMISSIONS_FACTOR_GRAY, EMISSIONS_FACTOR_BROWN, EMISSIONS_FACTOR_GREEN, KG_TO_TONS,
    M_P_GREEN_BASE, M_P_BROWN_BASE, M_P_GREY_BASE, MODEL_START_YEAR, MODEL_END_YEAR,
    INCOME_GROUPS, DWELLING_LABELS,
    FLAG_NAMES, OUTPUT_DIR,
    DEFAULT_LEARNING_TYPE
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
    
    def __init__(self, case_study: str, scenario: str, policy: str,
                 learning_type: str = DEFAULT_LEARNING_TYPE,
                 run_label: str = None,
                 base_path: str = ".",
                 output_root: str = None):
        """
        Initialize BENCH model.
        
        Args:
            case_study: "Netherlands-Overijssel" or "Spain-Navarre"
            scenario: "Ref_SSP2"
            policy: Price scenario policy
            learning_type: One of the supported learning modes
            run_label: Optional label used to make run IDs readable
            base_path: Path to project root
            output_root: Optional root directory for output folders
        """
        self.case_study = case_study
        self.scenario = scenario
        self.policy = policy
        self.learning_type = learning_type
        self.run_label = run_label
        self.base_path = base_path
        self.output_root = output_root or os.path.join(self.base_path, OUTPUT_DIR)
        self.run_id = self._generate_run_id()
        self.run_output_dir = os.path.join(self.output_root, self.run_id)
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
        
        # Simulation state
        self.year = MODEL_START_YEAR
        self.households = []
        self.n_households = 0
        
        # Market state initially
        self.prices = {
            'm_p_grey': M_P_GREY_BASE,
            'm_p_brown': M_P_BROWN_BASE,
            'm_p_green': M_P_GREEN_BASE,
        }


        
        # Components
        self.data_loader = DataLoader(base_path)
        self.utility_calculator = UtilityCalculator()
        self.decision_maker = DecisionMaker()
        self.learning_mechanism = LearningMechanism()
        self.statistics = StatisticsAggregator()
        self.exporter = ResultsExporter(output_dir=self.run_output_dir)
        
        # Configuration
        self.debug = False
        self.memory_recall = True
    
    def initialize(self) -> bool:
        """
        Initialize model: load data and create agents.
        
        Returns:
            True if successful, False otherwise
        """
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
        
        # Print summary
        #self.data_loader.print_summary()
        #self._print_agent_summary()
        
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
        Create household agents from loaded data.
        
        Returns:
            True if successful
        """
        try:
            household_data = self.data_loader.get_all_households_data()

            for _, row in household_data.iterrows():
                # Extract attributes from data
                h_id = int(row.get('id', len(self.households)))
                income_group = int(row.get('group id', 1))
                income = float(row.get('income', 20000))
                consumption = float(row.get('consumption', 3000))
                
                # Energy flag: interpret from data
                energy_flag = int(row.get('lce user?', 0))
                if energy_flag > 1:
                    energy_flag = 2
                
                dwelling_label = int(row.get('dw_el', 3)) if 'dw_el' in row else 3
                owner = bool(row.get('Owner', True))
                
                # Convert row to dictionary and pull out what you already handled explicitly, DONT WANT TO PASS THE STUFF THAT IS ALERADY EXTRACTED AS ARGUMENTS TO THE HOUSEHOLD CONSTRUCTOR
                row_dict = row.to_dict()
                for key in ['id', 'group id', 'income', 'consumption', 'lce user?', 'dw_el', 'Owner']:
                    row_dict.pop(key, None)
                    
                # Create household
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
                
                # Set initial behavioral attributes if available
                if 'knowledge' in row or 'know' in row:
                    household.know = float(row.get('knowledge', row.get('know', 0)))
                if 'cee_aw' in row:
                    household.cee_aw = float(row.get('cee_aw', 0))
                if 'ed_aw' in row:
                    household.ed_aw = float(row.get('ed_aw', 0))
                
                # Set norms if available
                if 'personal1' in row:
                    household.per_nab[0] = float(row.get('personal1', 0))
                    household.per_nab[1] = float(row.get('personal2', 0)) if 'personal2' in row else 0
                    household.per_nab[2] = float(row.get('personal3', 0)) if 'personal3' in row else 0
                
                if 'social1' in row:
                    household.su_nor[0] = float(row.get('social1', 0))
                    household.su_nor[1] = float(row.get('social2', 0)) if 'social2' in row else 0
                    household.su_nor[2] = float(row.get('social3', 0)) if 'social3' in row else 0
                
                if 'pbc1' in row:
                    household.pbc[0] = float(row.get('pbc1', 0))
                    household.pbc[1] = float(row.get('pbc2', 0)) if 'pbc2' in row else 0
                    household.pbc[2] = float(row.get('pbc3', 0)) if 'pbc3' in row else 0
                
                # Initial awareness calculation
                household.update_awareness()
                household.update_motivation(self.case_study)
                
                self.households.append(household)
            
            self.n_households = len(self.households)
            self._place_households_on_grid()
            print(f"✓ Created {self.n_households} household agents")
            return True
            
        except Exception as e:
            print(f"✗ Error creating agents: {e}")
            return False
    
    def _place_households_on_grid(self) -> None:
        """Place households onto a rectangular grid and index their neighbors."""
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

    def _get_grid_neighbors(self, household) -> List[Household]:
        """Return the 8 adjacent grid neighbors for a household."""
        if not hasattr(household, 'grid_x') or not hasattr(household, 'grid_y'):
            return []

        adjacent_positions = [
            (household.grid_x + dx, household.grid_y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        ]

        return [
            self.grid_index[pos]
            for pos in adjacent_positions
            if pos in self.grid_index
        ]

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
        """
        Execute one year of simulation.
        
        Returns:
            True if year completed successfully
        """
        if self.year > MODEL_END_YEAR:
            return False
        
        if self.debug:
            print(f"\n--- Year {self.year} ---")
        
        # Recall historical behavior (2015 only)
        if self.year == MODEL_START_YEAR and self.memory_recall:
            self._recall_memory()
        
        # Update prices based on policy
        self._update_prices()
        
        # For each household: behavioral decision process
        for household in self.households:
            # 1. Activate knowledge pathway
            self.decision_maker.activate_knowledge(household, self.case_study)
            
            # 2. Update motivation
            self.decision_maker.update_motivation(household, self.case_study)
            
            # 3. Consider constraints
            for action_type in range(3):
                self.decision_maker.consider_action(household, action_type)
            
            # 4. Calculate budget constraints
            household.calculate_budgets(self.prices)
            
            # 5. Normalize budgets across population
            self.utility_calculator.normalize_budget_values(household)
            
            # 6. Calculate utilities
            self.utility_calculator.calculate_all_utilities(household, {})
            
            # 7. Make decisions
            self.decision_maker.decide_action(household, {}, self.utility_calculator)
            
            # 8. Calculate outcomes
            self.decision_maker.calculate_energy_savings(household, self.prices)
            self.decision_maker.calculate_financial_outcomes(household, self.prices)
            self.decision_maker.calculate_emissions_avoided(household, self.prices)
            
            # 9. Update actual utility
            self.utility_calculator.calculate_actual_utility(household)
            
            # 10. Learning
            self.learning_mechanism.update_memory(household, self.year)
        
        # 11. Social learning (grid adjacency: each HH learns from direct neighbors)
        self._apply_social_learning()
        
        # 12. Aggregate statistics
        stats = self.statistics.aggregate_population_stats(self.households, self.year)
        self.statistics.store_annual_stats(self.year, stats)
        
        if self.debug:
            print(f"  LCE Share: {stats['lce_share_percent']:.1f}%")
            print(f"  Actions Taken: {stats['action_total_count']}")
        
        # Increment time
        self.year += 1
        
        return True
    
    def _recall_memory(self) -> None:
        """Apply 2015 memory recall for historical behavior."""
        for household in self.households:
            self.learning_mechanism.recall_memory(
                household, {}, household.h_income_group,
                household.flag, self.case_study
            )

    def _update_prices(self) -> None:
        """
        Update market electricity prices dynamically based on the active policy scenario.
        Extracts target carbon values using a clean constant dictionary mapping lookup.
        """
        # 1. Direct dictionary key resolution with a safe fallback default (0.0 for Ref/unknown)
        target_carbon_price_2030 = CARBON_POLICY_TARGETS.get(self.policy, 0.0)
                
        # 2. Compute the Carbon Tax Trajectory over our linear timeline window
        carbon_tax_per_kwh_grey = 0.0
        carbon_tax_per_kwh_brown = 0.0
        current_tax_per_ton = 0.0

        if self.year >= 2017 and target_carbon_price_2030 > 0.0:
            # Freeze the tax progression timeline at the year 2030 ceiling boundary
            tax_year = min(self.year, 2030)
            
            # Calculate linear step fraction scaling from 2017 to 2030
            progression = (tax_year - 2017) / (2030 - 2017)
            current_tax_per_ton = target_carbon_price_2030 * progression
            
            # Convert €/ton to €/kWh:
            tax_per_kg = current_tax_per_ton * KG_TO_TONS
            carbon_tax_per_kwh_grey = tax_per_kg * EMISSIONS_FACTOR_GRAY
            carbon_tax_per_kwh_brown = tax_per_kg * EMISSIONS_FACTOR_BROWN

        # 3. Apply final calculated values to the active price variables
        self.prices['m_p_grey'] = M_P_GREY_BASE + carbon_tax_per_kwh_grey
        self.prices['m_p_brown'] = M_P_BROWN_BASE + carbon_tax_per_kwh_brown
        self.prices['m_p_green'] = M_P_GREEN_BASE  # Renewable track avoids emissions penalties

    def _apply_social_learning(self) -> None:
        """Apply the selected learning algorithm to households after 2015."""
        if self.year < 2016 or self.learning_type == "No learning":
            return

        for household in self.households:
            # 1. Check the central household FIRST. 
            # If it hasn't done ANY of these actions, skip immediately.
            if not (household.act1 or household.act3 or household.act50):
                continue  # Skip to the next household

            # 2. Only fetch neighbors and do heavy math if the central household is active
            neighbors = self._get_grid_neighbors(household)
            
            self.learning_mechanism.apply_learning(
                household,
                neighbors,
                self.year,
                self.learning_type
            )

    def run(self, verbose: bool = True) -> bool:
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
        
        print(f"{'='*60}")
        print("SIMULATION RUNNING")
        print(f"{'='*60}\n")
        
        while self.year <= MODEL_END_YEAR:
            if not self.step():
                break
            
            if verbose and self.year % 5 == 0:
                print(f"✓ Year {self.year} completed")
        
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

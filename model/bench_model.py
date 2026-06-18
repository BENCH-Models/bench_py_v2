"""
Main BENCH Model simulation engine â€” matrix-based agent representation.
All agent state lives in a Population object (parallel numpy arrays).
Per-Household Python loops replaced by vectorized numpy operations.
"""

import os
import re
import math
import random
import datetime
from typing import Dict, List, Optional

import numpy as np

from model.loader import DataLoader
from model.population import Population
from model.vectorized import (
    update_awareness,
    apply_carbon_price_awareness,
    update_motivation,
    consider_constraints,
    set_income_for_year,
    calculate_budgets,
    normalize_budgets,
    calculate_all_expected_utilities,
    calculate_actual_utility,
    calculate_satisfaction,
    apply_regret,
    decide_action,
    calculate_energy_savings,
    calculate_financial_outcomes,
    calculate_emissions_avoided,
    update_energy_consumption,
    update_memory,
    update_economic_data,
    apply_social_learning,
    recall_memory,
)
from model.statistics import StatisticsAggregator
from model.output import ResultsExporter
from model.parameters import (
    M_P_GREEN_BASE, M_P_BROWN_BASE, M_P_GREY_BASE,
    MODEL_START_YEAR, MODEL_END_YEAR,
    OUTPUT_DIR,
    DEFAULT_LEARNING_TYPE,
    VERBOSE,
    PRIMES_NL_PRICES_FILE,
)
import pandas as pd


class BENCHModel:
    """
    Main BENCH Model implementation â€” Population-matrix edition.
    Agent state stored as parallel numpy arrays in self.pop (Population).
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
                 satisfaction_regret: bool = True):

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
        self.pop: Optional[Population] = None
        self.n_households = 0

        # Components
        self.data_loader = DataLoader(base_path)
        self.statistics = StatisticsAggregator()

        # Load CGE trajectories and price data
        self.cge_data = self.data_loader.load_cge_trajectories()
        self.price_trajectories: Dict = {}
        self._load_price_trajectories()
        self.prices = {
            'm_p_grey':  M_P_GREY_BASE,
            'm_p_brown': M_P_BROWN_BASE,
            'm_p_green': M_P_GREEN_BASE,
        }

        # Stochastic seed
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Build output subfolder
        if run_label and seed is not None:
            self.run_label = f"{run_label}_seed_{seed}"
        elif run_label:
            self.run_label = run_label

        self.exporter = ResultsExporter(output_dir=self.run_output_dir)

        self.run_config = {
            'case_study':   self.case_study,
            'scenario':     self.scenario,
            'policy':       self.policy,
            'learning_type': self.learning_type,
            'run_label':    self.run_label,
            'run_id':       self.run_id,
            'start_year':   MODEL_START_YEAR,
            'end_year':     MODEL_END_YEAR,
            'output_dir':   self.run_output_dir,
        }

        self.debug = False
        self.memory_recall = True

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        if not self.data_loader.load_all_data():
            return False
        return self._create_population()

    def _create_population(self) -> bool:
        """Load CSV data directly into Population arrays â€” no per-row loops."""
        try:
            pop, _id_map = self.data_loader.create_population(self.case_study)
            self.pop = pop
            self.n_households = pop.N
            self._place_households_on_grid()
            if VERBOSE:
                self._print_agent_summary()
            return True
        except Exception as e:
            print(f"âœ— Error creating population: {e}")
            import traceback; traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Grid placement + neighbor precompute
    # ------------------------------------------------------------------

    def _place_households_on_grid(self) -> None:
        pop = self.pop
        assert pop is not None
        n   = pop.N
        if n == 0:
            return

        grid_side = math.ceil(math.sqrt(n))
        gw = grid_side
        gh = math.ceil(n / gw)

        cells = [
            (x, y)
            for y in range(gh)
            for x in range(gw)
        ][:n]
        random.shuffle(cells)

        xs = np.array([c[0] for c in cells], dtype=np.int32)
        ys = np.array([c[1] for c in cells], dtype=np.int32)
        pop.grid_x[:] = xs
        pop.grid_y[:] = ys

        # Build (x, y) â†’ agent-index lookup
        xy_to_idx: Dict = {}
        for i in range(n):
            xy_to_idx[(int(xs[i]), int(ys[i]))] = i

        # Precompute 8-cell neighbor lists (indices, not Household objects)
        nbr_cache: List[np.ndarray] = []
        for i in range(n):
            x, y = int(xs[i]), int(ys[i])
            nbrs = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    j = xy_to_idx.get((x + dx, y + dy))
                    if j is not None:
                        nbrs.append(j)
            nbr_cache.append(np.array(nbrs, dtype=np.int32))

        pop.nbr_cache = nbr_cache

        # Padded (N, 8) neighbour matrix — -1 for missing slots (used by vectorized learning)
        for i, nbrs in enumerate(nbr_cache):
            pop.nbr_matrix[i, :len(nbrs)] = nbrs

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(self) -> bool:
        if self.year > MODEL_END_YEAR:
            return False

        pop = self.pop
        assert pop is not None

        # Memory recall (2015 only)
        if self.year == MODEL_START_YEAR and self.memory_recall:
            self._recall_memory()

        # Update prices
        self._update_prices()

        # Carbon price awareness (cpinfo)
        if self.carbon_price_awareness and self.year >= 2016:
            apply_carbon_price_awareness(pop, self.policy, self.year)

        # Knowledge activation (update_awareness recalculates guilt + K)
        update_awareness(pop)

        # Motivation
        update_motivation(pop, self.case_study)

        # Consideration â†’ delta
        consider_constraints(pop)

        # Income + budgets
        set_income_for_year(pop, self.year)
        calculate_budgets(pop, self.prices)

        # Population normalisation (normalize-1)
        normalize_budgets(pop)

        # Actual utilities (normalize-2 + utility_actual)
        calculate_actual_utility(pop, self.prices)

        # Satisfaction and regret (from 2017 onwards)
        if self.satisfaction_regret and self.year >= 2017:
            calculate_satisfaction(pop)
            apply_regret(pop, self.learning_type)

        # Expected utilities
        calculate_all_expected_utilities(pop)

        # Actions (from 2016 onwards)
        if self.year >= 2016:
            decide_action(pop)

        # Energy savings, financial outcomes, emissions
        calculate_energy_savings(pop, self.prices)
        calculate_financial_outcomes(pop, self.prices)
        calculate_emissions_avoided(pop)

        # CGE economic update (income Ã— multiplier, h_q Ã— multiplier, alpha)
        update_economic_data(pop, self.cge_data, self.year)

        # Social learning (vectorised — no per-agent loops)
        if self.year >= 2016:
            apply_social_learning(pop, self.learning_type)

        # Energy consumption update (subtract investment/conservation savings)
        update_energy_consumption(pop)

        # Statistics — must come before update_memory, which zeroes act1/act2/act3
        stats = self.statistics.aggregate_population_stats_pop(pop, self.year)
        self.statistics.store_annual_stats(self.year, stats)
        ig_stats = self.statistics.aggregate_by_income_group_pop(pop, self.year)
        self.statistics.store_income_group_stats(self.year, ig_stats)

        # Memory: cooldown counters + reset annual flags
        update_memory(pop)

        self.year += 1
        return True

    # ------------------------------------------------------------------
    # Memory recall (2015 only)
    # ------------------------------------------------------------------

    def _recall_memory(self) -> None:
        recall_memory(self.pop)

    # ------------------------------------------------------------------
    # Price management
    # ------------------------------------------------------------------

    def _load_price_trajectories(self) -> None:
        file_path = os.path.join(self.base_path, PRIMES_NL_PRICES_FILE)
        if not os.path.exists(file_path):
            print(f"Warning: Price file not found: {file_path}")
            return

        df = pd.read_csv(file_path, header=None)

        for year_idx, year in enumerate(range(2015, 2031)):
            if year_idx >= len(df):
                break
            self.price_trajectories[year] = {}
            if self.policy == "Ref":
                v = df.iloc[year_idx, 0]
                self.price_trajectories[year] = {
                    'm_p_grey': v, 'm_p_brown': v, 'm_p_green': v,
                }
            elif self.policy == "Carbon price pressure-10":
                self.price_trajectories[year] = {
                    'm_p_brown': df.iloc[year_idx, 1],
                    'm_p_grey':  df.iloc[year_idx, 2],
                    'm_p_green': M_P_GREEN_BASE,
                }
            elif self.policy == "Carbon price pressure-25":
                self.price_trajectories[year] = {
                    'm_p_brown': df.iloc[year_idx, 3],
                    'm_p_grey':  df.iloc[year_idx, 4],
                    'm_p_green': M_P_GREEN_BASE,
                }
            elif self.policy == "Carbon price pressure-50":
                self.price_trajectories[year] = {
                    'm_p_brown': df.iloc[year_idx, 5],
                    'm_p_grey':  df.iloc[year_idx, 6],
                    'm_p_green': M_P_GREEN_BASE,
                }
            elif self.policy == "Carbon price pressure-100":
                self.price_trajectories[year] = {
                    'm_p_brown': df.iloc[year_idx, 7],
                    'm_p_grey':  df.iloc[year_idx, 8],
                    'm_p_green': M_P_GREEN_BASE,
                }
            elif self.policy == "Carbon price pressure-2020":
                self.price_trajectories[year] = {
                    'm_p_brown': df.iloc[year_idx, 9],
                    'm_p_grey':  df.iloc[year_idx, 10],
                    'm_p_green': M_P_GREEN_BASE,
                }

    def _update_prices(self) -> None:
        if self.year in self.price_trajectories:
            self.prices = self.price_trajectories[self.year].copy()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def run(self) -> bool:
        if not self.initialize():
            print("âœ— Model initialization failed")
            return False

        while self.year <= MODEL_END_YEAR:
            self.step()

        return True

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        return self.statistics.get_cumulative_stats(MODEL_START_YEAR, MODEL_END_YEAR)

    def export_results(self) -> List[str]:
        return self.exporter.export_all_results(self, MODEL_START_YEAR, MODEL_END_YEAR)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _sanitize_string(self, value: str) -> str:
        safe = re.sub(r'[^A-Za-z0-9_-]+', '_', value.strip().replace(' ', '_'))
        return safe.strip('_')[:80]

    def _generate_run_id(self) -> str:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        parts = [
            timestamp,
            self._sanitize_string(self.case_study),
            self._sanitize_string(self.scenario),
            self._sanitize_string(self.policy),
        ]
        if self.run_label:
            parts.append(self._sanitize_string(self.run_label))
        return "_".join(parts)

    def _print_agent_summary(self) -> None:
        pop = self.pop
        n   = pop.N
        if n == 0:
            return
        grey_n  = int((pop.flag == 0).sum())
        brown_n = int((pop.flag == 1).sum())
        green_n = int((pop.flag == 2).sum())
        print(f"\nAgent Summary (Baseline 2015):")
        print(f"  Total Households: {n}")
        print(f"  Energy Source Distribution:")
        print(f"    - Gray electricity:  {grey_n}  ({grey_n/n*100:.1f}%)")
        print(f"    - Brown electricity: {brown_n} ({brown_n/n*100:.1f}%)")
        print(f"    - Green electricity: {green_n} ({green_n/n*100:.1f}%)")
        print(f"  Average Income:      â‚¬{float(pop.h_income.mean()):,.0f}")
        print(f"  Average Consumption: {float(pop.h_q.mean()):,.0f} kWh/year")

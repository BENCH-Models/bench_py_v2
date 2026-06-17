"""
Statistics aggregation for BENCH model outputs.
Upgrade 4: key population-level sums replaced with numpy array operations.
"""

from typing import Dict, List, Tuple

import numpy as np

from utils.constants import (
    FLAG_NAMES,
    DWELLING_LABEL_NAMES,
    EMISSIONS_FACTOR_GRAY,
    EMISSIONS_FACTOR_BROWN,
    EMISSIONS_FACTOR_GREEN,
)


class StatisticsAggregator:
    """
    Aggregates individual household data to population-level statistics.
    Handles income group and dwelling label breakdowns.
    """

    def __init__(self):
        self.annual_stats: Dict = {}
        self.income_group_annual_stats: Dict = {}  # {year: {group_id: {...}}}
        # Cached static arrays (income groups never change after init)
        self._ig_groups: np.ndarray = None
        self._ig_income: np.ndarray = None
        self._ig_masks: Dict[int, np.ndarray] = {}

    def _build_income_group_cache(self, households: List) -> None:
        self._ig_groups = np.array([hh.h_income_group for hh in households], dtype=np.int8)
        self._ig_income = np.array([hh.h_income        for hh in households], dtype=np.float64)
        self._ig_masks  = {g: self._ig_groups == g for g in range(1, 8)}

    def aggregate_population_stats(self, households: List, year: int) -> Dict:
        """
        Calculate population-level statistics for the given year.
        Vectorized: extracts attribute arrays once, then uses numpy operations.
        """
        n = len(households)
        if n == 0:
            return {'year': year, 'n_households': 0}

        # --- extract arrays once ---
        flags    = np.array([hh.flag    for hh in households], dtype=np.int8)
        h_q      = np.array([hh.h_q     for hh in households], dtype=np.float64)
        act1     = np.array([hh.act1    for hh in households], dtype=bool)
        act2     = np.array([hh.act2    for hh in households], dtype=bool)
        act3     = np.array([hh.act3    for hh in households], dtype=bool)
        h_conserv       = np.array([hh.h_conserv      for hh in households], dtype=np.float64)
        h_invest_save   = np.array([hh.h_invest_save  for hh in households], dtype=np.float64)
        h_invest        = np.array([hh.h_invest        for hh in households], dtype=np.float64)
        h_conserv_p     = np.array([hh.h_conserv_p    for hh in households], dtype=np.float64)
        h_switch        = np.array([hh.h_switch        for hh in households], dtype=np.float64)
        h_aware         = np.array([hh.h_aware         for hh in households], dtype=np.float64)
        guilt_high      = np.array([hh.guilt == 'H'    for hh in households], dtype=bool)
        h_motiv_avg     = np.array([sum(hh.h_motiv.values()) / 3 for hh in households], dtype=np.float64)
        em_avoided      = np.array([sum(hh.em_avoided.values()) for hh in households], dtype=np.float64)

        # --- consumption ---
        total_grey  = float(h_q[flags == 0].sum())
        total_brown = float(h_q[flags == 1].sum())
        total_green = float(h_q[flags == 2].sum())
        total_consumption = total_grey + total_brown + total_green
        green_share = total_green / total_consumption * 100 if total_consumption > 0 else 0.0

        # --- actions ---
        action_1_count = int(act1.sum())
        action_2_count = int(act2.sum())
        action_3_count = int(act3.sum())
        total_action_count = int((act1 | act2 | act3).sum())

        # --- energy / financial ---
        total_energy_saved    = float((h_conserv + h_invest_save).sum())
        total_investment      = float(h_invest.sum())
        total_conservation_savings = float(h_conserv_p.sum())
        total_switching_benefit    = float(h_switch.sum())

        # --- emissions ---
        total_emissions_avoided = float(em_avoided.sum())
        ef = np.where(flags == 0, EMISSIONS_FACTOR_GRAY,
             np.where(flags == 1, EMISSIONS_FACTOR_BROWN, EMISSIONS_FACTOR_GREEN))
        total_emissions = float((h_q * ef).sum())

        # --- behavioural ---
        avg_awareness  = float(h_aware.mean())
        high_guilt_count = int(guilt_high.sum())
        avg_motivation = float(h_motiv_avg.mean())

        return {
            'year': year,
            'n_households': n,
            # Consumption
            'consumption_grey': total_grey,
            'consumption_brown': total_brown,
            'consumption_green': total_green,
            'consumption_total': total_consumption,
            'green_share_percent': green_share,
            # Actions
            'action_1_count': action_1_count,
            'action_2_count': action_2_count,
            'action_3_count': action_3_count,
            'action_total_count': total_action_count,
            'action_1_percent': action_1_count / n * 100,
            'action_2_percent': action_2_count / n * 100,
            'action_3_percent': action_3_count / n * 100,
            # Financial
            'total_investment': total_investment,
            'total_conservation_savings_money': total_conservation_savings,
            'total_switching_benefit': total_switching_benefit,
            'total_energy_saved_kwh': total_energy_saved,
            # Environmental
            'total_emissions_avoided_kg_co2': total_emissions_avoided,
            'emissions_avoided_per_capita': total_emissions_avoided / n,
            'total_emissions_kg_co2': total_emissions,
            'total_emissions_tons_co2': total_emissions / 1000.0,
            'emissions_per_capita_kg_co2': total_emissions / n,
            'emissions_per_capita_tons': total_emissions / n / 1000.0,
            'co2_emitted_tons_per_capita': total_emissions / n / 1000.0,
            # Behavioural
            'avg_awareness': avg_awareness,
            'high_guilt_count': high_guilt_count,
            'high_guilt_percent': high_guilt_count / n * 100,
            'avg_motivation': avg_motivation,
        }

    def aggregate_by_income_group(self, households: List, year: int) -> Dict[int, Dict]:
        """Aggregate statistics by income group (1-7)."""
        # Build static cache on first call (income groups never change)
        if self._ig_groups is None:
            self._build_income_group_cache(households)

        # Dynamic arrays only (change each year)
        h_q      = np.array([hh.h_q      for hh in households], dtype=np.float64)
        act1     = np.array([hh.act1      for hh in households], dtype=bool)
        act2     = np.array([hh.act2      for hh in households], dtype=bool)
        act3     = np.array([hh.act3      for hh in households], dtype=bool)
        h_invest   = np.array([hh.h_invest                    for hh in households], dtype=np.float64)
        em_avoided = np.array([sum(hh.em_avoided.values())    for hh in households], dtype=np.float64)
        h_aware    = np.array([hh.h_aware                     for hh in households], dtype=np.float64)
        income     = self._ig_income  # static — doesn't change within a run

        stats_by_group: Dict[int, Dict] = {}
        for g in range(1, 8):
            mask = self._ig_masks[g]
            cnt = int(mask.sum())
            if cnt == 0:
                stats_by_group[g] = {}
                continue
            stats_by_group[g] = {
                'count': cnt,
                'avg_income': float(income[mask].mean()),
                'avg_consumption': float(h_q[mask].mean()),
                'action_1_count': int(act1[mask].sum()),
                'action_2_count': int(act2[mask].sum()),
                'action_3_count': int(act3[mask].sum()),
                'total_investment': float(h_invest[mask].sum()),
                'total_emissions_avoided': float(em_avoided[mask].sum()),
                'avg_awareness': float(h_aware[mask].mean()),
            }
        return stats_by_group

    def aggregate_by_dwelling_label(self, households: List, year: int) -> Dict[int, Dict]:
        """Aggregate statistics by dwelling energy label (1-6, A-F)."""
        labels   = np.array([hh.dw_el    for hh in households], dtype=np.int8)
        h_q      = np.array([hh.h_q      for hh in households], dtype=np.float64)
        act1     = np.array([hh.act1      for hh in households], dtype=bool)
        act2     = np.array([hh.act2      for hh in households], dtype=bool)
        act3     = np.array([hh.act3      for hh in households], dtype=bool)
        h_invest = np.array([hh.h_invest  for hh in households], dtype=np.float64)
        h_conserv = np.array([hh.h_conserv for hh in households], dtype=np.float64)

        stats_by_label: Dict[int, Dict] = {}
        for lbl in range(1, 7):
            mask = labels == lbl
            cnt = int(mask.sum())
            if cnt == 0:
                stats_by_label[lbl] = {}
                continue
            stats_by_label[lbl] = {
                'count': cnt,
                'avg_consumption': float(h_q[mask].mean()),
                'action_1_count': int(act1[mask].sum()),
                'action_2_count': int(act2[mask].sum()),
                'action_3_count': int(act3[mask].sum()),
                'total_investment': float(h_invest[mask].sum()),
                'total_energy_saved': float(h_conserv[mask].sum()),
            }
        return stats_by_label

    def store_annual_stats(self, year: int, stats: Dict) -> None:
        self.annual_stats[year] = stats

    def store_income_group_stats(self, year: int, stats: Dict[int, Dict]) -> None:
        self.income_group_annual_stats[year] = stats

    def get_cumulative_stats(self, start_year: int, end_year: int) -> Dict:
        cumulative = {
            'total_investment': 0.0,
            'total_energy_saved': 0.0,
            'total_emissions_avoided': 0.0,
            'total_conservation_savings': 0.0,
            'actions_cumulative': 0,
            'years': end_year - start_year + 1,
        }
        for year in range(start_year, end_year + 1):
            if year in self.annual_stats:
                s = self.annual_stats[year]
                cumulative['total_investment']          += s.get('total_investment', 0)
                cumulative['total_energy_saved']        += s.get('total_energy_saved_kwh', 0)
                cumulative['total_emissions_avoided']   += s.get('total_emissions_avoided_kg_co2', 0)
                cumulative['total_conservation_savings'] += s.get('total_conservation_savings_money', 0)
                cumulative['actions_cumulative']        += s.get('action_total_count', 0)
        return cumulative

    def get_trajectory(self, variable: str, start_year: int,
                       end_year: int) -> List[Tuple[int, float]]:
        return [
            (year, self.annual_stats[year].get(variable, 0))
            for year in range(start_year, end_year + 1)
            if year in self.annual_stats
        ]

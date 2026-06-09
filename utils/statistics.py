"""
Statistics aggregation for BENCH model outputs
"""

from typing import Dict, List, Tuple
from utils.constants import (
    FLAG_NAMES,
    DWELLING_LABEL_NAMES,
    EMISSIONS_FACTOR_FF,
    EMISSIONS_FACTOR_LCE,
    EMISSIONS_FACTOR_SLCE,
)


class StatisticsAggregator:
    """
    Aggregates individual household data to population-level statistics.
    Handles income group and dwelling label breakdowns.
    """
    
    def __init__(self):
        """Initialize statistics tracker."""
        self.annual_stats = {}  # Dict[year] = statistics dictionary
    
    def aggregate_population_stats(self, households: List, year: int) -> Dict:
        """
        Calculate population-level statistics for given year.
        
        Args:
            households: List of all Household objects
            year: Current year
            
        Returns:
            Dictionary with population statistics
        """
        n_households = len(households)
        
        # === CONSUMPTION ===
        total_ff = sum(hh.h_q for hh in households if hh.flag == 0)
        total_lce = sum(hh.h_q for hh in households if hh.flag == 1)
        total_slce = sum(hh.h_q for hh in households if hh.flag == 2)
        total_consumption = total_ff + total_lce + total_slce
        
        lce_share = (total_lce + total_slce) / total_consumption * 100 if total_consumption > 0 else 0
        
        # === ACTIONS ===
        action_1_count = sum(1 for hh in households if hh.act1)
        action_2_count = sum(1 for hh in households if hh.act2)
        action_3_count = sum(1 for hh in households if hh.act3)
        total_action_count = sum(1 for hh in households if hh.act1 or hh.act2 or hh.act3)
        
        # === ENERGY SAVINGS ===
        total_energy_saved = sum(hh.h_conserv + hh.h_invest_save for hh in households)
        total_investment = sum(hh.h_invest for hh in households)
        total_conservation_savings = sum(hh.h_conserv_p for hh in households)
        total_switching_benefit = sum(hh.h_switch for hh in households)
        
        # === EMISSIONS ===
        total_emissions_avoided = sum(sum(hh.em_avoided) for hh in households)
        total_emissions = 0.0
        for hh in households:
            if hh.flag == 0:
                total_emissions += hh.h_q * EMISSIONS_FACTOR_FF
            elif hh.flag == 1:
                total_emissions += hh.h_q * EMISSIONS_FACTOR_LCE
            elif hh.flag == 2:
                total_emissions += hh.h_q * EMISSIONS_FACTOR_SLCE
        
        # === BEHAVIORAL METRICS ===
        avg_awareness = sum(hh.h_aware for hh in households) / n_households if n_households > 0 else 0
        high_guilt_count = sum(1 for hh in households if hh.guilt == 'H')
        avg_motivation = sum(sum(hh.h_motiv) / 3 for hh in households) / n_households if n_households > 0 else 0
        
        stats = {
            'year': year,
            'n_households': n_households,
            
            # Consumption
            'consumption_ff': total_ff,
            'consumption_lce': total_lce,
            'consumption_slce': total_slce,
            'consumption_total': total_consumption,
            'lce_share_percent': lce_share,
            
            # Actions
            'action_1_count': action_1_count,
            'action_2_count': action_2_count,
            'action_3_count': action_3_count,
            'action_total_count': total_action_count,
            'action_1_percent': action_1_count / n_households * 100 if n_households > 0 else 0,
            'action_2_percent': action_2_count / n_households * 100 if n_households > 0 else 0,
            'action_3_percent': action_3_count / n_households * 100 if n_households > 0 else 0,
            
            # Financial
            'total_investment': total_investment,
            'total_conservation_savings_money': total_conservation_savings,
            'total_switching_benefit': total_switching_benefit,
            'total_energy_saved_kwh': total_energy_saved,
            
            # Environmental
            'total_emissions_avoided_kg_co2': total_emissions_avoided,
            'emissions_avoided_per_capita': total_emissions_avoided / n_households if n_households > 0 else 0,
            'total_emissions_kg_co2': total_emissions,
            'total_emissions_tons_co2': total_emissions / 1000.0,
            'emissions_per_capita_kg_co2': total_emissions / n_households if n_households > 0 else 0,
            'emissions_per_capita_tons': (total_emissions / n_households / 1000.0) if n_households > 0 else 0,
            'co2_emitted_tons_per_capita': (total_emissions / n_households / 1000.0) if n_households > 0 else 0,
            
            # Behavioral
            'avg_awareness': avg_awareness,
            'high_guilt_count': high_guilt_count,
            'high_guilt_percent': high_guilt_count / n_households * 100 if n_households > 0 else 0,
            'avg_motivation': avg_motivation,
        }
        
        return stats
    
    def aggregate_by_income_group(self, households: List, year: int) -> Dict[int, Dict]:
        """
        Aggregate statistics by income group (1-7).
        
        Args:
            households: List of all Household objects
            year: Current year
            
        Returns:
            Dictionary mapping income_group -> statistics
        """
        stats_by_group = {}
        
        for income_group in range(1, 8):
            group_hhs = [hh for hh in households if hh.h_income_group == income_group]
            
            if len(group_hhs) == 0:
                stats_by_group[income_group] = {}
                continue
            
            stats = {
                'count': len(group_hhs),
                'avg_income': sum(hh.h_income for hh in group_hhs) / len(group_hhs),
                'avg_consumption': sum(hh.h_q for hh in group_hhs) / len(group_hhs),
                'action_1_count': sum(1 for hh in group_hhs if hh.act1),
                'action_2_count': sum(1 for hh in group_hhs if hh.act2),
                'action_3_count': sum(1 for hh in group_hhs if hh.act3),
                'total_investment': sum(hh.h_invest for hh in group_hhs),
                'total_emissions_avoided': sum(sum(hh.em_avoided) for hh in group_hhs),
                'avg_awareness': sum(hh.h_aware for hh in group_hhs) / len(group_hhs),
            }
            
            stats_by_group[income_group] = stats
        
        return stats_by_group
    
    def aggregate_by_dwelling_label(self, households: List, year: int) -> Dict[int, Dict]:
        """
        Aggregate statistics by dwelling energy label (1-6, A-F).
        
        Args:
            households: List of all Household objects
            year: Current year
            
        Returns:
            Dictionary mapping label -> statistics
        """
        stats_by_label = {}
        
        for label in range(1, 7):
            label_hhs = [hh for hh in households if hh.dw_el == label]
            
            if len(label_hhs) == 0:
                stats_by_label[label] = {}
                continue
            
            stats = {
                'count': len(label_hhs),
                'avg_consumption': sum(hh.h_q for hh in label_hhs) / len(label_hhs),
                'action_1_count': sum(1 for hh in label_hhs if hh.act1),
                'action_2_count': sum(1 for hh in label_hhs if hh.act2),
                'action_3_count': sum(1 for hh in label_hhs if hh.act3),
                'total_investment': sum(hh.h_invest for hh in label_hhs),
                'total_energy_saved': sum(hh.h_conserv for hh in label_hhs),
            }
            
            stats_by_label[label] = stats
        
        return stats_by_label
    
    def store_annual_stats(self, year: int, stats: Dict) -> None:
        """Store annual statistics for later retrieval."""
        self.annual_stats[year] = stats
    
    def get_cumulative_stats(self, start_year: int, end_year: int) -> Dict:
        """
        Calculate cumulative statistics across a time period.
        
        Args:
            start_year: Start year (inclusive)
            end_year: End year (inclusive)
            
        Returns:
            Dictionary with cumulative statistics
        """
        cumulative = {
            'total_investment': 0,
            'total_energy_saved': 0,
            'total_emissions_avoided': 0,
            'total_conservation_savings': 0,
            'actions_cumulative': 0,
            'years': end_year - start_year + 1,
        }
        
        for year in range(start_year, end_year + 1):
            if year in self.annual_stats:
                stats = self.annual_stats[year]
                cumulative['total_investment'] += stats.get('total_investment', 0)
                cumulative['total_energy_saved'] += stats.get('total_energy_saved_kwh', 0)
                cumulative['total_emissions_avoided'] += stats.get('total_emissions_avoided_kg_co2', 0)
                cumulative['total_conservation_savings'] += stats.get('total_conservation_savings_money', 0)
                cumulative['actions_cumulative'] += stats.get('action_total_count', 0)
        
        return cumulative
    
    def get_trajectory(self, variable: str, start_year: int, 
                      end_year: int) -> List[Tuple[int, float]]:
        """
        Get time series of a specific variable.
        
        Args:
            variable: Variable name (e.g., 'lce_share_percent')
            start_year: Start year
            end_year: End year
            
        Returns:
            List of (year, value) tuples
        """
        trajectory = []
        
        for year in range(start_year, end_year + 1):
            if year in self.annual_stats:
                value = self.annual_stats[year].get(variable, 0)
                trajectory.append((year, value))
        
        return trajectory

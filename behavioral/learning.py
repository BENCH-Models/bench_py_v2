"""
Social learning and network influence mechanisms
"""

from typing import Dict, List
import random
from utils.constants import (
    ACTION_INVESTMENT,
    ACTION_SWITCHING,
    BEHAVIORAL_SCALE_MAX,
    MEMORY_RULES
)

class LearningMechanism:
    """
    Implements learning and social influence for households.
    Tracks peer learning and memory effects.
    """
    
    def __init__(self):
        """Initialize learning mechanism."""
        self.network_influence = {}  # Track influence between households
        self.learning_history = {}  # Historical learning data
    
    def apply_learning(self, source_household, neighboring_households: List,
                       year: int, learning_type: str) -> None:
        """
        Apply the selected learning algorithm to neighbors.

        Learning modes:
        - No learning: no neighbor influence.
        - Fast adaptation: all households in the active neighborhood learn from each other.
        - Slow adaptation: each active household interacts with up to two neighbors.
        """

        if learning_type == "Fast adaptation":
            neighbors_subsample = list(neighboring_households)
        elif learning_type == "Slow adaptation":
            sample_count = min(2, len(neighboring_households))
            neighbors_subsample = random.sample(neighboring_households, sample_count)
        else:
           ValueError(f"Unknown learning type: {learning_type}")

        # Build neighborhood statistics including the active source household.

        know_values = [hh.know for hh in neighbors_subsample]
        awareness_values = [hh.h_aware for hh in neighbors_subsample]
        su_invest_values = [hh.su_nor[ACTION_INVESTMENT] for hh in neighbors_subsample]
        pbc_switch_values = [hh.pbc[ACTION_SWITCHING] for hh in neighbors_subsample]

        def mean(values):
            return sum(values) / len(values) if values else 0.0

        def median(values):
            sorted_values = sorted(values)
            n = len(sorted_values)
            if n == 0:
                return 0.0
            mid = n // 2
            return (sorted_values[mid] if n % 2 == 1 else (sorted_values[mid - 1] + sorted_values[mid]) / 2.0)

        target_know = max(mean(know_values), median(know_values))
        target_awareness = max(mean(awareness_values), median(awareness_values))
        target_su_invest = max(mean(su_invest_values), median(su_invest_values))
        target_pbc_switch = max(mean(pbc_switch_values), median(pbc_switch_values))

        for neighbor in neighbors_subsample:#CENTRAL INDIVIDUAL TEACHES NEIGHBORS
            if neighbor.know < target_know:
                neighbor.know = min(neighbor.know * 1.05, BEHAVIORAL_SCALE_MAX)

            if neighbor.h_aware < target_awareness:
                neighbor.h_aware = min(neighbor.h_aware * 1.05, BEHAVIORAL_SCALE_MAX)
                neighbor.update_awareness()

            if neighbor.su_nor[ACTION_INVESTMENT] < target_su_invest:
                neighbor.su_nor[ACTION_INVESTMENT] = min(
                    neighbor.su_nor[ACTION_INVESTMENT] * 1.05,
                    BEHAVIORAL_SCALE_MAX
                )

            if neighbor.pbc[ACTION_SWITCHING] < target_pbc_switch:
                neighbor.pbc[ACTION_SWITCHING] = min(
                    neighbor.pbc[ACTION_SWITCHING] * 1.05,
                    BEHAVIORAL_SCALE_MAX
                )

        # Reinforce perceived behavioral control for the active household itself.
        for action_index in range(len(source_household.pbc)):
            source_household.pbc[action_index] = min(
                source_household.pbc[action_index] * 1.05,
                BEHAVIORAL_SCALE_MAX
            )


    def recall_memory(self, household, initial_actions: Dict, 
                     income_group: int, energy_flag: int,
                     case_study: str) -> None:
        """
        Recall and apply initial memory/historical behavior pattern (2015 only).
        
        Based on empirical data of what households in different groups
        had already done by 2015.
        
        Args:
            household: Household object
            initial_actions: Pre-loaded historical action data
            income_group: Household income group (1-7)
            energy_flag: Current energy source (0, 1, 2)
            case_study: "Netherlands-Overijssel" or "Spain-Navarre"
        """
        if case_study != "Netherlands-Overijssel":
            print(f"Memory recall only implemented for Netherlands-Overijssel. Skipping for {case_study}.")
            return  # Only apply to Netherlands for now
        
        # Memory recall rules (from NetLogo recallmemory procedure)
        # Different probabilities for different income groups and energy sources
        

        if income_group not in MEMORY_RULES:
            print(f"Warning: No memory rules for income group {income_group}")
            return 
        
        rules = MEMORY_RULES[income_group].get(energy_flag, {})
        
        for action_name, (probability, value) in rules.items():
            if random.random() < probability:
                if action_name == 'act11':
                    household.act11 = value
                    household.act1 = value
                    household.hh_actions[0] = 1
                elif action_name == 'act12':
                    household.act12 = value
                    household.act1 = value
                    household.hh_actions[1] = 1
                elif action_name == 'act31':
                    household.act31 = value
                    household.act3 = value
                    household.hh_actions[4] = 1
                elif action_name == 'act32':
                    household.act32 = value
                    household.act3 = value
                    household.hh_actions[5] = 1
                elif action_name == 'act21':
                    household.act21 = value
                    household.act2 = value
                    household.act50 = value
                    household.hh_actions[2] = 1
                elif action_name == 'act40':
                    household.act40 = value
                    household.act2 = value
                    household.act50 = value
                    household.hh_actions[3] = 1
    
    def update_memory(self, household, year: int) -> None:
        """
        Update household memory with current year's experience.
        
        Args:
            household: Household object
            year: Current simulation year
        """
        if year not in household.memory:
            household.memory[year] = {}
        
        household.memory[year].update({
            'actions': household.hh_actions.copy(),
            'consumption': household.h_q,
            'awareness': household.h_aware,
            'investment': household.h_invest,
            'savings': household.h_conserv,
        })


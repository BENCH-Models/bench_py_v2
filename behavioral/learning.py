"""
Social learning and network influence mechanisms
"""

from typing import Dict, List
import random
from utils.constants import (
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
        Matches NetLogo's learn procedure.
        
        Learning modes:
        - No learning: no neighbor influence (already handled in step())
        - Fast adaptation: source household influences ALL neighbors
        - Slow adaptation: source household influences up to 2 random neighbors
        - Observation: (not in your constants but in NetLogo) - higher influence
        - Promote switching: (not in your constants) - focused on switching
        """
        if learning_type == "No learning":
            return
        
        # === STEP 1: Reinforce PBC for the source household (active learner) ===
        # NetLogo does this BEFORE neighbor learning
        if learning_type == "Fast adaptation":
            # 5% increase for all PBC
            for action in source_household.pbc:
                if source_household.pbc[action] < 6.5:
                    source_household.pbc[action] = min(
                        source_household.pbc[action] * 1.05,
                        BEHAVIORAL_SCALE_MAX
                    )
        elif learning_type == "Slow adaptation":
            # 2% increase for all PBC (slower learning)
            for action in source_household.pbc:
                if source_household.pbc[action] < 6.5:
                    source_household.pbc[action] = min(
                        source_household.pbc[action] * 1.02,
                        BEHAVIORAL_SCALE_MAX
                    )
        # Note: "Observation" and "Promote switching" would have different rates
        
        # === STEP 2: Select neighbors to influence ===
        if not neighboring_households:
            return
        
        if learning_type == "Fast adaptation":
            neighbors_to_influence = list(neighboring_households)
        elif learning_type == "Slow adaptation":
            sample_count = min(2, len(neighboring_households))
            neighbors_to_influence = random.sample(neighboring_households, sample_count)
        else:
            # Unknown learning type - no learning
            return
        
        # === STEP 3: For each neighbor, calculate target values and apply learning ===
        for neighbor in neighbors_to_influence:
            # Get all neighbors of this neighbor (for statistics)
            # In NetLogo, this uses "households-on neighbors" which is the 8-cell neighborhood
            # For simplicity, we use the same neighboring_households list
            neighbor_hood = neighboring_households
            
            # Calculate target values (max of mean and median of neighborhood)
            # Knowledge (know)
            know_values = [hh.know for hh in neighbor_hood if hasattr(hh, 'know')]
            if know_values:
                know_mean = sum(know_values) / len(know_values)
                know_median = sorted(know_values)[len(know_values)//2]
                target_know = max(know_mean, know_median)
            else:
                target_know = neighbor.know
            
            # Climate/Environment Awareness (cee_aw)
            cee_values = [hh.cee_aw for hh in neighbor_hood if hasattr(hh, 'cee_aw')]
            if cee_values:
                cee_mean = sum(cee_values) / len(cee_values)
                cee_median = sorted(cee_values)[len(cee_values)//2]
                target_cee = max(cee_mean, cee_median)
            else:
                target_cee = neighbor.cee_aw
            
            # Education Awareness (ed_aw)
            ed_values = [hh.ed_aw for hh in neighbor_hood if hasattr(hh, 'ed_aw')]
            if ed_values:
                ed_mean = sum(ed_values) / len(ed_values)
                ed_median = sorted(ed_values)[len(ed_values)//2]
                target_ed = max(ed_mean, ed_median)
            else:
                target_ed = neighbor.ed_aw
            
            # Personal Norm for Investment (per_nab['investment'])
            per_invest_values = [hh.per_nab.get('investment', 0) for hh in neighbor_hood]
            if per_invest_values:
                per_mean = sum(per_invest_values) / len(per_invest_values)
                per_median = sorted(per_invest_values)[len(per_invest_values)//2]
                target_per_invest = max(per_mean, per_median)
            else:
                target_per_invest = neighbor.per_nab.get('investment', 0)
            
            # Social Norm for Investment (su_nor['investment'])
            su_invest_values = [hh.su_nor.get('investment', 0) for hh in neighbor_hood]
            if su_invest_values:
                su_mean = sum(su_invest_values) / len(su_invest_values)
                su_median = sorted(su_invest_values)[len(su_invest_values)//2]
                target_su_invest = max(su_mean, su_median)
            else:
                target_su_invest = neighbor.su_nor.get('investment', 0)
            
            # PBC for Switching (pbc['switching'])
            pbc_switch_values = [hh.pbc.get('switching', 0) for hh in neighbor_hood]
            if pbc_switch_values:
                pbc_mean = sum(pbc_switch_values) / len(pbc_switch_values)
                pbc_median = sorted(pbc_switch_values)[len(pbc_switch_values)//2]
                target_pbc_switch = max(pbc_mean, pbc_median)
            else:
                target_pbc_switch = neighbor.pbc.get('switching', 0)
            
            # === STEP 4: Apply learning to neighbor ===
            # Knowledge update
            if neighbor.know < target_know and neighbor.know < 6.5:
                if learning_type == "Fast adaptation":
                    neighbor.know = min(neighbor.know * 1.05, BEHAVIORAL_SCALE_MAX)
                elif learning_type == "Slow adaptation":
                    neighbor.know = min(neighbor.know * 1.05, BEHAVIORAL_SCALE_MAX)
            
            # Climate awareness update
            if neighbor.cee_aw < target_cee and neighbor.cee_aw < 6.5:
                if learning_type == "Fast adaptation":
                    neighbor.cee_aw = min(neighbor.cee_aw * 1.05, BEHAVIORAL_SCALE_MAX)
                elif learning_type == "Slow adaptation":
                    neighbor.cee_aw = min(neighbor.cee_aw * 1.05, BEHAVIORAL_SCALE_MAX)
            
            # Education awareness update
            if neighbor.ed_aw < target_ed and neighbor.ed_aw < 6.5:
                if learning_type == "Fast adaptation":
                    neighbor.ed_aw = min(neighbor.ed_aw * 1.05, BEHAVIORAL_SCALE_MAX)
                elif learning_type == "Slow adaptation":
                    neighbor.ed_aw = min(neighbor.ed_aw * 1.05, BEHAVIORAL_SCALE_MAX)
            
            # Update awareness after knowledge changes
            neighbor.update_awareness()
            
            # Personal norm for investment update
            if neighbor.per_nab.get('investment', 0) < target_per_invest:
                if learning_type == "Fast adaptation":
                    neighbor.per_nab['investment'] = min(
                        neighbor.per_nab.get('investment', 0) * 1.05,
                        BEHAVIORAL_SCALE_MAX
                    )
                elif learning_type == "Slow adaptation":
                    neighbor.per_nab['investment'] = min(
                        neighbor.per_nab.get('investment', 0) * 1.05,
                        BEHAVIORAL_SCALE_MAX
                    )
            
            # Social norm for investment update
            if neighbor.su_nor.get('investment', 0) < target_su_invest:
                if learning_type == "Fast adaptation":
                    neighbor.su_nor['investment'] = min(
                        neighbor.su_nor.get('investment', 0) * 1.07,
                        BEHAVIORAL_SCALE_MAX
                    )
                elif learning_type == "Slow adaptation":
                    neighbor.su_nor['investment'] = min(
                        neighbor.su_nor.get('investment', 0) * 1.07,
                        BEHAVIORAL_SCALE_MAX
                    )
            
            # PBC for switching update
            if neighbor.pbc.get('switching', 0) < target_pbc_switch:
                if learning_type == "Fast adaptation":
                    neighbor.pbc['switching'] = min(
                        neighbor.pbc.get('switching', 0) * 1.05,
                        BEHAVIORAL_SCALE_MAX
                    )
                elif learning_type == "Slow adaptation":
                    neighbor.pbc['switching'] = min(
                        neighbor.pbc.get('switching', 0) * 1.05,
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
        
        rules = MEMORY_RULES[income_group].get(energy_flag)
        
        if rules is None:
            return
        
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
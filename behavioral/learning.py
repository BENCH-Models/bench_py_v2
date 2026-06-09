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
    
    def learn_from_peers(self, household, neighboring_households: List, 
                        year: int) -> None:
        """
        Update household attributes based on neighboring households' experiences.
        
        Mechanisms:
        - Knowledge spreads from high-awareness to low-awareness neighbors
        - Action success influences motivation
        - Price experiences spread through network
        
        Args:
            household: Household object
            neighboring_households: List of nearby household objects
            year: Current simulation year
        """
        if not neighboring_households:
            return
        
        # Average peer characteristics (simplified social learning)
        avg_knowledge = 0.0
        avg_awareness = 0.0
        actions_count = 0
        
        for neighbor in neighboring_households:
            avg_knowledge += neighbor.know
            avg_awareness += neighbor.h_aware
            actions_count += sum(neighbor.hh_actions)
        
        n = len(neighboring_households)
        if n == 0:
            return
        
        avg_knowledge /= n
        avg_awareness /= n
        
        # Update own knowledge with influence factor (0.1 = 10% of peer influence)
        influence_factor = 0.1
        if household.know < avg_knowledge:
            household.know += influence_factor * (avg_knowledge - household.know)
            household.know = min(household.know, BEHAVIORAL_SCALE_MAX)
        
        if household.cee_aw < avg_awareness:
            household.cee_aw += influence_factor * (avg_awareness - household.cee_aw)
            household.cee_aw = min(household.cee_aw, BEHAVIORAL_SCALE_MAX)
        
        # Social norms: see peers take actions
        if actions_count > 0:
            action_fraction = actions_count / n
            # Increase social norms if peers are acting
            for i in range(3):
                household.su_nor[i] = min(
                    household.su_nor[i] + influence_factor * action_fraction,
                    BEHAVIORAL_SCALE_MAX
                )
        
        # Store learning event
        self._record_learning(household, year, avg_knowledge, avg_awareness)

    def apply_learning(self, source_household, neighboring_households: List,
                       year: int, learning_type: str) -> None:
        """
        Apply the selected learning algorithm to neighbors.
        
        Learning modes are based on NetLogo logic:
        - No learning: no neighbor influence.
        - Fast adaptation: rapid transmission from households that invested.
        - Slow adaptation: weaker influence over the same behaviors.
        - Observation: peers observe actions and slowly adjust awareness.
        - Promote switching: emphasis on switching norms and PBC.
        """
        if learning_type == "No learning" or year < 2016:
            return

        if not source_household.act1:
            return

        if not neighboring_households:
            return

        if learning_type == "Fast adaptation":
            sample_count = min(2, len(neighboring_households))
            influence = 0.35
        elif learning_type == "Slow adaptation":
            sample_count = min(2, len(neighboring_households))
            influence = 0.15
        elif learning_type == "Observation":
            sample_count = min(5, len(neighboring_households))
            influence = 0.10
        elif learning_type == "Promote switching":
            sample_count = min(3, len(neighboring_households))
            influence = 0.25
        else:
            sample_count = min(2, len(neighboring_households))
            influence = 0.10

        neighbors = random.sample(neighboring_households, sample_count)

        for neighbor in neighbors:
            if learning_type in ["Fast adaptation", "Slow adaptation"]:
                knowledge_gain = influence * (source_household.know - neighbor.know)
                awareness_gain = influence * (source_household.h_aware - neighbor.h_aware)
                neighbor.know = min(max(neighbor.know + max(knowledge_gain, 0.0), 0.0), BEHAVIORAL_SCALE_MAX)
                neighbor.cee_aw = min(max(neighbor.cee_aw + max(awareness_gain, 0.0), 0.0), BEHAVIORAL_SCALE_MAX)
                neighbor.update_awareness()
                neighbor.su_nor[ACTION_INVESTMENT] = min(
                    neighbor.su_nor[ACTION_INVESTMENT] + influence * 0.5,
                    BEHAVIORAL_SCALE_MAX
                )

            elif learning_type == "Observation":
                if source_household.act1:
                    neighbor.know = min(neighbor.know + influence * 0.5, BEHAVIORAL_SCALE_MAX)
                    neighbor.cee_aw = min(neighbor.cee_aw + influence * 0.25, BEHAVIORAL_SCALE_MAX)
                    neighbor.update_awareness()
                neighbor.su_nor[ACTION_INVESTMENT] = min(
                    neighbor.su_nor[ACTION_INVESTMENT] + influence * 0.25,
                    BEHAVIORAL_SCALE_MAX
                )

            elif learning_type == "Promote switching":
                neighbor.su_nor[ACTION_SWITCHING] = min(
                    neighbor.su_nor[ACTION_SWITCHING] + influence * 0.6,
                    BEHAVIORAL_SCALE_MAX
                )
                neighbor.pbc[ACTION_SWITCHING] = min(
                    neighbor.pbc[ACTION_SWITCHING] + influence * 0.4,
                    BEHAVIORAL_SCALE_MAX
                )
                neighbor.update_awareness()

        self._record_learning(source_household, year, source_household.know, source_household.h_aware)


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
    
    def _record_learning(self, household, year: int, 
                        peer_knowledge: float, peer_awareness: float) -> None:
        """Record learning event for analysis."""
        if year not in self.learning_history:
            self.learning_history[year] = []
        
        self.learning_history[year].append({
            'h_id': household.h_id,
            'knowledge_before': household.know - 0.1,  # Approximate
            'peer_knowledge': peer_knowledge,
            'awareness_change': peer_awareness - household.h_aware,
        })

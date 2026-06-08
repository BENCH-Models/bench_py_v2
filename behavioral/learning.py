"""
Social learning and network influence mechanisms
"""

from typing import Dict, List
import random
from utils.constants import BEHAVIORAL_SCALE_MAX


class LearningMechanism:
    """
    Implements learning and social influence for households.
    Tracks peer learning, regret, satisfaction, and memory effects.
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
    
    def update_satisfaction(self, household, year: int) -> None:
        """
        Update satisfaction based on action outcomes.
        
        Args:
            household: Household object
            year: Current simulation year
        """
        # Satisfaction improves if actions yielded benefits
        investment_benefit = household.h_invest_total > 0
        conservation_benefit = household.h_conserv > 0
        switching_benefit = household.h_switch > 0
        
        satisfaction_change = 0.0
        if investment_benefit:
            satisfaction_change += 0.1
        if conservation_benefit:
            satisfaction_change += 0.1
        if switching_benefit:
            satisfaction_change += 0.1
        
        household.satisfaction = min(household.satisfaction + satisfaction_change, 1.0)
    
    def update_regret(self, household, market_state: Dict, year: int) -> None:
        """
        Update regret based on divergence from optimal decisions.
        
        Args:
            household: Household object
            market_state: Current market state
            year: Current simulation year
        """
        # Regret if prices diverged unfavorably
        price_changes = market_state.get('price_changes', {})
        
        for i, (old_price, new_price) in price_changes.items():
            if old_price > new_price:
                # Prices dropped - regret for not switching earlier
                household.regret[i] = min(household.regret[i] + 0.05, 1.0)
            elif old_price < new_price:
                # Prices increased - no regret, decision was good
                household.regret[i] = max(household.regret[i] - 0.05, 0.0)
    
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
            return  # Only apply to Netherlands for now
        
        # Memory recall rules (from NetLogo recallmemory procedure)
        # Different probabilities for different income groups and energy sources
        
        memory_rules = {
            1: {  # Income group 1
                1: {  # LCE users: 57.14% did investment
                    'act11': (0.5714, True),
                    'act31': (0.1428, True),
                    'act32': (0.8572, True),
                    'act21': (0.2143, True),
                },
                0: {  # FF users: 58.14% did investment
                    'act12': (0.5814, True),
                    'act40': (0.0930, True),
                }
            },
            # Add more groups as needed...
        }
        
        if income_group not in memory_rules:
            return
        
        rules = memory_rules[income_group].get(energy_flag, {})
        
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
    
    def get_network_effect(self, household, neighboring_households: List) -> float:
        """
        Get aggregate network effect on household (0-1 scale).
        Higher = more positive influence from neighbors.
        
        Args:
            household: Household object
            neighboring_households: List of neighboring household objects
            
        Returns:
            Network effect score (0-1)
        """
        if not neighboring_households:
            return 0.5
        
        # Count neighbors who took any action
        actions_taken = sum(1 for hh in neighboring_households if sum(hh.hh_actions) > 0)
        
        effect = 0.5 + (0.5 * actions_taken / len(neighboring_households))
        
        return min(effect, 1.0)

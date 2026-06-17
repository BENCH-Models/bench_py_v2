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
                    year: int, learning_type: str, action_type: str,
                    get_neighbors_fn=None) -> None:
        """
        Apply learning for one specific action type (investment, conservation, or switching).

        Bug 2 fix: NetLogo has three independent learn blocks each gated by the action taken
        (act1 / act50 / act3) and updating only the norms for that action type.  The old
        unified block updated only investment norms regardless of which action fired.

        Bug 3 fix: NetLogo includes the source household's own attribute value when computing
        the neighbourhood target statistic for each neighbour.  The formula is:
            ngb_k = max(mean([nbr_neighbourhood_mean, source.know]),
                        median([nbr_neighbourhood_median, source.know]))
        For a two-element list mean == median only when the two values are equal, so we
        implement: (max(nbr_mean, nbr_median) + source_val) / 2

        Also corrected: per_nab / pbc / su_nor for neighbours increase unconditionally when
        < 6.5 (no comparison to a neighbourhood target), matching NetLogo exactly.
        """
        if learning_type not in ("Fast adaptation", "Slow adaptation"):
            return

        # === STEP 1: Self-reinforce PBC for the source household (action-specific) ===
        pbc_val = source_household.pbc.get(action_type, 0)
        if pbc_val < 6.5:
            rate = 1.05 if learning_type == "Fast adaptation" else 1.02
            source_household.pbc[action_type] = min(pbc_val * rate, BEHAVIORAL_SCALE_MAX)

        if not neighboring_households:
            return

        # === STEP 2: Select neighbours to influence ===
        if learning_type == "Slow adaptation":
            if len(neighboring_households) < 2:
                return  # NetLogo: only runs when count out-link-neighbors >= 2
            neighbors_to_influence = random.sample(neighboring_households, 2)
        else:
            neighbors_to_influence = list(neighboring_households)

        # === STEP 3: Update each neighbour ===
        for neighbor in neighbors_to_influence:
            # Get the neighbour's own 8-cell neighbourhood for target statistics
            if get_neighbors_fn is not None:
                nbr_nbrs = get_neighbors_fn(neighbor)
            else:
                nbr_nbrs = neighboring_households  # fallback (less accurate)

            # Bug 3: compute target as (max(nbr_mean, nbr_median) + source_val) / 2
            def _target(vals, source_val):
                if not vals:
                    return source_val
                n = len(vals)
                nbr_mean = sum(vals) / n
                s = sorted(vals)
                mid = n // 2
                nbr_median = s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2.0
                return (max(nbr_mean, nbr_median) + source_val) / 2.0

            target_know = _target([h.know for h in nbr_nbrs], source_household.know)
            target_cee  = _target([h.cee_aw for h in nbr_nbrs], source_household.cee_aw)
            target_ed   = _target([h.ed_aw for h in nbr_nbrs], source_household.ed_aw)

            # Knowledge / awareness: conditional on (< target AND < 6.5), 5% rate
            if neighbor.know < target_know and neighbor.know < 6.5:
                neighbor.know = min(neighbor.know * 1.05, BEHAVIORAL_SCALE_MAX)
            if neighbor.cee_aw < target_cee and neighbor.cee_aw < 6.5:
                neighbor.cee_aw = min(neighbor.cee_aw * 1.05, BEHAVIORAL_SCALE_MAX)
            if neighbor.ed_aw < target_ed and neighbor.ed_aw < 6.5:
                neighbor.ed_aw = min(neighbor.ed_aw * 1.05, BEHAVIORAL_SCALE_MAX)

            neighbor.update_awareness()

            # Action-specific norms: UNCONDITIONAL (just < 6.5), no target comparison
            # per_nab and pbc: 5% rate; su_nor: 7% rate (matches NetLogo Fast/Slow blocks)
            if neighbor.per_nab.get(action_type, 0) < 6.5:
                neighbor.per_nab[action_type] = min(
                    neighbor.per_nab.get(action_type, 0) * 1.05, BEHAVIORAL_SCALE_MAX
                )
            if neighbor.pbc.get(action_type, 0) < 6.5:
                neighbor.pbc[action_type] = min(
                    neighbor.pbc.get(action_type, 0) * 1.05, BEHAVIORAL_SCALE_MAX
                )
            if neighbor.su_nor.get(action_type, 0) < 6.5:
                neighbor.su_nor[action_type] = min(
                    neighbor.su_nor.get(action_type, 0) * 1.07, BEHAVIORAL_SCALE_MAX
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
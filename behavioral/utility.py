"""
Utility calculation and preference modeling for household decisions
"""

from typing import Dict, List
from utils.constants import (
    UTILITY_NORMALIZATION_FACTOR as ALPHA,
    BEHAVIORAL_SCALE_MAX
)


class UtilityCalculator:
    """
    Calculates expected and actual utilities for household actions.
    Implements normalization of budget constraints and preference modeling.
    """
    
    def __init__(self):
        """Initialize utility calculator with empty population metrics."""
        self.z_max = {}
    
    def normalize_budgets(self, households: List) -> None:
        """
        Scans the entire population to find the absolute maximum raw Z values
        across all 3 action types. Should be called ONCE per step.
        """
        z_brown_max = [0.0, 0.0, 0.0]
        z_grey_max = [0.0, 0.0, 0.0]
        z_green_max = [0.0, 0.0, 0.0]
        
        for hh in households:
            for i in range(3):
                z_brown_max[i] = max(z_brown_max[i], hh.z_brown[i])
                z_grey_max[i] = max(z_grey_max[i], hh.z_grey[i])
                z_green_max[i] = max(z_green_max[i], hh.z_green[i])
        
        self.z_max = {
            'z_brown': z_brown_max,
            'z_grey': z_grey_max,
            'z_green': z_green_max
        }

    def normalize_budget_values(self, household) -> None:
        """
        Normalizes a single household's raw Z values against the global population maximums.
        """
        z_brown_max = self.z_max.get('z_brown', [1.0, 1.0, 1.0])
        z_grey_max = self.z_max.get('z_grey', [1.0, 1.0, 1.0])
        z_green_max = self.z_max.get('z_green', [1.0, 1.0, 1.0])
        
        # Guard clause: Ensure target normalization tracking lists exist on the agent
        for attr in ['z_brown_norm', 'z_grey_norm', 'z_green_norm']:
            if not hasattr(household, attr) or not getattr(household, attr):
                setattr(household, attr, [0.0, 0.0, 0.0])

        for i in range(3):
            household.z_brown_norm[i] = (household.z_brown[i] / z_brown_max[i]) if z_brown_max[i] > 0 else 0.0
            household.z_grey_norm[i] = (household.z_grey[i] / z_grey_max[i]) if z_grey_max[i] > 0 else 0.0
            household.z_green_norm[i] = (household.z_green[i] / z_green_max[i]) if z_green_max[i] > 0 else 0.0
    
    def calculate_expected_utility(self, household, energy_source: int, 
                                   action_type: int, market_state: Dict) -> float:
        """
        Calculates the explicit expected utility score for an isolated scenario route.
        """
        # Select appropriate normalized budget constraint
        if energy_source == 0:    # GREY
            z_norm = household.z_grey_norm[action_type]
        elif energy_source == 1:  # BROWN
            z_norm = household.z_brown_norm[action_type]
        else:                     # GREEN
            z_norm = household.z_green_norm[action_type]
        
        # Resolve environmental preference parameters dynamically from agent profiles if present
        e_norm = getattr(household, 'e_norm', 0.3 if energy_source == 0 else 0.7)
        
        # Extract behavioral components
        delta = household.delta[action_type]
        K = household.K
        M = household.M[action_type]
        pbc_norm = household.pbc[action_type] / BEHAVIORAL_SCALE_MAX
        
        # Functional Utility Formula Calculation
        consumption_utility = (z_norm * (1 - ALPHA)) + (e_norm * ALPHA)
        behavioral_utility = K + M + pbc_norm
        
        return (consumption_utility * (1 - delta)) + (behavioral_utility * delta)
    
    def calculate_all_utilities(self, household, market_state: Dict) -> None:
        """
        Loops through all possible configurations to update the household's expected utility arrays.
        """
        # Initialize storage arrays on the household if missing
        for attr in ['utility_exp_grey', 'utility_exp_brown', 'utility_exp_green']:
            if not hasattr(household, attr) or not getattr(household, attr):
                setattr(household, attr, [0.0, 0.0, 0.0])

        for energy_source in [0, 1, 2]:
            for action_type in range(3):
                util = self.calculate_expected_utility(
                    household, energy_source, action_type, market_state
                )
                
                if energy_source == 0:
                    household.utility_exp_grey[action_type] = util
                elif energy_source == 1:
                    household.utility_exp_brown[action_type] = util
                else:
                    household.utility_exp_green[action_type] = util
    
    def calculate_actual_utility(self, household) -> None:
        """
        Maps expectation utilities to reality indexes depending on which fuel flag 
        the household is currently committed to.
        """
        # Clear out baseline matrix structures
        household.utility_grey = [0.0, 0.0, 0.0]
        household.utility_brown = [0.0, 0.0, 0.0]
        household.utility_green = [0.0, 0.0, 0.0]
        
        if household.flag == 0:    # Currently on GREY
            household.utility_grey = household.utility_exp_grey.copy()
        elif household.flag == 1:  # Currently on BROWN
            household.utility_brown = household.utility_exp_brown.copy()
        else:                      # Currently on GREEN
            household.utility_green = household.utility_exp_green.copy()
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
            Finds the maximum values for each color variable across the entire population.
            """
            # Track separate global maximums
            max_grey1 = max_grey2 = max_grey3 = 1.0
            max_brown1 = max_brown2 = max_brown3 = 1.0
            max_green1 = max_green2 = 1.0
            
            for hh in households:
                max_brown1 = max(max_brown1, getattr(hh, 'z_brown1'))
                max_brown2 = max(max_brown2, getattr(hh, 'z_brown2'))
                max_brown3 = max(max_brown3, getattr(hh, 'z_brown3'))
                
                max_grey1  = max(max_grey1, getattr(hh, 'z_grey1'))
                max_grey2  = max(max_grey2, getattr(hh, 'z_grey2'))
                max_grey3  = max(max_grey3, getattr(hh, 'z_grey3'))
                
                max_green1 = max(max_green1, getattr(hh, 'z_green1'))
                max_green2 = max(max_green2, getattr(hh, 'z_green2'))

            # Assign normalized variables directly back to the agents
            for hh in households:
                hh.z_brown1_norm = hh.z_brown1 / max_brown1 if max_brown1 > 0 else 1.0
                hh.z_brown2_norm = hh.z_brown2 / max_brown2 if max_brown2 > 0 else 1.0
                hh.z_brown3_norm = hh.z_brown3 / max_brown3 if max_brown3 > 0 else 1.0
                
                hh.z_grey1_norm  = hh.z_grey1 / max_grey1 if max_grey1 > 0 else 1.0
                hh.z_grey2_norm  = hh.z_grey2 / max_grey2 if max_grey2 > 0 else 1.0
                hh.z_grey3_norm  = hh.z_grey3 / max_grey3 if max_grey3 > 0 else 1.0
                
                hh.z_green1_norm = hh.z_green1 / max_green1 if max_green1 > 0 else 1.0
                hh.z_green2_norm = hh.z_green2 / max_green2 if max_green2 > 0 else 1.0

    def normalize_budget_values(self, household) -> None:
            """
            Normalizes a single household's raw Z values against the global population maximums.
            """
            # Safely fetch the 3-slot list of maximum values from the dictionary, falling back to 1.0s
            z_brown_max = self.z_max.get('z_brown', [1.0, 1.0, 1.0])
            z_grey_max = self.z_max.get('z_grey', [1.0, 1.0, 1.0])
            z_green_max = self.z_max.get('z_green', [1.0, 1.0, 1.0])

            # Safely loop through all 3 index slots (0 = Investment, 1 = Conservation, 2 = Switching)
            for i in range(3):
                # Guard against division by zero by checking if the specific slot maximum is greater than 0
                household.z_brown_norm[i] = (household.z_brown[i] / z_brown_max[i]) if z_brown_max[i] > 0 else 0.0
                household.z_grey_norm[i] = (household.z_grey[i] / z_grey_max[i]) if z_grey_max[i] > 0 else 0.0
                household.z_green_norm[i] = (household.z_green[i] / z_green_max[i]) if z_green_max[i] > 0 else 0.0
    
    def calculate_expected_utility(self, household, energy_source: int, action_type: int, market_state: dict) -> float:
            """
            Calculates utility matching NetLogo structures exactly.
            household.flag: 0 = Grey (FF), 1 = Brown (LCE), 2 = Green (Zero)
            action_type: 0 = Investment, 1 = Conservation, 2 = Switching
            """
            K = household.K
            alpha = market_state.get('alpha', 0.1)
            
            M = household.M[action_type]
            delta = household.delta[action_type]
            pbc = household.pbc[action_type]
            
            # Base consumption metric (e_norm)
            e_norm = household.h_q / max(1.0, household.h_q) 

            
            # Map the calculation paths directly to your explicit color variable names
            if household.flag == 1:   # Brown User Path (LCE)
                if action_type == 0:    # Investment
                    z_norm = getattr(household, 'z_brown1_norm', 1.0)
                elif action_type == 1:  # Conservation
                    z_norm = getattr(household, 'z_brown2_norm', 1.0)
                elif action_type == 2:  # Switching
                    z_norm = getattr(household, 'z_brown3_norm', 1.0)
                    
            elif household.flag == 0: # Grey User Path (FF)
                if action_type == 0:    # Investment
                    z_norm = getattr(household, 'z_grey1_norm', 1.0)
                elif action_type == 1:  # Conservation
                    z_norm = getattr(household, 'z_grey2_norm', 1.0)
                elif action_type == 2:  # Switching
                    z_norm = getattr(household, 'z_grey3_norm', 1.0)
                    
            elif household.flag == 2:  # Green User Path (Super Green / zero)
                        if action_type == 0:    # Investment (zero1)
                            z_norm = getattr(household, 'z_green1_norm', 1.0)
                        elif action_type == 1:  # Conservation (zero2)
                            z_norm = getattr(household, 'z_green2_norm', 1.0)
                        elif action_type == 2:  # Switching (Not allowed for green agents!)
                            # Return 0.0 utility because they are already at the peak green tier 
                            # and cannot switch away to grey or brown.
                            return 0.0

            

            if delta == 0:
                return 0.0

            # Exact formula from NetLogo utility_exp_NAT:
            utility = (((z_norm * (1 - alpha)) + (e_norm * alpha)) * (1 - delta)) + ((K + M + (pbc / 7.0)) * delta)
            return utility

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
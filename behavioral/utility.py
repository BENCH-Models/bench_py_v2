"""
Utility calculation and preference modeling for household decisions
"""

from typing import Dict, List, Tuple
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
        """Initialize utility calculator."""
        self.z_max = {}  # Store maximum z values for normalization
        self.e_norm = {}  # Store normalized energy parameters
    
    def normalize_budget_values(self, household) -> None:
        """
        Normalize budget values (z) to 0-1 scale using population max.
        
        Args:
            household: Household object to normalize
        """
        z_brown_max = self.z_max.get('z_brown')
        z_grey_max = self.z_max.get('z_grey')
        z_green_max = self.z_max.get('z_green')
        
        for i in range(3):
            household.z_brown_norm[i] = (household.z_brown[i] / z_brown_max[i])
            household.z_grey_norm[i] = (household.z_grey[i] / z_grey_max[i])
            household.z_green_norm[i] = (household.z_green[i] / z_green_max[i]) 
    
    def calculate_expected_utility(self, household, energy_source: int, 
                                  action_type: int, market_state: Dict) -> float:
        """
        Calculate expected utility for a household action.
        
        Expected Utility Formula:
        UE = ((z_norm * (1 - alpha)) + (e_norm * alpha)) * (1 - delta) + 
             ((K + M + pbc/7) * delta)
        
        Where:
        - z_norm: Normalized budget/consumption combination (0-1)
        - alpha: Weight on environmental component (typically 0.5)
        - e_norm: Normalized environmental preference
        - delta: Behavioral adjustment factor (0.2-0.7)
        - K: Guilt factor (0-1)
        - M: Motivation factor (0-1)
        - pbc: Perceived behavioral control (0-7)
        
        Args:
            household: Household object
            energy_source: 0=FF, 1=LCE, 2=SLCE
            action_type: 0=Investment, 1=Conservation, 2=Switching
            market_state: Dictionary with market parameters
            
        Returns:
            Expected utility value (0-1 scale, typically)
        """
        # Select appropriate z value
        if energy_source == 0:  # GREY
            z_norm = household.z_grey_norm[action_type]
        elif energy_source == 1:  # BROWN
            z_norm = household.z_brown_norm[action_type]
        else:  # GREEN
            z_norm = household.z_green_norm[action_type]
        
        # Environmental preference (proxy: higher for BROWN/GREEN)
        e_norm = 0.3 if energy_source == 0 else 0.7
        
        # Get behavioral factors
        delta = household.delta[action_type]
        K = household.K
        M = household.M[action_type]
        pbc_norm = household.pbc[action_type] / BEHAVIORAL_SCALE_MAX
        
        # Calculate utility
        consumption_utility = (z_norm * (1 - ALPHA)) + (e_norm * ALPHA)
        behavioral_utility = K + M + pbc_norm
        
        expected_utility = (consumption_utility * (1 - delta)) + (behavioral_utility * delta)
        
        return expected_utility
    
    def calculate_all_utilities(self, household, market_state: Dict) -> None:
        """
        Calculate all expected utilities for a household across all scenarios.
        
        Args:
            household: Household object
            market_state: Dictionary with market parameters
        """
        # For each energy source option
        for energy_source in [0, 1, 2]:
            for action_type in range(3):
                util = self.calculate_expected_utility(
                    household, energy_source, action_type, market_state
                )
                
                if energy_source == 0:  # FF
                    household.utility_exp_grey[action_type] = util
                elif energy_source == 1:  # LCE
                    household.utility_exp_brown[action_type] = util
                else:  # SLCE
                    household.utility_exp_green[action_type] = util
    
    def calculate_actual_utility(self, household) -> None:
        """
        Calculate actual utility experienced after taking action.
        Based on actual outcomes rather than expectations.
        
        Args:
            household: Household object
        """
        # Actual utilities are determined by outcomes (simplified version).
        
        if household.flag == 0:  # Currently on FF
            household.utility_grey = household.utility_exp_grey.copy()
            household.utility_brown = [0.0, 0.0, 0.0]
            household.utility_green = [0.0, 0.0, 0.0]
        elif household.flag == 1:  # Currently on LCE
            household.utility_brown = household.utility_exp_brown.copy()
            household.utility_grey = [0.0, 0.0, 0.0]
            household.utility_green = [0.0, 0.0, 0.0]
        else:  # Currently on SLCE
            household.utility_green = household.utility_exp_green.copy()
            household.utility_brown = [0.0, 0.0, 0.0]
            household.utility_grey = [0.0, 0.0, 0.0]
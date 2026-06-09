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
    
    def normalize_budgets(self, households: List) -> None:
        """
        Calculate maximum budget values across population for normalization.
        
        Args:
            households: List of Household objects
        """
        # Find maximum z values across all households and scenarios
        z_lce_max = [0.0, 0.0, 0.0]
        z_ff_max = [0.0, 0.0, 0.0]
        z_zero_max = [0.0, 0.0, 0.0]
        
        for hh in households:
            for i in range(3):
                z_lce_max[i] = max(z_lce_max[i], hh.z_lce[i])
                z_ff_max[i] = max(z_ff_max[i], hh.z_ff[i])
                z_zero_max[i] = max(z_zero_max[i], hh.z_zero[i])
        
        self.z_max = {
            'z_lce': z_lce_max,
            'z_ff': z_ff_max,
            'z_zero': z_zero_max
        }
    
    def normalize_budget_values(self, household) -> None:
        """
        Normalize budget values (z) to 0-1 scale using population max.
        
        Args:
            household: Household object to normalize
        """
        z_lce_max = self.z_max.get('z_lce', [1.0, 1.0, 1.0])
        z_ff_max = self.z_max.get('z_ff', [1.0, 1.0, 1.0])
        z_zero_max = self.z_max.get('z_zero', [1.0, 1.0, 1.0])
        
        for i in range(3):
            # Normalize with protection against division by zero
            household.z_lce_norm[i] = (household.z_lce[i] / z_lce_max[i]) if z_lce_max[i] > 0 else 0.0
            household.z_ff_norm[i] = (household.z_ff[i] / z_ff_max[i]) if z_ff_max[i] > 0 else 0.0
            household.z_zero_norm[i] = (household.z_zero[i] / z_zero_max[i]) if z_zero_max[i] > 0 else 0.0
    
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
        if energy_source == 0:  # FF
            z_norm = household.z_ff_norm[action_type]
        elif energy_source == 1:  # LCE
            z_norm = household.z_lce_norm[action_type]
        else:  # SLCE
            z_norm = household.z_zero_norm[action_type]
        
        # Environmental preference (proxy: higher for LCE/SLCE)
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
                    household.utility_exp_ff[action_type] = util
                elif energy_source == 1:  # LCE
                    household.utility_exp_lce[action_type] = util
                else:  # SLCE
                    household.utility_exp_zero[action_type] = util
    
    def calculate_actual_utility(self, household) -> None:
        """
        Calculate actual utility experienced after taking action.
        Based on actual outcomes rather than expectations.
        
        Args:
            household: Household object
        """
        # Actual utilities are determined by outcomes (simplified version).
        
        if household.flag == 0:  # Currently on FF
            household.utility_ff = household.utility_exp_ff.copy()
            household.utility_lce = [0.0, 0.0, 0.0]
            household.utility_zero = [0.0, 0.0, 0.0]
        elif household.flag == 1:  # Currently on LCE
            household.utility_lce = household.utility_exp_lce.copy()
            household.utility_ff = [0.0, 0.0, 0.0]
            household.utility_zero = [0.0, 0.0, 0.0]
        else:  # Currently on SLCE
            household.utility_zero = household.utility_exp_zero.copy()
            household.utility_lce = [0.0, 0.0, 0.0]
            household.utility_ff = [0.0, 0.0, 0.0]
    
    def get_max_expected_utility(self, household) -> Tuple[float, int]:
        """
        Determine which action has highest expected utility.
        
        Args:
            household: Household object
            
        Returns:
            Tuple of (max_utility, action_type_index)
        """
        if household.flag == 0:  # FF
            utils = household.utility_exp_ff
        elif household.flag == 1:  # LCE
            utils = household.utility_exp_lce
        else:  # SLCE
            utils = household.utility_exp_zero
        
        max_util = max(utils) if utils else 0.0
        max_action = utils.index(max_util) if utils else 0
        
        return max_util, max_action
    
    def should_take_action(self, household, action_type: int, 
                          threshold: float = 0.5) -> bool:
        """
        Determine if household should take action based on utility threshold.
        
        Args:
            household: Household object
            action_type: 0, 1, or 2
            threshold: Utility threshold (default 0.5)
            
        Returns:
            True if utility > threshold, False otherwise
        """
        if household.flag == 0:
            util = household.utility_exp_ff[action_type]
        elif household.flag == 1:
            util = household.utility_exp_lce[action_type]
        else:
            util = household.utility_exp_zero[action_type]
        
        return util > threshold
    
    def calculate_switching_benefit(self, household, 
                                   old_price: float, new_price: float) -> float:
        """
        Calculate benefit of switching energy sources.
        
        Args:
            household: Household object
            old_price: Current energy price (€/kWh)
            new_price: New energy source price (€/kWh)
            
        Returns:
            Annual savings/cost (negative = cost)
        """
        benefit = (old_price - new_price) * household.h_q
        return benefit
    
    def calculate_action_utility_comparison(self, household) -> Dict[int, float]:
        """
        Compare utilities across all actions for current energy source.
        
        Args:
            household: Household object
            
        Returns:
            Dictionary mapping action_type -> utility
        """
        if household.flag == 0:
            return {i: household.utility_exp_ff[i] for i in range(3)}
        elif household.flag == 1:
            return {i: household.utility_exp_lce[i] for i in range(3)}
        else:
            return {i: household.utility_exp_zero[i] for i in range(3)}

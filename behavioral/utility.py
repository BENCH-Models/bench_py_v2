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
        self.e_max = 1.0
    
    def normalize_budgets(self, households: List) -> None:
        """
        Finds the maximum values for each variable across the entire population.
        Matches NetLogo's normalize-1 procedure.
        """
        # Initialize max trackers using dictionary structure
        z_max = {
            'grey': {'conservation': -float('inf'), 'investment': -float('inf'), 'switching': -float('inf')},
            'brown': {'conservation': -float('inf'), 'investment': -float('inf'), 'switching': -float('inf')},
            'green': {'conservation': -float('inf'), 'investment': -float('inf')}  # No switching for green
        }
        
        e_max = -float('inf')
        
        for hh in households:
            # Grey values - using dictionary access
            if hasattr(hh, 'z_grey') and 'conservation' in hh.z_grey:
                z_max['grey']['conservation'] = max(z_max['grey']['conservation'], hh.z_grey['conservation'])
                z_max['grey']['investment'] = max(z_max['grey']['investment'], hh.z_grey['investment'])
                z_max['grey']['switching'] = max(z_max['grey']['switching'], hh.z_grey['switching'])
            
            # Brown values
            if hasattr(hh, 'z_brown') and 'conservation' in hh.z_brown:
                z_max['brown']['conservation'] = max(z_max['brown']['conservation'], hh.z_brown['conservation'])
                z_max['brown']['investment'] = max(z_max['brown']['investment'], hh.z_brown['investment'])
                z_max['brown']['switching'] = max(z_max['brown']['switching'], hh.z_brown['switching'])
            
            # Green values (only conservation and investment)
            if hasattr(hh, 'z_green') and 'conservation' in hh.z_green:
                z_max['green']['conservation'] = max(z_max['green']['conservation'], hh.z_green['conservation'])
                z_max['green']['investment'] = max(z_max['green']['investment'], hh.z_green['investment'])
            
            # Track max consumption
            if hh.h_q > e_max:
                e_max = hh.h_q
        
        # Replace -inf with 1.0 for safety
        for energy_type in z_max:
            for action in z_max[energy_type]:
                if z_max[energy_type][action] <= 0 or z_max[energy_type][action] == float('-inf'):
                    z_max[energy_type][action] = 1.0
        
        self.z_max = z_max
        self.e_max = e_max if e_max > 0 else 1.0
    
    def normalize_budget_values(self, household) -> None:
        """
        Normalizes a single household's Z values against the population maximums.
        """
        # Initialize normalized dictionaries if they don't exist
        if not hasattr(household, 'z_grey_norm'):
            household.z_grey_norm = {'conservation': 0.0, 'investment': 0.0, 'switching': 0.0}
            household.z_brown_norm = {'conservation': 0.0, 'investment': 0.0, 'switching': 0.0}
            household.z_green_norm = {'conservation': 0.0, 'investment': 0.0}
        
        # Grey normalization
        if hasattr(household, 'z_grey'):
            household.z_grey_norm['conservation'] = household.z_grey['conservation'] / self.z_max['grey']['conservation']
            household.z_grey_norm['investment'] = household.z_grey['investment'] / self.z_max['grey']['investment']
            household.z_grey_norm['switching'] = household.z_grey['switching'] / self.z_max['grey']['switching']
        
        # Brown normalization
        if hasattr(household, 'z_brown'):
            household.z_brown_norm['conservation'] = household.z_brown['conservation'] / self.z_max['brown']['conservation']
            household.z_brown_norm['investment'] = household.z_brown['investment'] / self.z_max['brown']['investment']
            household.z_brown_norm['switching'] = household.z_brown['switching'] / self.z_max['brown']['switching']
        
        # Green normalization
        if hasattr(household, 'z_green'):
            household.z_green_norm['conservation'] = household.z_green['conservation'] / self.z_max['green']['conservation']
            household.z_green_norm['investment'] = household.z_green['investment'] / self.z_max['green']['investment']
        
        # Store consumption normalization denominator
        household.e_norm_denom = self.e_max
    
    def calculate_expected_utility(self, household, energy_source: str, 
                                   action_type: str, alpha: float = ALPHA) -> float:
        """
        Calculates expected utility matching NetLogo's utility_exp_NAT exactly.
        
        Args:
            household: Household object
            energy_source: 'grey', 'brown', or 'green'
            action_type: 'investment', 'conservation', or 'switching'
            alpha: Share of income spent on composite good
        """
        # Check if household is low-paid (constraint from NetLogo)
        if hasattr(household, 'hh_sta') and household.hh_sta == "Low-paid1":
            return 0.0
        
        # Get behavioral factors - now using dictionary access
        K = household.K  # guilt factor (0-1)
        M = household.M.get(action_type, 0.0)  # Get motivation for this action
        pbc = household.pbc.get(action_type, 0.0)  # Get PBC for this action
        delta = household.delta.get(action_type, 0.0)  # Get delta for this action
        
        # Get normalized Z based on energy source and action type
        # Try dictionary access first, fallback to attribute access for backward compatibility
        z_norm = 1.0
        z_dict_name = f'z_{energy_source}_norm'
        if hasattr(household, z_dict_name):
            z_dict = getattr(household, z_dict_name)
            if action_type in z_dict:
                z_norm = z_dict[action_type]
        
        # Get normalized consumption
        e_norm = household.h_q / household.e_norm_denom if household.e_norm_denom > 0 else 1.0
        
        # Check if delta is zero (NetLogo condition)
        if delta == 0:
            return 0.0
        
        # Calculate utility using exact NetLogo formula
        economic_part = (z_norm * (1 - alpha)) + (e_norm * alpha)
        behavioral_part = K + M + (pbc / BEHAVIORAL_SCALE_MAX)
        
        utility = (economic_part * (1 - delta)) + (behavioral_part * delta)
        
        return utility
    
    def calculate_all_expected_utilities(self, household, alpha: float = ALPHA) -> None:
        """
        Calculate all expected utilities for a household using dictionary structure.
        Matches NetLogo's utility_exp_NAT procedure with action constraints.
        """
        # Initialize utility dictionaries if they don't exist
        if not hasattr(household, 'utility_exp'):
            household.utility_exp = {
                'grey': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0},
                'brown': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0},
                'green': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0}
            }
        
        # Reset all utilities to 0
        for energy in household.utility_exp:
            for action in household.utility_exp[energy]:
                household.utility_exp[energy][action] = 0.0
        
        # Calculate based on current flag (matching NetLogo's conditional logic)
        if household.flag == 1:  # Brown user
            # Action 1: Investment (only if not already taken)
            if not household.act11:
                household.utility_exp['brown']['investment'] = self.calculate_expected_utility(
                    household, 'brown', 'investment', alpha
                )
            
            # Action 2: Conservation (only if not already taken)
            if not household.act21:
                household.utility_exp['brown']['conservation'] = self.calculate_expected_utility(
                    household, 'brown', 'conservation', alpha
                )
            
            # Action 3: Switching to green (only if not already taken)
            if not (household.act31 or household.act32):
                household.utility_exp['brown']['switching'] = self.calculate_expected_utility(
                    household, 'brown', 'switching', alpha
                )
            
        elif household.flag == 0:  # Grey user
            # Action 1: Investment (only if not already taken)
            if not household.act12:
                household.utility_exp['grey']['investment'] = self.calculate_expected_utility(
                    household, 'grey', 'investment', alpha
                )
            
            # Action 2: Conservation (only if not already taken)
            if not household.act40:
                household.utility_exp['grey']['conservation'] = self.calculate_expected_utility(
                    household, 'grey', 'conservation', alpha
                )
            
            # Action 3: Switching to brown (only if not already taken)
            if not household.act32:
                household.utility_exp['grey']['switching'] = self.calculate_expected_utility(
                    household, 'grey', 'switching', alpha
                )
            
        else:  # Green user (flag == 2)
            # Action 1: Investment (only if not already taken)
            if not household.act11:
                household.utility_exp['green']['investment'] = self.calculate_expected_utility(
                    household, 'green', 'investment', alpha
                )
            
            # Action 2: Conservation (only if not already taken)
            if not household.act21:
                household.utility_exp['green']['conservation'] = self.calculate_expected_utility(
                    household, 'green', 'conservation', alpha
                )
            
            # Action 3: Switching - not applicable for green users
    
    def calculate_actual_utility(self, household, alpha: float = ALPHA) -> None:
        """
        Calculate actual utilities after actions are taken.
        Matches NetLogo's update_utilities_NAT procedure.
        """
        # First normalize for actual utility calculation
        self._normalize_actual_budgets(household)
        
        # Initialize actual utility dictionaries if they don't exist
        if not hasattr(household, 'utility_actual'):
            household.utility_actual = {
                'grey': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0},
                'brown': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0},
                'green': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0}
            }
        
        # Reset all actual utilities
        for energy in household.utility_actual:
            for action in household.utility_actual[energy]:
                household.utility_actual[energy][action] = 0.0
        
        # Get behavioral factors - using dictionary access
        K = household.K
        
        # Get normalized values for actual utility
        z_norm = getattr(household, 'z_norm_actual', 1.0)
        e_norm = getattr(household, 'e_norm_actual', 1.0)
        
        # Define actions to calculate based on flag
        if household.flag == 1:  # Brown user
            actions_to_calc = ['investment', 'conservation', 'switching']
            target = 'brown'
        elif household.flag == 0:  # Grey user
            actions_to_calc = ['investment', 'conservation', 'switching']
            target = 'grey'
        else:  # Green user
            actions_to_calc = ['investment', 'conservation']
            target = 'green'
        
        # Calculate utilities using dictionary access
        for action in actions_to_calc:
            M = household.M.get(action, 0.0)
            delta = household.delta.get(action, 0.0)
            
            if delta > 0:  # Only calculate if delta > 0
                household.utility_actual[target][action] = (
                    (z_norm * (1 - alpha) + e_norm * alpha) * (1 - delta)
                ) + ((K + M) * delta)
    
    def _normalize_actual_budgets(self, household) -> None:
        """Normalize for actual utility calculation (NetLogo's normalize-2 equivalent)."""
        # Simplified normalization for actual utility - find max Z across all types
        z_max = 1.0
        
        # Check all possible z values to find max
        if hasattr(household, 'z_brown'):
            for action in household.z_brown.values():
                if action > z_max:
                    z_max = action
        
        if hasattr(household, 'z_grey'):
            for action in household.z_grey.values():
                if action > z_max:
                    z_max = action
        
        if hasattr(household, 'z_green'):
            for action in household.z_green.values():
                if action > z_max:
                    z_max = action
        
        # Simple normalization - in practice, you'd use population-wide max
        household.z_norm_actual = 1.0
        household.e_norm_actual = household.h_q / max(1.0, household.h_q)
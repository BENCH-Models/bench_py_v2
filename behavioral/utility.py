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
        self.e_max_grey = 1.0   # Max consumption for grey (ff) users
        self.e_max_brown = 1.0  # Max consumption for brown (lce) users
        self.e_max_green = 1.0  # Max consumption for green (zero) users
    
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
        
        # Separate max consumption by energy source (matching NetLogo)
        e_max_grey = -float('inf')   # For flag? = 0 (grey/ff users)
        e_max_brown = -float('inf')  # For flag? = 1 (brown/lce users)  
        e_max_green = -float('inf')  # For flag? = 2 (green/zero users)
        
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
            
            # Track max consumption BY ENERGY SOURCE (NetLogo style)
            if hh.flag == 0:  # Grey/FF user
                if hh.h_q > e_max_grey:
                    e_max_grey = hh.h_q
            elif hh.flag == 1:  # Brown/LCE user
                if hh.h_q > e_max_brown:
                    e_max_brown = hh.h_q
            else:  # Green/Zero user (flag == 2)
                if hh.h_q > e_max_green:
                    e_max_green = hh.h_q
        
        # Replace -inf with 1.0 for safety
        for energy_type in z_max:
            for action in z_max[energy_type]:
                if z_max[energy_type][action] <= 0 or z_max[energy_type][action] == float('-inf'):
                    z_max[energy_type][action] = 1.0
        
        self.z_max = z_max
        self.e_max_grey = e_max_grey if e_max_grey > 0 else 1.0
        self.e_max_brown = e_max_brown if e_max_brown > 0 else 1.0
        self.e_max_green = e_max_green if e_max_green > 0 else 1.0
    
    def normalize_budget_values(self, household) -> None:
        """
        Normalizes a single household's Z values against the population maximums.
        Also normalizes consumption based on the household's energy source.
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
        
        # Store consumption normalization denominators based on energy source
        # This matches NetLogo's e_lce_norm and e_ff_norm
        household.e_norm_denom_grey = self.e_max_grey
        household.e_norm_denom_brown = self.e_max_brown
        household.e_norm_denom_green = self.e_max_green
    
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
        
        # Get behavioral factors - using dictionary access
        K = household.K  # guilt factor (0-1)
        M = household.M.get(action_type, 0.0)
        pbc = household.pbc.get(action_type, 0.0)
        delta = household.delta.get(action_type, 0.0)
        
        # Get normalized Z based on energy source and action type
        z_norm = 1.0
        z_dict_name = f'z_{energy_source}_norm'
        if hasattr(household, z_dict_name):
            z_dict = getattr(household, z_dict_name)
            if action_type in z_dict:
                z_norm = z_dict[action_type]
        
        # Get normalized consumption based on energy source (NetLogo style)
        # This is the key fix - use the correct denominator for each energy source
        if energy_source == 'grey':
            e_norm = household.h_q / household.e_norm_denom_grey if household.e_norm_denom_grey > 0 else 1.0
        elif energy_source == 'brown':
            e_norm = household.h_q / household.e_norm_denom_brown if household.e_norm_denom_brown > 0 else 1.0
        else:  # green
            e_norm = household.h_q / household.e_norm_denom_green if household.e_norm_denom_green > 0 else 1.0
        
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
    
    def calculate_actual_utility(self, households: List, prices: Dict[str, float], alpha: float = ALPHA) -> None:
        """
        Calculate actual utilities for ALL households.
        Matches NetLogo's update_utilities_NAT procedure.
        
        Args:
            households: List of all household agents
            prices: Current market prices dictionary
            alpha: Share of income spent on composite good
        """
        # First normalize for actual utility calculation (population-wide)
        self._normalize_actual_budgets(households, prices)
        
        for household in households:
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
            
            # Get behavioral factors
            K = household.K
            
            # Get normalized z value (already set by _normalize_actual_budgets)
            z_norm = getattr(household, 'z_norm_actual', 1.0)
            
            # Use the appropriate e_norm based on current flag
            if household.flag == 0:  # Grey user
                e_norm = household.h_q / self.e_max_grey if self.e_max_grey > 0 else 1.0
                target = 'grey'
                actions_to_calc = ['investment', 'conservation', 'switching']
            elif household.flag == 1:  # Brown user
                e_norm = household.h_q / self.e_max_brown if self.e_max_brown > 0 else 1.0
                target = 'brown'
                actions_to_calc = ['investment', 'conservation', 'switching']
            else:  # Green user (flag == 2)
                e_norm = household.h_q / self.e_max_green if self.e_max_green > 0 else 1.0
                target = 'green'
                actions_to_calc = ['investment', 'conservation']
            
            # Calculate utilities for each action
            for action in actions_to_calc:
                M = household.M.get(action, 0.0)
                delta = household.delta.get(action, 0.0)
                
                # Only calculate if delta > 0 (NetLogo condition)
                if delta > 0:
                    household.utility_actual[target][action] = (
                        (z_norm * (1 - alpha) + e_norm * alpha) * (1 - delta)
                    ) + ((K + M) * delta)

    def _normalize_actual_budgets(self, households: List, prices: Dict[str, float]) -> None:
        """
        Normalize for actual utility calculation (NetLogo's normalize-2 equivalent).
        This MUST be called on the entire population, not per household.
        
        Args:
            households: List of all household agents
            prices: Current market prices dictionary
        """
        # Get current prices
        m_p_grey = prices.get('m_p_grey', 0.15)
        m_p_brown = prices.get('m_p_brown', 0.15)
        
        # First pass: calculate raw z values for all households
        for household in households:
            if household.h_q <= 0:
                continue
            
            h_q = household.h_q
            h_income = household.h_income
            h_conserv_p = household.h_conserv_p
            h_switch = household.h_switch
            h_invest = household.h_invest
            h_invest_save = household.h_invest_save

            # Calculate z values for each energy type (matching NetLogo)
            # Note: In NetLogo, all three use m_p_lce (brown) for the base consumption calculation
            household.z_actual_brown = h_income - ((h_q * m_p_brown) + (h_invest_save * m_p_brown) + h_invest + h_conserv_p + h_switch)
            household.z_actual_grey = h_income - ((h_q * m_p_brown) + (h_invest_save * m_p_grey) + h_invest + h_conserv_p + h_switch)
            household.z_actual_green = h_income - ((h_q * m_p_brown) + (h_invest_save * m_p_brown) + h_invest + h_conserv_p + h_switch)
        
        # Second pass: find population-wide maximums
        z_brown_max = 1.0
        z_grey_max = 1.0
        z_green_max = 1.0
        
        # Find max for brown
        brown_values = [hh.z_actual_brown for hh in households if hasattr(hh, 'z_actual_brown')]
        if brown_values:
            z_brown_max = max(brown_values)
        
        # Find max for grey
        grey_values = [hh.z_actual_grey for hh in households if hasattr(hh, 'z_actual_grey')]
        if grey_values:
            z_grey_max = max(grey_values)
        
        # Find max for green
        green_values = [hh.z_actual_green for hh in households if hasattr(hh, 'z_actual_green')]
        if green_values:
            z_green_max = max(green_values)
        
        # Ensure no division by zero
        z_brown_max = z_brown_max if z_brown_max > 0 else 1.0
        z_grey_max = z_grey_max if z_grey_max > 0 else 1.0
        z_green_max = z_green_max if z_green_max > 0 else 1.0
        
        # Third pass: normalize using population maximums
        for household in households:
            if hasattr(household, 'z_actual_brown'):
                household.z_brown_norm_actual = household.z_actual_brown / z_brown_max
                household.z_grey_norm_actual = household.z_actual_grey / z_grey_max
                household.z_green_norm_actual = household.z_actual_green / z_green_max
                
                # Store the appropriate one based on current flag for easy access
                if household.flag == 0:  # Grey user
                    household.z_norm_actual = household.z_grey_norm_actual
                elif household.flag == 1:  # Brown user
                    household.z_norm_actual = household.z_brown_norm_actual
                else:  # Green user (flag == 2)
                    household.z_norm_actual = household.z_green_norm_actual
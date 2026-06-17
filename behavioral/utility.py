"""
Utility calculation and preference modeling for household decisions
"""

from typing import Dict, List

import numpy as np

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
        self.e_max = 1.0         # Global max h_q across ALL households (NetLogo: e_lce_max = e_ff_max = global max)
        self._e_max_actual = 1.0
    
    def normalize_budgets(self, households: List) -> None:
        """
        Finds the maximum values for each variable across the entire population.
        Matches NetLogo's normalize-1 procedure.
        """
        # Build arrays once — faster than per-household max() comparisons
        zgc = np.fromiter((hh.z_grey['conservation']  for hh in households), float, len(households))
        zgi = np.fromiter((hh.z_grey['investment']    for hh in households), float, len(households))
        zgs = np.fromiter((hh.z_grey['switching']     for hh in households), float, len(households))
        zbc = np.fromiter((hh.z_brown['conservation'] for hh in households), float, len(households))
        zbi = np.fromiter((hh.z_brown['investment']   for hh in households), float, len(households))
        zbs = np.fromiter((hh.z_brown['switching']    for hh in households), float, len(households))
        znc = np.fromiter((hh.z_green['conservation'] for hh in households), float, len(households))
        zni = np.fromiter((hh.z_green['investment']   for hh in households), float, len(households))
        e_arr = np.fromiter((hh.h_q for hh in households), float, len(households))

        def _safe_max(arr):
            v = float(arr.max())
            return v if v > 0 else 1.0

        self.z_max = {
            'grey':  {'conservation': _safe_max(zgc), 'investment': _safe_max(zgi), 'switching': _safe_max(zgs)},
            'brown': {'conservation': _safe_max(zbc), 'investment': _safe_max(zbi), 'switching': _safe_max(zbs)},
            'green': {'conservation': _safe_max(znc), 'investment': _safe_max(zni)},
        }
        e_max = float(e_arr.max())
        self.e_max = e_max if e_max > 0 else 1.0
    
    def normalize_budget_values(self, household) -> None:
        """
        Normalizes a single household's Z values against the population maximums.
        Also normalizes consumption based on the household's energy source.
        """
        gm = self.z_max['grey']
        bm = self.z_max['brown']
        nm = self.z_max['green']

        zg = household.z_grey
        household.z_grey_norm['conservation'] = zg['conservation'] / gm['conservation']
        household.z_grey_norm['investment']   = zg['investment']   / gm['investment']
        household.z_grey_norm['switching']    = zg['switching']    / gm['switching']

        zb = household.z_brown
        household.z_brown_norm['conservation'] = zb['conservation'] / bm['conservation']
        household.z_brown_norm['investment']   = zb['investment']   / bm['investment']
        household.z_brown_norm['switching']    = zb['switching']    / bm['switching']

        zn = household.z_green
        household.z_green_norm['conservation'] = zn['conservation'] / nm['conservation']
        household.z_green_norm['investment']   = zn['investment']   / nm['investment']

        # Bug 1 fix: single global denominator (NetLogo uses same max for all energy types)
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
        # Bug 4 fix: Low-paid1 only blocks investment expected utility (NetLogo gates only act1 utility)
        if action_type == 'investment' and household.hh_sta == "Low-paid1":
            return 0.0

        # Get behavioral factors
        K = household.K
        M = household.M.get(action_type, 0.0)
        pbc = household.pbc.get(action_type, 0.0)
        delta = household.delta.get(action_type, 0.0)

        # Get normalized Z — all z_*_norm dicts are guaranteed present (initialised in Household.__init__)
        z_norm = getattr(household, f'z_{energy_source}_norm').get(action_type, 1.0)

        # Bug 1 fix: single global e_norm denominator (NetLogo e_lce_max = e_ff_max = global max)
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
            # Reset all actual utilities
            for energy in household.utility_actual:
                for action in household.utility_actual[energy]:
                    household.utility_actual[energy][action] = 0.0
            
            # Get behavioral factors
            K = household.K
            
            # Get normalized z value (already set by _normalize_actual_budgets)
            z_norm = getattr(household, 'z_norm_actual', 1.0)
            
            # Bug 1 fix: single global e_norm denominator (NetLogo e_lce_max = e_ff_max = global max)
            e_norm = household.h_q / self._e_max_actual if self._e_max_actual > 0 else 1.0

            if household.flag == 0:
                target = 'grey'
                actions_to_calc = ['investment', 'conservation', 'switching']
            elif household.flag == 1:
                target = 'brown'
                actions_to_calc = ['investment', 'conservation', 'switching']
            else:
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
        
        # Second pass: find population-wide maximums using numpy (skip h_q<=0 households)
        # Bug 1 fix: e_max is global across all households (NetLogo: e_lce_max = e_ff_max = global)
        active = [hh for hh in households if hh.h_q > 0]
        if active:
            ab = np.fromiter((hh.z_actual_brown for hh in active), float, len(active))
            ag = np.fromiter((hh.z_actual_grey  for hh in active), float, len(active))
            an = np.fromiter((hh.z_actual_green for hh in active), float, len(active))
            eq = np.fromiter((hh.h_q            for hh in active), float, len(active))
            z_brown_max = float(ab.max()); z_brown_max = z_brown_max if z_brown_max > 0 else 1.0
            z_grey_max  = float(ag.max()); z_grey_max  = z_grey_max  if z_grey_max  > 0 else 1.0
            z_green_max = float(an.max()); z_green_max = z_green_max if z_green_max > 0 else 1.0
            self._e_max_actual = float(eq.max()); self._e_max_actual = self._e_max_actual if self._e_max_actual > 0 else 1.0
        else:
            z_brown_max = z_grey_max = z_green_max = 1.0
            self._e_max_actual = 1.0

        # Third pass: normalize (only active households; others keep 0.0 defaults)
        for household in active:
            household.z_brown_norm_actual = household.z_actual_brown / z_brown_max
            household.z_grey_norm_actual  = household.z_actual_grey  / z_grey_max
            household.z_green_norm_actual = household.z_actual_green / z_green_max

            if household.flag == 0:
                household.z_norm_actual = household.z_grey_norm_actual
            elif household.flag == 1:
                household.z_norm_actual = household.z_brown_norm_actual
            else:
                household.z_norm_actual = household.z_green_norm_actual
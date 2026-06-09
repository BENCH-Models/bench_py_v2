"""
Behavioral decision-making logic for household energy actions
Implements knowledge, motivation, consideration, and action determination
"""

from typing import Dict, List, Tuple
import random
from utils.constants import (
    ACTION_INVESTMENT, ACTION_CONSERVATION, ACTION_SWITCHING,
    INVESTMENT_PV_ANNUAL_COST, INVESTMENT_PV_ENERGY_OUTPUT,
    CONSERVATION_RATE, GUILT_HIGH, GUILT_LOW,
    BEHAVIORAL_SCALE_MAX
)


class DecisionMaker:
    """
    Implements behavioral decision-making processes for households.
    Includes knowledge formation, motivation, consideration, and action selection.
    """
    
    def __init__(self):
        """Initialize decision maker."""
        self.decision_history = {}  # Track decisions over time
    
    def activate_knowledge(self, household, case_study: str) -> None:
        """
        Activate knowledge pathway: awareness affects guilt and motivation.
        
        Knowledge calculation based on: know + climate_awareness + education_awareness
        
        Args:
            household: Household object
            case_study: "Netherlands-Overijssel" or "Spain-Navarre"
        """
        # Update awareness (average of three components)
        household.update_awareness()
        
        # If awareness crosses guilt threshold, household becomes motivated
        if household.guilt == GUILT_HIGH:
            household.K = household.h_aware / BEHAVIORAL_SCALE_MAX
        else:
            household.K = 0.0
    
    def update_motivation(self, household, case_study: str) -> None:
        """
        Update household motivation based on personal and social norms.
        
        Args:
            household: Household object
            case_study: Case study identifier
        """
        household.update_motivation(case_study)
    
    def consider_action(self, household, action_type: int) -> None:
        """
        Apply consideration logic based on PBC and constraints.
        
        Sets delta (adjustment factor) and consideration level.
        
        Args:
            household: Household object
            action_type: 0=Investment, 1=Conservation, 2=Switching
        """
        household.consider_constraints(action_type)
    
    def decide_action(self, household, market_state: Dict,
                     utility_calculator) -> List[bool]:
        """
        Make action decision based on utilities and thresholds.
        
        Args:
            household: Household object
            market_state: Current market conditions
            utility_calculator: UtilityCalculator instance
            
        Returns:
            List of [action1_taken, action2_taken, action3_taken]
        """
        actions_taken = [False, False, False]
        
        # Skip if no consumption or already took all actions
        if household.h_q <= 0:
            return actions_taken
        
        if all(household.hh_actions):  # All actions taken
            return actions_taken
        
        # Proceed based on current energy source
        if household.flag == 1:  # Currently brown electricity
            actions_taken = self._decide_lce_household(household, utility_calculator)
        
        elif household.flag == 0:  # Currently gray electricity
            actions_taken = self._decide_ff_household(household, utility_calculator)
        
        elif household.flag == 2:  # Currently green electricity
            actions_taken = self._decide_slce_household(household, utility_calculator)
        
        return actions_taken
    
    def _decide_lce_household(self, household, utility_calculator) -> List[bool]:
        """Make decisions for household on green energy."""
        actions = [False, False, False]
        
        # Action 1: Investment (PV)
        if not household.act11:
            if (household.utility_exp_lce[0] >= max(
                household.utility_exp_lce[0],
                household.utility_exp_lce[1],
                household.utility_exp_lce[2]
            ) and household.utility_exp_lce[0] >= household.utility_lce[0]):
                household.act1 = True
                household.act11 = True
                household.hh_actions[0] = 1
                actions[0] = True
        
        # Action 2: Conservation
        if not household.act21:
            if (household.utility_exp_lce[1] >= max(
                household.utility_exp_lce[0],
                household.utility_exp_lce[1],
                household.utility_exp_lce[2]
            ) and household.utility_exp_lce[1] >= household.utility_lce[1]):
                household.act2 = True
                household.act21 = True
                household.act50 = True
                household.hh_actions[2] = 1
                actions[1] = True
        
        # Action 3: Switching to green electricity
        if not (household.act31 or household.act32):
            if (household.utility_exp_lce[2] >= max(
                household.utility_exp_lce[0],
                household.utility_exp_lce[1],
                household.utility_exp_lce[2]
            ) and household.utility_exp_lce[2] >= household.utility_lce[2]):
                household.act3 = True
                household.act31 = True
                household.flag = 2  # Switch to green electricity
                household.hh_actions[4] = 1
                actions[2] = True
        
        return actions
    
    def _decide_ff_household(self, household, utility_calculator) -> List[bool]:
        """Make decisions for household on fossil fuel."""
        actions = [False, False, False]
        
        # Action 1: Investment (PV)
        if not household.act12:
            if (household.utility_exp_ff[0] >= max(
                household.utility_exp_ff[0],
                household.utility_exp_ff[1],
                household.utility_exp_ff[2]
            ) and household.utility_exp_ff[0] >= household.utility_ff[0]):
                household.act1 = True
                household.act12 = True
                household.hh_actions[1] = 1
                actions[0] = True
        
        # Action 2: Conservation
        if not household.act40:
            if (household.utility_exp_ff[1] >= max(
                household.utility_exp_ff[0],
                household.utility_exp_ff[1],
                household.utility_exp_ff[2]
            ) and household.utility_exp_ff[1] >= household.utility_ff[1]):
                household.act2 = True
                household.act50 = True
                household.act40 = True
                household.hh_actions[3] = 1
                actions[1] = True
        
        # Action 3: Switching to brown electricity
        if not household.act32:
            if (household.utility_exp_ff[2] >= max(
                household.utility_exp_ff[0],
                household.utility_exp_ff[1],
                household.utility_exp_ff[2]
            ) and household.utility_exp_ff[2] >= household.utility_ff[2]):
                household.act3 = True
                household.act32 = True
                household.flag = 1  # Switch to brown electricity
                household.hh_actions[5] = 1
                actions[2] = True
        
        return actions
    
    def _decide_slce_household(self, household, utility_calculator) -> List[bool]:
        """Make decisions for household already on super-green."""
        actions = [False, False, False]
        
        # Action 1: Further Investment
        if not household.act11:
            if (household.utility_exp_zero[0] >= max(
                household.utility_exp_zero[0],
                household.utility_exp_zero[1]
            ) and household.utility_exp_zero[0] >= household.utility_zero[0]):
                household.act1 = True
                household.act11 = True
                household.hh_actions[0] = 1
                actions[0] = True
        
        # Action 2: Conservation (already implemented)
        if not household.act21:
            if (household.utility_exp_zero[1] >= max(
                household.utility_exp_zero[0],
                household.utility_exp_zero[1]
            ) and household.utility_exp_zero[1] >= household.utility_zero[1]):
                household.act2 = True
                household.act21 = True
                household.hh_actions[2] = 1
                actions[1] = True
        
        # No switching action for SLCE (already at top tier)
        
        return actions
    
    def calculate_energy_savings(self, household, prices: Dict[str, float]) -> float:
        """
        Calculate energy saved from actions.
        
        Args:
            household: Household object
            prices: Dictionary with electricity prices
            
        Returns:
            Total energy saved (kWh)
        """
        energy_saved = 0.0
        
        # Investment saves fixed amount
        if household.act1 or household.act11 or household.act12:
            energy_saved += INVESTMENT_PV_ENERGY_OUTPUT
            household.h_invest_save += INVESTMENT_PV_ENERGY_OUTPUT
        
        # Conservation saves percentage of consumption
        if household.act2 or household.act21 or household.act40:
            conservation_amount = household.h_q * CONSERVATION_RATE
            energy_saved += conservation_amount
            household.h_conserv = conservation_amount
            
            # Calculate money saved
            price = prices.get('m_p_ff', 0.15) if household.flag == 0 else prices.get('m_p_lce', 0.15)
            household.h_conserv_p = conservation_amount * price
        
        return energy_saved
    
    def calculate_financial_outcomes(self, household, prices: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate investment and switching financial outcomes.
        
        Args:
            household: Household object
            prices: Dictionary with electricity prices
            
        Returns:
            Tuple of (total_investment, switching_benefit)
        """
        investment = 0.0
        switching_benefit = 0.0
        
        # Investment cost and payback
        if household.act1 or household.act11 or household.act12:
            if household.counter_invest < 10:  # 10-year payback
                investment = INVESTMENT_PV_ANNUAL_COST
                household.h_invest = investment
                household.counter_invest += 1
                household.h_invest_total += investment
        
        # Switching benefit (price difference)
        if household.act3 or household.act31 or household.act32:
            old_price = prices.get('m_p_ff', 0.15) if household.flag != 0 else prices.get('m_p_lce', 0.15)
            new_price = prices.get('m_p_lce', 0.15) if household.flag == 0 else prices.get('m_p_zero', 0.12)
            switching_benefit = (old_price - new_price) * household.h_q
            household.h_switch = switching_benefit
        
        return investment, switching_benefit
    
    def calculate_emissions_avoided(self, household, prices: Dict[str, float]) -> float:
        """
        Calculate CO2 emissions avoided by actions.
        
        Conversion: ~0.5 kg CO2 per kWh of fossil fuel electricity
        
        Args:
            household: Household object
            prices: Dictionary with market info
            
        Returns:
            Total CO2 avoided (kg)
        """
        co2_factor = 0.5  # kg CO2 per kWh of FF electricity
        emissions_avoided = 0.0
        
        # Investment savings
        if household.act1 or household.act11 or household.act12:
            emissions_avoided += INVESTMENT_PV_ENERGY_OUTPUT * co2_factor
            household.em_avoided[0] += INVESTMENT_PV_ENERGY_OUTPUT * co2_factor
        
        # Conservation savings (only if was on FF before)
        if household.act2 or household.act21 or household.act40:
            if household.flag == 0:  # Currently on FF
                conservation_amount = household.h_q * CONSERVATION_RATE
                emissions_avoided += conservation_amount * co2_factor
                household.em_avoided[1] += conservation_amount * co2_factor
        
        # Switching to renewable
        if household.act3 or household.act31 or household.act32:
            # Avoid emissions for entire consumption switched to renewable
            emissions_avoided += household.h_q * co2_factor
            household.em_avoided[2] += household.h_q * co2_factor
        
        return emissions_avoided
    
    def get_decision_summary(self, household) -> Dict:
        """Get summary of household's current decisions."""
        return {
            'h_id': household.h_id,
            'action_investment': household.act1,
            'action_conservation': household.act2,
            'action_switching': household.act3,
            'energy_source': household.flag,
            'actions_taken': sum(household.hh_actions),
        }

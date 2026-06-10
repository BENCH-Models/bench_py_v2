"""
Behavioral decision-making logic for household energy actions
Implements knowledge, motivation, consideration, and action determination
"""

from typing import Dict, List, Tuple
import random
from agents import household
from utils.constants import (
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
    
    def consider_action(self, household, action_name: str) -> None:
        """
        Apply consideration logic based on PBC and constraints.
        
        Args:
            household: Household object
            action_name: 'investment', 'conservation', or 'switching'
        """
        household.consider_constraints(action_name)

    def apply_carbon_price_awareness(self, household, policy: str, year: int) -> None:
        """Apply carbon price awareness effects (NetLogo's cpinfo)."""
        if year < 2016:
            return
        
        # Skip if policy is "Ref" (no carbon price)
        if policy == "Ref":
            return
        
        # Handle different carbon price policies
        if policy == "Carbon price pressure-10":
            if household.know < 6.5:
                household.know = min(household.know * 1.01, BEHAVIORAL_SCALE_MAX)
            if household.cee_aw < 6.5:
                household.cee_aw = min(household.cee_aw * 1.01, BEHAVIORAL_SCALE_MAX)
            if household.ed_aw < 6.5:
                household.ed_aw = min(household.ed_aw * 1.01, BEHAVIORAL_SCALE_MAX)
            household.update_awareness()
            
        elif policy == "Carbon price pressure-25":
            if household.know < 6.5:
                household.know = min(household.know * 1.02, BEHAVIORAL_SCALE_MAX)
            if household.cee_aw < 6.5:
                household.cee_aw = min(household.cee_aw * 1.02, BEHAVIORAL_SCALE_MAX)
            if household.ed_aw < 6.5:
                household.ed_aw = min(household.ed_aw * 1.02, BEHAVIORAL_SCALE_MAX)
            household.update_awareness()
            
            for action in ['investment', 'conservation', 'switching']:
                if household.su_nor[action] < 6.5:
                    household.su_nor[action] = min(household.su_nor[action] * 1.04, BEHAVIORAL_SCALE_MAX)
                if household.pbc[action] < 6.5:
                    household.pbc[action] = min(household.pbc[action] * 1.03, BEHAVIORAL_SCALE_MAX)
                    
        elif policy == "Carbon price pressure-50":
            if household.know < 6.5:
                household.know = min(household.know * 1.04, BEHAVIORAL_SCALE_MAX)
            if household.cee_aw < 6.5:
                household.cee_aw = min(household.cee_aw * 1.04, BEHAVIORAL_SCALE_MAX)
            if household.ed_aw < 6.5:
                household.ed_aw = min(household.ed_aw * 1.04, BEHAVIORAL_SCALE_MAX)
            household.update_awareness()
            
            for action in ['investment', 'conservation', 'switching']:
                if household.su_nor[action] < 6.5:
                    household.su_nor[action] = min(household.su_nor[action] * 1.06, BEHAVIORAL_SCALE_MAX)
                if household.pbc[action] < 6.5:
                    household.pbc[action] = min(household.pbc[action] * 1.04, BEHAVIORAL_SCALE_MAX)
                    
        elif policy == "Carbon price pressure-100":
            if household.know < 6.5:
                household.know = min(household.know * 1.06, BEHAVIORAL_SCALE_MAX)
            if household.cee_aw < 6.5:
                household.cee_aw = min(household.cee_aw * 1.06, BEHAVIORAL_SCALE_MAX)
            if household.ed_aw < 6.5:
                household.ed_aw = min(household.ed_aw * 1.06, BEHAVIORAL_SCALE_MAX)
            household.update_awareness()
            
            for action in ['investment', 'conservation', 'switching']:
                if household.su_nor[action] < 6.5:
                    household.su_nor[action] = min(household.su_nor[action] * 1.10, BEHAVIORAL_SCALE_MAX)
                if household.pbc[action] < 6.5:
                    household.pbc[action] = min(household.pbc[action] * 1.05, BEHAVIORAL_SCALE_MAX)
        
        # Note: "Carbon price pressure-2020" would go here if needed


    def decide_action(self, household, market_state: Dict, utility_calculator) -> List[bool]:
        """
        Make action decision based on utilities and thresholds.
        Uses dictionary-based utility access.
        """
        actions_taken = [False, False, False]
        
        if household.h_q <= 0:
            return actions_taken
        
        if household.flag == 1:  # Brown user
            # Get utilities for brown user
            exp_utils = household.utility_exp['brown']
            actual_utils = household.utility_actual['brown']
            
            # Find max expected utility
            max_util = max(exp_utils.values())
            
            # Action 1: Investment
            if not household.act11:
                if exp_utils['investment'] >= max_util and exp_utils['investment'] >= actual_utils['investment']:
                    household.act1 = True
                    household.act11 = True
                    household.hh_actions[0] = 1
                    actions_taken[0] = True
            
            # Action 2: Conservation
            if not household.act21:
                if exp_utils['conservation'] >= max_util and exp_utils['conservation'] >= actual_utils['conservation']:
                    household.act2 = True
                    household.act21 = True
                    household.act50 = True
                    household.hh_actions[2] = 1
                    actions_taken[1] = True
            
            # Action 3: Switching to green
            if not (household.act31 or household.act32):
                if exp_utils['switching'] >= max_util and exp_utils['switching'] >= actual_utils['switching']:
                    household.act3 = True
                    household.act31 = True
                    household.flag = 2
                    household.hh_actions[4] = 1
                    actions_taken[2] = True
        
        elif household.flag == 0:  # Grey user
            exp_utils = household.utility_exp['grey']
            actual_utils = household.utility_actual['grey']
            
            max_util = max(exp_utils.values())
            
            # Action 1: Investment
            if not household.act12:
                if exp_utils['investment'] >= max_util and exp_utils['investment'] >= actual_utils['investment']:
                    household.act1 = True
                    household.act12 = True
                    household.hh_actions[1] = 1
                    actions_taken[0] = True
            
            # Action 2: Conservation
            if not household.act40:
                if exp_utils['conservation'] >= max_util and exp_utils['conservation'] >= actual_utils['conservation']:
                    household.act2 = True
                    household.act50 = True
                    household.act40 = True
                    household.hh_actions[3] = 1
                    actions_taken[1] = True
            
            # Action 3: Switching to brown
            if not household.act32:
                if exp_utils['switching'] >= max_util and exp_utils['switching'] >= actual_utils['switching']:
                    household.act3 = True
                    household.act32 = True
                    household.flag = 1
                    household.hh_actions[5] = 1
                    actions_taken[2] = True
        
        elif household.flag == 2:  # Green user
            exp_utils = household.utility_exp['green']
            actual_utils = household.utility_actual['green']
            
            max_util = max(exp_utils['investment'], exp_utils['conservation'])
            
            # Action 1: Investment
            if not household.act11:
                if exp_utils['investment'] >= max_util and exp_utils['investment'] >= actual_utils['investment']:
                    household.act1 = True
                    household.act11 = True
                    household.hh_actions[0] = 1
                    actions_taken[0] = True
            
            # Action 2: Conservation
            if not household.act21:
                if exp_utils['conservation'] >= max_util and exp_utils['conservation'] >= actual_utils['conservation']:
                    household.act2 = True
                    household.act21 = True
                    household.hh_actions[2] = 1
                    actions_taken[1] = True
        
        return actions_taken

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
            price = prices.get('m_p_grey') if household.flag == 0 else prices.get('m_p_brown')

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
            old_price = prices.get('m_p_grey') if household.flag != 0 else prices.get('m_p_brown')
            new_price = prices.get('m_p_brown') if household.flag == 0 else prices.get('m_p_green')
            switching_benefit = (old_price - new_price) * household.h_q
            household.h_switch = switching_benefit
        
        return investment, switching_benefit
    
    def calculate_emissions_avoided(self, household, prices: Dict[str, float]) -> float:
        """
        Calculate CO2 emissions avoided by actions.
        """
        co2_factor = 0.5  # kg CO2 per kWh of FF electricity
        emissions_avoided = 0.0
        
        # Initialize em_avoided as dictionary if it's a list or doesn't exist
        if not hasattr(household, 'em_avoided') or isinstance(household.em_avoided, list):
            household.em_avoided = {
                'investment': 0.0,
                'conservation': 0.0,
                'switching': 0.0
            }
        
        # Investment savings
        if household.act1 or household.act11 or household.act12:
            saved = INVESTMENT_PV_ENERGY_OUTPUT * co2_factor
            emissions_avoided += saved
            household.em_avoided['investment'] += saved
        
        # Conservation savings (only if was on FF before)
        if household.act2 or household.act21 or household.act40:
            if household.flag == 0:  # Currently on FF/grey
                conservation_amount = household.h_q * CONSERVATION_RATE
                saved = conservation_amount * co2_factor
                emissions_avoided += saved
                household.em_avoided['conservation'] += saved
        
        # Switching to renewable
        if household.act3 or household.act31 or household.act32:
            saved = household.h_q * co2_factor
            emissions_avoided += saved
            household.em_avoided['switching'] += saved
        
        return emissions_avoided
        # DELETE EVERYTHING AFTER THIS LINE - there's duplicate old code
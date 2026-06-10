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

    def calculate_satisfaction(self, household) -> str:
        """
        Determine if household is satisfied or regrets their actions.
        Matches NetLogo's satisfy procedure.
        
        Returns:
            satisfaction status string:
            - "keepact1" / "regretact1" for investment
            - "keepact2" / "regretact2" for conservation
            - "keepact3" / "regretact3" for switching
            - "none" if no action taken this year
        """
        # Check Investment satisfaction (act1)
        if household.act1:
            if household.flag == 1:  # Brown user invested
                exp_util = household.utility_exp['brown']['investment']
                actual_util = household.utility_actual['brown']['investment']
                if actual_util >= exp_util:
                    return "keepact1"
                else:
                    return "regretact1"
            
            elif household.flag == 0:  # Grey user invested
                exp_util = household.utility_exp['grey']['investment']
                actual_util = household.utility_actual['grey']['investment']
                if actual_util >= exp_util:
                    return "keepact1"
                else:
                    return "regretact1"
            
            elif household.flag == 2:  # Green user invested
                exp_util = household.utility_exp['green']['investment']
                actual_util = household.utility_actual['green']['investment']
                if actual_util >= exp_util:
                    return "keepact1"
                else:
                    return "regretact1"
        
        # Check Conservation satisfaction (act50)
        if household.act50:
            if household.flag == 1:  # Brown user conserved
                exp_util = household.utility_exp['brown']['conservation']
                actual_util = household.utility_actual['brown']['conservation']
                if actual_util >= exp_util:
                    return "keepact2"
                else:
                    return "regretact2"
            
            elif household.flag == 0:  # Grey user conserved
                exp_util = household.utility_exp['grey']['conservation']
                actual_util = household.utility_actual['grey']['conservation']
                if actual_util >= exp_util:
                    return "keepact2"
                else:
                    return "regretact2"
        
        # Check Switching satisfaction (act3)
        if household.act3:
            if household.flag == 2:  # Switched to green (act31)
                exp_util = household.utility_exp['green']['switching']
                actual_util = household.utility_actual['green']['switching']
                if actual_util >= exp_util:
                    return "keepact3"
                else:
                    return "regretact3"
            
            elif household.flag == 1:  # Switched to brown (act32)
                exp_util = household.utility_exp['brown']['switching']
                actual_util = household.utility_actual['brown']['switching']
                if actual_util >= exp_util:
                    return "keepact3"
                else:
                    return "regretact3"
        
        return "none"


    def apply_regret(self, household, satisfaction: str, learning_type: str) -> None:
        """
        Apply regret effects when households are dissatisfied with their actions.
        Matches NetLogo's regret procedure.
        
        When a household regrets an action, it reduces:
        - Personal norms (per_nab)
        - Perceived Behavioral Control (pbc)
        - Social norms (su_nor)
        
        Args:
            household: Household object
            satisfaction: Satisfaction status from calculate_satisfaction
            learning_type: Current learning type (affects influence spread)
        """
        # Only Fast adaptation and Slow adaptation have regret effects
        if learning_type not in ["Fast adaptation", "Slow adaptation"]:
            return
        
        # === REGRET FROM CONSERVATION (act2 / act50) ===
        if satisfaction == "regretact2":
            if learning_type == "Fast adaptation":
                # Reduce personal norm for conservation
                if household.per_nab.get('conservation', 0) >= 1:
                    household.per_nab['conservation'] = max(0, household.per_nab['conservation'] * 0.95)
                
                # Reduce PBC for conservation
                if household.pbc.get('conservation', 0) >= 1:
                    household.pbc['conservation'] = max(0, household.pbc['conservation'] * 0.95)
                
                # Reduce social norm for conservation
                if household.su_nor.get('conservation', 0) >= 1:
                    household.su_nor['conservation'] = max(0, household.su_nor['conservation'] * 0.97)
            
            elif learning_type == "Slow adaptation":
                # In Slow adaptation, regret spreads to neighbors
                # This will be handled in the learning mechanism
                # Mark that this household has regret to spread
                if not hasattr(household, 'regret_to_spread'):
                    household.regret_to_spread = {}
                household.regret_to_spread['conservation'] = True
        
        # === REGRET FROM SWITCHING (act3) ===
        elif satisfaction == "regretact3":
            if learning_type == "Fast adaptation":
                # Reduce personal norm for switching
                if household.per_nab.get('switching', 0) >= 1:
                    household.per_nab['switching'] = max(0, household.per_nab['switching'] * 0.95)
                
                # Reduce PBC for switching
                if household.pbc.get('switching', 0) >= 1:
                    household.pbc['switching'] = max(0, household.pbc['switching'] * 0.95)
                
                # Reduce social norm for switching
                if household.su_nor.get('switching', 0) >= 1:
                    household.su_nor['switching'] = max(0, household.su_nor['switching'] * 0.97)
            
            elif learning_type == "Slow adaptation":
                if not hasattr(household, 'regret_to_spread'):
                    household.regret_to_spread = {}
                household.regret_to_spread['switching'] = True
        
        # === REGRET FROM INVESTMENT (act1) ===
        elif satisfaction == "regretact1":
            if learning_type == "Fast adaptation":
                # Reduce personal norm for investment
                if household.per_nab.get('investment', 0) >= 1:
                    household.per_nab['investment'] = max(0, household.per_nab['investment'] * 0.95)
                
                # Reduce PBC for investment
                if household.pbc.get('investment', 0) >= 1:
                    household.pbc['investment'] = max(0, household.pbc['investment'] * 0.95)
                
                # Reduce social norm for investment
                if household.su_nor.get('investment', 0) >= 1:
                    household.su_nor['investment'] = max(0, household.su_nor['investment'] * 0.97)
            
            elif learning_type == "Slow adaptation":
                if not hasattr(household, 'regret_to_spread'):
                    household.regret_to_spread = {}
                household.regret_to_spread['investment'] = True


    def decide_action(self, household, market_state: Dict, utility_calculator) -> List[bool]:
        """
        Make action decision based on utilities and thresholds.
        Uses dictionary-based utility access.
        Matches NetLogo's action procedure.
        
        Returns:
            List of [investment_taken, conservation_taken, switching_taken]
        """
        actions_taken = [False, False, False]
        
        if household.h_q <= 0:
            return actions_taken
        
        # Check if all actions already taken (early exit)
        if all(household.hh_actions):
            return actions_taken
        
        # === BROWN USER (flag == 1) ===
        if household.flag == 1:
            exp_utils = household.utility_exp['brown']
            actual_utils = household.utility_actual['brown']
            
            # Find max expected utility among available actions
            # Only consider actions that haven't been taken yet
            available_utilities = {}
            if not household.act11:
                available_utilities['investment'] = exp_utils['investment']
            if not household.act21:
                available_utilities['conservation'] = exp_utils['conservation']
            if not (household.act31 or household.act32):
                available_utilities['switching'] = exp_utils['switching']
            
            if not available_utilities:
                return actions_taken
            
            max_util = max(available_utilities.values())
            
            # Action 1: Investment (PV installation)
            if not household.act11:
                if (exp_utils['investment'] >= max_util and 
                    exp_utils['investment'] >= actual_utils['investment']):
                    household.act1 = True
                    household.act11 = True
                    household.hh_actions[0] = 1
                    actions_taken[0] = True
            
            # Action 2: Conservation (energy efficiency)
            if not household.act21:
                if (exp_utils['conservation'] >= max_util and 
                    exp_utils['conservation'] >= actual_utils['conservation']):
                    household.act2 = True
                    household.act21 = True
                    household.act50 = True
                    household.hh_actions[2] = 1
                    actions_taken[1] = True
            
            # Action 3: Switching to green electricity
            if not (household.act31 or household.act32):
                if (exp_utils['switching'] >= max_util and 
                    exp_utils['switching'] >= actual_utils['switching']):
                    household.act3 = True
                    household.act31 = True
                    household.flag = 2  # Switch to green
                    household.hh_actions[4] = 1
                    actions_taken[2] = True
        
        # === GREY USER (flag == 0) ===
        elif household.flag == 0:
            exp_utils = household.utility_exp['grey']
            actual_utils = household.utility_actual['grey']
            
            # Find max expected utility among available actions
            available_utilities = {}
            if not household.act12:
                available_utilities['investment'] = exp_utils['investment']
            if not household.act40:
                available_utilities['conservation'] = exp_utils['conservation']
            if not household.act32:
                available_utilities['switching'] = exp_utils['switching']
            
            if not available_utilities:
                return actions_taken
            
            max_util = max(available_utilities.values())
            
            # Action 1: Investment (PV installation)
            if not household.act12:
                if (exp_utils['investment'] >= max_util and 
                    exp_utils['investment'] >= actual_utils['investment']):
                    household.act1 = True
                    household.act12 = True
                    household.hh_actions[1] = 1
                    actions_taken[0] = True
            
            # Action 2: Conservation (energy efficiency)
            if not household.act40:
                if (exp_utils['conservation'] >= max_util and 
                    exp_utils['conservation'] >= actual_utils['conservation']):
                    household.act2 = True
                    household.act50 = True
                    household.act40 = True
                    household.hh_actions[3] = 1
                    actions_taken[1] = True
            
            # Action 3: Switching to brown electricity
            if not household.act32:
                if (exp_utils['switching'] >= max_util and 
                    exp_utils['switching'] >= actual_utils['switching']):
                    household.act3 = True
                    household.act32 = True
                    household.flag = 1  # Switch to brown
                    household.hh_actions[5] = 1
                    actions_taken[2] = True
        
        # === GREEN USER (flag == 2) ===
        elif household.flag == 2:
            exp_utils = household.utility_exp['green']
            actual_utils = household.utility_actual['green']
            
            # Only investment and conservation available for green users
            available_utilities = {}
            if not household.act11:
                available_utilities['investment'] = exp_utils['investment']
            if not household.act21:
                available_utilities['conservation'] = exp_utils['conservation']
            
            if not available_utilities:
                return actions_taken
            
            max_util = max(available_utilities.values())
            
            # Action 1: Further Investment
            if not household.act11:
                if (exp_utils['investment'] >= max_util and 
                    exp_utils['investment'] >= actual_utils['investment']):
                    household.act1 = True
                    household.act11 = True
                    household.hh_actions[0] = 1
                    actions_taken[0] = True
            
            # Action 2: Conservation
            if not household.act21:
                if (exp_utils['conservation'] >= max_util and 
                    exp_utils['conservation'] >= actual_utils['conservation']):
                    household.act2 = True
                    household.act21 = True
                    household.hh_actions[2] = 1
                    actions_taken[1] = True
            
            # No switching action for green users (already at lowest carbon)
        
        return actions_taken

    def calculate_energy_savings(self, household, prices: Dict[str, float]) -> float:
        """
        Calculate energy saved from actions and money saved from conservation.
        Matches NetLogo's save procedure.
        
        Args:
            household: Household object
            prices: Dictionary with electricity prices
            
        Returns:
            Total energy saved (kWh)
        """
        energy_saved = 0.0
        
        # === INVESTMENT (Action 1) ===
        # NetLogo uses act1 (the annual flag set in action procedure)
        if household.act1:
            # Fixed energy saving from PV installation (1700 kWh/year)
            energy_saved += INVESTMENT_PV_ENERGY_OUTPUT
            household.h_invest_save += INVESTMENT_PV_ENERGY_OUTPUT
            
            # Investment cost (487.59 €/year for 10 years)
            # This is handled in calculate_financial_outcomes
        
        # === CONSERVATION (Action 2 / act50) ===
        # NetLogo uses act50 (the annual conservation flag)
        if household.act2 or household.act50:
            # Conservation saves 50% of current consumption
            conservation_amount = household.h_q * CONSERVATION_RATE
            energy_saved += conservation_amount
            household.h_conserv = conservation_amount
            
            # Calculate money saved based on current energy source
            # NetLogo: if flag? = 1 (brown) use m_p_lce
            #          if flag? = 0 (grey) use m_p_ff
            #          if flag? = 2 (green) no money saved
            if household.flag == 0:  # Grey user
                price = prices.get('m_p_grey', 0.15)
                household.h_conserv_p = conservation_amount * price
            elif household.flag == 1:  # Brown user
                price = prices.get('m_p_brown', 0.15)
                household.h_conserv_p = conservation_amount * price
            else:  # Green user (flag == 2)
                household.h_conserv_p = 0.0
        
        return energy_saved

    def calculate_financial_outcomes(self, household, prices: Dict[str, float]) -> Tuple[float, float]:
        """
        Calculate investment and switching financial outcomes.
        Matches NetLogo's save procedure for investments and switching.
        
        Args:
            household: Household object
            prices: Dictionary with electricity prices
            
        Returns:
            Tuple of (total_investment, switching_benefit)
        """
        investment = 0.0
        switching_benefit = 0.0
        
        # === INVESTMENT COST (10-year payback) ===
        # NetLogo: if (act1 = True) [set h_invest 487.59]

        if household.act1:
            if household.counter_invest < 10:  # 10-year payback period
                investment = INVESTMENT_PV_ANNUAL_COST  # 487.59 € per year
                household.h_invest = investment
                household.counter_invest += 1
                household.h_invest_total += investment
        
        # === SWITCHING BENEFIT (price difference savings) ===
        # NetLogo: if act32 = True [set h_switch ((m_p_lce - m_p_ff) * h_q)]

        if household.act32:  # Grey -> Brown switching
            old_price = prices.get('m_p_grey', 0.15)
            new_price = prices.get('m_p_brown', 0.15)
            switching_benefit = (old_price - new_price) * household.h_q
            household.h_switch = switching_benefit
        
        elif household.act31:  # Brown -> Green switching
            old_price = prices.get('m_p_brown', 0.15)
            new_price = prices.get('m_p_green', 0.15)
            switching_benefit = (old_price - new_price) * household.h_q
            household.h_switch = switching_benefit
        
        return investment, switching_benefit
    
    def calculate_emissions_avoided(self, household, prices: Dict[str, float]) -> float:
        """
        Calculate CO2 emissions avoided by actions.
        Matches NetLogo's emissions procedure.
        
        Emission factors (from NetLogo):
        - Grey electricity (coal): 0.0009 tons CO2 per kWh (0.9 kg CO2 per kWh)
        - Brown electricity (gas): 0.0003 tons CO2 per kWh (0.3 kg CO2 per kWh)
        - Green electricity: 0.0
        
        Args:
            household: Household object
            prices: Dictionary with market info (used for price-based calculations)
            
        Returns:
            Total CO2 avoided (kg)
        """
        # Emission factors (converted to kg CO2 per kWh)
        # NetLogo uses 0.0009 tons/kWh = 0.9 kg/kWh for grey
        # NetLogo uses 0.0003 tons/kWh = 0.3 kg/kWh for brown
        CO2_FACTOR_GREY = 0.9   # kg CO2 per kWh for grey electricity
        CO2_FACTOR_BROWN = 0.3  # kg CO2 per kWh for brown electricity
        
        emissions_avoided = 0.0
        
        # Initialize em_avoided as dictionary if needed
        if not hasattr(household, 'em_avoided') or isinstance(household.em_avoided, list):
            household.em_avoided = {
                'investment': 0.0,
                'conservation': 0.0,
                'switching': 0.0
            }
        
        # === INVESTMENT EMISSIONS AVOIDED ===
        # NetLogo: if act1 = True and flag? = 0 (grey) or flag? = 1 (brown)
        if household.act1:
            if household.flag == 0:  # Grey user
                emissions_saved = INVESTMENT_PV_ENERGY_OUTPUT * CO2_FACTOR_GREY
                emissions_avoided += emissions_saved
                household.em_avoided['investment'] += emissions_saved
            elif household.flag == 1:  # Brown user
                emissions_saved = INVESTMENT_PV_ENERGY_OUTPUT * CO2_FACTOR_BROWN
                emissions_avoided += emissions_saved
                household.em_avoided['investment'] += emissions_saved
            # Green user: investment avoids no additional emissions (already green)
        
        # === CONSERVATION EMISSIONS AVOIDED ===
        # NetLogo: only applies if currently on grey or brown
        if household.act2 or household.act50:
            conservation_amount = household.h_q * CONSERVATION_RATE
            
            if household.flag == 0:  # Grey user
                emissions_saved = conservation_amount * CO2_FACTOR_GREY
                emissions_avoided += emissions_saved
                household.em_avoided['conservation'] += emissions_saved
            elif household.flag == 1:  # Brown user
                emissions_saved = conservation_amount * CO2_FACTOR_BROWN
                emissions_avoided += emissions_saved
                household.em_avoided['conservation'] += emissions_saved
            # Green user: conservation avoids no additional emissions
        
        # === SWITCHING EMISSIONS AVOIDED ===
        # NetLogo: switching from grey to brown or brown to green
        if household.act32:  # Grey -> Brown switching
            # Avoided emissions = entire consumption * (grey factor - brown factor)
            emissions_saved = household.h_q * (CO2_FACTOR_GREY - CO2_FACTOR_BROWN)
            emissions_avoided += emissions_saved
            household.em_avoided['switching'] += emissions_saved
        
        elif household.act31:  # Brown -> Green switching
            # Avoided emissions = entire consumption * brown factor
            emissions_saved = household.h_q * CO2_FACTOR_BROWN
            emissions_avoided += emissions_saved
            household.em_avoided['switching'] += emissions_saved
        
        return emissions_avoided
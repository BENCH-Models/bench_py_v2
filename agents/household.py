"""
Household Agent Class for BENCH Model
Represents individual household decision-making and energy consumption
"""

import uuid
from typing import Dict, List, Optional
from utils.constants import (
    FLAG_GRAY, FLAG_BROWN, FLAG_GREEN, FLAG_NAMES,
    GUILT_LOW, GUILT_HIGH,
    ACTION_INVESTMENT, ACTION_CONSERVATION, ACTION_SWITCHING,
    BEHAVIORAL_SCALE_MAX, BEHAVIORAL_SCALE_MIN,
    PBC_MIN, PBC_MAX,
    INVESTMENT_PV_ANNUAL_COST, INVESTMENT_PV_ENERGY_OUTPUT,
    CONSERVATION_RATE
)


class Household:
    """
    Represents a single household agent in the BENCH model.
    
    Attributes cover:
    - Demographic characteristics (income group, age, ownership)
    - Energy consumption (quantity, type, source)
    - Behavioral factors (knowledge, awareness, norms, motivation, PBC)
    - Utilities (expected and actual for each action)
    - Actions taken (investment, conservation, switching)
    - Financial outcomes (investments, savings)
    - Environmental outcomes (emissions avoided)
    - Memory and learning
    """
    
    def __init__(self, h_id: int, income_group: int, income: float, 
                 consumption_q: float, energy_flag: int, 
                 dwelling_label: int, owner: bool = True, **kwargs):
        """
        Initialize a household agent.
        
        Args:
            h_id: Unique household identifier
            income_group: Income group (1-7)
            income: Annual household income
            consumption_q: Baseline electricity consumption (kWh/year)
            energy_flag: Current energy source (0=gray, 1=brown, 2=green)
            dwelling_label: Energy efficiency label (1-6, A-F)
            owner: Whether household owns dwelling (affects investment eligibility)
            **kwargs: Additional attributes from data
        """
        # === IDENTIFICATION ===
        self.h_id = h_id
        self.unique_id = uuid.uuid4()  # For tracking in model
        
        # === DEMOGRAPHIC ===
        self.h_income_group = income_group  # 1-7
        self.h_income = income
        self.h_age = kwargs.get('h_age', None)
        self.owner = owner
        
        # === DWELLING ===
        self.dw_el = dwelling_label  # 1-6 (A-F equivalent)
        self.dw_st = kwargs.get('dw_st', 0)  # Dwelling structure type
        
        # === ENERGY CONSUMPTION ===
        self.h_q = consumption_q  # Base electricity consumption (kWh/year)
        self.flag = energy_flag  # 0=gray, 1=brown, 2=green
        
        # === BEHAVIORAL ATTRIBUTES (0-7 scale or specified range) ===
        self.know = kwargs.get('know', 0.0)  # Knowledge (0-7)
        self.cee_aw = kwargs.get('cee_aw', 0.0)  # Climate/environmental awareness
        self.ed_aw = kwargs.get('ed_aw', 0.0)  # Education/awareness
        self.h_aware = 0.0  # Average awareness: (know + cee_aw + ed_aw) / 3
        
        self.guilt = GUILT_LOW  # 'L' (low) or 'H' (high)
        self.K = 0.0  # Normalized guilt factor (0-1): h_aware / 7
        
        # === NORMS & ATTITUDES (0-7 scale) ===
        # Personal norms for 3 action types
        self.per_nab = [0.0, 0.0, 0.0]  # [action1, action2, action3]
        
        # Social/subjective norms for 3 action types
        self.su_nor = [0.0, 0.0, 0.0]
        
        # === MOTIVATION (0-7 scale, aggregated) ===
        self.h_motiv = [0.0, 0.0, 0.0]  # Average of personal + social norms
        self.M = [0.0, 0.0, 0.0]  # Normalized motivation (0-1): motiv / 7
        
        # Responsibility (behavioral factor)
        self.responsibility = False
        
        # === PERCEIVED BEHAVIORAL CONTROL (0-7 scale) ===
        self.pbc = [0.0, 0.0, 0.0]  # PBC for [action1, action2, action3]
        
        # === CONSIDERATION FACTORS (behavioral adjustment, 0.2-0.7) ===
        self.delta = [0.0, 0.0, 0.0]  # [delta1, delta2, delta3]
        self.consideration = [GUILT_LOW, GUILT_LOW, GUILT_LOW]  # 'L' or 'H' for each action
        
        # === UTILITIES (Budget/consumption combinations) ===
        # z_brown/z_grey/z_green: discretionary income for different scenarios
        self.z_brown = [0.0, 0.0, 0.0]  # z_brown1, z_brown2, z_brown3
        self.z_grey = [0.0, 0.0, 0.0]   # z_grey1, z_grey2, z_grey3
        self.z_green = [0.0, 0.0, 0.0] # z_green1, z_green2, z_green3
        
        # Normalized z values (0-1)
        self.z_brown_norm = [0.0, 0.0, 0.0]
        self.z_grey_norm = [0.0, 0.0, 0.0]
        self.z_green_norm = [0.0, 0.0, 0.0]
        
        # === EXPECTED UTILITIES ===
        # green electricity utilities
        self.utility_exp_brown = [0.0, 0.0, 0.0]  # UE_brown1, UE_brown2, UE_brown3
        
        # grey electricity utilities
        self.utility_exp_grey = [0.0, 0.0, 0.0]   # UE_grey1, UE_grey2, UE_grey3
        
        # Zero-carbon utilities
        self.utility_exp_green = [0.0, 0.0, 0.0]  # UE_green1, UE_green2, UE_green3
        
        # === ACTUAL UTILITIES ===
        self.utility_brown = [0.0, 0.0, 0.0]
        self.utility_grey = [0.0, 0.0, 0.0]
        self.utility_green = [0.0, 0.0, 0.0]
        
        # === ACTIONS TAKEN (boolean flags) ===
        # Action 1: Investment (PV installation)
        self.act1 = False
        self.act11 = False  # Specific to Brown
        self.act12 = False  # Specific to Grey
        
        # Action 2: Conservation (energy efficiency)
        self.act2 = False
        self.act50 = False  # Alternative naming for conservation
        self.act21 = False  # Conservation action
        self.act40 = False  # Alternative conservation action
        
        # Action 3: Switching (to renewable)
        self.act3 = False
        self.act31 = False  # Switch to brown
        self.act32 = False  # Switch from brown to green
        
        # Action vector: [act11, act12, act21, act40, act31, act32]
        self.hh_actions = [0, 0, 0, 0, 0, 0]
        
        # === FINANCIAL OUTCOMES ===
        # Investment (PV installation)
        self.h_invest = 0.0  # Annual investment amount
        self.h_invest_save = 0.0  # Cumulative energy saved by PV
        self.h_invest_total = 0.0  # Total invested
        self.counter_invest = 0  # Years invested (up to payback period)
        
        # Conservation (energy efficiency)
        self.h_conserv = 0.0  # Energy saved through conservation (kWh)
        self.h_conserv_p = 0.0  # Money saved through conservation
        self.h_conserv_1_7 = [0.0] * 7  # Conservation by income group (for tracking)
        
        # Switching (to renewable energy)
        self.h_switch = 0.0  # Money saved/spent on switching
        
        # === ENVIRONMENTAL OUTCOMES ===
        self.em_total = 0.0  # Total CO2 emissions (kg)
        self.em_avoided = [0.0, 0.0, 0.0]  # Emissions avoided by action type
        
        # === HOUSEHOLD STATUS ===
        self.hh_sta = "Normal"  # "Normal" or "Low-paid1/2/3" (income constraints)
        
        # === MEMORY & LEARNING ===
        self.memory = {}  # Store historical decisions
        self.influences = {
            'influence_know': 0.0,
            'influence_cee_aw': 0.0,
            'influence_ed_aw': 0.0,
            'influence_per': 0.0,
            'influence_su': 0.0,
        }
        self.experience = {}  # Track past experiences
        
    def update_awareness(self) -> None:
        """Calculate average awareness from components."""
        self.h_aware = (self.know + self.cee_aw + self.ed_aw) / 3.0
        
        # Determine guilt level based on awareness
        if self.h_aware < 5.21:
            self.guilt = GUILT_LOW
        else:
            self.guilt = GUILT_HIGH
        
        # Normalize guilt to K factor (0-1)
        if self.guilt == GUILT_HIGH:
            self.K = self.h_aware / BEHAVIORAL_SCALE_MAX
        else:
            self.K = 0.0
    
    def update_motivation(self, case_study: str) -> None:
        """
        Calculate motivation from personal and social norms.
        Sets responsibility flag based on thresholds (varies by case study).
        
        Args:
            case_study: "Netherlands-Overijssel" or "Spain-Navarre"
        """
        # Calculate average motivation for each action
        for i in range(3):
            self.h_motiv[i] = (self.per_nab[i] + self.su_nor[i]) / 2.0
            
            # Normalize motivation (0-1)
            self.M[i] = self.h_motiv[i] / BEHAVIORAL_SCALE_MAX
        
        # Determine responsibility based on case-specific thresholds
        self.responsibility = False
        
        if self.guilt == GUILT_HIGH:
            if case_study == "Spain-Navarre":
                thresholds = [
                    (5.67, 4.77),  # Action 1
                    (5.40, 4.45),  # Action 2
                    (5.78, 5.05),  # Action 3
                ]
            elif case_study == "Netherlands-Overijssel":
                thresholds = [
                    (5.67, 4.45),  # Action 1
                    (5.40, 3.66),  # Action 2
                    (5.78, 5.05),  # Action 3
                ]
            else:
                thresholds = [(0, 0), (0, 0), (0, 0)]
            
            for i in range(3):
                if (self.per_nab[i] >= thresholds[i][0] and 
                    self.su_nor[i] >= thresholds[i][1]):
                    self.responsibility = True
    
    def consider_constraints(self, action_type: int) -> None:
        """
        Determine consideration level and delta factor based on PBC.
        
        Args:
            action_type: 0 (investment), 1 (conservation), or 2 (switching)
        """
        pbc_value = self.pbc[action_type]
        
        # Determine consideration level
        if pbc_value < 4:
            self.consideration[action_type] = GUILT_LOW
        else:
            self.consideration[action_type] = GUILT_HIGH
        
        # Determine delta factor based on PBC value
        if pbc_value < 2:
            self.delta[action_type] = 0.2
        elif pbc_value < 3:
            self.delta[action_type] = 0.3
        elif pbc_value < 4:
            self.delta[action_type] = 0.4
        elif pbc_value < 5:
            self.delta[action_type] = 0.5
        elif pbc_value < 6:
            self.delta[action_type] = 0.6
        else:
            self.delta[action_type] = 0.7
    
    def calculate_budgets(self, prices: Dict[str, float]) -> None:
        """
        Calculate discretionary income (z) for different scenarios.
        
        Args:
            prices: Dictionary with keys 'm_p_grey', 'm_p_brown', 'm_p_green'
        """
        if self.h_q <= 0:
            return
        
        m_p_grey = prices.get('m_p_grey', 0.1)
        m_p_brown = prices.get('m_p_brown', 0.15)
        m_p_green = prices.get('m_p_green', 0.12)
        
        # Scenario 1: No action (baseline consumption)
        base_fixed = 1700 * m_p_brown + 487.59  # Fixed costs
        
        self.z_brown[0] = self.h_income - ((self.h_q * m_p_brown) + base_fixed + 
                                          self.h_conserv_p + self.h_switch)
        self.z_grey[0] = self.h_income - ((self.h_q * m_p_grey) + base_fixed + 
                                         self.h_conserv_p + self.h_switch)
        self.z_green[0] = self.h_income - ((self.h_q * m_p_green) + base_fixed + 
                                           self.h_conserv_p + self.h_switch)
        
        # Scenario 2: After investment
        self.z_brown[1] = self.h_income - ((self.h_q * m_p_brown) + self.h_invest_save * m_p_brown + 
                                          self.h_invest + (0.5 * self.h_q * m_p_brown) + self.h_switch)
        self.z_grey[1] = self.h_income - ((self.h_q * m_p_grey) + self.h_invest_save * m_p_grey + 
                                         self.h_invest + (0.5 * self.h_q * m_p_grey) + self.h_switch)
        self.z_green[1] = self.h_income - ((self.h_q * m_p_green) + self.h_invest_save * m_p_green + 
                                           self.h_invest + (0.5 * self.h_q * m_p_green) + self.h_switch)
        
        # Scenario 3: Complex switching scenarios
        self.z_brown[2] = self.h_income - ((self.h_q * m_p_green) + 
                                          self.h_invest_save * m_p_green + self.h_invest + 
                                          self.h_conserv_p + (m_p_green - m_p_brown))
        self.z_grey[2] = self.h_income - ((self.h_q * m_p_brown) + 
                                         self.h_invest_save * m_p_brown + self.h_invest + 
                                         self.h_conserv_p + (m_p_brown - m_p_grey))
        
        # Check for low-paid status
        if (self.z_grey[0] < 0 or self.z_grey[1] < 0 or self.z_brown[0] < 0 or 
            self.z_brown[1] < 0 or self.z_green[0] < 0):
            self.hh_sta = "Low-paid"
    
    def get_action_status(self) -> Dict[str, bool]:
        """Return current action status as dictionary."""
        return {
            'investment': self.act1,
            'conservation': self.act2,
            'switching': self.act3
        }
    
    def get_state_dict(self) -> Dict:
        """
        Return current household state as dictionary for output/logging.
        """
        return {
            'h_id': self.h_id,
            'year': None,  # Will be set by model
            'income_group': self.h_income_group,
            'income': self.h_income,
            'consumption': self.h_q,
            'energy_source': FLAG_NAMES.get(self.flag, 'Unknown'),
            'dwelling_label': self.dw_el,
            'awareness': self.h_aware,
            'guilt': self.guilt,
            'motivation_avg': sum(self.h_motiv) / 3,
            'pbc_avg': sum(self.pbc) / 3,
            'action_investment': self.act1,
            'action_conservation': self.act2,
            'action_switching': self.act3,
            'investment_total': self.h_invest_total,
            'conservation_savings': self.h_conserv,
            'emissions_avoided': sum(self.em_avoided),
        }

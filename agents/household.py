"""
Household Agent Class for BENCH Model
Represents individual household decision-making and energy consumption
"""

import uuid
from typing import Dict, List, Optional
from utils.constants import (
    FLAG_GRAY, FLAG_BROWN, FLAG_GREEN, FLAG_NAMES,
    GUILT_LOW, GUILT_HIGH,
    BEHAVIORAL_SCALE_MAX, BEHAVIORAL_SCALE_MIN,
    PBC_MIN, PBC_MAX,
    INVESTMENT_PV_ANNUAL_COST, INVESTMENT_PV_ENERGY_OUTPUT,
    CONSERVATION_RATE,
    ACTIONS
)


class Household:
    """
    Represents a single household agent in the BENCH model.
    """
    

    
    def __init__(self, h_id: int, income_group: int, income: float, 
                 consumption_q: float, energy_flag: int, 
                 dwelling_label: int, owner: bool = True, **kwargs):
        """
        Initialize a household agent.
        """
        # === IDENTIFICATION ===
        self.h_id = h_id
        self.unique_id = uuid.uuid4()
        
        # === DEMOGRAPHIC ===
        self.h_income_group = income_group
        self.h_income = income
        self.h_age = kwargs.get('h_age')
        self.owner = owner
        self.income_trajectory = kwargs.get('income_trajectory', {2015: income})
        
        # === DWELLING ===
        self.dw_el = dwelling_label
        self.dw_st = kwargs.get('dw_st')
        
        # === ENERGY CONSUMPTION ===
        self.h_q = consumption_q
        self.flag = energy_flag  # 0=grey, 1=brown, 2=green
        
        # === BEHAVIORAL ATTRIBUTES ===
        self.know = kwargs.get('know', 0.0)
        self.cee_aw = kwargs.get('cee_aw', 0.0)
        self.ed_aw = kwargs.get('ed_aw', 0.0)
        self.h_aware = 0.0
        self.guilt = GUILT_LOW
        self.K = 0.0
        
        # === NORMS & ATTITUDES - Now using dictionaries ===
        self.per_nab = {
            'investment': 0.0,
            'conservation': 0.0,
            'switching': 0.0
        }
        
        self.su_nor = {
            'investment': 0.0,
            'conservation': 0.0,
            'switching': 0.0
        }
        
        # === MOTIVATION ===
        self.h_motiv = {
            'investment': 0.0,
            'conservation': 0.0,
            'switching': 0.0
        }
        
        self.M = {
            'investment': 0.0,
            'conservation': 0.0,
            'switching': 0.0
        }
        
        self.responsibility = False
        
        # === PERCEIVED BEHAVIORAL CONTROL ===
        self.pbc = {
            'investment': 0.0,
            'conservation': 0.0,
            'switching': 0.0
        }
        
        # === CONSIDERATION FACTORS ===
        self.delta = {
            'investment': 0.0,
            'conservation': 0.0,
            'switching': 0.0
        }
        
        self.consideration = {
            'investment': GUILT_LOW,
            'conservation': GUILT_LOW,
            'switching': GUILT_LOW
        }
        
        # === BUDGETS (Z values) - Using dictionaries ===
        self.z_grey = {
            'conservation': 0.0,
            'investment': 0.0,
            'switching': 0.0
        }
        self.z_brown = {
            'conservation': 0.0,
            'investment': 0.0,
            'switching': 0.0
        }
        self.z_green = {
            'conservation': 0.0,
            'investment': 0.0
        }
        
        # Normalized budgets
        self.z_grey_norm = {
            'conservation': 0.0,
            'investment': 0.0,
            'switching': 0.0
        }
        self.z_brown_norm = {
            'conservation': 0.0,
            'investment': 0.0,
            'switching': 0.0
        }
        self.z_green_norm = {
            'conservation': 0.0,
            'investment': 0.0
        }
        
        # === EXPECTED UTILITIES - Using nested dictionaries ===
        self.utility_exp = {
            'grey': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0},
            'brown': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0},
            'green': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0}
        }
        
        # === ACTUAL UTILITIES ===
        self.utility_actual = {
            'grey': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0},
            'brown': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0},
            'green': {'investment': 0.0, 'conservation': 0.0, 'switching': 0.0}
        }
        
        # === ACTIONS TAKEN ===
        self.act1 = False  # General investment flag
        self.act11 = False  # Brown investment
        self.act12 = False  # Grey investment
        
        self.act2 = False  # General conservation flag
        self.act50 = False  # Conservation
        self.act21 = False  # Brown/green conservation
        self.act40 = False  # Grey conservation
        
        self.act3 = False  # General switching flag
        self.act31 = False  # Brown -> green
        self.act32 = False  # Grey -> brown
        
        # Action vector: [act11, act12, act21, act40, act31, act32]
        self.hh_actions = [0, 0, 0, 0, 0, 0]
        
        # === FINANCIAL OUTCOMES ===
        self.h_invest = 0.0
        self.h_invest_save = 0.0
        self.h_invest_total = 0.0
        self.counter_invest = 0
        
        self.h_conserv = 0.0
        self.h_conserv_p = 0.0
        self.h_conserv_1_7 = [0.0] * 7
        
        self.h_switch = 0.0
        
        # === ENVIRONMENTAL OUTCOMES ===
        self.em_total = 0.0
        self.em_avoided = {
            'investment': 0.0,
            'conservation': 0.0,
            'switching': 0.0
        }
        
        # === HOUSEHOLD STATUS ===
        self.hh_sta = "Normal"
        
        # === MEMORY & LEARNING ===
        self.memory = {}
        self.influences = {
            'influence_know': 0.0,
            'influence_cee_aw': 0.0,
            'influence_ed_aw': 0.0,
            'influence_per': 0.0,
            'influence_su': 0.0,
        }
        self.experience = {}
        
        # For normalization denominators
        self.e_norm_denom = 1.0

    def set_income_for_year(self, year: int) -> None:
        """Updates the active annual income using the time series trajectory."""
        if self.income_trajectory and year in self.income_trajectory:
            self.h_income = float(self.income_trajectory[year])
        
    def update_awareness(self) -> None:
        """Calculate average awareness from components."""
        self.h_aware = (self.know + self.cee_aw + self.ed_aw) / 3.0
        
        if self.h_aware < 5.21:
            self.guilt = GUILT_LOW
        else:
            self.guilt = GUILT_HIGH
        
        if self.guilt == GUILT_HIGH:
            self.K = self.h_aware / BEHAVIORAL_SCALE_MAX
        else:
            self.K = 0.0
    
    def update_motivation(self, case_study: str) -> None:
        """
        Calculate motivation from personal and social norms.
        Uses named actions instead of indices.
        """
        # Calculate average motivation for each action
        for action in ACTIONS:
            self.h_motiv[action] = (self.per_nab[action] + self.su_nor[action]) / 2.0
            self.M[action] = self.h_motiv[action] / BEHAVIORAL_SCALE_MAX
        
        # Determine responsibility based on case-specific thresholds
        self.responsibility = False
        
        if self.guilt == GUILT_HIGH:
            if case_study == "Spain-Navarre":
                thresholds = {
                    'investment': (5.67, 4.77),
                    'conservation': (5.40, 4.45),
                    'switching': (5.78, 5.05),
                }
            elif case_study == "Netherlands-Overijssel":
                thresholds = {
                    'investment': (5.67, 4.45),
                    'conservation': (5.40, 3.66),
                    'switching': (5.78, 5.05),
                }
            else:
                thresholds = {
                    'investment': (0, 0),
                    'conservation': (0, 0),
                    'switching': (0, 0),
                }
            
            for action in ACTIONS:
                if (self.per_nab[action] >= thresholds[action][0] and 
                    self.su_nor[action] >= thresholds[action][1]):
                    self.responsibility = True
    
    def consider_constraints(self, action: str) -> None:
        """
        Determine consideration level and delta factor based on PBC.
        
        Args:
            action: 'investment', 'conservation', or 'switching'
        """
        pbc_value = self.pbc[action]
        
        # Determine consideration level
        if pbc_value < 4:
            self.consideration[action] = GUILT_LOW
        else:
            self.consideration[action] = GUILT_HIGH
        
        # Determine delta factor based on PBC value
        if pbc_value < 2:
            self.delta[action] = 0.2
        elif pbc_value < 3:
            self.delta[action] = 0.3
        elif pbc_value < 4:
            self.delta[action] = 0.4
        elif pbc_value < 5:
            self.delta[action] = 0.5
        elif pbc_value < 6:
            self.delta[action] = 0.6
        else:
            self.delta[action] = 0.7
    
    def calculate_budgets(self, prices: Dict[str, float]) -> None:
        """Calculate discretionary income (z) for different scenarios."""
        if self.h_q <= 0:
            return
            
        self.hh_sta = "Normal"
        
        m_p_grey = prices.get('m_p_grey', 0.0)
        m_p_brown = prices.get('m_p_brown', 0.0)
        m_p_green = prices.get('m_p_green', 0.0)
        
        h_q = self.h_q
        h_income = self.h_income
        h_conserv_p = self.h_conserv_p
        h_switch = self.h_switch
        h_invest = self.h_invest
        h_invest_save = self.h_invest_save

        # Scenario 1: Conservation Path
        self.z_grey['conservation'] = h_income - ((h_q * m_p_grey) + (1700 * m_p_grey) + 487.59 + h_conserv_p + h_switch)
        self.z_brown['conservation'] = h_income - ((h_q * m_p_brown) + (1700 * m_p_brown) + 487.59 + h_conserv_p + h_switch)
        self.z_green['conservation'] = h_income - ((h_q * m_p_green) + (1700 * m_p_green) + 487.59 + h_conserv_p + h_switch)

        # Scenario 2: Investment Path
        self.z_grey['investment'] = h_income - ((h_q * m_p_grey) + (h_invest_save * m_p_grey) + h_invest + ((0.5 * h_q) * m_p_grey) + h_switch)
        self.z_brown['investment'] = h_income - ((h_q * m_p_brown) + (h_invest_save * m_p_brown) + h_invest + ((0.5 * h_q) * m_p_brown) + h_switch)
        self.z_green['investment'] = h_income - ((h_q * m_p_green) + (h_invest_save * m_p_green) + h_invest + (0.5 * h_q) + h_switch)
        
        # Scenario 3: Switching Path
        self.z_brown['switching'] = h_income - ((h_q * m_p_green) + (h_invest_save * m_p_green) + h_invest + h_conserv_p + (m_p_green - m_p_brown))
        self.z_grey['switching'] = h_income - ((h_q * m_p_brown) + (h_invest_save * m_p_brown) + h_invest + h_conserv_p + (m_p_brown - m_p_grey))

        # Apply low-income safety limits
        if self.z_grey['conservation'] < 0 or self.z_grey['investment'] < 0:
            self.hh_sta = "Low-paid1"
        elif self.z_brown['conservation'] < 0 or self.z_brown['investment'] < 0 or self.z_brown.get('switching', 0) < 0:
            self.hh_sta = "Low-paid1"
        elif self.z_green['conservation'] < 0 or self.z_green['investment'] < 0:
            self.hh_sta = "Low-paid1"
    
    def get_action_status(self) -> Dict[str, bool]:
        """Return current action status as dictionary."""
        return {
            'investment': self.act1,
            'conservation': self.act2,
            'switching': self.act3
        }
    
    def get_state_dict(self) -> Dict:
        """Return current household state as dictionary for output/logging."""
        return {
            'h_id': self.h_id,
            'year': None,
            'income_group': self.h_income_group,
            'income': self.h_income,
            'consumption': self.h_q,
            'energy_source': FLAG_NAMES.get(self.flag),
            'dwelling_label': self.dw_el,
            'awareness': self.h_aware,
            'guilt': self.guilt,
            'motivation_avg': sum(self.h_motiv.values()) / 3,
            'pbc_avg': sum(self.pbc.values()) / 3,
            'action_investment': self.act1,
            'action_conservation': self.act2,
            'action_switching': self.act3,
            'investment_total': self.h_invest_total,
            'conservation_savings': self.h_conserv,
            'emissions_avoided': sum(self.em_avoided.values()),
        }
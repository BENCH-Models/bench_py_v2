"""
Population: parallel numpy arrays for all agent state.
"""
import numpy as np
from typing import List

from model.parameters import MODEL_START_YEAR, MODEL_END_YEAR

# Column indices for (N, 3) action arrays
INV = 0  # investment
CON = 1  # conservation
SWI = 2  # switching

# hh_sta int8 encoding
HH_NORMAL = 0
HH_LOWPAID = 1
HH_SELFPRODUCER = 2
HH_EFFICIENT = 3

N_YEARS = MODEL_END_YEAR - MODEL_START_YEAR + 1  # 16 years (2015-2030)


class Population:
    """
    Flat numpy array store for all 3468 agent states.
    Replaces per-Household Python objects with parallel (N,) and (N,3) arrays.
    """

    def __init__(self, n: int):
        self.N = n

        # --- Static (set once during init, never change) ---
        self.h_id = np.zeros(n, dtype=np.int32)
        self.income_group = np.zeros(n, dtype=np.int8)   # 1â€“7
        self.dw_el = np.zeros(n, dtype=np.int8)          # 1â€“6 dwelling label
        self.owner = np.zeros(n, dtype=bool)
        self.ep = np.zeros(n, dtype=np.float32)          # energy patterns

        # Income trajectory: income_traj[i, year_idx] where year_idx = year - 2015
        self.income_traj = np.zeros((n, N_YEARS), dtype=np.float64)

        # --- Dynamic scalar state ---
        self.flag = np.zeros(n, dtype=np.int8)           # 0=grey, 1=brown, 2=green
        self.h_q = np.zeros(n, dtype=np.float64)         # consumption (kWh/yr)
        self.h_income = np.zeros(n, dtype=np.float64)    # annual income
        self.alpha = np.zeros(n, dtype=np.float64)

        # Awareness components
        self.know = np.zeros(n, dtype=np.float64)
        self.cee_aw = np.zeros(n, dtype=np.float64)
        self.ed_aw = np.zeros(n, dtype=np.float64)
        self.h_aware = np.zeros(n, dtype=np.float64)
        self.guilt = np.zeros(n, dtype=bool)             # True = GUILT_HIGH (aware >= 5.21)
        self.K = np.zeros(n, dtype=np.float64)
        self.responsibility = np.zeros(n, dtype=bool)

        # Norms: (N, 3) â€” columns [INV, CON, SWI]
        self.per_nab = np.zeros((n, 3), dtype=np.float64)
        self.su_nor = np.zeros((n, 3), dtype=np.float64)
        self.pbc = np.zeros((n, 3), dtype=np.float64)
        self.delta = np.zeros((n, 3), dtype=np.float64)
        self.h_motiv = np.zeros((n, 3), dtype=np.float64)
        self.M = np.zeros((n, 3), dtype=np.float64)

        # Budget Z values
        self.z_grey = np.zeros((n, 3), dtype=np.float64)    # (N, [INV,CON,SWI])
        self.z_brown = np.zeros((n, 3), dtype=np.float64)
        self.z_green = np.zeros((n, 2), dtype=np.float64)   # (N, [INV,CON]) no SWI
        self.z_grey_norm = np.zeros((n, 3), dtype=np.float64)
        self.z_brown_norm = np.zeros((n, 3), dtype=np.float64)
        self.z_green_norm = np.zeros((n, 2), dtype=np.float64)
        self.e_norm_denom = np.ones(n, dtype=np.float64)

        # Actual-utility intermediates
        self.z_actual_brown = np.zeros(n, dtype=np.float64)
        self.z_actual_grey = np.zeros(n, dtype=np.float64)
        self.z_actual_green = np.zeros(n, dtype=np.float64)
        self.z_norm_actual = np.zeros(n, dtype=np.float64)
        self._e_max_actual: float = 1.0

        # Utility arrays: (N, 3) for [INV, CON, SWI] relative to agent's current flag
        self.utility_exp = np.zeros((n, 3), dtype=np.float64)
        self.utility_actual = np.zeros((n, 3), dtype=np.float64)

        # Annual action flags (reset each year)
        self.act1 = np.zeros(n, dtype=bool)
        self.act2 = np.zeros(n, dtype=bool)
        self.act3 = np.zeros(n, dtype=bool)
        self.act50 = np.zeros(n, dtype=bool)

        # Permanent action flags (cleared on cooldown expiry)
        self.act11 = np.zeros(n, dtype=bool)   # brown/green investment
        self.act12 = np.zeros(n, dtype=bool)   # grey investment
        self.act21 = np.zeros(n, dtype=bool)   # brown/green conservation
        self.act40 = np.zeros(n, dtype=bool)   # grey conservation
        self.act31 = np.zeros(n, dtype=bool)   # switched to green
        self.act32 = np.zeros(n, dtype=bool)   # switched to brown

        # Action vector: (N, 6) = [act11, act12, act21, act40, act31, act32]
        self.hh_actions = np.zeros((n, 6), dtype=np.int8)

        # Cooldown counters
        self.act11_year = np.zeros(n, dtype=np.int32)
        self.act12_year = np.zeros(n, dtype=np.int32)
        self.act21_year = np.zeros(n, dtype=np.int32)
        self.act40_year = np.zeros(n, dtype=np.int32)
        self.act31_year = np.zeros(n, dtype=np.int32)
        self.act32_year = np.zeros(n, dtype=np.int32)

        # Financial
        self.h_invest = np.zeros(n, dtype=np.float64)
        self.h_invest_save = np.zeros(n, dtype=np.float64)
        self.h_invest_total = np.zeros(n, dtype=np.float64)
        self.h_conserv = np.zeros(n, dtype=np.float64)
        self.h_conserv_p = np.zeros(n, dtype=np.float64)
        self.h_switch = np.zeros(n, dtype=np.float64)
        self.counter_invest = np.zeros(n, dtype=np.int32)

        # Emissions avoided: (N, 3) = [INV, CON, SWI]
        self.em_avoided = np.zeros((n, 3), dtype=np.float64)

        # Household status (int8): 0=Normal, 1=Low-paid1, 2=self-producer, 3=efficient
        self.hh_sta = np.zeros(n, dtype=np.int8)

        # Satisfaction encoding: 0=none, 1=keepact1, 2=regretact1,
        #   3=keepact2, 4=regretact2, 5=keepact3, 6=regretact3
        self.satisfaction = np.zeros(n, dtype=np.int8)

        # Grid positions
        self.grid_x = np.zeros(n, dtype=np.int32)
        self.grid_y = np.zeros(n, dtype=np.int32)

        # Neighbour index store — populated by bench_model after grid placement
        # nbr_cache[i]: variable-length int32 array of up to 8 neighbour indices
        # nbr_matrix:   (N, 8) padded with -1 for missing slots (for vectorized stats)
        self.nbr_cache: List[np.ndarray] = [np.array([], dtype=np.int32)] * n
        self.nbr_matrix: np.ndarray = np.full((n, 8), -1, dtype=np.int32)

"""
Vectorized behavioral procedures operating on Population arrays.
All procedures are numpy-level â€” no per-agent Python loops.
"""
import numpy as np
from typing import Dict

from model.population import Population, INV, CON, SWI, HH_NORMAL, HH_LOWPAID, HH_SELFPRODUCER, HH_EFFICIENT
from model.parameters import (
    BEHAVIORAL_SCALE_MAX, BEHAVIORAL_SCALE_GUILT_THRESHOLD, BEHAVIORAL_CAP,
    MODEL_START_YEAR,
    UTILITY_NORMALIZATION_FACTOR as ALPHA,
    INVESTMENT_PV_ANNUAL_COST, INVESTMENT_PV_ENERGY_OUTPUT,
    INVESTMENT_PV_PAYBACK_YEARS, CONSERVATION_RATE, CONSERVATION_MIN_KWH,
    CONSERVATION_COOLDOWN_YEARS, SWITCH_COOLDOWN_YEARS,
    CO2_FACTOR_GREY, CO2_FACTOR_BROWN,
    DELTA_LEVELS, RESPONSIBILITY_THRESHOLDS, CGE_INCOME_GROUP_COLS,
    CPINFO_RATES, REGRET_PER_NAB_PBC_RATE, REGRET_SU_NOR_RATE,
    M_P_GREY_BASE, M_P_BROWN_BASE, M_P_GREEN_BASE,
    LEARN_AWARENESS_RATE_FAST, LEARN_AWARENESS_RATE_SLOW,
    LEARN_PER_NAB_PBC_RATE, LEARN_SU_NOR_RATE,
    SLOW_LEARN_MIN_NEIGHBORS, SLOW_LEARN_NUM_NEIGHBORS,
    MEMORY_RULES,
)

# Derived from DELTA_LEVELS at import time
_DELTA_DEFAULT = min(DELTA_LEVELS.keys())
_DELTA_STEPS = sorted(
    [(lo, val) for val, (lo, _) in DELTA_LEVELS.items() if lo > 0],
    key=lambda x: x[0],
)


# ---------------------------------------------------------------------------
# Awareness / knowledge
# ---------------------------------------------------------------------------

def update_awareness(pop: Population) -> None:
    """Recalculate h_aware, guilt flag, and K for the whole population."""
    pop.h_aware[:] = (pop.know + pop.cee_aw + pop.ed_aw) / 3.0
    pop.guilt[:] = pop.h_aware >= BEHAVIORAL_SCALE_GUILT_THRESHOLD
    pop.K[:] = np.where(pop.guilt, pop.h_aware / BEHAVIORAL_SCALE_MAX, 0.0)


def apply_carbon_price_awareness(pop: Population, policy: str, year: int) -> None:
    """Vectorized cpinfo: bump awareness + norms for carbon-price policies."""
    if year < 2016 or policy == "Ref":
        return
    rates = CPINFO_RATES.get(policy)
    if rates is None:
        return

    cap  = BEHAVIORAL_CAP
    bmax = BEHAVIORAL_SCALE_MAX
    aw_rate = rates["awareness"]

    for arr in (pop.know, pop.cee_aw, pop.ed_aw):
        m = arr < cap
        arr[m] = np.minimum(arr[m] * aw_rate, bmax)

    update_awareness(pop)

    if rates["su_nor"] is not None:
        su_rate  = rates["su_nor"]
        pbc_rate = rates["pbc"]
        m = pop.su_nor < cap
        pop.su_nor[m] = np.minimum(pop.su_nor[m] * su_rate, bmax)
        m = pop.pbc < cap
        pop.pbc[m] = np.minimum(pop.pbc[m] * pbc_rate, bmax)


# ---------------------------------------------------------------------------
# Motivation
# ---------------------------------------------------------------------------

def update_motivation(pop: Population, case_study: str) -> None:
    """Compute h_motiv, M, and responsibility for all agents."""
    pop.h_motiv[:] = (pop.per_nab + pop.su_nor) / 2.0
    pop.M[:] = pop.h_motiv / BEHAVIORAL_SCALE_MAX

    pop.responsibility[:] = False
    thresh_pair = RESPONSIBILITY_THRESHOLDS.get(case_study)
    if thresh_pair is None:
        return

    t_nab, t_su = thresh_pair
    for a in range(3):
        cond = (pop.guilt
                & (pop.per_nab[:, a] >= t_nab[a])
                & (pop.su_nor[:, a]  >= t_su[a]))
        pop.responsibility |= cond


# ---------------------------------------------------------------------------
# Constraints â†’ delta
# ---------------------------------------------------------------------------

def consider_constraints(pop: Population) -> None:
    """Compute delta for all agents × actions from PBC values."""
    pbc = pop.pbc  # (N, 3)
    d = np.full_like(pbc, _DELTA_DEFAULT)
    for thresh, val in _DELTA_STEPS:
        d[pbc >= thresh] = val
    # NetLogo gates switching on responsibility; non-responsible agents get delta_SWI=0
    # so utility_exp[SWI]=0 and they never switch (investment/conservation are ungated)
    d[~pop.responsibility, SWI] = 0.0
    pop.delta[:] = d


# ---------------------------------------------------------------------------
# Income + budgets
# ---------------------------------------------------------------------------

def set_income_for_year(pop: Population, year: int) -> None:
    yi = year - MODEL_START_YEAR
    if 0 <= yi < pop.income_traj.shape[1]:
        pop.h_income[:] = pop.income_traj[:, yi]


def calculate_budgets(pop: Population, prices: Dict) -> None:
    """Vectorized discretionary-income (Z) calculation for all agents."""
    m_g = prices.get('m_p_grey',  0.0)
    m_b = prices.get('m_p_brown', 0.0)
    m_n = prices.get('m_p_green', 0.0)

    active = pop.h_q > 0
    pop.hh_sta[active] = HH_NORMAL

    q   = pop.h_q
    inc = pop.h_income
    cp  = pop.h_conserv_p
    sw  = pop.h_switch
    inv = pop.h_invest
    isv = pop.h_invest_save

    # Conservation path (col CON=1)
    pop.z_grey[:, CON]  = inc - (q * m_g + INVESTMENT_PV_ENERGY_OUTPUT * m_g + INVESTMENT_PV_ANNUAL_COST + cp + sw)
    pop.z_brown[:, CON] = inc - (q * m_b + INVESTMENT_PV_ENERGY_OUTPUT * m_b + INVESTMENT_PV_ANNUAL_COST + cp + sw)
    pop.z_green[:, CON] = inc - (q * m_n + INVESTMENT_PV_ENERGY_OUTPUT * m_n + INVESTMENT_PV_ANNUAL_COST + cp + sw)

    # Investment path (col INV=0)
    pop.z_grey[:, INV]  = inc - (q * m_g + isv * m_g + inv + CONSERVATION_RATE * q * m_g + sw)
    pop.z_brown[:, INV] = inc - (q * m_b + isv * m_b + inv + CONSERVATION_RATE * q * m_b + sw)
    pop.z_green[:, INV] = inc - (q * m_n + isv * m_n + inv + CONSERVATION_RATE * q        + sw)

    # Switching path (col SWI=2) â€” only grey & brown
    pop.z_brown[:, SWI] = inc - (q * m_n + isv * m_n + inv + cp + (m_n - m_b))
    pop.z_grey[:,  SWI] = inc - (q * m_b + isv * m_b + inv + cp + (m_b - m_g))

    lowpaid = active & (
        (pop.z_grey[:,  CON] < 0) | (pop.z_grey[:,  INV] < 0) |
        (pop.z_brown[:, CON] < 0) | (pop.z_brown[:, INV] < 0) |
        (pop.z_brown[:, SWI] < 0) |
        (pop.z_green[:, CON] < 0) | (pop.z_green[:, INV] < 0)
    )
    pop.hh_sta[lowpaid] = HH_LOWPAID


# ---------------------------------------------------------------------------
# Budget normalisation
# ---------------------------------------------------------------------------

def _smax(arr: np.ndarray) -> float:
    v = float(arr.max())
    return v if v > 0 else 1.0


def normalize_budgets(pop: Population) -> None:
    """Compute population-wide Z maxima and normalise all z arrays in place."""
    grey_max  = np.array([_smax(pop.z_grey[:,  INV]),
                          _smax(pop.z_grey[:,  CON]),
                          _smax(pop.z_grey[:,  SWI])])
    brown_max = np.array([_smax(pop.z_brown[:, INV]),
                          _smax(pop.z_brown[:, CON]),
                          _smax(pop.z_brown[:, SWI])])
    green_max = np.array([_smax(pop.z_green[:, INV]),
                          _smax(pop.z_green[:, CON])])
    e_max = _smax(pop.h_q)

    pop.z_grey_norm[:]  = pop.z_grey  / grey_max
    pop.z_brown_norm[:] = pop.z_brown / brown_max
    pop.z_green_norm[:] = pop.z_green / green_max
    pop.e_norm_denom[:] = e_max


# ---------------------------------------------------------------------------
# Expected utilities
# ---------------------------------------------------------------------------

def calculate_all_expected_utilities(pop: Population, alpha: float = ALPHA) -> None:
    """Vectorised expected utility for every agent Ã— action."""
    pop.utility_exp[:] = 0.0

    active  = pop.h_q > 0
    e_denom = pop.e_norm_denom
    e_norm  = np.where(e_denom > 0, pop.h_q / e_denom, 1.0)

    g_mask = pop.flag == 0   # grey
    b_mask = pop.flag == 1   # brown
    n_mask = pop.flag == 2   # green

    # z_norm per agent per action based on their current energy source
    z_norm = np.empty((pop.N, 3), dtype=np.float64)
    z_norm[g_mask, INV] = pop.z_grey_norm[g_mask, INV]
    z_norm[g_mask, CON] = pop.z_grey_norm[g_mask, CON]
    z_norm[g_mask, SWI] = pop.z_grey_norm[g_mask, SWI]
    z_norm[b_mask, INV] = pop.z_brown_norm[b_mask, INV]
    z_norm[b_mask, CON] = pop.z_brown_norm[b_mask, CON]
    z_norm[b_mask, SWI] = pop.z_brown_norm[b_mask, SWI]
    z_norm[n_mask, INV] = pop.z_green_norm[n_mask, INV]
    z_norm[n_mask, CON] = pop.z_green_norm[n_mask, CON]
    z_norm[n_mask, SWI] = 0.0

    for a in range(3):
        d_a   = pop.delta[:, a]
        mask  = active & (d_a > 0)

        if a == INV:
            inv_taken = (
                ((b_mask | n_mask) & pop.act11) |
                (g_mask & pop.act12)
            )
            mask = mask & ~(pop.hh_sta == HH_LOWPAID) & ~inv_taken
        elif a == CON:
            con_taken = (
                ((b_mask | n_mask) & pop.act21) |
                (g_mask & pop.act40)
            )
            mask = mask & ~con_taken
        else:  # SWI
            mask = mask & ~n_mask  # green cannot switch
            swi_taken = (
                (b_mask & (pop.act31 | pop.act32)) |
                (g_mask & pop.act32)
            )
            mask = mask & ~swi_taken

        econ = z_norm[:, a] * (1 - alpha) + e_norm * alpha
        beh  = pop.K + pop.M[:, a] + (pop.pbc[:, a] / BEHAVIORAL_SCALE_MAX)
        util = econ * (1 - d_a) + beh * d_a
        pop.utility_exp[:, a] = np.where(mask, util, 0.0)


# ---------------------------------------------------------------------------
# Actual utilities
# ---------------------------------------------------------------------------

def calculate_actual_utility(pop: Population, prices: Dict, alpha: float = ALPHA) -> None:
    """Vectorised normalise-2 + actual utility for all agents."""
    m_b = prices.get('m_p_brown', M_P_BROWN_BASE)
    m_g = prices.get('m_p_grey',  M_P_GREY_BASE)

    q = pop.h_q; inc = pop.h_income
    cp = pop.h_conserv_p; sw = pop.h_switch
    inv = pop.h_invest; isv = pop.h_invest_save

    active = q > 0

    # First pass: raw z values (matching NetLogo)
    pop.z_actual_brown[:] = inc - (q * m_b + isv * m_b + inv + cp + sw)
    pop.z_actual_grey[:]  = inc - (q * m_b + isv * m_g + inv + cp + sw)
    pop.z_actual_green[:] = inc - (q * m_b + isv * m_b + inv + cp + sw)

    # Second pass: population maxima over active agents
    if active.any():
        def _amax(arr):
            v = float(arr[active].max())
            return v if v > 0 else 1.0
        z_bmax = _amax(pop.z_actual_brown)
        z_gmax = _amax(pop.z_actual_grey)
        z_nmax = _amax(pop.z_actual_green)
        pop._e_max_actual = _amax(q)
    else:
        z_bmax = z_gmax = z_nmax = pop._e_max_actual = 1.0

    # Third pass: normalise and select per flag
    pop.z_norm_actual[:] = np.where(
        pop.flag == 0, pop.z_actual_grey  / z_gmax,
        np.where(
            pop.flag == 1, pop.z_actual_brown / z_bmax,
                           pop.z_actual_green / z_nmax))

    pop.utility_actual[:] = 0.0
    e_norm = np.where(pop._e_max_actual > 0, q / pop._e_max_actual, 1.0)
    econ   = pop.z_norm_actual * (1 - alpha) + e_norm * alpha

    for a in range(3):
        d_a  = pop.delta[:, a]
        mask = active & (d_a > 0)
        if a == SWI:
            mask = mask & (pop.flag != 2)
        pop.utility_actual[:, a] = np.where(
            mask,
            econ * (1 - d_a) + (pop.K + pop.M[:, a]) * d_a,
            0.0
        )


# ---------------------------------------------------------------------------
# Satisfaction + regret
# ---------------------------------------------------------------------------

def calculate_satisfaction(pop: Population) -> None:
    """Vectorised satisfaction encoding into pop.satisfaction (int8)."""
    sat = np.zeros(pop.N, dtype=np.int8)

    # Investment (checked first)
    m1    = pop.act1
    keep1 = m1 & (pop.utility_actual[:, INV] >= pop.utility_exp[:, INV])
    sat[keep1]     = 1
    sat[m1 & ~keep1] = 2

    # Conservation (only if no investment action)
    m50   = pop.act50 & ~m1
    keep2 = m50 & (pop.utility_actual[:, CON] >= pop.utility_exp[:, CON])
    sat[keep2]      = 3
    sat[m50 & ~keep2] = 4

    # Switching (only if no investment or conservation)
    m3   = pop.act3 & ~m1 & ~pop.act50
    keep3 = m3 & (pop.utility_actual[:, SWI] >= pop.utility_exp[:, SWI])
    sat[keep3]      = 5
    sat[m3 & ~keep3] = 6

    pop.satisfaction[:] = sat


def apply_regret(pop: Population, learning_type: str) -> None:
    """Vectorised regret: reduce norms for dissatisfied Fast-adaptation agents."""
    if learning_type not in ("Fast adaptation", "Slow adaptation"):
        return
    if learning_type != "Fast adaptation":
        return  # Slow adaptation regret is handled via neighbor spreading (unchanged)

    r  = REGRET_PER_NAB_PBC_RATE
    sr = REGRET_SU_NOR_RATE
    for a, code in [(CON, 4), (SWI, 6)]:
        mask = pop.satisfaction == code
        if not mask.any():
            continue
        can = mask & (pop.per_nab[:, a] >= 1)
        pop.per_nab[can, a] = np.maximum(0, pop.per_nab[can, a] * r)
        can = mask & (pop.pbc[:, a] >= 1)
        pop.pbc[can, a] = np.maximum(0, pop.pbc[can, a] * r)
        can = mask & (pop.su_nor[:, a] >= 1)
        pop.su_nor[can, a] = np.maximum(0, pop.su_nor[can, a] * sr)


# ---------------------------------------------------------------------------
# Action decision
# ---------------------------------------------------------------------------

def _apply_inv(pop, agents, *, perm_flag: str, hh_col: int):
    if len(agents) == 0:
        return
    pop.act1[agents] = True
    getattr(pop, perm_flag)[agents] = True
    pop.hh_actions[agents, hh_col] = 1


def _apply_con(pop, agents, *, perm_flag: str, hh_col: int):
    if len(agents) == 0:
        return
    pop.act2[agents] = True
    pop.act50[agents] = True
    getattr(pop, perm_flag)[agents] = True
    pop.hh_actions[agents, hh_col] = 1


def _apply_swi(pop, agents, *, new_flag: int, perm_flag: str, hh_col: int):
    if len(agents) == 0:
        return
    pop.act3[agents] = True
    getattr(pop, perm_flag)[agents] = True
    pop.flag[agents] = new_flag
    pop.hh_actions[agents, hh_col] = 1


def decide_action(pop: Population) -> None:
    """Vectorised action decision for all agents."""
    active = pop.h_q > 0

    for f in range(3):
        flag_mask = (pop.flag == f) & active
        if not flag_mask.any():
            continue

        idx = np.where(flag_mask)[0]      # agent indices for this flag type
        exp    = pop.utility_exp[idx]     # (M, 3)
        actual = pop.utility_actual[idx]  # (M, 3)

        max_u = exp.max(axis=1, keepdims=True)
        take  = (exp >= max_u) & (exp >= actual)  # (M, 3)

        if f == 0:  # Grey
            i_inv = idx[take[:, INV] & ~pop.act12[idx]]
            i_con = idx[take[:, CON] & ~pop.act40[idx]]
            i_swi = idx[take[:, SWI] & ~pop.act32[idx]]
            _apply_inv(pop, i_inv, perm_flag='act12', hh_col=1)
            _apply_con(pop, i_con, perm_flag='act40', hh_col=3)
            _apply_swi(pop, i_swi, new_flag=1, perm_flag='act32', hh_col=5)

        elif f == 1:  # Brown
            i_inv = idx[take[:, INV] & ~pop.act11[idx]]
            i_con = idx[take[:, CON] & ~pop.act21[idx]]
            i_swi = idx[take[:, SWI] & ~(pop.act31[idx] | pop.act32[idx])]
            _apply_inv(pop, i_inv, perm_flag='act11', hh_col=0)
            _apply_con(pop, i_con, perm_flag='act21', hh_col=2)
            _apply_swi(pop, i_swi, new_flag=2, perm_flag='act31', hh_col=4)

        else:  # Green (no switching)
            i_inv = idx[take[:, INV] & ~pop.act11[idx]]
            i_con = idx[take[:, CON] & ~pop.act21[idx]]
            _apply_inv(pop, i_inv, perm_flag='act11', hh_col=0)
            _apply_con(pop, i_con, perm_flag='act21', hh_col=2)


# ---------------------------------------------------------------------------
# Outcomes: energy savings, financial, emissions
# ---------------------------------------------------------------------------

def calculate_energy_savings(pop: Population, prices: Dict) -> None:
    """Vectorised energy-saving and conservation-money calculations."""
    # Investment: accumulate saved energy
    pop.h_invest_save[pop.act1] += INVESTMENT_PV_ENERGY_OUTPUT

    # Conservation
    con_mask = pop.act2 | pop.act50
    if con_mask.any():
        amt = pop.h_q * CONSERVATION_RATE
        pop.h_conserv[con_mask] = amt[con_mask]

        m_g = prices.get('m_p_grey',  M_P_GREY_BASE)
        m_b = prices.get('m_p_brown', M_P_BROWN_BASE)

        grey_c  = con_mask & (pop.flag == 0)
        brown_c = con_mask & (pop.flag == 1)
        green_c = con_mask & (pop.flag == 2)

        pop.h_conserv_p[grey_c]  = amt[grey_c]  * m_g
        pop.h_conserv_p[brown_c] = amt[brown_c] * m_b
        pop.h_conserv_p[green_c] = 0.0


def calculate_financial_outcomes(pop: Population, prices: Dict) -> None:
    """Vectorised investment cost and switching benefit calculations."""
    inv_paying = pop.act1 & (pop.counter_invest < INVESTMENT_PV_PAYBACK_YEARS)
    if inv_paying.any():
        pop.h_invest[inv_paying]       = INVESTMENT_PV_ANNUAL_COST
        pop.counter_invest[inv_paying] += 1
        pop.h_invest_total[inv_paying] += INVESTMENT_PV_ANNUAL_COST

    # Switching benefit
    m_g = prices.get('m_p_grey',  M_P_GREY_BASE)
    m_b = prices.get('m_p_brown', M_P_BROWN_BASE)
    m_n = prices.get('m_p_green', M_P_GREEN_BASE)

    grey_to_brown = pop.act32  # grey â†’ brown switching
    pop.h_switch[grey_to_brown] = (m_g - m_b) * pop.h_q[grey_to_brown]

    brown_to_green = pop.act31  # brown â†’ green switching
    pop.h_switch[brown_to_green] = (m_b - m_n) * pop.h_q[brown_to_green]


def calculate_emissions_avoided(pop: Population) -> None:
    """Vectorised CO2 emissions-avoided calculation."""
    # Investment
    if pop.act1.any():
        inv_grey  = pop.act1 & (pop.flag == 0)
        inv_brown = pop.act1 & (pop.flag == 1)
        pop.em_avoided[inv_grey,  INV] += INVESTMENT_PV_ENERGY_OUTPUT * CO2_FACTOR_GREY
        pop.em_avoided[inv_brown, INV] += INVESTMENT_PV_ENERGY_OUTPUT * CO2_FACTOR_BROWN

    # Conservation
    con_mask = pop.act2 | pop.act50
    if con_mask.any():
        amt = pop.h_q * CONSERVATION_RATE
        con_grey  = con_mask & (pop.flag == 0)
        con_brown = con_mask & (pop.flag == 1)
        pop.em_avoided[con_grey,  CON] += amt[con_grey]  * CO2_FACTOR_GREY
        pop.em_avoided[con_brown, CON] += amt[con_brown] * CO2_FACTOR_BROWN

    # Switching
    swi_gb = pop.act32  # grey â†’ brown
    swi_bn = pop.act31  # brown â†’ green
    pop.em_avoided[swi_gb, SWI] += pop.h_q[swi_gb] * (CO2_FACTOR_GREY - CO2_FACTOR_BROWN)
    pop.em_avoided[swi_bn, SWI] += pop.h_q[swi_bn] * CO2_FACTOR_BROWN


# ---------------------------------------------------------------------------
# Energy consumption update
# ---------------------------------------------------------------------------

def update_energy_consumption(pop: Population) -> None:
    """Vectorised update_heq: reduce h_q after investment/conservation."""
    # Investment: subtract 1700 kWh
    inv_m = pop.act1 & (pop.h_q > 0)
    if inv_m.any():
        pop.h_q[inv_m] -= INVESTMENT_PV_ENERGY_OUTPUT
        self_prod = inv_m & (pop.h_q <= 0)
        pop.hh_sta[self_prod] = HH_SELFPRODUCER
        pop.h_q[self_prod] = 0.0

    # Conservation
    con_m = pop.act2 | pop.act50
    if con_m.any():
        high = con_m & (pop.h_q > CONSERVATION_MIN_KWH)
        pop.h_q[high] -= pop.h_conserv[high]
        below = high & (pop.h_q < CONSERVATION_MIN_KWH)
        pop.h_q[below] = CONSERVATION_MIN_KWH

        low = con_m & (pop.h_q <= CONSERVATION_MIN_KWH)
        pop.hh_sta[low] = HH_EFFICIENT
        pop.h_q[low] = CONSERVATION_MIN_KWH


# ---------------------------------------------------------------------------
# Memory update (cooldowns + flag resets)
# ---------------------------------------------------------------------------

def update_memory(pop: Population) -> None:
    """Vectorised cooldown increment, expiry, and annual flag reset."""
    # Increment counters for active permanent flags
    pop.act11_year[pop.act11] += 1
    pop.act12_year[pop.act12] += 1
    pop.act21_year[pop.act21] += 1
    pop.act40_year[pop.act40] += 1
    pop.act31_year[pop.act31] += 1
    pop.act32_year[pop.act32] += 1

    # Expire: investment cooldown
    m = pop.act11_year >= INVESTMENT_PV_PAYBACK_YEARS
    pop.hh_actions[m, 0] = 0; pop.act11[m] = False; pop.act11_year[m] = 0

    m = pop.act12_year >= INVESTMENT_PV_PAYBACK_YEARS
    pop.hh_actions[m, 1] = 0; pop.act12[m] = False; pop.act12_year[m] = 0

    # Expire: conservation cooldown
    m = pop.act21_year >= CONSERVATION_COOLDOWN_YEARS
    pop.hh_actions[m, 2] = 0; pop.act21[m] = False; pop.act21_year[m] = 0

    m = pop.act40_year >= CONSERVATION_COOLDOWN_YEARS
    pop.hh_actions[m, 3] = 0; pop.act40[m] = False; pop.act40_year[m] = 0

    # Expire: greyâ†’brown switching cooldown = 2 years
    m = pop.act32_year >= 2
    pop.hh_actions[m, 5] = 0; pop.act32[m] = False; pop.act32_year[m] = 0
    # act31 (brownâ†’green) has no cooldown

    # Reset annual action flags
    pop.act1[:]  = False
    pop.act2[:]  = False
    pop.act3[:]  = False
    pop.act50[:] = False


# ---------------------------------------------------------------------------
# CGE economic update
# ---------------------------------------------------------------------------

def update_economic_data(pop: Population, cge_data: dict, year: int) -> None:
    """Vectorised CGE-based income/consumption/alpha updates."""
    if year <= MODEL_START_YEAR:
        return

    year_idx = year - MODEL_START_YEAR
    income_data = cge_data.get('income',      {}).get('raw', [])
    cons_data   = cge_data.get('consumption', {}).get('raw', [])
    alpha_data  = cge_data.get('alpha',       {}).get('raw', [])

    if not income_data or year_idx >= len(income_data):
        return

    def _val(row, col):
        if isinstance(row, (list, tuple)):
            return float(row[col]) if col < len(row) else 1.0
        return float(row)

    # Income: same multiplier for all groups (col 0)
    income_row = income_data[year_idx]
    pop.h_income *= _val(income_row, 0)

    # Consumption & alpha: per income group
    # Groups 1â†’col0, 2â†’col1, 3â†’col2, 4â†’col3, 5/6/7â†’col4
    for g, col in CGE_INCOME_GROUP_COLS.items():
        mask = pop.income_group == g
        if not mask.any():
            continue
        if cons_data and year_idx < len(cons_data):
            pop.h_q[mask] *= _val(cons_data[year_idx], col)
        if alpha_data and year_idx < len(alpha_data):
            pop.alpha[mask] = _val(alpha_data[year_idx], col)


# ---------------------------------------------------------------------------
# Vectorized social learning
# ---------------------------------------------------------------------------

def apply_social_learning(pop: Population, learning_type: str) -> None:
    """
    Fully vectorized social learning — no AgentProxy, no per-agent loops.

    Fast adaptation: every actor influences all 8 grid neighbours.
    Slow adaptation: each actor samples SLOW_LEARN_NUM_NEIGHBORS neighbours at random.

    Awareness target formula (matching NetLogo Bug-3 fix):
        target = (max(mean(j_nbrs), median(j_nbrs)) + source_val) / 2
    """
    if learning_type == "No learning":
        return

    fast = learning_type == "Fast adaptation"
    pbc_self_rate = LEARN_AWARENESS_RATE_FAST if fast else LEARN_AWARENESS_RATE_SLOW

    # Precompute neighbourhood stats for ALL agents in one vectorised pass.
    # nbr_matrix is (N, 8) with -1 for missing slots.
    safe = np.maximum(pop.nbr_matrix, 0)          # (N, 8) — -1 → 0 (safe dummy index)
    valid = pop.nbr_matrix >= 0                    # (N, 8) boolean mask

    def _stats(arr1d: np.ndarray):
        """Mean and median of each agent's neighbourhood values. Returns (N,), (N,)."""
        vals = np.where(valid, arr1d[safe], np.nan)   # (N, 8)
        return np.nanmean(vals, axis=1), np.nanmedian(vals, axis=1)

    mean_know, med_know = _stats(pop.know)
    mean_cee,  med_cee  = _stats(pop.cee_aw)
    mean_ed,   med_ed   = _stats(pop.ed_aw)

    for a, act_flag in ((INV, pop.act1), (CON, pop.act50), (SWI, pop.act3)):
        actor_ids = np.where(act_flag)[0]
        if len(actor_ids) == 0:
            continue

        # 1. Actor self-boost of own PBC for this action type
        below = pop.pbc[actor_ids, a] < BEHAVIORAL_CAP
        pop.pbc[actor_ids[below], a] = np.minimum(
            pop.pbc[actor_ids[below], a] * pbc_self_rate, BEHAVIORAL_SCALE_MAX
        )

        # 2. Build flat (actor_idx, nbr_idx) pairs
        if fast:
            counts    = np.array([len(pop.nbr_cache[i]) for i in actor_ids])
            actor_arr = np.repeat(actor_ids, counts)
            nbr_arr   = np.concatenate([pop.nbr_cache[i] for i in actor_ids])
        else:
            actor_list, nbr_list = [], []
            for i in actor_ids:
                nbrs = pop.nbr_cache[i]
                if len(nbrs) < SLOW_LEARN_MIN_NEIGHBORS:
                    continue
                k = min(SLOW_LEARN_NUM_NEIGHBORS, len(nbrs))
                chosen = np.random.choice(nbrs, k, replace=False)
                actor_list.append(np.full(k, i, dtype=np.int32))
                nbr_list.append(chosen)
            if not actor_list:
                continue
            actor_arr = np.concatenate(actor_list)
            nbr_arr   = np.concatenate(nbr_list)

        if len(nbr_arr) == 0:
            continue

        # 3. Awareness targets and conditional boosts (always uses FAST rate per NetLogo)
        t_know = (np.maximum(mean_know[nbr_arr], med_know[nbr_arr]) + pop.know[actor_arr])   / 2
        t_cee  = (np.maximum(mean_cee[nbr_arr],  med_cee[nbr_arr])  + pop.cee_aw[actor_arr]) / 2
        t_ed   = (np.maximum(mean_ed[nbr_arr],   med_ed[nbr_arr])   + pop.ed_aw[actor_arr])  / 2

        for arr, target in ((pop.know, t_know), (pop.cee_aw, t_cee), (pop.ed_aw, t_ed)):
            eligible = (arr[nbr_arr] < target) & (arr[nbr_arr] < BEHAVIORAL_CAP)
            if eligible.any():
                np.multiply.at(arr, nbr_arr[eligible], LEARN_AWARENESS_RATE_FAST)
                affected = np.unique(nbr_arr[eligible])
                arr[affected] = np.minimum(arr[affected], BEHAVIORAL_SCALE_MAX)

        # 4. Norm boosts for action a — unconditional if < CAP
        for arr, rate in ((pop.per_nab, LEARN_PER_NAB_PBC_RATE),
                          (pop.pbc,     LEARN_PER_NAB_PBC_RATE),
                          (pop.su_nor,  LEARN_SU_NOR_RATE)):
            col   = arr[:, a]
            below = col[nbr_arr] < BEHAVIORAL_CAP
            if below.any():
                np.multiply.at(col, nbr_arr[below], rate)
                affected = np.unique(nbr_arr[below])
                col[affected] = np.minimum(col[affected], BEHAVIORAL_SCALE_MAX)

    # Recompute h_aware / guilt / K for the whole population after all updates
    update_awareness(pop)


# ---------------------------------------------------------------------------
# Vectorized memory recall (2015 only)
# ---------------------------------------------------------------------------

def _set_memory_action(pop: Population, agents: np.ndarray, action_name: str) -> None:
    """Apply a single memory-recall action to an array of agent indices."""
    name = action_name[len('always_'):] if action_name.startswith('always_') else action_name
    if name == 'act11':
        pop.act11[agents] = True; pop.act1[agents] = True; pop.hh_actions[agents, 0] = 1
    elif name == 'act12':
        pop.act12[agents] = True; pop.act1[agents] = True; pop.hh_actions[agents, 1] = 1
    elif name == 'act31':
        pop.act31[agents] = True; pop.act3[agents] = True; pop.hh_actions[agents, 4] = 1
    elif name == 'act32':
        pop.act32[agents] = True; pop.act3[agents] = True; pop.hh_actions[agents, 5] = 1
    elif name in ('act21', 'act2'):
        pop.act21[agents] = True; pop.act2[agents] = True
        pop.act50[agents] = True; pop.hh_actions[agents, 2] = 1
    elif name == 'act40':
        pop.act40[agents] = True; pop.act2[agents] = True
        pop.act50[agents] = True; pop.hh_actions[agents, 3] = 1


def recall_memory(pop: Population) -> None:
    """Vectorized 2015 memory recall — replaces per-agent loop + AgentProxy."""
    for group, flag_rules in MEMORY_RULES.items():
        group_mask = pop.income_group == group
        for flag_val, rules in flag_rules.items():
            if not isinstance(rules, dict):
                continue
            agents = np.where(group_mask & (pop.flag == flag_val))[0]
            if len(agents) == 0:
                continue
            for action_name, rule_val in rules.items():
                prob = rule_val[0] if isinstance(rule_val, tuple) else float(rule_val)
                triggered = agents[np.random.random(len(agents)) < prob]
                if len(triggered) > 0:
                    _set_memory_action(pop, triggered, action_name)

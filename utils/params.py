"""
Behavioral parameter constants for the BENCH model.

Groups the calibrated rates and thresholds that were previously scattered as
magic numbers in decision_making.py and learning.py.
"""

# --- Saturation ceiling shared by all norm/awareness variables ---
BEHAVIORAL_CAP = 6.5          # above this, no further increase is applied
BEHAVIORAL_SCALE_MAX = 7.0    # hard maximum of the 0-7 scale

# --- Social learning rates (applied by learning.py) ---
# Knowledge / awareness variables (know, cee_aw, ed_aw)
LEARN_AWARENESS_RATE_FAST = 1.05   # Fast adaptation
LEARN_AWARENESS_RATE_SLOW = 1.02   # Slow adaptation

# Action-specific norm variables (per_nab, pbc, su_nor)
LEARN_PER_NAB_PBC_RATE = 1.05     # both fast and slow
LEARN_SU_NOR_RATE      = 1.07     # both fast and slow

# Minimum neighbours required for Slow-adaptation learning
SLOW_LEARN_MIN_NEIGHBORS = 2

# --- Regret rates (applied by decision_making.py) ---
REGRET_PER_NAB_PBC_RATE = 0.95
REGRET_SU_NOR_RATE      = 0.97

# --- Carbon-price awareness rates (applied by decision_making.py / cpinfo) ---
# Keyed by the policy string used in BENCH_MODEL.
CPINFO_RATES = {
    "Carbon price pressure-10": {
        "awareness": 1.01,      # know / cee_aw / ed_aw multiplier
        "su_nor":    None,      # no su_nor effect at this level
        "pbc":       None,
    },
    "Carbon price pressure-25": {
        "awareness": 1.02,
        "su_nor":    1.04,
        "pbc":       1.03,
    },
    "Carbon price pressure-50": {
        "awareness": 1.04,
        "su_nor":    1.06,
        "pbc":       1.04,
    },
    "Carbon price pressure-100": {
        "awareness": 1.06,
        "su_nor":    1.10,
        "pbc":       1.05,
    },
}

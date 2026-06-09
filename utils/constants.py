"""
Global constants and configuration for BENCH model
"""

# Model Configuration
MODEL_START_YEAR = 2015
MODEL_END_YEAR = 2030
TIMESTEP_YEARS = 1

# Income Groups (7 groups)
INCOME_GROUPS = list(range(1, 8))

# Dwelling Energy Labels (6 labels: A-F equivalent to 1-6)
DWELLING_LABELS = list(range(1, 7))
DWELLING_LABEL_NAMES = ['A', 'B', 'C', 'D', 'E', 'F']

# Energy Source Flags
FLAG_GRAY = 0           # Gray electricity (fossil-based)
FLAG_BROWN = 1          # Brown electricity (mixed/less-clean)
FLAG_GREEN = 2          # Green electricity (renewable)
FLAG_NAMES = {0: 'gray', 1: 'brown', 2: 'green'}

# Guilt Levels
GUILT_LOW = 'L'
GUILT_HIGH = 'H'

# Action Types
ACTION_INVESTMENT = 0      # Solar PV installation
ACTION_CONSERVATION = 1    # Energy efficiency/conservation
ACTION_SWITCHING = 2       # Switch to renewable

ACTION_NAMES = {
    0: 'Investment (PV)',
    1: 'Conservation',
    2: 'Switching'
}

# Behavioral Scales (0-7)
BEHAVIORAL_SCALE_MIN = 0
BEHAVIORAL_SCALE_MAX = 7
BEHAVIORAL_SCALE_GUILT_THRESHOLD = 5.21  # Threshold for guilt level

# PBC Scale
PBC_MIN = 0
PBC_MAX = 7
PBC_THRESHOLD = 4  # Threshold for high consideration

# Delta Parameters (adjustment factors for actions)
DELTA_LEVELS = {
    0.2: (0, 2),        # PBC < 2
    0.3: (2, 3),        # 2 <= PBC < 3
    0.4: (3, 4),        # 3 <= PBC < 4
    0.5: (4, 5),        # 4 <= PBC < 5
    0.6: (5, 6),        # 5 <= PBC < 6
    0.7: (6, 7),        # 6 <= PBC <= 7
}

# Investment Parameters
INVESTMENT_PV_PAYBACK_YEARS = 10
INVESTMENT_PV_ANNUAL_COST = 487.59
INVESTMENT_PV_ENERGY_OUTPUT = 1700  # kWh/year

# Conservation Parameters
CONSERVATION_RATE = 0.5  # 50% energy reduction achievable
# Note: Original NetLogo mentions EnergySTAR ~12%, but uses 5% in some places

# Market Parameters
CASE_STUDIES = ["Netherlands-Overijssel", "Spain-Navarre"]
SCENARIOS = ["Ref_SSP2"]
POLICIES = [
    "Ref",
    "Carbon price pressure-10",
    "Carbon price pressure-25",
    "Carbon price pressure-50",
    "Carbon price pressure-100",
    "Carbon price pressure-2020"
]

# Emission factors by electricity source
EMISSIONS_FACTOR_GRAY = 0.5   # kg CO2 per kWh for gray electricity
EMISSIONS_FACTOR_BROWN = 0.3  # kg CO2 per kWh for brown electricity
EMISSIONS_FACTOR_GREEN = 0.0  # kg CO2 per kWh for green electricity

LEARNING_TYPES = [
    "No learning",
    "Fast adaptation",
    "Slow adaptation",
    "Observation",
    "Promote switching"
]
DEFAULT_LEARNING_TYPE = "No learning"

# File Paths (relative to project root)
DATA_DIR = "netlogo/data"
HOUSEHOLD_FILE = "netlogo/data/household.csv"
CGE_NL_CON_FILE = "netlogo/data/cge-nl-ssp2-con.csv"
CGE_NL_H_FILE = "netlogo/data/cge-nl-ssp2-h.csv"
CGE_NL_ALPHA_FILE = "netlogo/data/cge-nl-ssp2-alpha.csv"
PRIMES_NL_PRICES_FILE = "netlogo/data/primes-nl-ref-prices.csv"
PRIMES_NL_CON_FILE = "netlogo/data/primes-nl-ref-con.csv"
PRIMES_NL_NONCON_FILE = "netlogo/data/primes-nl-ref-noncon.csv"
PRIMES_SPN_PRICES_FILE = "netlogo/data/primes-spn-ref-prices.csv"

# Output Configuration
OUTPUT_DIR = "output"
OUTPUT_HOUSEHOLD_FILE = "annual_household_data.csv"
OUTPUT_AGGREGATES_FILE = "annual_aggregates.csv"
OUTPUT_ACTIONS_FILE = "actions_breakdown.csv"

# Normalization Parameters (utilities)
UTILITY_NORMALIZATION_FACTOR = 0.5  # Alpha parameter for utility calculation

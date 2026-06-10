"""
Global constants and configuration for BENCH model
"""

# Model Configuration
MODEL_START_YEAR = 2015
MODEL_END_YEAR = 2030
TIMESTEP_YEARS = 1
NUMBER_SEED_RUNS = 10

VERBOSE = False  # Set to True for detailed logging during model execution

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
CARBON_POLICY_TARGETS = {
    "Ref": 0.0,
    "Carbon price pressure-10": 10.0,
    "Carbon price pressure-25": 25.0,
    "Carbon price pressure-50": 50.0,
    "Carbon price pressure-100": 100.0,
}


# Emission factors by electricity source
EMISSIONS_FACTOR_GRAY = 0.5   # kg CO2 per kWh for gray electricity
EMISSIONS_FACTOR_BROWN = 0.3  # kg CO2 per kWh for brown electricity
EMISSIONS_FACTOR_GREEN = 0.0  # kg CO2 per kWh for green electricity
KG_TO_TONS = 0.001            # Conversion factor (1 kg = 0.001 metric tons)

LEARNING_TYPES = [
    "No learning",
    "Fast adaptation",
    "Slow adaptation",
]
DEFAULT_LEARNING_TYPE = "No learning"

# File Paths (relative to project root)
DATA_DIR = "data"
HOUSEHOLD_FILE = "data/household.csv"
CGE_NL_CON_FILE = "data/cge-nl-ssp2-con.csv"
CGE_NL_H_FILE = "data/cge-nl-ssp2-h.csv"
CGE_NL_ALPHA_FILE = "data/cge-nl-ssp2-alpha.csv"
PRIMES_NL_PRICES_FILE = "data/primes-nl-ref-prices.csv"
PRIMES_NL_CON_FILE = "data/primes-nl-ref-con.csv"
PRIMES_NL_NONCON_FILE = "data/primes-nl-ref-noncon.csv"
PRIMES_SPN_PRICES_FILE = "data/primes-spn-ref-prices.csv"

# Output Configuration
OUTPUT_DIR = "output"
OUTPUT_HOUSEHOLD_FILE = "annual_household_data.csv"
OUTPUT_AGGREGATES_FILE = "annual_aggregates.csv"
OUTPUT_ACTIONS_FILE = "actions_breakdown.csv"

# Normalization Parameters (utilities)
UTILITY_NORMALIZATION_FACTOR = 0.5  # Alpha parameter for utility calculation

MEMORY_RULES = {
    1: {  # Income group 1
        1: {  # energy_flag = 1
            'act11': (0.5714, True),
            'act31': (0.1428, True),
            'act32': (0.8572, True),  # Derived from remaining probability (100 - 14.28)
            'act21': (0.2143, True),
        },
        0: {  # energy_flag = 0
            'act12': (0.5814, True),
            'act40': (0.0930, True),
        }
    },
    2: {  # Income group 2
        1: {
            'act11': (0.7829, True),
            'act31': (0.1513, True),
            'act32': (0.8487, True),
            'act21': (0.1710, True),
        },
        0: {
            'act12': (0.6860, True),
            'act40': (0.0870, True),
        }
    },
    3: {  # Income group 3
        1: {
            'act11': (0.8240, True),
            'act31': (0.1410, True),
            'act32': (0.8590, True),
            'act21': (0.1700, True),
            'always_act2': (1.0000, True), # Note: Group 3, flag 1 always sets item 2 to 1 unconditionally
        },
        0: {
            'act12': (0.8048, True),
            'act40': (0.1155, True),
        }
    },
    4: {  # Income group 4
        1: {
            'act11': (0.8310, True),
            'act31': (0.1846, True),
            'act32': (0.8154, True),
            'act21': (0.1846, True),
        },
        0: {
            'act12': (0.4286, True),
            'act40': (0.1600, True),
        }
    },
    5: {  # Income group 5
        1: {
            'act11': (0.8847, True),
            'act31': (0.2310, True),
            'act32': (0.7690, True),
            'act21': (0.2310, True),
        },
        0: {
            'act12': (0.8789, True),
            'act40': (0.1818, True),
        }
    },
    6: {  # Income group 6
        1: {
            'always_act11': (1.0000, True), # Note: Group 6, flag 1 always executes action 11
            'act31': (0.5000, True),
            'act32': (0.5000, True),
            'act21': (0.50)
        },
        0: {
            'always_act12': (1.0000, True), # Note: Group 6, flag 0 always executes action 12
            'act40': (0.2500, True),
        }
    },
    7: {  # Income group 7
        1: {
            'act11': (0.6000, True),
            'act31': (0.2000, True),
            'act32': (0.8000, True),
            'act21': (0.2000, True),
        },
        0: {
            'act12': (0.8333, True),
            'act40': (0.0830, True),
        }
    }
}


M_P_GREY_BASE = 0.15
M_P_BROWN_BASE = 0.15
M_P_GREEN_BASE = 0.15


"""
Data loading and management for BENCH Model
Reads all CSV files and provides structured access to data
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from utils.constants import (
    HOUSEHOLD_FILE, CGE_NL_H_FILE, CGE_NL_CON_FILE, CGE_NL_ALPHA_FILE,
    PRIMES_NL_PRICES_FILE, PRIMES_NL_CON_FILE, PRIMES_NL_NONCON_FILE,
    DATA_DIR, MODEL_START_YEAR, MODEL_END_YEAR
)


class DataLoader:
    """
    Loads and manages all data files required for BENCH model simulation.
    Provides interfaces for accessing household data, prices, and parameters.
    """
    
    def __init__(self, base_path: str = "."):
        """
        Initialize data loader.
        
        Args:
            base_path: Base path to project directory (for locating data files)
        """
        self.base_path = base_path
        self.data_dir = os.path.join(base_path, DATA_DIR)
        
        # Data storage
        self.households_df = None
        self.prices_df = None
        self.consumption_params = {}
        self.cge_params = {}
        self.loaded_files = {}
    
    def load_all_data(self) -> bool:
        """
        Load all required data files.
        
        Returns:
            True if all files loaded successfully, False otherwise
        """
        try:
            print("Loading BENCH model data...")
            
            # Load household data
            self.load_households()
            print(f"✓ Loaded {len(self.households_df)} households")
            
            # Load price scenarios
            self.load_prices()
            print(f"✓ Loaded price scenarios")
            
            # Load consumption parameters
            self.load_consumption_data()
            print(f"✓ Loaded consumption data")
            
            # Load CGE parameters
            self.load_cge_data()
            print(f"✓ Loaded CGE parameters")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def load_households(self) -> None:
        """Load household data from CSV."""
        file_path = os.path.join(self.base_path, HOUSEHOLD_FILE)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Household file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Handle column name spacing
        df.columns = df.columns.str.strip()
        
        # Convert year to filter only 2015 baseline
        if 'year' in df.columns:
            df = df[df['year'] == MODEL_START_YEAR].reset_index(drop=True)
        
        self.households_df = df
        self.loaded_files['households'] = file_path
    
    def load_prices(self) -> None:
        """Load price scenarios from CSV."""
        file_path = os.path.join(self.base_path, PRIMES_NL_PRICES_FILE)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prices file not found: {file_path}")
        
        df = pd.read_csv(file_path, header=None)
        
        # Convert to numeric, handle empty values
        df = df.apply(pd.to_numeric, errors='coerce')
        
        self.prices_df = df
        self.loaded_files['prices'] = file_path
    
    def load_consumption_data(self) -> None:
        """Load consumption reference data."""
        file_path = os.path.join(self.base_path, PRIMES_NL_CON_FILE)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            self.consumption_params['reference'] = df
            self.loaded_files['consumption'] = file_path
    
    def load_cge_data(self) -> None:
        """Load CGE economic parameters."""
        try:
            # Load household economic data
            h_file = os.path.join(self.base_path, CGE_NL_H_FILE)
            if os.path.exists(h_file):
                self.cge_params['households'] = pd.read_csv(h_file)
                self.loaded_files['cge_h'] = h_file
            
            # Load consumption economic data
            c_file = os.path.join(self.base_path, CGE_NL_CON_FILE)
            if os.path.exists(c_file):
                self.cge_params['consumption'] = pd.read_csv(c_file)
                self.loaded_files['cge_con'] = c_file
            
            # Load alpha parameters
            a_file = os.path.join(self.base_path, CGE_NL_ALPHA_FILE)
            if os.path.exists(a_file):
                self.cge_params['alpha'] = pd.read_csv(a_file)
                self.loaded_files['cge_alpha'] = a_file
                
        except Exception as e:
            print(f"Warning: Could not load all CGE data: {e}")
    
    def get_household_count(self) -> int:
        """Return total number of households."""
        return len(self.households_df) if self.households_df is not None else 0
    
    def get_household_data(self, h_id: int) -> Optional[pd.Series]:
        """
        Get data for a specific household.
        
        Args:
            h_id: Household ID
            
        Returns:
            Pandas Series with household data, or None if not found
        """
        if self.households_df is None:
            return None
        
        # Handle different possible ID column names
        for id_col in ['id', 'h_id', 'ID']:
            if id_col in self.households_df.columns:
                matches = self.households_df[self.households_df[id_col] == h_id]
                if len(matches) > 0:
                    return matches.iloc[0]
        
        return None
    
    def get_all_households_data(self) -> pd.DataFrame:
        """Return all household data as DataFrame."""
        return self.households_df.copy() if self.households_df is not None else pd.DataFrame()
    
    def get_price_at_year(self, policy: str, year: int, column: int = 0) -> float:
        """
        Get electricity price for a specific year and policy.
        
        Args:
            policy: Policy scenario name
            year: Target year (2015-2030)
            column: Column index in price matrix (0=reference, 1+=scenarios)
            
        Returns:
            Price value
        """
        if self.prices_df is None or year < MODEL_START_YEAR or year > MODEL_END_YEAR:
            return 0.15  # Default fallback price
        
        # Map year to row index (2015=row 0, 2016=row 1, etc.)
        row_idx = year - MODEL_START_YEAR
        
        if row_idx >= len(self.prices_df):
            return self.prices_df.iloc[-1, column]
        
        value = self.prices_df.iloc[row_idx, column]
        
        # Handle NaN values
        return float(value) if pd.notna(value) else 0.15
    
    def get_price_scenario_index(self, policy: str) -> int:
        """
        Map policy scenario name to column index in price matrix.
        
        Args:
            policy: Policy name
            
        Returns:
            Column index (0=reference, 1+=scenarios)
        """
        policy_mapping = {
            'Ref': 0,
            'Carbon price pressure-10': 2,
            'Carbon price pressure-25': 4,
            'Carbon price pressure-50': 6,
            'Carbon price pressure-100': 8,
            'Carbon price pressure-2020': 10,
        }
        return policy_mapping.get(policy, 0)
    
    def get_household_income_groups(self) -> Dict[int, List[int]]:
        """
        Get list of household IDs grouped by income group.
        
        Returns:
            Dictionary mapping income_group -> [list of h_ids]
        """
        if self.households_df is None:
            return {}
        
        groups = {}
        
        # Identify income group column (try different names)
        income_col = None
        for col in ['group id', 'group_id', 'income_group', 'income group']:
            if col in self.households_df.columns:
                income_col = col
                break
        
        if income_col is None:
            # If no explicit column, create income-based groups
            return self._create_income_groups()
        
        # Get ID column
        id_col = None
        for col in ['id', 'h_id', 'ID']:
            if col in self.households_df.columns:
                id_col = col
                break
        
        if id_col is None:
            return {}
        
        # Group by income
        for group_num in range(1, 8):
            mask = self.households_df[income_col] == group_num
            groups[group_num] = self.households_df[mask][id_col].tolist()
        
        return groups
    
    def _create_income_groups(self) -> Dict[int, List[int]]:
        """
        Create income groups by dividing households by income percentile.
        Creates 7 groups from lowest to highest income.
        
        Returns:
            Dictionary mapping income_group -> [list of h_ids]
        """
        if self.households_df is None:
            return {}
        
        # Find income column
        income_col = None
        for col in ['income', 'h_income', 'Income']:
            if col in self.households_df.columns:
                income_col = col
                break
        
        if income_col is None:
            return {}
        
        id_col = None
        for col in ['id', 'h_id', 'ID']:
            if col in self.households_df.columns:
                id_col = col
                break
        
        if id_col is None:
            return {}
        
        # Sort by income and create 7 groups
        df = self.households_df.sort_values(income_col)
        group_size = len(df) // 7
        groups = {}
        
        for group_num in range(1, 8):
            if group_num == 7:
                # Last group gets all remaining
                mask = df.index >= (group_size * 6)
            else:
                start = group_size * (group_num - 1)
                end = group_size * group_num
                mask = (df.index >= start) & (df.index < end)
            
            groups[group_num] = df[mask][id_col].tolist()
        
        return groups
    
    def get_dwelling_statistics(self) -> Dict[int, int]:
        """
        Get count of households by dwelling energy label (1-6).
        
        Returns:
            Dictionary mapping label -> count
        """
        if self.households_df is None:
            return {}
        
        # Find dwelling label column
        label_col = None
        for col in ['dw_el', 'dwelling_label', 'dwelling_el']:
            if col in self.households_df.columns:
                label_col = col
                break
        
        if label_col is None:
            return {}
        
        stats = {}
        for label in range(1, 7):
            count = len(self.households_df[self.households_df[label_col] == label])
            stats[label] = count
        
        return stats
    
    def get_energy_source_distribution(self) -> Dict[int, int]:
        """
        Get count of households by energy source (flag: 0, 1, 2).
        
        Returns:
            Dictionary mapping flag -> count
        """
        if self.households_df is None:
            return {}
        
        # Find energy source column
        flag_col = None
        for col in ['lce user?', 'lce user', 'flag', 'energy_source']:
            if col in self.households_df.columns:
                flag_col = col
                break
        
        if flag_col is None:
            return {}
        
        stats = {}
        for flag in [0, 1, 2]:
            count = len(self.households_df[self.households_df[flag_col] == flag])
            stats[flag] = count
        
        return stats
    
    def print_summary(self) -> None:
        """Print summary of loaded data."""
        print("\n=== DATA LOADING SUMMARY ===")
        print(f"Households: {self.get_household_count()}")
        
        if self.households_df is not None:
            print(f"  Columns: {list(self.households_df.columns)[:10]}...")
            print(f"  Data shape: {self.households_df.shape}")
        
        if self.prices_df is not None:
            print(f"Prices: {self.prices_df.shape[0]} years, {self.prices_df.shape[1]} scenarios")
        
        print(f"Files loaded: {len(self.loaded_files)}")
        print("==========================\n")

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
            #print("Loading BENCH model data...")
            
            # Load household data
            self.load_households()
            #print(f"✓ Loaded {len(self.households_df)} households")
            
            # Load price scenarios
            self.load_prices()
            #print(f"✓ Loaded price scenarios")
            
            # Load consumption parameters
            self.load_consumption_data()
            #print(f"✓ Loaded consumption data")
            
            # Load CGE parameters
            self.load_cge_data()
            #print(f"✓ Loaded CGE parameters")
            
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

    def load_price_trajectories(self) -> Dict:
        """Load price trajectories from PRIMES data."""
        file_path = os.path.join(self.base_path, PRIMES_NL_PRICES_FILE)
        
        if not os.path.exists(file_path):
            print(f"Warning: Price file not found: {file_path}")
            return {}
        
        df = pd.read_csv(file_path, header=None)
        
        # Structure: rows = years, columns = different policy scenarios
        # Based on NetLogo indices:
        # Column 0: Ref prices (grey = brown = green)
        # Column 1-2: Carbon price 10 (brown, grey)
        # Column 3-4: Carbon price 25 (brown, grey)
        # Column 5-6: Carbon price 50 (brown, grey)
        # Column 7-8: Carbon price 100 (brown, grey)
        # Column 9-10: Carbon price 2020 (brown, grey)
        
        price_data = {
            'Ref': {'grey': [], 'brown': [], 'green': []},
            'Carbon price pressure-10': {'grey': [], 'brown': [], 'green': []},
            'Carbon price pressure-25': {'grey': [], 'brown': [], 'green': []},
            'Carbon price pressure-50': {'grey': [], 'brown': [], 'green': []},
            'Carbon price pressure-100': {'grey': [], 'brown': [], 'green': []},
            'Carbon price pressure-2020': {'grey': [], 'brown': [], 'green': []},
        }
        
        for year_idx in range(len(df)):
            # Ref scenario (all equal)
            price_data['Ref']['grey'].append(df.iloc[year_idx, 0])
            price_data['Ref']['brown'].append(df.iloc[year_idx, 0])
            price_data['Ref']['green'].append(df.iloc[year_idx, 0])
            
            # Carbon price 10
            price_data['Carbon price pressure-10']['brown'].append(df.iloc[year_idx, 1])
            price_data['Carbon price pressure-10']['grey'].append(df.iloc[year_idx, 2])
            price_data['Carbon price pressure-10']['green'].append(0.15)  # Green unchanged
            
            # Carbon price 25
            price_data['Carbon price pressure-25']['brown'].append(df.iloc[year_idx, 3])
            price_data['Carbon price pressure-25']['grey'].append(df.iloc[year_idx, 4])
            price_data['Carbon price pressure-25']['green'].append(0.15)
            
            # Carbon price 50
            price_data['Carbon price pressure-50']['brown'].append(df.iloc[year_idx, 5])
            price_data['Carbon price pressure-50']['grey'].append(df.iloc[year_idx, 6])
            price_data['Carbon price pressure-50']['green'].append(0.15)
            
            # Carbon price 100
            price_data['Carbon price pressure-100']['brown'].append(df.iloc[year_idx, 7])
            price_data['Carbon price pressure-100']['grey'].append(df.iloc[year_idx, 8])
            price_data['Carbon price pressure-100']['green'].append(0.15)
            
            # Carbon price 2020
            price_data['Carbon price pressure-2020']['brown'].append(df.iloc[year_idx, 9])
            price_data['Carbon price pressure-2020']['grey'].append(df.iloc[year_idx, 10])
            price_data['Carbon price pressure-2020']['green'].append(0.15)
        
        return price_data
    
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
    
    
    def get_all_households_data(self) -> pd.DataFrame:
        """Return all household data as DataFrame."""
        return self.households_df.copy() if self.households_df is not None else pd.DataFrame()
    

"""
Data loading and management for BENCH Model
Reads all CSV files and provides structured access to data
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Tuple
from model.parameters import (
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
            #print(f"âœ“ Loaded {len(self.households_df)} households")
            
            # Load price scenarios
            self.load_prices()
            #print(f"âœ“ Loaded price scenarios")
            
            # Load consumption parameters
            self.load_consumption_data()
            #print(f"âœ“ Loaded consumption data")
            
            # Load CGE parameters
            self.load_cge_data()
            #print(f"âœ“ Loaded CGE parameters")
            
            return True
            
        except Exception as e:
            print(f"âœ— Error loading data: {e}")
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

    def load_cge_trajectories(self) -> Dict:
        """
        Load CGE economic trajectories for income, consumption, and alpha.
        Returns dictionary with yearly multipliers by income group.
        
        The CGE files contain growth multipliers. For example:
        - A value of 1.02 means 2% growth
        - A value of 0.98 means 2% decrease
        
        File structure (from NetLogo):
        - Each row represents a year (2015, 2016, 2017, ...)
        - Columns represent different income group trajectories
        """
        cge_data = {
            'income': {},
            'consumption': {},
            'alpha': {}
        }
        
        # Load income growth data
        income_path = os.path.join(self.base_path, CGE_NL_H_FILE)
        if os.path.exists(income_path):
            try:
                # Try reading with header first
                df = pd.read_csv(income_path)
                # Check if it's a single column (growth factors by year)
                if len(df.columns) == 1:
                    cge_data['income']['raw'] = df.iloc[:, 0].tolist()
                else:
                    cge_data['income']['raw'] = df.values.tolist()
                #print(f"âœ“ Loaded CGE income data: {len(cge_data['income']['raw'])} years")
            except Exception as e:
                # Fallback: read without header
                df = pd.read_csv(income_path, header=None)
                cge_data['income']['raw'] = df.values.tolist()
                #print(f"âœ“ Loaded CGE income data (no header): {len(cge_data['income']['raw'])} years")
        else:
            print(f"Warning: CGE income file not found: {income_path}")
            # Provide default growth factors (1.0 = no growth) as fallback
            cge_data['income']['raw'] = [[1.0] for _ in range(16)]  # 2015-2030
        
        # Load consumption growth data
        cons_path = os.path.join(self.base_path, CGE_NL_CON_FILE)
        if os.path.exists(cons_path):
            try:
                df = pd.read_csv(cons_path)
                if len(df.columns) == 1:
                    cge_data['consumption']['raw'] = df.iloc[:, 0].tolist()
                else:
                    cge_data['consumption']['raw'] = df.values.tolist()
                #print(f"âœ“ Loaded CGE consumption data: {len(cge_data['consumption']['raw'])} years")
            except Exception:
                df = pd.read_csv(cons_path, header=None)
                cge_data['consumption']['raw'] = df.values.tolist()
                #print(f"âœ“ Loaded CGE consumption data (no header): {len(cge_data['consumption']['raw'])} years")
        else:
            print(f"Warning: CGE consumption file not found: {cons_path}")
            cge_data['consumption']['raw'] = [[1.0] for _ in range(16)]
        
        # Load alpha data
        alpha_path = os.path.join(self.base_path, CGE_NL_ALPHA_FILE)
        if os.path.exists(alpha_path):
            try:
                df = pd.read_csv(alpha_path)
                if len(df.columns) == 1:
                    cge_data['alpha']['raw'] = df.iloc[:, 0].tolist()
                else:
                    cge_data['alpha']['raw'] = df.values.tolist()
                #print(f"âœ“ Loaded CGE alpha data: {len(cge_data['alpha']['raw'])} years")
            except Exception:
                df = pd.read_csv(alpha_path, header=None)
                cge_data['alpha']['raw'] = df.values.tolist()
                #print(f"âœ“ Loaded CGE alpha data (no header): {len(cge_data['alpha']['raw'])} years")
        else:
            print(f"Warning: CGE alpha file not found: {alpha_path}")
            # Default alpha values by income group (from NetLogo survey data)
            default_alphas = {
                1: 0.0199,
                2: 0.0191,
                3: 0.0175,
                4: 0.0159,
                5: 0.0133,
                6: 0.0133,
                7: 0.0133,
            }
            # Create list of alpha values for each year (constant over time)
            alpha_list = []
            for year in range(16):  # 2015-2030
                year_data = [default_alphas.get(g, 0.015) for g in range(1, 8)]
                alpha_list.append(year_data)
            cge_data['alpha']['raw'] = alpha_list
            print("âœ“ Using default alpha values from NetLogo survey data")
        
        return cge_data

    def get_all_households_data(self) -> pd.DataFrame:
        """Return all household data as DataFrame."""
        return self.households_df.copy() if self.households_df is not None else pd.DataFrame()

    def create_population(self, case_study: str):
        """
        Build a Population directly from the loaded DataFrame without per-row Python loops.
        Eliminates the pandas iterrows()/to_dict() hotspot from _create_agents.

        Returns:
            (Population, id_to_index_map) where id_to_index_map maps h_id â†’ array index.
        """
        import numpy as np
        from model.population import Population, N_YEARS
        from model.parameters import MODEL_START_YEAR, BEHAVIORAL_SCALE_MAX

        raw = self.households_df
        if raw is None or raw.empty:
            raise ValueError("No household data loaded. Call load_all_data() first.")

        # --- Extract the 2015 baseline rows ---
        if 'year' in raw.columns:
            df2015 = raw[raw['year'] == MODEL_START_YEAR].drop_duplicates(subset=['id'], keep='first')
        else:
            df2015 = raw.drop_duplicates(subset=['id'], keep='first') if 'id' in raw.columns else raw

        n = len(df2015)
        pop = Population(n)

        # Helper to extract a column safely
        def _col(name, dtype=np.float64, default=0.0):
            if name in df2015.columns:
                vals = df2015[name].to_numpy(dtype=float)
            else:
                vals = np.full(n, default, dtype=float)
            return vals.astype(dtype)

        # --- Static attributes ---
        pop.h_id[:]           = _col('id',       np.int32)
        pop.income_group[:]   = _col('group id',  np.int8)
        pop.h_income[:]       = _col('income',    np.float64)
        pop.h_q[:]            = _col('consumption', np.float64)
        pop.dw_el[:]          = _col('dw_el',     np.int8, default=3)
        pop.owner[:]          = _col('Owner',     bool).astype(bool)
        pop.ep[:]             = _col('ep',        np.float32)

        # Energy flag: values > 1 â†’ green (2)
        flag_raw = _col('lce user?', np.int8)
        flag_raw[flag_raw > 1] = 2
        pop.flag[:] = flag_raw

        # --- Behavioral attributes ---
        # knowledge / awareness
        if 'knowledge' in df2015.columns:
            pop.know[:] = _col('knowledge')
        elif 'know' in df2015.columns:
            pop.know[:] = _col('know')

        pop.cee_aw[:] = _col('cee_aw')
        pop.ed_aw[:]  = _col('ed_aw')

        # Personal norms (INV=0, CON=1, SWI=2)
        pop.per_nab[:, 0] = _col('personal1')
        pop.per_nab[:, 1] = _col('personal2')
        pop.per_nab[:, 2] = _col('personal3')

        # Social norms
        pop.su_nor[:, 0] = _col('social1')
        pop.su_nor[:, 1] = _col('social2')
        pop.su_nor[:, 2] = _col('social3')

        # PBC
        pop.pbc[:, 0] = _col('pbc1')
        pop.pbc[:, 1] = _col('pbc2')
        pop.pbc[:, 2] = _col('pbc3')

        # --- Income trajectory: one column-read per year from raw data ---
        if 'year' in raw.columns and 'id' in raw.columns and 'income' in raw.columns:
            # Build id â†’ row index map
            id_to_idx = {int(hid): i for i, hid in enumerate(df2015['id'].to_numpy(dtype=np.int32))}
            ids_arr = df2015['id'].to_numpy(dtype=np.int32)

            for yi in range(N_YEARS):
                year = MODEL_START_YEAR + yi
                yr_df = raw[raw['year'] == year][['id', 'income']]
                if yr_df.empty:
                    # Copy previous year or use baseline
                    if yi == 0:
                        pop.income_traj[:, yi] = pop.h_income
                    else:
                        pop.income_traj[:, yi] = pop.income_traj[:, yi - 1]
                    continue
                yr_map = dict(zip(yr_df['id'].astype(int), yr_df['income'].astype(float)))
                for i, hid in enumerate(ids_arr):
                    pop.income_traj[i, yi] = yr_map.get(int(hid), pop.h_income[i])
        else:
            # No multi-year data: fill all years with baseline income
            for yi in range(N_YEARS):
                pop.income_traj[:, yi] = pop.h_income

        # --- Initial awareness + motivation ---
        pop.h_aware[:] = (pop.know + pop.cee_aw + pop.ed_aw) / 3.0
        pop.guilt[:]   = pop.h_aware >= 5.21
        pop.K[:]       = np.where(pop.guilt, pop.h_aware / BEHAVIORAL_SCALE_MAX, 0.0)

        # Initial motivation
        pop.h_motiv[:] = (pop.per_nab + pop.su_nor) / 2.0
        pop.M[:]       = pop.h_motiv / BEHAVIORAL_SCALE_MAX

        # update_motivation responsibility (case-study thresholds)
        from model.vectorized import update_motivation
        update_motivation(pop, case_study)

        id_to_idx = {int(hid): i for i, hid in enumerate(pop.h_id)}
        return pop, id_to_idx
    

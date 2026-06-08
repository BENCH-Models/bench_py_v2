"""
Quick test to verify project structure and imports
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...\n")
    
    try:
        from utils.constants import MODEL_START_YEAR, MODEL_END_YEAR
        print("✓ utils.constants")
    except Exception as e:
        print(f"✗ utils.constants: {e}")
        return False
    
    try:
        from agents.household import Household
        print("✓ agents.household")
    except Exception as e:
        print(f"✗ agents.household: {e}")
        return False
    
    try:
        from data_loader.loader import DataLoader
        print("✓ data_loader.loader")
    except Exception as e:
        print(f"✗ data_loader.loader: {e}")
        return False
    
    try:
        from behavioral.utility import UtilityCalculator
        print("✓ behavioral.utility")
    except Exception as e:
        print(f"✗ behavioral.utility: {e}")
        return False
    
    try:
        from behavioral.decision_making import DecisionMaker
        print("✓ behavioral.decision_making")
    except Exception as e:
        print(f"✗ behavioral.decision_making: {e}")
        return False
    
    try:
        from behavioral.learning import LearningMechanism
        print("✓ behavioral.learning")
    except Exception as e:
        print(f"✗ behavioral.learning: {e}")
        return False
    
    try:
        from utils.statistics import StatisticsAggregator
        print("✓ utils.statistics")
    except Exception as e:
        print(f"✗ utils.statistics: {e}")
        return False
    
    try:
        from utils.output import ResultsExporter
        print("✓ utils.output")
    except Exception as e:
        print(f"✗ utils.output: {e}")
        return False
    
    try:
        from model.bench_model import BENCHModel
        print("✓ model.bench_model")
    except Exception as e:
        print(f"✗ model.bench_model: {e}")
        return False
    
    return True


def test_household_creation():
    """Test that a household can be created."""
    print("\nTesting household creation...\n")
    
    try:
        from agents.household import Household
        
        hh = Household(
            h_id=1,
            income_group=3,
            income=30000,
            consumption_q=3000,
            energy_flag=0,
            dwelling_label=3,
            owner=True
        )
        
        print(f"✓ Created household: ID={hh.h_id}, Income=€{hh.h_income}, Consumption={hh.h_q} kWh")
        print(f"  - Energy source: {hh.flag} (0=FF, 1=LCE, 2=SLCE)")
        print(f"  - Income group: {hh.h_income_group}")
        print(f"  - Dwelling label: {hh.dw_el}")
        
        # Test method
        hh.update_awareness()
        print(f"✓ Updated awareness: {hh.h_aware:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Household creation failed: {e}")
        return False


def test_data_loader():
    """Test that data loader can initialize."""
    print("\nTesting data loader...\n")
    
    try:
        from data_loader.loader import DataLoader
        
        loader = DataLoader(str(project_root))
        print("✓ DataLoader initialized")
        print(f"  - Base path: {loader.base_path}")
        print(f"  - Data directory: {loader.data_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data loader failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("BENCH MODEL - STRUCTURE TEST")
    print("="*60 + "\n")
    
    all_pass = True
    
    if not test_imports():
        all_pass = False
    
    if not test_household_creation():
        all_pass = False
    
    if not test_data_loader():
        all_pass = False
    
    print("\n" + "="*60)
    if all_pass:
        print("✓ ALL TESTS PASSED")
        print("\nStructure is ready! Run: python main.py")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease check errors above")
    print("="*60 + "\n")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

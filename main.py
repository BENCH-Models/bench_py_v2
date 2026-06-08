"""
BENCH Model - Main Entry Point
Behavioral Energy Consumption Household Model in Pure Python

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model.bench_model import BENCHModel
from utils.constants import MODEL_START_YEAR, MODEL_END_YEAR


def main():
    """Main entry point for BENCH model simulation."""
    
    # Configuration
    CASE_STUDY = "Netherlands-Overijssel"  # or "Spain-Navarre"
    SCENARIO = "Ref_SSP2"
    POLICY = "Ref"  # or "Carbon price pressure-25", etc.
    
    # Create and run model
    model = BENCHModel(
        case_study=CASE_STUDY,
        scenario=SCENARIO,
        policy=POLICY,
        base_path=str(project_root)
    )
    
    # Optional: Enable debug output
    model.debug = False
    
    # Run simulation
    success = model.run(verbose=True)
    
    if not success:
        print("\n✗ Simulation failed")
        return 1
    
    # Print summary
    summary = model.get_summary()
    print("\nFINAL CUMULATIVE RESULTS:")
    print("-" * 60)
    print(f"Total Investment: €{summary['total_investment']:,.2f}")
    print(f"Total Energy Saved: {summary['total_energy_saved']:,.0f} kWh")
    print(f"Total Emissions Avoided: {summary['total_emissions_avoided']:,.0f} kg CO2")
    print(f"Total Actions: {summary['actions_cumulative']:,.0f}")
    
    # Export results
    print("\nExporting results...")
    files = model.export_results()
    print(f"\nResults exported to: {model.run_output_dir}")
    #print(f"Plots saved to: {model.exporter.plots_dir}")
    print(f"Files: {len(files)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

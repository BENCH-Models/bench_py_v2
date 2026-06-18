"""
BENCH Model - Single Test Run Script
Runs the model once with minimal configuration for testing purposes
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model.bench_model import BENCHModel
from model.parameters import DEFAULT_LEARNING_TYPE


def run_single_test():
    """
    Run a single test simulation of the BENCH model
    with default parameters for quick testing
    """
    print("=" * 80)
    print("BENCH MODEL - SINGLE TEST RUN")
    print("=" * 80)
    
    # Test configuration
    test_config = {
        "case_study": "Netherlands-Overijssel",
        "scenario": "Ref_SSP2",
        "policy": "Ref",
        "learning_type": DEFAULT_LEARNING_TYPE,
        "run_label": "test_run_single",
        "seed": 42,  # Fixed seed for reproducibility
        "debug": True,  # Enable debug output for testing
    }
    
    print(f"\nTest Configuration:")
    print(f"  Case Study: {test_config['case_study']}")
    print(f"  Scenario: {test_config['scenario']}")
    print(f"  Policy: {test_config['policy']}")
    print(f"  Learning Type: {test_config['learning_type']}")
    print(f"  Random Seed: {test_config['seed']}")
    print(f"  Run Label: {test_config['run_label']}")
    print("-" * 80)
    
    try:
        # Initialize the model
        print("\nInitializing BENCHModel...")
        model = BENCHModel(
            case_study=test_config["case_study"],
            scenario=test_config["scenario"],
            policy=test_config["policy"],
            learning_type=test_config["learning_type"],
            run_label=test_config["run_label"],
            base_path=str(project_root),
            output_root=None,  # Use default output directory
            seed=test_config["seed"],
        )
        
        # Enable debug output
        model.debug = test_config["debug"]
        
        # Run the simulation
        print("\nRunning simulation...")
        success = model.run()
        
        if not success:
            print("\nâœ— Simulation failed!")
            return False
        
        # Get and display results summary
        print("\n" + "=" * 80)
        print("SIMULATION RESULTS SUMMARY")
        print("=" * 80)
        
        summary = model.get_summary()
        if summary:
            print(f"\nTotal Investment: â‚¬{summary.get('total_investment', 0):,.2f}")
            print(f"Total Energy Saved: {summary.get('total_energy_saved', 0):,.0f} kWh")
            print(f"Total Emissions Avoided: {summary.get('total_emissions_avoided', 0):,.0f} kg CO2")
            print(f"Total Actions: {summary.get('actions_cumulative', 0):,.0f}")
            
            # Additional metrics if available
            if 'payback_period' in summary:
                print(f"Average Payback Period: {summary.get('payback_period', 0):.1f} years")
            if 'roi' in summary:
                print(f"Return on Investment: {summary.get('roi', 0):.1f}%")
        
        # Export results
        print("\nExporting results...")
        model.export_results()
        
        print("\nâœ“ Simulation completed successfully!")

        
        return True
        
    except Exception as e:
        print(f"\nâœ— Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BENCH MODEL TEST SUITE")
    print("=" * 80)
    
    # Run main test
    success = run_single_test()
    
    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Full Test: {'âœ“ PASSED' if success else 'âœ— FAILED'}")

    
    sys.exit(0 if success else 1)
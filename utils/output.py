"""
Output and results export functionality
"""

import os
import csv
import json
import pandas as pd
from typing import Dict, List
from utils.constants import OUTPUT_DIR, VERBOSE

try:
    import yaml
except ImportError:
    yaml = None


class ResultsExporter:
    """
    Exports simulation results to CSV and other formats.
    """
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        """
        Initialize exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        self.ensure_output_dir()
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.ensure_plots_dir()
    
    def ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            #print(f"Created output directory: {self.output_dir}")
    
    def ensure_plots_dir(self) -> None:
        """Create plots directory inside the run output folder."""
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            #print(f"Created plots directory: {self.plots_dir}")
    
    def export_annual_aggregates(self, stats_aggregator, 
                                start_year: int, end_year: int,
                                filename: str = "annual_aggregates.csv") -> str:
        """
        Export annual population-level statistics.
        
        Args:
            stats_aggregator: StatisticsAggregator instance
            start_year: First year to export
            end_year: Last year to export
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        rows = []
        for year in range(start_year, end_year + 1):
            if year in stats_aggregator.annual_stats:
                rows.append(stats_aggregator.annual_stats[year])
        
        if not rows:
            print("No annual stats to export")
            return output_path
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        #print(f"✓ Exported annual aggregates: {output_path}")
        
        return output_path
    
    def export_household_actions(self, households: List, year: int,
                                filename: str = None) -> str:
        """
        Export household-level action data for a specific year.
        
        Args:
            households: List of Household objects
            year: Year to export
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"household_actions_{year}.csv"
        
        output_path = os.path.join(self.output_dir, filename)
        
        rows = []
        for hh in households:
            row = {
                'h_id': hh.h_id,
                'year': year,
                'income_group': hh.h_income_group,
                'income': hh.h_income,
                'consumption': hh.h_q,
                'energy_source': hh.flag,
                'dwelling_label': hh.dw_el,
                'awareness': hh.h_aware,
                'guilt': hh.guilt,
                'motivation_avg': sum(hh.h_motiv) / 3,
                'pbc_avg': sum(hh.pbc) / 3,
                'action_investment': hh.act1,
                'action_conservation': hh.act2,
                'action_switching': hh.act3,
                'investment_total': hh.h_invest_total,
                'conservation_savings': hh.h_conserv,
                'switching_benefit': hh.h_switch,
                'emissions_avoided': sum(hh.em_avoided),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        #print(f"✓ Exported household actions: {output_path}")
        
        return output_path
    
    def export_actions_by_group(self, stats_aggregator,
                               start_year: int, end_year: int,
                               filename: str = "actions_by_income_group.csv") -> str:
        """
        Export actions aggregated by income group.
        
        Args:
            stats_aggregator: StatisticsAggregator instance
            start_year: First year
            end_year: Last year
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        rows = []
        for year in range(start_year, end_year + 1):
            # This would need income group stats stored in aggregator
            # Simplified version below
            rows.append({
                'year': year,
                'note': 'Income group breakdowns would be stored and exported here'
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def export_summary_report(self, stats_aggregator,
                             case_study: str, scenario: str, policy: str,
                             start_year: int, end_year: int,
                             filename: str = "summary_report.txt") -> str:
        """
        Export summary report of entire simulation.
        
        Args:
            stats_aggregator: StatisticsAggregator instance
            case_study: Case study name
            scenario: Scenario name
            policy: Policy name
            start_year: First year
            end_year: Last year
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = os.path.join(self.output_dir, filename)
        
        cumulative = stats_aggregator.get_cumulative_stats(start_year, end_year)
        
        with open(output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BENCH MODEL SIMULATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("SIMULATION CONFIGURATION\n")
            f.write("-" * 60 + "\n")
            f.write(f"Case Study: {case_study}\n")
            f.write(f"Scenario: {scenario}\n")
            f.write(f"Policy: {policy}\n")
            f.write(f"Time Period: {start_year}-{end_year}\n")
            f.write(f"Duration: {cumulative['years']} years\n\n")
            
            f.write("CUMULATIVE RESULTS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total Investments: €{cumulative['total_investment']:,.2f}\n")
            f.write(f"Total Energy Saved: {cumulative['total_energy_saved']:,.0f} kWh\n")
            f.write(f"Total Emissions Avoided: {cumulative['total_emissions_avoided']:,.0f} kg CO2\n")
            f.write(f"Total Conservation Savings: €{cumulative['total_conservation_savings']:,.2f}\n")
            f.write(f"Total Actions Taken: {cumulative['actions_cumulative']:,.0f}\n\n")
            
            # Final year metrics
            if end_year in stats_aggregator.annual_stats:
                final = stats_aggregator.annual_stats[end_year]
                f.write("FINAL YEAR METRICS (Year {})\n".format(end_year))
                f.write("-" * 60 + "\n")
                f.write(f"Renewable Energy Share: {final.get('lce_share_percent'):.1f}%\n")
                f.write(f"High Awareness Households: {final.get('high_guilt_percent'):.1f}%\n")
                f.write(f"Average Awareness: {final.get('avg_awareness'):.2f}\n\n")
            
            f.write("=" * 60 + "\n")
        
        #print(f"✓ Exported summary report: {output_path}")
        return output_path
    
    def export_run_config(self, config: Dict,
                          filename: str = "run_config.json") -> List[str]:
        """Export run configuration to JSON and optionally YAML."""
        output_paths = []
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        #print(f"✓ Exported run configuration: {output_path}")
        output_paths.append(output_path)

        if yaml is not None:
            yaml_path = os.path.join(self.output_dir, 'run_config.yaml')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, sort_keys=False)
            #print(f"✓ Exported run configuration: {yaml_path}")
            output_paths.append(yaml_path)

        return output_paths
    
    def export_trajectory(self, stats_aggregator, variable: str,
                         start_year: int, end_year: int,
                         filename: str = None) -> str:
        """
        Export time series for a specific variable.
        
        Args:
            stats_aggregator: StatisticsAggregator instance
            variable: Variable name
            start_year: First year
            end_year: Last year
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"trajectory_{variable}.csv"
        
        output_path = os.path.join(self.output_dir, filename)
        
        trajectory = stats_aggregator.get_trajectory(variable, start_year, end_year)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['year', variable])
            for year, value in trajectory:
                writer.writerow([year, value])
        
        #print(f"✓ Exported trajectory: {output_path}")
        return output_path
    
    def export_all_results(self, model, start_year: int, end_year: int) -> List[str]:
        """
        Export all available results from a completed model run.
        
        Args:
            model: BENCHModel instance
            start_year: First year
            end_year: Last year
            
        Returns:
            List of exported file paths
        """
        files = []
        if VERBOSE:
            print("\nExporting simulation results...")
        
        # Annual aggregates
        files.append(self.export_annual_aggregates(
            model.statistics, start_year, end_year
        ))
        
        # Summary report
        files.append(self.export_summary_report(
            model.statistics, model.case_study, model.scenario, 
            model.policy, start_year, end_year
        ))
        
        # Key trajectories
        trajectories = [
            'lce_share_percent',
            'action_1_count',
            'action_2_count',
            'action_3_count',
            'total_emissions_avoided_kg_co2',
            'total_energy_saved_kwh',
        ]
        
        for variable in trajectories:
            files.append(self.export_trajectory(
                model.statistics, variable, start_year, end_year
            ))
        if VERBOSE:
            print(f"\n✓ All results exported to: {self.output_dir}")
        return files

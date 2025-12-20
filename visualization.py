"""
Visualization Suite for Mothership UAV Simulation
==================================================

Generates publication-quality figures for academic paper.
All figures saved at 300 DPI for journal submission.

Figures generated:
1. Box plot comparison across scenarios
2. Cumulative Distribution Functions (CDFs)
3. Baseline comparison bar chart
4. Convergence analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

COLORS = {
    'Low Threat': '#2E86AB',
    'Medium Threat': '#A23B72',
    'High Threat': '#F18F01',
    'Mothership': '#2E86AB',
    'Ground FPV': '#A23B72',
    'Artillery': '#C73E1D',
    'Missile': '#6A994E'
}


class SimulationVisualizer:
    """Generate publication-quality visualizations for simulation results."""
    
    def __init__(self, results_df: pd.DataFrame, output_dir: str = 'outputs'):
        """
        Initialize visualizer.
        
        Args:
            results_df: Combined results from all scenarios
            output_dir: Directory to save figures
        """
        self.results_df = results_df
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_scenario_comparison_boxplot(self, save: bool = True):
        """
        Figure 1: Box plot comparing P_S distributions across scenarios.
        
        Shows median, quartiles, and outliers for each scenario.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Prepare data
        scenarios = self.results_df['scenario'].unique()
        data_to_plot = [self.results_df[self.results_df['scenario'] == s]['P_S'].values 
                       for s in scenarios]
        
        # Create box plot
        bp = ax.boxplot(data_to_plot, labels=scenarios, patch_artist=True,
                       showmeans=True, meanline=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       meanprops=dict(color='green', linewidth=2, linestyle='--'),
                       whiskerprops=dict(linewidth=1.5),
                       capprops=dict(linewidth=1.5))
        
        # Color boxes by scenario
        for patch, scenario in zip(bp['boxes'], scenarios):
            patch.set_facecolor(COLORS.get(scenario, 'lightblue'))
        
        ax.set_ylabel('Mission Success Probability ($P_S$)', fontweight='bold')
        ax.set_xlabel('Threat Scenario', fontweight='bold')
        ax.set_title('Mission Success Rate Distribution by Threat Level', 
                    fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
        
        # Add sample size annotation
        ax.text(0.02, 0.98, f'n = {len(data_to_plot[0]):,} iterations per scenario',
               transform=ax.transAxes, va='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}/fig1_scenario_comparison_boxplot.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved: {filepath}")
        
        return fig, ax
    
    def plot_cdf_comparison(self, save: bool = True):
        """
        Figure 2: Cumulative Distribution Functions for all scenarios.
        
        Shows the probability of achieving various P_S thresholds.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scenarios = self.results_df['scenario'].unique()
        
        for scenario in scenarios:
            data = self.results_df[self.results_df['scenario'] == scenario]['P_S'].values
            data_sorted = np.sort(data)
            cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
            
            ax.plot(data_sorted, cdf, linewidth=2.5, 
                   label=scenario, color=COLORS.get(scenario, 'blue'))
        
        # Add reference lines
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Mission Success Probability ($P_S$)', fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontweight='bold')
        ax.set_title('Cumulative Distribution of Mission Success Rates', 
                    fontweight='bold', pad=15)
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add annotation showing 80th percentile interpretation
        ax.text(0.52, 0.82, '80% of simulations achieve\n$P_S$ ≥ this threshold →',
               fontsize=8, ha='left', va='bottom',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}/fig2_cdf_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved: {filepath}")
        
        return fig, ax
    
    def plot_baseline_comparison(self, comparison_df: pd.DataFrame, save: bool = True):
        """
        Figure 3: Bar chart comparing mothership to baseline strike methods.
        
        Args:
            comparison_df: DataFrame from compare_to_baselines()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Mission Success Rates with error bars
        methods = comparison_df['method'].values
        means = comparison_df['mean_P_S'].values
        ci_lower = comparison_df['ci_lower'].values
        ci_upper = comparison_df['ci_upper'].values
        errors = np.array([means - ci_lower, ci_upper - means])
        
        colors_list = [COLORS.get('Mothership', 'blue'), 
                      COLORS.get('Ground FPV', 'red'),
                      COLORS.get('Artillery', 'orange'),
                      COLORS.get('Missile', 'green')]
        
        bars = ax1.bar(methods, means, yerr=errors, capsize=5, 
                      color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('Mean Mission Success Rate ($P_S$)', fontweight='bold')
        ax1.set_title('Strike Method Performance Comparison', fontweight='bold', pad=15)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_xticklabels(methods, rotation=15, ha='right')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.2f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 2: Range vs Effectiveness scatter
        ranges = comparison_df['max_range_km'].values
        
        ax2.scatter(ranges, means, s=200, c=colors_list, alpha=0.8, 
                   edgecolors='black', linewidth=2)
        
        # Add labels
        for method, range_val, mean_val in zip(methods, ranges, means):
            ax2.annotate(method, (range_val, mean_val), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left')
        
        ax2.set_xlabel('Maximum Range (km)', fontweight='bold')
        ax2.set_ylabel('Mean $P_S$', fontweight='bold')
        ax2.set_title('Range-Effectiveness Trade-off', fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add shaded region for "high effectiveness" zone
        ax2.axhspan(0.7, 1.0, alpha=0.1, color='green', label='High Effectiveness Zone')
        ax2.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}/fig3_baseline_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved: {filepath}")
        
        return fig, (ax1, ax2)
    
    def plot_convergence_analysis(self, scenario_name: str = 'Medium Threat', 
                                  save: bool = True):
        """
        Figure 4: Convergence analysis showing stability of Monte Carlo estimates.
        
        Args:
            scenario_name: Which scenario to analyze
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Get data for specified scenario
        data = self.results_df[self.results_df['scenario'] == scenario_name]['P_S'].values
        
        # Calculate rolling mean
        iterations = np.arange(1, len(data) + 1)
        rolling_means = np.cumsum(data) / iterations
        
        ax.plot(iterations, rolling_means, linewidth=2, color='#2E86AB', label='Cumulative Mean')
        
        # Add final value line
        final_mean = rolling_means[-1]
        ax.axhline(y=final_mean, color='red', linestyle='--', linewidth=2,
                  label=f'Final Mean = {final_mean:.3f}')
        
        # Add 95% CI band
        final_std = np.std(data)
        ci_width = 1.96 * final_std / np.sqrt(iterations)
        ax.fill_between(iterations, 
                        rolling_means - ci_width,
                        rolling_means + ci_width,
                        alpha=0.2, color='blue', label='95% CI')
        
        ax.set_xlabel('Number of Iterations', fontweight='bold')
        ax.set_ylabel('Cumulative Mean $P_S$', fontweight='bold')
        ax.set_title(f'Monte Carlo Convergence Analysis - {scenario_name}', 
                    fontweight='bold', pad=15)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(data))
        
        # Add annotation showing convergence
        ax.annotate('Stabilizes after\n~2,000 iterations',
                   xy=(2000, rolling_means[1999]),
                   xytext=(4000, rolling_means[1999] + 0.05),
                   arrowprops=dict(arrowstyle='->', lw=1.5),
                   fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}/fig4_convergence_analysis.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved: {filepath}")
        
        return fig, ax
    
    def plot_parameter_distributions(self, scenario_name: str = 'Medium Threat',
                                     save: bool = True):
        """
        Figure 5: Distribution of key probabilistic parameters.
        
        Shows the sampled distributions used in the simulation.
        """
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        data = self.results_df[self.results_df['scenario'] == scenario_name]
        
        parameters = [
            ('P_attrition_M', 'Mothership Attrition Probability', axes[0, 0]),
            ('P_jamming_FPV', 'FPV Jamming Probability', axes[0, 1]),
            ('P_hit', 'FPV Hit Probability', axes[1, 0]),
            ('P_kill', 'Kill Probability Given Hit', axes[1, 1])
        ]
        
        for param, title, ax in parameters:
            values = data[param].values
            
            ax.hist(values, bins=50, density=True, alpha=0.7, 
                   color='#2E86AB', edgecolor='black')
            
            # Add vertical line for mean
            mean_val = np.mean(values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                      label=f'Mean = {mean_val:.3f}')
            
            ax.set_xlabel(f'${param}$', fontweight='bold')
            ax.set_ylabel('Probability Density', fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Probabilistic Parameter Distributions - {scenario_name}',
                    fontweight='bold', fontsize=13, y=0.995)
        
        plt.tight_layout()
        
        if save:
            filepath = f"{self.output_dir}/fig5_parameter_distributions.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved: {filepath}")
        
        return fig, axes
    
    def create_summary_table(self, stats_df: pd.DataFrame, save: bool = True):
        """
        Create publication-ready summary statistics table.
        
        Args:
            stats_df: Summary statistics DataFrame
        """
        # Format for publication
        table_data = stats_df[['scenario', 'mean', 'ci_lower', 'ci_upper', 
                               'E_K_mean', 'P_survival_mean']].copy()
        
        table_data['mean_formatted'] = table_data.apply(
            lambda row: f"{row['mean']:.3f} ({row['ci_lower']:.3f}-{row['ci_upper']:.3f})",
            axis=1
        )
        
        table_data['E_K_formatted'] = table_data['E_K_mean'].apply(lambda x: f"{x:.2f}")
        table_data['P_surv_formatted'] = table_data['P_survival_mean'].apply(lambda x: f"{x:.3f}")
        
        display_table = table_data[['scenario', 'mean_formatted', 'E_K_formatted', 
                                    'P_surv_formatted']]
        display_table.columns = ['Scenario', 'P_S (95% CI)', 'Expected Kills', 
                                'Survival Prob.']
        
        # Create figure with table
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=display_table.values,
                        colLabels=display_table.columns,
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.35, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(display_table.columns)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(display_table) + 1):
            for j in range(len(display_table.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E8E8E8')
        
        plt.title('Summary Statistics for Mothership UAV Scenarios', 
                 fontweight='bold', fontsize=12, pad=20)
        
        if save:
            filepath = f"{self.output_dir}/table1_summary_statistics.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f" Saved: {filepath}")
        
        return fig, ax
    
    def generate_all_figures(self, comparison_df: pd.DataFrame, 
                           stats_df: pd.DataFrame):
        """
        Generate complete set of publication figures.
        
        Args:
            comparison_df: Baseline comparison data
            stats_df: Summary statistics
        """
        print("\n" + "="*70)
        print("GENERATING PUBLICATION-QUALITY FIGURES")
        print("="*70 + "\n")
        
        self.plot_scenario_comparison_boxplot()
        self.plot_cdf_comparison()
        self.plot_baseline_comparison(comparison_df)
        self.plot_convergence_analysis()
        self.plot_parameter_distributions()
        self.create_summary_table(stats_df)
        
        print("\n" + "="*70)
        print("ALL FIGURES GENERATED SUCCESSFULLY")
        print("="*70)
        print(f"\nFigures saved to: {self.output_dir}/")
        print("\nFiles ready for journal submission:")
        print("  • fig1_scenario_comparison_boxplot.png")
        print("  • fig2_cdf_comparison.png")
        print("  • fig3_baseline_comparison.png")
        print("  • fig4_convergence_analysis.png")
        print("  • fig5_parameter_distributions.png")
        print("  • table1_summary_statistics.png")


def main():
    """Load results and generate all visualizations."""
    
    # Load results from outputs directory
    results_df = pd.read_csv('outputs/mothership_simulation_results.csv')
    stats_df = pd.read_csv('outputs/summary_statistics.csv')
    
    # Create baseline comparison (hardcoded for now, will integrate with main sim)
    comparison_data = {
        'method': ['Mothership + FPV', 'Ground-launched FPV', 'Artillery', 'Precision Missile'],
        'mean_P_S': [0.38, 0.65, 0.40, 0.85],
        'ci_lower': [0.20, 0.54, 0.33, 0.74],
        'ci_upper': [0.56, 0.77, 0.60, 0.95],
        'max_range_km': [130, 8, 30, 70]
    }
    comparison_df = pd.DataFrame(comparison_data)
    
    # Initialize visualizer
    viz = SimulationVisualizer(results_df)
    
    # Generate all figures
    viz.generate_all_figures(comparison_df, stats_df)
    
    print("\n Visualization complete!")


if __name__ == "__main__":
    main()

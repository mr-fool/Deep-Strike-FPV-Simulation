"""
Sensitivity Analysis for Mothership UAV Simulation
===================================================

One-at-a-time (OAT) sensitivity analysis to identify critical parameters
affecting mission success rate (P_S).

For each parameter, varies it across its range while holding all others constant,
then calculates the impact on P_S.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mothership_simulation import MonteCarloSimulation
import os
import warnings
warnings.filterwarnings('ignore')

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'


class SensitivityAnalyzer:
    """
    Perform sensitivity analysis on mothership simulation.
    
    Uses one-at-a-time (OAT) methodology to isolate parameter impacts.
    """
    
    def __init__(self, base_scenario: dict, n_iterations: int = 10000):
        """
        Initialize sensitivity analyzer.
        
        Args:
            base_scenario: Baseline scenario (typically Medium Threat)
            n_iterations: Monte Carlo iterations per parameter value
        """
        self.base_scenario = base_scenario.copy()
        self.n_iterations = n_iterations
        self.sim = MonteCarloSimulation(n_iterations=n_iterations)
        self.results = {}
        
    def analyze_mothership_range(self, ranges: list = None) -> pd.DataFrame:
        """
        Analyze sensitivity to mothership operational range.
        
        Args:
            ranges: List of ranges to test (km)
        """
        if ranges is None:
            ranges = [60, 90, 120, 150, 180, 200]
        
        print("\nAnalyzing sensitivity to MOTHERSHIP RANGE...")
        results = []
        
        for R_M_max in ranges:
            scenario = self.base_scenario.copy()
            scenario['mothership']['R_M_max'] = R_M_max
            # CRITICAL FIX: Update target_distance to match new range
            # Default is 60% of max range (same as define_scenario default)
            scenario['target_distance'] = R_M_max * 0.6
            
            res_df = self.sim.run_scenario(scenario, target_type='armor')
            stats = self.sim.calculate_statistics(res_df)
            
            results.append({
                'parameter': 'R_M_max',
                'value': R_M_max,
                'P_S_mean': stats['mean'],
                'P_S_std': stats['std'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper']
            })
            print(f"  R_M_max = {R_M_max} km (target @ {R_M_max * 0.6:.0f} km) → P_S = {stats['mean']:.3f}")
        
        return pd.DataFrame(results)
    
    def analyze_fpv_count(self, counts: list = None) -> pd.DataFrame:
        """
        Analyze sensitivity to number of FPVs deployed.
        
        Args:
            counts: List of FPV counts to test
        """
        if counts is None:
            counts = [2, 3, 5, 7, 10]
        
        print("\nAnalyzing sensitivity to FPV COUNT...")
        results = []
        
        for N_FPV in counts:
            scenario = self.base_scenario.copy()
            scenario['N_FPV'] = N_FPV
            
            res_df = self.sim.run_scenario(scenario, target_type='armor')
            stats = self.sim.calculate_statistics(res_df)
            
            results.append({
                'parameter': 'N_FPV',
                'value': N_FPV,
                'P_S_mean': stats['mean'],
                'P_S_std': stats['std'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'E_K_mean': stats['E_K_mean']
            })
            print(f"  N_FPV = {N_FPV} → P_S = {stats['mean']:.3f}, E[K] = {stats['E_K_mean']:.2f}")
        
        return pd.DataFrame(results)
    
    def analyze_ad_density(self, densities: list = None) -> pd.DataFrame:
        """
        Analyze sensitivity to air defense density.
        
        Args:
            densities: List of AD densities to test (systems per 100 km²)
        """
        if densities is None:
            densities = [0.5, 1.0, 2.0, 3.0, 5.0]
        
        print("\nAnalyzing sensitivity to AD DENSITY...")
        results = []
        
        for rho_AD in densities:
            scenario = self.base_scenario.copy()
            scenario['rho_AD'] = rho_AD
            
            res_df = self.sim.run_scenario(scenario, target_type='armor')
            stats = self.sim.calculate_statistics(res_df)
            
            results.append({
                'parameter': 'rho_AD',
                'value': rho_AD,
                'P_S_mean': stats['mean'],
                'P_S_std': stats['std'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper'],
                'P_survival_mean': stats['P_survival_mean']
            })
            print(f"  ρ_AD = {rho_AD} → P_S = {stats['mean']:.3f}, P_survival = {stats['P_survival_mean']:.3f}")
        
        return pd.DataFrame(results)
    
    def analyze_guidance_type(self, types: list = None) -> pd.DataFrame:
        """
        Analyze sensitivity to FPV guidance type.
        
        Args:
            types: List of guidance types ('radio', 'fiber', 'ai')
        """
        if types is None:
            types = ['radio', 'fiber', 'ai']
        
        print("\nAnalyzing sensitivity to GUIDANCE TYPE...")
        results = []
        
        for guidance in types:
            scenario = self.base_scenario.copy()
            scenario['guidance_type'] = guidance
            
            res_df = self.sim.run_scenario(scenario, target_type='armor')
            stats = self.sim.calculate_statistics(res_df)
            
            results.append({
                'parameter': 'guidance_type',
                'value': guidance,
                'P_S_mean': stats['mean'],
                'P_S_std': stats['std'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper']
            })
            print(f"  Guidance = {guidance} → P_S = {stats['mean']:.3f}")
        
        return pd.DataFrame(results)
    
    def analyze_altitude(self, altitudes: list = None) -> pd.DataFrame:
        """
        Analyze sensitivity to mothership operational altitude.
        
        Args:
            altitudes: List of altitudes to test (meters)
        """
        if altitudes is None:
            altitudes = [1000, 1500, 2000, 2500, 3000]
        
        print("\nAnalyzing sensitivity to OPERATIONAL ALTITUDE...")
        results = []
        
        for H_M_op in altitudes:
            scenario = self.base_scenario.copy()
            scenario['mothership']['H_M_op'] = H_M_op
            
            res_df = self.sim.run_scenario(scenario, target_type='armor')
            stats = self.sim.calculate_statistics(res_df)
            
            results.append({
                'parameter': 'H_M_op',
                'value': H_M_op,
                'P_S_mean': stats['mean'],
                'P_S_std': stats['std'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper']
            })
            print(f"  H_M = {H_M_op} m → P_S = {stats['mean']:.3f}")
        
        return pd.DataFrame(results)
    
    def analyze_wind_speed(self, wind_speeds: list = None) -> pd.DataFrame:
        """
        Analyze sensitivity to wind speed.
        
        Args:
            wind_speeds: List of wind speeds to test (km/h)
        """
        if wind_speeds is None:
            wind_speeds = [0, 10, 20, 30, 40]
        
        print("\nAnalyzing sensitivity to WIND SPEED...")
        results = []
        
        for V_wind in wind_speeds:
            scenario = self.base_scenario.copy()
            scenario['V_wind'] = V_wind
            
            res_df = self.sim.run_scenario(scenario, target_type='armor')
            stats = self.sim.calculate_statistics(res_df)
            
            results.append({
                'parameter': 'V_wind',
                'value': V_wind,
                'P_S_mean': stats['mean'],
                'P_S_std': stats['std'],
                'ci_lower': stats['ci_lower'],
                'ci_upper': stats['ci_upper']
            })
            print(f"  V_wind = {V_wind} km/h → P_S = {stats['mean']:.3f}")
        
        return pd.DataFrame(results)
    
    def run_full_sensitivity_analysis(self) -> dict:
        """
        Run complete sensitivity analysis for all parameters.
        
        Returns:
            Dictionary of DataFrames, one per parameter
        """
        print("\n" + "="*70)
        print("SENSITIVITY ANALYSIS - MEDIUM THREAT BASELINE")
        print("="*70)
        
        results = {
            'range': self.analyze_mothership_range(),
            'fpv_count': self.analyze_fpv_count(),
            'ad_density': self.analyze_ad_density(),
            'guidance': self.analyze_guidance_type(),
            'altitude': self.analyze_altitude(),
            'wind': self.analyze_wind_speed()
        }
        
        self.results = results
        return results
    
    def calculate_sensitivity_coefficients(self) -> pd.DataFrame:
        """
        Calculate normalized sensitivity coefficients for tornado diagram.
        
        Returns:
            DataFrame with sensitivity metrics for each parameter
        """
        coefficients = []
        
        for param_name, df in self.results.items():
            if df.empty:
                continue
            
            # Calculate range of P_S across parameter variation
            P_S_range = df['P_S_mean'].max() - df['P_S_mean'].min()
            
            # Calculate relative sensitivity (normalized by baseline P_S)
            baseline_P_S = self.base_scenario.get('baseline_P_S', 0.38)
            relative_sensitivity = P_S_range / baseline_P_S
            
            coefficients.append({
                'parameter': param_name,
                'P_S_range': P_S_range,
                'relative_sensitivity': relative_sensitivity,
                'min_P_S': df['P_S_mean'].min(),
                'max_P_S': df['P_S_mean'].max()
            })
        
        return pd.DataFrame(coefficients).sort_values('P_S_range', ascending=False)
    
    def plot_tornado_diagram(self, save: bool = True):
        """
        Create tornado diagram showing parameter sensitivities.
        
        The width of each bar represents the range of P_S when that parameter is varied.
        """
        coef_df = self.calculate_sensitivity_coefficients()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        params = coef_df['parameter'].values
        param_labels = {
            'guidance': 'Guidance Type',
            'ad_density': 'AD Density (ρ_AD)',
            'altitude': 'Operational Altitude',
            'wind': 'Wind Speed',
            'fpv_count': 'FPV Count',
            'range': 'Mothership Range'
        }
        
        y_labels = [param_labels.get(p, p) for p in params]
        
        # Create horizontal bars showing min to max P_S
        y_pos = np.arange(len(params))
        
        for i, (param, min_val, max_val) in enumerate(zip(params, 
                                                          coef_df['min_P_S'], 
                                                          coef_df['max_P_S'])):
            # Width of bar = range of P_S
            width = max_val - min_val
            
            # Center bar around baseline
            baseline = (min_val + max_val) / 2
            left = baseline - width/2
            
            ax.barh(i, width, left=left, height=0.6,
                   color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add text showing range
            ax.text(max_val + 0.01, i, f'{width:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        # Baseline reference line - use actual baseline from scenario
        baseline_P_S = self.base_scenario.get('baseline_P_S', 0.38)
        ax.axvline(baseline_P_S, color='red', linestyle='--', linewidth=2,
                  label=f'Baseline P_S = {baseline_P_S:.3f}', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Mission Success Probability ($P_S$)', fontweight='bold')
        ax.set_title('Parameter Sensitivity Analysis - Tornado Diagram', 
                    fontweight='bold', pad=15)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, 0.8)
        
        plt.tight_layout()
        
        if save:
            os.makedirs('outputs', exist_ok=True)
            filepath = 'outputs/fig6_tornado_diagram.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nSaved: {filepath}")
        
        return fig, ax
    
    def plot_sensitivity_curves(self, save: bool = True):
        """
        Plot sensitivity curves showing P_S vs each parameter.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_configs = [
            ('ad_density', 'AD Density (systems/100km²)', 'rho_AD', axes[0]),
            ('altitude', 'Operational Altitude (m)', 'H_M_op', axes[1]),
            ('wind', 'Wind Speed (km/h)', 'V_wind', axes[2]),
            ('range', 'Mothership Range (km)', 'R_M_max', axes[3]),
            ('fpv_count', 'Number of FPVs', 'N_FPV', axes[4]),
        ]
        
        for param_key, xlabel, param_symbol, ax in plot_configs:
            if param_key not in self.results:
                continue
            
            df = self.results[param_key]
            
            # Plot mean line with confidence interval
            ax.plot(df['value'], df['P_S_mean'], 'o-', linewidth=2.5, 
                   markersize=8, color='#2E86AB', label='Mean $P_S$')
            ax.fill_between(df['value'], df['ci_lower'], df['ci_upper'],
                           alpha=0.3, color='#2E86AB', label='95% CI')
            
            ax.set_xlabel(xlabel, fontweight='bold')
            ax.set_ylabel('$P_S$', fontweight='bold')
            ax.set_title(f'Sensitivity to {xlabel.split("(")[0].strip()}', 
                        fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
            ax.set_ylim(0, 0.8)
        
        # Handle guidance type separately (categorical)
        if 'guidance' in self.results:
            ax = axes[5]
            df = self.results['guidance']
            
            x = np.arange(len(df))
            ax.bar(x, df['P_S_mean'], yerr=[df['P_S_mean'] - df['ci_lower'],
                                            df['ci_upper'] - df['P_S_mean']],
                  capsize=5, color=['#C73E1D', '#2E86AB', '#6A994E'],
                  alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(df['value'].values)
            ax.set_ylabel('$P_S$', fontweight='bold')
            ax.set_xlabel('Guidance Type', fontweight='bold')
            ax.set_title('Sensitivity to Guidance Type', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 0.8)
        
        fig.suptitle('Parameter Sensitivity Curves - Medium Threat Baseline',
                    fontweight='bold', fontsize=14, y=0.995)
        
        plt.tight_layout()
        
        if save:
            os.makedirs('outputs', exist_ok=True)
            filepath = 'outputs/fig7_sensitivity_curves.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        return fig, axes


def main():
    """Run sensitivity analysis and generate visualizations."""
    
    # Define medium threat baseline scenario
    sim = MonteCarloSimulation()
    base_scenario = sim.define_scenario(
        name='Medium Threat',
        rho_AD=2.0,
        rho_EW=1.0,
        V_wind=15,
        visibility=1.0,
        N_FPV=5,
        guidance_type='fiber'
    )
    
    # Calculate actual baseline P_S from simulation (not hardcoded)
    print("\n" + "="*70)
    print("CALCULATING BASELINE P_S FOR SENSITIVITY ANALYSIS")
    print("="*70)
    baseline_df = sim.run_scenario(base_scenario, target_type='armor')
    baseline_stats = sim.calculate_statistics(baseline_df)
    baseline_P_S = baseline_stats['mean']
    base_scenario['baseline_P_S'] = baseline_P_S
    print(f"Baseline P_S = {baseline_P_S:.4f} (95% CI: {baseline_stats['ci_lower']:.4f}-{baseline_stats['ci_upper']:.4f})")
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(base_scenario, n_iterations=10000)
    
    # Run full sensitivity analysis
    results = analyzer.run_full_sensitivity_analysis()
    
    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING SENSITIVITY VISUALIZATIONS")
    print("="*70)
    
    analyzer.plot_tornado_diagram()
    analyzer.plot_sensitivity_curves()
    
    # Calculate and display sensitivity coefficients
    coef_df = analyzer.calculate_sensitivity_coefficients()
    
    print("\n" + "="*70)
    print("SENSITIVITY COEFFICIENTS")
    print("="*70)
    print("\nParameters ranked by impact on P_S:")
    print(coef_df.to_string(index=False))
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save results
    combined_results = pd.concat([df.assign(param=name) 
                                 for name, df in results.items()], 
                                ignore_index=True)
    combined_results.to_csv('outputs/sensitivity_analysis_results.csv', index=False)
    print("\nSensitivity results saved to: outputs/sensitivity_analysis_results.csv")
    
    coef_df.to_csv('outputs/sensitivity_coefficients.csv', index=False)
    print("Coefficients saved to: outputs/sensitivity_coefficients.csv")
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

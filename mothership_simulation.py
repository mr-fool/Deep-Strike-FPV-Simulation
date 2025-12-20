"""
Monte Carlo Simulation for Mothership UAV Deep Strike Analysis
================================================================


This simulation models the operational effectiveness of carrier UAVs (motherships)
deploying FPV drones for deep strike missions in Anti-Access/Area Denial (A2/AD)
environments, comparing performance across threat scenarios and against traditional
strike methods.

Key A2/AD Features:
- Layered air defense penetration modeling
- Time-based exposure risk in contested airspace
- Electronic warfare and jamming effects
- Mission timeline analysis for A2/AD loiter duration

Methodology: Monte Carlo simulation with 10,000 iterations per scenario
Statistical Approach: Probabilistic parameter modeling using triangular, uniform, and beta distributions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, Tuple, List
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Publication settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGURE_DPI = 300


class MonteCarloSimulation:
    """
    Monte Carlo simulation engine for mothership UAV effectiveness analysis.
    
    Models mission success probability (P_S) as a function of:
    - Mothership vulnerability to air defense
    - FPV jamming susceptibility  
    - Terminal engagement effectiveness
    - Environmental conditions
    """
    
    def __init__(self, n_iterations: int = 10000, random_seed: int = RANDOM_SEED):
        """
        Initialize simulation parameters.
        
        Args:
            n_iterations: Number of Monte Carlo runs per scenario
            random_seed: Seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Storage for results
        self.results = {}
        self.scenarios = {}
        
    def define_parameters(self) -> Dict:
        """
        Define all simulation parameters based on Excel requirements and literature.
        
        Returns:
            Dictionary of parameter definitions organized by category
        """
        params = {
            # MOTHERSHIP PARAMETERS
            'mothership': {
                'R_M_max': 120,          # km - Maximum operational radius
                'V_M': 100,              # km/h - Cruise speed
                'H_M_op': 2000,          # m - Operational altitude
                'N_FPV_max': 5,          # units - Maximum FPV payload
                'RCS': 0.1,              # m² - Radar cross section
            },
            
            # FPV DRONE PARAMETERS
            'fpv': {
                'R_FPV_max': 10,         # km - FPV range from release point
                'V_FPV': 150,            # km/h - Attack speed
                'M_warhead': 1.5,        # kg - Warhead mass
            },
            
            # BASELINE COMPARISON SYSTEMS
            'baselines': {
                'ground_fpv': {
                    'R_max': 8,          # km
                    'P_S_dist': (0.5, 0.65, 0.8),  # Triangular: (min, mode, max)
                },
                'artillery': {
                    'R_max': 30,         # km
                    'P_S_dist': (0.3, 0.4, 0.6),
                },
                'missile': {
                    'R_max': 70,         # km  
                    'P_S_dist': (0.7, 0.85, 0.95),
                }
            }
        }
        
        return params
    
    def define_scenario(self, name: str, rho_AD: float, rho_EW: float, 
                       V_wind: float, visibility: float, 
                       N_FPV: int = 5, guidance_type: str = 'fiber',
                       target_distance: float = None, n_targets: int = 1) -> Dict:
        """
        Define a specific operational scenario.
        
        Args:
            name: Scenario identifier
            rho_AD: Air defense density (systems per 100 km²)
            rho_EW: EW system density (systems per 50 km²)
            V_wind: Wind speed (km/h)
            visibility: Detection multiplier (1.0=clear, 0.7=hazy, 0.4=poor)
            N_FPV: Number of FPVs deployed
            guidance_type: 'radio', 'fiber', or 'ai'
            target_distance: Distance to target (km), None = medium range
            n_targets: Number of targets
            
        Returns:
            Scenario parameter dictionary
        """
        params = self.define_parameters()
        
        scenario = {
            'name': name,
            'rho_AD': rho_AD,
            'rho_EW': rho_EW,
            'V_wind': V_wind,
            'visibility': visibility,
            'N_FPV': N_FPV,
            'guidance_type': guidance_type,
            'target_distance': target_distance or params['mothership']['R_M_max'] * 0.6,
            'n_targets': n_targets,
            # Include base parameters
            **params
        }
        
        return scenario
    
    def sample_mothership_vulnerability(self, scenario: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample mothership detection and attrition probabilities.
        
        Model: P_detect = min(1.0, 0.3 + 0.4*(H_M/2000) + 0.3*ρ_AD) × visibility
        P_attrition | detected ~ Triangular(0.05, 0.15, 0.40)
        
        Args:
            scenario: Scenario parameters
            
        Returns:
            Tuple of (P_detect, P_attrition) arrays of length n_iterations
        """
        H_M = scenario['mothership']['H_M_op']
        rho_AD = scenario['rho_AD']
        vis = scenario['visibility']
        
        # Detection probability model
        base_detect = 0.3 + 0.4 * (H_M / 2000) + 0.3 * rho_AD
        P_detect = np.minimum(1.0, base_detect * vis)
        P_detect = np.repeat(P_detect, self.n_iterations)
        
        # Attrition given detection - Triangular distribution
        # Parameters: min, mode, max
        P_attrition_given_detect = np.random.triangular(0.05, 0.15, 0.40, self.n_iterations)
        
        # Overall attrition probability
        P_attrition = P_detect * P_attrition_given_detect
        
        return P_detect, P_attrition
    
    def calculate_mission_timeline(self, scenario: Dict) -> Tuple[float, float, float]:
        """
        Calculate mission timeline and A2/AD exposure duration.
        
        Critical for A2/AD analysis: longer exposure in contested airspace
        increases detection and attrition risk.
        
        Args:
            scenario: Scenario parameters
            
        Returns:
            Tuple of (T_transit, T_deployment, T_total) in minutes
        """
        target_distance = scenario['target_distance']  # km
        V_M = scenario['mothership']['V_M']  # km/h
        N_FPV = scenario['N_FPV']
        T_launch_seconds = 30  # seconds per FPV deployment
        
        # Transit time to FPV release point (assumes 80% of target distance)
        release_distance = target_distance * 0.8
        T_transit = (release_distance / V_M) * 60  # Convert to minutes
        
        # FPV deployment time (exposure window while releasing drones)
        T_deployment = (N_FPV * T_launch_seconds) / 60  # Convert to minutes
        
        # Total time exposed in A2/AD envelope
        T_total = T_transit + T_deployment
        
        return T_transit, T_deployment, T_total
    
    def apply_time_exposure_penalty(self, base_P_attrition: np.ndarray, 
                                    T_total: float) -> np.ndarray:
        """
        Apply time-based attrition penalty for extended A2/AD exposure.
        
        Longer loiter times in contested environments increase cumulative
        detection and engagement opportunities for adversary systems.
        
        Model: 1% additional attrition risk per minute of exposure
        
        Args:
            base_P_attrition: Base attrition probability
            T_total: Total time in A2/AD envelope (minutes)
            
        Returns:
            Modified attrition probability accounting for time exposure
        """
        # Time exposure factor: 1% increase per minute
        exposure_factor = 1.0 + (0.01 * T_total)
        
        # Apply factor to attrition (capped at 1.0)
        P_attrition_modified = np.minimum(1.0, base_P_attrition * exposure_factor)
        
        return P_attrition_modified
    
    def sample_fpv_jamming(self, scenario: Dict) -> np.ndarray:
        """
        Sample FPV jamming probability based on guidance type and EW density.
        
        Model:
        - Radio-controlled: P_jamming ~ Uniform(0.5, 0.7)
        - Fiber optic: P_jamming = 0.05 (fixed, minimal)
        - AI-enabled: P_jamming ~ Uniform(0.15, 0.25)
        
        Args:
            scenario: Scenario parameters
            
        Returns:
            P_jamming array of length n_iterations
        """
        guidance = scenario['guidance_type']
        rho_EW = scenario['rho_EW']
        
        if guidance == 'radio':
            P_jamming_base = np.random.uniform(0.5, 0.7, self.n_iterations)
        elif guidance == 'fiber':
            P_jamming_base = np.full(self.n_iterations, 0.05)
        elif guidance == 'ai':
            P_jamming_base = np.random.uniform(0.15, 0.25, self.n_iterations)
        else:
            raise ValueError(f"Unknown guidance type: {guidance}")
        
        # EW density multiplier (simplified model)
        ew_factor = 1.0 + 0.2 * rho_EW
        P_jamming = np.minimum(0.95, P_jamming_base * ew_factor)
        
        return P_jamming
    
    def sample_terminal_effectiveness(self, scenario: Dict, 
                                      target_type: str = 'armor') -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample FPV hit and kill probabilities.
        
        P_hit ~ Triangular(0.5, 0.75, 0.9), adjusted for wind
        P_kill | hit ~ Depends on target type:
            - Armored vehicles: Beta(7, 3) → mean ≈0.7
            - Soft targets: Beta(9, 1) → mean ≈0.9
            - Fortifications: Beta(5, 5) → mean ≈0.5
        
        Args:
            scenario: Scenario parameters
            target_type: 'armor', 'soft', or 'fortification'
            
        Returns:
            Tuple of (P_hit, P_kill) arrays of length n_iterations
        """
        V_wind = scenario['V_wind']
        
        # Base hit probability - Triangular distribution
        P_hit_base = np.random.triangular(0.5, 0.75, 0.9, self.n_iterations)
        
        # Wind degradation for high winds
        if V_wind > 20:
            wind_factor = 1.0 - 0.01 * V_wind
            P_hit = P_hit_base * wind_factor
        else:
            P_hit = P_hit_base
        
        P_hit = np.clip(P_hit, 0.0, 1.0)
        
        # Kill probability given hit - Beta distributions
        if target_type == 'armor':
            P_kill = np.random.beta(7, 3, self.n_iterations)
        elif target_type == 'soft':
            P_kill = np.random.beta(9, 1, self.n_iterations)
        elif target_type == 'fortification':
            P_kill = np.random.beta(5, 5, self.n_iterations)
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        return P_hit, P_kill
    
    def calculate_mission_success(self, P_attrition_M: np.ndarray, 
                                  P_jamming_FPV: np.ndarray,
                                  P_hit: np.ndarray,
                                  P_kill: np.ndarray) -> np.ndarray:
        """
        Calculate overall mission success probability.
        
        Core Formula:
        P_S = (1 - P_attrition,M) × (1 - P_jamming,FPV) × P_hit × P_kill
        
        Args:
            P_attrition_M: Mothership attrition probability
            P_jamming_FPV: FPV jamming probability
            P_hit: Hit probability
            P_kill: Kill probability given hit
            
        Returns:
            Mission success probability array
        """
        P_S = (1 - P_attrition_M) * (1 - P_jamming_FPV) * P_hit * P_kill
        
        # Sanity checks
        assert np.all((P_S >= 0) & (P_S <= 1)), "P_S must be between 0 and 1"
        
        return P_S
    
    def run_scenario(self, scenario: Dict, target_type: str = 'armor') -> pd.DataFrame:
        """
        Execute Monte Carlo simulation for a single scenario.
        
        Args:
            scenario: Scenario parameters
            target_type: Target type for kill probability
            
        Returns:
            DataFrame with iteration-level results
        """
        print(f"Running scenario: {scenario['name']} ({self.n_iterations} iterations)")
        
        # Calculate mission timeline (A2/AD exposure duration)
        T_transit, T_deployment, T_total = self.calculate_mission_timeline(scenario)
        
        # Sample probabilistic parameters
        P_detect, P_attrition_M_base = self.sample_mothership_vulnerability(scenario)
        
        # Apply time-based exposure penalty for A2/AD environment
        P_attrition_M = self.apply_time_exposure_penalty(P_attrition_M_base, T_total)
        
        P_jamming_FPV = self.sample_fpv_jamming(scenario)
        P_hit, P_kill = self.sample_terminal_effectiveness(scenario, target_type)
        
        # Calculate mission success
        P_S = self.calculate_mission_success(P_attrition_M, P_jamming_FPV, P_hit, P_kill)
        
        # Calculate derived metrics
        N_FPV = scenario['N_FPV']
        E_K = N_FPV * P_S * P_kill  # Expected kills
        P_survival = 1 - P_attrition_M
        efficiency = E_K / N_FPV
        
        # Store results (including A2/AD time exposure metrics)
        results_df = pd.DataFrame({
            'iteration': np.arange(self.n_iterations),
            'P_detect': P_detect,
            'P_attrition_M': P_attrition_M,
            'P_jamming_FPV': P_jamming_FPV,
            'P_hit': P_hit,
            'P_kill': P_kill,
            'P_S': P_S,
            'P_survival': P_survival,
            'E_K': E_K,
            'efficiency': efficiency,
            'T_transit_min': T_transit,
            'T_deployment_min': T_deployment,
            'T_total_A2AD_exposure_min': T_total,
            'P_survival': P_survival,
            'E_K': E_K,
            'efficiency': efficiency
        })
        
        results_df['scenario'] = scenario['name']
        
        return results_df
    
    def calculate_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for reporting in academic paper.
        
        Args:
            results_df: Results from run_scenario
            
        Returns:
            Dictionary of statistics formatted for paper
        """
        P_S = results_df['P_S'].values
        
        stats_dict = {
            # Central tendency
            'mean': np.mean(P_S),
            'median': np.median(P_S),
            'std': np.std(P_S),
            
            # Confidence intervals (95%)
            'ci_lower': np.percentile(P_S, 2.5),
            'ci_upper': np.percentile(P_S, 97.5),
            
            # Interquartile range
            'q1': np.percentile(P_S, 25),
            'q3': np.percentile(P_S, 75),
            'iqr': np.percentile(P_S, 75) - np.percentile(P_S, 25),
            
            # Derived metrics  
            'E_K_mean': results_df['E_K'].mean(),
            'E_K_ci_lower': np.percentile(results_df['E_K'], 2.5),
            'E_K_ci_upper': np.percentile(results_df['E_K'], 97.5),
            
            'P_survival_mean': results_df['P_survival'].mean(),
            'P_survival_ci_lower': np.percentile(results_df['P_survival'], 2.5),
            'P_survival_ci_upper': np.percentile(results_df['P_survival'], 97.5),
            
            'efficiency_mean': results_df['efficiency'].mean(),
            
            # A2/AD Time Exposure Metrics
            'T_transit_mean': results_df['T_transit_min'].mean(),
            'T_deployment_mean': results_df['T_deployment_min'].mean(),
            'T_total_exposure_mean': results_df['T_total_A2AD_exposure_min'].mean(),
        }
        
        return stats_dict
    
    def compare_to_baselines(self, mothership_P_S_mean: float, 
                            mothership_P_S_ci: Tuple[float, float]) -> pd.DataFrame:
        """
        Compare mothership performance to baseline strike methods.
        
        Args:
            mothership_P_S_mean: Mean mission success rate for mothership
            mothership_P_S_ci: (lower, upper) confidence interval
            
        Returns:
            DataFrame with comparison statistics
        """
        params = self.define_parameters()
        baselines = params['baselines']
        
        comparison_data = []
        
        # Mothership
        comparison_data.append({
            'method': 'Mothership + FPV',
            'mean_P_S': mothership_P_S_mean,
            'ci_lower': mothership_P_S_ci[0],
            'ci_upper': mothership_P_S_ci[1],
            'max_range_km': params['mothership']['R_M_max'] + params['fpv']['R_FPV_max']
        })
        
        # Ground-launched FPV
        P_S_ground = np.random.triangular(*baselines['ground_fpv']['P_S_dist'], self.n_iterations)
        comparison_data.append({
            'method': 'Ground-launched FPV',
            'mean_P_S': np.mean(P_S_ground),
            'ci_lower': np.percentile(P_S_ground, 2.5),
            'ci_upper': np.percentile(P_S_ground, 97.5),
            'max_range_km': baselines['ground_fpv']['R_max']
        })
        
        # Artillery
        P_S_arty = np.random.triangular(*baselines['artillery']['P_S_dist'], self.n_iterations)
        comparison_data.append({
            'method': 'Artillery',
            'mean_P_S': np.mean(P_S_arty),
            'ci_lower': np.percentile(P_S_arty, 2.5),
            'ci_upper': np.percentile(P_S_arty, 97.5),
            'max_range_km': baselines['artillery']['R_max']
        })
        
        # Precision Missile
        P_S_missile = np.random.triangular(*baselines['missile']['P_S_dist'], self.n_iterations)
        comparison_data.append({
            'method': 'Precision Missile',
            'mean_P_S': np.mean(P_S_missile),
            'ci_lower': np.percentile(P_S_missile, 2.5),
            'ci_upper': np.percentile(P_S_missile, 97.5),
            'max_range_km': baselines['missile']['R_max']
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate relative improvement
        ground_mean = comparison_df[comparison_df['method'] == 'Ground-launched FPV']['mean_P_S'].values[0]
        comparison_df['improvement_vs_ground'] = (comparison_df['mean_P_S'] - ground_mean) / ground_mean * 100
        
        return comparison_df
    
    def print_paper_ready_statistics(self, scenario_name: str, stats: Dict, 
                                     comparison_df: pd.DataFrame = None):
        """
        Print statistics in format ready for copy-paste into academic paper.
        
        Args:
            scenario_name: Name of scenario
            stats: Statistics dictionary from calculate_statistics
            comparison_df: Optional baseline comparison
        """
        print("\n" + "="*70)
        print(f"RESULTS FOR: {scenario_name}")
        print("="*70)
        
        print("\nFOR METHODOLOGY SECTION:")
        print("-" * 70)
        print(f"Monte Carlo simulation with {self.n_iterations:,} iterations per scenario was")
        print("employed to estimate mission success rates under uncertainty.")
        print("Probabilistic parameters were modeled using triangular, uniform, and")
        print("beta distributions based on literature estimates and operational data.")
        
        print("\nFOR RESULTS SECTION:")
        print("-" * 70)
        mean = stats['mean']
        ci_l = stats['ci_lower']
        ci_u = stats['ci_upper']
        
        print(f"Under {scenario_name.lower()} conditions, the mothership concept achieved")
        print(f"a mean mission success rate of {mean:.2f} (95% CI: {ci_l:.2f}-{ci_u:.2f})")
        
        if comparison_df is not None:
            ground_row = comparison_df[comparison_df['method'] == 'Ground-launched FPV'].iloc[0]
            print(f", compared to {ground_row['mean_P_S']:.2f} ", end="")
            print(f"(95% CI: {ground_row['ci_lower']:.2f}-{ground_row['ci_upper']:.2f})")
            print(f"for ground-launched FPVs, representing a ")
            
            improvement = comparison_df[comparison_df['method'] == 'Mothership + FPV']['improvement_vs_ground'].values[0]
            print(f"{improvement:+.1f}% improvement in mission success probability.")
        
        print(f"\nExpected target kills: {stats['E_K_mean']:.2f} ", end="")
        print(f"(95% CI: {stats['E_K_ci_lower']:.2f}-{stats['E_K_ci_upper']:.2f})")
        
        print(f"Mothership survival probability: {stats['P_survival_mean']:.2f} ", end="")
        print(f"(95% CI: {stats['P_survival_ci_lower']:.2f}-{stats['P_survival_ci_upper']:.2f})")
        
        print(f"\nA2/AD TIME EXPOSURE ANALYSIS:")
        print(f"Transit to release point: {stats['T_transit_mean']:.1f} minutes")
        print(f"FPV deployment time: {stats['T_deployment_mean']:.1f} minutes")
        print(f"Total A2/AD exposure: {stats['T_total_exposure_mean']:.1f} minutes")
        print(f"(Time-based attrition penalty: {(stats['T_total_exposure_mean'] * 0.01):.1%})")
        
        print("\n" + "="*70 + "\n")


def main():
    """
    Main execution function - Phase 1: A2/AD Penetration Scenarios
    """
    print("\n" + "="*70)
    print("MOTHERSHIP UAV A2/AD PENETRATION ANALYSIS")
    print("Monte Carlo Simulation of Deep Strike in Contested Environments")
    print("="*70 + "\n")
    
    # Initialize simulation
    sim = MonteCarloSimulation(n_iterations=10000, random_seed=42)
    
    # Define A2/AD scenarios based on threat density
    scenarios = {
        'LOW_THREAT': sim.define_scenario(
            name='Permissive Environment (Minimal A2/AD)',
            rho_AD=0.5,
            rho_EW=0.2,
            V_wind=10,
            visibility=1.0,
            N_FPV=5,
            guidance_type='fiber'
        ),
        
        'MEDIUM_THREAT': sim.define_scenario(
            name='Contested A2/AD Environment',
            rho_AD=2.0,
            rho_EW=1.0,
            V_wind=15,
            visibility=1.0,
            N_FPV=5,
            guidance_type='fiber'
        ),
        
        'HIGH_THREAT': sim.define_scenario(
            name='Denied A2/AD Environment',
            rho_AD=5.0,
            rho_EW=3.0,
            V_wind=25,
            visibility=0.7,  # Hazy helps mothership
            N_FPV=5,
            guidance_type='fiber'
        ),
    }
    
    # Store all results
    all_results = {}
    all_statistics = {}
    
    # Run each scenario
    for scenario_key, scenario in scenarios.items():
        results_df = sim.run_scenario(scenario, target_type='armor')
        all_results[scenario_key] = results_df
        
        # Calculate statistics
        stats = sim.calculate_statistics(results_df)
        all_statistics[scenario_key] = stats
        
        # Compare to baselines (for Medium Threat)
        comparison_df = None
        if scenario_key == 'MEDIUM_THREAT':
            comparison_df = sim.compare_to_baselines(
                stats['mean'],
                (stats['ci_lower'], stats['ci_upper'])
            )
        
        # Print paper-ready statistics
        sim.print_paper_ready_statistics(scenario['name'], stats, comparison_df)
    
    # Create outputs directory if it doesn't exist
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results to CSV
    combined_results = pd.concat(all_results.values(), ignore_index=True)
    output_file = os.path.join(output_dir, 'mothership_simulation_results.csv')
    combined_results.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Save summary statistics
    stats_rows = []
    for scenario_key, stats in all_statistics.items():
        row = {'scenario': scenarios[scenario_key]['name'], **stats}
        stats_rows.append(row)
    
    stats_df = pd.DataFrame(stats_rows)
    stats_file = os.path.join(output_dir, 'summary_statistics.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Summary statistics saved to: {stats_file}")
    
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("- Review results and statistics")
    print("- Run visualization script (coming in Phase 2)")
    print("- Proceed to sensitivity analysis")
    
    return sim, all_results, all_statistics, scenarios


if __name__ == "__main__":
    sim, results, statistics, scenarios = main()

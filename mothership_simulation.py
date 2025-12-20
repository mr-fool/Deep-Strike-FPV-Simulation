"""
Monte Carlo Simulation for Mothership UAV Deep Strike Analysis
================================================================
"""

import numpy as np
import pandas as pd
import os
import warnings
from typing import Dict, Tuple, List
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class MonteCarloSimulation:
    def __init__(self, n_iterations: int = 10000, random_seed: int = RANDOM_SEED):
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def define_parameters(self) -> Dict:
        return {
            'mothership': {'R_M_max': 120, 'V_M': 100, 'H_M_op': 2000, 'N_FPV_max': 5, 'RCS': 0.1},
            'fpv': {'R_FPV_max': 10, 'V_FPV': 150, 'M_warhead': 1.5},
            'baselines': {
                'ground_fpv': {'R_max': 8, 'P_S_dist': (0.5, 0.65, 0.8)},
                'artillery': {'R_max': 30, 'P_S_dist': (0.3, 0.4, 0.6)},
                'missile': {'R_max': 70, 'P_S_dist': (0.7, 0.85, 0.95)}
            }
        }

    def define_scenario(self, name: str, rho_AD: float, rho_EW: float, V_wind: float, visibility: float, N_FPV: int = 5, guidance_type: str = 'fiber') -> Dict:
        params = self.define_parameters()
        return {
            'name': name, 'rho_AD': rho_AD, 'rho_EW': rho_EW, 'V_wind': V_wind, 
            'visibility': visibility, 'N_FPV': N_FPV, 'guidance_type': guidance_type,
            'target_distance': params['mothership']['R_M_max'] * 0.6,
            'mothership': params['mothership'], 'fpv': params['fpv'], 'baselines': params['baselines']
        }

    def sample_mothership_vulnerability(self, scenario: Dict):
        H_M = scenario['mothership']['H_M_op']
        base_detect = 0.3 + 0.4 * (H_M / 2000) + 0.3 * scenario['rho_AD']
        P_detect = np.minimum(1.0, base_detect * scenario['visibility'])
        P_detect_arr = np.repeat(P_detect, self.n_iterations)
        P_attrition_given_detect = np.random.triangular(0.05, 0.15, 0.40, self.n_iterations)
        return P_detect_arr, P_detect_arr * P_attrition_given_detect

    def sample_fpv_jamming(self, scenario: Dict):
        guidance = scenario['guidance_type']
        if guidance == 'radio': P_jam_base = np.random.uniform(0.5, 0.7, self.n_iterations)
        elif guidance == 'fiber': P_jam_base = np.full(self.n_iterations, 0.05)
        else: P_jam_base = np.random.uniform(0.15, 0.25, self.n_iterations)
        return np.minimum(0.95, P_jam_base * (1.0 + 0.2 * scenario['rho_EW']))

    def sample_terminal_effectiveness(self, scenario: Dict, target_type: str = 'armor'):
        P_hit = np.random.triangular(0.5, 0.75, 0.9, self.n_iterations)
        if scenario['V_wind'] > 20: P_hit *= (1.0 - 0.01 * scenario['V_wind'])
        P_hit = np.clip(P_hit, 0, 1)
        if target_type == 'armor': P_kill = np.random.beta(7, 3, self.n_iterations)
        elif target_type == 'soft': P_kill = np.random.beta(9, 1, self.n_iterations)
        else: P_kill = np.random.beta(5, 5, self.n_iterations)
        return P_hit, P_kill

    def run_scenario(self, scenario: Dict, target_type: str = 'armor') -> pd.DataFrame:
        P_det, P_att_M = self.sample_mothership_vulnerability(scenario)
        P_jam = self.sample_fpv_jamming(scenario)
        P_hit, P_kill = self.sample_terminal_effectiveness(scenario, target_type)
        P_S = (1 - P_att_M) * (1 - P_jam) * P_hit * P_kill
        
        df = pd.DataFrame({
            'iteration': np.arange(self.n_iterations), 'P_detect': P_det, 'P_attrition_M': P_att_M,
            'P_jamming_FPV': P_jam, 'P_hit': P_hit, 'P_kill': P_kill, 'P_S': P_S,
            'P_survival': 1 - P_att_M, 'E_K': scenario['N_FPV'] * P_S * P_kill,
            'efficiency': (scenario['N_FPV'] * P_S * P_kill) / scenario['N_FPV'],
            'scenario': scenario['name']
        })
        return df

    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        P_S = df['P_S'].values
        return {
            'mean': np.mean(P_S), 'std': np.std(P_S), 
            'ci_lower': np.percentile(P_S, 2.5), 'ci_upper': np.percentile(P_S, 97.5),
            'E_K_mean': df['E_K'].mean(), 'P_survival_mean': df['P_survival'].mean()
        }

def main():
    sim = MonteCarloSimulation()
    scenarios = [
        sim.define_scenario('Low Threat', 0.5, 0.2, 10, 1.0),
        sim.define_scenario('Medium Threat', 2.0, 1.0, 15, 1.0),
        sim.define_scenario('High Threat', 5.0, 3.0, 25, 0.7)
    ]
    
    os.makedirs('outputs', exist_ok=True)
    results, stats_list = [], []
    
    for scen in scenarios:
        res_df = sim.run_scenario(scen)
        results.append(res_df)
        stats = sim.calculate_statistics(res_df)
        stats['scenario'] = scen['name']
        stats_list.append(stats)
        print(f"Completed: {scen['name']}")

    pd.concat(results).to_csv('outputs/mothership_simulation_results.csv', index=False)
    pd.DataFrame(stats_list).to_csv('outputs/summary_statistics.csv', index=False)
    print("\nData saved to /outputs directory.")

if __name__ == "__main__":
    main()
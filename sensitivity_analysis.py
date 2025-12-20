"""
Sensitivity Analysis for Mothership UAV Simulation
===================================================
"""

import numpy as np
import pandas as pd
import os
from mothership_simulation import MonteCarloSimulation

class SensitivityAnalyzer:
    def __init__(self, base_scenario: dict, n_iterations: int = 10000):
        self.base_scenario = base_scenario
        self.sim = MonteCarloSimulation(n_iterations=n_iterations)
        self.results = {}

    def run_full_analysis(self):
        # Sensitivity to AD Density
        print("Analyzing AD Density...")
        ad_results = []
        for rho in [0.5, 1.0, 2.0, 5.0]:
            scen = self.base_scenario.copy()
            scen['rho_AD'] = rho
            res = self.sim.run_scenario(scen, target_type='armor')
            stats = self.sim.calculate_statistics(res)
            ad_results.append({'value': rho, 'P_S_mean': stats['mean'], 'param': 'AD Density'})
        self.results['ad_density'] = pd.DataFrame(ad_results)

        # Sensitivity to Wind
        print("Analyzing Wind Speed...")
        wind_results = []
        for wind in [0, 15, 30, 45]:
            scen = self.base_scenario.copy()
            scen['V_wind'] = wind
            res = self.sim.run_scenario(scen, target_type='armor')
            stats = self.sim.calculate_statistics(res)
            wind_results.append({'value': wind, 'P_S_mean': stats['mean'], 'param': 'Wind Speed'})
        self.results['wind'] = pd.DataFrame(wind_results)

    def save_results(self):
        os.makedirs('outputs', exist_ok=True)
        combined = pd.concat(self.results.values())
        combined.to_csv('outputs/sensitivity_analysis_results.csv', index=False)
        print("Sensitivity data saved to outputs/sensitivity_analysis_results.csv")

def main():
    sim = MonteCarloSimulation()
    base = sim.define_scenario('Medium Threat', 2.0, 1.0, 15, 1.0)
    analyzer = SensitivityAnalyzer(base)
    analyzer.run_full_analysis()
    analyzer.save_results()

if __name__ == "__main__":
    main()
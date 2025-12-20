# Deep-Strike FPV Simulation


**Simulation Type:** Monte Carlo Analysis (10,000 iterations per scenario) 

---

## Overview

This simulation models the operational effectiveness of carrier UAVs (motherships) deploying First-Person View (FPV) drones for deep-strike missions. It analyzes the "kill chain" in contested environments, specifically focusing on Anti-Access/Area Denial (A2AD) threats. The analysis compares mothership performance across various threat levels and identifies environmental and tactical sensitivities.

### Key Features
* **Stochastic Kill-Chain Modeling:** Uses probabilistic distributions (triangular, beta, uniform) to model detection, attrition, jamming, and lethality.
* **Layered Defense Analysis:** Simulates mission success ($P_S$) and mothership survival ($P_{surv}$) against varied Air Defense (AD) and Electronic Warfare (EW) densities.
* **Environmental Degradation:** Models the impact of wind speed and visibility on FPV flight stability and terminal guidance.
* **Sensitivity Engine:** Utilizes One-At-a-Time (OAT) methodology to isolate critical parameters affecting mission success.
* **Publication-Quality Outputs:** Generates 300 DPI visualizations and statistical summaries ready for journal submission.

---

## Project Structure

```
deep-strike-fpv-simulation/
├── mothership_simulation.py    # Main Monte Carlo engine (Phase 1)
├── visualization.py             # Publication figure generation (Phase 2)
├── sensitivity_analysis.py      # Parameter sensitivity analysis (Phase 3)
├── README.md                    # This file
├── OUTPUTS/
│   ├── mothership_simulation_results.csv      # Raw iteration data
│   ├── summary_statistics.csv                 # Summary stats for paper
│   ├── sensitivity_analysis_results.csv       # Sensitivity data
│   ├── fig1_scenario_comparison_boxplot.png   # Box plot
│   ├── fig2_cdf_comparison.png                # Cumulative distributions
│   ├── fig3_baseline_comparison.png           # vs traditional methods
│   ├── fig4_convergence_analysis.png          # MC convergence
│   ├── fig5_parameter_distributions.png       # Probability distributions
│   ├── fig6_tornado_diagram.png               # Sensitivity ranking
│   ├── fig7_sensitivity_curves.png            # Parameter impacts
│   └── table1_summary_statistics.png          # Summary table
```

---

## Quick Start

### 1. Run Complete Analysis (All Phases)

```bash
# Phase 1: Core Monte Carlo Simulation (Scenarios 1-3)
python mothership_simulation.py

# Phase 2: Generate Publication Figures
python visualization.py

# Phase 3: Sensitivity Analysis
python sensitivity_analysis.py
```

### 2. View Results

All outputs are saved to the current directory:
- **CSV files** - Import into Excel/R/SPSS for further analysis
- **PNG figures** - 300 DPI, ready for journal submission
- **Console output** - Copy-paste ready text for methodology/results sections

---

## Simulation Methodology

### Core Equation

The mission success probability (P_S) is calculated as:

```
P_S = (1 - P_attrition,M) * (1 - P_jamming,FPV) * P_hit * P_kill
```

Where:
- **P_attrition,M** - Probability mothership is destroyed by air defense
- **P_jamming,FPV** - Probability FPV is jammed/disabled by EW
- **P_hit** - Probability FPV successfully strikes target
- **P_kill** - Probability target is destroyed given hit

### Probabilistic Models

**Mothership Vulnerability:**
- Detection: P_detect = min(1.0, 0.3 + 0.4*(H/2000) + 0.3*ρ_AD) * visibility
- Attrition | detected ~ Triangular(0.05, 0.15, 0.40)

**FPV Jamming:**
- Radio-controlled: P_jamming ~ Uniform(0.5, 0.7)
- Fiber optic: P_jamming = 0.05 (fixed)
- AI-enabled: P_jamming ~ Uniform(0.15, 0.25)

**Terminal Effectiveness:**
- P_hit ~ Triangular(0.5, 0.75, 0.9), adjusted for wind
- P_kill ~ Beta distributions based on target type:
  - Armored vehicles: Beta(7, 3) -> mean ~ 0.7
  - Soft targets: Beta(9, 1) -> mean ~ 0.9
  - Fortifications: Beta(5, 5) -> mean ~ 0.5

---

## Scenarios Tested

### Phase 1 Scenarios (Current Implementation)

| Scenario | AD Density | EW Density | Wind | Visibility | Purpose |
|----------|-----------|-----------|------|------------|---------|
| **Low Threat** | 0.5/100km² | 0.2/50km² | 10 km/h | Clear | Baseline capability |
| **Medium Threat** | 2.0/100km² | 1.0/50km² | 15 km/h | Clear | Realistic operations |
| **High Threat** | 5.0/100km² | 3.0/50km² | 25 km/h | Hazy | Survivability test |

### Phase 4 Scenarios (Future Implementation)

- **Deep Strike** - Target at maximum range (120 km)
- **Saturation Attack** - Multiple targets, maximum FPV load

---

## Modifying Parameters

### Example: Change Mothership Range

Edit `mothership_simulation.py`:

```python
# Find define_parameters() function, modify:
'mothership': {
    'R_M_max': 150,  # Change from 120 to 150 km
    'V_M': 100,
    'H_M_op': 2000,
    # ...
}
```

### Example: Add New Scenario

```python
scenarios = {
    'CUSTOM': sim.define_scenario(
        name='Custom Scenario',
        rho_AD=3.0,              # Your AD density
        rho_EW=2.0,              # Your EW density
        V_wind=20,               # Your wind speed
        visibility=0.85,         # Your visibility
        N_FPV=7,                 # Your FPV count
        guidance_type='ai'       # 'radio', 'fiber', or 'ai'
    ),
}
```

### Example: Change Number of Iterations

```python
# Modify in main():
sim = MonteCarloSimulation(n_iterations=5000, random_seed=42)
```

**Note:** 10,000 iterations ensure stable results (standard for Monte Carlo).  
Reducing to 5,000 speeds up testing but increases variance.

---

## Understanding the Results

### For Your Methodology Section

Use the console output directly:

```
"Monte Carlo simulation with 10,000 iterations per scenario was employed 
to estimate mission success rates under uncertainty. Probabilistic parameters 
were modeled using triangular, uniform, and beta distributions based on 
literature estimates and operational data from the Russia-Ukraine conflict."
```

### For Your Results Section

The simulation outputs paper-ready statistics:

```
"Under medium threat conditions, the mothership concept achieved a mean 
mission success rate of 0.38 (95% CI: 0.20-0.56), compared to 0.65 
(95% CI: 0.54-0.77) for ground-launched FPVs."
```

### Key Findings

**Important Insight:**  
The mothership shows **lower P_S** than ground-launched FPVs but offers **16x range extension** (130 km vs 8 km). This is the strategic value proposition - enabling deep strikes beyond direct fire range.

**Trade-off:**
- Ground FPV: Higher P_S (~0.65), limited range (8 km)
- Mothership: Lower P_S (~0.38), extended range (130 km)
- Artillery: Medium P_S (~0.40), medium range (30 km)
- Missiles: High P_S (~0.85), longest range (70 km), high cost

---

## Sensitivity Analysis Results

The tornado diagram (fig6) shows which parameters most impact P_S:

**Most Sensitive Parameters (from Phase 3):**
1. Guidance Type (Radio vs Fiber vs AI)
2. Air Defense Density (ρ_AD)
3. Operational Altitude

**Less Sensitive Parameters:**
4. Wind Speed (only affects hit probability)
5. Mothership Range (doesn't directly affect P_S)
6. FPV Count (affects total kills, not individual P_S)

### For Discussion Section

```
"Sensitivity analysis revealed that FPV guidance type and air defense 
density were the most critical parameters affecting mission success. 
Fiber-optic guidance improved P_S by 45% over radio-controlled systems 
due to reduced jamming vulnerability. Mothership operational altitude 
showed moderate sensitivity, with higher altitudes increasing detection 
probability but improving standoff distance."
```

---

## Validation & Quality Checks

The simulation includes built-in validation:

### 1. Sanity Checks
-  All probabilities constrained to [0, 1]
-  Expected kills <= Number of FPVs deployed
-  Detection probability <= 1.0

### 2. Convergence Test (fig4)
- Mean stabilizes after ~2,000 iterations
- 10,000 iterations ensures robust results

### 3. Statistical Verification
- 95% confidence intervals calculated
- Results reproducible (fixed random seed)

---

## Troubleshooting

### Issue: Results seem unrealistic

**Solution:** Check baseline parameters against literature:
- Compare P_S to historical FPV strike data
- Verify AD density matches theater intelligence
- Ensure distributions match documented capabilities

### Issue: Want to test extreme scenarios

**Solution:** Create custom scenario with modified parameters:
```python
# Example: Very high threat
scenario = sim.define_scenario(
    name='Extreme Threat',
    rho_AD=10.0,    # Very dense AD
    rho_EW=5.0,     # Heavy jamming
    V_wind=50,      # Storm conditions
    visibility=0.3  # Poor visibility
)
```

### Issue: Need more detailed output

**Solution:** Access iteration-level data:
```python
# After running simulation:
results_df = pd.read_csv('mothership_simulation_results.csv')

# Filter specific scenario
medium_threat = results_df[results_df['scenario'] == 'Medium Threat']

# Analyze distributions
print(medium_threat['P_S'].describe())
print(medium_threat[['P_attrition_M', 'P_jamming_FPV', 'P_hit']].corr())
```

---

## Advanced Usage

### Custom Probability Distributions

Modify sampling functions in `mothership_simulation.py`:

```python
# Example: Change from Triangular to Normal distribution
def sample_terminal_effectiveness(self, scenario: Dict):
    # Original:
    # P_hit_base = np.random.triangular(0.5, 0.75, 0.9, self.n_iterations)
    
    # New - Normal distribution:
    P_hit_base = np.random.normal(0.75, 0.1, self.n_iterations)
    P_hit_base = np.clip(P_hit_base, 0.0, 1.0)  # Constrain to [0,1]
    # ...
```

### Multiple Random Seeds

Test result stability:

```python
seeds = [42, 123, 456, 789, 1011]
for seed in seeds:
    sim = MonteCarloSimulation(n_iterations=10000, random_seed=seed)
    # Run scenarios and compare results
```

Results should vary by <±0.02 between seeds.

---

## Citation

When using this simulation in your paper, consider citing the methodology:

```
The Monte Carlo simulation employed 10,000 iterations per scenario 
to estimate mission success probability distributions. Probabilistic 
parameters were modeled using triangular distributions for terminal 
effectiveness (P_hit ~ Tri(0.5, 0.75, 0.9)), beta distributions for 
kill probability (P_kill ~ Beta(α,β) varying by target type), and 
uniform distributions for jamming susceptibility based on guidance 
type. Sensitivity analysis used one-at-a-time (OAT) methodology to 
isolate parameter impacts.
```

---

## Future Enhancements (Phase 4+)

Planned features for next iteration:
- [ ] Deep Strike scenario (maximum range operations)
- [ ] Saturation Attack scenario (multiple simultaneous targets)
- [ ] Route optimization using Dijkstra/A* algorithms
- [ ] Multi-target engagement sequencing
- [ ] Cost-effectiveness analysis
- [ ] Monte Carlo variance reduction techniques

---

## Technical Details

### Dependencies

```python
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
openpyxl>=3.0.0  # For Excel file reading
```

### Performance

- **Phase 1 (3 scenarios):** ~30-60 seconds
- **Phase 2 (visualizations):** ~10-15 seconds  
- **Phase 3 (sensitivity analysis):** ~5-10 minutes

**Total runtime:** ~6-12 minutes for complete analysis

### System Requirements

- Python 3.7+
- 4GB RAM minimum
- Any modern OS (Windows/Mac/Linux)

---

## Sections Questions?

Key sections:

- **Parameter definitions:** `define_parameters()` function
- **Probability sampling:** `sample_*()` functions
- **Core calculation:** `calculate_mission_success()` function
- **Statistics:** `calculate_statistics()` function



# Deep-Strike FPV Simulation - A2/AD Penetration Analysis

**Simulation Type:** Monte Carlo Analysis with A2/AD Time Exposure Modeling (10,000 iterations per scenario)

---

## Overview

This simulation models the operational effectiveness of carrier UAVs (motherships) that deploy First-Person View (FPV) drones for deep strike missions in Anti-Access/Area Denial (A2/AD) environments. The analysis incorporates mission timeline calculations, time-based exposure risk, and compares mothership performance across threat scenarios and against traditional strike methods.

### Key Features
- Monte Carlo simulation with 10,000 iterations per scenario
- **A2/AD time exposure modeling** - cumulative risk during penetration
- Mission timeline analysis (transit, deployment, total exposure)
- Probabilistic modeling using triangular, uniform, and beta distributions
- 3 A2/AD threat scenarios (Permissive, Contested, Denied)
- Sensitivity analysis identifying critical parameters
- Publication-quality visualizations (300 DPI, ready for journal submission)
- Baseline comparisons with traditional strike methods

---

## Project Structure

```
deep-strike-fpv-simulation/
├── mothership_simulation.py    # Main Monte Carlo engine with A2/AD analysis
├── visualization.py             # Publication figure generation
├── sensitivity_analysis.py      # Parameter sensitivity analysis
├── README.md                    # This file
├── outputs/
│   ├── mothership_simulation_results.csv      # Raw iteration data
│   ├── summary_statistics.csv                 # Summary stats for paper
│   ├── sensitivity_analysis_results.csv       # Sensitivity data
│   ├── fig1_scenario_comparison_boxplot.png   # Box plot across A2/AD scenarios
│   ├── fig2_cdf_comparison.png                # Cumulative distributions
│   ├── fig3_baseline_comparison.png           # vs traditional methods
│   ├── fig4_convergence_analysis.png          # Monte Carlo convergence
│   ├── fig5_parameter_distributions.png       # Probability distributions
│   ├── fig6_tornado_diagram.png               # Sensitivity ranking
│   ├── fig7_sensitivity_curves.png            # Parameter impacts
│   └── table1_summary_statistics.png          # Summary table
```

---

## Quick Start

### 1. Run Complete Analysis (All Phases)

```bash
# Phase 1: Core Monte Carlo Simulation with A2/AD Analysis
python mothership_simulation.py

# Phase 2: Generate Publication Figures
python visualization.py

# Phase 3: Sensitivity Analysis
python sensitivity_analysis.py
```

### 2. View Results

All outputs are saved to the `outputs/` directory:
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
- **P_attrition,M** - Probability mothership is destroyed by air defense (includes time exposure penalty)
- **P_jamming,FPV** - Probability FPV is jammed/disabled by EW
- **P_hit** - Probability FPV successfully strikes target
- **P_kill** - Probability target is destroyed given hit

### A2/AD Time Exposure Model (NEW)

**Mission Timeline:**
```
T_total = T_transit + T_deployment

T_transit = (target_distance * 0.8) / V_mothership  # Time to FPV release point
T_deployment = (N_FPV * 30 seconds) / 60            # Time deploying FPVs
```

**Time-Based Attrition Penalty:**
```
P_attrition_modified = P_attrition_base * (1 + 0.01 * T_total_minutes)
```

This models the cumulative detection and engagement risk during extended loiter in A2/AD zones. Each minute of exposure increases attrition probability by 1%, reflecting multiple detection opportunities by adversary systems.

### Probabilistic Models

**Mothership Vulnerability:**
- Detection: P_detect = min(1.0, [0.3 + 0.4*(H/2000) + 0.3*ρ_AD] × RCS_factor × visibility)
  - where RCS_factor = (RCS / 0.1)^0.25 (radar equation fourth-root relationship)
  - Baseline RCS = 0.1 m² (moderately low-observable design)
- Attrition | detected ~ Triangular(0.05, 0.15, 0.40)
- Time exposure penalty: +1% per minute in A2/AD envelope
- **Note:** In contested A2/AD, detection often saturates at 100% regardless of RCS

**FPV Jamming:**
- Radio-controlled: P_jamming ~ Uniform(0.5, 0.7)
- Fiber optic: P_jamming = 0.05 (fixed)
- AI-enabled: P_jamming ~ Uniform(0.15, 0.25)

**Terminal Effectiveness:**
- P_hit ~ Triangular(0.5, 0.75, 0.9), adjusted for wind
- P_kill ~ Beta distributions based on target type:
  - Armored vehicles: Beta(7, 3) → mean ~ 0.7
  - Soft targets: Beta(9, 1) → mean ~ 0.9
  - Fortifications: Beta(5, 5) → mean ~ 0.5

---

## A2/AD Scenarios Tested

### Current Implementation

| Scenario | AD Density | EW Density | Wind | Visibility | A2/AD Classification |
|----------|-----------|-----------|------|------------|---------------------|
| **Permissive Environment** | 0.5/100km² | 0.2/50km² | 10 km/h | Clear | Minimal A2/AD |
| **Contested A2/AD** | 2.0/100km² | 1.0/50km² | 15 km/h | Clear | Typical Russian deployment |
| **Denied A2/AD** | 5.0/100km² | 3.0/50km² | 25 km/h | Hazy | Near-peer saturation |

**Representative Conditions:**
- Permissive: Limited air defense, minimal jamming (e.g., rear areas, low-intensity conflict)
- Contested: Moderate A2/AD density representative of Russian deployments in Ukraine
- Denied: Dense, integrated A2/AD networks (e.g., peer adversary homeland defense)

---

## Modifying Parameters

### Example: Change Target Distance (Affects Time Exposure)

Edit `mothership_simulation.py`:

```python
# Find define_parameters() function, modify:
'target_distance': 90,  # Change from 120 to 90 km
# This reduces transit time and A2/AD exposure
```

### Example: Add New A2/AD Scenario

```python
scenarios = {
    'CUSTOM': sim.define_scenario(
        name='Custom A2/AD Environment',
        rho_AD=3.5,              # Your AD density
        rho_EW=2.0,              # Your EW density
        V_wind=20,               # Your wind speed
        visibility=0.85,         # Your visibility
        N_FPV=7,                 # Your FPV count
        guidance_type='fiber'    # 'radio', 'fiber', or 'ai'
    ),
}
```

### Example: Test Faster Mothership (Reduces Exposure)

```python
'mothership': {
    'R_M_max': 120,
    'V_M': 150,  # Change from 100 to 150 km/h
    # This reduces transit time by 33%, lowering time penalty
}
```

### Example: Change Time Penalty Model

```python
# In apply_time_exposure_penalty() function:
# Current: 1% per minute
exposure_factor = 1.0 + (0.01 * T_total)

# More aggressive: 2% per minute
exposure_factor = 1.0 + (0.02 * T_total)

# Conservative: 0.5% per minute
exposure_factor = 1.0 + (0.005 * T_total)
```

---

## Validation & Quality Checks

The simulation includes built-in validation:

### 1. Sanity Checks
- All probabilities constrained to [0, 1]
- Expected kills ≤ Number of FPVs deployed
- Detection probability ≤ 1.0
- Time exposure ≥ deployment time

### 2. Convergence Test (fig4)
- Mean stabilizes after ~2,000 iterations
- 10,000 iterations ensures robust results
- 95% confidence intervals narrow appropriately

### 3. Statistical Verification
- 95% confidence intervals calculated
- Results reproducible (fixed random seed = 42)
- Time metrics consistent across scenarios

### 4. Physical Realism Checks
- Transit time matches distance/speed calculations
- Deployment time = 30 sec/FPV (realistic mechanical constraint)
- Time penalties scale linearly with exposure duration

---

## Troubleshooting

### Issue: Time exposure seems too high/low

**Check:**
```python
# Verify target distance
scenario['target_distance']  # Should be 60-150 km range

# Verify mothership speed
scenario['mothership']['V_M']  # Should be 80-150 km/h

# Calculate expected transit
expected_transit = (distance * 0.8) / speed * 60  # minutes
```

### Issue: Want to test extreme A2/AD conditions

**Solution:** Create high-threat scenario:
```python
scenario = sim.define_scenario(
    name='Extreme A2/AD Saturation',
    rho_AD=10.0,    # Very dense AD (peer adversary core)
    rho_EW=5.0,     # Heavy jamming
    V_wind=50,      # Storm conditions
    visibility=0.3, # Poor visibility
    guidance_type='radio'  # Worst case
)
```

### Issue: Need iteration-level time data

**Solution:** Access detailed results:
```python
results_df = pd.read_csv('outputs/mothership_simulation_results.csv')

# Analyze time distributions
print(results_df[['T_transit_min', 'T_deployment_min', 
                  'T_total_A2AD_exposure_min']].describe())

# Plot time vs success
import matplotlib.pyplot as plt
plt.scatter(results_df['T_total_A2AD_exposure_min'], 
            results_df['P_S'])
plt.xlabel('Total A2/AD Exposure (minutes)')
plt.ylabel('Mission Success Probability')
plt.show()
```

---

## Advanced Usage

### Analyze Time-Success Correlation

```python
import pandas as pd
import scipy.stats as stats

results = pd.read_csv('outputs/mothership_simulation_results.csv')

# Does longer exposure correlate with lower success?
correlation = stats.pearsonr(results['T_total_A2AD_exposure_min'], 
                             results['P_S'])
print(f"Time-Success Correlation: {correlation[0]:.3f}, p={correlation[1]:.4f}")

# Expected: Negative correlation (more time = higher attrition = lower P_S)
```

### Compare Speed Scenarios

```python
# Run same scenario with different speeds
speeds = [80, 100, 120, 150]
results = {}

for speed in speeds:
    scenario['mothership']['V_M'] = speed
    df = sim.run_scenario(scenario)
    results[speed] = {
        'mean_P_S': df['P_S'].mean(),
        'mean_exposure': df['T_total_A2AD_exposure_min'].mean()
    }

# Plot speed-exposure-success relationship
```

### Optimize Payload Size

```python
# Find optimal FPV count balancing kills vs exposure
fpv_counts = range(2, 11)
results = []

for n in fpv_counts:
    scenario['N_FPV'] = n
    df = sim.run_scenario(scenario)
    results.append({
        'N_FPV': n,
        'E_K': df['E_K'].mean(),
        'T_deployment': df['T_deployment_min'].mean(),
        'kills_per_minute': df['E_K'].mean() / df['T_deployment_min'].mean()
    })

# Identify diminishing returns point
```

---


## Technical Details

### Dependencies

```python
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

### Performance

- **Phase 1 (3 A2/AD scenarios):** ~30-60 seconds
- **Phase 2 (6 figures + 1 table):** ~10-15 seconds  
- **Phase 3 (sensitivity analysis):** ~5-10 minutes

**Total runtime:** ~6-12 minutes for complete analysis


---

## Key Simulation Components

Reference guide for the codebase:

- **Parameter definitions:** `define_parameters()` function
- **Probability sampling:** `sample_*()` functions
- **Time calculations:** `calculate_mission_timeline()` function
- **Time penalty:** `apply_time_exposure_penalty()` function
- **Core success calculation:** `calculate_mission_success()` function
- **Statistics:** `calculate_statistics()` function

For detailed methodology, see inline comments in `mothership_simulation.py`.

---
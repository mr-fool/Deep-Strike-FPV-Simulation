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
- Detection: P_detect = min(1.0, 0.3 + 0.4*(H/2000) + 0.3*ρ_AD) * visibility
- Attrition | detected ~ Triangular(0.05, 0.15, 0.40)
- Time exposure penalty: +1% per minute in A2/AD envelope

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

## Key Results Summary

### Mission Success Probability

| Scenario | Mean P_S | 95% CI | Expected Kills | Survival Prob | A2/AD Exposure |
|----------|----------|--------|----------------|---------------|----------------|
| Permissive | 0.37 | 0.19-0.55 | 1.83 | 0.77 | 37.1 min |
| Contested | 0.34 | 0.17-0.53 | 1.71 | 0.73 | 37.1 min |
| Denied | 0.25 | 0.13-0.39 | 1.26 | 0.73 | 37.1 min |

### Time Exposure Analysis

**Typical 120km Mission (Contested A2/AD):**
- Transit to release point: **57.6 minutes**
- FPV deployment time: **2.5 minutes** (5 FPVs @ 30 sec each)
- **Total A2/AD exposure: 60.1 minutes**
- Time-based attrition penalty: **+60%**

**Key Insight:** Extended loiter in contested airspace significantly increases cumulative detection risk, with time penalty growing linearly (1% per minute).

### Range vs. Time Trade-off

| Target Distance | Transit Time | Total Exposure | Cumulative Penalty |
|----------------|--------------|----------------|-------------------|
| 60 km | 28.8 min | 31.3 min | +31% attrition |
| 90 km | 43.2 min | 45.7 min | +46% attrition |
| 120 km | 57.6 min | 60.1 min | +60% attrition |
| 150 km | 72.0 min | 74.5 min | +75% attrition |

**Operational Implication:** Deeper targets impose significantly higher time-based risk. Optimal engagement range is 60-90 km for acceptable exposure duration.

---

## Understanding the Results

### For Your Methodology Section

```
Monte Carlo simulation with 10,000 iterations per scenario was employed to 
estimate mission success rates under uncertainty. Probabilistic parameters 
were modeled using triangular, uniform, and beta distributions based on 
documented capabilities and operational data from the Russia-Ukraine conflict.

Mission timelines were calculated to account for cumulative detection and 
engagement risk during A2/AD penetration. Time-based attrition penalties 
applied a 1% probability increase per minute of exposure in contested 
airspace, reflecting the reality that extended loiter provides multiple 
opportunities for adversary detection and engagement.
```

### For Your Results Section

```
MISSION SUCCESS PROBABILITY:
Against contested A2/AD environments (2 AD systems/100km², moderate jamming) 
representative of Russian deployments in Ukraine, the mothership concept 
achieved a mean mission success rate of 0.34 (95% CI: 0.17-0.53), compared 
to 0.65 (95% CI: 0.54-0.77) for ground-launched FPVs.

A2/AD TIME EXPOSURE ANALYSIS:
Penetration of contested A2/AD environments to 120km targets required mean 
exposure times of 60.1 minutes, comprising 57.6 minutes transit and 2.5 
minutes FPV deployment. Extended loiter in defended airspace imposed 
cumulative detection and engagement opportunities, modeled as a 1% attrition 
increase per minute, resulting in a 60% time-based penalty applied to base 
attrition probabilities.

RANGE-TIME TRADE-OFF:
Shorter-range operations (60km) reduced total A2/AD exposure to 31 minutes, 
decreasing time-based penalties to 31%. This demonstrates the range-time 
trade-off inherent in deep strike operations against layered defenses, 
suggesting mothership platforms are optimized for intermediate-depth targets 
(60-90km) where extended exposure risk remains acceptable.
```

### Critical Insight

**Strategic Trade-off:**  
The mothership shows **lower P_S** than ground-launched FPVs (0.34 vs 0.65) but offers **16x range extension** (130 km vs 8 km). The value proposition is **operational reach** enabling strikes against high-value targets beyond direct fire range, not higher success probability.

**Time-Speed-Payload Trade-off:**
- Faster motherships (150 km/h vs 100 km/h) reduce A2/AD exposure 33%
- Larger payloads (10 FPVs) increase deployment time, extending exposure 2 minutes
- Deeper targets (150 km) impose 75-minute exposure, 75% attrition penalty

**Comparison to Traditional Methods:**
- Ground FPV: Higher P_S (~0.65), limited range (8 km), no A2/AD penetration
- Mothership: Medium P_S (~0.34), extended range (130 km), 60-min A2/AD exposure
- Artillery: Medium P_S (~0.40), medium range (30 km), reveals battery position
- Missiles: High P_S (~0.85), longest range (70 km), high cost ($150k/round)

---

## Sensitivity Analysis Results

The tornado diagram (fig6) and sensitivity curves (fig7) show parameter impacts on P_S:

### Most Critical Parameters

**1. Guidance Type** (Dominant Factor)
- Radio-controlled: P_S = 0.11 (heavy jamming vulnerability)
- **Fiber-optic: P_S = 0.38** (jam-resistant, baseline)
- AI-enabled: P_S = 0.31 (moderate jamming)
- **Impact: 237% improvement** (fiber vs radio)

**2. Wind Speed** (Environmental)
- 0-20 km/h: P_S ≈ 0.38 (stable performance)
- 30 km/h: P_S = 0.26 (degraded, 30% reduction)
- 40 km/h: P_S = 0.23 (severe degradation)

**3. Air Defense Density** (Moderate Impact)
- Low (0.5/100km²): P_S = 0.39, Survival = 0.83
- Medium (2.0/100km²): P_S = 0.38, Survival = 0.80
- High (5.0/100km²): P_S = 0.38, Survival = 0.80

### Less Sensitive Parameters

**4. Operational Altitude** (Minimal Direct Impact)
- 1000m to 3000m: P_S varies by <1%
- Higher altitude increases detection but improves standoff

**5. Mothership Range** (No Direct Impact on P_S)
- Affects mission feasibility, not individual success rate
- Longer range = more exposure time (indirect impact)

**6. FPV Count** (Affects Total Kills, Not P_S)
- 2 FPVs: E[K] = 0.75
- 5 FPVs: E[K] = 1.89
- 10 FPVs: E[K] = 3.77
- P_S remains constant (~0.38) regardless of payload

### For Discussion Section

```
SENSITIVITY ANALYSIS INSIGHTS:

Guidance type emerged as the dominant parameter, with fiber-optic control 
improving mission success probability 237% over radio-controlled systems 
(P_S = 0.38 vs 0.11) due to reduced jamming vulnerability. This finding 
underscores the critical importance of jam-resistant communications in 
A2/AD environments.

Wind conditions above 20 km/h significantly degraded performance, reducing 
P_S by 30-40%, highlighting the need for weather-dependent mission planning. 
Air defense density showed moderate sensitivity, with survivability decreasing 
from 83% in permissive environments to 80% in contested zones.

Surprisingly, mothership operational altitude (1000-3000m) showed minimal 
direct impact on P_S (<1% variation), suggesting commanders have flexibility 
in altitude selection based on terrain masking and standoff requirements 
without sacrificing mission success probability.

The time exposure analysis revealed that payload size creates a tactical 
dilemma: larger FPV loads (10 vs 5) double strike volume but extend 
vulnerable deployment time from 2.5 to 5.0 minutes, increasing total A2/AD 
exposure and cumulative detection risk.
```

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

## Questions?

Key simulation components:

- **Parameter definitions:** `define_parameters()` function
- **Probability sampling:** `sample_*()` functions
- **Time calculations:** `calculate_mission_timeline()` function
- **Time penalty:** `apply_time_exposure_penalty()` function
- **Core success calculation:** `calculate_mission_success()` function
- **Statistics:** `calculate_statistics()` function

For detailed methodology, see inline comments in `mothership_simulation.py`.

---
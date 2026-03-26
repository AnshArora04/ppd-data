# Begenau, Liang & Siriwardane (2026, RFS) — "The Rise of Alternatives"
## Machine-Readable Replication Guide

---

## 1. DATA SOURCES

### 1.1 Primary: Public Plans Database (PPD)
- **Maintainer:** Center for Retirement Research (CRR), Boston College
- **URL:** https://publicplansdata.org/
- **Release used:** January 2023 (Covid analysis uses December 2024 release)
- **Unit of observation:** Pension *system*-year (plans pooled within systems)
- **Coverage:** 89% of pensions by count, 94% by assets have target shares
- **Years:** 2001–2021 (main analysis); 2019–2023 (Covid section)
- **Key variables available:**
  - `target_share_*` : target portfolio weights (cash, fixed income, public equity, PE, RE, HF, commodities, misc alts, other)
  - `actual_share_*` : actual portfolio weights
  - `gasb25_funding` : GASB 25 funding ratio (actuarial assets / actuarial liabilities)
  - `hurdle_rate` : assumed long-run return on assets (= liability discount rate)
  - `aum` : assets under management ($)
  - `required_contribution_payroll` : required actuarial contribution / payroll
  - `deductions_payroll` : total deductions / payroll
  - `annual_return` : annual net investment return
  - `pct_retired` : fraction of members who are retired
  - `n_members` : total members

### 1.2 Historical 1990s data: PENDAT Surveys
- **Administered by:** Public Pension Coordinating Council
- **Waves:** Multiple during 1990s (key years: 1990, 1996, 1998)
- **Aggregated to:** State level (asset-weighted)
- **Key variables:**
  - `target_equity_share_1990`, `target_equity_share_1996`
  - `investment_restrictions_1998` : indicator for constitutional investment restrictions
  - `min_qualifications_1998` : indicator for min professional qualifications
  - `political_board_fraction_2002` : fraction political board members (from Andonov et al. 2018)

### 1.3 Capital Market Assumptions (CMAs)
- **Source:** Hand-collected from 15 major consultancies (obtained under confidentiality)
- **Coverage:** ~78% of U.S. pension assets; 63% of system-year observations
- **Time range:** 2003–2021
- **Variables per consultant c, asset class a, year t:**
  - `mu_A[c,a,t]` : expected excess return on alternatives
  - `mu_E[c,t]` : expected excess return on public equities
  - `beta[c,a,t]` : CAPM beta of alternatives vs public equities
  - `sigma2_E[c,t]` : variance of public equity returns
  - `alpha[c,a,t]` : computed alpha = mu_A - beta * mu_E  (in bps)
- **Asset classes:** Private Equity (PE), Real Estate (RE), Hedge Funds (HF)

### 1.4 Consultant Identity
- **Collection method:** Hand-collected from CAFRs + FOIA requests
- **Coverage:** >98% of system-year observations
- **N distinct consultants:** 57 (with multiple pension clients in a given year)

### 1.5 Investment Advisory Representatives (IARs)
- **Source:** Updated version of Egan, Matvos & Seru (2019) dataset
- **Used for:** Instrumental variable (consultant geographic proximity)
- **Variable:** `consultant_active_in_state[c,s,t]` : indicator, consultant c has ≥1 IAR in state s at time t

### 1.6 Other Institutions (for Section 5.3 / Figure 7)
| Institution | Source | Variable |
|---|---|---|
| U.S. Corporate DB | Milliman Corporate 100 | actual portfolio shares, top 100 by assets |
| U.K. Corporate DB | U.K. Pension Protection Authority | actual portfolio shares |
| U.S. Endowments | NACUBO historical studies | actual portfolio shares |

### 1.7 Global Market Portfolio (GMP)
- **Source:** State Street (with adjustments for REITs and hedge funds)
- **Variable:** `gmp_share[asset_class, t]` : market-cap weight of each asset class
- **Asset classes:** public equities, private equity, real estate, hedge funds, commodities

### 1.8 BEA Funding Ratios
- **Source:** U.S. Bureau of Economic Analysis
- **Aggregation:** State level only
- **Discount rate:** AAA-rated corporate bond yield curve

### 1.9 Board Composition
- **Source:** Andonov, Hochberg & Rauh (2018) — available on Aleksandar Andonov's website
- **Key variable:** `political_board_fraction[system, t]`

### 1.10 CIO Identities
- **Source:** Lu, Mullally & Ray (2023)
- **Key variable:** `cio_tenure[system, t]` (years in role)

### 1.11 Covid Mortality
- **Source:** CDC age-adjusted mortality rates
- **Variable:** `covid_mortality_rate[state]` : avg deaths per 100,000, 2020–2022

### 1.12 Private-Sector Peers (for peer effects robustness)
- **Source:** S&P Money Market Directory
- **Variables:** `alt_to_risky_share[endowments/corp_pensions/unions, state, t]`

---

## 2. VARIABLE CONSTRUCTION

### 2.1 Asset Class Aggregation
```
fixed_income  = cash + fixed income
public_equity = public equities
alternatives  = PE + real_assets + HF + misc_alts + other
real_assets   = real_estate + commodities
risky         = public_equity + alternatives
```

### 2.2 Key Outcome Variables
```python
# Alternative-to-Risky Share (main outcome)
omega[p,t] = target_alternatives[p,t] / (target_alternatives[p,t] + target_public_equity[p,t])

# Risky Share
risky_share[p,t] = (target_alternatives[p,t] + target_public_equity[p,t]) / total_AUM[p,t]

# Constraint slack (Section 4.1)
l[p,t] = actual_risky_share[p,t] - target_risky_share[p,t]
l_bar[p]  = mean(l[p,t], t=2002..2021)          # time-average

# Residualized constraint slack (strip out market fluctuations)
# regress l[p,t] on portfolio_return[p,t], take residuals, then average
```

### 2.3 Consultant Alpha (from CMA data)
```python
# For consultant c, asset class a (PE/RE/HF), year t:
alpha[c,a,t] = mu_A[c,a,t] - beta[c,a,t] * mu_E[c,t]

# Equal-weighted average across asset classes (for pension p with consultant c):
alpha_consultant[c,t] = mean(alpha[c,'PE',t], alpha[c,'RE',t], alpha[c,'HF',t])
# (uses available asset classes if some are missing)

# Average consultant alpha (for Figure 4a):
alpha_bar[t] = median over consultants of alpha_consultant[c,t]
# Include only consultants with >= 10 years of data; start 2003
```

### 2.4 Peer Alternative-to-Risky Share (Section 3.3)
```python
# Geographic peer network (inverse-distance weighted)
# d[p,k,t] = distance in km between headquarters of pension p and k (using 5-digit zip)
# Set d=1.6 km for pensions in same zip code

delta[p,k] = (1/d[p,k,t]) / sum(1/d[p,j,t] for j != p)

n[p,t] = sum(delta[p,k] * omega[k,t] for k != p)
```

### 2.5 Instrument for Consultant Beliefs (Section 3.1.2 — Bartik IV)
```python
# N_active[p,k,t] = number of consultants active in pension p's state at time k
# w[j,p,k] = 1/N_active[p,k,t]  if consultant j active in p's state at k, else 0

# Instrument (contemporaneous weights, k=t):
z[p,t,t] = sum(w[j,p,t] * alpha_consultant[j,t] for j in all_consultants)

# Instrument (pre-period weights, k=2005):
z[p,2005,t] = sum(w[j,p,2005] * alpha_consultant[j,t] for j in all_consultants)
```

### 2.6 State-Level Equity Entry Timing (Section 3.2)
```python
# From PENDAT data (state-level asset-weighted averages):
Delta_Equity_1996_2002[s] = target_equity_share[s,2002] - target_equity_share[s,1996]
Delta_Equity_1990_2002[s] = target_equity_share[s,2002] - target_equity_share[s,1990]
# (use actual shares when target shares missing)

Delta_omega_2002_2021[s] = omega[s,2021] - omega[s,2002]
```

### 2.7 Diversification Distance from GMP (Section 6.1, Figure 8b)
```python
# Asset classes j: public_equity, PE, real_estate, HF, commodities
# Omega_j[t] = pension risky portfolio share of asset class j
# G_j[t]     = GMP share of asset class j

D[t] = 0.5 * sum(abs(Omega_j[t] - G_j[t]) for j in asset_classes)
# Equivalent to "Active Share" (Cremers & Petajisto 2009)
```

---

## 3. REGRESSION SPECIFICATIONS

### 3.1 Table 3 — Consultant Beliefs → Pension Allocations (OLS & IV)
```
omega[p,t] = lambda_t + beta * alpha_consultant[c(p),t] + Gamma * X[p,t] + eps[p,t]

Controls X[p,t]:
  - gasb25_funding[p,t]
  - log(aum[p,t])
  - hurdle_rate[p,t]
  - required_contribution_payroll[p,t]
  - deductions_payroll[p,t]
  - cash_share[p,t]
  - annual_return[p,t]
  - residualized (actual_risky - target_risky)[p,t]   # strip out annual return first

SE: double-clustered by consultant and year

# OLS col(1): no controls, time FE only
# OLS col(2): with controls, time FE
# IV  col(3): instrument = z[p,t,t]  (contemporaneous IAR weights)
# IV  col(4): instrument = z[p,2005,t] (2005 IAR weights)

Key results:
  beta_OLS  = 3.39  (t=2.98)  no controls
  beta_OLS  = 3.45  (t=3.19)  with controls
  beta_IV   = 3.27  (t=4.24)  contemporaneous instrument
  beta_IV   = 4.01  (t=5.27)  2005 instrument
```

### 3.2 Table 4 — Consultant Fixed Effects Variance Decomposition
```
omega[p,t] = lambda_t + lambda_c + Gamma * X[p,t] + eps[p,t]
# Also estimated with state-by-time FE instead of time FE

# Report:
#   Incremental R^2 from adding lambda_c (over controls + time FE): +17pp
#   Incremental R^2 with state-by-time FE: +12pp
#   F-test: lambda_c all equal (reject, p=0.00)

# Repeat for sub-shares: PE/credit, HF, real assets (relative to total risky)
```

### 3.3 Table 5 — Experience Effects (State-Level)
```
Delta_omega[s, 2002->2021] = c
  + beta1 * Delta_Equity[s, 1996->2002]
  + beta2 * Delta_Equity[s, 1990->2002]    # controls for total equity exposure
  + Gamma * X[s]
  + eps[s]

Controls X[s]:
  - Delta_hurdle_rate[s, 2002->2021]
  - Delta_pct_retired[s, 2002->2021]
  - bea_funding[s, 2002]
  - log(aum[s, 2002])
  - required_contribution_payroll[s, 2002]
  - fraction_restricted_assets_1998   (constitutional investment restriction indicator)
  - financial_sophistication_1998     (min qualifications indicator)
  - political_board_fraction_2002

SE: heteroskedasticity-robust (state level)

Key result: beta1 = 0.87** (t=3.43) in col(3)
            R^2 = 0.20 with active-change method

Placebo: replace LHS with Delta_risky_share -> beta1 = 0.24 (t=1.36), not significant
```

### 3.4 Table 6 — Peer Effects
```
omega[p,t] = lambda[c,t,d] + beta_z * n[p,t] + theta' * X[p,t] + eps[p,t]

# lambda[c,t,d] = consultant x year x census-division FE
# n[p,t] = inverse-distance-weighted peer alternative-to-risky share

Controls X[p,t]:
  - gasb25_funding[p,t]
  - hurdle_rate[p,t]
  - log(aum[p,t])
  - required_contribution_payroll[p,t]
  - admin_expenses_payroll[p,t]

SE: clustered by state and year

Col(1): baseline                          beta_z = 0.71** (t=3.30)
Col(2): interact with Established-CIO     interaction = 0.00 (t=0.03)
Col(3): interact with Well-Funded         interaction = -0.10 (t=-0.84)
Col(4): interact with High-Performing     interaction = -0.19** (t=-2.16)
         main effect for high-performers  = 0.75 - 0.19 = 0.56 (still significant)
Col(5): lagged peer share (robustness)    beta_z = 0.73** (t=3.21)
Col(6): non-pension peers (endow+corp)    beta_z = 0.43** (t=2.26)
         FE = consultant x year only
```

### 3.5 Table 7 — Risk-Seeking Motives
```
Delta_omega[p, 2002->2021] = a + beta * Delta_X[p] + eps[p]

Columns:
  (1) Delta_X = Delta_gasb25_funding     beta=-0.10 (t=-1.16)  R^2=0.01  N=118
  (2) Delta_X = Delta_bea_funding        beta=-0.06 (t=-0.28)  R^2=0.00  N=47 (state)
  (3) Delta_X = Delta_hurdle_rate        beta=-1.02 (t=-0.31)  R^2=0.00  N=117
  (4) Delta_X = Delta_pct_retired        beta=0.13  (t=0.76)   R^2=0.00  N=115
  (5)-(7): repeat (1),(3),(4) adding constraint controls (l_bar, Delta_cash_share)
           -> results essentially unchanged

SE: clustered by state (system-level regressions) or robust (state-level)
```

### 3.6 Table 8 — Covid Pandemic Spending Shock
```
Delta_Y[s] = a + b * CovidMortalityRate[s] + eps[s]
# CovidMortalityRate standardized to mean=0, sd=1

Outcomes:
  (1) Delta_LE[s, 2019->2021]        b=-0.22** (t=-4.67)   R^2=0.31
  (2) Delta_FractionRetired[s, 2019->2022]  b=-0.38** (t=-2.24)  R^2=0.07
  (3) Delta_omega[s, 2019->2022]     b=-0.00   (t=-0.99)   R^2=0.01
  (4) Delta_omega[s, 2019->2023]     b=-0.00   (t=-0.67)   R^2=0.00

SE: heteroskedasticity-robust
N=51 states (including DC)
```

---

## 4. SIMULATION EXERCISE (Section 5.1, Table 9)

### 4.1 Model Setup
The model is mean-variance (Markowitz 1952 / Merton 1969) with 3 assets:
- Riskless asset (return r_f)
- Public equities (excess log return r_E - r_f ~ N(mu_E, sigma2_E))
- Alternatives (excess log return r_A - r_f = alpha + beta*(r_E - r_f) + eta_A)

```python
# Implied moments:
mu_A    = alpha + beta * mu_E
sigma2_A = beta**2 * sigma2_E + sigma2_eta
sigma_AE = beta * sigma2_E

# Variance-covariance matrix Sigma:
Sigma = [[sigma2_A, sigma_AE],
         [sigma_AE, sigma2_E]]

# Jensen's inequality correction:
mu_tilde = [mu_A + sigma2_A/2,
            mu_E + sigma2_E/2]

# Optimal unconstrained weights on risky assets (eq. 1):
[omega_A, omega_E] = inv(Sigma) @ mu_tilde / gamma

# Alternative-to-risky share (eq. 2):
Omega = omega_A / (omega_A + omega_E)
     = (inv(Sigma) @ mu_tilde)[0] / sum(inv(Sigma) @ mu_tilde)
# NOTE: Omega is independent of gamma (Tobin separation)
```

### 4.2 Constrained Case
```python
# Constraint: omega_A + omega_E <= omega_max
# If constraint binds, solve:
# max E[U] s.t. omega_A + omega_E = omega_max, omega_A >= 0

# Partial derivative of Omega w.r.t. gamma when constrained:
# dOmega/dgamma = -K / (omega_max * gamma^2)
# where K is a function of beliefs
# K > 0 iff alternatives initially preferred (mu_A > beta * mu_E on margin)
```

### 4.3 Simulation Algorithm (Section 5.1)
```python
import numpy as np

N_sim = 100_000
np.random.seed(42)

# Observed aggregate data:
Omega_2001 = 0.14   # alternative-to-risky share in 2001
risky_2001 = 0.69   # overall risky share in 2001
Omega_2021 = 0.39   # alternative-to-risky share in 2021
risky_2021 = 0.76   # overall risky share in 2021
omega_min_f = 1 - risky_2021   # minimum fixed income share (constraint in 2021)
omega_max   = risky_2021       # maximum risky share in 2021

results = []
S_star = []

for i in range(N_sim * 10):  # oversample, keep 100k valid
    # Step 1: Draw beliefs
    mu_E     = np.random.uniform(0.02, 0.07)
    sigma2_E = np.random.uniform(0.02, 0.09)
    alpha    = np.random.uniform(0.00, 0.05)
    beta     = np.random.uniform(0.00, 1.50)

    # Step 2: Derived moments
    mu_A     = alpha + beta * mu_E
    sigma2_A = beta**2 * sigma2_E  # + sigma2_eta (sigma2_eta inferred below)
    sigma_AE = beta * sigma2_E

    # mu_tilde (Jensen correction)
    # sigma2_eta is inferred to match Omega_2001
    # From Omega = omega_A / (omega_A + omega_E), solve for sigma2_eta
    # (use eq. 2 analytically)

    # Step 3: Back out sigma2_eta to match Omega_2001
    # ... (solve closed form from eq. 2)

    # Step 4: Back out gamma_2001 to match risky_share_2001

    # Validity checks:
    # sigma2_eta > 0
    # gamma_2001 >= 1
    # sigma2_A = beta^2 * sigma2_E + sigma2_eta <= 0.25

    # Step 5 (Section 5.1 — reach-for-yield test):
    # Fix beliefs, impose omega_max constraint in 2021
    # Find gamma_2021 that matches Omega_2021
    # Check if Delta_Omega > 0 (i.e., can declining risk aversion explain the shift?)

    # Step 5 (Section 5.2 — belief shift):
    # Fix gamma, allow alpha to increase by Delta_alpha
    # Find Delta_alpha to match Omega_2021
    # Also find gamma_2021 to match risky_share_2021

# Key finding from Panel C of Table 9:
# - 99.6% of simulations: no change in risk aversion can explain Delta_Omega
# - Required belief shift: average Delta_alpha ≈ 70 bps
```

### 4.4 Belief-Shift Decomposition (Section 5.2)
```python
# Total required change in perceived alpha:
Delta_alpha_p ≈ 70 bps  (average across simulations)

# Component 1: Causal effect of consultants
# Delta_alpha_c = observed change in median consultant alpha ≈ 60 bps (2001-2021)
# zeta = causal influence of consultant beliefs on pension beliefs
#       (computed from b1 in Table 3 fed into mean-variance model)
# Consultant contribution ≈ 12% of total 70 bps

# Component 2: Peer network amplification
# Use Leontief (1986) / network multiplier approach
# Peer amplification ≈ 20% of remaining private belief shift (Delta_xi_p)
```

---

## 5. FIGURES — REPRODUCTION GUIDE

### Figure 1a — Aggregate Risky Share (PPD, post-2000)
```python
# Data: PPD, target risky share, aggregate (asset-weighted)
# X-axis: year 2001-2021
# Y-axis: aggregate_risky_share = sum(target_risky[p,t]*aum[p,t]) / sum(aum[p,t])
# Series: one line
# Range: 0.68 to ~0.78
```

### Figure 1b — Aggregate Risky Share (PPD + QSPP, post-1968)
```python
# Data: PPD (target) + U.S. Census QSPP (actual, starts 1968)
# QSPP URL: https://www.census.gov/programs-surveys/qspp.html
# X-axis: 1968-2021; two overlapping series from ~2001
# Note: QSPP risky = 1 - (fixed_income + cash) based on actual shares
```

### Figure 2a — Raw Portfolio Weights Over Time
```python
# Data: PPD aggregate, target shares
# Three area/line series: fixed_income, public_equity, alternatives
# X: 2001-2021
# Note: for every $1 out of fixed income since 2001:
#   $2.72 went to alternatives, $1.72 came from public equity
```

### Figure 2b — Subcategories of Alternatives (Bar Chart)
```python
# Two grouped bars: 2001 vs 2021
# Groups: real_assets (4%->12%), PE/credit (4%->10%), HF (0%->6%), Other
```

### Figure 2c — Alternative-to-Risky Share Aggregate
```python
# PPD aggregate, target: Omega[t] from 14% (2001) to 39% (2021)
```

### Figure 3a — Cross-Section of Omega Over Time (Box Plots)
```python
# Even years 2002-2020, one box plot per year
# Whiskers = 10th/90th percentiles
# 2021: p10=18%, p90=57%; 2001: spread = 26pp
```

### Figure 3b — Distribution of Delta_Omega (2002-2021)
```python
# Histogram of changes in omega across pension systems
# p25 = +15pp, p75 = +35pp
```

### Figure 3c — Within-Alternatives Heterogeneity (2021)
```python
# Box plots for PE-to-alt share, RA-to-alt share, HF-to-alt share
# Each shows distribution across pensions in 2021
# PE: p25=20%, p75=40%
```

### Figure 4a — Median Consultant Alpha Over Time
```python
# For each year t: median alpha_bar[t] across consultants
# Include only consultants with >= 10 years of data
# Start 2003; ends 2021
# Increases by 58 bps total
```

### Figure 4b — Binscatter: Consultant Alpha vs Pension Omega
```python
# After partialing out state-by-year FE + controls
# X: alpha_consultant[c(p),t]  (residualized)
# Y: omega[p,t]                (residualized)
# Binscatter with 20 equal-frequency bins
# Overlay OLS line: beta=3.39, t=2.98, Within-R^2=9%, N=1971
```

### Figure 5 — Distribution of Consultant Fixed Effects
```python
# Run regression (Table 4 col 2)
# Extract consultant FEs lambda_c
# Apply Casella (1992) empirical Bayes shrinkage
# Add back mean Omega for interpretation
# Histogram; p5 client = 7%, p95 client = 50%
```

### Figure 6a — Constraint Tightness vs Delta Omega
```python
# Binscatter: X = l_bar[p] (avg actual-target risky gap), Y = Delta_omega[p]
# beta=1.34, t=1.78, R^2=4%
```

### Figure 6b — Cash Share vs Delta Omega
```python
# Binscatter: X = Delta_cash_share[p], Y = Delta_omega[p]
# beta=-0.74, t=-1.49, R^2=2%
```

### Figure 7a — Alternative-to-Risky Across Institution Types
```python
# Four series: US Public (PPD target), US Endowment (NACUBO actual),
#              US Corporate (Milliman actual), UK Corporate (PPF actual)
# X: 2001-2021
# All four trend upward
```

### Figure 7b — Risky Share Across Institution Types
```python
# Same four series as 7a but risky share
# US Public & Endowment: slight increase
# US Corporate & UK Corporate: sharp decline
# UK: 69% (2004) -> 31% (2021)
```

### Figure 8a — GMP vs US Public Pension Alternative-to-Risky
```python
# X: 2001-2021
# Series 1: US Public Pension target omega (from PPD)
# Series 2: GMP alternative-to-risky (from State Street, adjusted)
# GMP: ~10pp (2001) -> ~23pp (2021)
# Pensions overweight alternatives by ~17pp relative to GMP by 2021
```

### Figure 8b — Distance from GMP Over Time
```python
# D[t] = 0.5 * sum|Omega_j[t] - G_j[t]| (Active Share metric)
# Two lines: target weights (blue), actual weights (red)
# Monotonically increasing from ~0.05 (2001) to ~0.18 (2021)
```

---

## 6. KEY NUMERICAL FACTS (for sanity checks)

```python
# Aggregate portfolio shifts (Table 1, Figure 2):
alt_share_2001 = 0.09    # 9% of total portfolio
alt_share_2021 = 0.30    # 30% of total portfolio
equity_share_2001 = 0.59
equity_share_2021 = 0.46
alt_to_risky_2001 = 0.14
alt_to_risky_2021 = 0.39
risky_share_2001 = 0.69
risky_share_2021 = 0.76

# Subcategory shares (Figure 2b):
real_assets_2001 = 0.04; real_assets_2021 = 0.12
pe_credit_2001   = 0.04; pe_credit_2021   = 0.10
hf_2001          = 0.00; hf_2021          = 0.06

# Consultant alpha trends (Table 2, in basis points):
alpha_all_2003 = 157; alpha_all_2021 = 215; Delta_alpha = 58
mu_A_all_2003  = 429; mu_A_all_2021  = 485; Delta_mu_A = 57
beta_all_2003  = 0.56; beta_all_2021  = 0.61

# Cross-section stats (Figure 3a):
p10_omega_2021 = 0.18; p90_omega_2021 = 0.57
spread_2001    = 0.26  # p90-p10 in 2001
spread_increase = 0.13  # pp increase in p90-p10 spread by 2021

# State heterogeneity examples:
high_adoption_states = ['Maine','New Mexico','Indiana','Wyoming','Texas']
high_adoption_avg_change = 0.58   # 58pp increase since 2001
low_adoption_states  = ['South Dakota','Nevada','Georgia','Iowa','Colorado']
low_adoption_change  ≈ 0         # essentially unchanged
```

---

## 7. SUGGESTED REPLICATION ORDER

1. **Figure 2** — Aggregate portfolio trends (PPD only, straightforward)
2. **Figure 1** — Risky share (add QSPP Census data)
3. **Figure 3** — Cross-sectional distribution (box plots & histogram)
4. **Table 7** — Risk-seeking regressions (single-variable OLS, easy baseline)
5. **Table 5** — Experience effects (state-level, need PENDAT + PPD)
6. **Figure 4a** — Consultant alpha time series (need CMA data)
7. **Table 3** — Consultant beliefs → allocations (need CMA + IAR data for IV)
8. **Table 4** — Consultant FE / variance decomposition (need consultant identity)
9. **Table 6** — Peer effects (need to compute n[p,t] inverse-distance weights)
10. **Table 8** — Covid design (need CDC mortality + PPD Dec 2024 release)
11. **Figure 8** — GMP comparison (need State Street GMP data)
12. **Table 9 / Section 5** — Simulation (self-contained, no external data needed)

---

## 8. NOTES ON RESTRICTED DATA

- **CMA data:** Obtained under confidentiality agreements from consultants; not publicly available
- **IAR data:** Requires request to Mark Egan (Egan, Matvos & Seru 2019 dataset, updated)
- **PENDAT surveys:** May require request to Public Pension Coordinating Council / GFOA
- **PPD:** Publicly downloadable at https://publicplansdata.org/
- **Census QSPP:** Publicly available at census.gov
- **NACUBO:** Requires purchase or institutional access
- **Milliman Corporate 100:** Requires purchase
- **UK PPF:** Publicly available at ppf.co.uk
- **State Street GMP:** Requires data agreement with State Street

---

## 9. SOFTWARE / ESTIMATION NOTES

- **Wild bootstrap** (Cameron, Gelbach & Miller 2008): use with G=15 consultant clusters
  - Null: beta=0; reported p_wild = 0.01 for all Table 3 columns
- **Oster (2019) bias adjustment:** Under assumption that observables and unobservables have equal
  explanatory power for alpha_consultant; bias-adjusted beta = 3.49 (vs OLS 3.45)
- **Empirical Bayes shrinkage** (Casella 1992, Eqs. 7.11 & 7.13): applied to consultant FEs in Figure 5
- **Bartik IV validity:** Goldsmith-Pinkham, Sorkin & Swift (2020) — exogeneity condition is
  that consultant state activity in year k is exogenous to pension preferences for alternatives

---
*Paper:* Begenau, J., Liang, P. & Siriwardane, E. (2026). "The Rise of Alternatives." Review of Financial Studies.
*Contacts:* begenau@stanford.edu, pliang20@stanford.edu, esiriwardane@hbs.edu

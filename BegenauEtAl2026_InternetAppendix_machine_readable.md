# Begenau, Liang & Siriwardane (2026, RFS) — "The Rise of Alternatives"
## Internet Appendix — Machine-Readable Guide
### 106 pages; LaTeX source (pdfTeX-1.40.25); June 2025

---

## APPENDIX STRUCTURE OVERVIEW

| Section | Title | Key Contents |
|---|---|---|
| A | Data | PPD cleaning, CMA construction, PENDAT, MMD, UK PPF, NACUBO, ASPP/QSPP |
| B | Additional Facts | Risky share cross-section, alt share levels |
| C | Robustness: Missing Target Shares | Calvet et al. (2009) imputation methodology |
| D | How Consultants Shape Pension Beliefs | Bayesian learning model, IV details, variance decomposition proofs |
| E | Belief-Based Explanations: Additional Evidence | Peer effect robustness, alt subcategory heterogeneity, rationality tests |
| F | Risk-Seeking Explanations: Additional Evidence | Model misspec tests, post-GFC, spending constraints, risky share |
| G | Aggregate Implications | Simulation details, belief-shift decomposition, peer multipliers |
| H | Other Explanations | Agency (return-smoothing, board composition, home bias), GMP supply-side |
| I | Derivations | Full CRRA + log-normal derivations, constrained/unconstrained solutions |

---

## SECTION A: DATA

### A.1 PPD Cleaning

**GitHub repository:** https://github.com/esiriwardane/ppd-cleaning-public
**Release used:** PPD 2023-07-28 (main analysis); December 2024 release (Covid section)

#### A.1.1 Asset Class Harmonization Rules
```
# Key reclassification rules (applied manually from CAFRs):
REITs          → public equities (NOT real estate)
Private credit → merged into private equity
Natural resources + infrastructure → commodities
Real estate    = core RE + private-equity RE (excludes REITs)
Real assets    = real estate + commodities
Alternatives   = PE/credit + real assets + HF + other alternatives
Risky          = public equities + alternatives
```

#### A.1.2 Data Filtering Rules
```python
# Step 1: Drop if missing market value of assets OR GASB 25 liabilities

# Step 2: Weight sum screen for pension p, fiscal year t:
A_pt = sum of actual portfolio weights
T_pt = sum of target portfolio weights

if abs(A_pt - 1) > 0.05 AND abs(T_pt - 1) > 0.05:
    DROP observation
elif abs(A_pt - 1) > 0.05 AND abs(T_pt - 1) <= 0.05:
    actual_weights = target_weights   # replace actual with target
elif abs(T_pt - 1) > 0.05 AND abs(A_pt - 1) <= 0.05:
    target_weights = actual_weights   # replace target with actual

# Step 3: Retain 2001-2021 only
# Step 4: Aggregate from plan → system level

# Result: 3,128 system-year observations
```

#### A.1.3 Special Plan Adjustments
```
1. Idaho PERSI:
   - Reports broad equity targets combining public + private equity
   - PPD records total as public equity, sets PE to zero
   - Fix: replace target shares with actual shares for this plan

2. NY State and Local Retirement System (FY 2001-2003):
   - PPD reports target shares not disclosed in annual reports
   - Fix: replace target shares with actual shares
```

#### A.1.4 Fiscal Year Note
Most plan fiscal years: July start → June end
Annual dates in PPD are fiscal years, not calendar years.
Results are robust to controlling for mid-year vs. end-year fiscal year ends.

#### A.1.5 High-Yield Debt Decomposition (Section A.1.3)
```
Data source: PPD "PensionCreditRating" dataset
Coverage: 2004-2018, not well-populated for all plans
Process:
  - Assign columns to IG or HY bucket (Moody's preferred; S&P used if Moody's coverage lower)
  - Aggregate to national level; use years with >= 20 plans (2005-2018)
  - Scale HY share by fixed income share from main PPD for portfolio-level figure

Key finding (Figure A1):
  avg HY/fixed_income share ≈ 30%
  avg HY/total_portfolio share ≈ 8%
  Both were relatively stable 2005-2018
```

---

### A.2 National Pension Data (ASPP + QSPP)

| Dataset | Frequency | Start Year | Coverage | URL |
|---|---|---|---|---|
| ASPP (Annual Survey of Public Pensions) | Annual | 1993 | All state/local DB plans | census.gov |
| QSPP (Quarterly Survey of Public Pensions) | Quarterly | 1968 | 100 largest plans (~90% of assets) | census.gov |

**Important:** QSPP asset class definitions changed materially in 2019. Use only through 2018 for time-series consistency.

**Validation:** PPD covers ≥85% of ASPP pension assets in all five-year sub-periods; ≥90% from 2006 onward.

---

### A.3 Capital Market Assumptions (CMAs)

**Consultants:** 14 consultants, obtained under NDA (names not disclosed)
**Coverage:** ~78% of U.S. pension assets
**PPD coverage with CMA data:** 63% of system-year observations

#### Asset Class Benchmarks Used
```
Fixed income   → U.S. core bonds
Public equity  → U.S. stock (or large-cap domestic equity if unavailable)
Real estate    → Private-label real estate (excludes REITs)
Hedge funds    → Most inclusive HF category; "absolute return" if HF unavailable
Private equity → PE aggregate (includes buyout + VC)
```

#### Alpha and Beta Construction
```python
# Step 1: Convert arithmetic returns to geometric:
mu_geometric = mu_arithmetic - 0.5 * sigma^2

# Step 2: Fill missing correlation matrices with prior year's values
# (assumption: correlations are stable year-to-year)

# Step 3: Compute beta for consultant c, asset class a, year t:
beta[c,a,t] = corr[c,a,E,t] * (sigma_A[c,a,t] / sigma_E[c,t])

# Step 4: Compute alpha:
alpha[c,a,t] = mu_A_excess[c,a,t] - beta[c,a,t] * mu_E_excess[c,t]

# Excluded from main analysis: infrastructure, private debt
# (CMA data only available for last few years)
# Proxy: real assets beliefs ← real estate; PE beliefs ← private equity
```

---

### A.4 PENDAT Surveys

**Administrator:** Public Pension Coordinating Council (PPCC)
**Waves used:** 1990, 1994, 1996, 1998
**Files used per wave:**
  1. `system` file — pension identity, fiscal year
  2. `investment` file — target + actual allocations, AUM
  3. `historical` file — actual allocations from prior 3 surveys (no targets)

**Missing value codes:** -9 = missing; -1 = not applicable → set both to missing

#### Coverage (Figure A3)
```
PENDAT covers virtually all U.S. pension assets in the 1990s
(closely matches ASPP totals for FY 1994, 1996, 1998)
```

#### Target Share Availability
```
1996: available for 72% of pensions by count, 89% by assets
1990: available for 32% of pensions by count, 57% by assets

Hierarchy for missing target shares:
  1. investment file of that survey year
  2. average of 1994 and 1998 investment files (for 1996 only)
  3. FOIA requests
  4. actual shares from investment/historical files (last resort)
```

#### State-Level Aggregation
```python
# Aggregate to state level using asset-weighted averages
# Reason: reduces influence of smaller pensions with more missing data

# Variables used from PENDAT:
# - target/actual equity share (1990, 1996, 2002)
# - constitutional investment restrictions (1998 wave, backup: 1996)
# - min professional qualifications indicator (1998 wave, backup: 1996)
# - tactical vs. long-term asset allocation indicator (1998/1996)
```

---

### A.5 S&P Money Market Directory (MMD)

**Purpose:** Alternative-to-risky and risky share of private-sector institutions (used in peer effects Section 3.3 and Section D.4.3)

**Tables used:**
- `MMASSET` — dollar and percent allocations by asset class, indexed by year
- `MMPLANS` — total assets, plan type (DB, endowment, etc.)
- `MMPROV` — consultant relationships per plan

**Quality filter:**
```python
# Keep if: |total_assets_MMASSET - total_assets_MMPLANS| / total_assets_MMPLANS <= 0.05

# Keep only plan types:
#   "Defined Benefit Plan"
#   "Endowment Fund"
#   "Closed or Frozen Defined Benefit Plan"

# Keep only sponsors classified as:
#   "Corporation", "Endowment", or "Union"
# AND domiciled in United States

# Aggregation for Section 3.3:
#   → state-year level; require > 5 private-sector pensions per cell

# Aggregation for Section D.4.3:
#   → consultant-year level (using MMPROV); require > 5 private-sector clients per consultant-year
```

---

### A.6 U.K. Corporate DB Pensions

**Source:** U.K. Pension Protection Fund (PPF) Purple Book (2022 edition)
**URL:** ppf.co.uk
**Coverage (March 2022):** 5,131 schemes, £1.7+ trillion of assets

**Asset class mapping:**
```
public equity  = UK quoted + overseas quoted
alternatives   = hedge funds + property + miscellaneous + unquoted/private equity
fixed income   = bonds + cash/deposits + insurance policies + annuities
risky          = all assets outside fixed income
```

---

### A.7 Endowments (NACUBO)

**Source:** NACUBO-TIAA Study of Endowments (NTSET); annual
**URL:** https://www.nacubo.org/Research/2022/Public-NTSE-Tables
**Coverage (2022):** 600+ institutions; $807 billion AUM

**Asset class mapping:**
```python
fixed_income = nacubo_fixed_income + cash
public_equity = nacubo_equity (pre-2009)
              = nacubo_domestic_equity + nacubo_international_equity (2009+)
alternatives  = residual
risky         = public_equity + alternatives
```

**Rebalancing policy note:** 92% of endowments had formal rebalancing policies (NACUBO 2021);
87% used market value-based rules (target + range).

---

## SECTION B: ADDITIONAL FACTS

### B.1.1 Aggregate Risky Share — Long-Run Decomposition

**Figure A4 data sources:**
- Panel (a): ASPP — decomposes FI into: cash, corporate bonds, agency debt, treasuries (1993–2016)
- Panel (b): QSPP — decomposes into: cash, corporate bonds, U.S. government-sponsored debt (1968–2018)

**Key historical facts:**
```
1970: corporate bonds = 56.4% of total portfolio; U.S. sponsored debt = 8.6%
1970s: cut corporate bonds by 15.4pp; increased sponsored debt by 9.1pp
1980s: cut corporate bonds by another 20.2pp; increased sponsored debt by 10.5pp
1990s: decline in FI driven by rotation out of Treasuries into public equities
       risky share increased ~17pp in the 1990s alone
Post-2000: risky share increased only ~8pp (much smaller)
```

### B.1.2 Cross-Section of Risky Share

```
2021: p10 risky share = 67%; p90 risky share = 85%; spread = 18pp
2002: spread = 21pp (slight decline in dispersion over time)

Notable outliers:
- Texas Municipal Retirement System (TMRS): was 100% fixed income pre-2009
  → House Bill 360, 81st Texas Legislature session → enabled diversification
- South Carolina RSIC: prohibited from public equities until 1997
  → rose from <25% risky to 74% by end of sample
- Texas County and District Retirement System (TCDRS): similar legal restriction story

Turnover (Table A1a):
  33% of bottom-quartile-risk pensions in 2002 remained in bottom quartile in 2021
  15% of bottom-quartile-risk pensions in 2021 came from top quartile in 2002
  29% of top-quartile-risk pensions in 2021 came from bottom quartile in 2002
```

### B.2 Alternative Share — Notable Outliers
```
PA Public School Employees RS (PSERS): alt-to-risky ≈ 75% since early 2010s
Indiana Public Retirement System: alt-to-risky ≈ 70% since 2012
```

---

## SECTION C: ROBUSTNESS — MISSING TARGET SHARES

### C.1 Calvet, Campbell & Sodini (2009) Imputation

#### Passive Share Construction
```python
# For pension with asset class weights w[j,t] at time t:
# r[j,t→T] = return on asset class j from t to T

# Passive share at T for asset j:
w_passive[j,T] = w[j,t] * (1 + r[j,t→T]) / sum_k(w[k,t] * (1 + r[k,t→T]))

# Active change in j between t and T:
A[j,t→T] = w[j,T] - w_passive[j,T]
```

#### Return Proxies

**PPD data:**
```python
# Use each pension's own return on asset class j
# If missing: use aggregate pension return on that asset class
```

**PENDAT data (1990→2002 changes):**
```python
# Asset class benchmarks:
domestic_equities      = CRSP value-weighted return (Ken French website)
international_equities = Developed market ex-US index (Ken French website)
real_estate_equities   = FTSE Nareit U.S. Real Estate Index
fixed_income           = Barclay's Aggregate Fixed Income Index
  (includes: govt debt + corporate bonds + real estate mortgages)
other (incl. cash)     = 1-month T-bill rate (ICE BofA US 1-Month T-Bill Index)
alternatives (for 1996→2002) = Cambridge Associates U.S. Private Equity Index
```

#### When to Apply Substitution (Table 5 context)
```python
# Replace Delta_Equity[s, 1990→2002] with active change A_e[s, 1990→2002] if:
#   > 50% of state's assets are missing target shares in 1990 or 2002
# Results robust to different thresholds and to using active changes for all states
```

#### Imputing Missing LEVELS of Target Shares (PPD)
```python
# Example: pension has actual shares from 2004 but target shares only from 2006

# Step 1: Compute active changes A[j, 2004→2006]
# Step 2: Infer 2004 target share:
#   target[j, 2004] = target[j, 2006] - A[j, 2004→2006]
# Step 3: Enforce constraints: short-sale constraint, leverage constraint, sum to 1

# Use first available year of target shares as base year
# Results robust to using different base years (e.g., 2020)
```

#### Robustness Results (Tables A2–A6)
```
Table A2 (vs Table 3 — consultant beliefs): similar magnitude, all significant
Table A3 (vs Table 4 — consultant FEs): large and significant FEs, comparable incremental R²
Table A4 (vs Table 5 — experience): similar coefficients and R² on Delta_Equity_1996→2002
Table A5 (vs Table 6 — peer effects): comparable magnitudes; same statistical significance patterns
Table A6 (vs Table 7 — risk-seeking): risk-seeking proxies still have little explanatory power
  Note: 1 fewer pension in A6 vs Table 7 (Philadelphia Board of Pensions lacks data in 2010)
```

---

## SECTION D: HOW CONSULTANTS SHAPE PENSION BELIEFS

### D.1 Bayesian Learning Model

#### D.1.1 Setup
```
Environment identical to Section 2.3 plus:
  - Pension p has prior about alpha: p0_p(alpha) = N(alpha_0p, 1/tau_0p)
  - Consultant c(p) provides noisy signal: alpha_c(p) = alpha_true + eps_p
    where eps_p ~ N(0, 1/tau_p) and tau_p = s_p * tau_0p

Interpretation of s_p:
  s_p > 1  → consultant signal is more precise than pension's own prior
  larger s_p → lower pension skill (more reliant on consultant)
```

#### D.1.2 Bayesian Updating
```python
# Posterior mean alpha for pension p:
alpha_1p = (1/(1+s_p)) * alpha_0p + (s_p/(1+s_p)) * alpha_c(p)

# Posterior precision:
tau_1p = (1 + s_p) * tau_0p

# Posterior moments for alternatives:
mu_A_p    = alpha_1p + beta * mu_E
sigma2_A_p = (1/(1+s_p)) * (1/tau_0p) + beta^2 * sigma2_E + sigma2_eta
sigma_AE  = beta * sigma2_E
```

#### D.1.3 Key Comparative Statics (Equations 6–12)

```python
# dOmega / d(Delta_c(p)) > 0:  [Eq. 8]
#   Pension always increases alternative-to-risky share when consultant is more bullish
#   Delta_c(p) = alpha_c(p) - alpha_0p = consultant signal minus pension's prior

# d^2_Omega / d(alpha_c(p)) * d(s_p):  [Eq. 13]
#   Sign is AMBIGUOUS in general for alternative-to-risky share
#   BUT d^2_omega_A / d(alpha_c) * d(s_p) > 0 always (raw alternative share)
#   For beta > 1: d^2_Omega / d(alpha_c) * d(s_p) > 0
#   For 0 < beta < 1: likely positive in practice (C term in Eq.12 negligibly small)

# Empirical finding (Figure A7):
#   Causal effect of consultants roughly TWICE as large for:
#     - smaller pensions (high s_p proxy)
#     - more politicized boards (high s_p proxy)
#   Consistent with d^2_Omega / d(alpha_c) * d(s_p) > 0 empirically
```

#### D.1.4 OLS Bias Decomposition (Equation 14)
```
beta_OLS = b1 (causal effect)
         + tau_B (bias from selection on beliefs: cov(alpha_0p, alpha_c(p)) != 0)
         + tau_NB (bias from selection on non-beliefs: cov(v_p, alpha_c(p)) != 0)

Oster (2019) bias-adjusted coefficient: 3.49 (vs OLS 3.45)
→ Selection bias is small
```

#### D.1.5 IV LATE Formula (Equation 15)
```
beta_IV = E[theta * s_p / (1 + s_p)]

Bartik instrument is valid if geographic location is exogenous to:
  - pension-specific belief parameters (alpha_0p, s_p)
  - non-belief preferences (v_p)
```

---

### D.2 Alpha Decomposition (Table A7, Equation 18)

```python
# Total change in average consultant alpha for asset class j:
Delta_alpha_bar_j = Delta_mu_bar_j          # expected return channel
                  - beta_bar_j1 * Delta_mu_E  # public equity risk premium channel
                  - mu_E0 * Delta_beta_bar_j  # diversification channel
                  - Delta_Cov(beta_j, mu_E)   # covariance term (omitted from table)

# Table A7 shows each channel as % of total Delta_alpha_bar_j

# Key finding: virtually all of the rise in average alpha is from higher expected returns
# Diversification benefits have DECLINED slightly (beta_bar increased from 0.56 to 0.61)

# Figure A8 shows:
#   Perceived volatility: stable or slightly declining (largest decline: PE)
#   Perceived correlation with public equities: INCREASED for all alt classes
#   → Beta rise driven by stronger perceived co-movement, not higher volatility
```

---

### D.3 Instrumental Variable — Technical Details

#### D.3.1 IAR Definition
```
Consultant c is "active in state s as of year k" if:
  c employs >= 1 Investment Advisory Representative (IAR) who is either:
    (a) registered to operate in state s, OR
    (b) maintains home office in state s

Data source: Updated version of Egan, Matvos & Seru (2019) dataset
Contact: Mark Egan (megan2@hbs.edu)

Key institutional facts about IARs:
  - Must pass Series 63 + Series 65 exams (FINRA)
  - Subject to fiduciary standard (Investment Advisers Act 1940)
  - Must comply with state-level registration requirements (not trivial)
  - Average consultancy is active in ~13 states (far from all 51 jurisdictions)
  → Entry costs create meaningful cross-state variation → relevant instrument
```

#### D.3.2 Instrument Validity
```
Exogeneity (k=t weighting): consultant location in year t is exogenous to
  contemporaneous pension preferences, conditional on Xp,t (which controls for size)

Exogeneity (k=2005 weighting): consultant location choices in 2005 are plausibly
  exogenous to future pension preferences for alternatives (2005 pre-dates acceleration)

Relevance (Table 3, cols 3-4):
  First-stage F-statistics: 84.62 (k=t) and 20.04 (k=2005)
  Both exceed 10 (Stock & Yogo threshold)
  Both exceed Andrews et al. (2019) robust threshold
```

---

### D.4 Variance Decomposition — Formal Proof

#### D.4.1 Without Conditioning on Observables
```
Result: Incremental R² from adding consultant FEs to regression provides a LOWER BOUND
        on fraction of cross-sectional variance in Omega attributable to beliefs

Condition required: no selection on non-beliefs (tau_NB = 0)

Intuition: Var(lambda_c) <= Var(belief component)
           (the within-consultant variance of beliefs reduces the tightness of the bound)

Lower bound = Var(lambda_c) / Var(Omega_p)
            = incremental R² from adding consultant FEs
```

#### D.4.2 Conditioning on Observables Xp (Weaker Assumption)
```
Modified condition: tau_NB = 0 conditional on Xp
  (no selection on non-beliefs, AFTER controlling for observables)

Lower bound B = VXp(lambda_c) / VXp(Omega_p)
              = incremental R² from adding consultant FEs to model
                that already includes controls Xp

Key result from Table 4:
  Incremental R² from consultant FEs (controlling for Xp, time FE):    +17pp
  Incremental R² from consultant FEs (controlling for Xp, state×time): +12pp
```

#### D.4.3 Private-Sector Clients Test (Figure A9)
```
Setup: For each consultant c in year t, compute:
  u_ac,t = avg alternative-to-risky share of PUBLIC-sector clients
  p_ac,t = avg alternative-to-risky share of PRIVATE-sector clients (from MMD)
  u_rc,t = avg risky share of public-sector clients
  p_rc,t = avg risky share of private-sector clients

Key results:
  Panel (a): beta = 0.58 (t=4.15), Within-R² = 17%
    → 10pp increase in private-sector clients' alt-to-risky share
       associated with 6pp increase in public-sector clients' alt-to-risky share
  Panel (b): beta = 0.13 (t=0.93), Within-R² ≈ 0%
    → NO relationship for risky shares

Interpretation: shared consultants cause similar COMPOSITION decisions but NOT
  similar total risk budgets. Supports beliefs (not agency) as the driver.
  Consistent with Foerster et al. (2017) advisor effects for Canadian households.

Requires: > 5 private-sector clients per consultant-year
```

---

## SECTION E: BELIEF-BASED EXPLANATIONS — ADDITIONAL EVIDENCE

### E.1 Peer Effects — Robustness

#### Ruling Out Consultant-Based Peer Networks (Table A8)
```python
# Baseline (col 1): n_pt = inverse-distance-weighted avg of ALL other pensions' Omega_kt
# Col 2: drop consultant FE, include time×census-division FE only
#   → estimates stable, but doesn't rule out consultant-based peer effects

# Col 3: construct n_no_share_pt = inverse-distance-weighted avg of pensions
#         that do NOT share a consultant with p
#   → if consultants are sole source: beta_z should drop to zero
#   → RESULT: beta_z comparable to col 1, strongly significant
#   → CONCLUSION: peer networks are distinct from shared consultants
```

#### Peer Network Formation Mechanisms (Section E.1)
```
Documented channels:
1. Conference travel and industry events
   - CALAPRS (California): annual General Assembly, roundtable meetings, training courses
   - NASRA annual conference (rotating locations)
   - NCPERS annual conference (rotating locations)
   Evidence: CalPERS Travel Transparency Report FY2023:
     162 trips classified "Conference/Forum" = 665 travel days
     38% occurred within California or Pacific census division

2. Labor mobility between pensions (avenue for future research)
3. Shared educational experiences (avenue for future research)

Data availability: FOIA requests for pension travel records could build national database
```

### E.2 Heterogeneity Across Alternative Subcategories (Figure A10)

```python
# Panel regression:
y_pt = alpha_t + lambda_c + theta * RS_pt + eps_pt

# where y_pt = target share of PE (or RA or HF) relative to total alternatives
# RS_pt = risk-seeking proxies:
#   GASB 25 funding level, liability discount rates, pct_retired,
#   required_contribution/payroll, total_spending/payroll, cash_share,
#   (actual_risky - target_risky), 1-year realized portfolio returns

# Key finding (Figure A10):
#   Incremental R² from RS_pt: NEVER exceeds 1pp (for any alt subcategory)
#   Incremental R² from lambda_c: 14-19pp (order of magnitude larger)

# Supporting test (Figure A25 / Regression 28):
#   Omega_apt = f_pt + beta * V_c(p)at + eps_pat
#   where V_c(p)at = alpha of alt type a reported by consultant c(p)
#   f_pt = pension-by-time FE (absorbs time-varying pension characteristics)
#   beta = 1.92 (t=6.83), Within-R² = 15%
#   → Pensions allocate more to PE vs. real assets when consultant reports PE has higher alpha
#   → Wild bootstrap p-value = 0.001
```

### E.3 Rationality of Beliefs

#### E.3.1 CAPM Estimation Methodology
```python
# For all alpha/beta estimates from historical data:
Ra_t - Rf_t = alpha + beta0 * (Rm_t - Rf_t) + beta1 * (Rm_t-1 - Rf_t-1) + eps_t

# where:
#   Ra_t = annual return of alternative asset (Cambridge Associates indices)
#   Rm_t = CRSP value-weighted index (Ken French website)
#   Rf_t = 1-month T-bill (Ken French website)
#   CAPM beta = beta0 + beta1 (Dimson 1979 correction for illiquidity)
# SE: heteroskedasticity robust
```

#### E.3.2 Benchmark Indices Used
```
Private equity:  Cambridge Associates U.S. Private Equity Index (1983-2020)
Real estate:     Cambridge Associates U.S. Real Estate Index (1986-2020)
Hedge funds:     Credit Suisse Hedge Fund Index (1994-2020)
```

#### E.3.3 Summary of Rationality Evidence (Figure A11, Figure A12)
```
Private equity (Figure A11):
  Consultant-perceived beta_PE ≈ 1.2 → slightly ABOVE historical point estimate
    but within 95% CI
  Historical alpha range from literature: 150-200 bps (Harris et al. 2014, NBIM 2023)
  Consultant-perceived alpha_PE ≈ 180 bps → within range

Real estate:
  All three metrics (alpha, beta, expected return) within 95% CI of history
  Consultants slightly pessimistic on expected returns

Hedge funds:
  alpha_HF ≈ 190 bps (consultant) vs. historical point est. ≈ similar
  beta_HF ≈ 0.35 (both consultant and history)
  Consistent with Jurek & Stafford (2015): HFRI annual alpha ≈ 360 bps (1996-2012)
    and CAPM beta ≈ 0.35

Ex-post performance by alternative-to-risky quartile (Figure A12):
  Omega_bar quartile assignment based on 2001-2021 average alt-to-risky share
  q1 avg Omega_bar ≈ 15%;  q4 avg Omega_bar ≈ 45%

  Figure A12a: alpha_alternatives → q4 slightly higher than q1, but
    p-value = 0.70, no monotonic pattern across q2-q4
  Figure A12b: net-of-fee total returns → low alt users slightly higher
    (driven by post-GFC equity bull market); cannot reject equal returns
  Figure A12c: total portfolio CAPM alpha → no meaningful diff across quartiles
  Figure A12d: Sharpe ratios → LOW alt users have SLIGHTLY HIGHER Sharpe ratios
    (challenges the return-smoothing narrative specifically)
```

---

## SECTION F: RISK-SEEKING EXPLANATIONS — ADDITIONAL EVIDENCE

### F.1.1 Model Misspecification Test (Figure A14)
```
# Reproduce scatterplots underlying Table 7 regressions
# Variables: Delta_omega vs Delta_gasb25_funding, Delta_hurdle, Delta_pct_retired
# Finding: No obvious non-linearities; poor fit is genuine, not functional form issue
```

### F.1.2 Post-GFC Analysis (Table A9)
```python
# Sample: 2010-2021 (to isolate low-interest-rate environment)
# Regression: Delta_Omega_p = a + beta * Delta_Xp + eps_p

# Col(1): Delta_X = Delta_GASB25_funding (2010-2021)
#   beta = POSITIVE (more underfunding → LESS shift to alts), marginally significant
#   R² = low
# Cols(3)-(4): hurdle rates, pct_retired → both statistically insignificant
# Cols(5)-(6): initial funding in 2010 → no predictive power

# Reconciliation with Mittal (2024) who shows underfunded pensions contribute more $ to alts:
# Example: Pension A (underfunded) increases RISKY SHARE from 70% to 90%, keeps alt-to-risky=30%
#   → Alt investment goes from $21 to $27 (+$6)
# Pension B (funded) keeps risky share=70%, increases alt-to-risky from 30% to 32%
#   → Alt investment goes from $21 to $22.4 (+$1.4)
# A commits more new dollars even though B increases alt-to-risky share more
# → Funding drives the risky share (not alt-to-risky share)
```

### F.1.3 Initial Levels as Predictors (Table A10)
```
# Test: do initial levels (2002) of risk proxies predict subsequent changes?
# Variables: GASB25 funding 2002, hurdle rate 2002, pct_retired 2002, log(AUM) 2002,
#            fraction of years missing ARC payment 2001-2005

# Key finding col(1): better-funded pensions in 2002 slightly MORE likely to increase
#   alt-to-risky share (positive coeff) → opposite of risk-seeking prediction

# Figure A15a: no relationship between ARC payment failures and alt adoption
# Figure A15b: smallest pensions shifted less into alts, but effect small and insignificant
#   → size not a strong predictor of alt-to-risky level or change
```

### F.1.4 Panel Analysis in Levels (Table A11)
```python
# Regression: Omega_pt = alpha_t + beta * X_pt + eps_pt
# Time FE included; SE double-clustered by state and year

# Results:
#   GASB25 funding:     beta statistically insignificant, Within-R² < 1%
#   BEA-adjusted funding: same
#   Hurdle rate:        negative sign (counterintuitive), insignificant
#   Pct_retired:        insignificant

# Figure A16: binned scatter plots confirm no non-linearities

# 2021 cross-section only (cols 5-8): same pattern; hurdle rate negative in 2021
```

### F.1.5 Portfolio Constraint Time-Series (Figure A18)
```python
# l_pt = actual_risky_share_pt - target_risky_share_pt

# Figure A18a: raw l_pt
#   In-sample average = -0.9 pp (actual slightly below target)
#   Heavily influenced by GFC 2008-2009 dip (mechanical decline in actual vs target)

# Figure A18b: residualized l_pt (after regressing on contemporaneous 1-year returns)
#   Residual average ≈ 0 pp, no persistent deviation from zero
#   → No evidence that U.S. pensions are CONSTRAINED IN AGGREGATE
```

### F.2 Spending Constraints

#### F.2.1 Forecasting Spending (Table A12a)
```python
# Test rational expectations version of spending-based channel:
# If Omega_p drives alternative adoption through expected spending,
# then Omega_p should FORECAST future spending:

# Regression: Spending_p[2012-2021_avg] = alpha + beta * Omega_p[2011] + eta

# Two spending measures:
#   CPR = contribution-to-payout ratio (low CPR = high spending relative to contributions)
#   Fraction of years missing ARC payments 2012-2021

# Results: Omega_p[2011] has essentially NO forecasting power for either spending measure
#   R² ≈ 0 in all specs; also true for Delta_Omega (changes in alt-to-risky) as predictor
```

#### F.2.2 Current Spending (Table A12b)
```python
# Panel regression:
y_pt = lambda_t + beta * CPR_pt + eps_pt
# y_pt = overall alt-to-risky share, PE-to-risky, or RA-to-risky

# Also: long-run change regression (2002-2021)
# Delta_y_p = a + b * Delta_CPR_p + eta_p

# Finding: NO economically or statistically significant relationship
#   between alternative use and current OR long-run spending
```

### F.3 The Risky Share (Table A13)
```python
# Regression: Delta_risky_share_p = a + beta * Delta_X_p + eps_p
# (parallel to Table 7 but for risky share rather than alt-to-risky share)

# Col(1): Delta_GASB25 → close to zero, insignificant
# Col(2): Delta_BEA_funding (state-level) → negative but small and insignificant
# Col(3): Delta_hurdle_rate → POSITIVE and SIGNIFICANT
#   Moving from p10 to p90 of hurdle changes → +7.3pp increase in risky share
#   (p10 of Delta_hurdle = -0.3pp; p90 = -1.6pp; so declining hurdles → rising risky share)
# Col(4): Delta_pct_retired → POSITIVE and SIGNIFICANT
#   Moving from p10 to p90 of pct_retired changes → +3.3pp increase in risky share
#   (Consistent with Andonov et al. 2017 but weaker — they study 1990-2012)
# Cols(5)-(8): initial levels (2002) → no predictive power; R² < 10%

# Takeaway: funding/accounting explain SOME cross-pension variation in risky share
#   but NOT in alt-to-risky share → evidence for two separate mechanisms
```

---

## SECTION G: AGGREGATE IMPLICATIONS

### G.1 Shadow Cost Calculation (Figure A21)
```python
# For the 379 simulations where declining risk aversion can match Delta_Omega:
#   Average implied reduction in gamma: gamma_2001 - gamma_2021 ≈ 4.51

# Shadow cost = utility from unconstrained tangency portfolio
#               minus utility from constrained portfolio in 2021

# Formula: ShadowCost = M_counterfactual - M_portfolio
#   where M = E[r_p] + 0.5*(1-gamma)*Var[r_p]  (power utility objective)

# Finding: Average shadow cost = 732 basis points per year
#   → "Pensions would pay 732bps/year to relax the constraint"
#   → Implausibly large given asset hurdle rates ≈ 700bps
#   → Further evidence against the constrained risk-taking channel
```

### G.2.1 Full Belief-Shift Simulation (Figure A22, A23)
```python
# Modify baseline: no constraint on risky share; allow alpha to shift

# For each of 100,000 simulations:
#   1. Find Delta_alpha_p that matches Omega_2021 = 0.39
#   2. Find gamma_2021 that matches risky_share_2021 = 0.76

# Key findings:
#   Average required Delta_alpha_p ≈ 70 bps
#   After allowing belief shifts, gamma_2021 distribution ≈ gamma_2001 distribution
#     → only SMALL risk aversion changes needed to match risky share (not large)
#   Why? Because d(risky_share)/d(alpha) = (1/gamma) * (1-beta)/sigma2_eta > 0 for beta < 1
#     → alpha shift itself generates most of the risky share increase (≈55%)
#     → remaining 45% from small decline in risk aversion

# Figure A23 (Delta_Omega as function of fixed Delta_alpha):
#   Delta_alpha = 20-70 bps → can reasonably explain full aggregate rise
#   depending on initial beliefs in 2001
```

### G.2.2 Incorporating Consultant Effects
```python
# Decompose: Delta_alpha_p = zeta * Delta_alpha_c + Delta_xi_p
# where zeta = causal influence of consultant on pension beliefs

# Infer zeta in each simulation from:
#   b1 (IV estimate ≈ 4) = Lambda * zeta
#   Lambda = d_Omega/d_alpha_p (from model, varies by simulation)
#   → zeta = b1 / Lambda

# Distribution of zeta:
#   Mean = 0.138
#   Median = 0.075

# With Delta_alpha_c ≈ 60 bps (from Figure 4a):
#   Consultant contribution to Delta_alpha_p: avg = zeta * 60 ≈ 0.138 * 60 ≈ 8-10 bps
#   That's ≈ 12% of total 70 bps needed

# Required private belief shift Delta_xi_p ≈ 60 bps (Figure A24)
```

### G.2.3 Peer Network Multiplier
```python
# Vector form of peer effects regression (Eq. 8):
# Omega_t = beta_z * D * Omega_t + X_tilde * Gamma + eps_t
# Solution: Omega_t = Psi * X_tilde * Gamma + Psi * eps_t
# where Psi = (I - beta_z * D)^{-1}  [Leontief inverse]

# D is P×P inverse-distance matrix with:
#   D[i,j] = (1/d_ij) / sum_k(1/d_ik)   [row-normalized]
#   D[i,i] = 0

# beta_z ≈ 0.4 (from high-performing pensions subgroup, which is less prone to herding)

# Peer multiplier for shock vector eps_t:
# M(Psi, eps_t) = (s_t' * Psi * eps_t) / (s_t' * eps_t)
# where s_t = AUM-based pension weights

# CalPERS isolated shock (eps_CalPERS = 1, all others = 0):
#   M(Psi, eps_t) = 1.23
#   Interpretation: peer effects account for (1.23-1)/1.23 = 19% of aggregate rise
#   (for CalPERS-originated shock; "about 20%" in the paper)
```

---

## SECTION H: OTHER EXPLANATIONS

### H.1.1 Agency — Return Smoothing Tests

#### Asset Smoothing Windows (Pre-2014 GASB Policy)
```python
# Pre-2014 GASB: pensions could smooth market asset fluctuations over chosen window
# Post-2014 GASB: required market valuations (announced 2012)
# ~60% of pensions used 5-year window; others: none or up to 10 years
# Most pensions (>50%) made NO changes to smoothing window 2001-2014

# Test: Omega_pt = lambda_t + beta * SW_pt + eps_pt
#   SW_pt = smoothing window in years
#   Results (pre-2014):
#     OLS: beta = 0.003 (t=0.440) → statistically insignificant
#     IV (instrument: SW in 2001): first-stage F >> 70; beta_IV = -0.000 (t=-0.013)
#   → No evidence smoothing incentives drive alt allocations in cross-section

# Also note: largest increases in alt-to-risky occurred BEFORE 2012 (when GASB change announced)
#   and growth slowed markedly thereafter — opposite of what return-smoothing would predict
```

#### Real Assets vs. Private Equity Test (Equation 28, Figure A25)
```python
# Regression 28:
# Omega_apt = f_pt + beta * V_c(p)at + eps_pat
#
# Omega_apt = target weight of pension p in alt type a at time t
# f_pt = pension-by-time FE  ← absorbs ALL time-varying pension chars
# V_c(p)at = alpha of alt type a reported by consultant c(p) at t
# SE: double-clustered by consultant and pension
#
# Key result: beta = 1.92 (t=6.83), Within-R² = 15%
# Interpretation: pension invests more in real assets OVER PE when consultant
#   says real assets have higher alpha — EVEN after controlling for any
#   time-varying desire to smooth returns or take risk
# Wild bootstrap p-value: 0.001
```

### H.1.2 Board Composition (Table A15)
```python
# Panel regression:
# Omega_pt = alpha_t + Gamma * B_pt + beta * X_pt + eps_pt
# B_pt = board composition measure (from Andonov et al. 2018, updated through 2020)
# SE: double-clustered by time and pension system

# Results (Table A15):
#   Col(1) no controls:     economically small, statistically insignificant
#   Col(2) with controls:   same
#   Within-R² both ≈ 0

# Board composition measures tested (following Andonov et al. 2018):
#   fraction of board members appointed by elected official
#   fraction serving ex officio
#   overall board politicization indicator

# Conclusion: board composition does NOT predict alternative use
```

### H.1.3 Home Bias (Table A16, Equation 29)
```python
# Regression 29:
# Delta_Omega_s[2002→2021] = a + b * Alt_Activity_s + eps_s

# Alt_Activity_s measures:
#   (1) Fraction of U.S.-based private capital funds domiciled in state s (vintages 2000-2022)
#   (2) Fraction of AUM domiciled in state s (same vintages)

# Data: WRDS Preqin "Fund Details" + "Managers Detail" files
#   Restricted to: U.S.-based GPs, vintages 2000-2022

# Results (Table A16):
#   Both measures: small coefficient, statistically insignificant, R² ≈ 0
#   → Home bias does NOT explain cross-state variation in alt adoption
```

### H.2 Global Market Portfolio — Cross-Sectional Tests (Table A17)

```python
# Equation 30:
# Delta_Omega_p[2002→2021] = beta0 + beta1 * D_p[2002] + eps_p
# D_p = "Active Share" distance from GMP in 2002 (using target or actual shares)

# Prediction if diversification motive: beta1 > 0 (poorly diversified pensions shift more)
# RESULT: beta1 < 0 (negative) and statistically insignificant → OPPOSITE of prediction

# Change in distance (cols 3-4):
# Delta_Omega_p ~ Delta_D_p[2002→2021]
# beta1 > 0 and significant → pensions that adopted alts became LESS diversified
# R² relatively large → decline in diversification driven by alt adoption

# Additional GMP facts:
#   2002: 59% of pensions UNDERWEIGHT alternatives vs. GMP
#   2021: only 16% underweight alternatives
#   Among overweight pensions in 2021: avg overweight = 19pp
#   "Heavily overweight" (>10pp above GMP):
#     2002 ≈ 10% of pensions; 2021 ≈ 60% of pensions (Figure A26b)
```

---

## SECTION I: DERIVATIONS

### I.1 CRRA Preference with Log-Normal Distribution
```
Problem: max E[W^(1-gamma) / (1-gamma)]
         ⟺ max (1-gamma)*r_p + 0.5*(1-gamma)^2 * sigma2_p   [log-normal case]
         ⟺ max r_p + 0.5*(1-gamma)*sigma2_p
```

### I.2 Unconstrained Solution
```python
# Optimal weights: w_r = (1/gamma) * Sigma^{-1} * (mu + sigma^2/2)

# Where:
# Sigma = [[sigma2_A, sigma_AE],
#          [sigma_AE, sigma2_E]]
#
# Sigma^{-1} = (1/(sigma2_A*sigma2_E - sigma_AE^2)) * [[sigma2_E, -sigma_AE],
#                                                        [-sigma_AE, sigma2_A]]

omega_A = (1/gamma) * [sigma2_E*(mu_A + sigma2_A/2) - sigma_AE*(mu_E + sigma2_E/2)] /
                       [sigma2_A*sigma2_E - sigma_AE^2]

omega_E = (1/gamma) * [sigma2_A*(mu_E + sigma2_E/2) - sigma_AE*(mu_A + sigma2_A/2)] /
                       [sigma2_A*sigma2_E - sigma_AE^2]

# Key property: Omega = omega_A/(omega_A+omega_E) is INDEPENDENT of gamma
# (Tobin separation theorem)
```

### I.3 Constrained Solution (binding omega_f >= omega_f_min)
```python
# Lagrangian: L = E[r_p] + 0.5*(1-gamma)*Var[r_p] - lambda*(omega_f_min + omega_A + omega_E - 1)

# Constrained optimal:
omega_A_constrained = (1/gamma) * K + (1 - omega_f_min) * C

# where:
K = [(mu_A + sigma2_A/2) - (mu_E + sigma2_E/2)] / (sigma2_A - 2*sigma_AE + sigma2_E)
C = (sigma2_E - sigma_AE) / (sigma2_A - 2*sigma_AE + sigma2_E)

# In CAPM notation (using alpha, beta, sigma_eta):
K = [(alpha + (beta-1)*mu_E)/sigma2_E] / [(beta-1)^2 + sigma2_eta/sigma2_E]
    + [0.5*(beta^2-1) + sigma2_eta/sigma2_E] / [(beta-1)^2 + sigma2_eta/sigma2_E]

C = (1-beta) / [(beta-1)^2 + sigma2_eta/sigma2_E]

# Note: when constraint binds, Omega IS sensitive to gamma:
# d(Omega)/d(gamma) = -K / (omega_max * gamma^2)
# Sign depends on K: K > 0 iff alternatives preferred to equities
#   (K < 0 for most initial beliefs consistent with 2001 portfolio → confirms simulation result)

# Minimum gamma to keep constraint non-binding:
gamma_min = [1/(1 - omega_f_min)] * (mu + sigma^2/2)' * Sigma^{-1} * iota
# where iota = [1,1]'
```

---

## APPENDIX FIGURES — REPRODUCTION GUIDE

| Figure | Data | Key Variables | Key Finding |
|---|---|---|---|
| A1 | PPD PensionCreditRating | HY/FI share, HY/portfolio share | HY share stable ~30% of FI (2005-2018) |
| A3 | PENDAT + ASPP | Total assets by year | PENDAT covers virtually all U.S. pension assets in 1990s |
| A4a | ASPP | Cash, corp bonds, agency, treasury shares | Treasuries drove FI decline in 1990s |
| A4b | QSPP | Cash, corp bonds, govt-sponsored debt | Corporate bonds dominated FI in 1970; both declined |
| A5a | PPD | Risky share distribution, even years 2002-2020 | Rising risky share, moderate dispersion |
| A5b | PPD | Change in risky share 2002-2021 | Wide heterogeneity; p25=+1pp, p75=+16pp |
| A6 | PPD | Alt share (total portfolio, not risky) | Mirrors alt-to-risky dispersion pattern |
| A7 | CMA + IAR + PPD | IV estimates by size/board quintile | Small/political boards: 2× larger consultant effect |
| A8 | CMA | Consultant-perceived vol and correlation | Correlation with equities increased; drives beta rise |
| A9 | PPD + MMD | Public vs private alt-to-risky by consultant | Strong correlation in alt-to-risky (β=0.58); near-zero for risky |
| A10 | PPD + CMA | R² from risk-seeking vs consultant FEs | Consultant FEs explain 10-19× more variance than risk-seeking |
| A11 | CA indices + CMA | Alpha, beta, expected return for PE/RE/HF | Consultant beliefs not "wildly optimistic" vs history |
| A12 | PPD | Performance by alt-to-risky quartile | No outperformance for high-alt users; Sharpe slightly lower |
| A14 | PPD | Scatterplots: Delta_omega vs risk proxies | No non-linearities; genuinely weak relationships |
| A15 | PPD | Alt adoption vs failed ARC payments; size | No pattern for missed contributions; small effect for size |
| A16 | PPD | Binscatters: Omega level vs risk proxies | No non-linearities in level regressions |
| A17 | PPD + State Street GMP | Distance from GMP; overweight status | Worse-diversified pensions didn't shift more |
| A18 | PPD | l_pt = actual - target risky over time | Average ≈ 0 after stripping returns; no persistent constraint |
| A19 | PPD | Binscatters: Delta_risky vs risk proxies | Poor model fits not from non-linearity |
| A20 | Simulation | Histograms of admissible beliefs S* | Distribution of mu_E, sigma2_E, alpha, beta |
| A21 | Simulation | Risk aversion reduction and shadow costs | Shadow cost ≈ 732 bps → implausibly large |
| A22 | Simulation | Belief shift needed; gamma distributions | Risk aversion barely changes when beliefs shift |
| A23 | Simulation | Delta_Omega as function of fixed Delta_alpha | 20-70 bps explains full aggregate rise |
| A24 | Simulation | Required private belief shift Delta_xi_p | Avg ≈ 60 bps (after subtracting consultant effect) |
| A25 | PPD + CMA | PE-to-alt vs. RA-to-alt share ~ consultant alpha | β=1.92, Within-R²=15%; within-pension variation |
| A26 | PPD + State Street GMP | Fraction overweight alts over time | 59% underweight in 2002; 84% overweight in 2021 |

---

## APPENDIX TABLES — REPRODUCTION GUIDE

| Table | Section | Corresponds to | Key Result |
|---|---|---|---|
| A1a | B.1.2 | Transition matrix: risky share quartiles 2002→2021 | 33% persist in bottom Q; 29% of top Q in 2021 were bottom Q in 2002 |
| A1b | B.2 | Transition matrix: alt-to-risky quartiles 2002→2021 | 21% of top Q in 2021 were bottom Q in 2002 |
| A2 | C.2 | Table 3 with Calvet imputation | Comparable magnitudes; all significant |
| A3 | C.2 | Table 4 with Calvet imputation | Large, significant consultant FEs; comparable incremental R² |
| A4 | C.2 | Table 5 (cols 1,3,4,5) with Calvet imputation | Similar beta on Delta_Equity_1996→2002 |
| A5 | C.2 | Table 6 with Calvet imputation | Comparable peer effect magnitudes |
| A6 | C.2 | Table 7 with Calvet imputation | Risk-seeking still weak |
| A7 | D.2 | Alpha decomposition by channel | Expected returns drive alpha rise; diversification WORSENED |
| A8 | D.3 | Table A8 — Peer effects robustness (consultant source test) | Peer effects persist when excluding consultant-shared pensions |
| A9 | F.1.2 | Table 7 for post-GFC period (2010-2021) | Risk-seeking still weak even in low-rate environment |
| A10 | F.1.3 | Initial levels of risk proxies → subsequent changes | No predictive power; initial funding slightly positive (counterintuitive) |
| A11 | F.1.4 | Panel levels regression (Eq. 23) | All covariates insignificant; Within-R² < 1% |
| A12a | F.2.1 | Forecasting future spending with Omega | Almost no predictive power |
| A12b | F.2.2 | Current spending ~ alt use panel | No relationship in levels or changes |
| A13 | F.3 | Table 7 for risky share (not alt-to-risky) | Hurdle rates and pct_retired significant for risky share |
| A14a | H.1.1 | Median realized volatility by asset class | Alternatives have lower realized vol than public equities |
| A14b | H.1.1 | Volatility comparison: low vs high alt users | Very similar total portfolio volatilities (offsetting factors) |
| A15 | H.1.2 | Board composition ~ alt-to-risky | No correlation; R² ≈ 0 |
| A16 | H.1.3 | Home bias ~ alt adoption (state-level) | No relationship |
| A17 | H.2 | GMP diversification motive test | Poorly-diversified pensions in 2002 did NOT shift more |

---

## NOTES ON ADDITIONAL DATA ACCESS

| Data | Availability | Contact / URL |
|---|---|---|
| PPD (2023-07-28 release) | Public | https://publicplansdata.org/ |
| PPD cleaning code | Public | https://github.com/esiriwardane/ppd-cleaning-public |
| CMA data (14 consultants) | Confidential (NDA) | Contact paper authors |
| IAR data (updated Egan et al. 2019) | Request required | Mark Egan, HBS |
| PENDAT surveys | Request PPCC / GFOA | Chicago: Government Finance Officers Association |
| S&P Money Market Directory | Commercial | https://www.spglobal.com/marketintelligence/en/solutions/money-market-directories |
| UK PPF Purple Book 2022 | Public | ppf.co.uk |
| NACUBO NTSET tables | Public | https://www.nacubo.org/Research/2022/Public-NTSE-Tables |
| State Street GMP | Data agreement | https://www.ssga.com/us/en/institutional/insights/global-market-portfolio-2024 |
| Cambridge Associates PE Index | Commercial | cambridgeassociates.com |
| Cambridge Associates RE Index | Commercial | cambridgeassociates.com |
| Credit Suisse Hedge Fund Index | Public historical | credit-suisse.com |
| CRSP value-weighted returns | Public | Ken French data library |
| FTSE Nareit U.S. RE Index | Public | reit.com/nareit |
| Preqin Fund Details (WRDS) | Institutional (WRDS) | wrds.upenn.edu |
| HFR hedge fund AUM | Commercial | hedgefundresearch.com |
| CalPERS Travel Transparency Reports | Public | calpers.ca.gov |
| Board composition data | Public | Aleksandar Andonov's website |

---
*Internet Appendix to:* Begenau, J., Liang, P. & Siriwardane, E. (2026). "The Rise of Alternatives." Review of Financial Studies.
*Contacts:* begenau@stanford.edu, pliang20@stanford.edu, esiriwardane@hbs.edu

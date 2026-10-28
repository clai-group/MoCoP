# MoCoP User Guide

Version 1.0.0

MoCoP (Monotone Counterfactual Propagation) estimates the effect of a binary exposure on a
binary outcome from electronic health record data. The exposure is propagated through
observed mediators (laboratory values and, optionally, one binary post-exposure event) into a
monotone-constrained outcome model, and patient-level counterfactual risks are contrasted by
g-computation.

## Contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Data specification](#3-data-specification)
4. [Configuration reference](#4-configuration-reference)
5. [Edge specification](#5-edge-specification)
6. [Running MoCoP](#6-running-mocop)
7. [Output reference](#7-output-reference)
8. [Interpretation](#8-interpretation)
9. [Methods](#9-methods)
10. [Data partitioning and leakage control](#10-data-partitioning-and-leakage-control)
11. [Data protection](#11-data-protection)
12. [Assumptions and limitations](#12-assumptions-and-limitations)
13. [Troubleshooting](#13-troubleshooting)
14. [Demo](#14-demo)
15. [Versioning and reproducibility](#15-versioning-and-reproducibility)

---

## 1. Overview

### 1.1 Variable roles

MoCoP assigns every variable one of four roles.

| Role | Variables | Treatment in the counterfactual |
|---|---|---|
| Frozen | Pre-exposure covariates; baseline diagnosis groups; outcome history | Identical in both arms |
| Exposure | Binary exposure | Not an input to the outcome model; acts only through edges |
| Mediator | Laboratory values on a common severity scale; optional binary event | Shifted by the edges in the exposed arm |
| Outcome | Binary outcome | Predicted |

### 1.2 Pipeline

| Step | Description | Section |
|---|---|---|
| 1 | Read and validate the input files | 3 |
| 2 | Split patients into training, validation and test sets | 10 |
| 3 | Convert laboratory values to a signed severity scale and summarize each lab over the baseline window (level, slope, range) | 9.1 |
| 4 | Fit the outcome model with non-negativity constraints on all mediator coefficients | 9.2 |
| 5 | Obtain exposure → mediator edges from a file, from the cohort, or both | 5, 9.3 |
| 6 | Compute counterfactual risks with and without the exposure for every patient | 9.4 |
| 7 | Aggregate effects by pathway, with confidence intervals and sensitivity analyses | 7, 9.5 |

### 1.3 Scope

MoCoP estimates the part of an exposure's effect that is transmitted through the modeled
mediators. Effects transmitted through unobserved pathways are not captured (Section 12).

---

## 2. Installation

Requirements: Python 3.9 or later; numpy, pandas, scipy, scikit-learn and pyyaml (installed
automatically). The optional knowledge-graph module additionally requires matplotlib.

```bash
pip install mocop                    # from PyPI
pip install mocop-1.0.0-py3-none-any.whl   # from a wheel file
pip install "mocop[knowledge]"       # with the optional knowledge-graph module
```

Installation into a dedicated virtual environment is recommended:

```bash
python3 -m venv ~/mocop_env
source ~/mocop_env/bin/activate      # Windows: mocop_env\Scripts\activate
pip install mocop
mocop --version
```

---

## 3. Data specification

MoCoP reads analysis-ready CSV files. Cohort definition, time zero, exposure and outcome are
defined by the analyst before MoCoP is run. Column names are configurable (Section 4.2); the
defaults are shown below.

| File | Required | Grain |
|---|---|---|
| Cohort | Yes | One row per patient |
| Laboratory | Yes | One row per laboratory result |
| Diagnosis | No | One row per diagnosis record |
| Edges | No | One row per exposure → mediator edge (Section 5) |

### 3.1 Conventions

- Identifiers are read as text; leading zeros are preserved. Identifiers must match across files.
- Dates may be in any format parsed by `pandas.to_datetime`. Times of day are ignored.
- Binary fields accept `0/1`, `true/false` and `yes/no` (case-insensitive).
- Day offsets are computed relative to the patient's index date: day 0 is the index date,
  negative days precede it.

### 3.2 Cohort file

| Column (default) | Required | Type | Description |
|---|---|---|---|
| `pid` | Yes | string | Patient identifier. Unique. |
| `index_date` | Yes | date | Time zero: the time at which the exposure decision is made. |
| `outcome` | Yes | binary | Outcome status at the end of follow-up. |
| `exposure` | Yes | binary or missing | 1 = exposed, 0 = comparison group. Missing excludes the patient from the analysis. |
| `eligible` | No | binary | 1 = the patient could have received the exposure at time zero. Missing is treated as 0. |
| `event` | No | binary | Post-exposure event used as a mediator (for example, hospitalization within a fixed window). Missing is treated as 0. |
| other columns | No | numeric or categorical | Pre-exposure covariates. |

Example:

| pid | index_date | outcome | exposure | eligible | event | age | sex | comorbidity_score | region |
|---|---|---|---|---|---|---|---|---|---|
| P000001 | 2021-03-14 | 0 | 1 | 1 | 0 | 67 | F | 3 | north |
| P000002 | 2022-11-02 | 1 | 0 | 1 | 1 | 54 | M | 1 | south |
| P000003 | 2020-07-30 | 0 | | 0 | 0 | 41 | F | 0 | west |

Requirements:

1. One row and one index date per patient. For data with repeated episodes, the index
   episode is selected beforehand, and all cohort values refer to that episode.
2. Covariates are measured before the exposure. Variables that can be affected by the
   exposure must not be supplied as covariates.
3. If `covariates` is not set in the configuration, every column other than the identifier,
   index date, outcome, exposure, eligibility and event columns is used as a covariate.
   Columns parsed as dates, and text columns with more than 50 distinct values, are excluded
   and reported in the log. Listing covariates explicitly is recommended.
4. Missing-value coding must not encode the exposure. A covariate that nearly separates
   exposed from unexposed patients triggers a warning (Section 8.3).

### 3.3 Laboratory file

| Column (default) | Required | Type | Description |
|---|---|---|---|
| `pid` | Yes | string | Patient identifier |
| `date` | Yes | date | Specimen date |
| `code` | Yes | string | Laboratory code. LOINC is recommended. |
| `value` | Yes | numeric | Result. Non-numeric results are ignored. |
| `ref_low` | No | numeric | Lower limit of the reference interval |
| `ref_high` | No | numeric | Upper limit of the reference interval |

Example:

| pid | date | code | value | ref_low | ref_high |
|---|---|---|---|---|---|
| P000001 | 2020-11-02 | 2160-0 | 1.12 | 0.6 | 1.2 |
| P000001 | 2021-03-01 | 1988-5 | 14.3 | 0.0 | 10.0 |
| P000001 | 2021-04-20 | 1988-5 | 6.1 | 0.0 | 10.0 |

Time windows used:

| Window | Days relative to index | Configuration | Used for |
|---|---|---|---|
| Baseline | `-lookback_days` to `baseline_end_day` | `time` | Mediator features (all analyses) |
| Post-exposure | `post_window_days` | `edges` | Training laboratory edges from the cohort only |

Rows outside both windows are ignored. Units must be consistent within a code. When a
reference interval is missing or invalid, the 1st and 99th percentiles of that code among
training patients are used instead.

Laboratory direction. Each code is assigned one of three directions:

| Direction | Meaning | Examples |
|---|---|---|
| `high_bad` | Higher values are clinically worse | C-reactive protein, creatinine |
| `high_good` | Lower values are clinically worse | Hemoglobin, albumin |
| `mid_best` | Both extremes are clinically worse | Potassium, sodium |

A default table covering common LOINC codes is included
(`mocop/data/lab_directions_default.csv`). Codes not in the table are assigned
`features.unknown_lab_direction` (default `mid_best`). A user table (`code,direction[,name]`)
replaces the default when `features.lab_directions` is set.

### 3.4 Diagnosis file

| Column (default) | Required | Type | Description |
|---|---|---|---|
| `pid` | Yes | string | Patient identifier |
| `date` | Yes | date | Record date |
| `code` | Yes | string | Diagnosis code (for example ICD-10). Vocabulary prefixes such as `ICD10:` are removed. |

Only records dated within the baseline window and strictly before `baseline_end_day` are used.
Codes are truncated to `dx_stem_length` characters (default 3, after removing periods), and
groups present in at least `min_dx_coverage` of training patients become frozen features.
Codes listed in `outcome_history_codes` are combined into a single outcome-history feature.

### 3.5 Pre-run checklist

- One row per patient in the cohort file; identifiers match across files.
- The index date marks the exposure decision.
- If the exposure can occur on the index day, `time.baseline_end_day` is set to `-1` so that
  same-day laboratory results are not treated as baseline.
- Covariates are pre-exposure; they are listed in the configuration.
- Laboratory data cover the baseline window and, for trained laboratory edges, the
  post-exposure window.
- `mocop check config.yaml` completes without errors.

---

## 4. Configuration reference

The configuration is a YAML file (or a Python dictionary). Only `data.cohort` and
`data.labs` are required. Relative paths are resolved against the configuration file's
directory. Unknown keys raise an error.

Minimal configuration:

```yaml
data:
  cohort: cohort.csv
  labs: labs.csv
columns:
  outcome: outcome_column_name
  exposure: exposure_column_name
covariates: [age, sex, comorbidity_score]
```

### 4.1 `data`

| Key | Type | Default | Description |
|---|---|---|---|
| `cohort` | path | required | Cohort file |
| `labs` | path | required | Laboratory file |
| `dx` | path | none | Diagnosis file |
| `output_dir` | path | `mocop_output` | Output directory (created if absent) |

### 4.2 `columns`

| Key | Type | Default | Description |
|---|---|---|---|
| `id` | string | `pid` | Cohort identifier column |
| `index_date` | string | `index_date` | Time-zero column |
| `outcome` | string | `outcome` | Outcome column |
| `exposure` | string | `exposure` | Exposure column |
| `eligible` | string | none | Eligibility column; none = all patients eligible |
| `event_mediator` | string | none | Event mediator column; none = laboratory mediators only |
| `labs` | mapping | `{id: pid, date: date, code: code, value: value, ref_low: ref_low, ref_high: ref_high}` | Laboratory file columns |
| `dx` | mapping | `{id: pid, date: date, code: code}` | Diagnosis file columns |

### 4.3 Top level

| Key | Type | Default | Description |
|---|---|---|---|
| `covariates` | list | none | Pre-exposure cohort columns; none = automatic selection (Section 3.2) |
| `categorical` | list | `[]` | Covariates treated as categorical regardless of type |

### 4.4 `time`

| Key | Type | Default | Description |
|---|---|---|---|
| `lookback_days` | int | 730 | Length of the baseline window before the index date |
| `baseline_end_day` | int | 0 | Last day included in the baseline window |
| `trend_split_days` | int | 180 | Boundary between the recent and earlier parts of the baseline window (slope feature) |

### 4.5 `features`

| Key | Type | Default | Description |
|---|---|---|---|
| `min_lab_coverage` | float | 0.35 | Minimum share of training patients with a baseline value for a laboratory code to become a mediator |
| `min_dx_coverage` | float | 0.05 | Minimum share of training patients with a diagnosis group for it to become a feature |
| `dx_stem_length` | int | 3 | Characters retained from diagnosis codes |
| `dx_strip_vocabulary` | bool | true | Remove vocabulary prefixes from diagnosis codes |
| `outcome_history_codes` | list | `[]` | Diagnosis code prefixes indicating prior occurrence of the outcome |
| `lab_directions` | path | none | User direction table; none = built-in table |
| `unknown_lab_direction` | string | `mid_best` | Direction for codes absent from the table |
| `max_lab_codes` | int | 200 | Maximum number of laboratory mediators (highest coverage retained) |

### 4.6 `edges`

| Key | Type | Default | Description |
|---|---|---|---|
| `modes` | list | `[trained]` | Any of `curated`, `trained`; all listed modes are reported |
| `file` | path | none | Curated edges file (Section 5.2); required for `curated` |
| `candidate_labs` | list | none | Laboratory codes tested as mediators; none = all laboratory mediators |
| `confounders` | list | none | Adjustment set for trained edges; none = `covariates` |
| `post_window_days` | [int, int] | [7, 90] | Post-exposure window for trained laboratory edges |
| `rule` | string | `propagate` | `propagate`: use every edge passing the checks; `threshold`: additionally require p < `p_threshold` |
| `p_threshold` | float | 0.05 | Significance level for `rule: threshold` |
| `min_treated` | int | 30 | Minimum number of exposed patients per edge |
| `require_balance` | bool | true | Exclude edges failing the balance or overlap check |
| `ps_truncate` | [float, float] | [0.001, 0.999] | Propensity score truncation limits |
| `balance_smd_max` | float | 0.10 | Maximum absolute standardized mean difference after weighting |
| `extreme_ps_max` | float | 0.05 | Maximum share of patients with propensity scores outside `ps_truncate` |
| `separation_smd_warn` | float | 1.0 | Standardized difference before weighting above which a separation warning is issued |

### 4.7 `model`

| Key | Type | Default | Description |
|---|---|---|---|
| `monotone` | bool | true | Constrain mediator coefficients to be non-negative |
| `constrain_slope` | bool | true | Apply the constraint to slope and range features as well as levels |
| `platt` | bool | true | Platt calibration on the validation set |
| `l2` | float | 0.001 | Ridge penalty |
| `split` | [float, float, float] | [0.70, 0.15, 0.15] | Training, validation and test fractions |
| `seed` | int | 42 | Random seed for the split and resampling |

### 4.8 `uncertainty`

| Key | Type | Default | Description |
|---|---|---|---|
| `n_boot` | int | 500 | Bootstrap resamples of patients |
| `n_psa` | int | 300 | Draws of the edges from their sampling distributions |
| `psa_subset` | int | 20000 | Patients used per edge draw |

### 4.9 `sensitivity`

| Key | Type | Default | Description |
|---|---|---|---|
| `scan_gammas` | list | [0.05, 0.10, 0.20, 0.30, 0.40, 0.50] | Uniform laboratory improvements evaluated in the scan |
| `tornado_pct` | float | 0.5 | Relative change applied to each edge in the tornado analysis |

### 4.10 `knowledge` (optional)

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | false | Enable the knowledge-graph module |
| `primekg_dir` | path | none | Directory containing PrimeKG `kg.csv` |
| `concept_map` | path | none | CSV (`code,name`) giving a concept name for each laboratory code |
| `exposure_name` | string | none | Exposure name as it appears in PrimeKG |
| `restrict_candidates` | bool | false | Test only laboratory codes with knowledge-graph support |

### 4.11 `output`

| Key | Type | Default | Description |
|---|---|---|---|
| `suppress_small_counts` | int | 11 | Counts from 1 to this value minus 1 are written as `<11`; 0 disables suppression |
| `save_patient_level` | bool | false | Write patient-level risks for the test set (Section 11) |
| `chunk_rows` | int | 5000000 | Rows read per chunk from large files |

---

## 5. Edge specification

### 5.1 Definition

An edge links the exposure to one mediator.

| Kind | Scale | Sign convention |
|---|---|---|
| Laboratory | Shift on the severity scale (range -1 to 1) | Negative: the exposure improves the laboratory value |
| Event | Risk ratio | Below 1: the exposure reduces the event risk |

Without edges, every estimated effect is zero by construction.

### 5.2 Curated edges

Curated edges are supplied in a CSV file, typically from randomized trials or published
estimates.

| Column | Required | Description |
|---|---|---|
| `mediator` | Yes | Laboratory code as in the laboratory file, or any label for the event |
| `kind` | Yes | `lab` or `event` |
| `value` | Yes | Laboratory: severity shift. Event: risk ratio. |
| `se` | No | Laboratory: standard error of the shift. Event: standard error of the log risk ratio. |
| `ci_low`, `ci_high` | No | 95% confidence interval on the scale of `value` (used when `se` is absent) |

Example:

| mediator | kind | value | se |
|---|---|---|---|
| 33762-6 | lab | -0.30 | 0.05 |
| 2823-3 | lab | 0.06 | |
| event | event | 0.70 | 0.08 |

In `curated` mode, mediators absent from the file are trained from the cohort (Section 5.3).

### 5.3 Trained edges

Trained edges are estimated from the cohort with an augmented inverse probability weighting
(AIPW) estimator (Section 9.3). Estimation uses eligible training patients only; the edges
are fixed before any counterfactual is computed. An edge is used only if:

1. at least `min_treated` exposed patients contribute;
2. with `require_balance: true`, all confounders have an absolute standardized mean
   difference of at most `balance_smd_max` after weighting, and at most `extreme_ps_max` of
   patients have propensity scores outside `ps_truncate`;
3. with `rule: threshold`, p < `p_threshold`.

The reason for excluding an edge is reported in `edges_trained.csv`.

### 5.4 Selecting a mode

| Situation | Recommended `modes` |
|---|---|
| Published estimates exist for the principal edges | `[curated]` |
| No published estimates | `[trained]` |
| Comparison of published and cohort-derived edges | `[curated, trained]` |

### 5.5 Knowledge-graph support (optional)

When `knowledge.enabled` is true, MoCoP reports for each laboratory mediator whether PrimeKG
links the exposure to the laboratory's concept directly (`direct`), through one intermediate
node (`2 hops`), or not at all (`none`), using case-insensitive text matching. The module does
not set edge magnitudes. With `restrict_candidates: true`, only laboratory codes with support
are tested as mediators.

---

## 6. Running MoCoP

### 6.1 Command-line interface

| Command | Description |
|---|---|
| `mocop check CONFIG` | Read and validate the configuration and input files; no model is fitted |
| `mocop run CONFIG [--quiet]` | Run the full pipeline |
| `mocop example FOLDER [--patients N]` | Write a synthetic dataset, configuration and true values to FOLDER |
| `mocop --version` | Print the version |

Exit status is 0 on success and 1 on a data or configuration error.

### 6.2 Python API

```python
from mocop import run, load_config

res = run("config.yaml", verbose=True)    # path or dict
res.summary()                             # str: main estimates, all patients
res["effects"]                            # pandas.DataFrame
res.tables.keys()                         # all output tables
res.log                                   # list of log messages
res.output_dir                            # str
```

`run(config, verbose=True) -> Results`

| Parameter | Type | Description |
|---|---|---|
| `config` | str, Path or dict | Configuration file or dictionary |
| `verbose` | bool | Print progress messages |

`Results.tables` contains the DataFrames listed in Section 7 (keys without the `.csv` suffix).

`load_config(config) -> Config` returns the validated configuration with defaults applied.

### 6.3 Resources

Measured on a synthetic dataset of 119,381 patients, 20 million laboratory rows and 62
laboratory codes: 292 seconds and 3.3 GB peak memory on a single CPU core. Memory scales with the number of patients
times the number of features; `output.chunk_rows` controls the memory used while reading.

---

## 7. Output reference

All files are written to `data.output_dir`. All files are aggregate (Section 11).

| File | Content |
|---|---|
| `effects.csv` | Effect estimates by mode, estimand and group |
| `edges_locked.csv` | Edges used in each mode (the causal graph) |
| `edges_trained.csv` | All edges estimated from the cohort, with diagnostics |
| `edges_balance.csv` | Standardized mean differences per confounder and edge |
| `model_performance.csv` | Outcome and event model performance |
| `outcome_model_weights.csv` | Standardized outcome model coefficients |
| `mediator_weights.csv` | Coefficients and coverage per laboratory mediator |
| `directionality_audit.csv` | Monotonicity checks |
| `sensitivity_scan.csv` | Effect under uniform laboratory improvements |
| `sensitivity_tornado.csv` | Effect under one-at-a-time changes to each edge |
| `knowledge_support.csv`, `graph.png` | Knowledge-graph module only |
| `run_info.json` | Version, run time, configuration (file names only), counts and log |

Feature names: `COV_<column>` and `COV_<column>=<level>` (covariates), `DX_<group>`,
`DX_OUTCOME_HISTORY`, `EVENT`, `LAB_<code>_now`, `LAB_<code>_slope`, `LAB_<code>_trend`.

### 7.1 `effects.csv`

| Column | Description |
|---|---|
| `mode` | `curated` or `trained` |
| `estimand` | Pathway (Section 8.1) |
| `description` | Text description of the estimand |
| `group` | Patient group (Section 8.2) |
| `n` | Number of patients |
| `risk_unexposed`, `risk_exposed` | Mean counterfactual risk without and with the exposure |
| `risk_difference` | `risk_exposed - risk_unexposed` |
| `rd_boot_low`, `rd_boot_high` | 95% bootstrap interval of the risk difference (patients only) |
| `risk_ratio`, `rr_ci_low`, `rr_ci_high` | Risk ratio and 95% confidence interval |
| `log_rr`, `se_log_rr` | Log risk ratio and its standard error |
| `se_log_rr_patients`, `se_log_rr_edges` | Components of `se_log_rr` |
| `odds_ratio` | Marginal odds ratio |
| `pct_patients_benefit` | Percentage of patients whose counterfactual risk decreases |

### 7.2 `edges_trained.csv`

| Column | Description |
|---|---|
| `mediator`, `kind` | Mediator (`LAB_<code>` or `EVENT`) and edge kind |
| `n`, `n_treated` | Patients and exposed patients contributing |
| `estimate`, `se`, `ci_low`, `ci_high`, `p` | Edge estimate (laboratory: severity shift; event: log risk ratio) |
| `max_smd_before`, `max_smd_after` | Largest absolute standardized mean difference before and after weighting |
| `pct_extreme_ps` | Share of patients with propensity scores outside `ps_truncate` |
| `ess_treated` | Effective sample size of the weighted exposed group |
| `balance_ok`, `used`, `reason_not_used` | Check result, inclusion and reason for exclusion |
| `risk_ratio`, `rr_ci_low`, `rr_ci_high`, `odds_ratio` | Event edges only |
| `risk_if_all_exposed`, `risk_if_none_exposed` | Event edges only: standardized event risks |

### 7.3 Other files

| File | Columns |
|---|---|
| `edges_locked.csv` | `mode, mediator, name, kind, scale, value, se, note` |
| `edges_balance.csv` | `mediator, covariate, smd_before, smd_after` |
| `model_performance.csv` | `model, split, n, auc, brier, ece, prevalence, mean_predicted` |
| `outcome_model_weights.csv` | `feature, weight, constrained_non_negative` |
| `mediator_weights.csv` | `lab, name, baseline_coverage, weight_now, weight_slope, weight_trend, has_edge` |
| `directionality_audit.csv` | `check, change_in_mean_risk, ok` |
| `sensitivity_scan.csv` | `gamma, risk_difference, risk_ratio, pct_patients_benefit` |
| `sensitivity_tornado.csv` | `mode, edge, edge_value, rr_base, rr_edge_low, rr_edge_high, range` |
| `knowledge_support.csv` | `code, concept, support, via` |

---

## 8. Interpretation

### 8.1 Estimands

| Estimand | Definition |
|---|---|
| `total` | Effect through laboratory mediators and the event |
| `labs` | Effect through laboratory mediators; event at its natural course in both arms |
| `event` | Effect through the event only |
| `event_held` | Effect through laboratory mediators with the event fixed at each patient's observed value; corresponds to an analysis adjusted for the event |

Without an event mediator, only `total` is reported.

### 8.2 Groups

| Group | Patients |
|---|---|
| `all patients` | All included patients, including the training set |
| `test patients` | Test set |
| `unexposed test patients` | Unexposed patients in the test set |
| `eligible unexposed test patients` | As above, eligible only (reported when an eligibility column is set) |

### 8.3 Diagnostics

The following diagnostics are reviewed before the effect estimates.

| Diagnostic | Location | Expected |
|---|---|---|
| Warnings | `run_info.json`, field `log` | Each warning addressed |
| Separation warning | log | None; otherwise review the coding of the named covariate |
| Edge inclusion | `edges_trained.csv` | Principal edges used; exclusions explained |
| Outcome model calibration | `model_performance.csv`, `ece` | Small relative to prevalence |
| Mediator coefficients | `mediator_weights.csv` | Edges to mediators with zero coefficient cannot transmit an effect |
| Monotonicity | `directionality_audit.csv` | `ok` true for all checks |

### 8.4 Null and small effects

An effect near zero indicates that the exposure does not change the outcome through the
modeled mediators. It is not evidence that the exposure has no effect through other
pathways. The `labs` and `event` estimands identify which pathway carries the effect, and
`sensitivity_scan.csv` indicates the magnitude of laboratory effect required to change the
outcome.

---

## 9. Methods

### 9.1 Severity scale and laboratory features

Each laboratory value v is normalized to [0, 1] with its reference interval,
v_n = clip((v - ref_low) / (ref_high - ref_low), 0, 1), or with the 1st and 99th training
percentiles of its code when no valid interval is available. The signed severity s in
[-1, 1] is 2(v_n - 0.5) for `high_bad`, -2(v_n - 0.5) for `high_good` and 2|v_n - 0.5| for
`mid_best`. For each laboratory code and patient, three baseline features are computed: the
most recent severity (`now`), the mean severity in the last `trend_split_days` days minus the
mean severity earlier in the baseline window (`slope`), and the range of severities
(`trend`). Patients without a baseline value receive 0.

### 9.2 Outcome model

A logistic regression with class weights (inverse outcome prevalence) and an L2 penalty is
fitted to standardized features of the training set by L-BFGS-B. Coefficients of all mediator
features are bounded below by 0, so that worse mediator values cannot reduce predicted risk.
Coefficients of frozen features are unconstrained. Predicted risks are calibrated by Platt
scaling on the validation set. The exposure is not a model input.

### 9.3 Edge estimation

For each candidate mediator, the average effect of the exposure among eligible training
patients is estimated by AIPW. A logistic propensity model for the exposure and an outcome
model for the mediator (linear regression for laboratory changes, logistic regression for the
event) are fitted on the confounders. Laboratory edges use the change from the last baseline
severity to the mean severity in the post-exposure window and additionally adjust for the
baseline severity. Propensity scores are truncated to `ps_truncate`. For the event, the
estimand is the log risk ratio of the standardized event risks. Standard errors are derived
from the influence function. Balance is assessed by standardized mean differences under
inverse probability weights.

### 9.4 Counterfactual propagation

For each patient, the baseline severity of each laboratory mediator is extrapolated along
the patient's slope. The extrapolation step is the largest fraction (up to 1) of the slope
that keeps both arms within [-1, 1]. The exposed arm adds the laboratory edges; ineligible
patients receive no edge. With an event mediator, the natural-course event probability p is
estimated from frozen and baseline laboratory features by a logistic model fitted on
unexposed eligible training patients. Risk in each arm is p r(1) + (1 - p) r(0), where r(e)
is the outcome model's risk with the event set to e; in the exposed arm, p is multiplied by
the event risk ratio. Effects are means of the patient-level risks.

### 9.5 Uncertainty and sensitivity

The standard error of the log risk ratio combines a nonparametric bootstrap over patients
(edges and model fixed) with Monte Carlo draws of every edge from a normal distribution with
its estimated standard error (model fixed): se² = se²_patients + se²_edges. Uncertainty from
fitting the outcome model is not included. The sensitivity scan applies a uniform improvement
gamma to all laboratory mediators. The tornado analysis multiplies each edge (the log risk
ratio for the event) by 1 ± `tornado_pct`.

---

## 10. Data partitioning and leakage control

Patients are partitioned into training, validation and test sets (default 70/15/15,
stratified by outcome) before any quantity is estimated. The partition depends only on the
outcome vector and `model.seed`.

| Quantity | Estimated on |
|---|---|
| Laboratory percentile limits (severity fallback) | Training patients |
| Laboratory and diagnosis coverage selection | Training patients |
| Covariate encoding (fill values, category levels, reference levels) | Training patients |
| Feature standardization | Training patients |
| Outcome model coefficients | Training patients |
| Platt calibration | Validation patients |
| Event natural-course model | Unexposed eligible training patients |
| Trained edges, propensity models, balance diagnostics | Eligible training patients |
| Reported model performance | Validation and test patients |

Category levels that do not occur among training patients are coded as `other` or as the
reference level. Post-exposure laboratory values are used only to train laboratory edges
and are never inputs to the outcome model. The package test suite includes a test that alters
the laboratory values, covariates and category levels of all validation and test patients
and verifies that every learned quantity is unchanged.

---

## 11. Data protection

- MoCoP runs locally and makes no network connections.
- Input files are read only; they are never modified or copied.
- Default outputs contain no identifiers, dates or patient-level values.
- Counts from 1 to `suppress_small_counts - 1` are written as `<11` (default threshold 11).
- `run_info.json` records file names only, not directory paths.
- `output.save_patient_level: true` writes identifiers and patient-level risks for the test
  set. It is disabled by default and should be enabled only under an approved data-use
  agreement.

---

## 12. Assumptions and limitations

1. Mediation. Only effects transmitted through the modeled mediators are estimated.
2. Confounding. Trained edges assume no unmeasured confounding of the exposure-mediator
   relationship given the supplied confounders and positivity among eligible patients.
   Curated edges assume transportability of the published estimates to the study
   population.
3. Monotonicity. Worse mediator values are assumed not to decrease outcome risk.
4. Baseline representation. The outcome model represents mediators by baseline values;
   effects acting through post-exposure laboratory changes can be attenuated.
5. Design. One time zero per patient; binary exposure and outcome; at most one event
   mediator.
6. Uncertainty. Confidence intervals include patient sampling and edge uncertainty but not
   uncertainty from fitting the outcome model, and may therefore be too narrow.
7. Comparability with other analyses. Comparisons with other estimators require identical
   patients, exposure and outcome definitions, and a matching estimand (Section 8.1).

---

## 13. Troubleshooting

| Message | Cause | Resolution |
|---|---|---|
| `Cohort file is missing column(s)` | Column names differ from the configuration | Set the names under `columns` |
| `Cohort must have one row per patient` | Duplicated identifiers | Select one index episode per patient |
| `Column 'x' must be binary` | Values other than 0/1, true/false, yes/no | Recode the column |
| `No lab rows fall inside the time window` | Identifier mismatch, unparsed dates, or no baseline data | Check identifiers, date format and window settings |
| `WARNING: confounder ... almost separates` | A covariate level nearly determines exposure | Review the coding of the covariate |
| `WARNING: no edges locked` | No edge passed the checks | Inspect `reason_not_used` in `edges_trained.csv` |
| `WARNING: ... event zero weight` | No association between event and outcome in the model | The event pathway transmits no effect |
| `failed balance/overlap check` | Exposed and unexposed patients are not comparable for that edge | Revise confounders, define eligibility, or supply a curated edge |
| Memory error | Insufficient memory while reading | Reduce `output.chunk_rows` or restrict the laboratory file |

---

## 14. Demo

A synthetic demonstration dataset with known true values is provided in `examples/demo`.
No real patient data are included.

| File | Description |
|---|---|
| `cohort.csv`, `labs.csv`, `dx.csv` | 3,000 simulated patients |
| `edges_curated.csv` | One curated event edge |
| `config.yaml` | Configuration |
| `truth.json` | True edge values and total effect |
| `run_demo.py` | Walkthrough using the Python API |
| `demo.ipynb` | The same walkthrough as a Jupyter notebook |
| `EXPECTED_RESULTS.md` | Reference output |

Data-generating process: exposure depends on age, comorbidity score and prior events; it
lowers post-exposure C-reactive protein (LOINC 1988-5) and multiplies the risk of a binary
event by 0.60; the outcome depends on C-reactive protein, creatinine, the event and
comorbidity.

```bash
cd examples/demo
python run_demo.py            # or: mocop run config.yaml
```

Reference results (MoCoP 1.0.0):

| Quantity | Estimate | True value |
|---|---|---|
| C-reactive protein edge (severity shift) | -0.306 (95% CI -0.394, -0.219) | -0.367 |
| Event edge (risk ratio) | 0.746 (95% CI 0.515, 1.080) | 0.600 |
| Total effect, trained edges (risk ratio) | 0.914 (95% CI 0.876, 0.954) | 0.791 |

Edges are recovered within their confidence intervals. The total effect is attenuated toward
1 as described in Section 12, item 4. A dataset of any size can be generated with
`mocop example FOLDER --patients N`.

---

## 15. Versioning and reproducibility

MoCoP follows semantic versioning. Results are determined by the package version, the
configuration and the input data. `run_info.json` records the version and the resolved
configuration; `edges_locked.csv` records the causal graph used. These two files constitute
the record of an analysis and can be archived before comparison with other estimators. A
specific version is installed with `pip install mocop==1.0.0`.

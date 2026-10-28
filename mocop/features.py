"""Feature matrix (MoCoP variable taxonomy).

FROZEN       pre-exposure characteristics, identical in both counterfactual arms:
             cohort covariates, baseline diagnosis code stems, outcome history flag.
CONSEQUENCE  mediators the exposure can move:
             labs, summarised over the baseline window as
               now   = most recent severity up to the end of baseline
               slope = mean severity in the recent part minus the earlier part
               trend = max - min severity
             and, optionally, one binary secondary-outcome EVENT (e.g. hospitalization).
OUTCOME      binary outcome. The exposure itself is NOT a model input.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EVENT = "EVENT"


@dataclass
class FeatureSet:
    X: np.ndarray
    cols: list
    col_idx: dict
    outcome_cols: list            # columns used by the outcome model
    of_idx: list
    mono: np.ndarray              # +1 = coefficient constrained >= 0
    frozen_cols: list
    lab_codes: list               # labs with features (order = lab channel)
    lab_coverage: pd.Series
    has_event: bool
    covariate_cols: list = field(default_factory=list)

    def now_idx(self):
        return [self.col_idx[f"LAB_{c}_now"] for c in self.lab_codes]

    def slope_idx(self):
        return [self.col_idx[f"LAB_{c}_slope"] for c in self.lab_codes]


def encode_covariates(df: pd.DataFrame, covs: list, categorical: list, fit_rows=None):
    """Numeric columns as is (missing filled with the training median); text or
    listed-categorical columns one-hot (top 30 training levels, others pooled as 'other',
    most frequent training level dropped). fit_rows: boolean mask of training patients;
    every fill value and level set is learned from those rows only."""
    fit = np.ones(len(df), bool) if fit_rows is None else np.asarray(fit_rows, bool)
    out = {}
    for c in covs:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        if c not in categorical and num[fit].notna().mean() >= 0.95:
            med = num[fit].median() if num[fit].notna().any() else 0.0
            out[c] = num.fillna(med).values.astype(float)
            continue
        lev = s.astype(str).where(s.notna(), "missing").str.strip()
        top = lev[fit].value_counts().index[:30]
        lev = lev.where(lev.isin(top), "other")
        ref = lev[fit].value_counts().index[0]
        # columns come from training levels only; a level never seen in training is
        # pooled into 'other' (or, if training has no 'other', coded like the reference)
        for k in sorted(lev[fit].unique()):
            if k != ref:
                out[f"{c}={k}"] = (lev == k).values.astype(float)
    return out


def build_features(co, labs: pd.DataFrame, dx, cfg, log: list, train_mask=None) -> FeatureSet:
    """Coverage thresholds and covariate encodings are learned from training patients
    (train_mask); the resulting features are then computed for every patient."""
    N = co.n
    train = np.ones(N, bool) if train_mask is None else np.asarray(train_mask, bool)
    n_train = int(train.sum())
    fcfg, tcfg = cfg["features"], cfg["time"]
    cols = {}

    # ---- FROZEN: covariates ----
    cov_feats = encode_covariates(co.df, co.covariates, list(cfg["categorical"] or []), fit_rows=train)
    cols.update({f"COV_{k}": v for k, v in cov_feats.items()})

    # ---- FROZEN: baseline diagnosis stems and outcome history ----
    dx_cols = []
    if dx is not None and len(dx):
        L = int(fcfg["dx_stem_length"])
        stems = dx.assign(stem=dx["code"].str.replace(".", "", regex=False).str[:L])[["i", "stem"]].drop_duplicates()
        n_by = stems[train[stems["i"].values]].groupby("stem")["i"].nunique()
        keep = sorted(n_by[n_by >= fcfg["min_dx_coverage"] * n_train].index)
        hist_prefixes = [str(p).upper().replace(".", "") for p in (fcfg["outcome_history_codes"] or [])]
        if hist_prefixes:
            keep = [s for s in keep if not any(p.startswith(s) or s.startswith(p) for p in hist_prefixes)]
        g = stems[stems["stem"].isin(keep)]
        for s, sub in g.groupby("stem"):
            v = np.zeros(N)
            v[sub["i"].values] = 1.0
            cols[f"DX_{s}"] = v
            dx_cols.append(f"DX_{s}")
        if hist_prefixes:
            code_nd = dx["code"].str.replace(".", "", regex=False)
            hit = dx.loc[code_nd.str.startswith(tuple(hist_prefixes)), "i"].unique()
            v = np.zeros(N)
            v[hit] = 1.0
            cols["DX_OUTCOME_HISTORY"] = v
            dx_cols.append("DX_OUTCOME_HISTORY")
        log.append(f"features: {len(dx_cols)} diagnosis features (stems in >= "
                   f"{fcfg['min_dx_coverage']:.0%} of training patients)")

    # ---- CONSEQUENCE: labs ----
    base = labs[labs["day"] <= int(tcfg["baseline_end_day"])]
    cov = (base[train[base["i"].values]].groupby("code", observed=True)["i"].nunique()
           .sort_values(ascending=False) / n_train)
    lab_codes = [str(c) for c in cov[cov >= fcfg["min_lab_coverage"]].index]
    forced = [str(c) for c in (cfg["edges"]["candidate_labs"] or [])]
    for c in forced:
        if c in cov.index and cov[c] > 0 and c not in lab_codes:
            lab_codes.append(c)
            log.append(f"lab {c}: baseline coverage {cov[c]:.1%}, kept because it is a listed candidate mediator")
        elif c not in cov.index:
            log.append(f"WARNING: candidate lab {c} has no baseline values; it cannot be a mediator")
    if len(lab_codes) > int(fcfg["max_lab_codes"]):
        lab_codes = lab_codes[: int(fcfg["max_lab_codes"])]
    log.append(f"features: {len(lab_codes)} labs (baseline coverage in training patients >= {fcfg['min_lab_coverage']:.0%}"
               f"{' or listed' if forced else ''})")

    cat_pos = {str(c): k for k, c in enumerate(labs["code"].cat.categories)}
    bl = base[base["code"].astype(str).isin(lab_codes)][["i", "code", "day", "bad"]]
    bl = bl.assign(lc=bl["code"].cat.codes.astype(np.int32), dbi=-bl["day"]).drop(columns=["code", "day"])
    keys = ["i", "lc"]
    split = int(tcfg["trend_split_days"])
    now = bl.sort_values("dbi").groupby(keys)["bad"].first()
    rec = bl[bl["dbi"] <= split].groupby(keys)["bad"].mean()
    early = bl[bl["dbi"] > split].groupby(keys)["bad"].mean()
    slope = rec.reindex(now.index).fillna(now) - early.reindex(now.index).fillna(now)
    grp = bl.groupby(keys)["bad"]
    trend = grp.max() - grp.min()
    for c in lab_codes:
        k = cat_pos[c]
        for nm, ser in (("now", now), ("slope", slope), ("trend", trend)):
            v = np.zeros(N)
            try:
                s_ = ser.xs(k, level="lc")
                v[s_.index.values.astype(np.int64)] = s_.values
            except KeyError:
                pass
            cols[f"LAB_{c}_{nm}"] = v

    # ---- CONSEQUENCE: event mediator ----
    has_event = co.event is not None
    if has_event:
        cols[EVENT] = co.event.astype(float)

    frozen_cols = list(f"COV_{k}" for k in cov_feats) + dx_cols
    lab_cols = [f"LAB_{c}_{nm}" for nm in ("now", "slope", "trend") for c in lab_codes]
    all_cols = frozen_cols + ([EVENT] if has_event else []) + lab_cols
    X = np.column_stack([cols[c] for c in all_cols]).astype(float)
    col_idx = {c: j for j, c in enumerate(all_cols)}
    mcfg = cfg["model"]

    def mono_of(c):
        if not mcfg["monotone"]:
            return 0
        if c == EVENT:
            return 1
        if c.startswith("LAB_"):
            kind = c.rsplit("_", 1)[-1]
            return 1 if (kind == "now" or mcfg["constrain_slope"]) else 0
        return 0

    mono = np.array([mono_of(c) for c in all_cols], dtype=int)
    log.append(f"features: X {X.shape[0]:,} x {X.shape[1]} | frozen {len(frozen_cols)}, "
               f"event {int(has_event)}, lab features {len(lab_cols)}")
    return FeatureSet(X=X, cols=all_cols, col_idx=col_idx, outcome_cols=all_cols,
                      of_idx=list(range(len(all_cols))), mono=mono, frozen_cols=frozen_cols,
                      lab_codes=lab_codes, lab_coverage=cov, has_event=has_event,
                      covariate_cols=[f"COV_{k}" for k in cov_feats])

"""Edges of the causal graph: exposure -> mediator, with magnitude and direction.

Two sources, usable alone or side by side (edges.modes):
  curated : numbers imported from a CSV (literature, trials, a knowledge base)
  trained : learned from the cohort with a doubly robust (AIPW) estimator

Trained edges use the TRAINING split only and are locked before any counterfactual is
computed. Only eligible patients are used; confounders must be pre-exposure.
In 'curated' mode, any edge without a curated value is trained.

Edge scales:
  lab edge   : shift on the severity scale (positive = exposure makes the lab worse)
  event edge : risk ratio of the event (stored as log RR)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .features import EVENT, encode_covariates


def _sig(v):
    return 1 / (1 + np.exp(-np.clip(v, -30, 30)))


def logit_fit(Xd, yv, iters=100, lam=1e-4):
    """Logistic regression by Newton-Raphson with a tiny ridge (intercept unpenalised)."""
    b = np.zeros(Xd.shape[1])
    R = lam * np.eye(Xd.shape[1])
    R[0, 0] = 0
    H = None
    for _ in range(iters):
        p = _sig(Xd @ b)
        H = Xd.T @ (Xd * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H, Xd.T @ (yv - p) - R @ b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b, np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))


def smd(Z, t, w=None):
    w = np.ones(len(t)) if w is None else w
    out = np.zeros(Z.shape[1])
    for j in range(Z.shape[1]):
        z1, z0, w1, w0 = Z[t == 1, j], Z[t == 0, j], w[t == 1], w[t == 0]
        m1, m0 = np.average(z1, weights=w1), np.average(z0, weights=w0)
        v1 = np.average((z1 - m1) ** 2, weights=w1)
        v0 = np.average((z0 - m0) ** 2, weights=w0)
        out[j] = abs(m1 - m0) / np.sqrt((v1 + v0) / 2 + 1e-12)
    return out


def aipw(yv, t, Z, binary, ps_truncate=(0.001, 0.999)):
    """Doubly robust average effect of t on yv.
    binary=True : log risk ratio (also log odds ratio); otherwise mean difference."""
    n = len(yv)
    one = np.ones((n, 1))
    Xp = np.hstack([one, Z])
    bps, _ = logit_fit(Xp, t)
    e_raw = _sig(Xp @ bps)
    e = np.clip(e_raw, *ps_truncate)
    Xo = np.hstack([one, t[:, None], Z])
    X1, X0 = Xo.copy(), Xo.copy()
    X1[:, 1], X0[:, 1] = 1, 0
    if binary:
        bo, _ = logit_fit(Xo, yv)
        m1, m0 = _sig(X1 @ bo), _sig(X0 @ bo)
    else:
        bo = np.linalg.lstsq(Xo, yv, rcond=None)[0]
        m1, m0 = X1 @ bo, X0 @ bo
    psi1 = m1 + t * (yv - m1) / e
    psi0 = m0 + (1 - t) * (yv - m0) / (1 - e)
    mu1, mu0 = psi1.mean(), psi0.mean()
    r = dict(mu1=mu1, mu0=mu0)
    if binary:
        a1, a0 = np.clip(mu1, 1e-6, 1 - 1e-6), np.clip(mu0, 1e-6, 1 - 1e-6)
        r["est"] = np.log(a1 / a0)
        r["se"] = ((psi1 - mu1) / a1 - (psi0 - mu0) / a0).std(ddof=1) / np.sqrt(n)
        r["log_or"] = np.log(a1 / (1 - a1)) - np.log(a0 / (1 - a0))
        r["se_log_or"] = ((psi1 - mu1) / (a1 * (1 - a1)) - (psi0 - mu0) / (a0 * (1 - a0))).std(ddof=1) / np.sqrt(n)
    else:
        r["est"] = mu1 - mu0
        r["se"] = ((psi1 - mu1) - (psi0 - mu0)).std(ddof=1) / np.sqrt(n)
    w = t / e + (1 - t) / (1 - e)
    r["smd_before"], r["smd_after"] = smd(Z, t), smd(Z, t, w)
    r["pct_extreme_ps"] = float(((e_raw < ps_truncate[0]) | (e_raw > ps_truncate[1])).mean())
    r["ess_treated"] = float(w[t == 1].sum() ** 2 / (w[t == 1] ** 2).sum())
    return r


# ----------------------------------------------------------------------------------
def read_edges_file(path, lab_codes, event_name, log):
    """Curated edges CSV. Columns: mediator, kind (lab|event), value, optional se or
    ci_low/ci_high. Lab value = severity shift; event value = risk ratio."""
    t = pd.read_csv(path, dtype={"mediator": str})
    t.columns = [c.strip().lower() for c in t.columns]
    need = {"mediator", "kind", "value"}
    if not need <= set(t.columns):
        raise ValueError(f"edges file needs columns {sorted(need)}; found {list(t.columns)}")
    t["kind"] = t["kind"].str.strip().str.lower()
    t["mediator"] = t["mediator"].str.strip()
    bad = t[~t["kind"].isin({"lab", "event"})]
    if len(bad):
        raise ValueError(f"edges file: kind must be 'lab' or 'event'; found {bad['kind'].unique()}")
    labs, event = {}, None
    for r in t.itertuples(index=False):
        se = float(getattr(r, "se")) if "se" in t.columns and pd.notna(getattr(r, "se", np.nan)) else 0.0
        if r.kind == "lab":
            if "ci_low" in t.columns and pd.notna(r.ci_low) and se == 0:
                se = (float(r.ci_high) - float(r.ci_low)) / (2 * 1.96)
            if r.mediator not in lab_codes:
                log.append(f"WARNING: curated lab edge {r.mediator} ignored (no features for this lab: "
                           f"missing or below coverage)")
                continue
            labs[r.mediator] = (float(r.value), se)
        else:
            if event_name is None:
                log.append("WARNING: curated event edge ignored (no event mediator column configured)")
                continue
            rr = float(r.value)
            if rr <= 0:
                raise ValueError("edges file: event value must be a risk ratio > 0")
            if "ci_low" in t.columns and pd.notna(r.ci_low) and se == 0:
                se = (np.log(float(r.ci_high)) - np.log(float(r.ci_low))) / (2 * 1.96)
            event = (np.log(rr), se)
    log.append(f"curated edges read: {len(labs)} lab, {int(event is not None)} event")
    return labs, event


def train_edges(co, fs, labs, cfg, tr, log):
    """AIPW edges on eligible training patients. Returns (table, balance table, locked dict)."""
    ecfg = cfg["edges"]
    conf = ecfg["confounders"] if ecfg["confounders"] is not None else co.covariates
    miss = [c for c in conf if c not in co.df.columns]
    if miss:
        raise ValueError(f"edges.confounders not found in cohort: {miss}")
    train_mask = np.zeros(co.n, bool)
    train_mask[tr] = True
    Zf = encode_covariates(co.df, conf, list(cfg["categorical"] or []), fit_rows=train_mask)
    znames = list(Zf)
    Zc = np.column_stack([Zf[k] for k in znames]) if znames else np.zeros((co.n, 0))
    if Zc.shape[1]:
        mu, sd = Zc[tr].mean(0), Zc[tr].std(0) + 1e-9
        Zc = (Zc - mu) / sd
    T = co.t
    train = np.zeros(co.n, bool)
    train[tr] = True
    elig = co.eligible == 1
    trunc = tuple(ecfg["ps_truncate"])
    rows_out, bal_out, locked = [], [], {"labs": {}, "event": None}
    sep_warned = set()

    def record(edge, kind, n, t_, r, names):
        n_t = int(t_.sum())
        p = 2 * stats.norm.sf(abs(r["est"] / r["se"])) if r["se"] > 0 else np.nan
        bal_ok = (r["smd_after"].max(initial=0) <= ecfg["balance_smd_max"]) and \
                 (r["pct_extreme_ps"] <= ecfg["extreme_ps_max"])
        reason = ""
        if n_t < ecfg["min_treated"]:
            reason = f"fewer than {ecfg['min_treated']} treated"
        elif ecfg["rule"] == "threshold" and not (p < ecfg["p_threshold"]):
            reason = f"p >= {ecfg['p_threshold']} (threshold rule)"
        elif ecfg["require_balance"] and not bal_ok:
            reason = "failed balance/overlap check"
        used = reason == ""
        row = dict(mediator=edge, kind=kind, n=n, n_treated=n_t, estimate=r["est"], se=r["se"],
                   ci_low=r["est"] - 1.96 * r["se"], ci_high=r["est"] + 1.96 * r["se"], p=p,
                   max_smd_before=r["smd_before"].max(initial=0), max_smd_after=r["smd_after"].max(initial=0),
                   pct_extreme_ps=r["pct_extreme_ps"], ess_treated=r["ess_treated"],
                   balance_ok=bal_ok, used=used, reason_not_used=reason)
        if kind == "event":
            row.update(risk_ratio=np.exp(r["est"]), rr_ci_low=np.exp(row["ci_low"]),
                       rr_ci_high=np.exp(row["ci_high"]), odds_ratio=np.exp(r["log_or"]),
                       risk_if_all_exposed=r["mu1"], risk_if_none_exposed=r["mu0"])
        rows_out.append(row)
        for nm, b_, a_ in zip(names, r["smd_before"], r["smd_after"]):
            bal_out.append(dict(mediator=edge, covariate=nm, smd_before=b_, smd_after=a_))
            if b_ > ecfg["separation_smd_warn"] and nm not in sep_warned:
                sep_warned.add(nm)
                log.append(f"WARNING: confounder '{nm}' almost separates exposed from unexposed "
                           f"(standardized difference {b_:.1f} before weighting). Check its coding: a "
                           f"value that encodes the exposure itself (for example a missing-value pattern) "
                           f"must not be used as a confounder.")
        if used:
            if kind == "lab":
                locked["labs"][edge.replace("LAB_", "", 1)] = (float(r["est"]), float(r["se"]))
            else:
                locked["event"] = (float(r["est"]), float(r["se"]))

    # lab edges: change in severity (post window mean minus last baseline value)
    cands = [str(c) for c in (ecfg["candidate_labs"] or fs.lab_codes) if str(c) in fs.lab_codes]
    w0, w1 = ecfg["post_window_days"]
    base_end = int(cfg["time"]["baseline_end_day"])
    if cands:
        L = labs[labs["code"].astype(str).isin(cands)]
        b0 = (L[L["day"] <= base_end].sort_values("day").groupby(["code", "i"], observed=True)["bad"].last())
        p1 = (L[(L["day"] >= w0) & (L["day"] <= w1)].groupby(["code", "i"], observed=True)["bad"].mean())
        both = pd.concat([b0.rename("base"), p1.rename("post")], axis=1, join="inner").reset_index()
        for code in cands:
            d_ = both[both["code"].astype(str) == code]
            rows = d_["i"].values.astype(np.int64)
            keep = train[rows] & elig[rows]
            d_, rows = d_[keep], rows[keep]
            t_ = T[rows]
            if len(rows) < 20 or t_.sum() < 5 or (1 - t_).sum() < 5:
                rows_out.append(dict(mediator=f"LAB_{code}", kind="lab", n=len(rows), n_treated=int(t_.sum()),
                                     used=False, reason_not_used="too few patients with baseline and post values"))
                continue
            bz = d_["base"].values
            bz = ((bz - bz.mean()) / (bz.std() + 1e-9))[:, None]
            r = aipw((d_["post"] - d_["base"]).values.astype(float), t_, np.hstack([Zc[rows], bz]), False, trunc)
            record(f"LAB_{code}", "lab", len(rows), t_, r, znames + ["baseline_lab_value"])

    # event edge
    if fs.has_event:
        rows = np.where(train & elig)[0]
        t_ = T[rows]
        r = aipw(co.event[rows].astype(float), t_, Zc[rows], True, trunc)
        record(EVENT, "event", len(rows), t_, r, znames)

    tab = pd.DataFrame(rows_out)
    log.append(f"trained edges: {len(tab)} estimated, {int(tab['used'].sum()) if len(tab) else 0} used "
               f"(rule {ecfg['rule']}, balance required {ecfg['require_balance']})")
    return tab, pd.DataFrame(bal_out), locked


def edge_sets(cfg, curated, trained):
    """Locked edge set per mode. curated / trained: {'labs': {code: (v, se)}, 'event': (logrr, se)|None}."""
    out = {}
    for mode in cfg["edges"]["modes"]:
        labs_ = dict(trained["labs"]) if trained else {}
        event = trained["event"] if trained else None
        if mode == "curated" and curated:
            labs_.update(curated["labs"])
            if curated["event"] is not None:
                event = curated["event"]
        out[mode] = {"labs": labs_, "event": event}
    return out


def edge_table(sets, lab_names):
    """Long table of the locked graph per mode (for deposit)."""
    rows = []
    for mode, es in sets.items():
        for code, (v, se) in es["labs"].items():
            rows.append(dict(mode=mode, mediator=f"LAB_{code}", name=lab_names.get(code, ""), kind="lab",
                             scale="severity shift (+ = worse)", value=v, se=se))
        if es["event"] is not None:
            rows.append(dict(mode=mode, mediator=EVENT, name="event mediator", kind="event",
                             scale="risk ratio", value=float(np.exp(es["event"][0])), se=es["event"][1],
                             note="se is on the log risk ratio scale"))
    cols = ["mode", "mediator", "name", "kind", "scale", "value", "se", "note"]
    return pd.DataFrame(rows, columns=cols)

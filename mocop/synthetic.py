"""Synthetic example data with a known answer (for trying MoCoP and for the tests).

Nothing here is real patient data. The data-generating process:
  covariates   age, sex, comorbidity score, region, prior_event
  eligibility  ~90% eligible; only eligible patients can be exposed
  exposure     more likely in older patients with more comorbidity (confounding by
               indication through measured covariates only)
  labs         creatinine, CRP, hemoglobin, potassium at baseline (3 draws in the 2 years
               before the index date) and after the index date (days 10 to 80).
               Exposure lowers CRP after the index date.
  event        binary secondary outcome; exposure multiplies its risk by EVENT_RR
  outcome      depends on sickness, post-index CRP and creatinine, the event, comorbidity
The true lab edge, event edge and total effect are computed from both potential worlds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_RR = 0.60
CRP_EFFECT = -3.0          # mg/L shift in post-index CRP if exposed
LABS = {  # code: (ref_low, ref_high, mean, sd, sickness loading)
    "2160-0": (0.6, 1.2, 0.95, 0.25, 0.20),    # creatinine (high_bad)
    "1988-5": (0.0, 10.0, 6.0, 4.0, 4.0),      # CRP (high_bad)
    "718-7": (12.0, 16.0, 13.8, 1.2, -0.8),    # hemoglobin (high_good)
    "2823-3": (3.5, 5.1, 4.3, 0.35, 0.05),     # potassium (mid_best)
}
DIRECTION = {"2160-0": "high_bad", "1988-5": "high_bad", "718-7": "high_good", "2823-3": "mid_best"}


def _sev(code, v):
    lo, hi = LABS[code][:2]
    vn = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    d = DIRECTION[code]
    if d == "high_bad":
        return np.clip((vn - 0.5) * 2, -1, 1)
    if d == "high_good":
        return np.clip(-(vn - 0.5) * 2, -1, 1)
    return np.clip(np.abs(vn - 0.5) * 2, 0, 1)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def generate(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    pid = np.array([f"P{k:06d}" for k in range(n)])
    idx = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 4 * 365, n), unit="D")
    age = np.clip(rng.normal(58, 15, n), 18, 95).round()
    female = rng.random(n) < 0.55
    score = rng.poisson(2.0, n)
    region = rng.choice(["north", "south", "west"], n, p=[0.5, 0.3, 0.2])
    prior = (rng.random(n) < 0.25).astype(int)
    sick = rng.normal(0, 1, n) + 0.25 * (score - 2) + 0.01 * (age - 58)
    eligible = (rng.random(n) < 0.9).astype(int)
    # exposure depends on MEASURED covariates only (no unmeasured confounding by design)
    pt = _sig(-2.6 + 0.03 * (age - 58) + 0.35 * score + 0.6 * prior)
    t = ((rng.random(n) < pt) & (eligible == 1)).astype(int)

    lab_rows = []
    post_world = {0: {}, 1: {}}
    for code, (lo, hi, mu, sd, load) in LABS.items():
        for _ in range(3):
            d = -rng.integers(1, 700, n)
            v = mu + load * sick + rng.normal(0, sd, n)
            lab_rows.append(pd.DataFrame({"pid": pid, "day": d, "code": code, "value": v}))
        d = rng.integers(10, 80, n)
        noise = rng.normal(0, sd, n)
        v0 = mu + load * sick + noise
        v1 = v0 + (CRP_EFFECT if code == "1988-5" else 0.0)
        post_world[0][code], post_world[1][code] = v0, v1
        v_obs = np.where(t == 1, v1, v0)
        lab_rows.append(pd.DataFrame({"pid": pid, "day": d, "code": code, "value": v_obs}))
    labs = pd.concat(lab_rows, ignore_index=True)
    labs = labs[rng.random(len(labs)) < 0.85]                     # some labs not drawn
    labs["value"] = np.clip(labs["value"], 0.01, None).round(2)
    labs["date"] = (idx.values[pd.Index(pid).get_indexer(labs["pid"])]
                    + pd.to_timedelta(labs["day"].values, unit="D"))
    labs["date"] = pd.to_datetime(labs["date"]).dt.date
    labs["ref_low"] = labs["code"].map({k: v[0] for k, v in LABS.items()})
    labs["ref_high"] = labs["code"].map({k: v[1] for k, v in LABS.items()})
    labs = labs[["pid", "date", "code", "value", "ref_low", "ref_high"]]

    pe0 = _sig(-2.3 + 0.7 * sick + 0.15 * score)
    pe1 = np.where(eligible == 1, EVENT_RR * pe0, pe0)
    u_ev = rng.random(n)
    ev0, ev1 = (u_ev < pe0).astype(int), (u_ev < pe1).astype(int)
    event = np.where(t == 1, ev1, ev0)

    def p_out(world, ev):
        crp = _sev("1988-5", post_world[world]["1988-5"])
        cre = _sev("2160-0", post_world[world]["2160-0"])
        return _sig(-2.0 + 0.5 * sick + 0.9 * crp + 0.5 * cre + 0.9 * ev + 0.1 * score)

    u_y = rng.random(n)
    y0, y1 = (u_y < p_out(0, ev0)).astype(int), (u_y < p_out(1, ev1)).astype(int)
    y = np.where(t == 1, y1, y0)

    codes = {"ICD10:E11.9": 0.25, "ICD10:I10": 0.35, "ICD10:N18.3": 0.12, "ICD10:J45.909": 0.10, "ICD10:F32.9": 0.15}
    dx_rows = []
    for code, base_p in codes.items():
        has = rng.random(n) < np.clip(base_p * (1 + 0.15 * (score - 2)), 0.01, 0.95)
        d = -rng.integers(1, 700, n)
        dx_rows.append(pd.DataFrame({"pid": pid[has], "date": (idx[has] + pd.to_timedelta(d[has], unit="D")),
                                     "code": code}))
    dx = pd.concat(dx_rows, ignore_index=True)

    cohort = pd.DataFrame({"pid": pid, "index_date": idx.date, "outcome": y, "exposure": t, "eligible": eligible,
                           "event": event, "age": age, "sex": np.where(female, "F", "M"),
                           "comorbidity_score": score, "region": region, "prior_event": prior})
    el = eligible == 1
    sev0 = _sev("1988-5", post_world[0]["1988-5"])
    sev1 = _sev("1988-5", post_world[1]["1988-5"])
    pw0 = pe0 * p_out(0, 1) + (1 - pe0) * p_out(0, 0)
    pw1 = pe1 * p_out(1, 1) + (1 - pe1) * p_out(1, 0)
    truth = dict(
        event_risk_ratio=EVENT_RR,
        crp_severity_shift_eligible=float((sev1 - sev0)[el].mean()),
        total_risk_ratio_all=float(pw1.mean() / pw0.mean()),
        total_risk_difference_all=float(pw1.mean() - pw0.mean()),
        note="Truth from both potential worlds. MoCoP's estimate is expected to be close "
             "in direction and size, not identical, because it models labs from baseline values.",
    )
    return dict(cohort=cohort, labs=labs, dx=dx, truth=truth)


EXAMPLE_CONFIG = """\
# MoCoP example config (synthetic data). Paths are relative to this file.
data:
  cohort: cohort.csv
  labs: labs.csv
  dx: dx.csv
  output_dir: output
columns:
  id: pid
  index_date: index_date
  outcome: outcome
  exposure: exposure
  eligible: eligible
  event_mediator: event
covariates: [age, sex, comorbidity_score, region, prior_event]
edges:
  modes: [curated, trained]
  file: edges_curated.csv
uncertainty:
  n_boot: 200
  n_psa: 100
"""


def write_example(folder, n=5000, seed=0):
    f = Path(folder)
    f.mkdir(parents=True, exist_ok=True)
    d = generate(n, seed)
    d["cohort"].to_csv(f / "cohort.csv", index=False)
    d["labs"].to_csv(f / "labs.csv", index=False)
    d["dx"].to_csv(f / "dx.csv", index=False)
    pd.DataFrame([dict(mediator="event", kind="event", value=EVENT_RR, se=0.05,
                       source="example: a typed-in literature value")]).to_csv(f / "edges_curated.csv", index=False)
    (f / "config.yaml").write_text(EXAMPLE_CONFIG)
    (f / "truth.json").write_text(json.dumps(d["truth"], indent=2))
    return str(f / "config.yaml")

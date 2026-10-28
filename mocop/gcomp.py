"""Counterfactual propagation (g-computation).

For each patient, both arms start from the patient's own baseline labs and extrapolate
the patient's own slope (with the boundary-aware step from the original algorithm, so a
lab never leaves [-1, 1]); the exposed arm adds the lab edges. If an event mediator is
present, risk is averaged over it: P(event) from the natural-course model in the
unexposed arm, P(event) times the event risk ratio in the exposed arm. Patients who are
not eligible receive no exposure effect.

Estimands
  total        through labs and the event (MoCoP's full counterfactual)
  labs         through labs only (event at its natural course in both arms)
  event        through the event only
  event_held   labs only, event held at each patient's observed value (answers the same
               question as a causal model that adjusts for the event)
"""
from __future__ import annotations

import numpy as np

from .features import EVENT

ESTIMAND_TEXT = {
    "total": "through labs and the event (total)",
    "labs": "through labs only",
    "event": "through the event only",
    "event_held": "labs only, event held at its observed value",
}


class Propagator:
    def __init__(self, fs, model, eligible, p_event=None, event_obs=None, chunk_rows=20000):
        self.fs, self.model = fs, model
        self.eligible = eligible
        self.p_event, self.event_obs = p_event, event_obs
        self.chunk = chunk_rows
        self.now_idx = fs.now_idx()
        self.slope_idx = fs.slope_idx()
        self.ch = {c: k for k, c in enumerate(fs.lab_codes)}
        self.ev_idx = fs.col_idx.get(EVENT)

    def estimands(self):
        return ["total", "labs", "event", "event_held"] if self.fs.has_event else ["total"]

    def arm_risks(self, rows, lab_shift):
        """Risk per arm; with an event mediator: risk if event = 1 and if event = 0."""
        X = self.fs.X
        eff = np.zeros(len(self.fs.lab_codes))
        for code, s in lab_shift.items():
            if code in self.ch:
                eff[self.ch[code]] = s
        keys = ("u1", "u0", "t1", "t0") if self.fs.has_event else ("u", "t")
        res = {k: [] for k in keys}
        for s0 in range(0, len(rows), self.chunk):
            rr = rows[s0:s0 + self.chunk]
            Xu = X[rr]
            Xt = Xu.copy()
            if len(self.now_idx):
                ec = eff[None, :] * self.eligible[rr][:, None]
                base = Xu[:, self.now_idx]
                slp = Xu[:, self.slope_idx]
                nz = np.abs(slp) > 1e-9
                s_ = np.where(nz, slp, 1.0)
                au = np.where(slp > 0, (1 - base) / s_, (-1 - base) / s_)
                at = np.where(slp > 0, (1 - ec - base) / s_, (-1 - ec - base) / s_)
                alpha = np.clip(np.minimum(au, at), 0, 1)
                alpha[~nz] = 1.0
                Xu[:, self.now_idx] = np.clip(base + alpha * slp, -1, 1)
                Xt[:, self.now_idx] = np.clip(base + alpha * slp + ec, -1, 1)
            if self.fs.has_event:
                for nm, M in (("u", Xu), ("t", Xt)):
                    M[:, self.ev_idx] = 1.0
                    res[nm + "1"].append(self.model.risk(M))
                    M[:, self.ev_idx] = 0.0
                    res[nm + "0"].append(self.model.risk(M))
            else:
                res["u"].append(self.model.risk(Xu))
                res["t"].append(self.model.risk(Xt))
        return {k: np.concatenate(v) for k, v in res.items()}

    def mix(self, ar, rows, event_logrr, estimand):
        """Per-patient risk unexposed (p0) and exposed (p1)."""
        if not self.fs.has_event:
            return ar["u"], ar["t"]
        if estimand == "event_held":
            ph0 = ph1 = self.event_obs[rows].astype(float)
        else:
            ph0 = self.p_event[rows]
            ph1 = ph0
            if estimand in ("total", "event") and event_logrr is not None:
                ph1 = np.where(self.eligible[rows] == 1, np.clip(np.exp(event_logrr) * ph0, 0, 1), ph0)
        return ph0 * ar["u1"] + (1 - ph0) * ar["u0"], ph1 * ar["t1"] + (1 - ph1) * ar["t0"]


def odds_ratio(a0, a1):
    return (a1 / (1 - a1)) / (a0 / (1 - a0))


def summarize(p0, p1, n_boot, rng):
    m0, m1 = p0.mean(), p1.mean()
    ates, lrr = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        k = rng.integers(0, len(p0), len(p0))
        a0, a1 = p0[k].mean(), p1[k].mean()
        ates[b], lrr[b] = a1 - a0, np.log(a1 / a0)
    return dict(risk_unexposed=m0, risk_exposed=m1, risk_difference=m1 - m0,
                rd_boot_low=np.percentile(ates, 2.5), rd_boot_high=np.percentile(ates, 97.5),
                risk_ratio=m1 / m0, log_rr=np.log(m1 / m0), se_log_rr_patients=lrr.std(ddof=1),
                odds_ratio=odds_ratio(m0, m1), pct_patients_benefit=float(((p1 - p0) < 0).mean() * 100))


def directionality_audit(prop, rows, edge_sets):
    """Improving a lab (or avoiding the event) must never raise mean predicted risk."""
    X = prop.fs.X
    base = prop.model.risk(X[rows]).mean()
    out = []
    shifts = {}
    for es in edge_sets.values():
        for code, (v, _) in es["labs"].items():
            if v != 0:
                shifts[code] = v
    for code, v in shifts.items():
        c = f"LAB_{code}_now"
        if c not in prop.fs.col_idx:
            continue
        Xp = X[rows].copy()
        j = prop.fs.col_idx[c]
        Xp[:, j] = np.clip(Xp[:, j] - abs(v), -1, 1)          # improvement
        d = prop.model.risk(Xp).mean() - base
        out.append(dict(check=f"improve LAB_{code} by {abs(v):.3f}", change_in_mean_risk=d, ok=bool(d <= 1e-6)))
    if prop.fs.has_event:
        X1, X0 = X[rows].copy(), X[rows].copy()
        X1[:, prop.ev_idx], X0[:, prop.ev_idx] = 1.0, 0.0
        d = prop.model.risk(X0).mean() - prop.model.risk(X1).mean()
        out.append(dict(check="avoid the event", change_in_mean_risk=d, ok=bool(d <= 1e-6)))
    return out

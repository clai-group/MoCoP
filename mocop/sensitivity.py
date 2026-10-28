"""Sensitivity of the estimate to the edges (from the original script 4).

scan    : effect if the exposure improved EVERY lab mediator by gamma on the severity
          scale; shows how large a lab effect would be needed to move the outcome.
tornado : each locked edge moved by +/- pct, one at a time; longer bar = the
          estimate depends more on that edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def scan(prop, rows, gammas, lab_codes):
    out = []
    for g in gammas:
        ar = prop.arm_risks(rows, {c: -float(g) for c in lab_codes})
        p0, p1 = prop.mix(ar, rows, None, "labs" if prop.fs.has_event else "total")
        out.append(dict(gamma=g, risk_difference=p1.mean() - p0.mean(), risk_ratio=p1.mean() / p0.mean(),
                        pct_patients_benefit=float(((p1 - p0) < 0).mean() * 100)))
    return pd.DataFrame(out)


def tornado(prop, rows, edge_sets, pct):
    out = []
    for mode, es in edge_sets.items():
        base_shift = {k: v for k, (v, _) in es["labs"].items()}
        ev = es["event"][0] if es["event"] is not None else None

        def rr(shift, ev_):
            ar = prop.arm_risks(rows, shift)
            p0, p1 = prop.mix(ar, rows, ev_, "total")
            return p1.mean() / p0.mean()

        base = rr(base_shift, ev)
        for code, v in base_shift.items():
            lo = dict(base_shift, **{code: v * (1 - pct)})
            hi = dict(base_shift, **{code: v * (1 + pct)})
            out.append(dict(mode=mode, edge=f"LAB_{code}", edge_value=v, rr_base=base,
                            rr_edge_low=rr(lo, ev), rr_edge_high=rr(hi, ev)))
        if ev is not None:
            out.append(dict(mode=mode, edge="EVENT", edge_value=float(np.exp(ev)), rr_base=base,
                            rr_edge_low=rr(base_shift, ev * (1 - pct)), rr_edge_high=rr(base_shift, ev * (1 + pct))))
    t = pd.DataFrame(out)
    if len(t):
        t["range"] = (t["rr_edge_high"] - t["rr_edge_low"]).abs()
        t = t.sort_values(["mode", "range"], ascending=[True, False])
    return t

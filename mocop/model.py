"""Outcome model and event (natural course) model.

Outcome model: class-weighted L2 logistic regression on standardised features, with
non-negativity bounds on every mediator feature (labs on the severity scale and the
event), so making a mediator worse can never lower predicted risk. Platt-calibrated on
the validation split. The exposure is not an input.

Event model: P(event | pre-exposure features) in unexposed eligible training patients,
used as the event's natural course in the counterfactual.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


def ece(p, y, bins=10):
    e = 0.0
    for k in range(bins):
        m = (p >= k / bins) & (p < (k + 1) / bins)
        if m.any():
            e += abs(p[m].mean() - y[m].mean()) * m.sum()
    return e / len(p)


class OutcomeModel:
    def __init__(self, mono, l2=1e-3, platt=True):
        self.mono, self.l2, self.platt = np.asarray(mono), l2, platt

    def fit(self, X, y, tr, va):
        Xt = X[tr]
        self.mu = Xt.mean(0)
        self.sd = Xt.std(0) + 1e-6
        Z = (Xt - self.mu) / self.sd
        yt = y[tr]
        pw = (yt == 0).sum() / max((yt == 1).sum(), 1)
        w = np.where(yt == 1, pw, 1.0)
        n, d = Z.shape
        l2 = self.l2

        def f(th):
            b, wt = th[0], th[1:]
            p = np.clip(1 / (1 + np.exp(-(b + Z @ wt))), 1e-7, 1 - 1e-7)
            return -(w * (yt * np.log(p) + (1 - yt) * np.log(1 - p))).mean() + l2 * (wt @ wt)

        def g(th):
            b, wt = th[0], th[1:]
            p = 1 / (1 + np.exp(-(b + Z @ wt)))
            r = w * (p - yt) / n
            return np.concatenate([[r.sum()], Z.T @ r + 2 * l2 * wt])

        bounds = [(None, None)] + [(0, None) if m > 0 else (None, None) for m in self.mono]
        res = minimize(f, np.zeros(d + 1), jac=g, method="L-BFGS-B", bounds=bounds)
        self.b, self.w = res.x[0], res.x[1:]
        if self.platt:
            s = self.raw(X[va]).reshape(-1, 1)
            self.cal = LogisticRegression(C=1e6, max_iter=1000).fit(s, y[va])
        return self

    def raw(self, M):
        return 1 / (1 + np.exp(-(self.b + ((M - self.mu) / self.sd) @ self.w)))

    def risk(self, M):
        r = self.raw(M)
        return self.cal.predict_proba(r.reshape(-1, 1))[:, 1] if self.platt else r

    def performance(self, X, y, rows):
        p = self.risk(X[rows])
        yy = y[rows]
        auc = roc_auc_score(yy, p) if 0 < yy.mean() < 1 else np.nan
        return dict(auc=auc, brier=brier_score_loss(yy, p), ece=ece(p, yy), prevalence=yy.mean(),
                    mean_predicted=p.mean())


class EventModel:
    """Natural-course probability of the event, fitted on unexposed eligible training patients."""

    def fit(self, Xpre, event, rows):
        self.mu = Xpre[rows].mean(0)
        self.sd = Xpre[rows].std(0) + 1e-6
        self.m = LogisticRegression(C=1.0, max_iter=3000).fit((Xpre[rows] - self.mu) / self.sd, event[rows])
        return self

    def predict(self, Xpre):
        return self.m.predict_proba((Xpre - self.mu) / self.sd)[:, 1]

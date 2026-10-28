"""Lab severity scale (MoCoP paper, Eq. 2-3).

Each lab value is normalised to [0, 1] with its reference interval (or, when no valid
interval is available for an analyte, its 1st-99th percentile in the data), then turned
into a signed severity in [-1, 1] where larger always means clinically worse:
    high_bad  : (v - 0.5) * 2
    high_good : -(v - 0.5) * 2
    mid_best  : |v - 0.5| * 2          (distance from the middle of the range)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_severity(labs: pd.DataFrame, directions: dict, unknown: str = "mid_best",
                 train_mask: np.ndarray | None = None) -> pd.DataFrame:
    """Add the signed severity column 'bad'. Percentile limits for labs without a valid
    reference range are computed from training patients only (train_mask, indexed by
    patient row), so nothing is learned from validation or test patients."""
    has_ref = (~np.isnan(labs["ref_low"].values) & ~np.isnan(labs["ref_high"].values)
               & (labs["ref_high"].values > labs["ref_low"].values))
    v = labs["value"].values
    vn = np.full(len(labs), np.nan)
    lo, hi = labs["ref_low"].values, labs["ref_high"].values
    vn[has_ref] = np.clip((v[has_ref] - lo[has_ref]) / (hi[has_ref] - lo[has_ref] + 1e-9), 0, 1)
    codes = labs["code"].cat.codes.values
    cats = labs["code"].cat.categories
    fit = np.ones(len(labs), bool) if train_mask is None else np.asarray(train_mask)[labs["i"].values]
    q_noref = labs.loc[~has_ref & fit].groupby("code", observed=True)["value"].quantile([0.01, 0.99]).unstack()
    q_all = labs.loc[fit].groupby("code", observed=True)["value"].quantile([0.01, 0.99]).unstack()
    q = q_noref.reindex(cats).combine_first(q_all.reindex(cats)).reindex(columns=[0.01, 0.99])
    p01, p99 = q[0.01].values[codes], q[0.99].values[codes]
    m = np.isnan(vn) & ~np.isnan(p01)
    vn[m] = np.clip((v[m] - p01[m]) / (p99[m] - p01[m] + 1e-9), 0, 1)
    vn[np.isnan(vn)] = 0.5
    d = np.array([directions.get(str(c), unknown) for c in cats])[codes]
    labs = labs.copy()
    labs["bad"] = np.select([d == "high_bad", d == "high_good"],
                            [np.clip((vn - 0.5) * 2, -1, 1), np.clip(-(vn - 0.5) * 2, -1, 1)],
                            np.clip(np.abs(vn - 0.5) * 2, 0, 1)).astype(np.float32)
    return labs.drop(columns=["value", "ref_low", "ref_high"])

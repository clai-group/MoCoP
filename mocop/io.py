"""Read and check the analysis-ready input files.

Patients are referred to internally by their row number in the cohort table (an
integer), and large files are read in chunks, so tens of millions of lab rows fit in
a few GB of memory.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRUE_VALUES = {"1", "1.0", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "0.0", "false", "f", "no", "n"}


def _to_binary(s: pd.Series, name: str, allow_missing: bool) -> pd.Series:
    """0/1 as float with NaN for missing; raises on anything else."""
    txt = s.astype(str).str.strip().str.lower()
    miss = s.isna() | txt.isin({"", "nan", "none", "na", "<na>"})
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[txt.isin(TRUE_VALUES) & ~miss] = 1.0
    out[txt.isin(FALSE_VALUES) & ~miss] = 0.0
    bad = out.isna() & ~miss
    if bad.any():
        ex = ", ".join(sorted(set(s[bad].astype(str)))[:5])
        raise ValueError(f"Column '{name}' must be binary (0/1, true/false, yes/no). Found: {ex}")
    if miss.any() and not allow_missing:
        raise ValueError(f"Column '{name}' has {int(miss.sum())} missing values; it must be complete.")
    return out


def _to_date(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, errors="coerce", utc=True)
    return d.dt.tz_localize(None).dt.normalize()


@dataclass
class CohortData:
    df: pd.DataFrame                 # one row per included patient, original columns kept
    pid: np.ndarray                  # patient ids (str)
    index_date: np.ndarray           # datetime64[D]
    y: np.ndarray                    # outcome 0/1
    t: np.ndarray                    # exposure 0/1
    eligible: np.ndarray             # 0/1 (all ones if no eligibility column)
    event: np.ndarray | None         # event mediator 0/1 or None
    covariates: list
    pid_to_i: pd.Series
    log: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.pid)


def load_cohort(cfg) -> CohortData:
    c = cfg["columns"]
    log = []
    df = pd.read_csv(cfg["data"]["cohort"], dtype={c["id"]: str}, low_memory=False)
    need = [c["id"], c["index_date"], c["outcome"], c["exposure"]]
    need += [x for x in (c["eligible"], c["event_mediator"]) if x]
    missing = [x for x in need if x not in df.columns]
    if missing:
        raise ValueError(f"Cohort file is missing column(s): {missing}. Found: {list(df.columns)[:30]}")
    df[c["id"]] = df[c["id"]].astype(str).str.strip()
    dup = df[c["id"]].duplicated()
    if dup.any():
        raise ValueError(f"Cohort must have one row per patient: {int(dup.sum())} duplicated ids in '{c['id']}'.")
    n0 = len(df)
    df["_idx"] = _to_date(df[c["index_date"]])
    df["_y"] = _to_binary(df[c["outcome"]], c["outcome"], allow_missing=True)
    df["_t"] = _to_binary(df[c["exposure"]], c["exposure"], allow_missing=True)
    keep = df["_idx"].notna() & df["_y"].notna() & df["_t"].notna()
    for lbl, m in (("missing or unreadable index date", df["_idx"].isna()),
                   ("missing outcome", df["_y"].isna()),
                   ("missing exposure (not in this comparison)", df["_t"].isna())):
        if m.any():
            log.append(f"excluded {int(m.sum()):,} patients with {lbl}")
    df = df[keep].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No patients left after removing missing index date / outcome / exposure.")
    elig = (_to_binary(df[c["eligible"]], c["eligible"], allow_missing=True).fillna(0).values
            if c["eligible"] else np.ones(len(df)))
    event = None
    if c["event_mediator"]:
        ev = _to_binary(df[c["event_mediator"]], c["event_mediator"], allow_missing=True)
        if ev.isna().any():
            log.append(f"event mediator '{c['event_mediator']}': {int(ev.isna().sum()):,} missing values set to 0")
        event = ev.fillna(0).values
    t = df["_t"].values
    y = df["_y"].values
    for lbl, a in (("treated", t), ("outcome events", y)):
        if a.sum() == 0 or a.sum() == len(a):
            raise ValueError(f"All or none of the patients have {lbl}; nothing to estimate.")
    covs = _pick_covariates(df, cfg, log)
    log.insert(0, f"cohort: {n0:,} rows read, {len(df):,} patients kept | treated {int(t.sum()):,} | "
                  f"outcome {int(y.sum()):,} ({y.mean():.1%})")
    pid = df[c["id"]].values
    return CohortData(df=df, pid=pid, index_date=df["_idx"].values.astype("datetime64[D]"),
                      y=y.astype(float), t=t.astype(float), eligible=np.asarray(elig, float),
                      event=None if event is None else event.astype(float), covariates=covs,
                      pid_to_i=pd.Series(np.arange(len(df), dtype=np.int32), index=pid), log=log)


def _pick_covariates(df, cfg, log):
    c = cfg["columns"]
    reserved = {c["id"], c["index_date"], c["outcome"], c["exposure"], c["eligible"],
                c["event_mediator"], "_idx", "_y", "_t"}
    if cfg["covariates"] is not None:
        covs = list(cfg["covariates"])
        miss = [x for x in covs if x not in df.columns]
        if miss:
            raise ValueError(f"covariates not found in cohort file: {miss}")
        bad = [x for x in covs if x in reserved]
        if bad:
            raise ValueError(f"covariates must not include the id, date, outcome, exposure, eligibility "
                             f"or event columns: {bad}")
        return covs
    covs, skipped = [], []
    for col in df.columns:
        if col in reserved:
            continue
        s = df[col]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() >= 0.95 or s.isna().all():
            if s.notna().any():
                covs.append(col)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            as_date = pd.to_datetime(s, errors="coerce")
        if as_date.notna().mean() > 0.9:
            skipped.append(f"{col} (looks like a date)")
            continue
        if s.nunique(dropna=True) > 50:
            skipped.append(f"{col} (more than 50 text categories)")
            continue
        covs.append(col)
    log.append(f"covariates detected automatically ({len(covs)}): {covs}")
    if skipped:
        log.append(f"columns not used as covariates: {skipped}")
    return covs


def _strip_vocab(code: pd.Series) -> pd.Series:
    """'ICD10:ICD10:E11.9' -> 'E11.9'; upper case, spaces removed."""
    return code.astype(str).str.strip().str.upper().str.replace(r"^([A-Z0-9_\-]+:)+", "", regex=True)


def load_labs(cfg, co: CohortData):
    """Long lab table restricted to [-lookback, post_window_end] days around each index date."""
    lc = cfg["columns"]["labs"]
    path = cfg["data"]["labs"]
    header = pd.read_csv(path, nrows=0).columns
    need = [lc["id"], lc["date"], lc["code"], lc["value"]]
    miss = [x for x in need if x not in header]
    if miss:
        raise ValueError(f"Lab file is missing column(s): {miss}. Found: {list(header)}")
    opt = [lc[k] for k in ("ref_low", "ref_high") if lc.get(k) and lc[k] in header]
    lo_d = -int(cfg["time"]["lookback_days"])
    hi_d = int(cfg["edges"]["post_window_days"][1])
    idx_days = co.index_date.astype("datetime64[D]").astype(np.int64)
    parts, n_read = [], 0
    for ch in pd.read_csv(path, usecols=need + opt, dtype={lc["id"]: str},
                          chunksize=int(cfg["output"]["chunk_rows"]), low_memory=False):
        n_read += len(ch)
        i = ch[lc["id"]].astype(str).str.strip().map(co.pid_to_i)
        ch = ch[i.notna()]
        i = i[i.notna()].astype(np.int64).values
        d = _to_date(ch[lc["date"]])
        val = pd.to_numeric(ch[lc["value"]], errors="coerce")
        ok = d.notna().values & val.notna().values
        dd = (d.values.astype("datetime64[D]").astype(np.int64) - idx_days[i])
        ok &= (dd >= lo_d) & (dd <= hi_d)
        if not ok.any():
            continue
        out = pd.DataFrame({
            "i": i[ok].astype(np.int32),
            "code": ch[lc["code"]].astype(str).str.strip().values[ok],
            "value": val.values[ok].astype(np.float64),
            "ref_low": (pd.to_numeric(ch[lc["ref_low"]], errors="coerce").values[ok].astype(np.float32)
                        if lc.get("ref_low") in ch.columns else np.full(ok.sum(), np.nan, np.float32)),
            "ref_high": (pd.to_numeric(ch[lc["ref_high"]], errors="coerce").values[ok].astype(np.float32)
                         if lc.get("ref_high") in ch.columns else np.full(ok.sum(), np.nan, np.float32)),
            "day": dd[ok].astype(np.int32),
        })
        parts.append(out)
    if not parts:
        raise ValueError("No lab rows fall inside the time window around the index dates. "
                         "Check the id column, the date format, and that labs cover the baseline window.")
    labs = pd.concat(parts, ignore_index=True)
    labs["code"] = labs["code"].astype("category")
    co.log.append(f"labs: {n_read:,} rows read, {len(labs):,} inside the window "
                  f"({labs['code'].nunique()} lab codes, {labs['i'].nunique():,} patients)")
    return labs


def load_dx(cfg, co: CohortData):
    """Pre-index diagnoses: (i, code) pairs with dates in [-lookback, baseline_end_day)."""
    if not cfg["data"]["dx"]:
        co.log.append("dx: no diagnosis file given")
        return None
    dc = cfg["columns"]["dx"]
    path = cfg["data"]["dx"]
    header = pd.read_csv(path, nrows=0).columns
    need = [dc["id"], dc["date"], dc["code"]]
    miss = [x for x in need if x not in header]
    if miss:
        raise ValueError(f"Diagnosis file is missing column(s): {miss}. Found: {list(header)}")
    lo_d = -int(cfg["time"]["lookback_days"])
    hi_d = int(cfg["time"]["baseline_end_day"])
    idx_days = co.index_date.astype("datetime64[D]").astype(np.int64)
    parts, n_read = [], 0
    for ch in pd.read_csv(path, usecols=need, dtype={dc["id"]: str},
                          chunksize=int(cfg["output"]["chunk_rows"]), low_memory=False):
        n_read += len(ch)
        i = ch[dc["id"]].astype(str).str.strip().map(co.pid_to_i)
        ch = ch[i.notna()]
        i = i[i.notna()].astype(np.int64).values
        d = _to_date(ch[dc["date"]])
        ok = d.notna().to_numpy(copy=True)
        dd = d.values.astype("datetime64[D]").astype(np.int64) - idx_days[i]
        ok &= (dd >= lo_d) & (dd < hi_d)
        if not ok.any():
            continue
        code = _strip_vocab(ch[dc["code"]]) if cfg["features"]["dx_strip_vocabulary"] \
            else ch[dc["code"]].astype(str).str.strip().str.upper()
        parts.append(pd.DataFrame({"i": i[ok].astype(np.int32), "code": code.values[ok]}).drop_duplicates())
    if not parts:
        co.log.append("dx: no diagnoses inside the baseline window")
        return None
    dx = pd.concat(parts, ignore_index=True).drop_duplicates()
    co.log.append(f"dx: {n_read:,} rows read, {len(dx):,} patient-code pairs in the baseline window")
    return dx


def load_lab_directions(cfg) -> dict:
    """code -> direction. Built-in table unless features.lab_directions is given."""
    from importlib import resources
    p = cfg["features"]["lab_directions"]
    if p:
        t = pd.read_csv(p, dtype=str)
    else:
        with resources.files("mocop").joinpath("data/lab_directions_default.csv").open() as f:
            t = pd.read_csv(f, dtype=str)
    t.columns = [x.strip().lower() for x in t.columns]
    if not {"code", "direction"} <= set(t.columns):
        raise ValueError("lab_directions file needs columns: code, direction")
    t["direction"] = t["direction"].str.strip()
    bad = t[~t["direction"].isin({"high_bad", "high_good", "mid_best"})]
    if len(bad):
        raise ValueError(f"lab_directions: direction must be high_bad, high_good or mid_best; "
                         f"found {bad['direction'].unique()[:5]}")
    names = dict(zip(t["code"].str.strip(), t.get("name", pd.Series([""] * len(t))).fillna("")))
    return dict(zip(t["code"].str.strip(), t["direction"])), names

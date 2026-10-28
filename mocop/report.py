"""Writing outputs. Aggregate tables only, with small counts suppressed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

COUNT_COLUMNS = {"n", "n_treated", "n_patients", "n_exposed", "n_unexposed"}


def suppress(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Blank out counts between 1 and threshold-1 (0 is kept: it reveals nothing)."""
    if not threshold or df is None or not len(df):
        return df
    df = df.copy()
    for c in df.columns:
        if c in COUNT_COLUMNS:
            v = pd.to_numeric(df[c], errors="coerce")
            small = (v > 0) & (v < threshold)
            df[c] = df[c].astype(object)
            df.loc[small, c] = f"<{threshold}"
    return df


def write_table(out_dir: Path, name: str, df: pd.DataFrame, threshold: int, written: list):
    if df is None:
        return
    suppress(df, threshold).to_csv(out_dir / name, index=False)
    written.append(name)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_run_info(out_dir: Path, info: dict, written: list):
    with open(out_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2, default=_jsonable)
    written.append("run_info.json")

"""Configuration: defaults, loading from YAML or dict, validation.

Every setting has a default, so a minimal config only needs the data paths and the
exposure and outcome column names. See docs/USER_GUIDE.md for every key.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "data": {
        "cohort": None,          # required: one row per patient
        "labs": None,            # required: long format, one row per lab result
        "dx": None,              # optional: long format, one row per diagnosis code
        "output_dir": "mocop_output",
    },
    "columns": {
        "id": "pid",
        "index_date": "index_date",
        "outcome": "outcome",          # binary 0/1
        "exposure": "exposure",        # binary 0/1; blank = patient excluded
        "eligible": None,              # optional binary column; None = everyone eligible
        "event_mediator": None,        # optional binary post-exposure event column
        "labs": {"id": "pid", "date": "date", "code": "code", "value": "value",
                 "ref_low": "ref_low", "ref_high": "ref_high"},
        "dx": {"id": "pid", "date": "date", "code": "code"},
    },
    "covariates": None,          # pre-exposure cohort columns; None = all other cohort columns
    "categorical": [],           # covariates to force as categorical
    "time": {
        "lookback_days": 730,    # baseline window before the index date
        "baseline_end_day": 0,   # last day (relative to index) that counts as baseline
        "trend_split_days": 180, # recent vs earlier part of the baseline window
    },
    "features": {
        "min_lab_coverage": 0.35,
        "min_dx_coverage": 0.05,
        "dx_stem_length": 3,
        "dx_strip_vocabulary": True,     # 'ICD10:E11.9' -> 'E11.9'
        "outcome_history_codes": [],     # dx code prefixes that mean the outcome occurred before
        "lab_directions": None,          # CSV (code,direction); None = built-in table
        "unknown_lab_direction": "mid_best",
        "max_lab_codes": 200,
    },
    "edges": {
        "modes": ["trained"],            # any of: curated, trained
        "file": None,                    # curated edges CSV (mediator,kind,value[,se])
        "candidate_labs": None,          # lab codes to test; None = all labs passing coverage
        "confounders": None,             # None = covariates
        "post_window_days": [7, 90],     # post-exposure lab window for lab edges
        "rule": "propagate",             # propagate | threshold
        "p_threshold": 0.05,
        "min_treated": 30,
        "require_balance": True,
        "ps_truncate": [0.001, 0.999],
        "balance_smd_max": 0.10,
        "extreme_ps_max": 0.05,
        "separation_smd_warn": 1.0,
    },
    "model": {
        "monotone": True,
        "constrain_slope": True,
        "platt": True,
        "l2": 1e-3,
        "split": [0.70, 0.15, 0.15],
        "seed": 42,
    },
    "uncertainty": {
        "n_boot": 500,
        "n_psa": 300,
        "psa_subset": 20000,
    },
    "sensitivity": {
        "scan_gammas": [0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
        "tornado_pct": 0.5,
    },
    "knowledge": {
        "enabled": False,
        "primekg_dir": None,             # folder with PrimeKG kg.csv and nodes.csv
        "concept_map": None,             # CSV: code,name (exposure, lab and outcome codes)
        "exposure_name": None,           # PrimeKG name of the exposure
        "restrict_candidates": False,    # True: test only labs with knowledge-graph support
    },
    "output": {
        "suppress_small_counts": 11,     # counts from 1 to this value minus 1 are written as "<11"
        "save_patient_level": False,
        "chunk_rows": 5_000_000,
    },
}

VALID_MODES = {"curated", "trained"}
VALID_RULES = {"propagate", "threshold"}
VALID_DIRECTIONS = {"high_bad", "high_good", "mid_best"}


def _merge(base, over, path=""):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise ValueError(f"Unknown config key: '{path}{k}'. See docs/USER_GUIDE.md for valid keys.")
        if isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v, f"{path}{k}.")
        else:
            out[k] = v
    return out


class Section(dict):
    """Dictionary with attribute access (cfg.edges.rule)."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Section(v) if isinstance(v, dict) else v


class Config(Section):
    pass


def load_config(src) -> Config:
    """Load a config from a YAML path, a dict, or an existing Config."""
    if isinstance(src, Config):
        return src
    base_dir = Path(".")
    if isinstance(src, (str, Path)):
        p = Path(src).expanduser()
        with open(p) as f:
            user = yaml.safe_load(f) or {}
        base_dir = p.parent
    elif isinstance(src, dict):
        user = src
    else:
        raise TypeError("config must be a YAML path or a dict")
    cfg = Config(_merge(DEFAULTS, user))
    _resolve_paths(cfg, base_dir)
    validate(cfg)
    return cfg


def _resolve_paths(cfg, base_dir):
    def rp(v):
        if v is None:
            return None
        p = Path(str(v)).expanduser()
        return str(p if p.is_absolute() else (base_dir / p))
    for k in ("cohort", "labs", "dx", "output_dir"):
        cfg["data"][k] = rp(cfg["data"][k])
    cfg["edges"]["file"] = rp(cfg["edges"]["file"])
    cfg["features"]["lab_directions"] = rp(cfg["features"]["lab_directions"])
    cfg["knowledge"]["primekg_dir"] = rp(cfg["knowledge"]["primekg_dir"])
    cfg["knowledge"]["concept_map"] = rp(cfg["knowledge"]["concept_map"])


def validate(cfg):
    errs = []
    for k in ("cohort", "labs"):
        if not cfg["data"][k]:
            errs.append(f"data.{k} is required")
        elif not Path(cfg["data"][k]).exists():
            errs.append(f"data.{k} not found: {cfg['data'][k]}")
    if cfg["data"]["dx"] and not Path(cfg["data"]["dx"]).exists():
        errs.append(f"data.dx not found: {cfg['data']['dx']}")
    modes = cfg["edges"]["modes"]
    if isinstance(modes, str):
        cfg["edges"]["modes"] = modes = [modes]
    if not modes or not set(modes) <= VALID_MODES:
        errs.append(f"edges.modes must be a list from {sorted(VALID_MODES)}")
    if "curated" in modes and not cfg["edges"]["file"]:
        errs.append("edges.modes includes 'curated' but edges.file is not set")
    if cfg["edges"]["file"] and not Path(cfg["edges"]["file"]).exists():
        errs.append(f"edges.file not found: {cfg['edges']['file']}")
    if cfg["edges"]["rule"] not in VALID_RULES:
        errs.append(f"edges.rule must be one of {sorted(VALID_RULES)}")
    lo, hi = cfg["edges"]["ps_truncate"]
    if not (0 < lo < hi < 1):
        errs.append("edges.ps_truncate must be two numbers with 0 < low < high < 1")
    s = cfg["model"]["split"]
    if len(s) != 3 or abs(sum(s) - 1) > 1e-6 or min(s) <= 0:
        errs.append("model.split must be three positive fractions summing to 1 (train, validation, test)")
    w = cfg["edges"]["post_window_days"]
    if len(w) != 2 or w[0] > w[1] or w[0] <= cfg["time"]["baseline_end_day"]:
        errs.append("edges.post_window_days must be [start, end] with start after time.baseline_end_day")
    if cfg["features"]["unknown_lab_direction"] not in VALID_DIRECTIONS:
        errs.append(f"features.unknown_lab_direction must be one of {sorted(VALID_DIRECTIONS)}")
    if cfg["knowledge"]["enabled"]:
        for k in ("primekg_dir", "concept_map", "exposure_name"):
            if not cfg["knowledge"][k]:
                errs.append(f"knowledge.enabled is true but knowledge.{k} is not set")
    if errs:
        raise ValueError("Config problems:\n  - " + "\n  - ".join(errs))
    return True

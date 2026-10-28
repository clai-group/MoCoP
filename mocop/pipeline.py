"""End-to-end run: read data -> features -> outcome model -> edges -> counterfactuals."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from . import __version__
from .config import load_config
from .edges import edge_sets, edge_table, read_edges_file, train_edges
from .features import EVENT, build_features
from .gcomp import ESTIMAND_TEXT, Propagator, directionality_audit, summarize
from .io import load_cohort, load_dx, load_lab_directions, load_labs
from .model import EventModel, OutcomeModel
from .report import write_run_info, write_table
from .sensitivity import scan, tornado


class Log(list):
    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

    def append(self, msg):
        super().append(msg)
        if self.verbose:
            print("  " + msg, flush=True)


@dataclass
class Results:
    tables: dict
    log: list
    output_dir: str
    files: list = field(default_factory=list)

    def __getitem__(self, k):
        return self.tables[k]

    def summary(self):
        e = self.tables["effects"]
        main = e[(e["group"] == "all patients")]
        cols = ["mode", "estimand", "risk_unexposed", "risk_exposed", "risk_difference", "risk_ratio",
                "rr_ci_low", "rr_ci_high"]
        return main[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")


def _redacted(cfg):
    """Config for run_info.json with file paths reduced to file names, so the record
    does not reveal user names or folder structure."""
    import copy
    c = copy.deepcopy(dict(cfg))
    for sec, keys in (("data", ("cohort", "labs", "dx", "output_dir")), ("edges", ("file",)),
                      ("features", ("lab_directions",)), ("knowledge", ("primekg_dir", "concept_map"))):
        for k in keys:
            if c[sec].get(k):
                c[sec][k] = Path(str(c[sec][k])).name
    return c


def make_split(y, cfg):
    """Stratified train / validation / test split of patient rows (depends only on the
    outcome vector and the seed, so it is reproducible)."""
    s_tr, s_va, s_te = cfg["model"]["split"]
    seed = int(cfg["model"]["seed"])
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=s_va + s_te, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=s_te / (s_va + s_te), stratify=y[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def run(config, verbose=True) -> Results:
    t0 = time.time()
    cfg = load_config(config)
    out = Path(cfg["data"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    log = Log(verbose)
    if verbose:
        print(f"MoCoP {__version__} | output -> {out}")

    # ---------------- data ----------------
    co = load_cohort(cfg)
    for m in co.log:
        log.append(m)
    co.log = log
    labs = load_labs(cfg, co)
    dx = load_dx(cfg, co)

    # ---------------- split (first, so nothing below learns from test patients) ----------------
    tr, va, te = make_split(co.y, cfg)
    train_mask = np.zeros(co.n, bool)
    train_mask[tr] = True
    log.append(f"split: train {len(tr):,} / validation {len(va):,} / test {len(te):,}")

    directions, lab_names = load_lab_directions(cfg)
    from .severity import add_severity
    labs = add_severity(labs, directions, cfg["features"]["unknown_lab_direction"], train_mask)
    unknown = sorted(set(map(str, labs["code"].cat.categories)) - set(directions))
    if unknown:
        log.append(f"{len(unknown)} lab codes have no direction in the table and use "
                   f"'{cfg['features']['unknown_lab_direction']}' (list them in features.lab_directions to change)")

    # ---------------- knowledge graph (optional) ----------------
    support = None
    if cfg["knowledge"]["enabled"]:
        from .knowledge import lab_support
        base_codes = sorted(set(map(str, labs.loc[labs["day"] <= cfg["time"]["baseline_end_day"], "code"].unique())))
        support = lab_support(cfg, base_codes, log)
        if cfg["knowledge"]["restrict_candidates"]:
            keep = support.loc[support["support"].isin(["direct", "2 hops"]), "code"].tolist()
            cfg["edges"]["candidate_labs"] = keep
            log.append(f"knowledge graph: testing only the {len(keep)} labs with support as mediators")

    # ---------------- features ----------------
    fs = build_features(co, labs, dx, cfg, log, train_mask)
    X, y = fs.X, co.y
    seed = int(cfg["model"]["seed"])
    idx = np.arange(co.n)

    # ---------------- outcome model ----------------
    model = OutcomeModel(fs.mono, l2=cfg["model"]["l2"], platt=cfg["model"]["platt"]).fit(X, y, tr, va)
    perf = [dict(model="outcome", split=s, n=len(r), **model.performance(X, y, r))
            for s, r in (("validation", va), ("test", te))]
    log.append(f"outcome model: test AUC {perf[1]['auc']:.3f}, Brier {perf[1]['brier']:.3f}, "
               f"calibration error {perf[1]['ece']:.3f}")
    wts = pd.DataFrame(dict(feature=fs.cols, weight=model.w, constrained_non_negative=fs.mono.astype(bool)))
    wts = wts.reindex(wts["weight"].abs().sort_values(ascending=False).index)

    # ---------------- event natural-course model ----------------
    p_event = None
    if fs.has_event:
        pre = [j for j, c in enumerate(fs.cols) if c != EVENT]
        unexp = co.t == 0
        fit_rows = tr[unexp[tr] & (co.eligible[tr] == 1)]
        if co.event[fit_rows].min() == co.event[fit_rows].max():
            raise ValueError("Event mediator has only one value among unexposed eligible training patients.")
        em = EventModel().fit(X[:, pre], co.event, fit_rows)
        p_event = em.predict(X[:, pre])
        vr = va[unexp[va] & (co.eligible[va] == 1)]
        ev_auc = roc_auc_score(co.event[vr], p_event[vr]) if 0 < co.event[vr].mean() < 1 else np.nan
        perf.append(dict(model="event (natural course)", split="validation", n=len(vr), auc=ev_auc,
                         brier=brier_score_loss(co.event[vr], p_event[vr]), ece=np.nan,
                         prevalence=co.event[vr].mean(), mean_predicted=p_event[vr].mean()))
        w_ev = float(model.w[fs.col_idx[EVENT]])
        log.append(f"event model: validation AUC {ev_auc:.3f} | outcome-model weight on the event {w_ev:+.3f}")
        if w_ev <= 0:
            log.append("WARNING: the outcome model gives the event zero weight; the event pathway cannot carry an effect")

    # ---------------- edges (training split only, then locked) ----------------
    curated = None
    if cfg["edges"]["file"]:
        cl, ce = read_edges_file(cfg["edges"]["file"], fs.lab_codes, cfg["columns"]["event_mediator"], log)
        curated = {"labs": cl, "event": ce}
    tr_tab, bal_tab, trained = train_edges(co, fs, labs, cfg, tr, log)
    sets = edge_sets(cfg, curated, trained)
    locked_tab = edge_table(sets, lab_names)
    for mode, es in sets.items():
        ev = "none" if es["event"] is None else f"RR {np.exp(es['event'][0]):.3f}"
        log.append(f"locked edges [{mode}]: {len(es['labs'])} lab edges | event {ev}")
        if not es["labs"] and es["event"] is None:
            log.append(f"WARNING: no edges locked in mode '{mode}': the estimated effect is zero by "
                       f"construction. See edges_trained.csv (column reason_not_used).")
    del labs, dx

    lab_w = []
    for c in fs.lab_codes:
        lab_w.append(dict(lab=c, name=lab_names.get(c, ""), baseline_coverage=float(fs.lab_coverage.get(c, 0)),
                          weight_now=float(model.w[fs.col_idx[f"LAB_{c}_now"]]),
                          weight_slope=float(model.w[fs.col_idx[f"LAB_{c}_slope"]]),
                          weight_trend=float(model.w[fs.col_idx[f"LAB_{c}_trend"]]),
                          has_edge=any(c in es["labs"] for es in sets.values())))
    lab_w = pd.DataFrame(lab_w)

    # ---------------- counterfactuals ----------------
    prop = Propagator(fs, model, co.eligible, p_event, co.event)
    audit = pd.DataFrame(directionality_audit(prop, te, sets))
    if len(audit) and not audit["ok"].all():
        log.append("WARNING: directionality audit failed for: " + ", ".join(audit.loc[~audit["ok"], "check"]))
    unexp_te = te[co.t[te] == 0]
    groups = {"all patients": idx, "test patients": te, "unexposed test patients": unexp_te}
    if cfg["columns"]["eligible"]:
        groups["eligible unexposed test patients"] = unexp_te[co.eligible[unexp_te] == 1]
    rng = np.random.default_rng(seed)
    ucfg = cfg["uncertainty"]
    cache, rows_out = {}, []

    def arms(gname, shift):
        key = (gname, tuple(sorted((k, round(v, 12)) for k, v in shift.items())))
        if key not in cache:
            cache[key] = prop.arm_risks(groups[gname], shift)
        return cache[key]

    for mode, es in sets.items():
        shift = {k: v for k, (v, _) in es["labs"].items()}
        ev = es["event"][0] if es["event"] is not None else None
        for gname, rows in groups.items():
            if not len(rows):
                continue
            for est in prop.estimands():
                ar = arms(gname, {} if est == "event" else shift)
                p0, p1 = prop.mix(ar, rows, ev, est)
                rows_out.append(dict(mode=mode, estimand=est, description=ESTIMAND_TEXT[est], group=gname,
                                     n=len(rows), **summarize(p0, p1, int(ucfg["n_boot"]), rng)))
    eff = pd.DataFrame(rows_out)
    eff["se_log_rr_edges"] = 0.0

    # edge uncertainty: redraw every edge with an SE from N(estimate, SE), model fixed
    for mode, es in sets.items():
        lab_unc = {k: v for k, v in es["labs"].items() if v[1] > 0}
        ev_unc = es["event"] is not None and es["event"][1] > 0
        if not lab_unc and not ev_unc:
            continue
        base_shift = {k: v for k, (v, _) in es["labs"].items()}
        for gname in ("all patients", "unexposed test patients"):
            rows_g = groups[gname]
            if not len(rows_g):
                continue
            sub = rows_g[rng.choice(len(rows_g), min(int(ucfg["psa_subset"]), len(rows_g)), replace=False)]
            ar_empty = prop.arm_risks(sub, {})
            ar_fix = prop.arm_risks(sub, base_shift)
            draws = {e: [] for e in prop.estimands()}
            for _ in range(int(ucfg["n_psa"])):
                sh = dict(base_shift)
                sh.update({k: rng.normal(v, s) for k, (v, s) in lab_unc.items()})
                ar_l = prop.arm_risks(sub, sh) if lab_unc else ar_fix
                ev_d = rng.normal(*es["event"]) if ev_unc else (es["event"][0] if es["event"] else None)
                for est in prop.estimands():
                    p0, p1 = prop.mix(ar_empty if est == "event" else ar_l, sub, ev_d, est)
                    draws[est].append(np.log(p1.mean() / p0.mean()))
            for est in prop.estimands():
                m = (eff["mode"] == mode) & (eff["group"] == gname) & (eff["estimand"] == est)
                eff.loc[m, "se_log_rr_edges"] = float(np.std(draws[est], ddof=1))
    eff["se_log_rr"] = np.sqrt(eff["se_log_rr_patients"] ** 2 + eff["se_log_rr_edges"] ** 2)
    eff["rr_ci_low"] = np.exp(eff["log_rr"] - 1.96 * eff["se_log_rr"])
    eff["rr_ci_high"] = np.exp(eff["log_rr"] + 1.96 * eff["se_log_rr"])
    order = ["mode", "estimand", "description", "group", "n", "risk_unexposed", "risk_exposed",
             "risk_difference", "rd_boot_low", "rd_boot_high", "risk_ratio", "rr_ci_low", "rr_ci_high",
             "log_rr", "se_log_rr", "se_log_rr_patients", "se_log_rr_edges", "odds_ratio", "pct_patients_benefit"]
    eff = eff[order]

    # ---------------- sensitivity ----------------
    sc = scan(prop, unexp_te, cfg["sensitivity"]["scan_gammas"], fs.lab_codes) if fs.lab_codes else None
    tor = tornado(prop, unexp_te, sets, float(cfg["sensitivity"]["tornado_pct"]))

    # ---------------- write ----------------
    thr = int(cfg["output"]["suppress_small_counts"] or 0)
    written = []
    tables = dict(effects=eff, edges_trained=tr_tab, edges_balance=bal_tab, edges_locked=locked_tab,
                  model_performance=pd.DataFrame(perf), outcome_model_weights=wts, mediator_weights=lab_w,
                  directionality_audit=audit, sensitivity_scan=sc, sensitivity_tornado=tor,
                  knowledge_support=support)
    for name, df in tables.items():
        write_table(out, f"{name}.csv", df, thr, written)
    if cfg["knowledge"]["enabled"]:
        try:
            from .knowledge import draw_graph
            first = locked_tab[locked_tab["mode"] == cfg["edges"]["modes"][0]] if len(locked_tab) else None
            if draw_graph(out / "graph.png", cfg["knowledge"]["exposure_name"], first, support,
                          cfg["columns"]["outcome"]):
                written.append("graph.png")
        except ImportError:
            log.append("graph figure skipped (install mocop[knowledge] for matplotlib)")
    if cfg["output"]["save_patient_level"]:
        es = sets[cfg["edges"]["modes"][0]]
        ar = prop.arm_risks(te, {k: v for k, (v, _) in es["labs"].items()})
        p0, p1 = prop.mix(ar, te, es["event"][0] if es["event"] else None, "total")
        pd.DataFrame({"pid": co.pid[te], "risk_unexposed": p0, "risk_exposed": p1, "difference": p1 - p0}
                     ).to_csv(out / "patient_level_test.csv", index=False)
        written.append("patient_level_test.csv  (PATIENT-LEVEL)")
        log.append("WARNING: patient-level file written (output.save_patient_level = true)")
    info = dict(mocop_version=__version__, run_seconds=round(time.time() - t0, 1), config=_redacted(cfg),
                n_patients=int(co.n), n_exposed=int(co.t.sum()), lab_codes_used=fs.lab_codes,
                covariates=co.covariates, log=list(log))
    write_run_info(out, info, written)
    if verbose:
        print(f"\nDone in {time.time() - t0:.0f} s. Files in {out}:")
        for f in written:
            print("  " + f)
    return Results(tables=tables, log=list(log), output_dir=str(out), files=written)

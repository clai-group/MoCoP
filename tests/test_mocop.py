"""Tests: estimators recover planted truth; the pipeline runs in every configuration."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mocop import load_config, run
from mocop.edges import aipw
from mocop.report import suppress
from mocop.synthetic import write_example


# ------------------------------------------------------------------ estimators
def _confounded(n, seed):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    t = (rng.random(n) < 1 / (1 + np.exp(-(-3 + 1.2 * x1 + 0.6 * x2)))).astype(float)
    return rng, x1, x2, t


def test_aipw_binary_recovers_rr_with_wrong_outcome_model():
    rng, x1, x2, t = _confounded(60000, 1)
    base = 1 / (1 + np.exp(-(-2.6 + 0.9 * x1 + 0.4 * x1 ** 2 + 0.4 * x2)))   # outcome model will be wrong
    y = (rng.random(len(t)) < np.where(t == 1, 0.70 * base, base)).astype(float)
    r = aipw(y, t, np.column_stack([x1, x2]), True)
    crude = np.log(y[t == 1].mean() / y[t == 0].mean())
    assert abs(r["est"] - np.log(0.70)) < 3 * r["se"]
    assert abs(crude - np.log(0.70)) > 0.3          # confounding is real


def test_aipw_continuous_recovers_shift():
    ests = []
    for seed in range(8):
        rng, x1, x2, t = _confounded(40000, seed)
        d = -0.05 * t + 0.3 * x1 + 0.2 * np.sin(2 * x1) + rng.normal(size=len(t)) * 0.2
        ests.append(aipw(d, t, np.column_stack([x1, x2]), False)["est"])
    assert abs(np.mean(ests) + 0.05) < 0.01


# ------------------------------------------------------------------ pipeline
@pytest.fixture(scope="module")
def example(tmp_path_factory):
    f = tmp_path_factory.mktemp("ex")
    cfg_path = write_example(f, n=20000, seed=3)
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    cfg["uncertainty"] = {"n_boot": 100, "n_psa": 50}
    Path(cfg_path).write_text(yaml.safe_dump(cfg))
    res = run(cfg_path, verbose=False)
    return f, res, json.loads((f / "truth.json").read_text())


def test_end_to_end_recovers_edges(example):
    f, res, truth = example
    t = res["edges_trained"].set_index("mediator")
    crp = t.loc["LAB_1988-5"]
    assert abs(crp["estimate"] - truth["crp_severity_shift_eligible"]) < 3 * crp["se"] + 0.02
    ev = t.loc["EVENT"]
    assert abs(ev["estimate"] - np.log(truth["event_risk_ratio"])) < 3 * ev["se"]


def test_end_to_end_direction_and_files(example):
    f, res, truth = example
    e = res["effects"]
    tot = e[(e["estimand"] == "total") & (e["group"] == "all patients")]
    assert set(tot["mode"]) == {"curated", "trained"}
    assert (tot["risk_ratio"] < 1).all()                         # planted benefit
    assert res["directionality_audit"]["ok"].all()
    out = Path(res.output_dir)
    for name in ("effects.csv", "edges_trained.csv", "edges_locked.csv", "run_info.json"):
        assert (out / name).exists()
    assert not (out / "patient_level_test.csv").exists()         # aggregate only by default
    locked = res["edges_locked"]
    cur_ev = locked[(locked["mode"] == "curated") & (locked["mediator"] == "EVENT")]["value"].iloc[0]
    assert abs(cur_ev - 0.60) < 1e-9                             # curated value used as given


def _base_cfg(f):
    return {"data": {"cohort": str(f / "cohort.csv"), "labs": str(f / "labs.csv"), "output_dir": str(f / "o2")},
            "uncertainty": {"n_boot": 50, "n_psa": 20}}


def test_minimal_config_no_event_no_dx_auto_covariates(example):
    f, _, _ = example
    co = pd.read_csv(f / "cohort.csv").drop(columns=["event", "eligible"])
    co["enrolled_on"] = "2020-01-01"                              # a date column: must be skipped
    co.to_csv(f / "cohort_min.csv", index=False)
    cfg = _base_cfg(f)
    cfg["data"]["cohort"] = str(f / "cohort_min.csv")
    res = run(cfg, verbose=False)
    assert set(res["effects"]["estimand"]) == {"total"}
    assert any("enrolled_on (looks like a date)" in m for m in res.log)


def test_curated_lab_edge_file(example):
    f, _, _ = example
    pd.DataFrame([dict(mediator="1988-5", kind="lab", value=-0.2, se=0.05)]).to_csv(f / "e.csv", index=False)
    cfg = _base_cfg(f)
    cfg["edges"] = {"modes": ["curated"], "file": str(f / "e.csv")}
    res = run(cfg, verbose=False)
    lk = res["edges_locked"]
    assert lk.loc[lk["mediator"] == "LAB_1988-5", "value"].iloc[0] == pytest.approx(-0.2)


def test_ineligible_patients_get_no_effect(example):
    f, _, _ = example
    cfg = yaml.safe_load((f / "config.yaml").read_text())
    for k in ("cohort", "labs", "dx", "output_dir"):
        cfg["data"][k] = str(f / cfg["data"][k]) + ("3" if k == "output_dir" else "")
    cfg["edges"] = {"modes": ["trained"]}
    cfg["output"] = {"save_patient_level": True}
    cfg["uncertainty"] = {"n_boot": 20, "n_psa": 10}
    res = run(cfg, verbose=False)
    pl = pd.read_csv(Path(res.output_dir) / "patient_level_test.csv", dtype={"pid": str})
    co = pd.read_csv(f / "cohort.csv", dtype={"pid": str})[["pid", "eligible"]]
    m = pl.merge(co, on="pid")
    assert np.allclose(m.loc[m["eligible"] == 0, "difference"], 0)
    assert (m.loc[m["eligible"] == 1, "difference"] != 0).any()


def test_small_counts_suppressed():
    df = pd.DataFrame({"n": [3, 50, 0], "n_treated": [12, 5, 0], "estimate": [0.1, 0.2, 0.3]})
    s = suppress(df, 11)
    assert list(s["n"]) == ["<11", 50, 0]
    assert list(s["n_treated"]) == [12, "<11", 0]


def test_config_errors(example, tmp_path):
    f, _, _ = example
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config({**_base_cfg(f), "edgez": {}})
    with pytest.raises(ValueError, match="edges.file"):
        load_config({**_base_cfg(f), "edges": {"modes": ["curated"]}})
    bad = pd.read_csv(f / "cohort.csv").drop(columns=["exposure"])
    bad.to_csv(tmp_path / "c.csv", index=False)
    cfg = _base_cfg(f)
    cfg["data"]["cohort"] = str(tmp_path / "c.csv")
    with pytest.raises(ValueError, match="missing column"):
        run(cfg, verbose=False)


def test_knowledge_graph_support_and_restriction(example, tmp_path):
    f, _, _ = example
    kg = tmp_path / "kg"
    kg.mkdir()
    pd.DataFrame([
        dict(x_name="exampledrug", relation="drug_protein", y_name="C-reactive protein"),
        dict(x_name="C-reactive protein", relation="protein_protein", y_name="interleukin 6"),
        dict(x_name="exampledrug", relation="drug_effect", y_name="anemia"),
        dict(x_name="anemia", relation="disease_phenotype_positive", y_name="low hemoglobin"),
    ]).to_csv(kg / "kg.csv", index=False)
    pd.DataFrame(dict(node_name=["exampledrug"])).to_csv(kg / "nodes.csv", index=False)
    pd.DataFrame([dict(code="1988-5", name="C-reactive protein"), dict(code="718-7", name="hemoglobin"),
                  dict(code="2160-0", name="creatinine")]).to_csv(tmp_path / "map.csv", index=False)
    cfg = _base_cfg(f)
    cfg["data"]["output_dir"] = str(tmp_path / "o")
    cfg["covariates"] = ["age", "sex", "comorbidity_score", "region", "prior_event"]
    cfg["knowledge"] = {"enabled": True, "primekg_dir": str(kg), "concept_map": str(tmp_path / "map.csv"),
                        "exposure_name": "exampledrug", "restrict_candidates": True}
    res = run(cfg, verbose=False)
    sup = res["knowledge_support"].set_index("code")["support"]
    assert sup["1988-5"] == "direct" and sup["718-7"] == "2 hops" and sup["2160-0"] == "none"
    tested = set(res["edges_trained"]["mediator"])
    assert "LAB_1988-5" in tested and "LAB_2160-0" not in tested


def test_no_leakage_from_test_patients(example, tmp_path):
    """Everything learned (edges, outcome-model weights, feature set) must be identical
    when the lab values and covariates of validation and test patients are altered."""
    from mocop.config import load_config as _lc
    from mocop.pipeline import make_split
    f, _, _ = example
    cfg = _base_cfg(f)
    cfg["covariates"] = ["age", "sex", "comorbidity_score", "region", "prior_event"]
    cfg["columns"] = {"event_mediator": "event", "eligible": "eligible"}
    cfg["data"]["output_dir"] = str(tmp_path / "a")
    a = run(cfg, verbose=False)

    co = pd.read_csv(f / "cohort.csv", dtype={"pid": str})
    tr, va, te = make_split(co["outcome"].values.astype(float), _lc(cfg))
    held_out = set(co["pid"].values[np.concatenate([va, te])])
    rng = np.random.default_rng(0)
    co2 = co.copy()
    m = co2["pid"].isin(held_out)
    co2.loc[m, "age"] = co2.loc[m, "age"] + 30
    co2.loc[m, "region"] = "unseen_region"
    co2.loc[m, "comorbidity_score"] = rng.integers(0, 20, m.sum())
    co2.to_csv(tmp_path / "cohort.csv", index=False)
    labs = pd.read_csv(f / "labs.csv", dtype={"pid": str})
    lm = labs["pid"].isin(held_out)
    labs.loc[lm, "value"] = labs.loc[lm, "value"] * rng.uniform(0.2, 5.0, lm.sum())
    labs.loc[lm, "ref_low"] = np.nan                      # forces the percentile fallback path
    labs.to_csv(tmp_path / "labs.csv", index=False)

    cfg_b = dict(cfg, data={"cohort": str(tmp_path / "cohort.csv"), "labs": str(tmp_path / "labs.csv"),
                            "output_dir": str(tmp_path / "b")})
    b = run(cfg_b, verbose=False)
    ea = a["edges_trained"].set_index("mediator")["estimate"]
    eb = b["edges_trained"].set_index("mediator")["estimate"]
    assert list(ea.index) == list(eb.index)
    assert np.allclose(ea.values, eb.values, rtol=0, atol=1e-10)
    wa = a["outcome_model_weights"].set_index("feature")["weight"].sort_index()
    wb = b["outcome_model_weights"].set_index("feature")["weight"].sort_index()
    assert list(wa.index) == list(wb.index)
    assert np.allclose(wa.values, wb.values, rtol=0, atol=1e-8)

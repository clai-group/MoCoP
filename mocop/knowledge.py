"""Optional knowledge-graph support for edges (generalised from the original script 1).

Uses PrimeKG (kg.csv and nodes.csv, downloaded by the user) and a user-supplied concept
map (code,name) that gives a plain name for each lab code. For the exposure (by its
PrimeKG name) it reports, for every lab mediator, whether the knowledge graph links the
exposure to that lab's concept directly (1 hop) or through one intermediate node (2 hops),
using the same text matching as the original script.

It never sets edge sizes. It can only (a) document knowledge support for each edge and
(b) optionally restrict which labs are tested as mediators (knowledge.restrict_candidates).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

CAUSAL_RELATIONS = {
    "indication", "contraindication", "off-label use", "drug_protein", "drug_effect", "drug_drug",
    "disease_protein", "disease_disease", "disease_phenotype_positive", "disease_phenotype_negative",
    "exposure_disease", "exposure_protein", "protein_protein", "phenotype_protein",
}
MIN_TERM = 4


def _load_kg(primekg_dir):
    p = Path(primekg_dir)
    if not (p / "kg.csv").exists():
        raise FileNotFoundError(f"PrimeKG kg.csv not found in {p}")
    nbr = defaultdict(set)
    for ch in pd.read_csv(p / "kg.csv", chunksize=500_000, usecols=["x_name", "relation", "y_name"],
                          dtype=str):
        ch = ch[ch["relation"].isin(CAUSAL_RELATIONS)]
        for x, y in zip(ch["x_name"].str.lower(), ch["y_name"].str.lower()):
            nbr[x].add(y)
            nbr[y].add(x)
    return nbr


def _match(term, name):
    return len(term) >= MIN_TERM and len(name) >= MIN_TERM and (term in name or name in term)


def lab_support(cfg, lab_codes, log):
    k = cfg["knowledge"]
    cmap = pd.read_csv(k["concept_map"], dtype=str)
    cmap.columns = [c.strip().lower() for c in cmap.columns]
    if not {"code", "name"} <= set(cmap.columns):
        raise ValueError("knowledge.concept_map needs columns: code, name")
    names = dict(zip(cmap["code"].str.strip(), cmap["name"].str.strip().str.lower()))
    nbr = _load_kg(k["primekg_dir"])
    ex = str(k["exposure_name"]).lower().strip()
    ex_nodes = [n for n in nbr if n == ex] or [n for n in nbr if _match(ex, n)]
    if not ex_nodes:
        log.append(f"WARNING: knowledge graph: exposure '{k['exposure_name']}' not found in PrimeKG")
    hop1 = set().union(*[nbr[n] for n in ex_nodes]) if ex_nodes else set()
    hop2 = {}
    for m in hop1:
        for n2 in nbr[m]:
            hop2.setdefault(n2, m)
    rows = []
    for code in lab_codes:
        term = names.get(code)
        if not term:
            rows.append(dict(code=code, concept="", support="no concept name", via=""))
            continue
        h1 = next((n for n in hop1 if _match(term, n)), None)
        if h1:
            rows.append(dict(code=code, concept=term, support="direct", via=h1))
            continue
        h2 = next((n for n in hop2 if _match(term, n)), None)
        rows.append(dict(code=code, concept=term, support="2 hops" if h2 else "none",
                         via=f"{hop2[h2]} -> {h2}" if h2 else ""))
    t = pd.DataFrame(rows)
    log.append(f"knowledge graph: exposure matched to {len(ex_nodes)} node(s); lab support: "
               f"{t['support'].value_counts().to_dict()}")
    return t


def draw_graph(path, exposure, edge_tab, support=None, outcome="outcome"):
    """Simple tiered figure: exposure -> mediators (locked edges) -> outcome."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if edge_tab is None or not len(edge_tab):
        return None
    meds = list(dict.fromkeys(edge_tab["mediator"]))
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(meds)), 4))
    xs = {m: k for k, m in enumerate(meds)}
    cx = (len(meds) - 1) / 2
    ax.text(cx, 2, exposure, ha="center", va="center", fontsize=11, weight="bold",
            bbox=dict(boxstyle="round", fc="#dbe9f6"))
    ax.text(cx, 0, outcome, ha="center", va="center", fontsize=11, weight="bold",
            bbox=dict(boxstyle="round", fc="#f6dbdb"))
    sup = {} if support is None else dict(zip("LAB_" + support["code"], support["support"]))
    for m in meds:
        v = edge_tab.loc[edge_tab["mediator"] == m, "value"].iloc[0]
        kind = edge_tab.loc[edge_tab["mediator"] == m, "kind"].iloc[0]
        lbl = f"{m}\nshift {v:+.3f}" if kind == "lab" else f"{m}\nRR {v:.2f}"
        fc = "#e3f2e1" if sup.get(m) in ("direct", "2 hops") else "#f2f2f2"
        ax.text(xs[m], 1, lbl, ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round", fc=fc))
        ax.annotate("", (xs[m], 1.15), (cx, 1.85), arrowprops=dict(arrowstyle="->", color="#555"))
        ax.annotate("", (cx, 0.15), (xs[m], 0.85), arrowprops=dict(arrowstyle="->", color="#555"))
    ax.set_xlim(-0.8, len(meds) - 0.2)
    ax.set_ylim(-0.5, 2.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

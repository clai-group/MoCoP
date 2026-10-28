"""MoCoP demo: synthetic cohort with a known answer.

Run from this folder:
    python run_demo.py

The data are simulated (no real patients). The exposure lowers C-reactive protein
(LOINC 1988-5) after the index date and lowers the risk of a secondary event (risk
ratio 0.60); both raise the risk of the outcome. truth.json holds the true values,
computed from both potential worlds of every simulated patient.
"""
import json
from pathlib import Path

import pandas as pd

from mocop import run

HERE = Path(__file__).resolve().parent
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

# 1. Run the full pipeline. Settings are in config.yaml; outputs go to ./output
res = run(HERE / "config.yaml", verbose=False)
truth = json.loads((HERE / "truth.json").read_text())

# 2. Edges learned from the cohort (training split only)
edges = res["edges_trained"].set_index("mediator")
print("Trained edges")
print(edges[["kind", "n_treated", "estimate", "ci_low", "ci_high", "max_smd_after", "used"]].round(3))
print(f"\n  CRP severity shift : estimated {edges.loc['LAB_1988-5', 'estimate']:+.3f}, "
      f"true {truth['crp_severity_shift_eligible']:+.3f}")
print(f"  Event risk ratio   : estimated {edges.loc['EVENT', 'risk_ratio']:.3f}, "
      f"true {truth['event_risk_ratio']:.3f}")

# 3. Outcome model quality
print("\nModel performance")
print(res["model_performance"][["model", "split", "auc", "brier", "ece"]].round(3).to_string(index=False))

# 4. Effect of the exposure on the outcome, by pathway (all patients)
eff = res["effects"]
main = eff[eff["group"] == "all patients"]
print("\nEffect on the outcome (all patients)")
print(main[["mode", "estimand", "risk_unexposed", "risk_exposed", "risk_ratio", "rr_ci_low", "rr_ci_high"]]
      .round(3).to_string(index=False))
print(f"\n  True total risk ratio: {truth['total_risk_ratio_all']:.3f}")

# 5. Safety checks
print("\nDirectionality audit")
print(res["directionality_audit"].to_string(index=False))
print(f"\nAll output files are in {res.output_dir}")

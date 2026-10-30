# MoCoP

Monotone Counterfactual Propagation for estimating treatment effects from electronic health
records.

MoCoP estimates the effect of a binary exposure on a binary outcome by propagating the
exposure through observed mediators (laboratory values and, optionally, one binary
post-exposure event) into a monotone-constrained outcome model, and contrasting
patient-level counterfactual risks by g-computation. Exposure → mediator edges are imported
from published estimates, estimated from the cohort with a doubly robust estimator, or both.

## Installation

```bash
pip install "git+https://github.com/clai-group/MoCoP.git"
```

Python 3.9 or later. Dependencies: numpy, pandas, scipy, scikit-learn, pyyaml.
The demo, templates and tests are in this repository
(`git clone https://github.com/clai-group/MoCoP.git`).

## Quick start

From the root of the repository:

```bash
cd examples/demo
mocop run config.yaml
```

```python
from mocop import run

res = run("config.yaml")
print(res.summary())
```

## Input

| File | Grain | Required |
|---|---|---|
| Cohort | One row per patient: identifier, index date, outcome, exposure, covariates | Yes |
| Laboratory | One row per result: identifier, date, code, value, reference interval | Yes |
| Diagnosis | One row per record: identifier, date, code | No |
| Edges | One row per exposure → mediator edge | No |

## Documentation

[docs/USER_GUIDE.md](docs/USER_GUIDE.md): data specification, configuration reference,
output reference, methods, data partitioning, data protection, limitations and the demo.

## Repository layout

| Path | Content |
|---|---|
| `mocop/` | Package source |
| `docs/USER_GUIDE.md` | User guide |
| `examples/demo/` | Synthetic demonstration dataset, script, notebook and reference results |
| `examples/*.csv`, `examples/config_example.yaml` | Templates for edges, laboratory directions, concept maps and configuration |
| `tests/` | Test suite (`pytest tests`) |

## Data protection

MoCoP runs locally, makes no network connections and writes aggregate outputs only by
default, with small-count suppression. See Section 11 of the user guide.

## Citation

If you use MoCoP, please cite it using the metadata in [CITATION.cff](CITATION.cff)
(GitHub: "Cite this repository").

## License

MIT. See [LICENSE](LICENSE).

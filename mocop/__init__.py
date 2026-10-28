"""MoCoP: Monotone Counterfactual Propagation.

Estimates the effect of a binary exposure on a binary outcome by propagating the
exposure through mediators (lab values and, optionally, one secondary-outcome event)
into a monotone-constrained outcome model, then contrasting counterfactual risks.

Quick start:
    from mocop import run
    results = run("config.yaml")
"""
__version__ = "1.0.0"

from .config import load_config, Config          # noqa: E402,F401
from .pipeline import run                        # noqa: E402,F401

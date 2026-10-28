"""Command line:  mocop run CONFIG | mocop check CONFIG | mocop example FOLDER"""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _check(path):
    from .config import load_config
    from .io import load_cohort, load_dx, load_labs
    cfg = load_config(path)
    co = load_cohort(cfg)
    load_labs(cfg, co)
    load_dx(cfg, co)
    print("Config and data look usable:")
    for m in co.log:
        print("  " + m)


def main(argv=None):
    p = argparse.ArgumentParser(prog="mocop", description=f"MoCoP {__version__}: Monotone Counterfactual Propagation")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run MoCoP with a YAML config")
    r.add_argument("config")
    r.add_argument("--quiet", action="store_true")
    c = sub.add_parser("check", help="check the config and input files without running the model")
    c.add_argument("config")
    e = sub.add_parser("example", help="write a synthetic example dataset and config to a folder")
    e.add_argument("folder")
    e.add_argument("--patients", type=int, default=5000)
    a = p.parse_args(argv)
    try:
        if a.cmd == "run":
            from .pipeline import run
            res = run(a.config, verbose=not a.quiet)
            print("\nMain estimates (all patients):")
            print(res.summary())
        elif a.cmd == "check":
            _check(a.config)
        elif a.cmd == "example":
            from .synthetic import write_example
            cfg_path = write_example(a.folder, n=a.patients)
            print(f"Example written. Run it with:\n  mocop run {cfg_path}")
    except (ValueError, FileNotFoundError) as err:
        print(f"\nERROR: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

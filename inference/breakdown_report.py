#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Print the per-component comm & memory breakdown saved in one or more
# results.json files; with two or more, also print a side-by-side.
#
#     python inference/breakdown_report.py <cse_fsl>/results.json <fsl_sage>/results.json
#
# Each row names the paper step that produced the bytes (see
# src/utils/breakdown.py and docs/CSE_FSL_FSL_SAGE_BREAKDOWN.md).
# ------------------------------------------------------------------------------
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

from utils.breakdown import render, render_side_by_side   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', nargs='+', help='results.json path(s)')
    args = ap.parse_args()

    bds = []
    for path in args.results:
        with open(path) as f:
            res = json.load(f)
        bd = res.get('breakdown')
        if bd is None:
            print(f"[skip] {path}: no 'breakdown' key (run predates it)")
            continue
        bds.append(bd)
        print(f"<!-- {path} -->")
        print(render(bd))
        print()

    if len(bds) > 1:
        print(render_side_by_side(bds))


if __name__ == '__main__':
    main()

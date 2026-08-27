#!/usr/bin/env python
# ------------------------------------------------------------------------------
# One-command orchestrator: sweeps every MNIST method across both
# distributions, builds the ranked comparison table, and generates
# accuracy/comm-load plots -- the single entry point referenced by the Colab
# notebook's "run everything" cell.
#
# Must be run with `inference/` as the working directory (same precondition
# `plot_results.py` already has, since it resolves `src.utils.plot_util` via a
# `sys.path.append("../")` relative to the process's cwd at import time).
# ------------------------------------------------------------------------------
import argparse
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from results_loader import find_latest_run, load_run
from benchmark_table import build_table, load_config as load_table_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, 'src')

DEFAULT_METHODS = [
    'sl_multi_server', 'sl_single_server', 'cse_fsl', 'fsl_sage', 'ho_sfl',
    'mu_splitfed',
]
DISTRIBUTIONS = [('iid', None), ('noniid_dirichlet', 0.5)]

# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Sweep all MNIST methods, build the ranked comparison table, and "
            "generate accuracy/comm-load plots. Run with `inference/` as cwd."
        )
    )
    p.add_argument('--rounds', type=int, default=3,
                    help="rounds=3 is a pipeline check, not a reportable "
                         "result -- raise this for a real run (README default: 200)")
    p.add_argument('--seed', type=int, default=200)
    p.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                    help="algorithm registry keys to sweep")
    p.add_argument('--device', default=None,
                    help="cuda or cpu; default auto-detects via "
                         "torch.cuda.is_available()")
    p.add_argument('--num_clients', type=int, default=None,
                    help="override cfg.num_clients; default leaves the "
                         "harness's own default (10)")
    p.add_argument('--skip_sweep', action='store_true',
                    help="skip training and just (re)build the table/plots "
                         "from whatever is already under saves/")
    return p.parse_args()

# ------------------------------------------------------------------------------
def run_sweep(methods, rounds, seed, device, num_clients):
    failures = []
    for algo in methods:
        for distribution, alpha in DISTRIBUTIONS:
            cmd = [
                sys.executable, 'main.py',
                f'algorithm={algo}', 'model=resnet18', 'dataset=mnist',
                f'dataset.distribution={distribution}',
                f'rounds={rounds}', f'seed={seed}', 'save=True',
                f'device={device}',
            ]
            if alpha is not None:
                cmd.append(f'dataset.alpha={alpha}')
            if num_clients is not None:
                cmd.append(f'num_clients={num_clients}')

            tag = f'{algo} / mnist-{distribution}'
            tag += f' (alpha={alpha})' if alpha is not None else ''
            print(f"\n=== {tag} ===")

            env = dict(os.environ, WANDB_MODE='offline')
            result = subprocess.run(cmd, cwd=SRC_DIR, env=env)
            if result.returncode != 0:
                print(f"[FAILED] {tag} (exit {result.returncode})")
                failures.append((algo, distribution))

    if failures:
        print(f"\nSweep finished with failures: {failures}")
    else:
        print("\nSweep finished: all runs succeeded.")
    return failures

# ------------------------------------------------------------------------------
def make_plots(cfg):
    # Local import: plot_results.py appends '../' to sys.path and imports
    # `src.utils.plot_util`, which only resolves correctly when this
    # process's cwd is `inference/` -- same precondition plot_results.py has
    # always had.
    import plot_results as pr

    columns = [('iid', None), ('noniid_dirichlet', cfg['alpha'])]
    plots_root = os.path.join(os.path.dirname(__file__), 'plots_mnist')

    scatter_dict = {name: {} for name in cfg['methods']}

    pr.setup()
    for distribution, alpha in columns:
        save_dicts = {}
        for name, meta in cfg['methods'].items():
            path = find_latest_run(
                cfg['prefix_dir'], meta['key'], cfg['model'], cfg['dataset'],
                distribution, alpha=alpha
            )
            if path is None:
                continue
            run = load_run(path)
            # accuracy_plot indexes save_dicts values as v[0][...] -- wrap in
            # a 1-element list to match its multi-seed convention rather than
            # relying on its single-dict code path.
            save_dicts[name] = [run]
            # proxy alpha for the "iid" column so metrics_vs_comm_load_scatter's
            # log-scaled point-size convention (which assumes alpha > 0) still
            # applies; 100.0 stands in for "close to infinite" (iid).
            scatter_dict[name][alpha if alpha is not None else 100.0] = run

        if not save_dicts:
            print(f"[skip plots] no runs found yet for mnist-{distribution}")
            continue

        plot_dir = os.path.join(plots_root, distribution)
        os.makedirs(plot_dir, exist_ok=True)
        test_ids = list(range(len(save_dicts)))
        pr.accuracy_plot(
            save_dicts, ['test_acc', 'test_loss'],
            ['Test Accuracy', 'Test Loss'], test_ids=test_ids,
            metric_minimize=[False, True], plots_dir=plot_dir
        )
        pr.accuracy_plot(
            save_dicts, ['test_acc', 'test_loss'],
            ['Test Accuracy', 'Test Loss'], test_ids=test_ids,
            metric_minimize=[False, True], x_comm_load=True, plots_dir=plot_dir
        )
        print(f"[plots] wrote accuracy/loss vs round + comm-load to {plot_dir}")

    scatter_dict = {k: v for k, v in scatter_dict.items() if v}
    if scatter_dict:
        scatter_dir = os.path.join(plots_root, 'comm_load_scatter')
        pr.metrics_vs_comm_load_scatter(
            scatter_dict, ['test_acc', 'test_loss'],
            ['Test Accuracy', 'Test Loss'],
            test_ids=list(range(len(scatter_dict))),
            metric_minimize=[False, True], plots_dir=scatter_dir
        )
        print(f"[plots] wrote accuracy/loss vs comm-load scatter to {scatter_dir}")

# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(SRC_DIR):
        raise SystemExit(
            f"Expected to find {SRC_DIR} -- run this script with `inference/` "
            "as the working directory."
        )

    if not args.skip_sweep:
        run_sweep(args.methods, args.rounds, args.seed, device, args.num_clients)

    cfg = load_table_config()
    build_table(cfg)
    make_plots(cfg)

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------

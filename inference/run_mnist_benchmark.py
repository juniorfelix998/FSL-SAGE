#!/usr/bin/env python
# ------------------------------------------------------------------------------
# One-command orchestrator: sweeps every MNIST method across both
# distributions, builds the comparison table (comm cut/weights/total, cut vs.
# weights share, accuracy, latency, peak memory), and generates
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
    'sl_multi_server', 'sl_single_server', 'vanilla_sl', 'cse_fsl',
    'fsl_sage', 'ho_sfl', 'mu_splitfed', 'dsl_aux', 'han_locloss',
    'fedsplitx', 'hosl', 'locfedmix_sl',
]

# These two process exactly ONE batch per client per round -- faithful to their
# reference implementation, which draws `next(loader)` each round. Every other
# method here takes one optimizer step per BATCH, so on MNIST with 10 clients
# (6000 samples each, batch 256 -> 24 batches) an equal `rounds` budget hands
# these two 24x fewer updates; the reference itself runs ~375 effective rounds.
# Scaling their round count is what makes the accuracy column mean anything.
# The communication columns stay comparable regardless, because Comm-to-target
# is read off at a fixed ACCURACY rather than at a fixed round.
ONE_BATCH_PER_ROUND_METHODS = ('ho_sfl', 'mu_splitfed')
ZO_ROUND_MULTIPLIER = 24
DISTRIBUTIONS = [('iid', None), ('noniid_dirichlet', 0.5)]
ALL_CUTS = ['shallow', 'middle', 'deep']

# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Sweep all MNIST methods, build the comparison table, and "
            "generate accuracy/comm-load plots. Run with `inference/` as cwd."
        )
    )
    p.add_argument('--rounds', type=int, default=3,
                    help="rounds=3 is a pipeline check, not a reportable "
                         "result -- raise this for a real run (README default: 200)")
    p.add_argument('--seed', type=int, default=200)
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                    help="run and average over several seeds -- CLAUDE.md's "
                         "protocol is 3. Overrides --seed; the table then "
                         "reports mean+/-std per cell instead of one run")
    p.add_argument('--target_acc', type=float, default=None,
                    help="accuracy at which Comm-to-target is read off "
                         "(default: config.yaml's target_acc, 0.97 for MNIST)")
    p.add_argument('--zo_round_multiplier', type=int,
                    default=ZO_ROUND_MULTIPLIER,
                    help="multiply --rounds by this for ho_sfl/mu_splitfed, "
                         "which do one batch per client per round (see the "
                         "module comment). Pass 1 to run every method at "
                         "--rounds")
    p.add_argument('--comm_threshold_mb', type=float, default=None,
                    help="raise the communication budget that early-stops a "
                         "run (config.yaml default 204800 = 200 GiB). "
                         "MU-SplitFed at 10 clients burns ~890 MB/round and "
                         "would otherwise truncate silently at ~230 rounds")
    p.add_argument('--measure_memory', default=None,
                    choices=['true', 'false'],
                    help="per-side memory instrumentation (default on). "
                         "Set false for a pure throughput run")
    p.add_argument('--methods', nargs='+', default=DEFAULT_METHODS,
                    help="algorithm registry keys to sweep")
    p.add_argument('--cuts', nargs='+', default=['middle'], choices=ALL_CUTS,
                    help="which cut(s) to sweep (see hydra_config/cut/*.yaml) "
                         "-- default is just 'middle' (this harness's original "
                         "behavior); pass e.g. --cuts shallow middle deep to "
                         "compare a method's sensitivity to cut depth "
                         "(each cut produces its own table/plots)")
    p.add_argument('--device', default=None,
                    help="cuda or cpu; default auto-detects via "
                         "torch.cuda.is_available()")
    p.add_argument('--num_clients_list', type=int, nargs='+', default=None,
                    help="sweep multiple client counts (e.g. --num_clients_list "
                         "2 10 100), each producing its own table/plots -- "
                         "default runs once at the harness's own default (10), "
                         "same as before this became a sweep dimension")
    p.add_argument('--skip_sweep', action='store_true',
                    help="skip training and just (re)build the table/plots "
                         "from whatever is already under saves/")
    return p.parse_args()

# ------------------------------------------------------------------------------
def rounds_for(algo, rounds, zo_round_multiplier):
    '''Round budget for one method. See ONE_BATCH_PER_ROUND_METHODS above for
    why two of them get a larger one.'''
    if algo in ONE_BATCH_PER_ROUND_METHODS:
        return max(1, rounds * max(1, zo_round_multiplier))
    return rounds


def run_sweep(methods, rounds, seeds, device, num_clients_values, cuts,
               zo_round_multiplier=ZO_ROUND_MULTIPLIER, target_acc=None,
               comm_threshold_mb=None, measure_memory=None):
    failures = []
    for cut in cuts:
        for num_clients in num_clients_values:
            for algo in methods:
                algo_rounds = rounds_for(algo, rounds, zo_round_multiplier)
                for seed in seeds:
                    for distribution, alpha in DISTRIBUTIONS:
                        cmd = [
                            sys.executable, 'main.py',
                            f'algorithm={algo}', 'model=resnet18',
                            'dataset=mnist', f'cut={cut}',
                            f'dataset.distribution={distribution}',
                            f'rounds={algo_rounds}', f'seed={seed}',
                            'save=True', f'device={device}',
                        ]
                        if alpha is not None:
                            cmd.append(f'dataset.alpha={alpha}')
                        if num_clients is not None:
                            cmd.append(f'num_clients={num_clients}')
                        if target_acc is not None:
                            cmd.append(f'target_acc={target_acc}')
                        if comm_threshold_mb is not None:
                            cmd.append(f'comm_threshold_mb={comm_threshold_mb}')
                        if measure_memory is not None:
                            cmd.append(f'measure_memory={measure_memory}')

                        tag = (f'{algo} / mnist-{distribution} / cut={cut} / '
                               f'num_clients={num_clients} / seed={seed} / '
                               f'rounds={algo_rounds}')
                        tag += f' (alpha={alpha})' if alpha is not None else ''
                        print(f"\n=== {tag} ===")

                        env = dict(os.environ, WANDB_MODE='offline')
                        result = subprocess.run(cmd, cwd=SRC_DIR, env=env)
                        if result.returncode != 0:
                            print(f"[FAILED] {tag} (exit {result.returncode})")
                            failures.append(
                                (algo, distribution, cut, num_clients, seed)
                            )

    if failures:
        print(f"\nSweep finished with failures: {failures}")
    else:
        print("\nSweep finished: all runs succeeded.")
    return failures

# ------------------------------------------------------------------------------
DISTRIBUTION_MARKERS = {'iid': 'o', 'noniid_dirichlet': '^'}


def make_plots(cfg, cut='middle', num_clients=None):
    # Local import: plot_results.py appends '../' to sys.path and imports
    # `src.utils.plot_util`, which only resolves correctly when this
    # process's cwd is `inference/` -- same precondition plot_results.py has
    # always had.
    import plot_results as pr
    import matplotlib.pyplot as plt

    columns = [('iid', None), ('noniid_dirichlet', cfg['alpha'])]
    plots_dir_name = 'plots_mnist' if cut == 'middle' else f'plots_mnist_{cut}'
    if num_clients is not None:
        plots_dir_name += f'_nc{num_clients}'
    plots_root = os.path.join(os.path.dirname(__file__), plots_dir_name)

    pr.setup()
    # Fix one color per method up front so it stays identical across every
    # plot below, regardless of which subset of methods has data for a given
    # distribution (relying on axes.prop_cycle position instead would let
    # colors drift whenever a run is missing).
    cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    method_colors = {
        name: cycle_colors[i % len(cycle_colors)]
        for i, name in enumerate(cfg['methods'])
    }

    runs_by_distribution = {}
    for distribution, alpha in columns:
        runs = {}
        for name, meta in cfg['methods'].items():
            path = find_latest_run(
                cfg['prefix_dir'], meta['key'], cfg['model'], cfg['dataset'],
                distribution, alpha=alpha, cut=cut, num_clients=num_clients
            )
            if path is None:
                continue
            runs[name] = load_run(path)
        runs_by_distribution[distribution] = runs

        if not runs:
            print(f"[skip plots] no runs found yet for mnist-{distribution}")
            continue

        dist_label = 'IID' if distribution == 'iid' else f'α={alpha}'
        plot_dir = os.path.join(plots_root, distribution)
        os.makedirs(plot_dir, exist_ok=True)
        save_dicts = {name: [run] for name, run in runs.items()}
        test_ids = list(range(len(save_dicts)))
        for x_comm_load in (False, True):
            pr.accuracy_plot(
                save_dicts, ['test_acc', 'test_loss'],
                ['Test Accuracy', 'Test Loss'], test_ids=test_ids,
                metric_minimize=[False, True], x_comm_load=x_comm_load,
                plots_dir=plot_dir, title=f'MNIST ({dist_label})'
            )
        print(f"[plots] wrote accuracy/loss vs round + comm-load to {plot_dir}")

    _comm_load_scatter(runs_by_distribution, method_colors, plots_root)


def _comm_load_scatter(runs_by_distribution, method_colors, plots_root):
    '''Final test_acc vs total comm_load, one point per (method, distribution).

    Deliberately not `plot_results.metrics_vs_comm_load_scatter`: that
    function's point-size legend is built for a continuous sweep over many
    real Dirichlet alpha values, which doesn't fit having just two discrete
    conditions here (IID vs alpha=0.5) -- it produces a confusing legend
    entry and overlapping same-size dots instead of a clean comparison.
    '''
    import matplotlib.pyplot as plt

    scatter_dir = os.path.join(plots_root, 'comm_load_scatter')
    os.makedirs(scatter_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    seen_methods = set()
    for distribution, runs in runs_by_distribution.items():
        marker = DISTRIBUTION_MARKERS[distribution]
        for name, run in runs.items():
            x = run['comm_load'][-1] / (1024 ** 3)
            y = run['test_acc'][-1] * 100.0
            ax.scatter(
                x, y, marker=marker, s=80, color=method_colors[name],
                edgecolors='black', linewidths=0.5, alpha=0.9
            )
            seen_methods.add(name)

    method_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=name,
                    markerfacecolor=method_colors[name], markersize=8)
        for name in method_colors if name in seen_methods
    ]
    method_legend = ax.legend(
        handles=method_handles, loc='lower right', fontsize=8, title='Method'
    )
    ax.add_artist(method_legend)

    dist_handles = [
        plt.Line2D([0], [0], marker=DISTRIBUTION_MARKERS[d], color='black',
                    linestyle='', label=('IID' if d == 'iid' else 'α=0.5'),
                    markerfacecolor='white', markersize=8)
        for d in runs_by_distribution if runs_by_distribution[d]
    ]
    ax.legend(handles=dist_handles, loc='lower left', fontsize=8, title='Distribution')

    ax.set_xlabel('Communication Load (GB)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Final Accuracy vs Communication Load')
    ax.grid(True, which='both', axis='both', linestyle='dotted', linewidth=0.5,
            color='gray', alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(scatter_dir, 'test_acc_vs_commload_scatter.png'))
    fig.savefig(os.path.join(scatter_dir, 'test_acc_vs_commload_scatter.eps'))
    plt.close(fig)
    print(f"[plots] wrote accuracy vs comm-load scatter to {scatter_dir}")

# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.isdir(SRC_DIR):
        raise SystemExit(
            f"Expected to find {SRC_DIR} -- run this script with `inference/` "
            "as the working directory."
        )

    num_clients_values = args.num_clients_list or [None]
    seeds = args.seeds or [args.seed]

    if not args.skip_sweep:
        run_sweep(
            args.methods, args.rounds, seeds, device, num_clients_values,
            args.cuts, zo_round_multiplier=args.zo_round_multiplier,
            target_acc=args.target_acc,
            comm_threshold_mb=args.comm_threshold_mb,
            measure_memory=args.measure_memory,
        )

    cfg = load_table_config()
    # pass seeds through only for a genuine multi-seed sweep, so a single-seed
    # run keeps matching whatever seed is on disk rather than pinning one
    table_seeds = seeds if len(seeds) > 1 else None
    for cut in args.cuts:
        for num_clients in num_clients_values:
            build_table(cfg, cut=cut, num_clients=num_clients,
                        seeds=table_seeds, target_acc=args.target_acc)
            make_plots(cfg, cut=cut, num_clients=num_clients)

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------

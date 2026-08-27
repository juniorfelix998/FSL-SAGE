# ------------------------------------------------------------------------------
# Cross-method, ranked comparison table for the MNIST phase of the benchmark
# (see CLAUDE.md's "What Crosses the Cut?" target table). Kept separate from
# `exp_config.yaml`/`plot_results.py::make_table` -- that schema is purpose-built
# for the per-experiment line-plot functions and hand-maintained timestamped
# paths, not a ranking table spanning multiple methods automatically.
#
# Auto-discovers the most recent matching run per method/distribution via
# `results_loader.find_latest_run` rather than requiring hand-edited paths, so
# this can be re-run as more methods/seeds are added without config edits.
# ------------------------------------------------------------------------------
import os
import yaml
import numpy as np
from scipy.stats import rankdata
from prettytable import PrettyTable

from results_loader import find_latest_run, load_run

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), 'benchmark_table_config.yaml'
)
DEFAULT_OUT_PATH = os.path.join(
    os.path.dirname(__file__), 'benchmark_table_mnist.txt'
)
REIMPL_FOOTNOTE = (
    "* Third-party CV/ResNet18 reimplementation (HKU-WILL-Lab/HO-SFL repo), "
    "not the original MU-SplitFed authors' code."
)

# ------------------------------------------------------------------------------
def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------------------------
def collect_cell(cfg, algo_key, distribution, alpha=None):
    '''Find the latest run for (algo_key, distribution[, alpha]) and pull its
    final-round test accuracy (%) and total comm_load (bytes). Returns None if
    no matching run exists yet.'''
    path = find_latest_run(
        cfg['prefix_dir'], algo_key, cfg['model'], cfg['dataset'], distribution,
        alpha=alpha
    )
    if path is None:
        return None
    run = load_run(path)
    return {'acc': run['test_acc'][-1] * 100.0, 'comm_load': run['comm_load'][-1]}

# ------------------------------------------------------------------------------
def columns_spec(cfg):
    '''(display_name, distribution, alpha) for each table column.'''
    return [
        ('MNIST α=∞ (IID)', 'iid', None),
        (f"MNIST α={cfg['alpha']}", 'noniid_dirichlet', cfg['alpha']),
    ]

# ------------------------------------------------------------------------------
def compute_ranks(methods, cells, num_columns):
    '''Per column: rank by accuracy (higher better) and by comm_load (lower
    better), average those two ranks. Final R per method = mean of its
    per-column combined ranks, over columns where it has data.'''
    combined_ranks = {name: [] for name in methods}
    for col_idx in range(num_columns):
        present = [
            (name, cells[name][col_idx]) for name in methods
            if cells[name][col_idx] is not None
        ]
        if not present:
            continue
        names = [n for n, _ in present]
        accs = np.array([c['acc'] for _, c in present])
        comms = np.array([c['comm_load'] for _, c in present])
        acc_ranks = rankdata(-accs, method='average')
        comm_ranks = rankdata(comms, method='average')
        for n, ar, cr in zip(names, acc_ranks, comm_ranks):
            combined_ranks[n].append((ar + cr) / 2.0)
    return combined_ranks

# ------------------------------------------------------------------------------
def build_table(cfg=None, out_path=None):
    cfg = cfg or load_config()
    columns = columns_spec(cfg)
    methods = list(cfg['methods'].keys())

    cells = {
        name: [
            collect_cell(cfg, cfg['methods'][name]['key'], dist, alpha)
            for _, dist, alpha in columns
        ] for name in methods
    }
    ranks = compute_ranks(methods, cells, len(columns))

    table = PrettyTable()
    table.field_names = ['Method'] + [c[0] for c in columns] + ['R']

    footnote_needed = False
    for name in methods:
        meta = cfg['methods'][name]
        is_reimpl = meta.get('reimplementation', False)
        footnote_needed = footnote_needed or is_reimpl
        label = f"{name}*" if is_reimpl else name

        row = [label]
        for cell in cells[name]:
            row.append(f"{cell['acc']:.2f}" if cell is not None else 'N/A')
        r_vals = ranks[name]
        row.append(f"{np.mean(r_vals):.2f}" if r_vals else 'N/A')
        table.add_row(row)

    print(table)
    if footnote_needed:
        print(REIMPL_FOOTNOTE)

    out_path = out_path or DEFAULT_OUT_PATH
    with open(out_path, 'w') as f:
        print(table, file=f)
        if footnote_needed:
            print(REIMPL_FOOTNOTE, file=f)

    return table

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    build_table()

# ------------------------------------------------------------------------------

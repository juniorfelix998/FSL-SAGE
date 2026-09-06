# ------------------------------------------------------------------------------
# Cross-method measurement table for the MNIST phase of the benchmark (comm
# cut/weights/total, cut vs. weights share, accuracy, latency, peak memory --
# see CLAUDE.md's "What Crosses the Cut?" target table). Kept separate from
# `exp_config.yaml`/`plot_results.py::make_table` -- that schema is purpose-built
# for the per-experiment line-plot functions and hand-maintained timestamped
# paths, not a comparison table spanning multiple methods automatically.
#
# Auto-discovers the most recent matching run per method/distribution via
# `results_loader.find_latest_run` rather than requiring hand-edited paths, so
# this can be re-run as more methods/seeds are added without config edits.
# ------------------------------------------------------------------------------
import os
import yaml
from prettytable import PrettyTable

from results_loader import find_latest_run, load_run

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), 'benchmark_table_config.yaml'
)
def default_out_path(cut='middle', num_clients=None):
    suffix = '' if cut == 'middle' else f'_{cut}'
    if num_clients is not None:
        suffix += f'_nc{num_clients}'
    return os.path.join(
        os.path.dirname(__file__), f'benchmark_table_mnist{suffix}.txt'
    )
# per-method provenance footnotes, keyed by the algorithm's config key
# (`cfg['methods'][name]['key']`) -- each row that's flagged `reimplementation:
# true` in benchmark_table_config.yaml gets labeled with its own symbol, and
# only the distinct footnotes actually used by a row in the table get printed
# (not concatenated/shared across methods with different provenance stories).
REIMPL_FOOTNOTES = {
    'mu_splitfed': (
        '*',
        "* Third-party CV/ResNet18 reimplementation (HKU-WILL-Lab/HO-SFL repo), "
        "not the original MU-SplitFed authors' code."
    ),
    'dsl_aux': (
        '†',
        "† AI-assisted no-code reimplementation of DSL-Aux (arXiv:2601.19261), "
        "adapted from a partial third-party reference implementation "
        "(juniorfelix998/sl-fl-dgl); not validated against the paper's own "
        "reported numbers."
    ),
}
DEFAULT_REIMPL_SYMBOL = '*'

# ------------------------------------------------------------------------------
def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------------------------
def collect_cell(cfg, algo_key, distribution, alpha=None, cut='middle', num_clients=None):
    '''Find the latest run for (algo_key, distribution[, alpha], cut[,
    num_clients]) and pull its final-round test accuracy (%),
    cut/weights/total comm_load (bytes), and the run's latency (s) and peak
    memory (MB). Returns None if no matching run exists yet.

    `latency_s`/`peak_memory_mb` are bare scalars in `results.json` (unlike
    `test_acc`/`comm_load*`, which are per-round lists) -- not `[-1]`-indexed.
    '''
    path = find_latest_run(
        cfg['prefix_dir'], algo_key, cfg['model'], cfg['dataset'], distribution,
        alpha=alpha, cut=cut, num_clients=num_clients
    )
    if path is None:
        return None
    run = load_run(path)
    comm_cut = run['comm_load_cut'][-1]
    comm_weights = run['comm_load_weights'][-1]
    comm_total = run['comm_load'][-1]
    return {
        'acc': run['test_acc'][-1] * 100.0,
        'comm_load_cut': comm_cut,
        'comm_load_weights': comm_weights,
        'comm_load': comm_total,
        'cut_share_pct': 100.0 * comm_cut / comm_total if comm_total > 0 else 0.0,
        'latency_s': run['latency_s'],
        'peak_memory_mb': run['peak_memory_mb'],
    }

# ------------------------------------------------------------------------------
def columns_spec(cfg):
    '''(display_name, distribution, alpha) for each table column.'''
    return [
        ('MNIST α=∞ (IID)', 'iid', None),
        (f"MNIST α={cfg['alpha']}", 'noniid_dirichlet', cfg['alpha']),
    ]

# ------------------------------------------------------------------------------
def to_mb(comm_load_bytes):
    return comm_load_bytes / (1024 ** 2)

# ------------------------------------------------------------------------------
def build_table(cfg=None, out_path=None, cut='middle', num_clients=None):
    cfg = cfg or load_config()
    columns = columns_spec(cfg)
    methods = list(cfg['methods'].keys())

    cells = {
        name: [
            collect_cell(
                cfg, cfg['methods'][name]['key'], dist, alpha,
                cut=cut, num_clients=num_clients
            )
            for _, dist, alpha in columns
        ] for name in methods
    }

    table = PrettyTable()
    col_names = ['Method']
    for name, _, _ in columns:
        col_names += [
            f'{name} Comm-cut (MB)', f'{name} Comm-weights (MB)',
            f'{name} Comm-total (MB)', f'{name} Cut vs. weights',
            f'{name} Acc (%)', f'{name} Latency (s)', f'{name} Peak mem (MB)',
        ]
    table.field_names = col_names

    footnotes_used = []
    for name in methods:
        meta = cfg['methods'][name]
        is_reimpl = meta.get('reimplementation', False)
        symbol, footnote = REIMPL_FOOTNOTES.get(
            meta['key'], (DEFAULT_REIMPL_SYMBOL, None)
        )
        if is_reimpl and footnote is not None and footnote not in footnotes_used:
            footnotes_used.append(footnote)
        label = f"{name}{symbol}" if is_reimpl else name

        row = [label]
        for cell in cells[name]:
            if cell is not None:
                row += [
                    f"{to_mb(cell['comm_load_cut']):.2f}",
                    f"{to_mb(cell['comm_load_weights']):.2f}",
                    f"{to_mb(cell['comm_load']):.2f}",
                    f"{cell['cut_share_pct']:.0f}% cut",
                    f"{cell['acc']:.2f}",
                    f"{cell['latency_s']:.2f}",
                    f"{cell['peak_memory_mb']:.2f}",
                ]
            else:
                row += ['N/A'] * 7
        table.add_row(row)

    header = f"Cut: {cut}, num_clients: {num_clients if num_clients is not None else 'default'}"
    print(header)
    print(table)
    for footnote in footnotes_used:
        print(footnote)

    out_path = out_path or default_out_path(cut, num_clients)
    with open(out_path, 'w') as f:
        print(header, file=f)
        print(table, file=f)
        for footnote in footnotes_used:
            print(footnote, file=f)

    return table

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    build_table()

# ------------------------------------------------------------------------------

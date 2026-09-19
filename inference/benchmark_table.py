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
import math
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
    'han_locloss': (
        '‡',
        "‡ AI-assisted no-code reimplementation of Han et al., \"Accelerating "
        "FL with SL on Locally Generated Losses\" (FL-ICML 2021); not "
        "validated against the paper's own reported numbers."
    ),
    'fedsplitx': (
        '§',
        "§ AI-assisted no-code reimplementation of FedSplitX (arXiv:2310.14579), "
        "run at a single shared cut (this benchmark's protocol) rather than "
        "the paper's own multiple simultaneous depth-levels; not validated "
        "against the paper's own reported numbers."
    ),
    'hosl': (
        '¶',
        "¶ AI-assisted no-code reimplementation of HOSL (arXiv:2601.10940); "
        "not validated against the paper's own reported numbers."
    ),
    'locfedmix_sl': (
        '‖',
        "‖ AI-assisted no-code reimplementation of LocFedMix-SL (ACM WWW "
        "2022); not validated against the paper's own reported numbers."
    ),
}
DEFAULT_REIMPL_SYMBOL = '*'

# ------------------------------------------------------------------------------
def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------------------------
def _summarise(runs, key, scale=1.0, per_round=False):
    '''(mean, std, n) of one metric across the seed runs for a cell.

    Returns (None, None, 0) when no run reports the metric -- which happens for
    a metric added after those runs were produced. That is deliberately
    propagated as "not available" rather than silently coerced to 0.
    '''
    vals = []
    for run in runs:
        v = run.get(key)
        if per_round:
            v = v[-1] if v else None
        if v is None:
            continue
        vals.append(float(v) * scale)
    if not vals:
        return None, None, 0
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var), len(vals)


def _summarise_breakdown(runs, keys, scale=1.0):
    '''(mean, std, n) of a SUM of `comm_breakdown` sub-categories.

    The per-category ledger is kept in `comm_breakdown` (a cumulative snapshot
    per round), not at the top level, so `_summarise` cannot reach it. Used to
    surface how much of a method's weight traffic is server-side aggregation --
    a convention this harness charges and the reference implementations do not.
    '''
    vals = []
    for run in runs:
        bd = run.get('comm_breakdown')
        if not bd:
            continue
        last = bd[-1]
        vals.append(sum(float(last.get(k, 0.0)) for k in keys) * scale)
    if not vals:
        return None, None, 0
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var), len(vals)


# ------------------------------------------------------------------------------
def collect_cell(cfg, algo_key, distribution, alpha=None, cut='middle',
                  num_clients=None, seeds=None):
    '''Collect one table cell, averaged over `seeds` (CLAUDE.md: "Seeds: 3
    default"). With `seeds=None` the single most recent matching run is used,
    which is the pre-multi-seed behaviour.

    Also returns the run's own (rounds, num_clients, cut, seed) so `build_table`
    can flag cells that are not actually comparable -- the previous version
    silently compared a 5-round 1-client run against a 3-round 2-client one.
    '''
    seed_list = seeds if seeds else [None]
    runs, paths = [], []
    for seed in seed_list:
        path = find_latest_run(
            cfg['prefix_dir'], algo_key, cfg['model'], cfg['dataset'],
            distribution, alpha=alpha, cut=cut, num_clients=num_clients,
            seed=seed
        )
        if path is not None:
            runs.append(load_run(path))
            paths.append(path)
    if not runs:
        return None

    MB = 1024 ** 2
    cut_mean, _, _ = _summarise(runs, 'comm_load_cut', per_round=True)
    tot_mean, _, _ = _summarise(runs, 'comm_load', per_round=True)

    def fields(*specs):
        return {label: _summarise(runs, key, scale, per_round)
                for label, key, scale, per_round in specs}

    cell = fields(
        ('acc',            'test_acc',                      100.0, True),
        ('comm_cut',       'comm_load_cut',              1.0 / MB, True),
        ('comm_weights',   'comm_load_weights',          1.0 / MB, True),
        ('comm_total',     'comm_load',                  1.0 / MB, True),
        ('latency_s',      'latency_s',                       1.0, False),
        ('client_mem',     'peak_client_mem_mb',              1.0, False),
        ('server_mem',     'peak_server_mem_mb',              1.0, False),
        ('system_mem',     'peak_system_live_mb',             1.0, False),
        ('held_mem',       'client_mem_held_across_cut_mb',   1.0, False),
        ('process_rss',    'peak_memory_mb',                  1.0, False),
    )
    # How much of `comm_weights` is server-side aggregation: averaging the
    # per-client SERVER replicas of a multi-server method. Those replicas all
    # live on one host, so every reference implementation charges this as ZERO.
    # This harness charges it, because charging some methods and not others for
    # the identical operation would bias the ranked column -- but a reader
    # comparing against a paper's own accounting needs to be able to subtract
    # it, and until now it was invisible. Dominant for SplitFedv1 / Han-et-al /
    # FedSplitX / MU-SplitFed; exactly 0 for every single-server method.
    cell['weights_server'] = _summarise_breakdown(
        runs, ('weights.server_up', 'weights.server_down'), 1.0 / MB
    )
    cell['cut_share_pct'] = (
        100.0 * cut_mean / tot_mean if cut_mean is not None and tot_mean else 0.0
    )
    cell['n_seeds'] = len(runs)
    # Provenance. Which results.json fed a cell used to be invisible, so two
    # methods resolving to the SAME run printed as two corroborating rows. Kept
    # per cell and cross-checked in build_table.
    cell['paths'] = paths
    cell['config_sha256'] = [
        r.get('run_manifest', {}).get('config_sha256') for r in runs
    ]
    cell['settings'] = [
        (
            r.get('run_manifest', {}).get('rounds', len(r.get('test_acc', []))),
            r.get('run_manifest', {}).get('num_clients'),
            r.get('run_manifest', {}).get('cut'),
            r.get('run_manifest', {}).get('device'),
        )
        for r in runs
    ]
    return cell

# ------------------------------------------------------------------------------
def columns_spec(cfg):
    '''(display_name, distribution, alpha) for each table column.'''
    return [
        ('MNIST α=∞ (IID)', 'iid', None),
        (f"MNIST α={cfg['alpha']}", 'noniid_dirichlet', cfg['alpha']),
    ]


# ------------------------------------------------------------------------------
# per-column metrics, in table order. The benchmark ranks on `Comm-total`:
# every method runs the SAME number of rounds and a round means one local epoch
# for all of them (see docs/METRICS.md), so cumulative bytes at the final round
# is a like-for-like comparison. This is checked, not assumed -- build_table
# refuses to emit a table whose cells disagree on round count.
METRIC_COLUMNS = [
    # (column label, cell key, format precision)
    ('Comm-cut (MB)',      'comm_cut',       2),
    ('Comm-weights (MB)',  'comm_weights',   2),
    ('  of which server-side', 'weights_server', 2),
    ('Comm-total (MB)',    'comm_total',     2),
    ('Acc (%)',            'acc',            2),
    ('Latency (s)',        'latency_s',      2),
    ('Client mem (MB)',    'client_mem',     2),
    ('Held-across-cut (MB)', 'held_mem',     3),
    ('Server mem (MB)',    'server_mem',     2),
    # The simultaneity peak: live bytes summed across BOTH sides at one instant.
    # The two per-side columns above are maxed independently, so only this one
    # can show that synchronous SL holds the client's activations while the
    # server runs and a decoupled method does not -- the effect DSL-Aux's paper
    # reports as a peak-memory saving.
    ('System peak (MB)',   'system_mem',     2),
]


def _fmt(cell, key, prec):
    '''mean+/-std across seeds, or "n/a" when the metric is absent from the
    run's results.json (e.g. an older run predating a metric).'''
    mean, std, n = cell.get(key, (None, None, 0))
    if mean is None:
        return 'n/a'
    if n > 1:
        return f"{mean:.{prec}f}+/-{std:.{prec}f}"
    return f"{mean:.{prec}f}"


def build_table(cfg=None, out_path=None, cut='middle', num_clients=None,
                 seeds=None, rounds=None):
    cfg = cfg or load_config()
    columns = columns_spec(cfg)
    methods = list(cfg['methods'].keys())

    cells = {
        name: [
            collect_cell(
                cfg, cfg['methods'][name]['key'], dist, alpha,
                cut=cut, num_clients=num_clients, seeds=seeds
            )
            for _, dist, alpha in columns
        ] for name in methods
    }

    table = PrettyTable()
    col_names = ['Method']
    for name, _, _ in columns:
        col_names += [f'{name} {label}' for label, _, _ in METRIC_COLUMNS]
        col_names.append(f'{name} Cut vs. weights')
    table.field_names = col_names

    footnotes_used = []
    observed_settings = set()
    seed_counts = set()
    cell_sources = {}
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
                row += [_fmt(cell, key, prec) for _, key, prec in METRIC_COLUMNS]
                row.append(f"{cell['cut_share_pct']:.0f}% cut")
                observed_settings.update(cell['settings'])
                seed_counts.add(cell['n_seeds'])
                for pth in cell['paths']:
                    cell_sources.setdefault(pth, []).append(f'{label} ({name})')
            else:
                row += ['N/A'] * (len(METRIC_COLUMNS) + 1)
        table.add_row(row)

    header = f"Cut: {cut}, num_clients: {num_clients if num_clients is not None else 'default'}"
    header += ", matched rounds (1 round = 1 local epoch for every method)"
    header += ", ranked on Comm-total"

    # Comparability guard. Cumulative comm and wall-clock latency are only
    # comparable across cells that ran the same protocol, and memory is not
    # comparable across devices at all (cuDNN saves a different set of
    # intermediates than the CPU kernels). Previously the table just printed
    # whatever it found; now a mismatch is stated on the face of it.
    warnings = []
    # Two cells reading the same results.json is not a near-miss, it is a bug:
    # it prints one run twice and reads as independent corroboration. This is
    # exactly how a DSL-Aux row once came out byte-identical to Vanilla-SL's.
    for pth, users in sorted(cell_sources.items()):
        if len(set(users)) > 1:
            warnings.append(
                "!! DUPLICATE SOURCE: " + ", ".join(sorted(set(users)))
                + f" all read {pth}. These are not independent results."
            )
    distinct = {s for s in observed_settings if any(v is not None for v in s)}
    observed_rounds = {d[0] for d in distinct}
    if len(observed_rounds) > 1:
        warnings.append(
            "!! INVALID TABLE: cells were run for different round counts "
            f"({sorted(observed_rounds)}). The benchmark ranks on Comm-total at "
            "matched rounds, so this table is not a valid comparison -- re-run "
            "every cell at the same --rounds."
        )
    if rounds is not None and observed_rounds and observed_rounds != {rounds}:
        warnings.append(
            f"!! INVALID TABLE: expected every cell at rounds={rounds}, found "
            f"{sorted(observed_rounds)}. A cell short of the requested budget "
            "was truncated (check comm_threshold_mb) or is a stale run."
        )
    if len({d[1] for d in distinct if d[1] is not None}) > 1:
        warnings.append(
            "!! NOT COMPARABLE: cells were run with different client counts "
            f"({sorted({d[1] for d in distinct if d[1] is not None})}). "
            "Comm and weight traffic scale with the number of clients."
        )
    if len({d[2] for d in distinct if d[2] is not None}) > 1:
        warnings.append(
            "!! NOT COMPARABLE: cells were run at different cuts "
            f"({sorted({d[2] for d in distinct if d[2] is not None})})."
        )
    devices = {d[3] for d in distinct if d[3] is not None}
    if len(devices) > 1:
        warnings.append(
            f"!! MEMORY NOT COMPARABLE: cells span devices ({sorted(devices)}). "
            "Retained-activation bytes differ between CPU and cuDNN kernels."
        )
    if seed_counts and max(seed_counts) < 3:
        warnings.append(
            f"NOTE: at most {max(seed_counts)} seed(s) per cell; CLAUDE.md's "
            "protocol is 3. Values shown without +/- are single runs."
        )

    def emit(stream=None):
        print(header, file=stream) if stream else print(header)
        print(table, file=stream) if stream else print(table)
        for line in warnings + footnotes_used:
            print(line, file=stream) if stream else print(line)

    emit()
    out_path = out_path or default_out_path(cut, num_clients)
    with open(out_path, 'w') as f:
        emit(f)

    return table

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    build_table()

# ------------------------------------------------------------------------------

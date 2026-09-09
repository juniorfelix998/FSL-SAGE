# ------------------------------------------------------------------------------
# Shared auto-discovery helper for locating run results under `saves/`, without
# needing to hand-maintain timestamped paths the way `exp_config.yaml` does.
# Used by both `benchmark_table.py` and `run_mnist_benchmark.py` so there is a
# single place to change if/when multi-seed averaging is added later.
# ------------------------------------------------------------------------------
import os
import glob
import json

# ------------------------------------------------------------------------------
def find_latest_run(prefix_dir, algo_key, model, dataset, distribution,
                     alpha=None, cut='middle', num_clients=None, seed=None):
    '''Find the most recently produced `results.json` for a given
    (algorithm, model, dataset, distribution[, alpha], cut[, num_clients])
    combination.

    Mirrors the save path built by `create_save_dir()` in `src/utils/utils.py`:
        <prefix_dir>/<algo_key>/<model>/<cut>/<dataset>-<distribution>/<train_info>/<timestamp>/results.json
    where `train_info` looks like `R{rounds}m{num_clients}E{epoch}B{batch}...`,
    optionally followed by `-alp{alpha:.2e}` when
    `distribution == 'noniid_dirichlet'`. `num_clients=None` matches any
    client count (glob `*`); pass an int to filter to a specific sweep value.
    `cut` defaults to 'middle' (this harness's original, pre-cut-support
    behavior) -- pass 'shallow'/'deep' to look up those cuts' runs instead.
    `seed=None` matches any seed; pass an int to pin one, which is how
    `benchmark_table.collect_cell` gathers a cell's per-seed runs to average.

    Returns the path to the most recent matching `results.json` (by directory
    name, which sorts chronologically since timestamps are `%y%m%d-%H%M%S`), or
    None if no run matches.
    '''
    base = os.path.join(prefix_dir, algo_key, model, cut, f"{dataset}-{distribution}")
    filters = []
    if num_clients is not None:
        filters.append(f"m{num_clients}E")
    if alpha is not None:
        filters.append(f"alp{alpha:.2e}")
    if seed is not None:
        filters.append(f"seed{seed}")
    train_info_glob = "*" + "*".join(filters) + "*" if filters else "*"
    pattern = os.path.join(base, train_info_glob, "**", "results.json")
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[-1] if matches else None

# ------------------------------------------------------------------------------
def load_run(path):
    '''Load a `results.json` file. Returns the dict as-is: `test_acc`,
    `comm_load`, `comm_load_cut`, `comm_load_weights`, etc. are all per-round
    lists (index -1 is the final round).'''
    with open(path, 'r') as f:
        return json.load(f)

# ------------------------------------------------------------------------------

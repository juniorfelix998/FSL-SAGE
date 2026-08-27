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
def find_latest_run(prefix_dir, algo_key, model, dataset, distribution, alpha=None):
    '''Find the most recently produced `results.json` for a given
    (algorithm, model, dataset, distribution[, alpha]) combination.

    Mirrors the save path built by `create_save_dir()` in `src/utils/utils.py`:
        <prefix_dir>/<algo_key>/<model>/<dataset>-<distribution>/<train_info>/<timestamp>/results.json
    where `train_info` includes a `-alp{alpha:.2e}` substring when
    `distribution == 'noniid_dirichlet'`, which is why `alpha` needs its own
    glob term rather than just filtering on `distribution`.

    Returns the path to the most recent matching `results.json` (by directory
    name, which sorts chronologically since timestamps are `%y%m%d-%H%M%S`), or
    None if no run matches.
    '''
    base = os.path.join(prefix_dir, algo_key, model, f"{dataset}-{distribution}")
    train_info_glob = f"*alp{alpha:.2e}*" if alpha is not None else "*"
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

# ------------------------------------------------------------------------------
# Categorised communication accounting for the BP-free split-learning benchmark.
#
# WHY: `comm_load_cut` / `comm_load_weights` used to be bumped by copy-pasted
# `x.numel() * x.element_size()` lines inside each of the 13 method files.
# Nothing enforced that two methods counted the same things -- which is exactly
# the risk when the paper's entire contribution is a fairness claim. This ledger
# is the single place a transfer gets priced, and it additionally records WHICH
# KIND of traffic each byte was, so the headline ranking is inspectable:
#   * CSE-FSL / FSL-SAGE should show cut.grad_down == 0 (no gradient crosses back)
#   * HO-SFL should show full-tensor cut traffic (activation up, gradient down
#     every batch) but near-zero weights.*: its aggregation is dimension-free,
#     P seeds + P scalars per iteration, never a weight vector
#   * the zeroth-order methods' cost should sit almost entirely in cut.act_up
#   * vanilla SL should show weights.client_relay only -- it never aggregates,
#     but it does MOVE one model from client to client, which is not the same
#     thing as having no weight traffic (see src/algos/vanilla_sl.py)
#
# `comm_load_cut`, `comm_load_weights` and `comm_load` remain exactly what they
# were (bytes, cumulative over the run) -- they are now derived sums over the
# categories, so every existing consumer of results.json keeps working.
# ------------------------------------------------------------------------------

# Cut traffic: anything crossing the client/server split during training.
CUT_CATEGORIES = (
    'act_up',        # smashed activations, client -> server
    'grad_down',     # gradient at the cut, server -> client
    'scalar_up',     # scalar probes/losses, client -> server
    'scalar_down',   # scalar loss differences, server -> client (ZO methods)
    'labels_up',     # labels, client -> server (only when loss is server-side)
)

# Weight traffic: model/aggregation transfers, ~0 for non-federated methods.
WEIGHT_CATEGORIES = (
    'client_up', 'client_down',
    'client_relay',  # sequential-SL handover: client i -> client i+1, ONE way
    'aux_up', 'aux_down',
    'server_up', 'server_down',
    # HO-SFL's dimension-free aggregation (paper Alg.1/3): no weight vector,
    # only P seeds + P scalars per iteration, itemised so each can be counted
    'zo_seeds_down',    # P seeds broadcast per iteration (Alg.1 L4)
    'zo_scalars_up',    # P projection scalars per iteration (Alg.1 L23)
    'zo_scalars_down',  # P averaged scalars broadcast per iteration (Alg.1 L26)
    'zo_sync_down',     # catch-up seeds + scalars for missed iterations (Alg.3 L3)
)


# ------------------------------------------------------------------------------
def tensor_bytes(t):
    '''Wire size of a tensor. Uses element_size() so `use_64bit: true` is
    reflected automatically, and numel() so a ragged final batch is priced
    correctly rather than assumed full.'''
    return t.numel() * t.element_size()


# ------------------------------------------------------------------------------
class CommLedger:
    '''Cumulative, per-category byte counters for one algorithm run.'''

    def __init__(self):
        self.cut = {k: 0.0 for k in CUT_CATEGORIES}
        self.weights = {k: 0.0 for k in WEIGHT_CATEGORIES}
        # how many TRANSFERS each category saw, so bytes can be read as
        # count x size-per-transfer and checked against a paper's schedule
        # (e.g. "smashed data every h batches", "aux model once per l rounds")
        self.counts = {f'cut.{k}': 0 for k in CUT_CATEGORIES}
        self.counts.update({f'weights.{k}': 0 for k in WEIGHT_CATEGORIES})

    # -- charging --------------------------------------------------------
    def charge_cut(self, category, nbytes):
        if category not in self.cut:
            raise KeyError(f"unknown cut category {category!r}")
        self.cut[category] += float(nbytes)
        self.counts[f'cut.{category}'] += 1

    def charge_weights(self, category, nbytes):
        if category not in self.weights:
            raise KeyError(f"unknown weight category {category!r}")
        self.weights[category] += float(nbytes)
        self.counts[f'weights.{category}'] += 1

    # -- totals ----------------------------------------------------------
    @property
    def cut_total(self):
        return sum(self.cut.values())

    @property
    def weights_total(self):
        return sum(self.weights.values())

    @property
    def total(self):
        return self.cut_total + self.weights_total

    # -- reporting -------------------------------------------------------
    def snapshot(self):
        '''Flat {'cut.act_up': bytes, 'weights.client_up': bytes, ...} for the
        per-round record in results.json.'''
        out = {f'cut.{k}': v for k, v in self.cut.items()}
        out.update({f'weights.{k}': v for k, v in self.weights.items()})
        # event counts under a separate `count.` prefix, so any consumer that
        # sums the `cut.*` / `weights.*` byte keys is unaffected
        out.update({f'count.{k}': v for k, v in self.counts.items()})
        return out

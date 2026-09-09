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
#   * HO-SFL should show near-zero weights.* against a nonzero cut.grad_down
#     (dimension-free aggregation: P scalars + P seeds, never a weight vector)
#   * the zeroth-order methods' cost should sit almost entirely in cut.act_up
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
    'aux_up', 'aux_down',
    'server_up', 'server_down',
    'scalars',       # HO-SFL's dimension-free aggregation (scalars + seeds)
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

    # -- charging --------------------------------------------------------
    def charge_cut(self, category, nbytes):
        if category not in self.cut:
            raise KeyError(f"unknown cut category {category!r}")
        self.cut[category] += float(nbytes)

    def charge_weights(self, category, nbytes):
        if category not in self.weights:
            raise KeyError(f"unknown weight category {category!r}")
        self.weights[category] += float(nbytes)

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
        return out

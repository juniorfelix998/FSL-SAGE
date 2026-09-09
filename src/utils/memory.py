# ------------------------------------------------------------------------------
# Per-side (client vs. server) memory accounting for the BP-free split-learning
# benchmark.
#
# WHY THIS EXISTS: `main.py`'s `get_peak_memory_mb()` reports whole-process RSS,
# which is dominated by the interpreter, torch, the dataset held in RAM, and
# *every* simulated client's model being resident at once. That number is
# method-blind (all methods land within a few percent of each other) and it even
# points the wrong way -- CSE-FSL measured *higher* than SplitFedv2 on identical
# settings. It cannot express CLAUDE.md's metric 5 ("peak client + server
# memory"), which is a statement about what ONE device needs.
#
# WHAT WE REPORT INSTEAD, per side:
#     peak = static + working
#   static  = params + buffers + grads + optimizer state  (computed exactly)
#   working = high-water mark of retained activation / held bytes  (measured)
#
# The static half is exactly computable for a single device and is where a
# whole-process measurement is hopeless. The working half cannot be derived
# analytically (it depends on which tensors autograd chose to save, on
# `inplace=True` fusions, and on storage sharing) and so must be measured.
#
# TWO DISTINCT METRICS FALL OUT OF THIS, and they are NOT the same axis:
#   * peak activation bytes  -- separates zeroth-order clients (which run the
#     client forward under `torch.no_grad()` and retain nothing) from every
#     method that backprops on the client.
#   * bytes held across the cut -- client-owned autograd bytes still live at the
#     instant the server phase runs. Separates synchronous split learning (the
#     client stalls holding its graph waiting for the server's gradient) from
#     decoupled methods (CSE-FSL, FSL-SAGE, ...) that backward locally first.
#   Note the aux-model methods legitimately score HIGHER on the first metric --
#   they hold the client graph AND an auxiliary head's activations at once. They
#   buy communication savings by spending client memory. Reporting only one
#   number would hide that trade.
# ------------------------------------------------------------------------------
import logging
import weakref
from contextlib import contextmanager

import torch
import torch.nn as nn


# ------------------------------------------------------------------------------
def module_static_bytes(*modules):
    """(param_and_buffer_bytes, grad_bytes) for the given modules.

    Buffers count: BatchNorm running stats are real device memory. Gradients
    count too, and must be *measured* rather than assumed zero for zeroth-order
    methods -- `_perturb_accumulate_grad` in ho_sfl/hosl/mu_splitfed allocates a
    real `.grad` for every perturbed parameter.
    """
    param_size = 0
    grad_size = 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            param_size += p.nelement() * p.element_size()
            if p.grad is not None:
                grad_size += p.grad.nelement() * p.grad.element_size()
        for b in m.buffers():
            param_size += b.nelement() * b.element_size()
    return param_size, grad_size


# ------------------------------------------------------------------------------
def optimizer_state_bytes(*optimizers):
    """Bytes held in optimizer state (Adam/AdamW -> ~2x params, SGD with
    momentum -> ~1x, plain SGD -> 0).

    This is a large, method-varying term that a parameter count alone misses:
    ho_sfl forces AdamW while mu_splitfed forces plain SGD, so their true
    client-side footprints differ by ~2x the client model even before any
    activation is allocated. Deduplicated by storage so shared state is counted
    once.
    """
    seen = set()
    total = 0
    for opt in optimizers:
        if opt is None:
            continue
        for state in opt.state.values():
            for v in state.values():
                if torch.is_tensor(v):
                    key = v.untyped_storage().data_ptr()
                    if key in seen:
                        continue
                    seen.add(key)
                    total += v.untyped_storage().nbytes()
    return total


# ------------------------------------------------------------------------------
class _SavedSlot:
    """Finalization sentinel for one autograd saved-tensor slot.

    Deliberately holds no reference to the tensor -- only the storage key and a
    weak reference to the meter -- so pairing it with the tensor cannot create
    an uncollectable reference cycle through the autograd graph.
    """

    __slots__ = ('key', 'meter')

    def __init__(self, key, meter):
        self.key = key
        self.meter = weakref.ref(meter)

    def __del__(self):
        try:
            meter = self.meter()
            if meter is not None:
                meter._release_saved(self.key)
        except Exception:                   # interpreter shutdown
            pass


# ------------------------------------------------------------------------------
class MemoryMeter:
    """High-water mark of live memory, attributed to whichever phase
    ('client', i) / ('server',) allocated it.

    Mechanism: `torch.autograd.graph.saved_tensors_hooks`. This sees exactly the
    tensors autograd retains for backward -- which is the quantity we want, and
    which is zero under `torch.no_grad()`. Two alternatives were evaluated and
    rejected:
      * `torch.cuda.max_memory_allocated` brackets -- CUDA-only (every run so far
        has been CPU), and cross-phase *retention* shows up as an inflated
        baseline in the next phase rather than as attributable bytes. Kept only
        as a GPU cross-check.
      * forward hooks summing module outputs -- measures outputs, not what
        autograd saved. Wrong for `inplace=True` ReLU (which ResNetClient uses)
        and for BatchNorm (which saves save_mean/save_invstd, not its output),
        and identical with or without `no_grad`, which destroys the exact
        discrimination the zeroth-order methods need.

    Two kinds of bytes are tracked separately, because they answer different
    questions and conflating them makes the metric names lie:
      SAVED -- retained by autograd for backward. Zero under `no_grad`, so this
               is what discriminates a zeroth-order client.
      HELD  -- explicitly declared long-lived tensors (the input batch, a
               downloaded cut-gradient, a round buffer). Real memory, but
               resident whether or not the client backprops.
    `peak` is over their sum (what the device must provide); the
    held-across-the-cut and activation-peak metrics read SAVED alone.

    Implementation notes that are load-bearing:
      * Keyed on `untyped_storage().data_ptr()` so views and shared storages are
        counted once (e.g. a ReLU output shared with the next conv).
      * Release of a SAVED tensor is detected EXACTLY, via a sentinel stored
        alongside the tensor in the saved-tensor slot. When the graph is freed
        (normally at the end of backward) the slot is dropped, the sentinel is
        finalized, and the bytes are released.

        Tracking a SAVED tensor with a plain `weakref` instead is wrong: a
        tensor whose graph has already been backwarded can still be referenced
        by an ordinary local variable in the caller, and would then keep
        counting as "retained by autograd" when it no longer is. That produced a
        spurious non-zero held-across-the-cut figure for CSE-FSL from an 80-byte
        stale local -- understating exactly the decoupling this benchmark exists
        to demonstrate.

        The sentinel holds NO reference to the tensor, so it cannot form the
        uncollectable cycle a tensor-holding wrapper would.
      * Parameters and parameter *views* (nn.Linear saves `weight.t()`, a
        non-leaf sharing the parameter's storage) are excluded, because they are
        already counted exactly in the static term.
    """

    SAVED = 'saved'
    HELD = 'held'

    def __init__(self, enabled=True):
        self.enabled = enabled
        # storage_ptr -> [nbytes, owner, kind, weakref_or_None, refcount]
        self._entries = {}
        self._cur = {}          # (owner, kind) -> live bytes
        self.peak = {}          # owner -> high-water bytes (saved + held)
        self.peak_saved = {}    # owner -> high-water autograd-retained bytes
        self._stack = []        # phase-owner stack
        self._param_ptrs = set()

    # -- ownership -------------------------------------------------------
    @property
    def owner(self):
        return self._stack[-1] if self._stack else None

    def register_params(self, *modules):
        """Storages never to count as activations. Call once per round: FedAvg
        rebuilds `aggregated_client` every round, and several algorithms
        deep-copy or rebind models. In-place `optimizer.step()` and
        `load_state_dict` keep data_ptrs stable, so the set is valid for a round.
        """
        ptrs = set()
        for m in modules:
            if m is None:
                continue
            for t in list(m.parameters()) + list(m.buffers()):
                ptrs.add(t.untyped_storage().data_ptr())
        self._param_ptrs = ptrs

    # -- bookkeeping -----------------------------------------------------
    def _sweep(self):
        """Drop HELD entries whose tensor has been garbage collected. SAVED
        entries are released exactly, by sentinel, and are never swept here."""
        dead = [
            k for k, e in self._entries.items()
            if e[3] is not None and e[3]() is None
        ]
        for k in dead:
            self._drop(k)

    def _drop(self, key):
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        nbytes, owner, kind = entry[0], entry[1], entry[2]
        self._cur[(owner, kind)] = self._cur.get((owner, kind), 0) - nbytes

    def _release_saved(self, key):
        """Called from a sentinel's finalizer when a saved-tensor slot is
        dropped, i.e. the autograd graph holding it has been freed."""
        entry = self._entries.get(key)
        if entry is None or entry[2] != self.SAVED:
            return
        entry[4] -= 1                       # one fewer slot references it
        if entry[4] <= 0:
            self._drop(key)

    def _account(self, t, owner, kind):
        if owner is None or t is None or not torch.is_tensor(t):
            return None
        storage = t.untyped_storage()
        key = storage.data_ptr()
        if key in self._param_ptrs:
            return None
        self._sweep()

        existing = self._entries.get(key)
        if existing is not None:            # shared storage, already counted
            if kind == self.SAVED and existing[2] == self.SAVED:
                existing[4] += 1
                return key
            return None

        nbytes = storage.nbytes()
        self._entries[key] = [
            nbytes, owner, kind,
            weakref.ref(t) if kind == self.HELD else None,
            1,
        ]
        self._cur[(owner, kind)] = self._cur.get((owner, kind), 0) + nbytes

        saved = self._cur.get((owner, self.SAVED), 0)
        total = saved + self._cur.get((owner, self.HELD), 0)
        if total > self.peak.get(owner, 0):
            self.peak[owner] = total
        if saved > self.peak_saved.get(owner, 0):
            self.peak_saved[owner] = saved
        return key

    # -- autograd hooks --------------------------------------------------
    def _pack(self, t):
        if isinstance(t, nn.Parameter) or (t.is_leaf and t.requires_grad):
            return t
        key = self._account(t, self.owner, self.SAVED)
        if key is None:
            return t
        # Pair the tensor with a sentinel whose finalizer fires when autograd
        # drops this slot -- the precise moment the bytes stop being retained.
        return (t, _SavedSlot(key, self))

    @staticmethod
    def _unpack(p):
        return p[0] if isinstance(p, tuple) else p

    # -- public API ------------------------------------------------------
    def hold(self, owner, *tensors):
        """Declare long-lived, non-autograd tensors resident on `owner` -- the
        downloaded cut-gradient, HO-SFL's per-round buffer, the input batch.
        These are real device memory the saved-tensor hooks never see.
        """
        if not self.enabled:
            return
        for t in tensors:
            self._account(t, owner, self.HELD)

    @contextmanager
    def phase(self, owner):
        if not self.enabled:
            yield
            return
        self._stack.append(owner)
        try:
            with torch.autograd.graph.saved_tensors_hooks(self._pack, self._unpack):
                yield
        finally:
            self._stack.pop()

    def live_bytes(self, owner, saved_only=False):
        """Live bytes for `owner`. `saved_only=True` counts just the
        autograd-retained ones -- the right reading for "what is this client
        forced to keep alive because it is waiting on the server", since the
        input batch and downloaded buffers are resident either way.
        """
        self._sweep()
        total = self._cur.get((owner, self.SAVED), 0)
        if not saved_only:
            total += self._cur.get((owner, self.HELD), 0)
        return max(0, total)

    def reset_live(self):
        """Drop all liveness tracking (but keep the peaks).

        Called between instrumented steps. Necessary because
        `torch.autograd.grad(..., create_graph=True)` -- FSL-SAGE's surrogate
        gradient emitter -- legitimately leaves saved tensors alive past the hook
        scope, so without a reset the high-water mark would drift upward across
        iterations instead of measuring a single step.
        """
        self._entries.clear()
        self._cur.clear()

    def end_step(self, warn=True):
        """Report autograd bytes still live after a step. Non-zero means a graph
        was built and never backwarded. No current method does this; this catches
        a future one rather than letting it silently inflate the peak.
        """
        self._sweep()
        residual = {
            k[0]: b for k, b in self._cur.items()
            if b > 0 and k[1] == self.SAVED
        }
        if warn and residual:
            logging.debug(f"[mem] residual live autograd bytes after step: {residual}")
        self.reset_live()
        return residual

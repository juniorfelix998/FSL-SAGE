# How every benchmark number is measured

**"What Crosses the Cut?" — BP-free split-learning benchmark**

This document is the measurement contract. The benchmark's whole contribution is
a *fair common measurement layer*, so "how is this measured?" is not a footnote —
it is the claim. Every metric below is given in the same shape:

> **what we claim to measure → the code that does it → what it includes and
> excludes → how to verify it on a real run**

Every citation is `file:line` against this repo, so anything here can be checked
against the source rather than taken on trust.

---

## 0. The unit of work: what a "round" is

**One round = one local epoch over each client's shard, for every method.**

The shared training loop is the only loop; no method has its own
(`src/algos/__init__.py:538-570`):

```
for t in range(cfg.rounds)            # round        <- src/algos/__init__.py:540
    for i in range(cfg.num_clients)   # client       <- :555
        for j in range(clients[i].epochs)   # local epoch  <- :556
            for k, (x, y) in enumerate(train_loader)  # batch  <- :566
                alg.client_step(...)
    alg.aggregate()
    alg.evaluate()
```

On MNIST with 10 clients, batch 256, 6000 samples/client, that is **24 optimizer
steps per client per round** — the same for all twelve methods.

> **Why this matters, and what changed.** `ho_sfl` and `mu_splitfed` used to
> no-op on every batch except the round's first, faithful to their reference
> implementations, whose "round" *is* one batch. Under an equal round budget that
> handed them **1/24 of everyone else's updates**, so the sweep compensated with
> a 24× round multiplier for those two methods alone — which in turn made
> Comm-total and Latency incomparable across rows (a method run 24× longer pays
> 24× the per-round overhead; MU-SplitFed's 65.8 GB in the previous report was
> that artifact, not a property of the method). The multiplier is gone and those
> two methods now step on every batch like the rest. Their zeroth-order update
> rule is untouched — only how many batches it is applied to.
>
> Enforced, not assumed: `test/check_accounting.py`'s
> `check_every_batch_trains` inspects every registered algorithm's `client_step`
> source and fails if any of them reintroduces a batch gate.

**There is no `Comm-to-target` metric.** Because rounds are matched and a round
means the same thing everywhere, cumulative **Comm-total at the final round** is
already a like-for-like comparison, and that is what the benchmark ranks on. The
table refuses to emit if its cells disagree on round count
(`inference/benchmark_table.py`, comparability guard).

---

## 1. Communication

### What we claim to measure
Bytes that would cross a network link during training, split into **cut traffic**
(activations/gradients/scalars across the client↔server split) and **weight
traffic** (model/aggregation transfers). The ranked number is their sum.

### The code that does it
One ledger prices every transfer — `CommLedger`, `src/utils/comm.py:47-84`, one
instance per run (`src/algos/__init__.py:100`). Methods never compute bytes
themselves; they call typed helpers on the base class
(`src/algos/__init__.py:119-151`), so two methods cannot silently disagree about
what a transfer costs.

Wire size is analytic (`src/utils/comm.py:39-44`):

```python
def tensor_bytes(t):
    return t.numel() * t.element_size()
```

`numel()` so a ragged final batch is priced correctly rather than assumed full;
`element_size()` so `use_64bit: true` is reflected automatically.

**Categories** (`src/utils/comm.py:21-37`):

| Cut | meaning |
|---|---|
| `act_up` | smashed activations, client → server |
| `grad_down` | gradient at the cut, server → client |
| `scalar_up` | scalar probes/losses, client → server |
| `scalar_down` | scalar loss differences, server → client (zeroth-order) |
| `labels_up` | labels, client → server (only when the loss is computed server-side) |

| Weights | meaning |
|---|---|
| `client_up/down`, `aux_up/down`, `server_up/down` | full model transfers |
| `scalars` | HO-SFL's dimension-free aggregation (scalars + seeds, never a weight vector) |

**Granularity.** Cut traffic is charged **per batch, per client**, inside
`client_step`. Weight traffic is charged **once per round** inside `aggregate()`,
inside a per-client loop, so it sums over clients. Counters are cumulative and
never reset; the per-round snapshot in `results.json` is cumulative-to-date
(`src/utils/comm.py:79-84`). Model weights include **buffers** — BatchNorm
running stats are real bytes on the wire (`src/utils/utils.py:22-32`). Labels are
int64, so 8 B/sample.

### Includes / excludes
- **Includes**: every tensor and scalar that would have to be transmitted, at
  full precision, including BN buffers.
- **Excludes**: serialization overhead, protocol headers, compression,
  quantization, retransmission. These are analytic byte counts, not packet
  captures — a deliberate choice, because it makes the comparison
  implementation-independent, and because any real transport would scale all
  methods by roughly the same factor.

### Two conventions worth stating plainly
1. **Labels are charged to the cut** whenever the loss is computed server-side
   (`src/algos/__init__.py:125-130`), and deliberately *not* charged for
   client-local-loss branches. That asymmetry is part of what a local-loss method
   buys, so hiding it would understate the methods being benchmarked.
2. **Server-side aggregation is charged a full model round-trip** in every
   multi-server method — SplitFedv1, FedSplitX, Han-et-al, MU-SplitFed
   (`src/algos/baselines.py:187-193`). The reference implementations charge this
   as *zero*, on the grounds that per-client server replicas live on one host, so
   averaging them is a memory copy rather than network traffic. This harness
   charges it because the alternative — charging some methods and not others for
   the identical operation — biases the ranked column. **The consequence is real
   and should be read with the table**: it inflates Comm-weights for the four
   multi-server methods relative to their papers' own accounting.

### How to verify
`python test/check_accounting.py` asserts **closed-form** byte counts per method
against the real shared loop, plus the per-method signatures below. For a real
run, read `comm_breakdown[-1]` out of `results.json`.

### Expected accounting signature per method
A row that violates its signature is a bug, not a finding.

| Method | `cut.grad_down` | `weights` | note |
|---|---|---|---|
| SplitFedv1 / SplitFedv2 | > 0 | > 0 | full BP across the cut, FedAvg |
| Vanilla-SL | > 0 | **0** | non-federated; one shared model pair |
| CSE-FSL | **0** | > 0 | local aux head; uploads only every `q`-th batch |
| FSL-SAGE | **0** | > 0 | surrogate synthesises the cut gradient; aux charged download-only |
| DSL-Aux | **0** | **0** | decoupled *and* non-federated (see §6) |
| Han-et-al (LGL-SL) | **0** | > 0 | local losses both sides |
| FedSplitX | **0** | > 0 | local collaborative loss both sides |
| HOSL | **0** | **0** | zeroth-order, non-federated; `scalar_down` only |
| HO-SFL | > 0 | > 0, `scalars` only | client needs the cut gradient for its ZO probe; aggregation is scalars + seeds |
| MU-SplitFed | **0** | > 0 | 3× `act_up` (fixed + two probes), 1 `scalar_down` |
| LocFedMix-SL | > 0 | > 0 | server-side Mixup, InfoPro decoder |

HO-SFL and MU-SplitFed were cross-checked line-for-line against the authors'
own accounting in `HKU-WILL-Lab/HO-SFL`'s `src/core/communicator.py`; both match
category-for-category (HO-SFL's reference really does log an activation-gradient
downlink, `ho_sfl_runner.py:157`).

---

## 2. Memory

### What we claim to measure
**Peak memory one device must provide** — one client, and the server host —
not what the whole simulated population sums to.

### Why the obvious approach was rejected
`main.py`'s whole-process RSS is dominated by the interpreter, torch, the dataset
in RAM, and *every* simulated client's model being resident at once. It is
method-blind (all methods land within a few percent) and it pointed the *wrong
way*: CSE-FSL measured higher than SplitFedv2 on identical settings
(`src/utils/memory.py:5-11`). `torch.cuda.max_memory_allocated()` has the same
problem in this harness — it is process-global, so with every client simulated in
one process it cannot attribute bytes to a device, and without an explicit
`reset_peak_memory_stats` between methods it is monotonic across a whole session.

### The code that does it
Per side, **`peak = static + working`**:

- **static, computed exactly** — params + buffers + `.grad`
  (`src/utils/memory.py:45-64`) and optimizer state, deduplicated by storage
  pointer so Adam ≈ 2× params while plain SGD is 0
  (`src/utils/memory.py:68-91`). Gradients are *measured*, not assumed zero for
  zeroth-order methods, because their perturbation step really does allocate a
  `.grad` for every parameter.
- **working, measured** — `MemoryMeter` (`src/utils/memory.py:119-337`) installs
  `torch.autograd.graph.saved_tensors_hooks` inside a phase bracket and tracks a
  high-water mark of autograd-**saved** bytes plus explicitly **held** bytes (the
  input batch, a downloaded cut-gradient). Keyed on storage pointer, so views
  are not double-counted; released exactly via a finalizer. Parameters are
  excluded so they are not counted twice.

Which modules belong to which side is declared per method and overridable
(`src/algos/__init__.py:208-218`) — e.g. CSE-FSL puts its aux head on the client,
SplitFedv1 puts per-client server replicas on the server.

**Reduction is `max` at every level** — across batches, epochs, rounds, and
finally across clients (`src/algos/__init__.py:230-269`). Never a mean: the
metric is what a *single* device must provide.

**Sampling cadence**: instrumentation is on only for the first
`mem_probe_batches` (default 2) batches of each round's first local epoch
(`src/algos/__init__.py:187-201`). The hooks cost ~4% when always on, and every
method's heaviest step is at or near the round's first iteration.

### Two axes that must not be conflated
- `client_act_peak_mem_mb` — peak retained activation bytes. Separates
  zeroth-order clients (forward under `no_grad`, retain ~0) from every method
  that backprops on the client.
- `client_mem_held_across_cut_mb` — client-owned autograd bytes still live *at
  the instant the server runs*. Separates synchronous split learning (the client
  stalls holding its graph, waiting for a gradient) from decoupled methods that
  backward locally first. **Exactly 0 is the decoupling signature.**

An auxiliary-head method can legitimately score *higher* on the first axis — it
holds the client graph and an aux head's activations at once. It buys
communication with client memory. Reporting one number would hide that trade.

### Includes / excludes
- **Includes**: params, buffers, grads, optimizer state, retained activations,
  explicitly held tensors.
- **Excludes**: interpreter and framework overhead, the dataset, allocator
  fragmentation, cuDNN workspace. `peak_process_mem_mb` (RSS,
  `src/main.py:36-42`) and `peak_cuda_memory_mb` are recorded as **diagnostics
  only** and must not be used as the benchmark's memory metric.

### How to verify
`test/check_accounting.py` section [1] asserts exact byte counts on known shapes,
zero under `no_grad`, release after backward, and no double-counting of views.

---

## 3. Latency

### What we claim to measure
Wall-clock time for the whole application run.

### The code that does it
`src/main.py:33` stamps `_APP_START_TIME` at import; `src/main.py:235` takes
`time.time() - _APP_START_TIME` just before writing `results.json`.

### Includes / excludes
- **Includes**: dataset construction, model setup, all training, **all per-round
  evaluation passes**, and wandb init/finish.
- **Excludes**: the final `torch.save` of models, which happens after the stamp.

State this plainly when presenting: it is an **application-level** number, not
pure training time. For a compute-only breakdown use
`client_model_compute_time` / `server_model_compute_time`, accumulated inside the
same `phase()` bracket that measures memory (`src/algos/__init__.py:160-180`),
and the `*_agg_compute_time` entries from each `aggregate()`.

Because it is wall-clock on shared hardware it is the **noisiest** column;
treat differences of a few percent as nothing.

---

## 4. Accuracy

Full pass over the test set under `no_grad`, through the method's own
`full_model(x)` (the concatenated client+server path), once per round after
aggregation (`src/algos/__init__.py:318-337`, called at `:624-625`).
`acc = correct / len(test_loader.dataset)`.

Two methods define `full_model` non-trivially and this is deliberate:
- **Vanilla-SL** evaluates the single shared client model — there is only one.
- **FedSplitX** evaluates the **ensemble of all auxiliary heads plus the final
  output** (paper §3), which is how that method is specified to infer.

---

## 5. Provenance and reproducibility

Every run writes `run_manifest` into `results.json`
(`src/utils/utils.py:83-129`): git commit + dirty flag, algorithm, model,
dataset, distribution, alpha, cut, num_clients, rounds, seed, batch size, local
epochs, dtype, device, and a `config_sha256` of the fully-resolved config. Also
written per run: `settings.yml` (full config dump), `output.log`, `metrics.pt`.

Save path layout (`src/utils/utils.py:146-181`):

```
saves/<algo>/<model>/<cut>/<dataset>-<distribution>/R{rounds}m{clients}E{ep}B{batch}[...]-seed{seed}/<timestamp>/results.json
```

**Table provenance.** `inference/benchmark_table.py` records the resolved
`results.json` path and `config_sha256` behind every cell, and emits a
`DUPLICATE SOURCE` warning if two rows resolved to the same file. One run printed
twice reads as two corroborating results — which is exactly how a DSL-Aux row
once came out byte-identical to Vanilla-SL's.

---

## 6. Worked example: DSL-Aux, and the bug this section exists to prevent

DSL-Aux (Zihad, Owino, Tang & Huang, *Decoupled Split Learning via Auxiliary
Loss*, arXiv:2601.19261) is the clearest case of measurement following from the
algorithm, and it was previously implemented backwards.

### What the paper specifies
- **Alg. 1 / §III-B**: the server computes ∂L/∂z internally, but "**this gradient
  is not transmitted to the client**. It is only used for the server's weight
  updates."
- **§III-C**: λ = 0, "to avoid any BP signals from the server to the client".
- **§IV-A**: the auxiliary classifier is "a single fully-connected layer that
  maps the activations z to 10/100 classes, plus a softmax layer".
- **Obs. 2**: communication falls ~50%, because the backward half disappears.
- **Obs. 3**: peak client memory falls up to 58%, because no activations are
  retained for a cross-cut backward pass.
- **Obs. 4**: per-epoch runtime is *moderately higher* — the aux head is extra
  client compute.

### The corrected implementation (`src/algos/dsl_aux.py`)

```python
with self.phase('client', i):                    # client phase OPENS and CLOSES
    self.clients[i].optimizer.zero_grad()        # before the server runs, so no
    self.clients[i].auxiliary_model.optimizer.zero_grad()   # graph is held
    z = self.clients[i].model(x)                 # across the cut  -> Obs. 3
    aux_out = self.clients[i].auxiliary_model.forward_inner(z.flatten(1))
    aux_loss = self.criterion(aux_out, y)        # single FC + softmax  -> SIV-A
    aux_loss.backward()                          # the client's ONLY signal
    self.clients[i].optimizer.step()
    self.clients[i].auxiliary_model.optimizer.step()

smashed_data = z.detach()
self.charge_cut_activation(smashed_data)         # -> cut.act_up
self.charge_cut_labels(y)                        # -> cut.labels_up (loss is server-side)
#  NO charge_cut_gradient  -- SIII-B: the gradient is never transmitted  -> Obs. 2

with self.phase('server', i):
    self.server.optimizer.zero_grad()
    out = self.server.model(smashed_data)
    loss = self.criterion(out, y)
    loss.backward()                              # dL/dz computed, never sent
    self.server.optimizer.step()
```

Each measurement call is a direct consequence of a line in the paper:

| paper | code | metric effect |
|---|---|---|
| §III-B "not transmitted" | no `charge_cut_gradient` | `cut.grad_down == 0` |
| Obs. 2 (~50% less comm) | only `act_up` + `labels_up` charged | Comm-cut ≈ half Vanilla-SL's |
| Obs. 3 (memory) | client phase closes before server phase | `client_mem_held_across_cut_mb == 0` |
| §IV-A (single FC) | `LinearGradScalarAuxiliaryModel` | small `client_param_mem_mb` |
| non-federated | `aggregate()` averages nothing | `comm_load_weights == 0` |
| Obs. 4 (slower) | extra aux forward/backward | Latency *above* Vanilla-SL |

### How it was wrong, and why the table showed it

The previous implementation charged `charge_cut_gradient(grad_at_cut)` and called
`splitting_output.backward(grad_at_cut)` — i.e. conventional split learning with
an extra auxiliary term. Its own header recorded the decision: it had adapted the
third-party reference's `train_standard_split` and explicitly *rejected*
`train_dgl`, the function that carries the comment "DGL has NO backward comm from
Server to Client". That is precisely backwards.

Because it charged the identical tensors on the identical schedule as
`vanilla_sl.py`, its communication was **identical by construction** — which is
exactly what the previous report showed (2813.87 MB for both, 100% cut for both).

The two references, side by side, are the whole distinction:

```python
# juniorfelix998/sl-fl-dgl -- train_standard_split (conventional SL)
tracker.log_comm(client_out, "fwd")
tracker.log_comm(grad_at_cut, "bwd")      # <- both directions

# juniorfelix998/sl-fl-dgl -- train_dgl (decoupled: what DSL-Aux is)
tracker.log_comm(features, "fwd")
# DGL has NO backward comm from Server to Client
```

### Measured result after the fix

ResNet-18 / MNIST / middle cut / IID / 2 clients / 1 round / seed 200, CPU —
`dsl_aux` and `vanilla_sl` on an otherwise identical config:

| metric | Vanilla-SL | DSL-Aux | paper's claim | verdict |
|---|---|---|---|---|
| Comm-cut | 0.916 GiB | **0.458 GiB** | ~50% less (Obs. 2) | exactly 50.0% |
| Comm-weights | 0.000 | 0.000 | non-federated | both 0 |
| Held-across-cut | 94.01 MiB | **0.00 MiB** | decoupled (SIII-B) | holds nothing |
| Peak client mem | 109.44 MiB | 107.76 MiB | lower (Obs. 3) | lower |
| Latency | 227.23 s | **232.62 s** | *higher* (Obs. 4) | 2.4% slower |
| Test acc | 97.70% | 97.11% | on par (Obs. 1) | comparable |

Before the fix these two rows were identical on every column.

> **One caveat to state when presenting.** The paper reports up to a **58%**
> reduction in peak memory; this harness shows only a small reduction in
> `peak_client_mem_mb`. That is a difference in *metric definition*, not a
> failure to reproduce. The paper measures whole-process peak GPU memory, where
> conventional SL must hold the client's activations **while the server is also
> running** -- the saving comes from that simultaneity. This harness measures
> per-device peak separately for client and server (S2), so the simultaneity
> effect does not appear in `peak_client_mem_mb` at all; it appears in
> `client_mem_held_across_cut_mb`, which goes **94.01 MiB -> 0**. That column is
> the like-for-like evidence for the paper's memory claim here. DSL-Aux still
> builds its own client graph for its local backward, so its *instantaneous*
> client peak is necessarily similar to SL's.

### Locked in by tests
`test/check_accounting.py` now asserts, against a Vanilla-SL run on an
*identical* config:
- `DSL-Aux cut.grad_down is ZERO`
- `DSL-Aux cut total == Vanilla-SL's minus the gradient half`
- `DSL-Aux cut traffic is ~50% of Vanilla-SL's (paper Obs. 2)`
- `DSL-Aux charges no weight traffic (non-federated)`

A regression to plain SL cannot pass silently.

---

## 7. Paper fidelity

| Method | Source | Code? | Status |
|---|---|---|---|
| SplitFedv1 / v2 | Thapa et al., AAAI 2022 | — | In-harness baselines |
| Vanilla-SL | Gupta & Raskar 2018 | — | Sequential/relay SL, one shared model pair |
| CSE-FSL | Mu & Shen, TMC 2025 | ✔ | Authors' method, in-harness |
| FSL-SAGE | Nair et al., ICML 2025 | ✔ | This repo's own method |
| HO-SFL | Chen et al., ICML 2026 | ✔ | Accounting verified against authors' `communicator.py` |
| MU-SplitFed | Liang et al., NeurIPS 2025 | ✔ (3rd-party) | Verified against `HKU-WILL-Lab/HO-SFL`'s CV reimplementation, **not** the original authors' code |
| **DSL-Aux** | arXiv:2601.19261 | — | Reimpl. **Corrected** to Alg. 1 (see §6) |
| **FedSplitX** | arXiv:2310.14579 | — | Reimpl. **Rewritten** — was a byte-identical copy of Han-et-al (see below) |
| Han-et-al (LGL-SL) | Han et al., FL-ICML 2021 | — | Reimpl., unvalidated |
| HOSL | arXiv:2601.10940 | — | Reimpl., unvalidated |
| LocFedMix-SL | Oh et al., WWW 2022 | — | Reimpl., unvalidated |

### The FedSplitX duplication

`fedsplitx.py` was a byte-for-byte copy of `han_locloss.py` with the class name
changed, producing two identical benchmark rows (4450.87 MB / 2967.24 MB /
98.73% / 76.02% in the previous report). Its header defended this as an honest
degeneration at a single shared cut — and at M=1 that is arithmetically true, but
M=1 is not FedSplitX.

FedSplitX's defining mechanism (arXiv:2310.14579 §2.1-2.2, Fig. 1) is an
auxiliary network at **every** partition point and a **collaborative loss**
summing the loss over all intermediate logits each side owns:

```
client:  F_c = sum_{i=1..d_k}   l( a_[i](f_c[:i](x)), y )
server:  F_s = sum_{i=d_k+1..M} l( a_[i](f_s[:i](z)), y )  +  l( f(z), y )
```

Partition points are a property of the *architecture*, not the client
population, so this is fully implementable at one shared cut. On ResNet-18 there
are **M = 3** of them (after `layer1`/`layer2`/`layer3`; `layer4`'s output is
l_{M+1}), and the harness's `cut` selects the depth-level all clients share:

| cut | client holds | server holds |
|---|---|---|
| shallow | a₁ | a₂, a₃ |
| middle | a₁, a₂ | a₃ |
| deep | a₁, a₂, a₃ | — |

Inference uses the **ensemble of all heads** (§3). What remains deliberately not
exercised is client *heterogeneity*: with one shared cut every client sits at the
same depth-level, so `heteroavg` (§2.3) degenerates to plain FedAvg. That is a
disclosed limitation of running FedSplitX under this benchmark's protocol, not a
shortcut.

### Measured result after the rewrite

Same config as above -- the two rows are now distinct on every column:

| metric | Han-et-al | FedSplitX |
|---|---|---|
| aux networks | 1 (at the cut) | 2 client + 1 server (M = 3) |
| Comm-cut | 0.458 GiB | 0.458 GiB |
| Comm-weights | 0.198 GiB | 0.167 GiB |
| Comm-total | 0.656 GiB | 0.625 GiB |
| Peak client mem | 148.80 MiB | 107.68 MiB |
| Latency | 271.64 s | 231.17 s |
| Test acc | 98.12% | 96.40% |

(FedSplitX's lower client memory is its lightweight per-stage linear heads
against Han-et-al's full mirrored-ResNet auxiliary; both correctly report
`Held-across-cut = 0`, as neither sends a gradient back.)

### The guard that was missing
Every other assertion in the test suite is per-method: it can confirm
`fedsplitx` charges the bytes `fedsplitx` should charge and still not notice the
file is a clone. `test_no_cloned_implementations` now compares the normalized
source of every registered `client_step` pairwise and fails on an exact match.
Verified to fire on the pre-fix file.

---

## 8. Known limitations

State these when presenting; none of them are hidden in the code.

1. **Bytes are analytic, not wire-measured** — no serialization, headers, or
   compression modelled (§1).
2. **Server-side aggregation is charged a full round-trip**, where the reference
   implementations charge zero. Uniform across methods, but it inflates the four
   multi-server methods (§1).
3. **Clients are simulated in one process on one device.** Per-side memory is
   reconstructed analytically + by autograd hooks precisely because a
   whole-process measurement cannot express "what one device needs" (§2).
4. **Memory is sampled on 2 batches per round**, not continuously (§2).
5. **Latency includes setup and evaluation**, and is the noisiest column (§3).
6. **Six methods are AI-assisted no-code reimplementations** and are *not yet
   validated against their papers' own reported numbers* (§7). Two of the six
   were found to deviate; the other four have not been independently confirmed,
   so their rows should be read as provisional.
7. **`comm_threshold_mb` can truncate a run** before `cfg.rounds`
   (`src/algos/__init__.py:661`). Matched rounds are only matched if this never
   fires — raise it for long sweeps, and check `run_manifest['rounds']` against
   the executed round count. The table warns on a mismatch.
8. **`fsl_sage` with `warm_start: true`** builds a fresh `CommLedger` for the
   second phase while inheriting the warm-start phase's series, so the
   cumulative comm curve restarts from 0 and is non-monotonic. Avoid warm-start
   for benchmark runs, or repair the series before plotting.

---

## 9. Reproducing the benchmark

```bash
# 1. correctness gate -- seconds, and a wrong byte counter invalidates a sweep
python test/check_accounting.py

# 2. one method, one config
cd src && python main.py algorithm=dsl_aux model=resnet18 dataset=mnist \
  cut=middle dataset.distribution=iid rounds=3 seed=200 save=True device=cuda

# 3. the full matched-round sweep: 12 methods x 3 seeds x {IID, Dirichlet 0.5}
cd inference && python run_mnist_benchmark.py \
  --cuts middle --seeds 200 201 202 --rounds <R> \
  --comm_threshold_mb <high enough that it never fires> --device cuda
```

The table lands in `inference/benchmark_table_mnist.txt`. It is only valid if it
prints **no** `INVALID TABLE`, `NOT COMPARABLE`, or `DUPLICATE SOURCE` warning.

### Environment note (local macOS venv)

This repo's venv runs **Python 3.14**, whose `argparse` added a `_check_help`
validation that rejects hydra 1.3.x's lazily-evaluated `--shell-completion` help
object, so `python main.py` aborts before any of our code runs with
`ValueError: badly formed help string`. It is a hydra/Python incompatibility,
not a harness bug, and it does not affect Colab (Python 3.11/3.12). Upgrading to
hydra 1.3.7 does not fix it. To run locally, neutralise that one hook:

```bash
cd src && python -c "
import argparse, runpy, sys, os
argparse.ArgumentParser._check_help = lambda self, a: None
sys.path.insert(0, os.getcwd()); sys.argv = ['main.py'] + sys.argv[1:]
runpy.run_path('main.py', run_name='__main__')
" algorithm=dsl_aux model=resnet18 dataset=mnist cut=middle rounds=1 device=cpu
```

`hydra-joblib-launcher` must also be installed (`config.yaml` selects it).
`test/check_accounting.py` is unaffected -- it builds its config with OmegaConf
directly and never touches hydra's CLI.

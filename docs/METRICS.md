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
(`src/algos/__init__.py:577`):

```
for t in range(cfg.rounds)                     # round
    for i in range(cfg.num_clients)            # client
        for j in range(clients[i].epochs)      # local epoch
            for k, (x, y) in enumerate(loader) # batch
                alg.client_step(...)
    alg.aggregate()
    alg.evaluate()
```

On MNIST with 10 clients, batch 256, 6000 samples/client, that is **24 optimizer
steps per client per round** — the same for all twelve methods.

> **Why this matters.** `ho_sfl` and `mu_splitfed` used to no-op on every batch
> except the round's first, faithful to their references, whose "round" *is* one
> batch. Under an equal round budget that handed them **1/24 of everyone else's
> updates**, so the sweep compensated with a 24× round multiplier for those two
> alone — which in turn made Comm-total and Latency incomparable across rows.
> The multiplier is gone and those two now step on every batch. Their
> zeroth-order update rule is untouched — only how many batches it is applied to.
> Enforced by `check_every_batch_trains`, which inspects every registered
> algorithm's `client_step` source and fails if a batch gate returns.

**There is no `Comm-to-target` metric.** Because rounds are matched and a round
means the same thing everywhere, cumulative **Comm-total at the final round** is
already a like-for-like comparison, and that is what the benchmark ranks on. The
table refuses to emit if its cells disagree on round count.

---

## 1. Communication

### What we claim to measure
Bytes that would cross a network link during training, split into **cut traffic**
(activations/gradients/scalars across the client↔server split) and **weight
traffic** (model/aggregation transfers). The ranked number is their sum.

### The code that does it
One ledger prices every transfer — `CommLedger`, `src/utils/comm.py:47`, one
instance per run (`src/algos/__init__.py:100`). Methods never compute bytes
themselves; they call typed helpers on the base class
(`src/algos/__init__.py:119` onward), so two methods cannot silently disagree
about what a transfer costs.

```python
# src/utils/comm.py:39 -- analytic wire size
def tensor_bytes(t):
    return t.numel() * t.element_size()
```

`numel()` so a ragged final batch is priced correctly rather than assumed full;
`element_size()` so `use_64bit: true` is reflected automatically.

**Categories** (`src/utils/comm.py:21` and `:30`):

| Cut | meaning |
|---|---|
| `act_up` | smashed activations, client → server |
| `grad_down` | gradient at the cut, server → client |
| `scalar_up` | scalar probes/losses, client → server |
| `scalar_down` | scalar loss differences, server → client (zeroth-order) |
| `labels_up` | labels, client → server (only when the loss is computed server-side) |

Weight categories are `client_up/down`, `aux_up/down`, `server_up/down`, plus
`scalars` for HO-SFL's dimension-free aggregation (scalars + seeds, never a
weight vector).

### Granularity
Cut traffic is charged **per batch, per client**, inside `client_step`. Weight
traffic is charged **once per round** inside `aggregate()`, in a per-client loop,
so it sums over clients. Counters are cumulative and never reset. Model weights
include **buffers** — BatchNorm running stats are real bytes on the wire
(`src/utils/utils.py:22`). Labels are int64, so 8 B/sample.

### Includes / excludes
- **Includes** every tensor and scalar that would have to be transmitted, at full
  precision, BN buffers included.
- **Excludes** serialization overhead, protocol headers, compression,
  quantization, retransmission. These are analytic byte counts, not packet
  captures — deliberate, because it makes the comparison implementation-
  independent, and any real transport would scale all methods by roughly the
  same factor.

### Two conventions worth stating plainly
1. **Labels are charged to the cut** whenever the loss is computed server-side
   (`src/algos/__init__.py:134`), and deliberately *not* for a purely
   client-local loss. Note `han_locloss` and `fedsplitx` *do* charge labels
   despite being local-loss methods: their servers compute their own
   cross-entropy, so the labels really do cross.
2. **Server-side aggregation is charged a full model round-trip** in every
   multi-server method — SplitFedv1, FedSplitX, Han-et-al, MU-SplitFed. The
   reference implementations charge this as *zero*, on the grounds that
   per-client server replicas live on one host, so averaging them is a memory
   copy rather than network traffic. This harness charges it because the
   alternative — charging some methods and not others for the identical
   operation — biases the ranked column. **The consequence is real and should be
   read with the table**: it inflates Comm-weights for those four methods
   relative to their papers' own accounting.

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
| FedSplitX | **0** | > 0 | collaborative local loss both sides, M=3 auxiliary heads |
| HOSL | **0** | **0** | zeroth-order, non-federated; **2Q+1 `act_up` and 2Q `scalar_down` per batch** |
| HO-SFL | > 0 | > 0, `scalars` only | client needs the cut gradient for its ZO probe |
| MU-SplitFed | **0** | > 0 | 3× `act_up` (fixed + two probes), 1 `scalar_down` |
| LocFedMix-SL | > 0 | > 0 | server-side Mixup; the *regularizer* is local, the cut gradient is real |

HO-SFL and MU-SplitFed were cross-checked line-for-line against the authors' own
accounting in `HKU-WILL-Lab/HO-SFL`'s `src/core/communicator.py`; both match
category-for-category (HO-SFL's reference really does log an activation-gradient
downlink, `ho_sfl_runner.py:157`).

---

## 2. Memory

Three columns, three different questions. They are **not** interchangeable, and
most confusion about this benchmark's memory results comes from reading one as
if it were another.

| Metric | Question it answers |
|---|---|
| `peak_client_mem_mb` | What must **one client device** provide? |
| `client_mem_held_across_cut_mb` | How much is the client forced to keep alive **while waiting** on the server? |
| `peak_system_live_mb` | What must **one host** provide for both sides **at once**? |

### 2a. Per-device peak

**Why the obvious approach was rejected.** Whole-process RSS is dominated by the
interpreter, torch, the dataset in RAM, and *every* simulated client's model
being resident at once. It is method-blind, and it pointed the *wrong way*:
CSE-FSL measured higher than SplitFedv2 on identical settings
(`src/utils/memory.py:5-11`). `torch.cuda.max_memory_allocated()` has the same
problem — process-global, so with every client simulated in one process it cannot
attribute bytes to a device.

**What is reported instead, per side: `peak = static + working`.**

- **Static, computed exactly** — params + buffers + `.grad`
  (`src/utils/memory.py:45`) and optimizer state, deduplicated by storage pointer
  so Adam ≈ 2× params while plain SGD is 0 (`:68`).
- **Working, measured** — `MemoryMeter` (`src/utils/memory.py:119`) installs
  `saved_tensors_hooks` inside a phase bracket and tracks a high-water mark of
  autograd-**saved** plus explicitly **held** bytes, keyed on storage pointer so
  views are not double-counted, released exactly via a finalizer.

**Reduction is `max` at every level** — across batches, epochs, rounds, and
finally across clients (`src/algos/__init__.py:244`). Never a mean: the metric is
what a *single* device must provide. Instrumentation runs only on the first
`mem_probe_batches` (default 2) batches of each round's first local epoch
(`:201`).

### 2b. Held across the cut — the decoupling signature

Client-owned autograd bytes still live *at the instant the server phase opens*
(`src/algos/__init__.py:174`). Exactly `0` means the client never stalls holding
its graph. This column is sensitive to **statement order**, so two methods were
corrected so it measures the algorithm rather than the port's code layout:

- **FSL-SAGE** now does its local backward *above* the server block on every
  iteration except an alignment one, where it genuinely must wait for the
  server-refreshed surrogate. Note this does not change the reported *number*:
  `held_across_cut` is a max, and `t % align_interval == 0` is true at `t = 0`
  for any interval, so every run contains one alignment iteration. What changed
  is how *often* the client stalls, which a max cannot express.
- **LocFedMix-SL** now frees its Infopro decoder subgraph eagerly via
  `torch.autograd.grad` instead of retaining it. On ResNet-18 at the middle cut
  that subgraph is dominated by a single `F.interpolate` back to input
  resolution — **128 MiB, 64× the smashed data** — which is why this column read
  241 MB where the backbone alone accounts for ~94. It now holds exactly what
  Vanilla-SL holds, asserted as such in the test suite.

### 2c. System (simultaneity) peak — the paper-comparable number

`peak_client_mem_mb` and `peak_server_mem_mb` are maxed **independently**, so
their sum is an upper bound that may never have occurred. Neither can express
that conventional SL keeps the client's activations alive **at the same instant**
as the server's, while a decoupled method frees them first. That simultaneity is
the whole of DSL-Aux's memory claim.

`peak_system_live_mb` (`src/algos/__init__.py:299`) is the high-water mark of
live bytes summed across **all owners at one instant**. It is maintained at the
meter's single choke point — `_account` (`src/utils/memory.py:242`) is the only
place live bytes are added and `_drop` (`:224`) the only place they are removed —
and it is sound to sum because `_entries` is keyed by storage `data_ptr`
**globally**, so a storage is counted once no matter which side touched it.
`peak_system_mem_mb` adds the static term (worst client + server).

**Measured on this harness**, ResNet-18 / batch 256, `vanilla_sl` vs `dsl_aux`:

| cut | CSL system peak | DSL system peak | saving | CSL held | DSL held |
|---|---|---|---|---|---|
| shallow | 115.04 MB | **79.01 MB** | **31.3%** | 76.00 MB | **0.00** |
| middle | 113.04 MB | 97.01 MB | 14.2% | 94.01 MB | **0.00** |
| deep | 112.04 MB | 106.02 MB | 5.4% | 103.02 MB | **0.00** |

This reproduces the paper's signature exactly (arXiv:2601.19261 Fig. 6, Obs. 3):
conventional SL is **near-constant across cuts** — "BP requires storing
intermediate activations regardless of where the cut layer is placed" — while
DSL **rises with cut depth**, so the saving is **largest at the shallow cut**.

We measure 31.3% at shallow where the paper reports up to 58% on ResNet-110. Same
mechanism, different magnitude: ResNet-110 has far more early-stage blocks, so
the client's share at a shallow cut is a smaller fraction of the whole. Two
further reasons our percentage is the more conservative one: the paper's
ResNet-110 has ~1.7M parameters, so its static term is negligible, whereas
ResNet-18's is ~43 MB (≈170 MB resident under Adam) and dilutes any percentage
that includes it — which is why `peak_system_live_mb` (measured live bytes only)
is the column to compare, not `peak_system_mem_mb`.

### 2d. Zeroth-order clients carry no gradient buffer

HOSL's paper states outright that ZO needs no gradient storage (Eq. 15,
`M_grad = 0`) and no stored activations (Eq. 18, `M_act = 0`), because the update
is applied in place by regenerating the perturbation from its seed (Alg. 3
`ZOUPDATE`). `hosl` and `mu_splitfed` now do exactly that — no `.grad` tensor and
no optimizer state is ever allocated, asserted in the test suite. For
`mu_splitfed` this is *mathematically identical* to what it did before, since its
reference uses plain SGD; only the allocation disappears.

**`ho_sfl` is deliberately excluded**: its own reference uses AdamW, whose moment
state the algorithm genuinely needs. Its client legitimately carries gradient and
optimizer memory, and forcing a stateless update there would change the method
rather than just the accounting.

### Includes / excludes
- **Includes**: params, buffers, grads, optimizer state, retained activations,
  explicitly held tensors (input batch, uploaded activation, downloaded gradient).
- **Excludes**: interpreter and framework overhead, the dataset, allocator
  fragmentation, cuDNN workspace. `peak_process_mem_mb` (RSS,
  `src/main.py:36`) and `peak_cuda_memory_mb` are **diagnostics only**.

---

## 3. Latency

Wall clock for the whole application run: `src/main.py:33` stamps
`_APP_START_TIME` at import; `:235` takes the delta just before writing
`results.json`.

- **Includes** dataset construction, model setup, all training, **all per-round
  evaluation passes**, and wandb init/finish.
- **Excludes** the final `torch.save`, which happens after the stamp.

State this plainly when presenting: it is an **application-level** number, not
pure training time. For a compute-only breakdown use
`client_model_compute_time` / `server_model_compute_time`, accumulated inside the
same `phase()` bracket that measures memory (`src/algos/__init__.py:174`).
Because it is wall-clock on shared hardware it is the **noisiest** column — treat
differences of a few percent as nothing.

---

## 4. Accuracy

Full pass over the test set under `no_grad`, through the method's own
`full_model(x)`, once per round after aggregation (`src/algos/__init__.py:354`,
called at `:661`). `acc = correct / len(test_loader.dataset)`.

Two methods define `full_model` non-trivially, deliberately:
- **Vanilla-SL** evaluates the single shared client model — there is only one.
- **FedSplitX** evaluates the **ensemble of all auxiliary heads plus the final
  output** (paper §3), which is how that method is specified to infer.

---

## 5. Provenance

Every run writes `run_manifest` into `results.json` (`src/utils/utils.py:83`):
git commit + dirty flag, algorithm, model, dataset, distribution, alpha, cut,
num_clients, rounds, seed, batch size, local epochs, dtype, device, and a
`config_sha256` of the fully-resolved config. Also written: `settings.yml`,
`output.log`, `metrics.pt`. Path layout at `src/utils/utils.py:146`.

**Table provenance.** `inference/benchmark_table.py` records the resolved
`results.json` path and `config_sha256` behind every cell, and emits a
`DUPLICATE SOURCE` warning if two rows resolved to the same file. One run printed
twice reads as two corroborating results — which is exactly how a DSL-Aux row
once came out byte-identical to Vanilla-SL's.

---

## 6. Worked example: DSL-Aux

DSL-Aux (Zihad, Owino, Tang & Huang, *Decoupled Split Learning via Auxiliary
Loss*, arXiv:2601.19261) is the clearest case of measurement following from the
algorithm, and it was previously implemented backwards.

### What the paper specifies
- **Alg. 1 / §III-B** — the server computes ∂L/∂z internally, but "**this
  gradient is not transmitted to the client**. It is only used for the server's
  weight updates."
- **§III-C** — λ = 0, "to avoid any BP signals from the server to the client".
- **§IV-A** — the auxiliary classifier is "a single fully-connected layer that
  maps the activations z to 10/100 classes, plus a softmax layer".
- **Obs. 2 / 3 / 4** — communication falls ~50%; peak memory up to 58%;
  per-epoch runtime is *moderately higher*.

### The corrected implementation (`src/algos/dsl_aux.py:122`)

```python
with self.phase('client', i):          # client phase OPENS and CLOSES
    ...                                # before the server runs, so no
    z = self.clients[i].model(x)       # graph is held across the cut -> Obs. 3
    aux_out = aux.forward_inner(z.flatten(1))
    aux_loss = self.criterion(aux_out, y)   # single FC + softmax  -> SIV-A
    aux_loss.backward()                     # the client's ONLY signal
    self.clients[i].optimizer.step(); aux.optimizer.step()

smashed_data = z.detach()
self.charge_cut_activation(smashed_data)   # -> cut.act_up
self.charge_cut_labels(y)                  # -> cut.labels_up
# NO charge_cut_gradient -- SIII-B: never transmitted -> Obs. 2

with self.phase('server', i):
    out = self.server.model(smashed_data)
    loss = self.criterion(out, y)
    loss.backward()                        # dL/dz computed, never sent
    self.server.optimizer.step()
```

| paper | code | metric effect |
|---|---|---|
| §III-B | no `charge_cut_gradient` | `cut.grad_down == 0` |
| Obs. 2 | only act_up + labels_up charged | Comm-cut ≈ half Vanilla-SL's |
| Obs. 3 | client phase closes before server phase | `held_across_cut == 0`, lower system peak |
| §IV-A | single linear head | small `client_param_mem_mb` |
| non-federated | `aggregate()` averages nothing | `comm_load_weights == 0` |
| Obs. 4 | extra aux forward/backward | Latency *above* Vanilla-SL |

### How it was wrong

The previous implementation charged `charge_cut_gradient(grad_at_cut)` and called
`splitting_output.backward(grad_at_cut)` — conventional split learning with an
extra auxiliary term. Its own header recorded the decision: it had adapted the
third-party reference's `train_standard_split` and explicitly *rejected*
`train_dgl`, the function carrying the comment "DGL has NO backward comm from
Server to Client". That is precisely backwards. Because it charged the identical
tensors on the identical schedule as `vanilla_sl.py`, its communication was
**identical by construction** — which is exactly what the previous report showed.

```python
# juniorfelix998/sl-fl-dgl -- train_standard_split (conventional SL)
tracker.log_comm(client_out,  "fwd")
tracker.log_comm(grad_at_cut, "bwd")      # <- both directions

# juniorfelix998/sl-fl-dgl -- train_dgl (decoupled: what DSL-Aux is)
tracker.log_comm(features, "fwd")
# DGL has NO backward comm from Server to Client
```

### Measured result

ResNet-18 / MNIST / middle cut / IID / 2 clients / 1 round / seed 200, CPU --
`dsl_aux` and `vanilla_sl` on an otherwise identical config:

| metric | Vanilla-SL | DSL-Aux | paper's claim | verdict |
|---|---|---|---|---|
| Comm-cut | 0.916 GiB | **0.458 GiB** | ~50% less (Obs. 2) | **exactly 50.0%** |
| Comm-weights | 0.000 | 0.000 | non-federated | both 0 |
| Held-across-cut | 94.01 MiB | **0.00 MiB** | decoupled (SIII-B) | holds nothing |
| **System live peak** | **113.05 MiB** | **97.02 MiB** | lower (Obs. 3) | **14.2% lower** |
| Peak client mem | 109.44 MiB | 107.76 MiB | lower (Obs. 3) | marginally lower |
| Client activations | 94.01 MiB | 94.02 MiB | -- | *identical* |
| Latency | 251.54 s | 241.92 s | *higher* (Obs. 4) | within noise |
| Test acc | 97.70% | 97.11% | on par (Obs. 1) | comparable |

Before the fix these two rows were identical on every column.

Three things in this table are worth reading carefully:

- **Client activations are identical (94.01 vs 94.02 MiB), and that is correct.**
  The paper says so itself in SIII-C: "the memory cost for the client includes
  storing activations for its own BP... In conventional SL, the client would
  similarly store those, but need to hold them **longer**, waiting for the
  server's gradient." The saving is in *duration and simultaneity*, not in how
  much the client stores. That is why `peak_client_mem_mb` barely moves while
  `held_across_cut` goes to zero and the system peak drops 14%.
- **The system peak is the column that carries Obs. 3.** Cross-check: the same
  quantity measured directly through the meter (S2c) gives 113.04 / 97.01 for
  this cut -- agreeing with this end-to-end run to two decimals.
- **Latency does not resolve.** DSL-Aux came out 3.8% *faster* here where the
  paper predicts moderately slower. Wall-clock on a shared CPU is the noisiest
  column in this benchmark (S3); a single-round, few-percent difference in
  either direction is not evidence. Obs. 4 is neither confirmed nor refuted by
  this run.

### Locked in by tests
Against a Vanilla-SL run on an identical config: `cut.grad_down` is zero; cut
total equals Vanilla-SL's minus the gradient half; cut traffic is ~50% of
Vanilla-SL's; no weight traffic; system peak strictly below Vanilla-SL's. A
regression to plain SL cannot pass silently.

---

## 7. Paper fidelity

Four methods have public code; the rest are AI-assisted no-code
reimplementations, all now checked against their papers.

| Method | Source | Code? | Verified against | Remaining disclosed deviation |
|---|---|---|---|---|
| SplitFedv1 / v2 | Thapa et al., AAAI 2022 | — | in-harness baselines | server-side FedAvg charged as network traffic (§1) |
| Vanilla-SL | Gupta & Raskar 2018 | — | sequential/relay SL | — |
| CSE-FSL | Mu & Shen, TMC 2025 | ✔ | authors' method, in-harness | — |
| FSL-SAGE | Nair et al., ICML 2025 | ✔ | this repo's own method | — |
| HO-SFL | Chen et al., ICML 2026 | ✔ | authors' `communicator.py`, category-for-category | client carries AdamW state (its reference's own optimizer) |
| MU-SplitFed | Liang et al., NeurIPS 2025 | 3rd-party | `HKU-WILL-Lab/HO-SFL` runner, line-for-line; every hyperparameter matches | runs ~47 ZO steps/round vs the reference's 1 (our matched-rounds rule) |
| **DSL-Aux** | arXiv:2601.19261 | — | Alg. 1, §III-B, §III-C, §IV-A | not validated against the paper's reported accuracy |
| **FedSplitX** | arXiv:2310.14579 | — | §2.1–2.2 Fig. 1: auxiliary net at every partition point, collaborative loss, ensemble inference (§3) | client heterogeneity not exercised — one shared cut means `heteroavg` degenerates to FedAvg |
| **Han-et-al** | Han et al., FL-ICML 2021 | — | §3 Eq. 1–5: two local losses, no cut gradient; auxiliary now ~0.18% of model, inside the paper's stated 0.1–0.6% | **server-side replicas + server FedAvg are this harness's own extrapolation**, not in the paper |
| **HOSL** | arXiv:2601.10940 | — | Def. 3 / Eq. 6–7 two-sided estimator, Alg. 1 three phases, App. VII-A (ε=1e-3, Q=10, SGD no momentum), Eq. 15/18 no grad or activation storage | not validated against the paper's reported accuracy (its experiments are OPT/GLUE, not vision) |
| **LocFedMix-SL** | Oh et al., WWW 2022 | — | Eq. 4 real cut gradient, Eq. 5–7 smashed-data mixup detached from the lower segment, Eq. 8 split update, Eq. 9–10 aggregation | decoder architecture is a stand-in (paper specifies only that h maps smashed → reconstruction); `T_c = 1`, the paper's own best-accuracy endpoint on its Fig. 6 trade-off |

### Two clones that were found and fixed

`fedsplitx.py` was once a byte-for-byte copy of `han_locloss.py` with the class
name changed, producing two identical benchmark rows. Every other assertion in
the suite is per-method — it can confirm `fedsplitx` charges the bytes
`fedsplitx` should charge and still not notice the file is a clone.
`test_no_cloned_implementations` now compares the normalized source of every
registered `client_step` pairwise and fails on an exact match; it was verified to
fire on the pre-fix file.

The other was DSL-Aux (§6), caught because its row was identical to Vanilla-SL's.

---

## 8. Known limitations

State these when presenting; none are hidden in the code.

1. **Bytes are analytic, not wire-measured** — no serialization, headers, or
   compression modelled (§1).
2. **Server-side aggregation is charged a full round-trip** where the references
   charge zero. Uniform across methods, but it inflates the four multi-server
   methods.
3. **Clients are simulated in one process on one device.** Per-side memory is
   reconstructed analytically plus by autograd hooks precisely because a
   whole-process measurement cannot express "what one device needs" (§2).
4. **Memory is sampled on 2 batches per round**, not continuously.
5. **Latency includes setup and evaluation**, and is the noisiest column (§3).
6. **MU-SplitFed's near-chance accuracy is a REPRODUCTION, not a failure.** The
   HO-SFL paper's own Figure 3 reports MU-SplitFed flat at ~10–15% for its entire
   run, as a deliberately weak backprop-free baseline that HO-SFL improves on.
   Our port matches the reference line-for-line on every published
   hyperparameter; at its own ε = 5e-3 the perturbation is 12–27% of the weight
   norm and the update diverges from the first step. It is not tuned, by
   decision — tuning it would make it a different method.
7. **Five reimplementations remain unvalidated against their papers' reported
   numbers.** Their *mechanisms* are now checked equation by equation (§7), but
   reproducing their published accuracy is a separate exercise.
8. **`comm_threshold_mb` can truncate a run** before `cfg.rounds`
   (`src/algos/__init__.py:697`). Matched rounds are only matched if this never
   fires — raise it for long sweeps. The table warns on a mismatch.
9. **`fsl_sage` with `warm_start: true`** builds a fresh `CommLedger` for the
   second phase while inheriting the warm-start phase's series, so the cumulative
   comm curve restarts from 0. Avoid warm-start for benchmark runs.
10. **HOSL at the paper's Q = 10 is expensive** — 21 activation uploads and 21
    client forward passes per batch, roughly 7× its previous cost. That is the
    honest trade its own Limitations section concedes, and it dominates sweep
    runtime.

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
object, so `python main.py` aborts before any harness code runs with
`ValueError: badly formed help string`. A hydra/Python incompatibility, not a
harness bug; it does not affect Colab (Python 3.11/3.12), and hydra 1.3.7 does
not fix it. To run locally, neutralise that one hook:

```bash
cd src && python -c "
import argparse, runpy, sys, os
argparse.ArgumentParser._check_help = lambda self, a: None
sys.path.insert(0, os.getcwd()); sys.argv = ['main.py'] + sys.argv[1:]
runpy.run_path('main.py', run_name='__main__')
" algorithm=dsl_aux model=resnet18 dataset=mnist cut=middle rounds=1 device=cpu
```

`hydra-joblib-launcher` must also be installed. `test/check_accounting.py` is
unaffected — it builds its config with OmegaConf directly and never touches
hydra's CLI.

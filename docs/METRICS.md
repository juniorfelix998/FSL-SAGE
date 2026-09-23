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
(`src/algos/__init__.py:590`):

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

### Is this a running log, or a conceptual calculation?

Both, and the distinction matters enough to state before anything else:

- **The event is real.** Every charge fires *during the run*, inside
  `client_step()` or `aggregate()`, at the actual point in the algorithm where
  the transfer would happen, with the actual tensor. Nothing is extrapolated
  from a formula afterwards. A run that stops early logs only what it actually
  did, and a ragged final batch is priced at its real size, not a nominal one.
- **The price is analytic.** `tensor_bytes(t) = t.numel() * t.element_size()`.
  There are no sockets in the simulation, so there is no wire to measure: no
  serialization, no protocol headers, no compression, no retransmission.

So: **a running log of analytically priced events.** The useful consequence is
that it is checkable two independent ways -- the ledger produces a number at
runtime, *and* every cell reproduces from a closed form by hand. When those two
disagree, something is wrong. Worked example, from a 5-client / 3-round sweep:

```
client-side ResNet-18 @ middle cut (params + buffers) = 2,740,048 B = 2.6131 MiB
server-side                                           = 42,025,080 B = 40.0782 MiB

SplitFedv2  = 3 rounds x 5 clients x 2 (up+down) x 2.6131            =   78.39 MiB
SplitFedv1  = 3 x 5 x 2 x (2.6131 + 40.0782)                         = 1280.74 MiB
CSE-FSL     = (3 x 5 x 2 x 2.6131) + (3 x 10 x 8.0294)               =  319.27 MiB
FSL-SAGE    = (3 x 5 x 2 x 2.6131) + (5 x 8.0294, one-way)           =  118.54 MiB
Vanilla-SL  = 3 x 5 x 2.6131 (relay, ONE way per handover)           =   39.20 MiB
```

Each of those matches its reported cell exactly. If a number looks wrong, this is
the first thing to check -- and if the arithmetic reproduces it, the disagreement
is not about the sums but about **which events count as transfers**, which is the
next subsection.

### What counts as a transfer

This is where the accounting has actually been wrong before, so it is worth
being explicit. The test is **"does a model cross the wire?"** -- not "does the
method aggregate?". Three methods report near-zero weight traffic for two
completely different reasons:

| Method | Weights | Why |
|---|---|---|
| DSL-Aux, HOSL | **0** | Every client owns an independently constructed model that is never shared, averaged, or handed on. Nothing crosses the wire. Physically zero. |
| Vanilla-SL | **the relay** | Never aggregates either -- but it *relays* one client-side model from each client to the next. Movement without aggregation still costs bytes. |
| HO-SFL | **0.04** | Also shares one model, but prices its synchronization as the paper's dimension-free scalar + seed exchange rather than as a model transfer. |

**Vanilla-SL's relay used to be charged zero, and that was a bug.**
`src/algos/vanilla_sl.py` aliases a single `nn.Module` across every client and
the driver trains them sequentially, so client *i*'s updated weights are consumed
by client *i+1*. The file's own comment describes this ("sequential training
naturally carries each client's just-updated weights forward to the next") -- in
a deployment each handover is a network transfer of the client-side model. The
simulation got it free because no copy is ever made. The old justification, "no
aggregation event ever happens", conflated aggregation with movement.

It is now charged as `weights.client_relay`: **one** transmission per handover
(peer-to-peer, client *i* straight to client *i+1* -- not a round-trip, nothing
comes back), and `num_clients` handovers per round, counting the wrap back to
client 0 that begins the next round. At 5 clients that is `5 x 2.6131 = 13.07`
MiB/round.

This was the same class of bug as the one already fixed one layer up: the inline
`BUGFIX` note at `src/algos/baselines.py:188-193` records SplitFedv1's
server-side exchange having been charged zero for exactly the same reason.

### What we claim to measure
Bytes that would cross a network link during training, split into **cut traffic**
(activations/gradients/scalars across the client↔server split) and **weight
traffic** (model/aggregation transfers). The ranked number is their sum.

### The code that does it
One ledger prices every transfer — `CommLedger`, `src/utils/comm.py:51`, one
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

**Categories** (`src/utils/comm.py:24` and `:30`):

| Cut | meaning |
|---|---|
| `act_up` | smashed activations, client → server |
| `grad_down` | gradient at the cut, server → client |
| `scalar_up` | scalar probes/losses, client → server |
| `scalar_down` | scalar loss differences, server → client (zeroth-order) |
| `labels_up` | labels, client → server (only when the loss is computed server-side) |

Weight categories are `client_up/down`, `aux_up/down`, `server_up/down`,
`client_relay` (the sequential-SL handover -- one way, no aggregator involved),
plus `scalars` for HO-SFL's dimension-free aggregation (scalars + seeds, never a
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
   operation — biases the ranked column.

   **This dominates those methods' weight traffic**, so the table carries an
   `of which server-side` sub-column to make it separable. That column is
   important enough to explain on its own — see immediately below.

### Reading the `of which server-side` column

**It is not an extra cost.** It is a *breakdown of* `Comm-weights`, naming how
much of that number comes from one specific operation: **averaging the
per-client copies of the SERVER-side model**.

Four methods — SplitFedv1, MU-SplitFed, Han-et-al, FedSplitX — keep **one server
replica per client**, so each round they FedAvg those replicas as well as the
client-side models. The server-side model is 40.08 MB against the client-side
2.61 MB — **15× larger** — so that single operation dominates everything else
the method sends. Worked from a real run (`rounds × clients = 6`):

```
client-side aggregation = 6 x 2 x  2.6131 =  31.36 MB   <- every FedAvg method pays this
server-side aggregation = 6 x 2 x 40.0782 = 480.94 MB   <- only the multi-server methods
                                            ---------
SplitFedv1 Comm-weights                    = 512.30 MB
```

**Why it is broken out.** Every reference implementation charges this as *zero*:
the replicas share one host, so averaging them is a memory copy, not network
traffic. This harness charges it anyway, because otherwise SplitFedv1 and
FedSplitX would be billed for an operation that SplitFedv2 performs internally
for free — two methods billed differently for the same work. But that is *our*
convention, not the literature's, so the column exists to let anyone remove it.

Subtract it and the picture changes completely:

| Method | Comm-weights | of which server-side | **net** |
|---|---|---|---|
| SplitFedv1 | 512.30 | 480.94 | **31.36** |
| SplitFedv2 | 31.36 | 0.00 | **31.36** |
| MU-SplitFed | 512.30 | 480.94 | **31.36** |
| Han-et-al | 513.23 | 480.94 | **32.29** |
| FedSplitX | 512.50 | 481.06 | **31.44** |
| CSE-FSL | 127.71 | 0.00 | 127.71 |
| FSL-SAGE | 47.42 | 0.00 | 47.42 |
| LocFedMix-SL | 32.22 | 0.00 | 32.22 |
| Vanilla-SL | 15.68 | 0.00 | 15.68 |
| HO-SFL | 0.04 | 0.00 | 0.04 |
| DSL-Aux, HOSL | 0.00 | 0.00 | 0.00 |

**The four apparently-expensive methods collapse to ≈31.4 MB — the same as
SplitFedv2.** Their 16× disadvantage is entirely this accounting convention, not
a property of the algorithms. And the ranking inverts: under the papers' own
convention **CSE-FSL (127.71) becomes the most weight-expensive method**, because
its auxiliary network really is FedAvg'd across the network every round.

Neither column is "the right one" — they answer different questions. *Comm-weights*
asks "what would this protocol cost if every participant were a separate
machine", which is the deployment question and is why it stays in the ranked
Comm-total. *Net of server-side* asks "what do the papers count", which is the
right column for checking our numbers against a published one. Quote whichever
you are arguing about, but say which.

### Expected accounting signature per method
A row that violates its signature is a bug, not a finding.

| Method | `cut.grad_down` | `weights` | note |
|---|---|---|---|
| SplitFedv1 / SplitFedv2 | > 0 | > 0 | full BP across the cut, FedAvg |
| Vanilla-SL | > 0 | `client_relay` only | never aggregates, but relays one model client-to-client |
| CSE-FSL | **0** | > 0 | local aux head; uploads only every `q`-th batch |
| FSL-SAGE | **0** | > 0 | surrogate synthesises the cut gradient; aux charged download-only |
| DSL-Aux | **0** | **0** | decoupled *and* genuinely shares nothing (see §6) |
| Han-et-al (LGL-SL) | **0** | > 0 | local losses both sides |
| FedSplitX | **0** | > 0 | collaborative local loss both sides, M=3 auxiliary heads |
| HOSL | **0** | **0** | zeroth-order, shares nothing; **2Q+1 `act_up` and 2Q `scalar_down` per batch** |
| HO-SFL | > 0 | > 0, `scalars` only | client needs the cut gradient for its ZO probe |
| MU-SplitFed | **0** | > 0 | 3× `act_up` (fixed + two probes), 1 `scalar_down` |
| LocFedMix-SL | > 0 | > 0 | server-side Mixup; the *regularizer* is local, the cut gradient is real |

HO-SFL and MU-SplitFed were cross-checked line-for-line against the authors' own
accounting in `HKU-WILL-Lab/HO-SFL`'s `src/core/communicator.py`; both match
category-for-category (HO-SFL's reference really does log an activation-gradient
downlink, `ho_sfl_runner.py:157`).

---

## 2. Memory

Three reported columns, three different questions. They are **not**
interchangeable, and most confusion about this benchmark's memory results comes
from reading one as if it were another. §2.0 first answers the prior question —
how any of them can be separated at all when both sides share one GPU.

| Metric | Question it answers |
|---|---|
| `peak_client_mem_mb` | What must **one client device** provide? |
| `client_mem_held_across_cut_mb` | How much is the client forced to keep alive **while waiting** on the server? |
| `peak_system_live_mb` | What must **one host** provide for both sides **at once**? |

### 2.0 If client and server share ONE GPU, how are they tracked separately?

The sweeps run on a single Colab T4 with both sides resident in the same VRAM,
so this is the first question the memory columns have to answer.

**The short answer: we never ask the device.** The CUDA allocator has no notion
of "which model requested this block", so `torch.cuda.max_memory_allocated()`
cannot be split by side no matter how it is bracketed. Instead every byte is
attributed **at the moment of allocation, by which side caused it**. Two
mechanisms, covering the two kinds of memory:

**(a) Static bytes are computed exactly, never measured.** Each algorithm
*declares* which modules live on which side — `client_side_modules(i)` /
`server_side_modules()` (`src/algos/__init__.py:235-245`), overridden by methods
that put an auxiliary head on the client or keep per-client server replicas.
`module_static_bytes()` then sums `nelement() * element_size()` over exactly
those modules (`src/utils/memory.py:45`), and `optimizer_state_bytes()` does the
same for optimizer state, deduplicated by storage pointer. No device query is
involved, so this half is exact and identical on CPU and T4.

**(b) Working bytes are attributed by the phase bracket that was open when
autograd saved them.** `MemoryMeter.phase()` installs
`torch.autograd.graph.saved_tensors_hooks` and pushes an owner
(`src/utils/memory.py`); every tensor autograd retains while that bracket is open
is charged to that owner:

```python
with self.phase('client', i):     # owner = ('client', i)
    z = self.clients[i].model(x)  # every saved tensor -> the client
with self.phase('server', i):     # owner = ('server',)
    out = self.server.model(z)    # every saved tensor -> the server
```

So the split is by **causal ownership in program order**, not by physical
location. Both tensors sit in the same T4 VRAM; we know which side put them
there. Keyed on `untyped_storage().data_ptr()` so views and in-place-shared
storages count once, and released exactly via a finalizer when the graph is
freed — which is what makes the result an instantaneous high-water mark rather
than a running total.

**Why this is the right answer here, not merely a different one.** On a
simulation the allocator's number describes the *simulation*: all N clients'
models resident at once, the dataset, cuDNN workspace, fragmentation, allocator
caching. A real deployment has one client model on the client device.
`max_memory_allocated` therefore answers a question nobody asked, and cannot
answer the one we did.

**The four memory numbers a T4 run produces:**

| Number | Answers | Device-dependent? |
|---|---|---|
| `peak_client_mem_mb` | what **one client device** would need if deployed for real | no — analytic + ownership |
| `peak_server_mem_mb` | what the **server host** would need | no |
| `peak_system_live_mb` / `peak_system_mem_mb` | what **one host running both sides** must hold at one instant (the paper-comparable number) | no |
| `peak_cuda_memory_mb` | what the **T4's allocator actually peaked at** during the simulation | yes |

These must satisfy

```
peak_cuda_memory_mb  >=  peak_system_mem_mb  >=  max(peak_client, peak_server)
```

and `src/main.py` asserts exactly that on every CUDA run, logging `OK` or
`VIOLATED`. A violation would mean the meter is counting bytes the allocator
never handed out — double counting or mis-attribution — so it is a real check.
The *gap* between the first two is logged too: it is the simulation's own
overhead (N−1 extra client models, dataset, workspace, fragmentation), i.e.
precisely the part a real deployment would not pay.

**The honest limitation.** Because attribution is analytic, we measure the
*logical* memory requirement. Allocator caching, fragmentation and cuDNN
workspace are excluded by construction. That is the trade that makes the number
portable across CPU and T4 and comparable between methods — but it means our
figure is not what `nvidia-smi` shows.

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
finally across clients (`src/algos/__init__.py:257`). Never a mean: the metric is
what a *single* device must provide. Instrumentation runs only on the first
`mem_probe_batches` (default 2) batches of each round's first local epoch
(`:201`).

### 2b. Held across the cut — the decoupling signature

Client-owned autograd bytes still live *at the instant the server phase opens*
(`src/algos/__init__.py:187`). Exactly `0` means the client never stalls holding
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

`peak_system_live_mb` (`src/algos/__init__.py:312`) is the high-water mark of
live bytes summed across **all owners at one instant**. It is maintained at the
meter's single choke point — `_account` (`src/utils/memory.py:247`) is the only
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
same `phase()` bracket that measures memory (`src/algos/__init__.py:187`).
Because it is wall-clock on shared hardware it is the **noisiest** column — treat
differences of a few percent as nothing.

---

## 4. Accuracy

Full pass over the test set under `no_grad`, through the method's own
`full_model(x)`, once per round after aggregation (`src/algos/__init__.py:367`,
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
| Comm-weights | 0.005 GiB (relay) | **0.000** | — | DSL shares nothing at all |
| Comm-total | 0.921 GiB | **0.458 GiB** | — | 49.7% |
| Held-across-cut | 94.01 MiB | **0.00 MiB** | decoupled (SIII-B) | holds nothing |
| **System live peak** | **113.05 MiB** | **97.02 MiB** | lower (Obs. 3) | **14.2% lower** |
| Peak client mem | 109.44 MiB | 107.76 MiB | lower (Obs. 3) | marginally lower |
| Client activations | 94.01 MiB | 94.02 MiB | — | *identical* |
| Latency | 257.50 s | 247.32 s | *higher* (Obs. 4) | within noise |
| Test acc | 97.70% | 97.11% | on par (Obs. 1) | comparable |

Before the fix these two rows were identical on every column.

Note Vanilla-SL's Comm-weights is no longer zero: at 2 clients its relay is
`2 x 2.6131 = 5.23 MiB` per round (§1). DSL-Aux stays at exactly 0 because each
client owns an independent model that never moves — the two are zero-vs-nonzero
for a real reason, not by convention.

Four things in this table are worth reading carefully:

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
   charge zero. Uniform across methods, but it dominates the four multi-server
   methods — 94% of SplitFedv1's weight traffic — which is why the table now
   breaks it out as a sub-column (§1).
3. **The sequential-SL relay is a modelling choice.** Vanilla-SL's handover is
   charged peer-to-peer, one transmission per handover, `num_clients` handovers
   per round. Routing it through the server instead would double it. The
   convention is stated so it can be disagreed with explicitly rather than
   silently assumed.
4. **Clients are simulated in one process on one device.** Per-side memory is
   attributed by declared module ownership plus autograd phase brackets (§2.0),
   precisely because a whole-process allocator figure cannot express "what one
   device needs". The consequence is that we report the *logical* requirement:
   allocator caching, fragmentation and cuDNN workspace are excluded by
   construction, so our number is not what `nvidia-smi` shows. On CUDA runs the
   harness asserts `cuda >= system >= max(side)` as a check on the attribution.
5. **Memory is sampled on 2 batches per round**, not continuously.
6. **Latency includes setup and evaluation**, and is the noisiest column (§3).
7. **MU-SplitFed's near-chance accuracy is a REPRODUCTION, not a failure.** The
   HO-SFL paper's own Figure 3 reports MU-SplitFed flat at ~10–15% for its entire
   run, as a deliberately weak backprop-free baseline that HO-SFL improves on.
   Our port matches the reference line-for-line on every published
   hyperparameter; at its own ε = 5e-3 the perturbation is 12–27% of the weight
   norm and the update diverges from the first step. It is not tuned, by
   decision — tuning it would make it a different method.
8. **Five reimplementations remain unvalidated against their papers' reported
   numbers.** Their *mechanisms* are now checked equation by equation (§7), but
   reproducing their published accuracy is a separate exercise.
9. **`comm_threshold_mb` can truncate a run** before `cfg.rounds`
   (`src/algos/__init__.py:710`). Matched rounds are only matched if this never
   fires — raise it for long sweeps. The table warns on a mismatch.
10. **`fsl_sage` with `warm_start: true`** builds a fresh `CommLedger` for the
   second phase while inheriting the warm-start phase's series, so the cumulative
   comm curve restarts from 0. Avoid warm-start for benchmark runs.
11. **HOSL at the paper's Q = 10 is expensive** — 21 activation uploads and 21
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

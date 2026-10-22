# FSL-SAGE: MNIST IID, 10 clients, 3 rounds

> **Superseded.** These results were produced with the **original** CSE-FSL and
> FSL-SAGE code, before it was changed to follow each paper's pseudocode. The
> differences (4 uploads per round instead of 5; FSL-SAGE alignment at the round
> boundary, initial x_c⁰ broadcast, first-order client gradient) are listed in
> [`docs/CSE_FSL_FSL_SAGE_BREAKDOWN.md`](../../CSE_FSL_FSL_SAGE_BREAKDOWN.md#paper-vs-original-code-changed-after-review).
> Current results: [`../mnist-iid-n10-r11/`](../mnist-iid-n10-r11/). This page is
> kept as a record; statements a review found wrong are corrected inline.

**Paper:** Nair et al., *Federated Split Learning via Smashed Activation
Gradient Estimation*, arXiv:2505.23182 (ICML 2025).

**Run:** Colab, Tesla T4. ResNet-18, middle cut, MNIST resized to 32×32 and
repeated to 3 channels, IID, seed 200, batch 256.

```
!python main.py rounds=3 save=False device=cuda algorithm=fsl_sage \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true
```

**Setup that every number depends on:**

| Symbol | Meaning | Value |
|---|---|---|
| N | clients | 10 |
| R | rounds | 3 |
| K | local batches per client per round (6,000 images / 256) | 24 (23 full + 1 of 112) |
| h | smashed-data upload period (paper: K/Q, p.8 "every 5 local steps") | 5 |
| l | aux-model alignment period (paper p.8) | 10 |
| \|x_c\| | client model (ResNet-18 up to stage 2) | 2.6131 MiB (685K params) |
| \|x_a\| | auxiliary model (paper p.8: "first server ResNet block + FC") | 8.0294 MiB (2.1M params) |
| \|x_s\| | server model (stages 3–4 + classifier) | 40.08 MiB (10.5M params) |

All units are MiB (1024² bytes), which is what the benchmark table calls "MB".

---

## Communication: 537.37 MiB total

Every row shows how many times something was sent × its size. **Expected** is
the count from the original code's schedule. Every row matched (✓), but that
only shows the code followed its own rule: the papers' pseudocode gives 4
uploads per round, not 5 (and FSL-SAGE also broadcasts x_c⁰ first).

| # | What is sent | Direction | When (paper) | Count | Expected | Size each | MiB |
|---|---|---|---|---:|---:|---:|---:|
| C1 | Smashed activations z_f | client → server | every h = 5 batches (Alg.1 l.11–12) | 150 | 10 × 3 × ⌈24/5⌉ = 150 ✓ | 2.0000 | **300.00** |
| C2 | Labels y for those batches | client → server | with every C1 upload (Alg.3 l.3–4) | 150 | 150 ✓ | 0.0020 | **0.29** |
| C3 | Gradient at the cut | server → client | **never**: true grads stay on the server (Alg.3 l.11–12) | 0 | 0 ✓ | – | **0.00** |
| W1 | Client model x_c | client → server | every round, for FedAvg (Alg.2 l.6) | 30 | 10 × 3 = 30 ✓ | 2.6131 | **78.39** |
| W2 | Averaged client model x_c | server → client | every round (Alg.2 l.7–8) | 30 | 30 ✓ | 2.6131 | **78.39** |
| W3 | Aux model x_a | client → server | **never**: aux models are not averaged | 0 | 0 ✓ | – | **0.00** |
| W4 | Aligned aux model x_a | server → client | once every l = 10 rounds (Alg.1 l.3–4, Fig.1) | 10 | 10 × ⌈3/10⌉ = 10 ✓ | 8.0294 | **80.29** |
| W5 | Server-model FedAvg | – | **none**: one server model (p.8) | 0 | 0 ✓ | – | **0.00** |

| Total | How it adds up | MiB |
|---|---|---:|
| **Cut** | C1 + C2 = 300.00 + 0.29 | **300.29** |
| **Weights** | W1 + W2 + W4 = 78.39 + 78.39 + 80.29 | **237.08** |
| of which Server-model FedAvg | W5 | 0.00 |
| **Total** | Cut + Weights | **537.37** |
| Cut share | 300.29 / 537.37 | 56% |

**How to read it:**
- **Cut traffic (C1, C2).** Each client has 24 batches per round and uploads
  one every 5th batch (k = 0, 5, 10, 15, 20), so 5 uploads per client per
  round. One upload is 256 images × 2,048 floats × 4 B = 2 MiB. The ragged last
  batch (k = 23) is never uploaded, so every upload is exactly 2 MiB.
- **No gradient comes back (C3 = 0).** This is the point of FSL-SAGE. The
  server uses the true gradients only to train ("align") the aux model, then
  sends the aux model instead.
- **Aux model sent once (W4 = 10 sends).** Alignment happens in rounds 0, 10,
  20, …, so with R = 3 only round 0 aligns. That is one 8.03 MiB download per
  client. With 200 rounds it would be 20 downloads per client.
- **W1 + W2** is the ordinary FedAvg of the client model, the same as
  SplitFed and CSE-FSL.

---

## Memory (per device)

> **Updated: aux memory moved to the side that actually uses it.** In
> FSL-SAGE the **client only runs** the aux model, to estimate the cut
> gradient every batch (Alg.1). The **server trains (aligns)** it every l
> rounds (Alg.3), and the aux optimizer only steps there. So the client holds
> the aux model's parameters only, and the aux gradients and Adam state are on
> the server. The first Colab run charged them to the client.
>
> **Confirmed by a T4 rerun after the fix:** client 133.76, server 218.60,
> system 334.07. Working memory, Held-across-cut (94.01) and the live system
> peak (123.30) did not change.
>
> | | Before | After |
> |---|---:|---:|
> | Client | 157.82 | **133.76** (−24.06 aux grads + Adam) |
> | Server | 186.52 | **218.60** (+32.09 aux copy being aligned) |

### Client: 133.76 MiB (worst single client)

| Item | Params + buffers | Grads | Adam state | MiB |
|---|---:|---:|---:|---:|
| Client model x_c (trained here) | 2.61 | 2.61 | 5.21 | 10.43 |
| Auxiliary model x_a (only **run** here, never trained) | 8.03 | 0.00 | 0.00 | 8.03 |
| Working-memory peak (activations + input batch) | | | | 115.30 |
| ↳ of which autograd activations | | | | 112.30 |
| **Client total** | | | | **133.76** |

| Item | MiB | Meaning |
|---|---:|---|
| **Held-across-cut** | **94.01** (artifact; 0 under the paper) | Part of the client's working memory that stays alive *while the client waits for the server*. **Correction:** this came from the original code aligning in the middle of iteration 0; under the paper (Alg.1 l.3–4) the aligned aux model arrives at the start of a round and Held = 0. It happens only on alignment iterations (round 0, first batch here): the client must wait for the freshly aligned aux model before it can backpropagate. It is already inside the 115.30, not extra. |

### Server: ≈ 606 MiB in total (10 clients) — lower bound, see correction below

The server holds three groups of things. Only the first group is the value in the
benchmark table's Server column.

| Item | Params + buffers | Grads | Adam state | MiB |
|---|---:|---:|---:|---:|
| Server model x_s (one copy, shared by all clients) | 40.08 | 40.05 | 80.10 | 160.22 |
| Aux model x_a, the copy being aligned right now (Alg.3 l.9–13) | 8.03 | 8.02 | 16.04 | 32.09 |
| Working-memory peak (activations + uploaded batch + stored z_f) | | | | 26.29 |
| ↳ of which autograd activations | | | | 22.29 |
| **Server total (table value)** (per-device peak, used in the benchmark table) | | | | **218.60** |
| Aux models of the other 9 clients (params + grads + Adam), 9 × 32.09 | | | | 288.79 |
| Alignment buffers × 10 clients (z_f + true gradients + labels) | | | | 98.20 |
| **Total server memory, all 10 clients** | | | | **≈ 606** (605.59) |

**Why the table uses 218.60 and not 606.** The server aligns one client's aux
model at a time, so 218.60 is what it needs to do that work. The other two
lines are state it must *keep* for every client between alignments:
- **Aux models of the other 9 clients:** Alg.3 gives each client its own aux
  model and Adam state.
- **Alignment buffers:** paper p.5 "Store for client i". Per client: up to
  1,000 stored z_f (7.81 MiB), true gradients from the last alignment (256
  samples, 2 MiB) and labels.

Both grow with the number of clients, while 218.60 does not. Use **606** when
asking "how much memory does an FSL-SAGE server need for 10 clients", and
**218.60** when comparing per-device peaks across methods.

**Correction (review).** These server numbers are lower bounds:
- 98.20 is an end-of-run snapshot (1,000 z_f but only 256 true gradients, from
  round 0). At steady state each buffer holds 1,000 of both, about 15.6 MiB
  per client, which puts the full total near 664 MiB.
- The working-memory line (26.29) only saw the 256-sample round-0 alignment;
  the full 1,000-sample alignment happens in round 10.
- About 2 MiB of the current client's z_f is in both the working-memory line
  and the buffers line.

All three are fixed in the 11-round results.

### System peak

| Item | MiB |
|---|---:|
| **System peak** (live bytes, client + server at the same instant) | **123.30** |
| with static model + optimizer bytes added | 334.07 |

---

## Accuracy, time and process diagnostics

Values are from the T4 rerun after the fix.

| Item | Value |
|---|---:|
| **Test accuracy** (round 2) | **98.71%** |
| **Latency** (application start → end) | **116.23 s** |
| Client compute time (sum) | 0.66 s |
| Server compute time (sum), including alignment | 3.11 s |
| Client FedAvg compute time | 0.51 s |
| Peak process RSS (diagnostic) | 1854.51 MiB |
| Peak CUDA allocator (diagnostic) | 1582.93 MiB |
| Ordering check cuda ≥ system ≥ max(side) | OK |

Server compute (3.11 s) is much higher than CSE-FSL's (0.22 s) because of
alignment. The server reruns its model over the stored buffer to get true
gradients, then trains the aux model on them.

The CUDA and RSS figures are not benchmark metrics. They include all 10
simulated clients, the dataset and the cuDNN workspace on one GPU.

## Questions and answers

### 1. Is the aux model on the client or on the server?

**Both, with different jobs.** The paper and the code agree.

| | Client | Server |
|---|---|---|
| **What it does with the aux model** | **Runs** it every batch to estimate the gradient at the cut, then backprops its own model with that estimate. It never trains the aux model. | **Trains (aligns)** it every l = 10 rounds against the true gradients, then sends the new copy down. |
| **Paper** | Alg.1: the client keeps x_{a,i} from the last round and computes its gradient estimate with it | Alg.3 l.9–13: the S-server aligns x_{a,i} on the stored (z_f, y). Fig.1: "S → C_i: x_{a,i}, once in l rounds" |
| **Code** | `fsl_sage.py`, `__client_backward`, inside `phase('client')` | `fsl_sage.py`, `refresh_data()` + `align()`, inside `phase('server')` |
| **Memory** | aux **params only** (8.03 MiB) | aux params + grads + Adam state (32.09 MiB per copy) |
| **Communication** | receives it (W4) | sends it (W4: 10 sends × 8.03 MiB) |

The client never changes the aux model, so the server's copy is always the
current one. That is why FSL-SAGE never uploads the aux model (W3 = 0).

### 2. What does "↳ of which autograd activations" mean?

**"↳ of which" means the line is a part of the line directly above it, not an
extra cost.** For the server, the 22.29 is already inside the 26.29:
160.22 + 32.09 + 26.29 = 218.60, with nothing added for 22.29.

- **Working-memory peak (26.29):** all the temporary memory the server needs at
  its busiest moment, beyond the models themselves. It has two kinds of tensors:
  - **Autograd activations:** during a forward pass, each layer saves its
    intermediate outputs so the backward pass can compute gradients. They exist
    only between the forward and backward pass, then get freed. This is usually
    the biggest chunk.
  - **Explicitly held data:** tensors the server keeps on purpose, not for
    autograd. Here that's the uploaded smashed batch (2 MiB) and the stored z_f
    alignment buffer.
- **↳ of which autograd activations (22.29):** the first kind on its own,
  measured at its own highest point.

**How to read the gap:** roughly 26.29 − 22.29 ≈ 4 MiB is the held data, mainly
the 2 MiB uploaded batch plus the stored z_f at that moment. It's "roughly"
because the two figures can peak at slightly different instants, so the
difference isn't an exact accounting.

**Why it's shown:** it tells you what drives the working memory. For the server
it's mostly backprop through the server model (22.29 of 26.29). On the client
the same split shows 112.30 of 115.30 is activations, the client model's own
forward graph.

### 3. What does "Server total (table value)" mean?

It is **the single number that goes in the benchmark table's Server column**,
the one methods are compared and ranked on. For FSL-SAGE it is **218.60 MiB**.
(Earlier versions of this doc called it the "headline".)

- **In the Server total (table value):** the server model (160.22), one aux
  model being aligned (32.09) and working memory (26.29).
- **Not in it:** the other 9 clients' aux models (288.79) and the alignment
  buffers (98.20). They are real memory and are listed separately.

The client works the same way: its table value is 133.76.

### 4. Why does the Server total hold one aux model, with "the other 9" listed separately?

In FSL-SAGE **every client has its own aux model**, so with 10 clients there are
10 aux models, and the server is the one that trains them (Alg.3).

- **One copy in the table value:** the server aligns them one after another,
  so at any instant only one is being trained. That one, with its gradients and
  Adam state, is the 32.09 in the Server total.
- **The other 9 listed separately:** the server still has to keep them, plus
  their Adam state, ready for their turn. That is the 288.79 (9 × 32.09).

Together that's all 10 aux models: 32.09 + 288.79 = 320.88 = 10 × 32.09.

| Piece | MiB | In the Server total (table value)? |
|---|---:|---|
| Server model x_s | 160.22 | yes |
| Aux model being aligned now (1 client) | 32.09 | yes |
| Working memory | 26.29 | yes |
| **Server total (table value)** | **218.60** | |
| Aux models of the other 9 clients | 288.79 | no |
| Alignment buffers of all 10 clients | 98.20 | no |
| **Total server memory, all 10 clients** | **≈ 606** | |

**Why the extras are kept out of the table value:** the table value is a
per-device peak, comparable across methods. The extras grow with the number of
clients (double the clients and they roughly double), while 218.60 stays the
same. Listing them separately shows the client-count cost without changing the
ranked column.

The 606 is approximate: the working-memory line already includes part of one
client's stored z_f, so adding the full 98.20 counts a few MiB twice.

### 5. Why is the true-gradient buffer smaller than the activation buffer (z_f)?

**Per sample they are the same size.** A true gradient has exactly the shape of
the activation it belongs to: 128×4×4 = 2,048 floats × 4 B = 8 KiB. The
difference is in **how many samples** each holds.

| Buffer (per client, end of run) | Samples | Size |
|---|---:|---:|
| `data_x` (z_f) | 1,000 (the cap) | 1,000 × 8 KiB = **7.81 MiB** |
| `data_y` (true gradient) | 256 | 256 × 8 KiB = **2.00 MiB** |
| `data_labels` | 1,000 | 1,000 × 8 B = 0.01 MiB |
| **Per client** | | **9.82 MiB** → × 10 clients = **98.20** |

**Why the counts differ:** the two buffers are filled at different times.
- **z_f grows every upload.** Every 5th batch the server appends the 256
  uploaded samples (`add_datapoint`). Over 3 rounds that's 15 uploads × 256 =
  3,840 samples, trimmed to the newest 1,000.
- **The true gradient is only computed at alignment.** `refresh_data()` runs
  the server over the whole buffer and stores the gradients in `data_y`. That
  happens only when `t % 10 == 0` on the first batch. In a 3-round run that's
  once, at round 0, batch 0, when the buffer held just the first upload (256
  samples).

So by the end of the run, `data_y` is a stale snapshot from round 0. 98.20 MiB
is the end-of-run size, not the peak. At the next alignment (round 10),
`data_y` would be recomputed over all 1,000 samples and become 7.81 MiB, the
same as z_f, giving about 15.6 MiB per client (≈ 156 MiB for 10 clients).

### 6. Why is the alignment buffer capped at 1,000 samples?

**It's a setting in the original FSL-SAGE authors' code. The paper doesn't give
a number.**

- **Where it's set:** `src/hydra_config/model/auxiliary/base_auxiliary.yaml`
  sets `max_dataset_size: 1000`, together with `align_batch_size: 1000` and
  `align_epochs: 100`. It came in with the upstream authors' commits (late
  2024), not anything added in this benchmark.
- **What the paper says:** the buffer grows every round, and in practice "one
  can discard the older entries in order to maintain a fixed maximum size"
  (p.5). For its theory it assumes unlimited storage. It never names a size.
- **Why a cap at all:** without one, the buffer and the alignment cost grow
  forever. Each alignment reruns the server model over the whole buffer, then
  trains the aux model on it for 100 passes.
- **Why 1,000 specifically:** not stated. A likely reason (my inference): with
  `align_batch_size` also 1,000, each alignment pass is a single batch over the
  whole buffer, which keeps alignment cheap.

**What it means in practice:** the buffer keeps the newest 1,000 samples, about
4 uploads' worth (4 × 256 = 1,024, trimmed to 1,000). So when the aux model is
aligned, it learns only from the most recent activations, not the whole
history.

### 7. What is "Server-model FedAvg" (formerly "of which server")?

The part of **Weights** spent averaging **per-client copies of the server-side
model**. It is already included in Weights, not added on top
(Total = Cut + Weights).

- **How it's computed:** `weights.server_up + weights.server_down` =
  N clients × R rounds × 2 (up + down) × |x_s|. Example, SplitFedv1 on MNIST
  with 2 clients × 3 rounds: 2 × 3 × 2 × 40.08 = 480.94 MiB.
- **Who pays it:** only methods that keep one server model per client
  (SplitFedv1, MU-SplitFed, Han et al., FedSplitX).
- **FSL-SAGE pays 0:** it keeps a single server model (p.8: "a single
  server-side model is maintained").

<details>
<summary>Raw breakdown output (T4 rerun after the aux memory re-split)</summary>

```
> Round 2, avg tr g_loss: 0.06, avg tr g_acc: 0.98, ts. loss: 0.04, ts. acc: 98.71%, comm cut: 0.293 GiB, comm weights: 0.232 GiB, comm total: 0.525 GiB.
> Peak client mem: 133.76 MiB (act 112.30, held across cut 94.01), peak server mem: 218.60 MiB, system live peak: 123.30 MiB.

## Breakdown: fsl_sage  —  FSL-SAGE, Nair et al., arXiv:2505.23182
N = 10 clients, R = 3 rounds, local iters/round = [24 x 10], h = 5, l = 10.  Units: MiB (= table 'MB').

### Communication
| # | What is sent | Paper | Column | Count | Expected | MiB / transfer | MiB | |
| C1 | Smashed activations z_f, client -> server | Alg.1 l.11-12; p.8: every K/Q = h local steps | Cut | 150 | 150 | 2.0000 | 300.00 | ✓ |
| C2 | Labels y for each uploaded batch, client -> server | Alg.3 l.3-4: server receives (z_f, y) | Cut | 150 | 150 | 0.0020 | 0.29 | ✓ |
| C3 | Gradient at the cut, server -> client | Alg.3 l.11-12: true grads used only on the server -> never sent | Cut | 0 | 0 | 0.0000 | 0.00 | ✓ |
| W1 | Client model x_c upload (for FedAvg) | Alg.2 l.6: every round | Weights | 30 | 30 | 2.6131 | 78.39 | ✓ |
| W2 | Averaged client model x_c download | Alg.2 l.7-8: every round | Weights | 30 | 30 | 2.6131 | 78.39 | ✓ |
| W3 | Aux model x_a upload | Alg.2: aux models are never averaged -> never sent up | Weights | 0 | 0 | 0.0000 | 0.00 | ✓ |
| W4 | Aligned aux model x_a download, server -> client i | Alg.1 l.3-4, Fig.1: once every l rounds | Weights | 10 | 10 | 8.0294 | 80.29 | ✓ |
| W5 | Server-model FedAvg (per-client x_s copies) | p.8: "a single server-side model is maintained" -> none | Weights | 0 | 0 | 0.0000 | 0.00 | ✓ |
- Cut = 300.29 (rows sum ✓) | Weights = 237.08 (rows sum ✓), Server-model FedAvg 0.00 | Total = 537.37 | Cut share 56%

### Memory — client (worst client = 0)
| client model x_c        | 2.61 | 2.61 | 5.21 | 10.43 |
| auxiliary model x_a     | 8.03 | 0.00 | 0.00 |  8.03 |
| working-memory peak (activations at own peak: 112.30) | | | | 115.30 |
- Client total = 133.76 MiB (lines sum ✓) | Held-across-cut = 94.01 MiB

### Memory — server
| server model x_s                         | 40.08 | 40.05 | 80.10 | 160.22 |
| auxiliary model x_a (copy being aligned) |  8.03 |  8.02 | 16.04 |  32.09 |
| working-memory peak (activations at own peak: 22.29) | | | | 26.29 |
- Server total = 218.60 MiB (lines sum ✓)
- not in Server total: alignment buffers x 10 clients 98.20 MiB; aux models of the other 9 clients 288.79 MiB
- System peak (live) = 123.30 MiB; with static = 334.07 MiB

Summary of net compute times: client 0.66s, server 3.11s, client_agg 0.51s
Total latency: 116.23s | Peak process memory: 1854.51 MiB | Peak CUDA memory: 1582.93 MiB
Memory attribution on Tesla T4: client 133.76 | server 218.60 | system 334.07 | cuda-allocator 1582.93 MiB -- ordering OK
```
</details>

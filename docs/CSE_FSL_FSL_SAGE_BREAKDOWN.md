# CSE-FSL and FSL-SAGE: where every MB comes from

This doc explains each line of the breakdown that every run now prints at the
end and saves to `breakdown.md` (and to `results.json["breakdown"]`). Each
line is one thing that crosses the network or sits in memory. It names the paper
step that causes it and shows

```
count  x  size per transfer  =  MiB          (paper count, ✓/✗)
```

**What ✓ means:** the run's count equals the count derived from the paper's
pseudocode. The formula and the paper line it comes from are printed in each
row, so you can check the derivation yourself. A ✗ means the run did not
follow the paper's schedule. All units are MiB (1024² bytes), which is what the
benchmark table labels "MB".

Papers:
- **CSE-FSL**: Mu & Shen, *Communication and Storage Efficient Federated Split Learning*, arXiv:2302.05599
- **FSL-SAGE**: Nair et al., *Federated Split Learning via Smashed Activation Gradient Estimation*, arXiv:2505.23182 (ICML 2025)

Neither paper gives a closed-form communication or memory formula. Both
describe the protocol and report measured totals. The schedules below are read
directly off their algorithm steps.

Notation: N clients, R rounds, K local iterations per round per client, h =
upload period (`server_update_interval`, 5), l = alignment period
(`align_interval`, 10, FSL-SAGE only). |x_c|, |x_a|, |x_s| = client, aux and
server model sizes.

---

## Paper vs original code (changed after review)

Both methods now follow their paper's pseudocode. The original
implementations differed in the ways below; each change is also documented at
the top of `src/algos/cse_fsl.py` / `src/algos/fsl_sage.py`.

| # | Method | Paper | Original code | Now | Effect (N = 10, K = 24 per round) |
|---|---|---|---|---|---|
| A | FSL-SAGE | Alg.1 l.11 / Alg.3 l.3: upload when k ≡ 0 mod K/Q, k ∈ [K] = 1..K | local_iter 0, 5, …, 20 (first batch of each window) | batches 5, 10, 15, 20 (last batch of each window) | 4 uploads per round, not 5 |
| B | FSL-SAGE | Alg.1 l.3–4: aligned x_a arrives at the **start** of round t; Eq.(2) aligns on the Qt samples from the t rounds before; Alg.3 l.9 aligns after the uploads | aligned in the middle of iteration 0 (on that batch too), while the client's graph was alive | aligns in `begin_round(t)`, before any client trains | Held-across-cut 94.01 → **0**. At t = 0 the buffer is empty (Qt = 0), so rounds 0–9 train with an **unaligned** aux model |
| C | FSL-SAGE | Alg.2 l.4: broadcast x_c⁰, plus l.8 after every round | only the post-aggregation broadcasts | initial broadcast charged too | W2 = N(R+1), +N transfers |
| D | CSE-FSL | Step 2 says "m mod h = 0", but Eq.(8) and Fig.2 say the server gets g_{x_c^{t,h−1}} "when m = h", and Eq.(6) gives windows of h steps, m = 0..h−1 | first batch of each window (local_iter 0, 5, …) | last batch of each window (model x_c^{t,h−1}), as Eq.(8)/Fig.2 | 4 uploads per round, not 5 |
| E | FSL-SAGE (memory only) | Alg.1 l.14 needs only the **value** of the gradient estimate | built the second-order graph (`create_graph=True`) on every client step | client uses `create_graph=False`; `align()` keeps it | same numbers; the aux forward graph is freed after the client step instead of staying alive |

**Why D reads Step 2 as "m = h":** Step 2's text alone would allow the first
batch (m = 0), but Eq.(8) and Fig.2 both say the server receives the output of
the client model *after h−1 updates* (x_c^{t,h−1}). That is the h-th batch of
the window. Eq.(6) confirms each window has h steps, m = 0..h−1. Two of the
three statements agree, and they are the ones with formulas.

**Why A uses k ∈ 1..K:** Alg.1 writes its loop as k = 1..K−1, but the text says
"K local iterations" and Alg.3 uses k ∈ [K]. The two readings give the same
count whenever h does not divide K (true for K = 24 and K = 118).

With A and D, both methods upload on the same batches (h, 2h, …), so their cut
traffic is identical again, now at the papers' 4 per round instead of 5.

---

## Communication

### Cut (traffic across the split, during training)

| # | What is sent | CSE-FSL | FSL-SAGE | Count |
|---|---|---|---|---|
| C1 | Smashed activations z_f, client → server | Eq.(8), Fig.2 (p.3–4): the last batch of each h-batch window ("when m = h") | Alg.1 l.11–12 / Alg.3 l.3: k ∈ 1..K with k ≡ 0 mod K/Q (= h) | N·R·⌊K/h⌋ |
| C2 | Labels y for that batch | Step 2: "uploads the smashed data **and the labels**" | Alg.3 l.3–4: server receives (z_f, y) | N·R·⌊K/h⌋ |
| C3 | Gradient at the cut, server → client | **Never.** p.3: clients "update the models locally without waiting for the transmission of the gradients" | **Never.** Alg.3 l.11–12: true gradients are computed and used only on the server, to align the aux model | 0 |

C1 and C2 are **identical** for the two methods: same h, same batches (h, 2h, …).

### Weights (model transfers)

| # | What is sent | CSE-FSL | FSL-SAGE |
|---|---|---|---|
| W1 | Client model x_c up, for FedAvg | Step 4, Eq.(9): every round → N·R | Alg.2 l.6: every round → N·R |
| W2 | x_c down | Step 1: start of every round → N·R | Alg.2 l.4 (initial x_c⁰) + l.8 (after every round) → N(R+1) |
| W3 | Aux model up | Step 4, Eq.(9): aux **is averaged** every round → N·R | **Never**: aux is not averaged → 0 |
| W4 | Aux model down | Step 1: every round → N·R | Alg.1 l.3–4, Fig.1 caption: at the start of every round t ≡ 0 mod l ("once in l rounds") → N·⌈R/l⌉ |
| W5 | **Server-model FedAvg** | 0 | 0 |

**Almost the whole Weights difference between the two methods is aux-model
traffic** (W3 + W4). CSE-FSL sends the aux model up and down every round;
FSL-SAGE only sends it down, once every l rounds. The remainder is FSL-SAGE's
initial x_c⁰ broadcast (W2 has N extra transfers; Alg.2 l.4).

### Server-model FedAvg (formerly "of which server")

**What it is:** the part of Weights spent averaging **per-client copies of
the server-side model**. It is already **inside** Weights, not added on top:
Total = Cut + Weights.

**How we get it:**

```
Server-model FedAvg = weights.server_up + weights.server_down          (ledger)
                    = N clients x R rounds x 2 (up + down) x |x_s|
```

It is charged by `charge_weights_roundtrip(server_model, 'server')`, once per client per round.

Worked example, SplitFedv1, MNIST, 2 clients × 3 rounds: 2 × 3 × 2 × 40.08 = **480.94 MiB**.

**Who pays it:** only methods that keep one server model per client
(SplitFedv1, MU-SplitFed, Han et al., FedSplitX). Both papers here keep
**one** server model, so it is 0:
- CSE-FSL p.3: "keep only a single server-side model instead of multiple models"
- FSL-SAGE p.8: "a single server-side model is maintained"

### Worked check: MNIST, ResNet-18 middle cut, 2 clients × 3 rounds

Sizes: |x_c| = 2.6131 MiB, |x_a| = 8.0294 MiB, smashed batch = 256 × 2048
floats × 4 B = 2 MiB, labels = 256 × 8 B = 2 KiB. K = 118 batches; uploads at
paper batches k = 5, 10, …, 115 give ⌊118/5⌋ = 23 per client-round. The ragged
last batch (k = 118) is never an upload batch, so every upload is a full 2 MiB.

| | CSE-FSL | FSL-SAGE |
|---|---:|---:|
| C1+C2 Cut: 6 × 23 × (2 MiB + 2 KiB) | 276.27 | 276.27 |
| W1 client up: 6 × 2.6131 | 15.68 | 15.68 |
| W2 client down: CSE 6 × 2.6131; SAGE (2 + 6) × 2.6131 (initial + per round) | 15.68 | 20.90 |
| W3+W4 aux: CSE 6 × 2 × 8.0294; SAGE 2 × 8.0294 (t = 0 only) | 96.35 | 16.06 |
| W5 Server-model FedAvg | 0 | 0 |
| **Weights** | **127.71** | **52.64** |
| **Total** | **403.98** | **328.91** |

The original code gave 288.28 cut (24 uploads per client-round), and 47.42
FSL-SAGE weights (no initial broadcast). The benchmark table's CSE-FSL and
FSL-SAGE cells change accordingly on the next sweep.

---

## Memory

Memory is per device: the worst single client, and the server. The breakdown
splits each into lines that **sum exactly to the table number**:

**Client** (table "Client"):
- `client model x_c`: params + buffers, grads, Adam state.
- `auxiliary model x_a`: both methods keep the aux model on the client
  permanently, since it replaces the server's gradient. Who **trains** it
  differs:
  - **CSE-FSL:** the client trains it every batch on its local loss (Step 3,
    Eq.5). So the client holds params + grads + Adam state (32.09 MiB).
  - **FSL-SAGE:** the client only **runs** it to estimate the cut gradient
    (Alg.1). The aux optimizer only steps in `align()` on the server (Alg.3).
    So the client holds **params only** (8.03 MiB), and the grads and Adam
    state are counted on the server.
- `working-memory peak`: autograd activations + the input batch at their
  highest point.

**Held-across-cut** (table "Held"): how much of the client's autograd graph is
still alive while the server is working, i.e. the client is stalled waiting.
It is already part of the working-memory line, not extra.
- CSE-FSL: 0. The client never waits for the server.
- FSL-SAGE: 0. The aligned aux model arrives at the start of a round (Alg.1
  l.3–4), so the client never waits mid-step. The original code reported
  94.01 MiB because it aligned in the middle of iteration 0; that was an
  artifact of statement order, not of the algorithm (fix B).

**Server** (table "Server"):
- `server model x_s`: params, grads, Adam state. One copy for both methods.
- FSL-SAGE only: `auxiliary model x_a (copy being aligned)`: params + grads +
  Adam state (32.09 MiB). The server aligns one client's aux model at a time.
- `working-memory peak`: activations + the uploaded smashed batch. For
  FSL-SAGE also the alignment pass (`refresh_data()` + `align()` on the buffer
  of the client being aligned) and that client's buffer (z_f + true gradients).
  The alignment pass is only at full size once the buffer is full, i.e. from
  the alignment in round l = 10 on, so **FSL-SAGE server memory is only
  representative for runs of at least 11 rounds**.

**Server-side extras (FSL-SAGE, printed separately, NOT in the Server total):**
Alg.3 keeps on the S-server, per client, an alignment buffer (z_f, true
gradient, y; ≤ 1000 samples, p.5 "Store for client i") and that client's aux
model plus its optimizer. The Server total (table value) already holds the
client being aligned (its aux copy and its buffer). The extras lines cover the
**other N−1 clients**, so nothing is counted twice:
- their aux models + Adam state;
- their buffers **at capacity** (steady state: 1,000 × (z_f + true gradient)
  + labels). The measured end-of-run size of all buffers is printed next to
  it, since in a short run the true-gradient buffer can still be a smaller
  snapshot from the last alignment.

The breakdown also prints **Total server memory incl. all clients' state** =
Server total + these lines.

**System peak:** live bytes on both sides **at the same instant**, so it
shows whether the client and server need memory simultaneously.

---

## How to get it

On Colab, each run prints the breakdown at the end of its cell:

```
!python main.py rounds={ROUNDS} save=False device=cuda algorithm=cse_fsl {COMMON}
!python main.py rounds={ROUNDS} save=False device=cuda algorithm=fsl_sage {COMMON}
```

With `save=True` it is also written to `<save_path>/breakdown.md` and
`results.json`. To compare two saved runs side by side:

```
python inference/breakdown_report.py <cse_fsl>/results.json <fsl_sage>/results.json
```

Code: `src/utils/breakdown.py` (rows and paper references),
`src/utils/comm.py` (byte and transfer counters),
`FLAlgorithm.memory_report()` (memory lines).
`test/check_accounting.py` asserts that every count matches the paper-derived
count, that FSL-SAGE holds nothing across the cut, and that the memory probe
samples a server step for both methods.

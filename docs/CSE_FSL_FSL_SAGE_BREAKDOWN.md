# CSE-FSL and FSL-SAGE: where every MB comes from

This doc explains each line of the breakdown that every run now prints at the
end and saves to `breakdown.md` (and to `results.json["breakdown"]`). Each
line is one thing that crosses the network or sits in memory. It names the paper
step that causes it and shows

```
count  x  size per transfer  =  MiB          (expected count from the paper, ✓/✗)
```

A ✗ means the run did not follow the paper's schedule. All units are MiB
(1024² bytes), which is what the benchmark table labels "MB".

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

## Communication

### Cut (traffic across the split, during training)

| # | What is sent | CSE-FSL | FSL-SAGE | Count |
|---|---|---|---|---|
| C1 | Smashed activations z_f, client → server | Sec.III-B Step 2 (p.3): "if current batch number m satisfies m mod h = 0, each client … uploads the smashed data" | Alg.1 l.11–12; p.8: "cut-layer features are sent to the server-side model every 5 local steps" | N·R·⌈K/h⌉ |
| C2 | Labels y for that batch | Step 2: "uploads the smashed data **and the labels**" | Alg.3 l.3–4: server receives (z_f, y) | N·R·⌈K/h⌉ |
| C3 | Gradient at the cut, server → client | **Never.** p.3: clients "update the models locally without waiting for the transmission of the gradients" | **Never.** Alg.3 l.11–12: true gradients are computed and used only on the server, to align the aux model | 0 |

C1 and C2 are **identical** for the two methods: same h, same batches.

### Weights (model transfers)

| # | What is sent | CSE-FSL | FSL-SAGE |
|---|---|---|---|
| W1 | Client model x_c up, for FedAvg | Step 4, Eq.(9): every round → N·R | Alg.2 l.6: every round → N·R |
| W2 | Averaged x_c down | Step 1: every round → N·R | Alg.2 l.7–8: every round → N·R |
| W3 | Aux model up | Step 4, Eq.(9): aux **is averaged** every round → N·R | **Never**: aux is not averaged → 0 |
| W4 | Aux model down | Step 1: every round → N·R | Alg.1 l.3–4, Fig.1 caption: "the updated x_{a,i} is sent to C_i once in l rounds" → N·⌈R/l⌉ |
| W5 | **Server-model FedAvg** | 0 | 0 |

**The whole Weights difference between the two methods is aux-model traffic**
(W3 + W4). CSE-FSL sends the aux model up and down every round. FSL-SAGE only
sends it down, once every l rounds.

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
k = 0, 5, …, 115 give 24 per client-round.

| | CSE-FSL | FSL-SAGE |
|---|---:|---:|
| C1+C2 Cut: 6 × 24 × (2 MiB + 2 KiB) | 288.28 | 288.28 |
| W1+W2 client FedAvg: 6 × 2 × 2.6131 | 31.36 | 31.36 |
| W3+W4 aux: CSE 6 × 2 × 8.0294, SAGE 2 × 8.0294 (t=0 only) | 96.35 | 16.06 |
| W5 Server-model FedAvg | 0 | 0 |
| **Weights** | **127.71** | **47.42** |
| **Total** | **415.99** | **335.70** |

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
- FSL-SAGE: non-zero only on alignment iterations (t % l == 0, first batch).
  There the client's gradient must come from the freshly aligned aux model.

**Server** (table "Server"):
- `server model x_s`: params, grads, Adam state. One copy for both methods.
- FSL-SAGE only: `auxiliary model x_a (copy being aligned)`: params + grads +
  Adam state (32.09 MiB). The server aligns one client's aux model at a time.
- `working-memory peak`: activations + the uploaded smashed batch (+ FSL-SAGE's
  stored z_f alignment buffer).

**Server-side extras (FSL-SAGE, printed separately, NOT in the Server total):**
Alg.3 keeps on the S-server, per client, an alignment buffer (z_f, true
gradient, y; ≤ 1000 samples, p.5 "Store for client i") and that client's aux
model plus its optimizer. The Server total (table value) counts only z_f and the
one aux copy being aligned. These lines show the rest (all clients' buffers and
the other N−1 aux copies), so you can decide whether to add them.

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
`test/check_accounting.py` asserts that every count matches the paper schedule.

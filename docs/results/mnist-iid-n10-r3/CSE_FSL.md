# CSE-FSL: MNIST IID, 10 clients, 3 rounds

**Paper:** Mu & Shen, *Communication and Storage Efficient Federated Split
Learning*, arXiv:2302.05599.

**Run:** Colab, Tesla T4. ResNet-18, middle cut, MNIST resized to 32×32 and
repeated to 3 channels, IID, seed 200, batch 256.

```
!python main.py rounds=3 save=False device=cuda algorithm=cse_fsl \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true
```

**Setup that every number depends on:**

| Symbol | Meaning | Value |
|---|---|---|
| N | clients | 10 |
| R | rounds | 3 |
| K | local batches per client per round (6,000 images / 256) | 24 (23 full + 1 of 112) |
| h | smashed-data upload period (paper Step 2: "m mod h = 0") | 5 |
| \|x_c\| | client model (ResNet-18 up to stage 2) | 2.6131 MiB (685K params) |
| \|a_c\| | auxiliary network (same 2.1M-param head as FSL-SAGE, for a fair comparison) | 8.0294 MiB |
| \|x_s\| | server model (stages 3–4 + classifier) | 40.08 MiB (10.5M params) |

All units are MiB (1024² bytes), which is what the benchmark table calls "MB".

---

## Communication: 938.84 MiB total

Every row shows how many times something was sent × its size. **Expected** is
the count the paper's algorithm predicts. Every row matched (✓).

| # | What is sent | Direction | When (paper) | Count | Expected | Size each | MiB |
|---|---|---|---|---:|---:|---:|---:|
| C1 | Smashed activations z_f | client → server | only when m mod h = 0 (Sec.III-B Step 2, p.3) | 150 | 10 × 3 × ⌈24/5⌉ = 150 ✓ | 2.0000 | **300.00** |
| C2 | Labels y for those batches | client → server | with every C1 upload (Step 2: "the smashed data and the labels") | 150 | 150 ✓ | 0.0020 | **0.29** |
| C3 | Gradient at the cut | server → client | **never**: client trains on its local aux loss (p.3) | 0 | 0 ✓ | – | **0.00** |
| W1 | Client model x_c | client → server | every round, for FedAvg (Step 4, Eq.9) | 30 | 10 × 3 = 30 ✓ | 2.6131 | **78.39** |
| W2 | Averaged client model x_c | server → client | start of every round (Step 1) | 30 | 30 ✓ | 2.6131 | **78.39** |
| W3 | Aux model a_c | client → server | **every round**: aux is FedAvg'd too (Step 4, Eq.9) | 30 | 10 × 3 = 30 ✓ | 8.0294 | **240.88** |
| W4 | Averaged aux model a_c | server → client | start of every round (Step 1) | 30 | 30 ✓ | 8.0294 | **240.88** |
| W5 | Server-model FedAvg | – | **none**: "keep only a single server-side model" (p.3) | 0 | 0 ✓ | – | **0.00** |

| Total | How it adds up | MiB |
|---|---|---:|
| **Cut** | C1 + C2 = 300.00 + 0.29 | **300.29** |
| **Weights** | W1 + W2 + W3 + W4 = 78.39 + 78.39 + 240.88 + 240.88 | **638.55** |
| of which Server-model FedAvg | W5 | 0.00 |
| **Total** | Cut + Weights | **938.84** |
| Cut share | 300.29 / 938.84 | 32% |

**How to read it:**
- **Cut traffic (C1, C2).** This is identical to FSL-SAGE: same h, same
  batches, 5 uploads per client per round of 2 MiB each.
- **No gradient comes back (C3 = 0).** The client gets its training signal
  from a small local auxiliary head instead of the server.
- **The aux model dominates (W3 + W4 = 481.76 MiB, 51% of everything sent).**
  CSE-FSL averages the aux model like the client model, so every client sends
  it up and gets it back every round. Here the aux head (8.03 MiB) is 3× bigger
  than the client model (2.61 MiB), so it costs 3× more than the client FedAvg
  itself.
- That is why Weights (68%) outweighs Cut (32%) for CSE-FSL at 10 clients.

---

## Memory (per device)

### Client: 148.80 MiB (worst single client)

| Item | Params + buffers | Grads | Adam state | MiB |
|---|---:|---:|---:|---:|
| Client model x_c | 2.61 | 2.61 | 5.21 | 10.43 |
| Auxiliary model a_c | 8.03 | 8.02 | 16.04 | 32.09 |
| Working-memory peak (activations + input batch) | | | | 106.28 |
| ↳ of which autograd activations | | | | 103.28 |
| **Client total** | | | | **148.80** ✓ |

| Item | MiB | Meaning |
|---|---:|---|
| **Held-across-cut** | **0.00** | The client never waits for the server. It finishes its own forward + backward (against the aux head) before uploading, so no client memory is held while the server works. |

### Server: 176.26 MiB

| Item | Params + buffers | Grads | Adam state | MiB |
|---|---:|---:|---:|---:|
| Server model x_s (one copy, shared by all clients) | 40.08 | 40.05 | 80.10 | 160.22 |
| Working-memory peak (activations + uploaded batch) | | | | 16.04 |
| ↳ of which autograd activations | | | | 14.04 |
| **Server total** | | | | **176.26** ✓ |

CSE-FSL keeps nothing else on the server between uploads. Averaging the aux
model needs a temporary copy during aggregation only.

### System peak

| Item | MiB |
|---|---:|
| **System peak** (live bytes, client + server at the same instant) | **106.28** |
| with static model + optimizer bytes added | 309.02 |

---

## Accuracy, time and process diagnostics

Values are from the T4 rerun. Every comm and memory number was identical to
the first run; CSE-FSL is unaffected by the FSL-SAGE aux fix.

| Item | Value |
|---|---:|
| **Test accuracy** (round 2) | **98.80%** |
| **Latency** (application start → end) | **98.26 s** |
| Client compute time (sum) | 1.22 s |
| Server compute time (sum) | 0.22 s |
| Client FedAvg compute time | 0.42 s |
| Aux FedAvg compute time | 0.20 s |
| Peak process RSS (diagnostic) | 1721.52 MiB |
| Peak CUDA allocator (diagnostic) | 934.55 MiB |
| Ordering check cuda ≥ system ≥ max(side) | OK |

The CUDA and RSS figures are not benchmark metrics. They include all 10
simulated clients, the dataset and the cuDNN workspace on one GPU.

<details>
<summary>Raw breakdown output (T4 rerun; identical to the first run apart from timing)</summary>

```
> Round 2, avg tr l_loss: 0.06, avg tr l_acc: 0.98, avg tr g_loss: 0.06, avg tr g_acc: 0.98, ts. loss: 0.04, ts. acc: 98.80%, comm cut: 0.293 GiB, comm weights: 0.624 GiB, comm total: 0.917 GiB.
> Peak client mem: 148.80 MiB (act 103.28, held across cut 0.00), peak server mem: 176.26 MiB, system live peak: 106.28 MiB.

## Breakdown: cse_fsl  —  CSE-FSL, Mu & Shen, arXiv:2302.05599
N = 10 clients, R = 3 rounds, local iters/round = [24, 24, 24, 24, 24, 24, 24, 24, 24, 24], h = 5, l = None.  Units: MiB (= table 'MB').

### Communication
| # | What is sent | Paper | Column | Count | Expected | MiB / transfer | MiB | |
|---|---|---|---|---:|---:|---:|---:|---|
| C1 | Smashed activations z_f, client -> server | Sec.III-B Step 2 (p.3): uploaded only when m mod h = 0 | Cut | 150 | 150 | 2.0000 | 300.00 | ✓ |
| C2 | Labels y for each uploaded batch, client -> server | Step 2 (p.3): "uploads the smashed data and the labels" | Cut | 150 | 150 | 0.0020 | 0.29 | ✓ |
| C3 | Gradient at the cut, server -> client | p.3: never sent; client trains on its local aux loss | Cut | 0 | 0 | 0.0000 | 0.00 | ✓ |
| W1 | Client model x_c upload (for FedAvg) | Step 4, Eq.(9) (p.4): every round | Weights | 30 | 30 | 2.6131 | 78.39 | ✓ |
| W2 | Averaged client model x_c download | Step 1 (p.3): start of every round | Weights | 30 | 30 | 2.6131 | 78.39 | ✓ |
| W3 | Aux model a_c upload (for FedAvg) | Step 4, Eq.(9) (p.4): aux is averaged every round | Weights | 30 | 30 | 8.0294 | 240.88 | ✓ |
| W4 | Averaged aux model a_c download | Step 1 (p.3): start of every round | Weights | 30 | 30 | 8.0294 | 240.88 | ✓ |
| W5 | Server-model FedAvg (per-client x_s copies) | p.3: "keep only a single server-side model" -> none | Weights | 0 | 0 | 0.0000 | 0.00 | ✓ |

- Cut = 300.29 MiB  (rows sum 300.29 ✓)
- Weights = 638.55 MiB  (rows sum 638.55 ✓)
  - of that, Server-model FedAvg = 0.00 MiB
- Total = Cut + Weights = 938.84 MiB
- Cut share = 32%

### Memory — client (worst client = 0)
| client model x_c        | 2.61 | 2.61 | 5.21  | 10.43 |
| auxiliary model x_a     | 8.03 | 8.02 | 16.04 | 32.09 |
| working-memory peak (activations at own peak: 103.28) | | | | 106.28 |
- Client total = 148.80 MiB  (lines sum 148.80 ✓)
- Held-across-cut = 0.00 MiB

### Memory — server
| server model x_s        | 40.08 | 40.05 | 80.10 | 160.22 |
| working-memory peak (activations at own peak: 14.04) | | | | 16.04 |
- Server total = 176.26 MiB  (lines sum 176.26 ✓)

- System peak (live) = 106.28 MiB; with static = 309.02 MiB
Summary of net compute times: client 1.22s, server 0.22s, client_agg 0.42s, auxiliary_agg 0.20s
Total latency: 98.26s | Peak process memory: 1721.52 MiB | Peak CUDA memory: 934.55 MiB
Memory attribution on Tesla T4: client 148.80 | server 176.26 | system 309.02 | cuda-allocator 934.55 MiB -- ordering OK
```
</details>

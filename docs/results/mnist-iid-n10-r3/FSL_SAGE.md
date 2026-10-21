# FSL-SAGE: MNIST IID, 10 clients, 3 rounds

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
the count the paper's algorithm predicts. Every row matched (✓).

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
| **Held-across-cut** | **94.01** | Part of the client's working memory that stays alive *while the client waits for the server*. It happens only on alignment iterations (round 0, first batch here): the client must wait for the freshly aligned aux model before it can backpropagate. It is already inside the 115.30, not extra. |

### Server: 218.60 MiB

| Item | Params + buffers | Grads | Adam state | MiB |
|---|---:|---:|---:|---:|
| Server model x_s (one copy, shared by all clients) | 40.08 | 40.05 | 80.10 | 160.22 |
| Aux model x_a, the copy being aligned (Alg.3 l.9–13) | 8.03 | 8.02 | 16.04 | 32.09 |
| Working-memory peak (activations + uploaded batch + stored z_f) | | | | 26.29 |
| ↳ of which autograd activations | | | | 22.29 |
| **Server total** | | | | **218.60** |

The server aligns one client's aux model at a time, so the headline carries one
aux copy with its gradients and Adam state.

**Also kept on the server by the protocol, but NOT in the 218.60:**

| Item | MiB | Why |
|---|---:|---|
| Alignment buffers × 10 clients | 98.20 | Paper p.5 / Alg.3 "Store for client i". Per client: up to 1,000 stored z_f (7.81 MiB) + true gradients from the last alignment (256 samples, 2 MiB) + labels. Only z_f is counted in the headline. |
| Aux models of the other 9 clients (params + grads + Adam) | 288.79 | Alg.3: each client has its own aux model and Adam state, which the server keeps between alignments. 9 × 32.09. |

Adding these would bring a real FSL-SAGE server to about 218.60 + 98.20 +
288.79 ≈ 606 MiB at 10 clients. They grow with the number of clients, while
the headline Server number does not.

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

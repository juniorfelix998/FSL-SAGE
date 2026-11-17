# FedSplitX: MNIST IID, 10 clients, 11 rounds

**Paper:** Shin et al., *FedSplitX: Federated Split Learning for Computationally-Constrained Heterogeneous Clients*, arXiv:2310.14579.

**Code:** the authors released none, so this is a **reimplementation** written from the paper's Algorithm 1. See [`docs/FEDSPLITX_BREAKDOWN.md`](../../FEDSPLITX_BREAKDOWN.md).

**Run:**
- Colab T4, ResNet-18, middle cut (paper depth level 3), seed 200;
- batch 256, random init;
- **all 10 clients train every round** (C = 1.0, `client_fraction` in the yaml);
- SGD lr 0.01 (the paper gives no learning rate).

```
python main.py rounds=11 save=False device=cuda algorithm=fedsplitx \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true
```

| Result | Value |
|---|---:|
| Test accuracy, full model (ensemble of all heads + output) | **96.28%** |
| Test accuracy, client side (ensemble of the 3 client heads) | 91.67% |
| Communication, total | **14555.56 MiB** (Cut 5161.29 + Weights 9394.27) |
| Client memory | **102.54 MiB** |
| Server memory, per device (table value) | **181.17 MiB** |
| Server memory, total for all clients' state | **1324.60 MiB** |
| Latency | **1265.48 s** |

Every breakdown row ✓. Units: MiB = 1024² bytes (the benchmark table's "MB").

## How FedSplitX works here

Each round, all 10 clients do the following. In the paper they run in parallel; the code runs them one after another, each starting from the same global models.

1. **Download.** The client downloads the global client model and its 3 aux heads (L5). The server takes a copy of the global server model for that client (L6).
2. **One upload of everything.** Before training, the client runs **all 6,000 of its images** through the model it just downloaded. It sends the cut activations and labels to the server **in one go** (GetSmashedData, L7/L18).
3. **Client trains alone.** The client trains on 3 local losses, one per aux head: after the stem, after layer1, after layer2 (L13–16). It never waits for the server.
4. **Server trains alone.** At the same time, the server trains that client's copy on the uploaded activations (L9–11).
5. **Average.** Each client uploads its model and aux heads, and the server averages the client models and the server copies (heteroavg, L12). With every client at the same depth level, this is a plain average.

**No gradient ever crosses the cut.**

## Inputs

- **Images per client:** 6,000, in 24 batches (23 × 256 + 1 of 112).
- **One activation per image:** 128 channels × 4 × 4 × 4 B = 8,192 B.
- **Client model:** 683,072 params + BatchNorm buffers = 2,740,048 B = 2.6131 MiB.
- **Server model:** 10,498,570 params + buffers = 42,025,080 B = 40.0782 MiB.
- **Aux heads:** (64·10+10) + (64·10+10) + (128·10+10) = 2,590 params = 10,360 B = 0.0099 MiB.
- **Client-rounds of training:** 10 × 11 = **110**.

## Communication: 14555.56 MiB

| What is sent | Per transfer (MiB) | Count | MiB |
|---|---:|---:|---:|
| Activations of all 6,000 images, one upload per client per round (L7/L18) | 46.8750 | 110 | 5156.25 |
| Labels with them (8 B each) | 0.0458 | 110 | 5.04 |
| Gradient back to client | – | 0 | 0.00 |
| Client model up (L12) | 2.6131 | 110 | 287.44 |
| Client model down (L5) | 2.6131 | 110 | 287.44 |
| Aux heads up | 0.0099 | 110 | 1.09 |
| Aux heads down | 0.0099 | 110 | 1.09 |
| Server copy down (L6) + up (L12): Server-model FedAvg | 40.0782 | 220 | 8817.21 |
| **Total** | | | **14555.56** (Cut 5161.29 + Weights 9394.27) |

- **Where the bytes come from:**
  - every row is once per client per round: 10 × 11 = 110 (the server copy: 2 × 110 = 220);
  - one upload is 6,000 × 8,192 B = 49,152,000 B = 46.875 MiB.
- **Server-model FedAvg is 61% of the total.** It is the benchmark's convention of charging server-copy averaging (as for SplitFedv1). Without it: 5738.35 MiB.
- **The cut traffic is one activation per image per round,** the same as HO-SFL's uploads (5156.25 MiB). FedSplitX sends no gradient back, so its cut is half of HO-SFL's.

Code: `src/algos/fedsplitx.py`: `client_step` (L5–L11) and `aggregate` (L12).

## Memory

| Client: 102.54 MiB | MiB |
|---|---:|
| Client model: params 2.61 + grads 2.61 (plain SGD: no optimizer state) | 5.22 |
| 3 aux heads: params 0.01 + grads 0.01 | 0.02 |
| Working memory: autograd activations 94.29 + input batch 3.00 + small outputs 0.01 | 97.30 |
| **Total** | **102.54** |

| Server | MiB |
|---|---:|
| Global server model (params + buffers) | 40.08 |
| Server copy being trained: params 40.08 + grads 40.05 | 80.13 |
| Working memory: the client's whole upload 46.88 + labels 0.05 + autograd activations 14.04 + small outputs 0.01 | 60.97 |
| **Per device (table value)** | **181.17** |
| + server copies of the other 9 clients (9 × 80.13, params + grads) | 721.14 |
| + uploads of the other 9 clients (9 × 49,200,000 B) | 422.28 |
| **Total for all clients' state** | **1324.60** |

- **Why the other 9 uploads count.** In Alg.1 all clients run in parallel, so the server holds all 10 uploads while it trains the 10 copies. The run's breakdown lists the 9 copies (721.14, so 902.32 in its last line); the 9 uploads are added here from the upload size.
- **Held-across-cut = 0:** the client never waits for the server.
- **System peak:** 144.17 MiB live; 269.62 with all model bytes added.

## Time

| | Value |
|---|---:|
| Latency | **1265.48 s** |
| Client compute (mean per client over 11 rounds) | 69.83 s |
| Server compute (mean per client over 11 rounds) | 23.58 s |
| Averaging, client / server (total) | 0.16 / 0.14 s |

- **Per client batch**, the client runs a forward + backward pass with 3 aux heads, and the server runs a forward + backward on **every** batch (CSE-FSL's server only on 1 in 5).
- **Each client also runs one extra forward pass** over all its data for the upload.
- **The compute times are CPU-clock timings** that do not wait for the GPU; compare latency, not the compute times.

## Accuracy

**Full model: 96.28%.** This is the benchmark number.
- After every round, the 10,000 test images go through the global client model and the global server model.
- The prediction is the **ensemble** of 4 predictions: the 3 aux heads and the server's output (paper Sec. 3: "uses the ensemble of all auxiliary networks").
- Each of the 4 gives a probability for each digit. The 4 probabilities are averaged, and the digit with the highest average wins.

**Client side: 91.67%.**
- The same ensemble, but of the **3 client aux heads only**, without the server: what a client could predict **on its own**, using only the model it holds.
- It comes about because every partition point has a small classifier (global average pool + one linear layer) trained during Client_Update. After the round, the averaged heads are kept with the averaged client model.

**The full model is 4.6 points above the client side**, because the server's layer3 and layer4 add depth. The paper shows the same pattern (Table 1, ResNet-18, level 3, CIFAR-10: client side 83.81 / full model 84.4).

**Below CSE-FSL (99.13%) and FSL-SAGE (98.96%).**
- Each round averages 10 models that each made 24 plain-SGD steps (lr 0.01) from the same start, so the global model moves about one client's worth of steps per round.
- The other two methods train with Adam.

## Differences from the paper (deliberate, for a fair benchmark)

- **All clients at one depth level** (the middle cut = level 3), the same cut as every method. The paper's mix of levels is not used, so heteroavg is a plain average.
- **All 10 clients every round (C = 1.0)**, so FedSplitX sees the same data per round as every other method. The paper uses C = 10% of 50 clients (5 per round).
- **Batch 256**, the batch every method uses (the paper: 64).
- **Random init**, like every method.
- **Not given in the paper, chosen here:**
  - learning rate (SGD 0.01);
  - aux head design;
  - ensemble rule;
  - how the aux heads are averaged.

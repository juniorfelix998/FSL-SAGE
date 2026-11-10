# FedSplitX: MNIST IID, 10 clients, 11 rounds
## NOT YET DONE JUST PUSHED FIRSTG
**Paper:** Shin et al., *FedSplitX: Federated Split Learning for Computationally-Constrained Heterogeneous Clients*, arXiv:2310.14579.

**Code:** the authors released none, so this is a **reimplementation** written from the paper's Algorithm 1. See [`docs/FEDSPLITX_BREAKDOWN.md`](../../FEDSPLITX_BREAKDOWN.md).

**Run:**
- Colab T4, ResNet-18, middle cut (paper depth level 3), seed 200;
- batch 256, random init;
- **10% of the clients per round** (paper p.5), i.e. 1 of 10;
- SGD lr 0.01 (the paper gives no learning rate).

```
python main.py rounds=11 save=False device=cuda algorithm=fedsplitx \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true
```

| Result | Value |
|---|---:|
| Test accuracy, full model (ensemble of all heads + output) | **95.85%** |
| Test accuracy, client side (ensemble of the 3 client heads) | 91.61% |
| Communication, total | **1455.56 MiB** |
| Client memory | **102.53 MiB** |
| Server memory, per device (table value) | **181.17 MiB** |
| Latency | **144.27 s** |

Units: MiB = 1024² bytes (the benchmark table's "MB").

## How FedSplitX works here

Each round, **one client** is picked at random (L3). Then:

1. **Download.** The client downloads the global client model and its 3 aux heads (L5). The server takes a copy of the global server model for that client (L6).
2. **One upload of everything.** Before training, the client runs **all 6,000 of its images** through the model it just downloaded. It sends the cut activations and labels to the server **in one go** (GetSmashedData, L7/L18).
3. **Client trains alone.** The client trains on 3 local losses, one per aux head: after the stem, after layer1, after layer2 (L13–16). It never waits for the server.
4. **Server trains alone.** At the same time, the server trains its copy on the uploaded activations (L9–11).
5. **Average.** The client uploads its model and aux heads, and the server copy is averaged too (L12). With one client per round, the average is just that client's model.

**No gradient ever crosses the cut.**

## Inputs

- **Images per client:** 6,000, in 24 batches (23 × 256 + 1 of 112).
- **One activation per image:** 128 channels × 4 × 4 × 4 B = 8,192 B.
- **Client model:** 683,072 params + BatchNorm buffers = 2,740,048 B = 2.6131 MiB.
- **Server model:** 10,498,570 params + buffers = 42,025,080 B = 40.0782 MiB.
- **Aux heads:** (64·10+10) + (64·10+10) + (128·10+10) = 2,590 params = 10,360 B = 0.0099 MiB.

## Communication: 1455.56 MiB

| What is sent | Count | MiB |
|---|---|---:|
| Activations of all 6,000 images, one upload per round (L7/L18) | 11 × 6,000 × 8,192 B | 515.63 |
| Labels with them (8 B each) | 11 × 6,000 × 8 B | 0.50 |
| Gradient back to client | 0 (never sent) | 0.00 |
| Client model up (L12) | 11 × 2.6131 | 28.74 |
| Client model down (L5) | 11 × 2.6131 | 28.74 |
| Aux heads up | 11 × 0.0099 | 0.11 |
| Aux heads down | 11 × 0.0099 | 0.11 |
| Server copy down (L6) + up (L12): Server-model FedAvg | 22 × 40.0782 | 881.72 |
| **Total** | Cut 516.13 + Weights 939.43 | **1455.56** |

- **Server-model FedAvg is 61% of the total.** It is the benchmark's convention of charging server-copy averaging (as for SplitFedv1). Without it: 573.84 MiB.
- **Only 11 client-rounds train in total** (1 per round), against 110 for the methods that train every client every round.

Code: `src/algos/fedsplitx.py`: `client_step` (L5–L11) and `aggregate` (L12).

## Memory

| Client: 102.53 MiB | MiB |
|---|---:|
| Client model: params 2.61 + grads 2.61 (plain SGD: no optimizer state) | 5.22 |
| 3 aux heads: params 0.01 + grads 0.01 | 0.02 |
| Working memory: input batch 3.00 + autograd activations 94.29 | 97.29 |
| **Total** | **102.53** |

| Server: 181.17 MiB | MiB |
|---|---:|
| Global server model (params + buffers) | 40.08 |
| Server copy being trained: params 40.08 + grads 40.05 | 80.13 |
| Working memory: the round's upload 46.88 + labels 0.05 + autograd activations 14.04 | 60.96 |
| **Per device (table value)** | **181.17** |

- **The working memory includes the whole upload.** The server holds all 6,000 activations (46.88 MiB) for the round, because the client sends them in one go.
- **Total for all copies:** the same 181.17. Only 1 client is sampled, so there are no other server copies.
- **Held-across-cut = 0:** the client never waits for the server.
- **System peak:** 144.16 MiB live; 269.61 with all model bytes added.

## Time

- **Latency:** 144.27 s, the fastest so far, because only one client trains per round.
- **Compute** (mean per sampled client, summed over its rounds):
  - client 5.15 s, including the extra forward pass over all its data for the upload;
  - server 3.70 s.
- **Averaging:** 0.05 s (client) + 0.04 s (server) in total.

## Accuracy

- **95.85% full model; 91.61% client side.** This is the paper's own pattern: the full model is above the client-side model (Table 1, ResNet-18 level 3: 83.81 / 84.4 on CIFAR-10).
- **Below CSE-FSL and FSL-SAGE (~99%).** In 11 rounds FedSplitX trains on only 11 × 6,000 = 66,000 images (about 1.1 passes over MNIST), against 11 passes for them. That follows from the paper's 10% sampling.

## Differences from the paper (deliberate, for a fair benchmark)

- **All clients at one depth level** (the middle cut = level 3), the same cut as every method. The paper's mix of levels is not used, so heteroavg is a plain average.
- **10 clients, so 10% = 1 client per round** (the paper: 50 clients, 5 per round).
- **Batch 256**, the batch every method uses (the paper: 64).
- **Random init**, like every method.
- **Not given in the paper, chosen here:**
  - learning rate (SGD 0.01);
  - aux head design;
  - ensemble rule;
  - how the aux heads are averaged.

# Han et al. (locally generated losses): MNIST IID, 10 clients, 11 rounds

**Paper:** Han, Bhatti, Lee, Moon, *Accelerating Federated Learning with Split Learning on Locally Generated Losses*, ICML 2021 Workshop on Federated Learning (FL-ICML'21).

**Code:** the authors released none, so this is a **reimplementation** written from the paper's Sec. 3.2 (the protocol is prose; there is no algorithm box). It has **not yet reproduced** the paper's own numbers (Table 2). See [`docs/HAN_LOCLOSS_BREAKDOWN.md`](../../HAN_LOCLOSS_BREAKDOWN.md).

**Run:**
- Colab T4, ResNet-18, middle cut, seed 200;
- batch 256, random init;
- **all 10 clients train every round**;
- SGD lr 0.01, momentum 0.9 (the paper's own setting, p.6).

```
python main.py rounds=11 save=False device=cuda algorithm=han_locloss \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true
```

| Result | Value |
|---|---:|
| Test accuracy (client model → server model) | **98.91%** |
| Communication, total | **5753.37 MiB** (Cut 5161.29 + Weights 592.08) |
| Client memory | **105.09 MiB** |
| Server memory, per device (table value) | **176.30 MiB** |
| Server memory, total for all clients' state | **1257.89 MiB** |
| Latency | **586.71 s** |

Every breakdown row ✓. Units: MiB = 1024² bytes (the benchmark table's "MB").

## How Han et al. works here

Each round, all 10 clients do the following. In the paper they run in parallel; the code runs them one after another.

1. **Download.** The client downloads the global client model and the aux head (Step 1).
2. **Per minibatch:**
   - **Upload.** The client runs the batch through its model and sends the cut activations and labels to the main server (Step 2).
   - **Client trains alone.** The client computes a cross-entropy loss on its aux head, a single linear layer on the flattened activation. It updates its model and the head from that loss only (Eq. 3, 4).
   - **Server trains alone.** The main server trains **this client's own copy** of the server model on the uploaded activations, with the real output loss (Eq. 5).
3. **Average.**
   - The fed server averages the client models and aux heads; each client uploads both (Step 3).
   - The main server averages its 10 server copies itself (Step 4b, p.4).

**No gradient ever crosses the cut** (p.6: "the downlink communication for transmitting the backpropagated signals is not required").

## Inputs

- **Images per client:** 6,000, in 24 batches (23 × 256 + 1 of 112).
- **One activation per image:** 128 channels × 4 × 4 × 4 B = 8,192 B. One label: 8 B.
- **Client model:** 2,740,048 B = 2.6131 MiB.
- **Server model:** 42,025,080 B = 40.0782 MiB.
- **Aux head:** Linear(2048, 10) = 20,490 params = 81,960 B = 0.0782 MiB, about 0.18% of the model. The paper's head is 0.1–0.6% of its model.
- **Client batches:** 10 × 24 × 11 = **2,640**. **Client-rounds:** 10 × 11 = **110**.

## Communication: 5753.37 MiB

| What is sent | Per transfer (MiB) | Count | MiB |
|---|---:|---:|---:|
| C1 activations, every minibatch (Step 2) | 2.0000 (1.9531 on average, with the batch of 112) | 2,640 | 5156.25 |
| C2 labels with them (Fig. 1(c)) | 0.0020 | 2,640 | 5.04 |
| C3 gradient back to client | – | 0 | 0.00 |
| W1 client model up (Step 3) | 2.6131 | 110 | 287.44 |
| W2 client model down (Step 1) | 2.6131 | 110 | 287.44 |
| W3 aux head up (Step 3) | 0.0782 | 110 | 8.60 |
| W4 aux head down (Step 1) | 0.0782 | 110 | 8.60 |
| W5 server copies averaged (Step 4b): on the main server, no network | – | 0 | 0.00 |
| **Total** | | | **5753.37** (Cut 5161.29 + Weights 592.08) |

**Exact bytes:**

| | Formula | Bytes |
|---|---|---:|
| Cut | 6,000 × (8,192 + 8) × 110 | 5,412,000,000 |
| Weights | 2 × (2,740,048 + 81,960) × 110 | 620,841,760 |
| **Total** | | **6,032,841,760 = 5753.37 MiB** |

The Colab log's GiB line agrees: 5.040 + 0.578 = 5.619 GiB.

- **90% of the traffic is the cut.** Every image is uploaded once per round: the same bytes as FedSplitX (5161.29), but sent batch by batch, not in one bulk upload. No gradient comes back, so the cut is half of HO-SFL's.
- **The server copies are not charged.** The paper averages them on the main server (Step 4b, p.4), which holds every copy, so nothing crosses a network. This is the same rule as SplitFedv1, FedSplitX and MU-SplitFed. The copies count as server memory instead (below).
- **Weights are 15.02 MiB above FedSplitX's** (592.08 vs 577.06). The difference is the aux head: one flattened linear head (0.0782 MiB) against FedSplitX's three pooled heads (0.0099 MiB). 2 × 110 × (0.0782 − 0.0099) = 15.02.
- **The aux head is 2.9% of the weight traffic** (17.20 of 592.08). The paper's Table 1 leaves it out ("|a| ≪ |w|").

Code: `src/algos/han_locloss.py`: `client_step` (Steps 2–4a) and `aggregate` (Steps 3b, 4b).

## Memory

| Client: 105.09 MiB | MiB |
|---|---:|
| Client model: params 2.61 + grads 2.61 + momentum 2.61 | 7.82 |
| Aux head: params 0.08 + grads 0.08 + momentum 0.08 | 0.23 |
| Working memory: autograd activations 94.02 + input batch 3.00 + small outputs 0.01 | 97.03 |
| **Total** | **105.09** |

| Server | MiB |
|---|---:|
| Global server model (params + buffers) | 40.08 |
| Server copy being trained: params 40.08 + grads 40.05 + momentum 40.05 | 120.18 |
| Working memory: one uploaded batch (2.00) + autograd activations 14.04 + small outputs | 16.05 |
| **Per device (table value)** | **176.30** |
| + server copies of the other 9 clients (9 × 120.18: params + grads + momentum) | 1081.58 |
| **Total for all clients' state** | **1257.89** |

- **Why the client is 105.09, not FedSplitX's 102.54.** The momentum buffers add a third copy of the client model (+2.61), and the flattened head is bigger than three pooled heads.
- **The per-device server number is close to CSE-FSL's (176.26).**
  - Both servers handle one batch at a time.
  - CSE-FSL holds one server model with Adam: params + grads + 2 Adam states = 160.22.
  - Han holds the global model (40.08) and one copy with SGD momentum (120.18): 160.25 from exact bytes (168,038,720 B), nearly the same.
- **The other 9 copies.** In Step 4a the main server updates every client's copy in parallel, so all 10 copies, with gradients and momentum, exist at once. They are listed separately (1081.58), giving 1257.89 in total.
  - FedSplitX's copies are cheaper (80.13 each, no momentum), but its server also holds every client's whole upload (422.29).
- **Parallel vs one after another.**
  - **Results are the same.** The clients share nothing until the averaging step. Running them in reverse order gives a bit-identical model (`test_han_locloss`).
  - **Memory is measured one client at a time.**
- **Held-across-cut = 0:** the client frees its graph before it uploads and never waits for the server.
- **System peak:** 97.03 MiB live; 265.34 with all model bytes added.

## Time

| | Value |
|---|---:|
| Latency | **586.71 s** |
| Client compute (mean per client over 11 rounds) | 3.16 s |
| Server compute (mean per client over 11 rounds) | 22.12 s |
| Averaging, client / aux / server (total) | 1.93 / 0.03 / 2.16 s |

- **The server runs a forward and backward on every batch** (CSE-FSL's server: 1 in 5), so Han is slower than CSE-FSL (338.87 s).
- **It is about twice as fast as FedSplitX (1253.75 s):** there is no extra forward pass over all the data for a bulk upload, and the client has one small head instead of three.
- **The compute times are CPU-clock timings** that do not wait for the GPU; compare latency, not the compute times.

## Accuracy

**98.91%.** After every round, the 10,000 test images go through the global client model and the global server model: [w_C^T, w_S^T], as the paper evaluates (Step 5). The aux head is not used at test time.

- **How it learned.**
  - The client learns only from its local aux loss.
  - The server learns from the real output loss, on activations that keep changing under it.
  - By round 10, the per-batch training accuracy is 0.99 for the client's head and 1.00 for the server's output, and the test loss is 0.03.
- **Compared with the others.**
  - Han is within 0.2 points of CSE-FSL (99.13%) and FSL-SAGE (98.96%), and 2.6 points above FedSplitX (96.28%), with no gradient across the cut.
  - FedSplitX uses plain SGD without momentum. Han uses the paper's momentum 0.9, and its server trains on fresh per-batch activations, not on a bulk upload made before the client trained.
- **Compared with the paper.**
  - The method behaved as the paper describes. Both sides trained on their own losses, the full model reached high MNIST accuracy, and nothing was sent back across the cut.
  - The paper's setting is very different: an AlexNet-like CNN, 1000 clients with 300 per round, batch 10, and about 190 rounds. So this run is **not** a reproduction of its Table 2 (97.73% MNIST IID). That reproduction is still open.

## Differences from the paper (deliberate, for a fair benchmark)

- **ResNet-18 at the middle cut** instead of the paper's CNN / VGG-11, the same cut as every method.
- **All 10 clients every round.** The paper samples K = 300 of N = 1000.
- **Batch 256**, the batch every method uses. The paper uses 10: 6 local steps per round there, 24 here.
- **Random init**, like every method.
- **The aux head's transfers are charged** (W3, W4), which the paper's Table 1 leaves out.
- **Not given in the paper, chosen here:**
  - the per-batch order: forward → upload → client step → server step;
  - the aux head design: flatten + one linear layer;
  - momentum buffers kept per client and per server copy;
  - no weight decay and no lr schedule.

# MU-SplitFed: MNIST IID, 10 clients, 11 rounds

Paper: Liang et al., *Towards Straggler-Resilient Split Federated Learning: An
Unbalanced Update Approach*, NeurIPS 2025 (arXiv:2510.21155). Code: the
authors' repo is LLM-only and does not run as published, so this is the vision
reimplementation from HKU-WILL-Lab/HO-SFL, checked step by step against the
paper's Algorithm 1 (a **reimplementation**; see
[`docs/MU_SPLITFED_BREAKDOWN.md`](../../MU_SPLITFED_BREAKDOWN.md)).

**Run:**
- Colab T4, ResNet-18, middle cut, seed 200;
- **batch 256**, random init;
- **5 of the 10 clients sampled per step** (the paper's 50%);
- τ = 2, λ = 0.005, η_c = 0.005, η_s = 0.01, η_g = 0.3 (paper Table 6).

```
python main.py rounds=11 save=False device=cuda algorithm=mu_splitfed \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true \
    model.client.batch_size=256
```

| Result | Value |
|---|---:|
| Test accuracy (after the 11th round) | **20.08%** |
| Communication, total | **29271.03 MiB** |
| Client memory | **5.62 MiB** |
| Server memory, per device (table value) | 86.16 MiB |
| Server memory, total for all copies | **246.47 MiB** |
| Latency | **548.80 s** |

Units: MiB = 1024² bytes (the benchmark table's "MB").

## How MU-SplitFed works here

Each **step** takes one batch from each of 5 clients, picked at random among
those that still have batches left this round, so every client still uses all
its data every round. Paper Algorithm 1:

1. **Pull.** Each picked client downloads the global client model. The
   server starts a copy of the global server model for that client.
2. **Three forwards, no backward.** The client runs its batch through its
   model three times: once normally (h), and once each with its weights nudged
   up and down along one random direction (h⁺, h⁻). It sends all **3
   activations** and the labels.
3. **Server steps.** The server takes **τ = 2** steps on its copy for that
   client, using the normal activation h. Each step also nudges the weights
   along a random direction, instead of backprop.
4. **One number back.** The server sends **one number**: how much the loss
   changes between h⁺ and h⁻ (using its updated copy). The client uses it to
   update its model along its random direction.
5. **Upload and average.** Each client uploads its model. The server averages
   the 5 client models and its 5 server copies, then moves the global models
   0.3 of the way towards the averages (η_g).

**No gradient ever crosses the cut.** Both sides learn from random directions
only.

## Inputs

- **Batches:** 24 per client per round (23 × 256 + 1 of 112), so 10 × 11 × 24
  = **2,640 client batches**, done in steps of 5 clients.
- **One activation:** 256 images × 8,192 B = 2 MiB.
- **One client model:** 2,740,048 B = 2.6131 MiB.

## Communication: 29271.03 MiB

| What is sent | Count | MiB |
|---|---|---:|
| Activations h, h⁺, h⁻, client → server | 3 × 2,640 = 7,920 | 15468.75 |
| Labels | 2,640 | 5.04 |
| Gradient back to client | 0 (never sent) | 0.00 |
| Loss-difference number, server → client | 2,640 × 4 B | 0.01 |
| Client model up (for averaging) | 2,640 | 6898.62 |
| Client model down (pull) | 2,640 | 6898.62 |
| Server copies averaging | 0: done inside the server, not over the network | 0.00 |
| **Total** | Cut 15473.80 + Weights 13797.24 | **29271.03** |

- **Activations: 3 tensors per batch** (3 × 6,000 images × 8,192 B per client
  per round).
- **Weights:** because the models are averaged after **every step**, each
  client sends its model up and pulls it down once per batch it trains on.

Code: `src/algos/mu_splitfed.py` `group_step` (steps marked with the paper's
line numbers).

## Memory

| Client: 5.62 MiB | MiB |
|---|---:|
| Client model, params only (no gradients or optimizer state: updated in place) | 2.61 |
| Working memory: input batch 3.00 (no activations kept: forward only) | 3.00 |
| **Total** | **5.62** |

| Server | MiB |
|---|---:|
| Global server model (params only) | 40.08 |
| Copy being trained for one client (params only) | 40.08 |
| Working memory: the 3 uploaded activations (h, h⁺, h⁻) held at once; no autograd activations | 6.00 |
| **Per device (table value)** | **86.16** |
| + copies for the other 4 sampled clients (4 × 40.08) | 160.31 |
| **Total for all copies** | **246.47** |

- **Held-across-cut = 0:** the client keeps no activations.
- **System peak:** 21.01 MiB live (5 clients' input batches 15.01 + 3
  uploads 6.00 at one moment), or 103.78 with all model bytes added.

## Time

- **Latency:** 548.80 s.
- **Compute (mean per client over 11 rounds):**
  - client 2.42 s
  - server 14.26 s (6 server forwards per client batch)
- **Averaging:** 14.57 s in total (after every step).

## Accuracy

**20.08%**, the lowest in the benchmark. Both sides learn only from random
directions (the server has 10.5 million weights), starting from random
weights, so progress is slow and noisy; the training loss is still high
(14.72). This matches the HO-SFL paper's result for this reimplementation:
14.0% on CIFAR-10 (Table 3, p.23; Fig.3, p.7), there with pretrained weights.

## Differences from the paper (deliberate, for a fair benchmark)

- **Vision instead of LLMs.** The method itself (no LoRA, full-parameter
  zeroth-order) maps directly onto ResNet-18 with the benchmark's middle cut.
- **Random init**, like every other method.
- **Fair sampling.** 5 of 10 clients per step, picked at random among the
  clients with batches left, so every client uses all its data every round.
  The paper doesn't say how its 5 are chosen.
- **Batch 256**, the batch every other method uses (the paper: 32).
- **Random directions are Gaussian**, as in both codebases; the paper's theory
  draws them from a sphere.
